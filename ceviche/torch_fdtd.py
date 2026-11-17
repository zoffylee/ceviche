"""Yee FDTD with functional time steps and optional exact checkpointing."""

import math

import torch
from torch.utils.checkpoint import checkpoint

from .constants import C_0, EPSILON_0
from .tensor import as_tensor, real_dtype

AXES = "xyz"
STATE_KEYS = tuple(
    prefix + axis for prefix in ["E", "H", "D", "ICE", "IH", "ICH", "ID"] for axis in AXES
)


def curl(fields, axis, dl, forward):
    j, k = (axis + 1) % 3, (axis + 2) % 3
    if forward:
        return (
            torch.roll(fields[k], -1, j) - fields[k] - torch.roll(fields[j], -1, k) + fields[j]
        ) / dl
    return (fields[k] - torch.roll(fields[k], 1, j) - fields[j] + torch.roll(fields[j], 1, k)) / dl


class fdtd:
    def __init__(self, eps_r, dL, npml, *, device=None, dtype=None):
        eps = as_tensor(eps_r, device=device, dtype=dtype)
        if eps.is_complex():
            raise ValueError("FDTD currently supports real, positive permittivity")
        if eps.ndim > 3:
            raise ValueError("Batched FDTD grids are not implemented")
        eps = eps.reshape(*eps.shape, *((1,) * (3 - eps.ndim)))
        if min(eps.shape) < 1 or len(npml) != 3:
            raise ValueError("Expected a nonempty grid and three PML widths")
        if any(int(p) != p or p < 0 or 2 * p > n for p, n in zip(npml, eps.shape)):
            raise ValueError("PML widths must fit inside the grid")
        self.device, self.dtype = eps.device, real_dtype(eps.dtype)
        self.grid_shape = tuple(eps.shape)
        self.Nx, self.Ny, self.Nz = self.grid_shape
        self._dL = dL
        self._npml = tuple(npml)
        self._set_time_step()
        self._compute_sigmas()
        self.eps_r = eps

    @property
    def dL(self):
        return self._dL

    @dL.setter
    def dL(self, value):
        self._dL = value
        self._set_time_step()
        self._compute_sigmas()
        self.initialize_fields()

    @property
    def npml(self):
        return self._npml

    @npml.setter
    def npml(self, value):
        if len(value) != 3 or any(
            int(p) != p or p < 0 or 2 * p > n for p, n in zip(value, self.grid_shape)
        ):
            raise ValueError("PML widths must fit inside the grid")
        self._npml = tuple(value)
        self._compute_sigmas()
        self.initialize_fields()

    @property
    def eps_r(self):
        return self._eps_r

    @eps_r.setter
    def eps_r(self, value):
        eps = as_tensor(value, device=self.device)
        if eps.is_complex() or not torch.isfinite(eps).all() or (eps <= 0).any():
            raise ValueError("FDTD permittivity must be real, finite, and positive")
        if tuple(eps.shape) != self.grid_shape:
            raise ValueError("Changing grid shape requires a new simulation")
        self._eps_r = eps.to(self.dtype)
        self.eps_arr, self.N = self._eps_r.flatten(), self._eps_r.numel()
        self.initialize_fields()

    def _set_time_step(self, stability_factor=0.5):
        if not math.isfinite(self.dL) or self.dL <= 0 or not 0 < stability_factor <= 1:
            raise ValueError("Invalid grid spacing or stability factor")
        self.dt = self.dL / (math.sqrt(3) * C_0) * stability_factor

    def _compute_sigmas(self):
        shape = tuple(2 * n for n in self.grid_shape)
        sigmas = []
        for axis, width in enumerate(self.npml):
            sigma = torch.zeros(shape, dtype=self.dtype, device=self.device)
            for i in range(2 * width):
                value = 0.5 * EPSILON_0 / self.dt * (i / (2 * width)) ** 3
                for position in [2 * width - i + 1, 2 * self.grid_shape[axis] - 2 * width + i]:
                    selection = [slice(None)] * 3
                    selection[axis] = position
                    sigma[tuple(selection)] = value
            sigmas.append(sigma)
        self._sig_h = (
            sigmas[0][1::2, ::2, ::2],
            sigmas[1][::2, 1::2, ::2],
            sigmas[2][::2, ::2, 1::2],
        )
        self._sig_d = (
            sigmas[0][::2, 1::2, 1::2],
            sigmas[1][1::2, ::2, 1::2],
            sigmas[2][1::2, 1::2, ::2],
        )

    def _coefficients(self):
        coefficients = {}
        for prefix, sigmas, sign in [("H", self._sig_h, -1), ("D", self._sig_d, 1)]:
            for axis in range(3):
                own = sigmas[axis]
                sj, sk = sigmas[(axis + 1) % 3], sigmas[(axis + 2) % 3]
                denom = (
                    1 / self.dt
                    + (sj + sk) / (2 * EPSILON_0)
                    + sj * sk * self.dt / (4 * EPSILON_0**2)
                )
                coefficients[prefix + AXES[axis]] = (
                    (
                        1 / self.dt
                        - (sj + sk) / (2 * EPSILON_0)
                        - sj * sk * self.dt / (4 * EPSILON_0**2)
                    )
                    / denom,
                    sign * C_0 / denom,
                    sign * C_0 * self.dt * own / EPSILON_0 / denom,
                    -self.dt * sj * sk / EPSILON_0**2 / denom,
                )
        self.eps_xx, self.eps_yy, self.eps_zz = [
            (self.eps_r + torch.roll(self.eps_r, 1, axis)) / 2 for axis in range(3)
        ]
        coefficients["eps"] = (self.eps_xx, self.eps_yy, self.eps_zz)
        return coefficients

    def initialize_fields(self):
        self._coeffs = self._coefficients()
        self._state = tuple(
            torch.zeros(self.grid_shape, device=self.device, dtype=self.dtype) for _ in STATE_KEYS
        )
        self.t_index = 0
        self._publish()

    def _publish(self):
        state = dict(zip(STATE_KEYS, self._state))
        for key, value in state.items():
            setattr(self, key, value)
        self.fields = {key: state[key] for key in STATE_KEYS[:9]}

    def _step(self, state, source, coefficients):
        old = dict(zip(STATE_KEYS, state))
        new = dict(old)
        e = [old["E" + a] for a in AXES]
        for i, a in enumerate(AXES):
            ce = curl(e, i, self.dL, True)
            new["ICE" + a] = old["ICE" + a] + ce
            new["IH" + a] = old["IH" + a] + old["H" + a]
            c1, c2, c3, c4 = coefficients["H" + a]
            new["H" + a] = c1 * old["H" + a] + c2 * ce + c3 * new["ICE" + a] + c4 * new["IH" + a]
        h = [new["H" + a] for a in AXES]
        for i, a in enumerate(AXES):
            ch = curl(h, i, self.dL, False)
            new["ICH" + a] = old["ICH" + a] + ch
            new["ID" + a] = old["ID" + a] + old["D" + a]
            c1, c2, c3, c4 = coefficients["D" + a]
            new["D" + a] = (
                c1 * old["D" + a] + c2 * ch + c3 * new["ICH" + a] + c4 * new["ID" + a] + source[i]
            )
            new["E" + a] = new["D" + a] / coefficients["eps"][i]
        return tuple(new[key] for key in STATE_KEYS)

    def forward(self, Jx=None, Jy=None, Jz=None):
        source = []
        for current in (Jx, Jy, Jz):
            value = (
                torch.zeros_like(self.eps_r)
                if current is None
                else as_tensor(current, device=self.device, dtype=self.dtype)
            )
            if tuple(value.shape) != self.grid_shape:
                raise ValueError(f"Source shape must be {self.grid_shape}")
            source.append(value)
        self._state = self._step(self._state, source, self._coeffs)
        self.t_index += 1
        self._publish()
        return self.fields

    def run(self, sources, *, checkpoint_steps=0, reset=True):
        """Run a dict of Jx/Jy/Jz arrays shaped (steps, Nx, Ny, Nz).

        Checkpointing recomputes complete state segments, preserving the full
        gradient. Sources and coefficient graphs must not be mutated before backward.
        """
        if not sources or set(sources) - {"Jx", "Jy", "Jz"}:
            raise ValueError("Provide one or more Jx/Jy/Jz source sequences")
        if int(checkpoint_steps) != checkpoint_steps or checkpoint_steps < 0:
            raise ValueError("checkpoint_steps must be a nonnegative integer")
        inputs = {k: as_tensor(v, device=self.device, dtype=self.dtype) for k, v in sources.items()}
        first = next(iter(inputs.values()))
        if first.ndim != 4 or tuple(first.shape[1:]) != self.grid_shape:
            raise ValueError("Source sequences must have shape (steps, Nx, Ny, Nz)")
        if any(v.shape != first.shape for v in inputs.values()):
            raise ValueError("All source sequences must have matching shapes")
        if reset:
            self.initialize_fields()
        currents = torch.stack([inputs.get("J" + a, torch.zeros_like(first)) for a in AXES], dim=1)
        coefficients = self._coeffs

        def advance(*args):
            state, segment = args[:-1], args[-1]
            for source in segment:
                state = self._step(state, source, coefficients)
            return state

        step = int(checkpoint_steps) or max(1, first.shape[0])
        for start in range(0, first.shape[0], step):
            segment = currents[start : start + step]
            if checkpoint_steps and torch.is_grad_enabled():
                self._state = checkpoint(advance, *self._state, segment, use_reentrant=False)
            else:
                self._state = advance(*self._state, segment)
        self.t_index += first.shape[0]
        self._publish()
        return self.fields
