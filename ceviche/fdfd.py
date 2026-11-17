"""Differentiable frequency-domain Maxwell solvers using Torch tensors."""

import math

import torch

from .constants import EPSILON_0, MU_0
from .derivatives import derivative_entries, product_pattern
from .solvers import SolverOptions, matvec, sparse_solve
from .tensor import as_tensor, complex_dtype, real_dtype


class fdfd:
    def __init__(
        self,
        omega,
        dL,
        eps_r,
        npml,
        bloch_phases=None,
        *,
        device=None,
        dtype=None,
        solver_options=None,
    ):
        material = as_tensor(eps_r, device=device, dtype=dtype)
        if material.ndim != 2 or min(material.shape) < 1:
            raise ValueError("FDFD requires a non-empty 2D permittivity grid")
        if not math.isfinite(omega) or omega <= 0 or not math.isfinite(dL) or dL <= 0:
            raise ValueError("omega and dL must be positive finite scalars")
        if len(npml) != 2 or any(
            int(p) != p or p < 0 or 2 * p > n for p, n in zip(npml, material.shape)
        ):
            raise ValueError("npml must contain two nonnegative widths fitting the grid")
        self.omega, self.dL = omega, dL
        self.device = material.device
        self.dtype = real_dtype(material.dtype)
        self.complex_dtype = complex_dtype(material.dtype)
        self.shape = tuple(material.shape)
        self.Nx, self.Ny = self.shape
        self.N = material.numel()
        self.npml = tuple(npml)
        self.bloch_x, self.bloch_y = (tuple(bloch_phases or []) + (0.0, 0.0))[:2]
        self.solver_options = solver_options or SolverOptions(
            rtol=1e-10 if self.dtype == torch.float64 else 1e-5
        )
        self.eps_r = material
        self._derivatives = derivative_entries(
            omega,
            self.shape,
            npml,
            dL,
            (self.bloch_x, self.bloch_y),
            device=self.device,
            dtype=self.complex_dtype,
        )
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = [
            torch.sparse_coo_tensor(i, v, (self.N, self.N)).coalesce() for v, i in self._derivatives
        ]
        for name, (values, indices) in zip(["Dxf", "Dxb", "Dyf", "Dyb"], self._derivatives):
            setattr(self, f"entries_{name}", values)
            setattr(self, f"indices_{name}", indices)
        self._ez_x = product_pattern(self._derivatives[0], self._derivatives[1], self.N)
        self._ez_y = product_pattern(self._derivatives[2], self._derivatives[3], self.N)
        self._hz_x = product_pattern(self._derivatives[1], self._derivatives[0], self.N)
        self._hz_y = product_pattern(self._derivatives[3], self._derivatives[2], self.N)
        diagonal = torch.arange(self.N, device=self.device)
        self._diagonal = torch.stack((diagonal, diagonal))

    @property
    def eps_r(self):
        return self._eps_r

    @eps_r.setter
    def eps_r(self, value):
        material = as_tensor(value, device=self.device)
        if tuple(material.shape) != self.shape:
            raise ValueError("Changing grid shape requires a new simulation")
        self._eps_r = material.to(self.complex_dtype if material.is_complex() else self.dtype)

    def _mult(self, index, x):
        values, indices = self._derivatives[index]
        return matvec(values, indices, x)

    def _grid_to_vec(self, grid):
        return as_tensor(grid, device=self.device, dtype=self.complex_dtype).flatten()

    def _vec_to_grid(self, vec):
        return vec.reshape(self.shape)

    def _grid_average_2d(self, eps):
        eps = eps.reshape(self.shape)
        return ((eps + torch.roll(eps, 1, 1)) / 2).flatten(), (
            (eps + torch.roll(eps, 1, 0)) / 2
        ).flatten()

    def _Ez_to_Hx_Hy(self, ez):
        return -self._mult(3, ez) / (1j * self.omega * MU_0), self._mult(1, ez) / (
            1j * self.omega * MU_0
        )

    def solve(self, source_z):
        source = as_tensor(source_z, device=self.device, dtype=self.complex_dtype)
        if tuple(source.shape) != self.shape:
            raise ValueError(f"Source shape must be {self.shape}")
        values, indices = self._make_A(self.eps_r.flatten())
        field = sparse_solve(
            values, indices, 1j * self.omega * source.flatten(), options=self.solver_options
        )
        return tuple(self._vec_to_grid(f) for f in self._fields(field))


class fdfd_ez(fdfd):
    def _make_A(self, eps):
        vx, ix, _ = self._ez_x
        vy, iy, _ = self._ez_y
        return torch.cat((-EPSILON_0 * self.omega**2 * eps, -vx / MU_0, -vy / MU_0)), torch.cat(
            (self._diagonal, ix, iy), dim=1
        )

    def _fields(self, field):
        return (*self._Ez_to_Hx_Hy(field), field)


class fdfd_hz(fdfd):
    def _make_A(self, eps):
        xx, yy = self._grid_average_2d(eps)
        vx, ix, kx = self._hz_x
        vy, iy, ky = self._hz_y
        diagonal = torch.full_like(eps, MU_0 * self.omega**2)
        return torch.cat(
            (diagonal, vx / (yy[kx] + 1e-5) / EPSILON_0, vy / (xx[ky] + 1e-5) / EPSILON_0)
        ), torch.cat((self._diagonal, ix, iy), dim=1)

    def _fields(self, field):
        xx, yy = self._grid_average_2d(self.eps_r.flatten())
        ex = self._mult(2, field) / (1j * self.omega * EPSILON_0 * (xx + 1e-5))
        ey = -self._mult(0, field) / (1j * self.omega * EPSILON_0 * (yy + 1e-5))
        return ex, ey, field


class fdfd_mf_ez(fdfd_ez):
    def __init__(
        self, omega, dL, eps_r, omega_mod, delta, phi, Nsb, npml, bloch_phases=None, **kwargs
    ):
        super().__init__(omega, dL, eps_r, npml, bloch_phases, **kwargs)
        if int(Nsb) != Nsb or Nsb < 0:
            raise ValueError("Nsb must be a nonnegative integer")
        self.Nsb, self.omega_mod = int(Nsb), omega_mod
        self.delta, self.phi = delta, phi
        self._frequencies = (
            omega + torch.arange(-Nsb, Nsb + 1, device=self.device, dtype=self.dtype) * omega_mod
        )
        if (self._frequencies == 0).any():
            raise ValueError("Zero-frequency sidebands are unsupported")

    def _modulation(self):
        delta = as_tensor(self.delta, device=self.device, dtype=self.dtype)
        phi = as_tensor(self.phi, device=self.device, dtype=self.dtype)
        if delta.ndim != 3 or phi.shape != delta.shape or tuple(delta.shape[1:]) != self.shape:
            raise ValueError("delta and phi must have matching (harmonics, Nx, Ny) shapes")
        if delta.shape[0] > self.Nsb:
            raise ValueError("Nsb must be at least the number of modulation harmonics")
        return delta.flatten(1), phi.flatten(1)

    def _make_A(self, eps, delta=None, phi=None):
        if delta is None:
            delta, phi = self._modulation()
        m, n = 2 * self.Nsb + 1, self.N
        w = self._frequencies
        vx, ix, _ = self._ez_x
        vy, iy, _ = self._ez_y
        base_v = torch.cat((-vx / MU_0, -vy / MU_0))
        base_i = torch.cat((ix, iy), dim=1)
        diagonal = torch.arange(m * n, device=self.device)
        values = [(-EPSILON_0 * w[:, None] ** 2 * eps[None, :]).flatten(), base_v.repeat(m)]
        indices = [
            torch.stack((diagonal, diagonal)),
            torch.cat([base_i + k * n for k in range(m)], dim=1),
        ]
        for k in range(delta.shape[0]):
            shift = k + 1
            upper = -0.5 * EPSILON_0 * delta[k] * torch.exp(1j * phi[k])
            lower = -0.5 * EPSILON_0 * delta[k] * torch.exp(-1j * phi[k])
            values.extend(
                (
                    (w[:-shift, None] ** 2 * upper[None, :]).flatten(),
                    (w[shift:, None] ** 2 * lower[None, :]).flatten(),
                )
            )
            row = torch.arange((m - shift) * n, device=self.device)
            indices.extend(
                (torch.stack((row, row + shift * n)), torch.stack((row + shift * n, row)))
            )
        return torch.cat(values), torch.cat(indices, dim=1)

    def solve(self, source_z):
        m = 2 * self.Nsb + 1
        source = as_tensor(source_z, device=self.device, dtype=self.complex_dtype)
        if tuple(source.shape) != (m, *self.shape):
            raise ValueError(f"Multifrequency source shape must be {(m, *self.shape)}")
        values, indices = self._make_A(self.eps_r.flatten())
        rhs = (1j * self._frequencies[:, None] * source.reshape(m, self.N)).flatten()
        ez = sparse_solve(values, indices, rhs, options=self.solver_options).reshape(m, self.N)
        hx = torch.stack(
            [-self._mult(3, ez[k]) / (1j * self._frequencies[k] * MU_0) for k in range(m)]
        )
        hy = torch.stack(
            [self._mult(1, ez[k]) / (1j * self._frequencies[k] * MU_0) for k in range(m)]
        )
        return tuple(f.reshape(m, *self.shape) for f in (hx, hy, ez))
