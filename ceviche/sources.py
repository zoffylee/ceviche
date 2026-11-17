"""Differentiable plane-wave total-field/scattered-field sources."""

import math

import torch

from .constants import C_0
from .solvers import matvec
from .tensor import as_tensor


def b_TFSF(fdfd, inside_mask, theta):
    """Return a source grid accepted by a single-frequency FDFD solve.

    The outside-field mask Q generates (Q A - A Q) f / (i omega).
    """
    mask = as_tensor(inside_mask, device=fdfd.device, dtype=fdfd.dtype)
    if tuple(mask.shape) != fdfd.shape:
        raise ValueError("TFSF mask must match the simulation grid")
    field = compute_f(
        theta,
        2 * math.pi * C_0 / fdfd.omega,
        fdfd.dL,
        fdfd.shape,
        device=fdfd.device,
        dtype=fdfd.dtype,
    )
    values, indices = fdfd._make_A(fdfd.eps_r.flatten())
    q = 1 - mask.flatten()
    rhs = q * matvec(values, indices, field) - matvec(values, indices, q * field)
    return (rhs / (1j * fdfd.omega)).reshape(fdfd.shape)


def compute_Q(inside_mask):
    mask = as_tensor(inside_mask).flatten()
    ids = torch.arange(mask.numel(), device=mask.device)
    return torch.sparse_coo_tensor(torch.stack((ids, ids)), 1 - mask, (mask.numel(), mask.numel()))


def compute_f(theta, lambda0, dL, shape, *, device=None, dtype=torch.float64):
    theta = as_tensor(theta, device=device, dtype=dtype)
    x = torch.arange(shape[0], device=theta.device, dtype=dtype)[:, None]
    y = torch.arange(shape[1], device=theta.device, dtype=dtype)[None, :]
    return torch.exp(
        1j * 2 * math.pi / lambda0 * dL * (x * theta.sin() - y * theta.cos())
    ).flatten()
