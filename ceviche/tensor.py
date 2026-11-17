"""Tensor conversion policy shared by the numerical kernels."""

import torch


def as_tensor(value, *, device=None, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if dtype is None:
        candidate = torch.as_tensor(value)
        dtype = torch.complex128 if candidate.is_complex() else torch.float64
    return torch.as_tensor(value, device=device, dtype=dtype)


def real_dtype(dtype):
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    if dtype in (torch.float32, torch.complex64):
        return torch.float32
    raise ValueError("Only float32/64 and complex64/128 are supported")


def complex_dtype(dtype):
    return torch.complex128 if real_dtype(dtype) == torch.float64 else torch.complex64


def to_numpy(value):
    """Explicit, non-differentiable boundary for plotting and CPU preprocessing."""
    if isinstance(value, torch.Tensor):
        return value.detach().resolve_conj().cpu().numpy()
    return value
