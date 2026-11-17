"""Differentiable COO primitives. Duplicate entries retain their gradients."""

import torch

from .solvers import sparse_solve
from .tensor import as_tensor


def sp_mult(entries, indices, x):
    values = as_tensor(entries)
    x = as_tensor(x, device=values.device)
    indices = torch.as_tensor(indices, dtype=torch.long, device=values.device)
    dtype = torch.promote_types(values.dtype, x.dtype)
    return torch.zeros_like(x, dtype=dtype).index_add(0, indices[0], values * x[indices[1]])


def sp_solve(entries, indices, b):
    return sparse_solve(entries, indices, b)


def spsp_mult(entries_a, indices_a, entries_x, indices_x, N):
    """Multiply square COO matrices without removing structural zero products.

    Enumerates matching intermediate indices; memory scales with actual product
    paths, not N squared. Indices are fixed, values remain differentiable.
    """
    a = as_tensor(entries_a)
    b = as_tensor(entries_x, device=a.device)
    ia = torch.as_tensor(indices_a, dtype=torch.long, device=a.device)
    ib = torch.as_tensor(indices_x, dtype=torch.long, device=a.device)
    values, indices = [], []
    for k in range(N):
        left = torch.where(ia[1] == k)[0]
        right = torch.where(ib[0] == k)[0]
        values.append((a[left, None] * b[right][None, :]).flatten())
        indices.append(
            torch.stack(
                (ia[0, left].repeat_interleave(right.numel()), ib[1, right].repeat(left.numel()))
            )
        )
    if not values:
        return a[:0], ia[:, :0]
    return torch.cat(values), torch.cat(indices, dim=1)
