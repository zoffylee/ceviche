"""Fixed modes: explicit, nondifferentiable SciPy CPU preprocessing.

Returned Torch profiles use the input device. CUDA inputs travel through CPU.
"""

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch

from .constants import C_0, MU_0
from .derivatives import derivative_entries
from .tensor import as_tensor, to_numpy


def get_modes(eps_cross, omega, dL, npml, m=1, filtering=True):
    eps = as_tensor(eps_cross)
    if eps.requires_grad:
        raise ValueError("Mode preprocessing is not differentiable; pass detached permittivity")
    n = eps.numel()
    if not 1 <= m <= n or int(m) != m:
        raise ValueError("m must be between 1 and the cross-section size")
    operators = derivative_entries(
        omega, (n, 1), (npml, 0), dL, (0.0, 0.0), device="cpu", dtype=torch.complex128
    )
    dxf, dxb = [
        sp.coo_matrix((to_numpy(v), to_numpy(i)), shape=(n, n)).tocsc() for v, i in operators[:2]
    ]
    values = to_numpy(eps).flatten()
    matrix = sp.diags(values) + dxf @ dxb * (C_0 / omega) ** 2
    vals, vecs = solver_eigs(matrix, int(m), guess_value=np.max(values.real))
    order = np.argsort(vals.real)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if filtering:
        vals, vecs = filter_modes(vals, vecs, [lambda v: v.real > 0])
    if vals.size == 0:
        raise RuntimeError("Could not find propagating modes")
    vecs *= np.exp(-1j * np.angle(vecs[np.argmax(np.abs(vecs), axis=0), np.arange(vecs.shape[1])]))
    return (as_tensor(vals, device=eps.device), as_tensor(normalize_modes(vecs), device=eps.device))


def insert_mode(omega, dx, x, y, epsr, target=None, npml=0, m=1, filtering=False):
    eps = as_tensor(epsr)
    if target is None:
        target = torch.zeros_like(eps, dtype=torch.complex128)
    else:
        target = as_tensor(target, device=eps.device).to(torch.complex128).clone()
    _, profiles = get_modes(eps[x, y], omega, dx, npml, m=m, filtering=filtering)
    if profiles.shape[1] < m:
        raise ValueError(f"Only {profiles.shape[1]} modes survived filtering")
    target[x, y] = profiles[:, m - 1].squeeze()
    return target


def solver_eigs(A, Neigs, guess_value=1.0):
    if Neigs >= A.shape[0] - 1:
        values, vectors = scipy.linalg.eig(A.toarray())
        chosen = np.argsort(np.abs(values - guess_value))[:Neigs]
        return values[chosen], vectors[:, chosen]
    shift = guess_value + max(1.0, abs(guess_value)) * 1e-10
    return spl.eigs(A, k=Neigs, sigma=shift, v0=np.ones(A.shape[0]), which="LM")


def filter_modes(values, vectors, filters=None):
    keep = np.ones(values.shape, dtype=bool)
    for fn in filters or []:
        keep &= fn(values)
    return values[keep], vectors[:, keep]


def normalize_modes(vectors):
    if isinstance(vectors, torch.Tensor):
        return vectors / vectors.abs().square().sum(dim=0).sqrt()
    return vectors / np.sqrt(np.sum(np.abs(vectors) ** 2, axis=0))


def Ez_to_H(Ez, omega, dL, npml):
    from .solvers import matvec

    field = as_tensor(Ez).to(torch.complex128)
    operators = derivative_entries(
        omega, (field.numel(), 1), (npml, 0), dL, (0.0, 0.0), device=field.device, dtype=field.dtype
    )
    hx = -matvec(*operators[3], field.flatten()) / (1j * omega * MU_0)
    hy = matvec(*operators[1], field.flatten()) / (1j * omega * MU_0)
    return hx.reshape(field.shape), hy.reshape(field.shape)
