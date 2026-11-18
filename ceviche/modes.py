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


def get_modes(eps_cross, omega, dL, npml, m=1, filtering=True, *, return_neff=False):
    """Return (eps_eff, profiles) for a transverse relative-permittivity profile.

    The discrete Helmholtz operator has eigenvalues lambda = (beta/k0)**2,
    where k0 = omega/C_0. The default first tensor contains these eigenvalues,
    the effective relative permittivities for use directly in an effective-index model.
    With return_neff=True it contains the principal square root n_eff = beta/k0.
    This root has nonnegative real part and positive imaginary part on the
    negative real axis. For an evanescent mode, the direction of decay also
    depends on the sign in the longitudinal propagation factor.

    The solver selects m eigenpairs near the maximum real permittivity and
    orders them by decreasing real(lambda). With filtering=True it keeps
    only real(lambda) > 0. For k retained modes, the tensors have shapes
    (k,) and (eps_cross.numel(), k), with a unit-norm profile in each column.
    SciPy runs on CPU with detached inputs and returns tensors on the input device.
    """
    eps = as_tensor(eps_cross)
    if eps.requires_grad:
        raise ValueError("Mode preprocessing is not differentiable; pass detached permittivity")
    n = eps.numel()
    if not 1 <= m <= n or int(m) != m:
        raise ValueError("m must be between 1 and the cross-section size")
    values = to_numpy(eps).flatten()
    operators = derivative_entries(
        omega, (n, 1), (npml, 0), dL, (0.0, 0.0), device="cpu", dtype=torch.complex128
    )
    dxf, dxb = [
        sp.coo_matrix((to_numpy(v), to_numpy(i)), shape=(n, n)).tocsc() for v, i in operators[:2]
    ]
    matrix = sp.diags(values) + dxf @ dxb * (C_0 / omega) ** 2
    eigenvalues, vecs = solver_eigs(matrix, int(m), guess_value=np.max(np.real(values)))
    order = np.argsort(np.real(eigenvalues))[::-1]
    eigenvalues, vecs = eigenvalues[order], vecs[:, order]
    if filtering:
        eigenvalues, vecs = filter_modes(eigenvalues, vecs, [lambda v: v.real > 0])
    if eigenvalues.size == 0:
        raise RuntimeError("Could not find propagating modes")
    vecs *= np.exp(-1j * np.angle(vecs[np.argmax(np.abs(vecs), axis=0), np.arange(vecs.shape[1])]))
    modal_values = eigenvalues
    if return_neff:
        # A lossless periodic operator is Hermitian. Imaginary roundoff could
        # otherwise select either side of the square-root cut for evanescent modes.
        if npml == 0 and np.all(np.isreal(values)):
            modal_values = eigenvalues.real
        modal_values = np.sqrt(modal_values + 0j)
    return as_tensor(modal_values, device=eps.device), as_tensor(
        normalize_modes(vecs), device=eps.device
    )


def insert_mode(omega, dx, x, y, epsr, target=None, npml=0, m=1, filtering=False):
    """Insert the m-th unit-norm mode profile into the line indexed by (x, y).

    epsr contains relative permittivity. The mode number is one-based and
    follows get_modes ordering and filtering. The result is a complex field
    grid on the input device, initialized from a copy of target or from zeros.
    Use get_modes with return_neff=True to obtain effective indices.
    """
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
