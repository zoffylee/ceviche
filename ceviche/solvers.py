"""Sparse solves with explicit CPU/GPU execution and implicit differentiation.

The CPU direct path uses SciPy. CUDA uses restarted, Jacobi-preconditioned GMRES
entirely on device. No dense fallback or implicit device transfer is performed.
"""

import math
import warnings
from dataclasses import dataclass

import torch

from .tensor import as_tensor, to_numpy


@dataclass(frozen=True)
class SolverOptions:
    method: str = "auto"
    rtol: float = 1e-10
    atol: float = 0.0
    maxiter: int = 2000
    restart: int = 80

    def __post_init__(self):
        if self.method not in ("auto", "scipy", "gmres", "dense", "native"):
            raise ValueError(f"Unknown solver method: {self.method}")
        if (
            not math.isfinite(self.rtol)
            or not math.isfinite(self.atol)
            or self.rtol <= 0
            or self.atol < 0
            or self.maxiter < 1
            or self.restart < 1
            or not isinstance(self.maxiter, int)
            or not isinstance(self.restart, int)
        ):
            raise ValueError("Invalid solver tolerances or iteration limits")


def matvec(values, indices, x):
    """COO matvec, including duplicate and structural-zero entries."""
    return torch.zeros_like(x).index_add(0, indices[0], values * x[indices[1]])


def _gmres(values, indices, b, options):
    n = b.numel()
    norm_b = torch.linalg.vector_norm(b)
    threshold = options.atol + options.rtol * norm_b
    if norm_b == 0:
        return torch.zeros_like(b)
    diag = torch.zeros_like(b).index_add(
        0, indices[0], torch.where(indices[0] == indices[1], values, 0)
    )
    inv_diag = torch.where(diag.abs() > torch.finfo(b.real.dtype).tiny, 1 / diag, 1)
    apply = lambda x: inv_diag * matvec(values, indices, x)
    rhs = inv_diag * b
    x = torch.zeros_like(b)
    used = 0
    while used < options.maxiter:
        residual = rhs - apply(x)
        beta = torch.linalg.vector_norm(residual)
        if torch.linalg.vector_norm(b - matvec(values, indices, x)) <= threshold:
            return x
        if not torch.isfinite(beta) or beta == 0:
            break
        m = min(options.restart, n, options.maxiter - used)
        basis = [residual / beta]
        h = torch.zeros((m + 1, m), device=b.device, dtype=b.dtype)
        target = torch.zeros(m + 1, device=b.device, dtype=b.dtype)
        target[0] = beta
        candidate = x
        for j in range(m):
            v = apply(basis[j])
            # Twice modified Gram-Schmidt limits loss of orthogonality.
            for _ in range(2):
                for i in range(j + 1):
                    coefficient = torch.vdot(basis[i], v)
                    h[i, j] += coefficient
                    v = v - coefficient * basis[i]
            h[j + 1, j] = torch.linalg.vector_norm(v)
            used += 1
            breakdown = h[j + 1, j].abs() <= torch.finfo(b.real.dtype).eps
            if not breakdown:
                basis.append(v / h[j + 1, j])
            if breakdown or (j + 1) % 8 == 0 or j == m - 1:
                coefficients = torch.linalg.lstsq(
                    h[: j + 2, : j + 1], target[: j + 2], driver="gels"
                ).solution
                candidate = x + torch.stack(basis[: j + 1], dim=1) @ coefficients
                if torch.linalg.vector_norm(b - matvec(values, indices, candidate)) <= threshold:
                    return candidate
            if breakdown:
                break
        x = candidate
    residual = torch.linalg.vector_norm(b - matvec(values, indices, x)) / norm_b
    raise RuntimeError(
        f"GMRES failed after {used} iterations; relative residual={residual.item():.3e}"
    )


def _solve_raw(values, indices, b, options):
    method = options.method
    if method == "auto":
        method = "gmres" if b.is_cuda else "scipy"
    if method == "scipy":
        if b.device.type != "cpu":
            raise ValueError("The scipy solver is CPU-only; select gmres for CUDA")
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl

        matrix = sp.coo_matrix(
            (to_numpy(values), to_numpy(indices)), shape=(b.numel(), b.numel())
        ).tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter("error", spl.MatrixRankWarning)
            try:
                result = spl.spsolve(matrix, to_numpy(b))
            except spl.MatrixRankWarning as exc:
                raise RuntimeError("Singular linear system") from exc
        x = torch.as_tensor(result, dtype=b.dtype, device=b.device)
    elif method == "gmres":
        x = _gmres(values, indices, b, options)
    else:
        matrix = torch.sparse_coo_tensor(indices, values, (b.numel(), b.numel())).coalesce()
        if method == "dense":
            if b.numel() > 512:
                raise ValueError("Dense solver is limited to 512 unknowns for validation")
            x = torch.linalg.solve(matrix.to_dense(), b)
        else:
            x = torch.sparse.spsolve(matrix.to_sparse_csr(), b)
    if not torch.isfinite(x).all():
        raise RuntimeError("Linear solve returned non-finite values")
    residual = torch.linalg.vector_norm(matvec(values, indices, x) - b)
    bound = options.atol + options.rtol * torch.linalg.vector_norm(b)
    if residual > bound:
        raise RuntimeError(
            f"Linear solve residual {residual.item():.3e} exceeds {bound.item():.3e}"
        )
    return x


@torch.autograd.function.once_differentiable
def _solve_backward(ctx, grad_output):
    values, indices, x = ctx.saved_tensors
    adjoint = _SparseSolve.apply(values.conj(), indices.flip(0), grad_output, ctx.options)
    grad_values = -adjoint[indices[0]] * x[indices[1]].conj()
    return grad_values, None, adjoint, None


class _SparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(values, indices, b, options):
        return _solve_raw(values, indices, b, options)

    @staticmethod
    def setup_context(ctx, inputs, output):
        values, indices, _b, options = inputs
        ctx.options = options
        ctx.save_for_backward(values, indices, output)
        ctx.save_for_forward(values, indices, output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        return _solve_backward(ctx, grad_output)

    @staticmethod
    def jvp(ctx, *grad_inputs):
        dvalues, _dindices, db, _doptions = grad_inputs
        values, indices, x = ctx.saved_tensors
        rhs = torch.zeros_like(x) if db is None else db
        if dvalues is not None:
            rhs = rhs - matvec(dvalues, indices, x)
        return _SparseSolve.apply(values, indices, rhs, ctx.options)


def sparse_solve(values, indices, b, *, options=None):
    values = as_tensor(values)
    b = as_tensor(b, device=values.device)
    dtype = torch.promote_types(values.dtype, b.dtype)
    values, b = values.to(dtype), b.to(dtype)
    indices = torch.as_tensor(indices, device=values.device, dtype=torch.long)
    if values.ndim != 1 or b.ndim != 1 or indices.shape != (2, values.numel()):
        raise ValueError("Expected values (nnz,), indices (2, nnz), and b (n,)")
    if b.numel() == 0 or (indices.numel() and (indices.min() < 0 or indices.max() >= b.numel())):
        raise ValueError("Sparse indices are outside the system shape")
    if not torch.isfinite(values).all() or not torch.isfinite(b).all():
        raise ValueError("Linear system contains NaN or Inf")
    return _SparseSolve.apply(values, indices, b, options or SolverOptions())


def solve_linear(A, b, iterative_method=False, *, options=None):
    """Solve a Torch sparse matrix, or explicitly CPU-based SciPy input."""
    from .utils import get_entries_indices

    if options is not None and iterative_method:
        raise ValueError("Pass options or iterative_method, not both")
    if iterative_method:
        if iterative_method not in (True, "gmres"):
            raise ValueError("The Torch iterative solver supports gmres")
        options = SolverOptions(method="gmres")
    values, indices = get_entries_indices(A)
    return sparse_solve(values, indices, b, options=options)
