"""Sparse solves with explicit CPU/GPU execution and implicit differentiation.

CPU defaults to PARDISO for double precision and SciPy for single precision.
CUDA supports Torch GMRES, CG and BiCGSTAB. No dense fallback or implicit device
transfer is performed.
"""

import math
import warnings
from dataclasses import dataclass, replace

import torch

from .tensor import as_tensor, to_numpy

ITERATIVE_METHODS = ("gmres", "bicgstab", "cg")


@dataclass(frozen=True)
class SolverOptions:
    method: str = "auto"
    rtol: float = 1e-10
    atol: float = 0.0
    maxiter: int = 2000
    restart: int = 80

    def __post_init__(self):
        if self.method not in ("auto", "pardiso", "scipy", "dense", "native", *ITERATIVE_METHODS):
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


def _residual_norm(vector):
    """Accumulate physical RHS and residual norms in double precision."""
    dtype = torch.complex128 if vector.is_complex() else torch.float64
    return torch.linalg.vector_norm(vector, dtype=dtype)


def _jacobi_inverse(values, indices, b):
    diag = torch.zeros_like(b).index_add(
        0, indices[0], torch.where(indices[0] == indices[1], values, 0)
    )
    return torch.where(diag.abs() > torch.finfo(b.real.dtype).tiny, 1 / diag, 1)


def _cg(values, indices, b, options):
    """Preconditioned CG for Hermitian positive-definite matrices only."""
    x = torch.zeros_like(b)
    residual = b.clone()
    threshold = options.atol + options.rtol * _residual_norm(b)
    if _residual_norm(residual) <= threshold:
        return x
    inv_diag = _jacobi_inverse(values, indices, b)
    z = inv_diag * residual
    direction = z.clone()
    rho = torch.vdot(residual, z)
    for _ in range(options.maxiter):
        product = matvec(values, indices, direction)
        curvature = torch.vdot(direction, product)
        if not torch.isfinite(curvature) or curvature.real <= 0 or rho == 0:
            break
        alpha = rho / curvature
        x = x + alpha * direction
        residual = residual - alpha * product
        if _residual_norm(residual) <= threshold:
            residual = b - matvec(values, indices, x)
            if _residual_norm(residual) <= threshold:
                return x
            # Restart after residual replacement to avoid accumulated roundoff.
            z = inv_diag * residual
            direction = z.clone()
            rho = torch.vdot(residual, z)
            continue
        z = inv_diag * residual
        next_rho = torch.vdot(residual, z)
        direction = z + (next_rho / rho) * direction
        rho = next_rho
    raise RuntimeError("CG failed to converge; the matrix must be Hermitian positive definite")


def _bicgstab(values, indices, b, options):
    """Right Jacobi-preconditioned BiCGSTAB for general complex matrices."""
    x = torch.zeros_like(b)
    residual = b.clone()
    shadow = residual.clone()
    threshold = options.atol + options.rtol * _residual_norm(b)
    if _residual_norm(residual) <= threshold:
        return x
    inv_diag = _jacobi_inverse(values, indices, b)
    direction = torch.zeros_like(b)
    product = torch.zeros_like(b)
    rho_prev = alpha = omega = b.new_ones(())
    for _ in range(options.maxiter):
        rho = torch.vdot(shadow, residual)
        if not torch.isfinite(rho) or rho == 0 or omega == 0:
            break
        beta = (rho / rho_prev) * (alpha / omega)
        direction = residual + beta * (direction - omega * product)
        conditioned = inv_diag * direction
        product = matvec(values, indices, conditioned)
        denominator = torch.vdot(shadow, product)
        if not torch.isfinite(denominator) or denominator == 0:
            break
        alpha = rho / denominator
        intermediate = residual - alpha * product
        candidate = x + alpha * conditioned
        if _residual_norm(b - matvec(values, indices, candidate)) <= threshold:
            return candidate
        conditioned_s = inv_diag * intermediate
        t = matvec(values, indices, conditioned_s)
        norm_t_squared = torch.vdot(t, t).real
        if not torch.isfinite(norm_t_squared) or norm_t_squared == 0:
            break
        omega = torch.vdot(t, intermediate) / norm_t_squared
        x = candidate + omega * conditioned_s
        if _residual_norm(b - matvec(values, indices, x)) <= threshold:
            return x
        residual = intermediate - omega * t
        rho_prev = rho
    raise RuntimeError("BiCGSTAB failed to converge or encountered a breakdown")


def _gmres(values, indices, b, options):
    n = b.numel()
    norm_b = _residual_norm(b)
    threshold = options.atol + options.rtol * norm_b
    if norm_b == 0:
        return torch.zeros_like(b)
    inv_diag = _jacobi_inverse(values, indices, b)
    apply = lambda x: inv_diag * matvec(values, indices, x)
    rhs = inv_diag * b
    x = torch.zeros_like(b)
    used = 0
    while used < options.maxiter:
        residual = rhs - apply(x)
        beta = torch.linalg.vector_norm(residual)
        if _residual_norm(b - matvec(values, indices, x)) <= threshold:
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
                if _residual_norm(b - matvec(values, indices, candidate)) <= threshold:
                    return candidate
            if breakdown:
                break
        x = candidate
    residual = _residual_norm(b - matvec(values, indices, x)) / norm_b
    raise RuntimeError(
        f"GMRES failed after {used} iterations; relative residual={residual.item():.3e}"
    )


def _solve_raw(values, indices, b, options):
    method = options.method
    if method == "auto":
        if b.is_cuda:
            method = "gmres"
        else:
            method = "pardiso" if b.dtype in (torch.float64, torch.complex128) else "scipy"
    cpu_iterative = method in ITERATIVE_METHODS and method != "gmres"
    if method in ("pardiso", "scipy") or (cpu_iterative and b.device.type == "cpu"):
        if b.device.type != "cpu":
            raise ValueError(f"The {method} solver is CPU-only; select gmres for CUDA")
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl

        matrix = sp.coo_matrix((to_numpy(values), to_numpy(indices)), shape=(b.numel(), b.numel()))
        matrix = matrix.tocsr() if method == "pardiso" else matrix.tocsc()
        if method == "pardiso":
            if b.dtype not in (torch.float64, torch.complex128):
                raise ValueError(
                    "py-mkl-pardiso computes in float64/complex128 only; "
                    "use auto or scipy for single precision"
                )
            from pymklpardiso import MTYPE_COMPLEX_NONSYM, MTYPE_REAL_NONSYM, PardisoSolver

            matrix.sum_duplicates()
            matrix.sort_indices()
            mtype = MTYPE_COMPLEX_NONSYM if b.is_complex() else MTYPE_REAL_NONSYM
            solver = PardisoSolver(matrix, mtype)
            try:
                result = solver.solve(to_numpy(b))
            finally:
                solver.release()
        elif method == "scipy":
            with warnings.catch_warnings():
                warnings.simplefilter("error", spl.MatrixRankWarning)
                try:
                    result = spl.spsolve(matrix, to_numpy(b))
                except spl.MatrixRankWarning as exc:
                    raise RuntimeError("Singular linear system") from exc
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                try:
                    result, info = getattr(spl, method)(
                        matrix,
                        to_numpy(b),
                        rtol=options.rtol,
                        atol=options.atol,
                        maxiter=options.maxiter,
                    )
                except RuntimeWarning as exc:
                    raise RuntimeError(f"{method} encountered a numerical breakdown") from exc
            if info != 0:
                raise RuntimeError(f"{method} failed to converge or broke down (info={info})")
        x = torch.as_tensor(result, dtype=b.dtype, device=b.device)
    elif method == "gmres":
        x = _gmres(values, indices, b, options)
    elif method == "cg":
        x = _cg(values, indices, b, options)
    elif method == "bicgstab":
        x = _bicgstab(values, indices, b, options)
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
    residual = _residual_norm(matvec(values, indices, x) - b)
    bound = options.atol + options.rtol * _residual_norm(b)
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
        # Implicit derivatives may have sparse/basis right-hand sides for which
        # BiCGSTAB breaks down even though the primal solve converged. The
        # derivative equations are solver-independent; use GMRES for these RHSs.
        ctx.options = replace(options, method="gmres") if options.method == "bicgstab" else options
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
        method = "gmres" if iterative_method is True else iterative_method
        if method not in ITERATIVE_METHODS:
            raise ValueError(f"Unknown iterative solver: {method}; choose from {ITERATIVE_METHODS}")
        options = SolverOptions(method=method)
    values, indices = get_entries_indices(A)
    return sparse_solve(values, indices, b, options=options)
