import unittest

import torch

from ceviche.jacobians import jacobian
from ceviche.solvers import SolverOptions, matvec, solve_linear, sparse_solve
from ceviche.tensor import as_tensor


class TestTorchSparseSolve(unittest.TestCase):
    def system(self, device):
        indices = torch.tensor([[0, 0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 0, 2]], device=device)
        values = torch.tensor(
            [2.0, 1.0 + 0.2j, 0.1j, 4.0, 0.2, -0.1j, 3.0],
            dtype=torch.complex128,
            device=device,
            requires_grad=True,
        )
        b = torch.tensor(
            [1.0 + 0.2j, 0.3, -0.7j], dtype=torch.complex128, device=device, requires_grad=True
        )
        return values, indices, b

    def test_solve_and_complex_gradients(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                a, idx, b = self.system(device)
                x = sparse_solve(a, idx, b)
                torch.testing.assert_close(matvec(a, idx, x), b, atol=1e-9, rtol=1e-9)
                self.assertEqual(x.device, b.device)
                self.assertTrue(
                    torch.autograd.gradcheck(
                        lambda a, b, idx=idx: sparse_solve(a, idx, b),
                        (a, b),
                        check_forward_ad=True,
                        atol=1e-5,
                        rtol=1e-4,
                    )
                )

    def test_jvp_matches_dense(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            a, idx, b = self.system(device)
            da, db = torch.ones_like(a) * 0.1, torch.ones_like(b) * 0.2j
            actual = torch.func.jvp(
                lambda a, b, idx=idx: sparse_solve(a, idx, b), (a, b), (da, db)
            )[1]
            matrix = torch.sparse_coo_tensor(idx, a, (3, 3)).to_dense()
            x = torch.linalg.solve(matrix, b)
            expected = torch.linalg.solve(matrix, db - matvec(da, idx, x))
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)

    def test_solver_failures_and_zero_source(self):
        a, idx, b = self.system("cpu")
        torch.testing.assert_close(sparse_solve(a, idx, b * 0), b * 0)
        with self.assertRaises(RuntimeError):
            sparse_solve(a * 0, idx, b)
        with self.assertRaises(RuntimeError):
            sparse_solve(a, idx, b, options=SolverOptions(method="gmres", maxiter=1, restart=1))
        with self.assertRaises(ValueError):
            sparse_solve(a * float("nan"), idx, b)
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                sparse_solve(a.cuda(), idx.cuda(), b.cuda(), options=SolverOptions(method="scipy"))

    def test_conversion_preserves_graph(self):
        x = torch.ones(3, requires_grad=True)
        as_tensor(x, dtype=torch.float64).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_cpu_direct_backends_precision_and_gradients(self):
        for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            double = dtype in (torch.float64, torch.complex128)
            for method in ("auto", "scipy", "pardiso"):
                with self.subTest(dtype=dtype, method=method):
                    a, idx, b = self.system("cpu")
                    if not dtype.is_complex:
                        a, b = a.real, b.real
                    a = a.detach().to(dtype).requires_grad_()
                    b = b.detach().to(dtype).requires_grad_()
                    options = SolverOptions(method=method, rtol=1e-10 if double else 1e-5)
                    if method == "pardiso" and not double:
                        with self.assertRaisesRegex(ValueError, "single precision"):
                            sparse_solve(a, idx, b, options=options)
                        continue
                    actual = sparse_solve(a, idx, b, options=options)
                    matrix = torch.sparse_coo_tensor(idx, a, (3, 3)).to_dense()
                    expected = torch.linalg.solve(matrix, b)
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, expected)
                    actual_grad = torch.autograd.grad(actual.abs().square().sum(), (a, b))
                    expected_grad = torch.autograd.grad(expected.abs().square().sum(), (a, b))
                    for result, reference in zip(actual_grad, expected_grad):
                        torch.testing.assert_close(result, reference)

    def test_pardiso_rejects_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        a, idx, b = self.system("cuda")
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            sparse_solve(a, idx, b, options=SolverOptions(method="pardiso"))

    def test_iterative_methods_and_derivatives(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for method in ("gmres", "bicgstab", "cg"):
                with self.subTest(device=device, method=method):
                    a, idx, b = self.system(device)
                    base = torch.sparse_coo_tensor(idx, a, (3, 3)).to_dense().detach()
                    # CG requires an HPD matrix. Vary it through an HPD parameterization
                    # so finite differences also remain inside the solver's domain.
                    if method == "cg":
                        base = base.mH @ base + torch.eye(3, device=device)
                    rows, cols = torch.meshgrid(
                        torch.arange(3, device=device),
                        torch.arange(3, device=device),
                        indexing="ij",
                    )
                    idx = torch.stack((rows.flatten(), cols.flatten()))
                    design = torch.ones(3, dtype=torch.float64, device=device, requires_grad=True)
                    options = SolverOptions(method=method)

                    def solve(design, rhs, base=base, idx=idx, options=options):
                        matrix = base + torch.diag(design)
                        return sparse_solve(matrix.flatten(), idx, rhs, options=options)

                    actual = solve(design, b)
                    matrix = base + torch.diag(design)
                    expected = torch.linalg.solve(matrix, b)
                    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
                    self.assertEqual(actual.device, b.device)
                    legacy_entry = solve_linear(
                        matrix.detach().to_sparse(), b, iterative_method=method
                    )
                    torch.testing.assert_close(legacy_entry, expected, atol=1e-9, rtol=1e-9)
                    self.assertTrue(
                        torch.autograd.gradcheck(
                            solve, (design, b), check_forward_ad=True, atol=1e-5, rtol=1e-4
                        )
                    )
                    tangent = (torch.ones_like(design) * 0.1, torch.ones_like(b) * 0.2j)
                    actual_jvp = torch.func.jvp(solve, (design, b), tangent)[1]
                    expected_jvp = torch.linalg.solve(matrix, tangent[1] - tangent[0] * expected)
                    torch.testing.assert_close(actual_jvp, expected_jvp, atol=1e-9, rtol=1e-9)

    def test_iterative_failures_and_zero_source(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for method in ("gmres", "bicgstab", "cg"):
                with self.subTest(device=device, method=method):
                    a, idx, b = self.system(device)
                    matrix = torch.sparse_coo_tensor(idx, a, (3, 3)).to_dense().detach()
                    matrix = matrix.mH @ matrix + torch.eye(3, device=device)
                    sparse = matrix.to_sparse().coalesce()
                    a, idx = sparse.values(), sparse.indices()
                    options = SolverOptions(method=method)
                    torch.testing.assert_close(sparse_solve(a, idx, b * 0, options=options), b * 0)
                    with self.assertRaises(RuntimeError):
                        sparse_solve(a * 0, idx, b, options=options)
                    with self.assertRaises(RuntimeError):
                        sparse_solve(
                            a, idx, b, options=SolverOptions(method=method, maxiter=1, restart=1)
                        )
        with self.assertRaises(ValueError):
            SolverOptions(method="unknown")

    def test_jacobian_complex_coordinates(self):
        a, idx, b = self.system("cpu")
        fn = lambda value: sparse_solve(value, idx, b).abs().square()
        reverse = jacobian(fn)(a)
        forward = jacobian(fn, mode="forward")(a)
        numeric = jacobian(fn, mode="numerical")(a)
        self.assertEqual(reverse.shape, (3, 2 * a.numel()))
        torch.testing.assert_close(reverse, forward, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(reverse, numeric, atol=1e-7, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
