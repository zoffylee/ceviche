import unittest

import torch

from ceviche.jacobians import jacobian
from ceviche.solvers import SolverOptions, matvec, sparse_solve
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
            _, actual = torch.func.jvp(
                lambda a, b, idx=idx: sparse_solve(a, idx, b), (a, b), (da, db)
            )
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
