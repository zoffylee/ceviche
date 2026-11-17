import math
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ceviche import fdfd_ez, fdtd, viz
from ceviche.filters import make_rho, operator_blur, operator_proj
from ceviche.modes import get_modes, insert_mode
from ceviche.optimizers import adam_optimize
from ceviche.sources import b_TFSF, compute_f
from ceviche.utils import get_spectrum, measure_fields


class TestApplications(unittest.TestCase):
    def test_adam_bounds_callbacks_and_gradient_interfaces(self):
        for mode in [None, True, "callable"]:
            for direction, sign in [("min", 1), ("max", -1)]:
                history = []
                objective = lambda x, sign=sign: sign * (x - 0.8).square().sum()
                jac = (lambda x, sign=sign: 2 * sign * (x - 0.8)) if mode == "callable" else mode
                if mode is True:
                    objective = lambda x, sign=sign: (
                        sign * (x - 0.8).square().sum(),
                        2 * sign * (x - 0.8),
                    )
                result, losses = adam_optimize(
                    objective,
                    [0.1, 0.2],
                    jac,
                    Nsteps=30,
                    step_size=0.08,
                    bounds=(0.0, 0.6),
                    direction=direction,
                    verbose=False,
                    callback=lambda i, _h, _p, history=history: history.append(i),
                )
                self.assertEqual(history, list(range(30)))
                self.assertEqual(len(losses), 30)
                torch.testing.assert_close(result, torch.full_like(result, 0.6))

    def test_filter_fdfd_optimization(self):
        shape = (6, 5)
        rho = torch.full(shape, 0.5, dtype=torch.float64, requires_grad=True)
        mask = torch.ones_like(rho)

        def fn(p):
            eps = 1 + operator_proj(make_rho(p, mask, 1), beta=3)
            sim = fdfd_ez(2 * math.pi * 200e12, 80e-9, eps, [2, 2])
            src = torch.zeros_like(eps)
            src[2, 2] = 1
            return sim.solve(src)[2].abs().square().mean()

        initial = fn(rho).item()
        result, _ = adam_optimize(fn, rho, Nsteps=3, bounds=(0, 1), verbose=False)
        self.assertLess(fn(result).item(), initial)
        self.assertTrue(torch.autograd.gradcheck(lambda x: operator_blur(x, 1), (rho,)))
        torch.testing.assert_close(operator_blur(rho, 0), rho)

    def test_modes_and_plotting_tensor_boundaries(self):
        eps = torch.ones(20, dtype=torch.float64)
        eps[7:13] = 4
        vals, profiles = get_modes(eps, 2 * math.pi * 200e12, 80e-9, 2, m=2)
        self.assertEqual(vals.numel(), 2)
        torch.testing.assert_close(
            profiles.abs().square().sum(0), torch.ones(2, dtype=torch.float64)
        )
        with self.assertRaises(ValueError):
            get_modes(eps.requires_grad_(), 2 * math.pi * 200e12, 80e-9, 2)
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            grid = eps.detach().repeat(5, 1).to(device)
            source = insert_mode(2 * math.pi * 200e12, 80e-9, 2, slice(None), grid)
            self.assertEqual(source.device, grid.device)
            axis = viz.abs(source.requires_grad_(), outline=grid)
            self.assertEqual(len(axis.images), 1)
            plt.close("all")

    def test_spectrum_time_axis_and_gradients(self):
        count, frequency = 64, 8
        t = torch.arange(count, dtype=torch.float64)
        signal = torch.cos(2 * math.pi * frequency * t / count).requires_grad_()
        freqs, spectrum = get_spectrum(signal, 1 / count)
        self.assertEqual(spectrum.shape, (count, 1))
        self.assertEqual(freqs[spectrum[: count // 2, 0].abs().argmax()].item(), frequency)
        spectrum.abs().square().sum().backward()
        assert signal.grad is not None
        self.assertTrue(torch.isfinite(signal.grad).all())
        eps = torch.full((4, 4, 1), 2.0, dtype=torch.float64, requires_grad=True)
        sim = fdtd(eps, 80e-9, [1, 1, 0])
        measured = measure_fields(sim, lambda _t: torch.ones_like(eps), 3, torch.ones_like(eps))
        self.assertEqual(measured.shape, (3, 1))
        measured.sum().backward()
        assert eps.grad is not None
        self.assertGreater(eps.grad.abs().sum(), 0)

    def test_tfsf_commutator_and_uniform_mode(self):
        omega, dl = 2 * math.pi * 200e12, 80e-9
        vals, _ = get_modes(torch.full((20,), 2.0, dtype=torch.float64), omega, dl, 0)
        torch.testing.assert_close(vals.real, torch.tensor([2.0], dtype=torch.float64))
        eps = torch.full((6, 5), 2.0, dtype=torch.float64)
        sim = fdfd_ez(omega, dl, eps, [2, 2])
        mask = torch.zeros_like(eps)
        mask[2:4, 2:4] = 1
        angle = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        source = b_TFSF(sim, mask, angle)
        from ceviche.constants import C_0

        values, indices = sim._make_A(eps.flatten())
        matrix = torch.sparse_coo_tensor(indices, values, (30, 30)).to_dense()
        q = torch.diag(1 - mask.flatten()).to(matrix.dtype)
        plane = compute_f(angle, 2 * math.pi * C_0 / omega, dl, eps.shape)
        expected = (q @ matrix - matrix @ q) @ plane / (1j * omega)
        torch.testing.assert_close(source.flatten(), expected)
        source.abs().square().sum().backward()
        assert angle.grad is not None
        self.assertTrue(torch.isfinite(angle.grad))


if __name__ == "__main__":
    unittest.main()
