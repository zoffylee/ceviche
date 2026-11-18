import math
import unittest

import torch

from ceviche.constants import EPSILON_0, MU_0
from ceviche.fdfd import fdfd_ez, fdfd_hz, fdfd_mf_ez
from ceviche.solvers import SolverOptions


class TestTorchFDFD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.eps = 1.5 + torch.rand((6, 5), dtype=torch.float64)
        self.src = torch.zeros((6, 5), dtype=torch.complex128)
        self.src[2, 2] = torch.tensor(1 + 0.2j, dtype=self.src.dtype)
        self.omega, self.dl = 2 * math.pi * 200e12, 80e-9

    def check_direction(self, fn, value):
        value = value.detach().requires_grad_()
        direction = torch.randn_like(value)
        loss = fn(value)
        (gradient,) = torch.autograd.grad(loss, value)
        reverse = (gradient.conj() * direction).real.sum()
        forward = torch.func.jvp(fn, (value,), (direction,))[1]
        h = 1e-5
        numerical = (fn(value + h * direction) - fn(value - h * direction)) / (2 * h)
        torch.testing.assert_close(reverse, forward, rtol=2e-6, atol=1e-12)
        torch.testing.assert_close(reverse, numerical, rtol=2e-5, atol=1e-10)

    def test_real_and_complex_material_and_source_gradients(self):
        for cls in [fdfd_ez, fdfd_hz]:
            for complex_material in [False, True]:
                for pml, phase in [([2, 2], [0.0, 0.0]), ([0, 0], [0.23, -0.17])]:
                    with self.subTest(cls=cls.__name__, complex=complex_material, pml=pml):
                        material = self.eps + (0.03j if complex_material else 0)
                        sim = cls(self.omega, self.dl, material, pml, bloch_phases=phase)

                        def objective(eps, sim=sim):
                            sim.eps_r = eps
                            fields = sim.solve(eps * self.src)
                            return sum(f.abs().square().sum() for f in fields)

                        self.check_direction(objective, material)
                        sim.eps_r = material
                        self.check_direction(
                            lambda src, sim=sim: sum(
                                f.abs().square().sum() for f in sim.solve(src)
                            ),
                            self.src,
                        )

    def test_high_contrast_hz_fields(self):
        eps = torch.ones_like(self.eps)
        eps[1:4, 2:4] = 3.48**2
        hz = torch.randn_like(self.src)
        # Manufacture a periodic source from the two discrete curl equations.
        iw = 1j * self.omega
        eps_x = (eps + eps.roll(1, 1)) / 2
        eps_y = (eps + eps.roll(1, 0)) / 2
        ex = (hz.roll(-1, 1) - hz) / (self.dl * iw * EPSILON_0 * eps_x)
        ey = -(hz.roll(-1, 0) - hz) / (self.dl * iw * EPSILON_0 * eps_y)
        source = (ex - ex.roll(1, 1) - ey + ey.roll(1, 0)) / self.dl - iw * MU_0 * hz
        fields = fdfd_hz(self.omega, self.dl, eps, [0, 0]).solve(source)
        for actual, expected in zip(fields, (ex, ey, hz)):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_hz_singular_edge_permittivity(self):
        sim = fdfd_hz(self.omega, self.dl, self.eps, [0, 0])
        for axis in (0, 1):
            with self.subTest(axis=axis):
                eps = torch.ones_like(self.eps)
                eps.select(axis, 1).fill_(-1)
                sim.eps_r = eps
                with self.assertRaisesRegex(ValueError, "edge.*permittivity"):
                    sim.solve(self.src)
        for value in (0.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                sim.eps_r = torch.full_like(self.eps, value)
                with self.assertRaisesRegex(ValueError, "edge.*permittivity"):
                    sim.solve(self.src)

    def test_1d_material_gradients(self):
        eps = self.eps[:, :1].clone()
        eps[2:4] = 3.48**2
        src = self.src[:, 2:3]
        for cls in (fdfd_ez, fdfd_hz):
            with self.subTest(cls=cls.__name__):
                sim = cls(self.omega, self.dl, eps, [0, 0], bloch_phases=[0.0, -0.3])

                def objective(material, sim=sim):
                    sim.eps_r = material
                    return sum(field.abs().square().sum() for field in sim.solve(src))

                self.check_direction(objective, eps)

    def test_multifrequency_design_gradients_and_uncoupled_limit(self):
        src = torch.stack((self.src * 0, self.src, self.src * 0))
        params = torch.cat((self.eps.flatten(), torch.full((30,), 0.08), torch.full((30,), 0.3)))

        def objective(p):
            sim = fdfd_mf_ez(
                self.omega,
                self.dl,
                p[:30].reshape(6, 5),
                0.1 * self.omega,
                p[30:60].reshape(1, 6, 5),
                p[60:].reshape(1, 6, 5),
                1,
                [2, 2],
            )
            return sim.solve(src)[2].abs().square().sum()

        self.check_direction(objective, params)
        zero = torch.zeros((1, 6, 5), dtype=torch.float64)
        mf = fdfd_mf_ez(self.omega, self.dl, self.eps, 0.1 * self.omega, zero, zero, 1, [2, 2])
        single = fdfd_ez(self.omega, self.dl, self.eps, [2, 2])
        for bands, field in zip(mf.solve(src), single.solve(self.src)):
            torch.testing.assert_close(bands[1], field)
            torch.testing.assert_close(bands[[0, 2]], torch.zeros_like(bands[[0, 2]]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cpu_cuda_fields_and_gradients(self):
        for cls in [fdfd_ez, fdfd_hz, fdfd_mf_ez]:
            results = []
            for device in ["cpu", "cuda"]:
                eps = self.eps.to(device).detach().requires_grad_()
                if issubclass(cls, fdfd_mf_ez):
                    delta = torch.full((1, 6, 5), 0.08, device=device, dtype=eps.dtype)
                    sim = cls(self.omega, self.dl, eps, 0.1 * self.omega, delta, delta, 1, [2, 2])
                    src = torch.stack((self.src * 0, self.src, self.src * 0))
                else:
                    sim = cls(self.omega, self.dl, eps, [2, 2])
                    src = self.src
                fields = sim.solve(src)
                (grad,) = torch.autograd.grad(
                    torch.stack([f.abs().square().sum() for f in fields]).sum(), eps
                )
                self.assertEqual(fields[0].device, eps.device)
                results.append([f.detach().cpu() for f in (*fields, grad)])
            for cpu, gpu in zip(*results):
                torch.testing.assert_close(cpu, gpu, rtol=2e-6, atol=1e-12)

    def test_single_precision_and_validation(self):
        sim = fdfd_ez(self.omega, self.dl, self.eps.float(), [2, 2])
        self.assertEqual(sim.solve(self.src)[2].dtype, torch.complex64)
        with self.assertRaises(ValueError):
            sim.solve(torch.zeros(3))
        with self.assertRaises(ValueError):
            sim.eps_r = torch.ones((2, 2))
        with self.assertRaises(RuntimeError):
            fdfd_ez(
                self.omega,
                self.dl,
                self.eps,
                [2, 2],
                solver_options=SolverOptions(method="gmres", maxiter=1),
            ).solve(self.src)

    def test_direct_and_bicgstab_fields_and_gradients(self):
        for cls in (fdfd_ez, fdfd_hz, fdfd_mf_ez):
            cases = [("cpu", "scipy"), ("cpu", "pardiso"), ("cpu", "bicgstab")]
            if torch.cuda.is_available():
                cases.append(("cuda", "bicgstab"))
            reference = None
            for device, method in cases:
                with self.subTest(polarization=cls.__name__, device=device, method=method):
                    eps = self.eps.to(device).detach().requires_grad_()
                    options = SolverOptions(method=method)
                    if cls is fdfd_mf_ez:
                        delta = torch.full((1, 6, 5), 0.08, device=device, dtype=eps.dtype)
                        sim = cls(
                            self.omega,
                            self.dl,
                            eps,
                            0.1 * self.omega,
                            delta,
                            delta,
                            1,
                            [2, 2],
                            solver_options=options,
                        )
                        src = torch.stack((self.src * 0, self.src, self.src * 0))
                    else:
                        sim = cls(self.omega, self.dl, eps, [2, 2], solver_options=options)
                        src = self.src
                    fields = sim.solve(src)
                    (gradient,) = torch.autograd.grad(
                        sum(field.abs().square().sum() for field in fields), eps
                    )
                    actual = [field.detach().cpu() for field in (*fields, gradient)]
                    if reference is None:
                        reference = actual
                    else:
                        for expected, result in zip(reference, actual):
                            torch.testing.assert_close(result, expected, rtol=2e-6, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
