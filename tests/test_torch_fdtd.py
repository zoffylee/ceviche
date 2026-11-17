import unittest

import torch

from ceviche.torch_fdtd import fdtd


class TestTorchFDTD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.eps = 1.5 + torch.rand((6, 5, 3), dtype=torch.float64)
        self.sources = torch.randn((12, 6, 5, 3), dtype=torch.float64) * 0.1

    def objective(self, eps, source, checkpoint_steps=0, field="Ez"):
        sim = fdtd(eps, 80e-9, [2, 2, 1])
        return sim.run({"Jz": source}, checkpoint_steps=checkpoint_steps)[field].square().sum()

    def test_electric_magnetic_forward_reverse_and_numerical_gradients(self):
        for field in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            eps = self.eps.detach().requires_grad_()
            source = self.sources.detach().requires_grad_()
            direction, ds = torch.randn_like(eps), torch.randn_like(source)
            fn = lambda e, s, field=field: self.objective(e, s, field=field)
            loss = fn(eps, source)
            ge, gs = torch.autograd.grad(loss, (eps, source))
            reverse = (ge * direction).sum() + (gs * ds).sum()
            _, forward = torch.func.jvp(fn, (eps, source), (direction, ds))
            h = 1e-5
            numerical = (
                fn(eps + h * direction, source + h * ds) - fn(eps - h * direction, source - h * ds)
            ) / (2 * h)
            torch.testing.assert_close(reverse, forward, rtol=1e-8, atol=1e-12)
            torch.testing.assert_close(reverse, numerical, rtol=1e-5, atol=1e-10)

    def test_checkpoint_fields_gradients_and_cuda_parity(self):
        results = []
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for chunk in [0, 4, 7]:
                eps = self.eps.to(device).detach().requires_grad_()
                source = self.sources.to(device).detach().requires_grad_()
                sim = fdtd(eps, 80e-9, [2, 2, 1])
                fields = sim.run({"Jz": source}, checkpoint_steps=chunk)
                loss = sum(f.square().sum() for f in fields.values())
                grads = torch.autograd.grad(loss, (eps, source))
                self.assertEqual(fields["Ez"].device, eps.device)
                results.append([x.detach().cpu() for x in (*fields.values(), *grads)])
        for result in results[1:]:
            for expected, actual in zip(results[0], result):
                torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-12)

    def test_reset_continuation_and_step_equivalence(self):
        sim = fdtd(self.eps, 80e-9, [2, 2, 1])
        expected = sim.run({"Jz": self.sources})
        sim.initialize_fields()
        for source in self.sources:
            actual = sim.forward(Jz=source)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        sim.run({"Jz": self.sources[:5]})
        actual = sim.run({"Jz": self.sources[5:]}, reset=False, checkpoint_steps=3)
        self.assertEqual(sim.t_index, 12)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        sim.eps_r = self.eps * 2
        self.assertEqual(sim.t_index, 0)
        self.assertEqual(sim.Ez.count_nonzero(), 0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            fdtd(-self.eps, 80e-9, [2, 2, 1])
        with self.assertRaises(ValueError):
            fdtd(self.eps.to(torch.complex128), 80e-9, [2, 2, 1])
        sim = fdtd(self.eps, 80e-9, [2, 2, 1])
        with self.assertRaises(ValueError):
            sim.run({"Jz": torch.ones((2, 3))})


if __name__ == "__main__":
    unittest.main()
