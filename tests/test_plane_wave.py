import cmath
import math
import unittest

import torch

from ceviche import fdfd_ez, fdfd_hz
from ceviche.constants import EPSILON_0, MU_0
from ceviche.derivatives import compute_derivative_matrices


class TestPlaneWave(unittest.TestCase):
    def test_single_cell_bloch_stencils(self):
        dl = 80e-9
        for phases in ((0.0, 0.0), (0.3, -0.7)):
            matrices = compute_derivative_matrices(1.0, (1, 1), (0, 0), dl, *phases)
            for phase, forward, backward in zip(phases, matrices[::2], matrices[1::2]):
                df = (cmath.exp(1j * phase) - 1) / dl
                db = (1 - cmath.exp(-1j * phase)) / dl
                for matrix, expected in ((forward, df), (backward, db)):
                    actual = matrix.to_dense()
                    torch.testing.assert_close(
                        actual, torch.full_like(actual, expected), rtol=1e-14, atol=0
                    )

    def test_discrete_periodic_plane_wave(self):
        nx, ny = 16, 12
        dl, omega, eps = 80e-9, 2 * math.pi * 200e12, 2.0
        px, py = 2 * math.pi / nx, 2 * math.pi / ny
        x = torch.arange(nx, dtype=torch.float64)[:, None]
        y = torch.arange(ny, dtype=torch.float64)[None, :]
        wave = torch.exp(1j * (px * x + py * y))
        k2 = 4 * (math.sin(px / 2) ** 2 + math.sin(py / 2) ** 2) / dl**2
        for cls in [fdfd_ez, fdfd_hz]:
            sim = cls(omega, dl, torch.full((nx, ny), eps, dtype=torch.float64), [0, 0])
            eigenvalue = (
                k2 / MU_0 - EPSILON_0 * omega**2 * eps
                if cls is fdfd_ez
                else MU_0 * omega**2 - k2 / (EPSILON_0 * (eps + 1e-5))
            )
            actual = sim.solve(wave * eigenvalue / (1j * omega))[2]
            torch.testing.assert_close(actual, wave, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
