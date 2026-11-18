import math
import unittest
from typing import cast

import torch
from scipy.optimize import brentq

from ceviche.constants import C_0
from ceviche.modes import get_modes


class TestEffectiveIndex(unittest.TestCase):
    wavelength = 1.55e-6
    omega = 2 * math.pi * C_0 / wavelength

    def test_uniform_default_permittivity_and_explicit_index(self):
        for index in (1.5, 1.5 + 0.1j, 1.5 - 0.1j):
            with self.subTest(index=index):
                eps = torch.full((24,), index**2, dtype=torch.complex128)
                eigenvalues, profiles = get_modes(eps, self.omega, 50e-9, 0)
                n_eff, _ = get_modes(eps, self.omega, 50e-9, 0, return_neff=True)
                torch.testing.assert_close(n_eff, torch.full_like(n_eff, index))
                torch.testing.assert_close(eigenvalues, eps[:1])
                torch.testing.assert_close(profiles.abs(), torch.full_like(profiles.real, 24**-0.5))

    def test_evanescent_branch_and_filtering(self):
        # Both nonconstant modes of this three-cell periodic grid have
        # lambda = eps_r - 3 / (k0 * dL)**2 = -2.
        eps = torch.ones(3, dtype=torch.float64)
        dl = self.wavelength / (2 * math.pi)
        indices, _ = get_modes(eps, self.omega, dl, 0, m=3, filtering=False, return_neff=True)
        expected = torch.tensor([1, 1j * math.sqrt(2), 1j * math.sqrt(2)], dtype=indices.dtype)
        torch.testing.assert_close(indices, expected)
        eps_eff, _ = get_modes(eps, self.omega, dl, 0, m=3, filtering=False)
        torch.testing.assert_close(eps_eff, torch.tensor([1, -2, -2], dtype=eps_eff.dtype))
        filtered, _ = get_modes(eps, self.omega, dl, 0, m=3, return_neff=True)
        torch.testing.assert_close(filtered, expected[:1])

    def test_step_index_slab_and_eim(self):
        n_core, n_clad = 2.0, 1.45
        dl = 10e-9
        core_cells = 40
        eps = torch.full((540,), n_clad**2, dtype=torch.float64)
        eps[250 : 250 + core_cells] = n_core**2

        # Even TE slab dispersion: u tan(u) = sqrt(V**2 - u**2), u = kx * half_width.
        k0a = math.pi * core_cells * dl / self.wavelength
        v = k0a * math.sqrt(n_core**2 - n_clad**2)
        u = cast(
            float,
            brentq(lambda u: u * math.tan(u) - math.sqrt(v**2 - u**2), 0, v, full_output=False),
        )
        analytic_index = math.sqrt(n_core**2 - (u / k0a) ** 2)
        n_eff, _ = get_modes(eps, self.omega, dl, 0, return_neff=True)
        torch.testing.assert_close(n_eff, torch.full_like(n_eff, analytic_index), atol=2e-4, rtol=0)

        # A laterally uniform EIM cross section must reproduce the slab index.
        eps_eff, _ = get_modes(eps, self.omega, dl, 0)
        lateral_index, _ = get_modes(eps_eff.expand(24), self.omega, dl, 0, return_neff=True)
        torch.testing.assert_close(lateral_index, n_eff)
