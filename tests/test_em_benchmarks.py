"""Physical benchmarks at 1.55 microns. All lengths are in metres."""

import math
import os
import unittest
from typing import cast

import numpy as np
import torch
from scipy.optimize import brentq

from ceviche import fdfd_ez, fdtd
from ceviche.constants import C_0, EPSILON_0, MU_0
from ceviche.modes import get_modes
from ceviche.solvers import SolverOptions

WAVELENGTH = 1.55e-6
K0 = 2 * math.pi / WAVELENGTH
OMEGA = C_0 * K0


def sheet_fields(index, pml_cells):
    """A current sheet radiates toward both ends of a 320-cell uniform medium."""
    dl = WAVELENGTH / 48
    eps = torch.full((320, 1), index**2, dtype=torch.complex128)
    source = torch.zeros_like(eps)
    source[160] = 1.0
    _, hy, ez = fdfd_ez(OMEGA, dl, eps, [pml_cells, 0]).solve(source)
    return dl, hy[:, 0].numpy(), ez[:, 0].numpy()


def periodic_field_error(device, dtype, method):
    """Relative L2 error against three exact Fourier modes of the discrete Ez operator."""
    nx, ny = 64, 48
    dl = WAVELENGTH / 16
    eps = 2.25 - 0.3j
    x, y = np.ogrid[:nx, :ny]
    expected = np.zeros((nx, ny), dtype=np.complex128)
    rhs = np.zeros_like(expected)
    for mx, my, amplitude in [(1, 2, 1), (-3, 1, 0.4j), (4, -2, -0.3)]:
        px, py = 2 * math.pi * mx / nx, 2 * math.pi * my / ny
        wave = amplitude * np.exp(1j * (px * x + py * y))
        k2 = 4 * (math.sin(px / 2) ** 2 + math.sin(py / 2) ** 2) / dl**2
        expected += wave
        rhs += (k2 / MU_0 - EPSILON_0 * OMEGA**2 * eps) * wave
    material = torch.full((nx, ny), eps, dtype=dtype, device=device)
    source = torch.as_tensor(rhs / (1j * OMEGA), dtype=dtype, device=device)
    options = SolverOptions(method=method, rtol=3e-5 if dtype == torch.complex64 else 1e-10)
    ez = fdfd_ez(OMEGA, dl, material, [0, 0], solver_options=options).solve(source)[2]
    assert ez.dtype == dtype and ez.device.type == device
    return np.linalg.norm(ez.cpu().numpy() - expected) / np.linalg.norm(expected)


class TestCPUPhysics(unittest.TestCase):
    def test_fdtd_plane_wave_time_of_flight(self):
        index = 1.5
        dl = WAVELENGTH / 40
        sim = fdtd(torch.full((384, 4, 1), index**2, dtype=torch.float64), dl, [0, 0, 0])
        sheet = torch.zeros_like(sim.eps_r)
        sheet[40] = 1
        probes = (70, 130)
        traces = []
        # End before the pulse traveling left can wrap around to either probe.
        for step in range(820):
            pulse = math.exp(-0.5 * ((step - 140) / 36) ** 2) * math.cos(
                OMEGA * sim.dt * (step - 140)
            )
            wave = sim.forward(Jz=pulse * sheet)["Ez"]
            traces.append([wave[p, 0, 0].item() for p in probes])
        first, second = np.array(traces).T
        delay = np.argmax(np.correlate(second, first, "full")) - (len(first) - 1)
        expected_steps = (probes[1] - probes[0]) * dl * index / (C_0 * sim.dt)
        self.assertGreater(np.max(np.abs(first)), 0.1, "FDTD incident pulse missing")
        self.assertAlmostEqual(delay / expected_steps, 1.0, delta=0.02, msg="FDTD transit time")
        self.assertAlmostEqual(
            np.linalg.norm(second) / np.linalg.norm(first),
            1.0,
            delta=0.01,
            msg="FDTD pulse amplitude in lossless dielectric",
        )

    def test_fdfd_pml_reflection_and_attenuation(self):
        dl, _, ez = sheet_fields(1.0, 32)
        phase = 2 * math.asin(K0 * dl / 2)
        # Project the source-free interior onto the two discrete plane waves.
        for direction in (1, -1):
            with self.subTest(direction=direction):
                distances = np.arange(8, 104)
                incident_wave = np.exp(-1j * phase * distances)
                basis = np.column_stack((incident_wave, incident_wave.conj()))
                incident, reflected = np.linalg.lstsq(
                    basis, ez[160 + direction * distances], rcond=None
                )[0]
                reflectance = abs(reflected / incident) ** 2
                outer = ez[-3:] if direction == 1 else ez[:3]
                self.assertLess(reflectance, 1e-6, "FDFD PML reflected power fraction")
                self.assertLess(
                    np.max(abs(outer / incident)) ** 2,
                    1e-6,
                    "FDFD field intensity at outer PML boundary",
                )

    def test_fdtd_pml_pulse_absorption(self):
        # A localized displacement pulse splits into two packets. Both reach
        # the boundaries within 400 steps; a periodic control retains the energy.
        cells = 192
        dl = WAVELENGTH / 24
        x = torch.arange(cells, dtype=torch.float64) - cells // 2
        pulse = (torch.cos(K0 * dl * x) * torch.exp(-0.5 * (x / 13) ** 2))[:, None, None]
        energies = []
        for width in (0, 24):
            sim = fdtd(torch.ones_like(pulse), dl, [width, 0, 0])
            sim.forward(Jz=pulse)
            for _ in range(550):
                sim.forward()
            # Ceviche FDTD normalizes H by the vacuum impedance.
            energies.append((sim.Ez.square() + sim.Hy.square()).sum().item())
        self.assertAlmostEqual(
            energies[0] / pulse.square().sum().item(),
            1.0,
            delta=0.05,
            msg="FDTD periodic control energy",
        )
        self.assertLess(
            energies[1] / energies[0], 1e-6, "FDTD residual energy after PML absorption"
        )

    def test_fdfd_power_balance_and_material_loss(self):
        for extinction in (0.0, 0.04):
            with self.subTest(extinction=extinction):
                index = 1.5 - 1j * extinction
                dl, hy, ez = sheet_fields(index, 32)
                # Hy[j] straddles Ez[j-1] and Ez[j]. Flux is per unit transverse area.
                flux = -0.25 * np.real((ez + np.roll(ez, 1)) * hy.conj())
                a, b = 185, 265
                absorbed = (
                    -0.5 * OMEGA * EPSILON_0 * (index**2).imag * dl * np.sum(abs(ez[a:b]) ** 2)
                )
                self.assertGreater(flux[a], 0, "incident Poynting flux")
                self.assertAlmostEqual(
                    (flux[b] + absorbed) / flux[a],
                    1.0,
                    delta=1e-8,
                    msg="FDFD Poynting flux and integrated material loss",
                )
                transmission = math.exp(-2 * K0 * extinction * (b - a) * dl)
                self.assertAlmostEqual(
                    flux[b] / flux[a],
                    transmission,
                    delta=0.005,
                    msg="FDFD Beer-Lambert power transmission",
                )

    def test_straight_slab_propagation_convergence(self):
        half_width = 0.24e-6
        v = K0 * half_width * math.sqrt(2.0**2 - 1.0**2)
        # Fundamental even TE root: u tan(u) = sqrt(V^2 - u^2).
        u = cast(
            float,
            brentq(
                lambda u: u * math.tan(u) - math.sqrt(v**2 - u**2),
                0,
                min(v, math.pi / 2),
                full_output=False,
            ),
        )
        beta = math.sqrt((2.0 * K0) ** 2 - (u / half_width) ** 2)
        errors = []
        for dl in (80e-9, 40e-9, 20e-9):
            nx, ny = round(6.4e-6 / dl), round(3.2e-6 / dl)
            half_core = round(half_width / dl)
            cross = torch.ones(ny, dtype=torch.float64)
            cross[ny // 2 - half_core : ny // 2 + half_core] = 2.0**2
            n_eff, modes = get_modes(cross, OMEGA, dl, 0, return_neff=True)
            # Separate transverse mode-solver error from longitudinal FDFD dispersion.
            self.assertLess(
                abs(n_eff[0].item() * K0 / beta - 1),
                0.02 * (dl / 80e-9) ** 2,
                f"transverse slab mode index at {dl:g} m",
            )
            profile = modes[:, 0]
            sim = fdfd_ez(OMEGA, dl, cross.expand(nx, ny), [round(1.28e-6 / dl), 0])
            source = torch.zeros((nx, ny), dtype=torch.complex128)
            source[nx // 3] = profile
            ez = sim.solve(source)[2]
            projected = ez[nx // 2 : 3 * nx // 4] @ profile.conj()
            beta_measured = -torch.angle(torch.vdot(projected[:-1], projected[1:])).item() / dl
            error = abs(beta_measured / beta - 1)
            if errors:
                self.assertLess(error, 0.35 * errors[-1], f"slab beta convergence at {dl:g} m")
            errors.append(error)
        self.assertLess(errors[-1], 0.003, "finest-grid FDFD slab propagation constant")

    def test_fdfd_precision_and_cpu_solvers(self):
        for dtype, method, limit in [
            (torch.complex64, "scipy", 2e-5),
            (torch.complex64, "gmres", 4e-5),
            (torch.complex128, "scipy", 1e-10),
            (torch.complex128, "pardiso", 1e-10),
        ]:
            with self.subTest(dtype=dtype, method=method):
                self.assertLess(
                    periodic_field_error("cpu", dtype, method), limit, "relative Ez L2 error"
                )


@unittest.skipUnless(
    os.environ.get("CEVICHE_TEST_CUDA") == "1" and torch.cuda.is_available(),
    "optional CUDA benchmarks require CEVICHE_TEST_CUDA=1 and CUDA",
)
class TestCUDAPhysics(unittest.TestCase):
    def test_fdfd_gmres_precision(self):
        for dtype, limit in [(torch.complex64, 4e-5), (torch.complex128, 1e-9)]:
            with self.subTest(dtype=dtype):
                self.assertLess(
                    periodic_field_error("cuda", dtype, "gmres"), limit, "relative Ez L2 error"
                )
