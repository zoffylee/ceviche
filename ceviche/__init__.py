"""Differentiable electromagnetic solvers powered by PyTorch."""

from . import modes, utils, viz
from .fdfd import fdfd_ez, fdfd_hz, fdfd_mf_ez
from .fdtd import fdtd
from .jacobians import jacobian
from .solvers import SolverOptions

__version__ = "0.1.3"
__all__ = [
    "SolverOptions",
    "fdfd_ez",
    "fdfd_hz",
    "fdfd_mf_ez",
    "fdtd",
    "jacobian",
    "modes",
    "utils",
    "viz",
]
