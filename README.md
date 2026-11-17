# ceviche

Electromagnetic Simulation Tools + Automatic Differentiation.  Code for paper [Forward-Mode Differentiation of Maxwell's Equations](https://arxiv.org/abs/1908.10507).

<img src="/img/horizontal-color.png" title="ceviche" alt="ceviche">

## What is ceviche?

`ceviche` provides two core electromagnetic simulation tools for solving Maxwell's equations:

- finite-difference frequency-domain (FDFD)

- finite-difference time-domain (FDTD)

Both use PyTorch tensors and support forward-mode and reverse-mode automatic differentiation. CPU FDFD uses a SciPy sparse direct solver with an implicit Torch derivative; CUDA FDFD uses Torch GMRES. FDTD runs entirely in Torch on the selected device.

This allows you to write code to solve your E&M problem, and then use automatic differentiation on your results.

As a result, you can do gradient-based optimization, sensitivity analysis, or plug your E&M solver into a machine learning model without having to go through the tedious process of deriving your derivatives by hand.

## Examples

The original NumPy/Autograd tutorial is available at [this link](https://github.com/fancompute/workshop-invdesign) with several historical notebook examples (their objectives need conversion to Torch for this version):
1. [Running FDFD simulations in ceviche.](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/01_First_simulation.ipynb)
2. [Performing inverse design of a mode converter.](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/02_Invdes_intro.ipynb)
3. [Adding fabrication constraints and device parameterizations.](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/03_Invdes_parameterization.ipynb)
4. [Inverse design of a wavelength-division multiplexer and advanced topics.](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/04_Invdes_wdm_scheduling.ipynb)

There are also a few examples in the `examples/*` directory.

## Installation

For local development, use Python 3.12 and [uv](https://docs.astral.sh/uv/) from the repository root:

```sh
uv sync --locked
uv run python -c "import ceviche; print(ceviche.__version__)"
uv build
```

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.
Packages are built with the `uv_build` backend.
The project uses the CERNET PyPI and Python download mirrors configured in `pyproject.toml`.

PyTorch is pinned to the CUDA 12.6 build of 2.14.1 from its official index.
On the tested Windows/Tesla P100 system (driver 539.41), set CUDA lazy loading
before starting Python; eager loading fails during CUDA initialization:

```powershell
$env:CUDA_MODULE_LOADING = "LAZY"
uv run python -c "import torch; print(torch.ones(1, device='cuda'))"
```

Development checks use the locked Ruff and ty dependencies:

```sh
uv run ruff format .
uv run ruff check .
uv run ty check
```

## Simulation and gradients

Initial grids and sources may be NumPy arrays or tensors; fields and Jacobians
are always Torch tensors. Use Torch operations inside differentiable objectives.
NumPy inputs default to double precision; tensors retain their dtype and device.
Set `dtype=torch.float64` explicitly when constructing design tensors.

```python
import math

import torch

from ceviche import fdfd_ez

# Use "cuda" for the P100, after setting CUDA_MODULE_LOADING as shown above.
device = "cpu"
eps = torch.full((16, 12), 2.0, dtype=torch.float64, device=device, requires_grad=True)
source = torch.zeros_like(eps)
source[7, 5] = 1.0
simulation = fdfd_ez(2 * math.pi * 200e12, 80e-9, eps, [3, 3])
hx, hy, ez = simulation.solve(source)
loss = ez.abs().square().mean()
loss.backward()
print(eps.grad.norm().item())
```

`fdfd_hz` returns `(Ex, Ey, Hz)`. `fdfd_mf_ez` uses source and field shapes
`(2*Nsb+1, Nx, Ny)` and modulation arrays `(harmonics, Nx, Ny)`.
PML widths and Bloch phases remain constructor arguments. Grid shape is fixed
for a simulation; assigning `eps_r` preserves the design's computation graph.

`SolverOptions(method="gmres", rtol=1e-10, maxiter=2000, restart=80)` controls
iterative solves. The default selects SciPy on CPU and GMRES on CUDA. GMRES
uses Jacobi preconditioning and checks the true residual; convergence depends
on the problem. Failure raises an error. It never silently moves a CUDA solve
to CPU or expands a large sparse matrix into a dense one. Sparse FDFD solves
support first-order reverse gradients and JVPs, not second-order derivatives
or batched `vmap` solves. Single precision may need looser tolerances.

`jacobian(fun, mode="reverse" | "forward" | "numerical")` returns a matrix
with flattened outputs as rows and inputs as columns. Complex coordinates are
represented as interleaved real/imaginary components, so each complex element
occupies two real coordinates. For scalar losses, prefer `torch.autograd.grad`
or `loss.backward()` to forming a full Jacobian.

`fdtd(eps, dL, [px, py, pz])` supports real positive permittivity and real
current sources. `forward(Jz=source)` advances one step and returns a field
dictionary. `run({"Jz": sources}, checkpoint_steps=32)` accepts a source history
shaped `(steps, Nx, Ny, Nz)` and returns final fields. Checkpointing recomputes
complete state segments during reverse differentiation to reduce stored
activations; use `checkpoint_steps=0` for forward-mode JVPs. `run` resets by
default; `reset=False` continues the current state. `initialize_fields()` or
assigning `eps_r` starts a new trajectory. Do not mutate source tensors before
backward. Multiple backward calls through the same trajectory follow normal
PyTorch graph-lifetime rules.

`adam_optimize` uses native Torch Adam; `jac=None` computes a gradient from
the objective, a callable supplies a gradient, and `jac=True` means the objective
returns `(loss, gradient)`. Bounds project each update; callbacks receive
`(iteration, loss_history, detached_parameters)` before the update.

`modes.get_modes` and `modes.insert_mode` are explicitly nondifferentiable CPU
preprocessing via SciPy, including when the surrounding simulation uses CUDA.
They return tensors on the input device and require detached permittivity.
`viz.real`, `viz.abs`, and `utils.imarr` detach tensors and transfer them to CPU
for plotting. Use `tensor.to_numpy` explicitly at other external boundaries.

## Examples and tests

The scripts and four notebooks under `examples/` use the Torch API. Density
filters live in `ceviche.filters`; no scikit-image or Angler installation is
required. Run notebooks with `examples/` as the working directory so their
included data paths resolve. Full optimization and time-domain examples can
be expensive; reduce their iteration/time-step settings for a quick run.

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run python -m compileall -q ceviche tests examples
uv run python -m unittest discover -s tests -v
uv build
```

Tests cover complex sparse derivatives, real/complex material and source
gradients, PML/Bloch conditions, multifrequency coupling, an analytic discrete
plane wave, FDTD checkpointing, optimizers, filters, modes, and FFTs. CUDA
comparisons run when CUDA is available; set the documented environment variable
before running on the tested Windows/P100 system. Temporary migration probes
and reference data are not part of the repository.

## Credits

If you use this for your research or work, please cite

    @article{hughes2019forward,
      title={Forward-Mode Differentiation of Maxwell’s Equations},
      author={Hughes, Tyler W and Williamson, Ian AD and Minkov, Momchil and Fan, Shanhui},
      journal={ACS Photonics},
      volume={6},
      number={11},
      pages={3010--3016},
      year={2019},
      publisher={ACS Publications}
    }

Our logo was created by [@nagilmer](http://nadinegilmer.com/)
