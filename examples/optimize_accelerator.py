# %% Optimize energy gain of dielectric particle accelerator
import matplotlib.pylab as plt
import numpy as np
import torch

from ceviche import fdfd_hz
from ceviche.constants import C_0
from ceviche.optimizers import adam_optimize
from ceviche.tensor import as_tensor
from ceviche.utils import get_value, imarr

# %% Simulation parameters

# whether to plot the setup images
PLOT = False
NIter = 200

# make parameters
wavelength = 1973e-9  # free space wavelength
omega = 2 * np.pi * C_0 / wavelength  # angular frequency
beta = 0.53  # speed of electron / speed of light
dL = wavelength / 100.0  # grid size (m)

spc = int(2 * wavelength / dL)  # space between edge of domain and the structure (m)
gap = int(300e-9 / dL)  # gap size (m)
mat_L = int(2e-6 / dL)

eps_max = 4
# eps_max = 3.6730**2                       # relative permittivity of material

# %% Build the accelerator unit cell
Nx, Ny = 2 * spc + gap + 2 * mat_L, int(beta * wavelength / dL)

# creates the source and initializes permittivity
eps_r = np.ones((Nx, Ny))
source = np.zeros((Nx, Ny))  # this is 'Mz' (magnetic current density)
source[30, :] = 1
# source[-30-1, :] = -1               # dual side drive, uncomment for single side

# make design region
design_region = np.zeros((Nx, Ny))
design_region[spc : Nx // 2 - gap // 2, :] = 1
design_region[Nx // 2 + gap // 2 : Nx - spc, :] = 1
eps_r[design_region == 1] = eps_max  # start with material in the design region

# pml thicknesses
npml = [20, 0]

# make the accelration probe
eta = np.zeros((Nx, Ny), dtype=np.complex128)
channel_ys = np.arange(Ny)
eta[Nx // 2, :] = np.exp(1j * 2 * np.pi * channel_ys / Ny)

# plot the probe through channel
if PLOT:
    plt.plot(np.real(imarr(eta[Nx // 2, :])), label=r"RE\{eta\}")
    plt.xlabel("position along channel (y)")
    plt.ylabel("eta (y)")
    plt.show()

eps_r = as_tensor(eps_r)
source = as_tensor(source)
eta = as_tensor(eta)
design_region = as_tensor(design_region)

# %% Reference field and normalization
F = fdfd_hz(omega, dL, eps_r, npml)
Ex, Ey, Hz = F.solve(source)
E_mag = (Ex.abs().square() + Ey.abs().square()).sqrt()
E0 = E_mag[spc:-spc, :].max().detach().item()
print(f"E0 = {E0} V/m")

# plot the reference fields
if PLOT:
    plt.imshow(Ey.real / E0, cmap="RdBu")
    plt.title("E_y / E0 (<-)")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.colorbar()
    plt.show()


# %% Acceleration gradient objective
def Emax(Ex, Ey, eps_r):
    E_mag = torch.sqrt(torch.square(torch.abs(Ex)) + torch.square(torch.abs(Ey)))
    material_density = (eps_r - 1) / (eps_max - 1)
    return torch.max(E_mag * material_density)


# average electric field magnitude in the domain
def Eavg(Ex, Ey):
    E_mag = torch.sqrt(torch.square(torch.abs(Ex)) + torch.square(torch.abs(Ey)))
    return torch.mean(E_mag)


# defines the acceleration gradient as a function of the relative permittivity grid
def accel_gradient(eps_arr, mode="max"):

    # set the permittivity of the FDFD and solve the fields
    F.eps_r = eps_arr.reshape((Nx, Ny))
    Ex, Ey, _Hz = F.solve(source)

    # compute the gradient and normalize if you want
    G = torch.sum(Ey * as_tensor(eta) / Ny)

    if mode == "max":
        return -G.abs() / Emax(Ex, Ey, F.eps_r)
    elif mode == "avg":
        return -G.abs() / Eavg(Ex, Ey)
    else:
        return -(G / E0).abs()


# %% Optimize within the permittivity bounds
lower = torch.ones_like(eps_r)
upper = lower + (eps_max - 1) * design_region
optimum, history = adam_optimize(
    accel_gradient, eps_r.flatten(), Nsteps=NIter, bounds=(lower.flatten(), upper.flatten())
)
F.eps_r = optimum.reshape(Nx, Ny)


# %% Optimized permittivity and fields
def stack(arr, num_periods=1):
    # returns an array that is stacked `num_periods` times along y.  For plotting many periods of the accelerator unit cell.
    arr_orig = get_value(arr).copy()
    arr_big = get_value(arr).copy()
    for i in range(num_periods):
        arr_big = np.hstack([arr_big, arr_orig])
    return arr_big


num_periods = 4

# plot the final permittivity
eps_disp = imarr(stack(get_value(F.eps_r), num_periods=num_periods))

plt.imshow(np.real(eps_disp), cmap="nipy_spectral")
plt.colorbar()
plt.show()

# plot the accelerating fields
fields = F.solve(source)

[Ex, Ey, Hz] = [get_value(f) for f in fields]

plt.imshow(imarr(np.real(stack(Ey, num_periods=num_periods))) / E0, cmap="RdBu")
plt.title("E_y / E0 (<-)")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()
