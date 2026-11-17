import sys

import matplotlib.pylab as plt
import numpy as np
import torch

from ceviche.optimizers import adam_optimize
from ceviche.tensor import as_tensor

sys.path.append("../ceviche")

from ceviche import fdfd_hz, jacobian
from ceviche.utils import imarr

""" Optimize intensity focusing through a box with continuous varying permittivity """

# whether to plot setup stuff
PLOT = False

# make parameters
omega = 2 * np.pi * 200e12  # lambda = 1.5 um
dL = 4e-8
eps_max = 2
npml = 10
spc = 10
L = 3e-6
Nx, Ny = 2 * npml + 4 * spc + int(L / dL), 2 * npml + 4 * spc + int(L / dL)
eps_r = np.ones((Nx, Ny))

# make source
source = np.zeros((Nx, Ny))
source[npml + spc, Ny // 2] = 1

# make design region
box_region = np.zeros((Nx, Ny))
box_region[
    npml + 2 * spc : npml + 2 * spc + int(L / dL), npml + 2 * spc : npml + 2 * spc + int(L / dL)
] = 1

# make the accelration probe
probe = np.zeros((Nx, Ny), dtype=np.complex128)
probe[-npml - spc, Ny // 2] = 1

# plot the probe through channel
if PLOT:
    plt.imshow(np.abs(imarr(probe + box_region + source)))
    plt.show()

eps_r = as_tensor(eps_r)
source = as_tensor(source)
probe = as_tensor(probe)
box_region = as_tensor(box_region)

# vacuum test, get normalization
F = fdfd_hz(omega, dL, eps_r, [npml, npml])
Ex, Ey, Hz = F.solve(source)
E_mag = (Ex.abs().square() + Ey.abs().square()).sqrt()
H_mag = Hz.abs()
I_E0 = (E_mag * probe).sum().square().abs()
I_H0 = (H_mag * probe).sum().square().abs().detach()

print(f"I_H0 = {I_H0}")

# plot the vacuum fields
if PLOT:
    plt.imshow(np.real(imarr(Hz)), cmap="RdBu")
    plt.title("real(Hz)")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.colorbar()
    plt.show()


# defines the intensity on the other side of the box as a function of the relative permittivity grid
def intensity(eps_arr):

    eps_r = eps_arr.reshape((Nx, Ny))
    # set the permittivity of the FDFD and solve the fields
    F.eps_r = eps_r
    _Ex, _Ey, Hz = F.solve(source)

    # compute the gradient and normalize if you want
    I = torch.sum(torch.square(torch.abs(Hz * probe)))
    return -I / I_H0


# Define the gradient with PyTorch
grad_I = jacobian(intensity, mode="reverse")

# initialize the design region with some eps
eps_r[box_region == 1] = eps_max


lower = torch.ones_like(eps_r)
upper = lower + (eps_max - 1) * box_region
optimum, history = adam_optimize(
    intensity, eps_r.flatten(), Nsteps=100, bounds=(lower.flatten(), upper.flatten())
)
F.eps_r = optimum.reshape(Nx, Ny)

# plot the final permittivity
plt.imshow(imarr(F.eps_r), cmap="nipy_spectral")
plt.colorbar()
plt.show()

# plot the fields
Ex, Ey, Hz = F.solve(source)
plt.imshow(np.real(imarr(Hz)), cmap="RdBu")
plt.title("real(H_z)")
plt.xlabel("y")
plt.ylabel("x")
plt.colorbar()
plt.show()
