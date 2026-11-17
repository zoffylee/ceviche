"""Differentiable density filters for inverse design."""

import math

import torch
from torch.nn import functional

from .tensor import as_tensor


def operator_proj(rho, eta=0.5, beta=100):
    rho = as_tensor(rho)
    return (math.tanh(beta * eta) + torch.tanh(beta * (rho - eta))) / (
        math.tanh(beta * eta) + math.tanh(beta * (1 - eta))
    )


def operator_blur(rho, radius=2):
    rho = as_tensor(rho)
    if rho.ndim != 2 or int(radius) != radius or radius < 0:
        raise ValueError("Blur requires a 2D grid and nonnegative integer radius")
    radius = int(radius)
    axis = torch.arange(-radius, radius + 1, device=rho.device)
    kernel = (axis[:, None].square() + axis[None, :].square() < (radius + 1) ** 2).to(rho.dtype)
    kernel = kernel / kernel.sum()
    return functional.conv2d(rho[None, None], kernel[None, None], padding=radius)[0, 0]


def make_rho(rho, design_region, radius=2):
    rho = as_tensor(rho)
    mask = as_tensor(design_region, device=rho.device, dtype=rho.dtype)
    return operator_blur(rho, radius) * mask + rho * (mask == 0)
