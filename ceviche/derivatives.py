"""Fixed-grid Yee derivative operators, assembled with Torch on the target device."""

import torch

from .constants import EPSILON_0, ETA_0


def stretch(omega, dl, size, thickness, forward, *, device, dtype):
    if thickness == 0:
        return torch.ones(size, device=device, dtype=dtype)
    coord = torch.arange(size, device=device, dtype=torch.float64)
    left = dl * (thickness - coord + (0.5 if forward else 1.0))
    right = dl * (coord - (size - thickness) - (0.5 if forward else 1.0))
    distance = torch.where(
        coord <= thickness, left, torch.where(coord > size - thickness, right, 0.0)
    )
    width = thickness * dl
    sigma = 60 / (ETA_0 * width) * (distance / width) ** 3
    return (1 - 1j * sigma / (omega * EPSILON_0)).to(dtype)


def derivative_entries(omega, shape, npml, dl, phases, *, device, dtype):
    nx, ny = shape
    row = torch.arange(nx * ny, device=device)
    grid = row.reshape(shape)
    result = []
    for axis, size in [(0, nx), (1, ny)]:
        for forward in [True, False]:
            factor = stretch(omega, dl, size, npml[axis], forward, device=device, dtype=dtype)
            scaling = (
                (1 / factor[:, None]).expand(shape)
                if axis == 0
                else (1 / factor[None, :]).expand(shape)
            )
            scaling = scaling.flatten()
            if size == 1:
                result.append((scaling, torch.stack((row, row))))
                continue
            step = -1 if forward else 1
            col = torch.roll(grid, shifts=step, dims=axis).flatten()
            coord = row // ny if axis == 0 else row % ny
            edge = coord == (size - 1 if forward else 0)
            angle = torch.as_tensor(phases[axis], device=device, dtype=torch.float64)
            phase = torch.exp((1j if forward else -1j) * angle).to(dtype)
            neighbor = torch.where(edge, phase, 1.0) * (1 if forward else -1)
            diagonal = torch.full_like(neighbor, -1 if forward else 1)
            entries = torch.cat((diagonal * scaling, neighbor * scaling)) / dl
            result.append((entries, torch.stack((torch.cat((row, row)), torch.cat((row, col))))))
    return tuple(result)


def product_pattern(left, right, n):
    """Symbolic product for two fixed, at-most-two-entry-per-row stencils.

    The original entries are diagonal block then neighbor block. Keep every
    path (including zero and duplicate entries) to preserve material gradients.
    """
    lv, li = left
    rv, ri = right
    width = rv.numel() // n
    paths = li[1][:, None] + torch.arange(width, device=lv.device)[None, :] * n
    row = li[0][:, None].expand_as(paths).flatten()
    col = ri[1][paths].flatten()
    middle = li[1][:, None].expand_as(paths).flatten()
    values = (lv[:, None] * rv[paths]).flatten()
    return values, torch.stack((row, col)), middle


def compute_derivative_matrices(
    omega, shape, npml, dL, bloch_x=0.0, bloch_y=0.0, *, device=None, dtype=torch.complex128
):
    """Return four Torch sparse COO derivative matrices in Yee order."""
    n = shape[0] * shape[1]
    return tuple(
        torch.sparse_coo_tensor(i, v, (n, n)).coalesce()
        for v, i in derivative_entries(
            omega, shape, npml, dL, (bloch_x, bloch_y), device=device, dtype=dtype
        )
    )


def curl_E(axis, Ex, Ey, Ez, dL):
    from .fdtd import curl
    from .tensor import as_tensor

    return curl(tuple(map(as_tensor, (Ex, Ey, Ez))), axis, dL, True)


def curl_H(axis, Hx, Hy, Hz, dL):
    from .fdtd import curl
    from .tensor import as_tensor

    return curl(tuple(map(as_tensor, (Hx, Hy, Hz))), axis, dL, False)
