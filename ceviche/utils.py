"""Tensor utilities; NumPy conversion is confined to plotting boundaries."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from .tensor import as_tensor, to_numpy


def make_sparse(entries, indices, shape):
    values = as_tensor(entries)
    indices = torch.as_tensor(indices, device=values.device, dtype=torch.long)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def get_entries_indices(matrix):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.to_sparse_coo().coalesce()
        return matrix.values(), matrix.indices()
    coo = matrix.tocoo()
    return as_tensor(coo.data), torch.tensor(np.stack((coo.row, coo.col)), dtype=torch.long)


def transpose_indices(indices):
    return torch.as_tensor(indices).flip(0)


def block_4(A, B, C, D):
    return torch.cat((torch.cat((A, B), dim=1), torch.cat((C, D), dim=1)), dim=0)


def make_IO_matrices(indices, N):
    indices = torch.as_tensor(indices, dtype=torch.long)
    m = indices.shape[1]
    seq = torch.arange(m, device=indices.device)
    ones = torch.ones(m, device=indices.device, dtype=torch.float64)
    return (
        make_sparse(ones, torch.stack((indices[0], seq)), (N, m)),
        make_sparse(ones, torch.stack((seq, indices[1])), (m, N)),
    )


def make_rand(N):
    return torch.rand(N, dtype=torch.float64) - 0.5


def make_rand_complex(N):
    return make_rand(N) + 1j * make_rand(N)


def make_rand_indeces(N, M):
    return torch.randint(N, (2, M))


def make_rand_entries_indices(N, M):
    return make_rand_complex(M), make_rand_indeces(N, M)


def make_rand_sparse(N, M):
    return make_sparse(*make_rand_entries_indices(N, M), (N, N))


def make_rand_sparse_density(N, density=1):
    return make_rand_sparse(N, int(N * N * density))


def der_num(fn, arg, index, delta):
    arg = as_tensor(arg)
    direction = torch.zeros_like(arg)
    direction.flatten()[index] = delta / 2
    return (fn(arg + direction) - fn(arg - direction)) / delta


def grad_num(fn, arg, delta=1e-6):
    from .jacobians import jacobian

    arg = as_tensor(arg)
    result = jacobian(fn, mode="numerical", step_size=delta)(arg)
    if result.shape[0] != 1:
        raise ValueError("grad_num requires a real scalar objective")
    if arg.is_complex():
        return torch.view_as_complex(result.reshape(-1, 2).contiguous()).reshape(arg.shape)
    return result.reshape(arg.shape)


def jac_num(fn, arg, step_size=1e-7):
    from .jacobians import jacobian

    return jacobian(fn, mode="numerical", step_size=step_size)(arg)


def grid_center_to_xyz(Q_mid, averaging=True):
    q = as_tensor(Q_mid)
    return tuple((q + torch.roll(q, 1, axis)) / 2 if averaging else q.clone() for axis in range(3))


def grid_xyz_to_center(Q_xx, Q_yy, Q_zz):
    return tuple(
        (q + torch.roll(q, 1, axis)) / 2
        for axis, q in enumerate(map(as_tensor, (Q_xx, Q_yy, Q_zz)))
    )


def vec_zz_to_xy(info_dict, vec_zz, grid_averaging=True):
    xx, yy, _ = grid_center_to_xyz(
        as_tensor(vec_zz).reshape(*info_dict["shape"], 1), grid_averaging
    )
    return xx.flatten(), yy.flatten()


def float_2_array(x):
    return as_tensor(x).reshape(-1) if np.isscalar(x) else as_tensor(x)


def reshape_to_ND(arr, N):
    arr = as_tensor(arr)
    if arr.ndim > N:
        raise ValueError(f"Input has more than {N} dimensions")
    return arr.reshape(*arr.shape, *((1,) * (N - arr.ndim)))


def get_value(x):
    """Detach and transfer to CPU for plotting or external, nondifferentiable code."""
    return to_numpy(x)


get_value_arr = get_value


def get_shape(x):
    return tuple(as_tensor(x).shape) or (1,)


def vjp_maker_num(fn, arg_inds, steps):
    from .jacobians import jacobian

    def maker(index, step):
        def setup(_out, *args):
            matrix = jacobian(fn, argnum=index, mode="numerical", step_size=step)(*args)

            def vjp(v):
                v = as_tensor(v, device=matrix.device)
                packed = torch.view_as_real(v).flatten() if v.is_complex() else v.flatten()
                result = matrix.T @ packed
                arg = as_tensor(args[index])
                if arg.is_complex():
                    result = torch.view_as_complex(result.reshape(-1, 2).contiguous())
                return result.reshape(arg.shape)

            return vjp

        return setup

    return tuple(maker(i, s) for i, s in zip(arg_inds, steps))


def aniplot(F, source, steps, component="Ez", num_panels=10):
    F.initialize_fields()
    _, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4), squeeze=False)
    every = max(1, steps // num_panels)
    with torch.no_grad():
        for t in range(steps):
            fields = F.forward(Jz=source(t))
            if t % every == 0 and t // every < num_panels:
                ax = axes[0, t // every]
                ax.imshow(imarr(fields[component]), cmap="RdBu")
                ax.set_title(f"time = {F.dt * t:.3e} s")
    plt.show()


def measure_fields(F, source, steps, probes, component="Ez"):
    F.initialize_fields()
    probes = probes if isinstance(probes, list) else [probes]
    probes = [as_tensor(p, device=F.device, dtype=F.dtype) for p in probes]
    measured = []
    for t in range(steps):
        field = F.forward(Jz=source(t))[component]
        measured.append(torch.stack([(field * p).sum() for p in probes]))
    return (
        torch.stack(measured)
        if measured
        else torch.empty((0, len(probes)), device=F.device, dtype=F.dtype)
    )


def imarr(arr):
    value = to_numpy(arr)
    if value.ndim == 3:
        value = value[:, :, 0]
    return np.flipud(value.T)


def my_fft(x):
    return torch.fft.fft(as_tensor(x))


def get_spectrum(series, dt):
    series = as_tensor(series)
    steps = series.shape[0]
    window = torch.hamming_window(
        steps, periodic=False, device=series.device, dtype=series.real.dtype
    )
    signal = torch.fft.fft(series.reshape(steps, -1) * window[:, None], dim=0)
    freqs = torch.fft.fftfreq(steps, d=dt, device=series.device, dtype=series.real.dtype)
    return freqs, signal


def get_max_power_freq(series, dt):
    freqs, power = get_spectral_power(series, dt)
    return freqs[power.sum(dim=1).argmax()]


def get_spectral_power(series, dt):
    freqs, signal = get_spectrum(series, dt)
    return freqs, signal.abs().square()


def plot_spectral_power(series, dt, f_top=2e14):
    freqs, power = get_spectral_power(series, dt)
    n = len(freqs) // 2
    plt.plot(to_numpy(freqs[:n]), to_numpy(power[:n]))
    plt.xlim(0, f_top)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("power (|signal|^2)")
    plt.show()
