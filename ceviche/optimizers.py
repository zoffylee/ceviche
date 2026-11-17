"""Adam optimization for real Torch design parameters."""

import time

import torch

from .tensor import as_tensor


def adam_optimize(
    objective,
    params,
    jac=None,
    step_size=1e-2,
    Nsteps=100,
    bounds=None,
    direction="min",
    beta1=0.9,
    beta2=0.999,
    callback=None,
    verbose=True,
):
    """Optimize a scalar loss; jac=None uses autograd, jac=True returns (loss, grad).

    Callable jac is also accepted. Callbacks run before updates and receive
    (iteration, loss history, detached parameter copy). Bounds project each
    update. Optimization steps themselves are not differentiable.
    """
    if direction not in ("min", "max"):
        raise ValueError("direction must be min or max")
    if int(Nsteps) != Nsteps or Nsteps < 0:
        raise ValueError("Nsteps must be a nonnegative integer")
    parameter = as_tensor(params).detach().clone().requires_grad_()
    if parameter.is_complex():
        raise ValueError("Design parameters must be real")
    optimizer = torch.optim.Adam(
        [parameter], lr=step_size, betas=(beta1, beta2), maximize=direction == "max"
    )
    limits = None
    if bounds is not None:
        limits = tuple(as_tensor(v, device=parameter.device, dtype=parameter.dtype) for v in bounds)
        if len(limits) != 2 or torch.any(limits[0] > limits[1]):
            raise ValueError("bounds must be (lower, upper) with lower <= upper")
    history = []
    for iteration in range(int(Nsteps)):
        if callback is not None:
            callback(iteration, list(history), parameter.detach().clone())
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if jac is True:
            loss, gradient = objective(parameter)
        else:
            loss = objective(parameter)
            if jac is None:
                (gradient,) = torch.autograd.grad(loss, parameter)
            else:
                gradient = jac(parameter)
        loss = as_tensor(loss)
        gradient = as_tensor(gradient, device=parameter.device, dtype=parameter.dtype).reshape_as(
            parameter
        )
        if (
            loss.numel() != 1
            or loss.is_complex()
            or not torch.isfinite(loss).all()
            or not torch.isfinite(gradient).all()
        ):
            raise ValueError(
                "Objective and gradient must be finite; objective must be a real scalar"
            )
        history.append(loss.detach().item())
        parameter.grad = gradient.detach()
        optimizer.step()
        if limits is not None:
            with torch.no_grad():
                parameter.copy_(torch.clamp(parameter, min=limits[0], max=limits[1]))
        if verbose:
            print(
                f"Epoch: {iteration + 1:3d}/{Nsteps} | Duration: {time.perf_counter() - start:.2f} secs | Value: {history[-1]:.5e}"
            )
    return parameter.detach(), history


def step_adam(gradient, mopt_old, vopt_old, iteration, beta1, beta2, epsilon=1e-8):
    gradient = as_tensor(gradient)
    mopt = beta1 * mopt_old + (1 - beta1) * gradient
    vopt = beta2 * vopt_old + (1 - beta2) * gradient.square()
    update = (mopt / (1 - beta1 ** (iteration + 1))) / (
        (vopt / (1 - beta2 ** (iteration + 1))).sqrt() + epsilon
    )
    return update, mopt, vopt
