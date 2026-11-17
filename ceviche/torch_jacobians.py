"""Dense Jacobian compatibility helper; use torch.autograd.grad for scalar losses.

Rows and columns flatten output and input respectively. Complex tensors use
interleaved real/imaginary coordinates, giving an unambiguous real Jacobian.
"""

import torch

from .tensor import as_tensor


def jacobian(fun, argnum=0, mode="reverse", step_size=1e-6):
    if mode not in ("reverse", "forward", "numerical"):
        raise ValueError("mode must be reverse, forward, or numerical")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    def evaluate(*args, **kwargs):
        x = as_tensor(args[argnum])
        packed = torch.view_as_real(x).reshape(-1) if x.is_complex() else x.reshape(-1)

        def wrapped(p):
            if x.is_complex():
                value = torch.view_as_complex(p.reshape(*x.shape, 2).contiguous())
            else:
                value = p.reshape(x.shape)
            actual = list(args)
            actual[argnum] = value
            result = as_tensor(fun(*actual, **kwargs), device=x.device)
            return (
                torch.view_as_real(result).reshape(-1)
                if result.is_complex()
                else result.reshape(-1)
            )

        if mode == "reverse":
            return torch.autograd.functional.jacobian(wrapped, packed, vectorize=False)
        columns = []
        for i in range(packed.numel()):
            direction = torch.zeros_like(packed)
            direction[i] = 1
            if mode == "forward":
                _, column = torch.func.jvp(wrapped, (packed,), (direction,))
            else:
                column = (
                    wrapped(packed + step_size * direction)
                    - wrapped(packed - step_size * direction)
                ) / (2 * step_size)
            columns.append(column)
        return torch.stack(columns, dim=1)

    return evaluate
