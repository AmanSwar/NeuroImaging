"""Deterministic adaptive average pooling.

``nn.AdaptiveAvgPool{1,3}d`` has no deterministic backward on CUDA (PyTorch
warns under ``use_deterministic_algorithms``) and is missing on Apple MPS.
Adaptive average pooling is separable and linear, so it is exactly a matrix
product per axis with the window weights below. Matmuls are deterministic on
every backend, so this keeps GPU runs reproducible and removes the MPS->CPU
fallback, with outputs equal to the PyTorch op (up to float rounding).
"""

import math

import torch
import torch.nn as nn


def _pool_matrix(n_in: int, n_out: int, device, dtype) -> torch.Tensor:
    """(n_in, n_out) matrix whose column i averages PyTorch's window
    ``[floor(i*n_in/n_out), ceil((i+1)*n_in/n_out))``."""
    m = torch.zeros(n_in, n_out, dtype=torch.float32)
    for i in range(n_out):
        lo = (i * n_in) // n_out
        hi = math.ceil((i + 1) * n_in / n_out)
        m[lo:hi, i] = 1.0 / (hi - lo)
    return m.to(device=device, dtype=dtype)


def adaptive_avg_pool(x: torch.Tensor, output_size) -> torch.Tensor:
    """Adaptive average pool over all dims after (batch, channel)."""
    nd = x.dim() - 2
    out = (output_size,) * nd if isinstance(output_size, int) else tuple(output_size)
    if len(out) != nd:
        raise ValueError(f"output_size {output_size!r} does not match a {nd}-D feature map")
    for d, n_out in enumerate(out):
        axis = 2 + d
        n_in = x.shape[axis]
        if n_in == n_out:
            continue
        p = _pool_matrix(n_in, n_out, x.device, x.dtype)
        x = torch.movedim(torch.tensordot(x, p, dims=([axis], [0])), -1, axis)
    return x


class AdaptiveAvgPool(nn.Module):
    """Drop-in replacement for ``nn.AdaptiveAvgPool{1,2,3}d`` (no parameters)."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return adaptive_avg_pool(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"
