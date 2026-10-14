"""The deterministic adaptive pool must match torch's adaptive pooling exactly."""

import pytest
import torch
import torch.nn as nn

from src.models.pooling import AdaptiveAvgPool, adaptive_avg_pool


@pytest.mark.parametrize("shape,out", [
    ((2, 3, 500), 16),          # non-divisible 1-D (the temporal encoder case)
    ((2, 3, 64), 16),           # divisible 1-D
    ((2, 4, 8, 8, 8), 4),       # divisible 3-D (the spatial encoder case)
    ((2, 4, 9, 7, 11), (4, 3, 2)),  # non-divisible, anisotropic 3-D
    ((2, 5, 6, 6, 6), 1),       # global squeeze (SE block)
    ((1, 2, 5, 5, 5), 5),       # identity
])
def test_matches_torch_adaptive_pool_forward_and_backward(shape, out):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    nd = len(shape) - 2
    ref_mod = {1: nn.AdaptiveAvgPool1d, 3: nn.AdaptiveAvgPool3d}[nd](out)
    ref, got = ref_mod(x), adaptive_avg_pool(x2, out)
    torch.testing.assert_close(got, ref)
    w = torch.randn_like(ref)
    (ref * w).sum().backward(); (got * w).sum().backward()
    torch.testing.assert_close(x2.grad, x.grad)


def test_module_wrapper_and_validation():
    pool = AdaptiveAvgPool((2, 2, 2))
    assert pool(torch.randn(1, 3, 4, 4, 4)).shape == (1, 3, 2, 2, 2) and "output_size" in repr(pool)
    with pytest.raises(ValueError):
        adaptive_avg_pool(torch.randn(1, 3, 4, 4), (2, 2, 2))


def test_backward_is_bitwise_repeatable():
    x = torch.randn(2, 3, 500, requires_grad=True)
    grads = []
    for _ in range(2):
        x.grad = None
        adaptive_avg_pool(x, 16).pow(2).sum().backward()
        grads.append(x.grad.clone())
    assert torch.equal(grads[0], grads[1])
