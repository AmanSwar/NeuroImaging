"""1D-CNN latency backbone: VEP waveform -> temporal tokens.

v2: multi-scale temporal modeling. VEP pathology expresses itself at
different time scales simultaneously -- the P100 peak is a ~25-35 ms
feature while the N75-P100-N145 complex spans ~80 ms at clinical sampling
rates. A single fixed kernel size cannot cover both. Parallel kernels of
3/9/19 samples (per stage) are fused after each downsampling stage so the
encoder sees fine waveform detail *and* complex-level context.
"""

import torch
import torch.nn as nn

from src.config import ModelConfig, DEFAULT_CONFIG
from src.models.pooling import AdaptiveAvgPool


class MultiScaleResidual1D(nn.Module):
    """Parallel multi-kernel temporal convolutions fused with a residual."""

    def __init__(self, channels: int, kernels: tuple[int, ...] = (3, 9, 19)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, bias=False)
            for k in kernels
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(kernels), channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.act(x + self.fuse(branches))


class TemporalEncoder1D(nn.Module):
    """(B, 1, L) -> (B, temporal_tokens, d_model) temporal token sequence."""

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model, in_channels: int = 1):
        super().__init__()
        width_stages = [16, 32, 64, cfg.d_model]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width_stages[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(width_stages[0]),
            nn.GELU(),
        )
        stages = []
        prev = width_stages[0]
        for width in width_stages[1:]:
            stages.append(
                nn.Sequential(
                    nn.Conv1d(prev, width, kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                )
            )
            stages.append(MultiScaleResidual1D(width))
            prev = width
        self.net = nn.Sequential(*stages)
        self.pool = AdaptiveAvgPool(cfg.temporal_tokens)
        self.d_model = cfg.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)         # (B, 16, L)
        x = self.net(x)          # (B, d_model, L')
        x = self.pool(x)         # (B, d_model, temporal_tokens)
        tokens = x.transpose(1, 2)  # (B, temporal_tokens, d_model)
        return tokens
