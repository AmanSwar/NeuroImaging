"""3D-CNN / ResNet backbone: structural MRI volume -> spatial tokens.

v2: residual blocks gain Squeeze-and-Excitation channel recalibration
(Hu et al., 2018). Brain MRI has strong inter-slice intensity variation;
channel attention lets the encoder suppress uninformative feature channels
before tokens are pooled, at negligible parameter cost.
"""

import torch
import torch.nn as nn

from src.config import ModelConfig, DEFAULT_CONFIG
from src.models.pooling import AdaptiveAvgPool


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation over the channel dimension of a 3D feature map."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.squeeze = AdaptiveAvgPool(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        weights = self.excite(self.squeeze(x).flatten(1)).view(b, c, 1, 1, 1)
        return x * weights


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 se_reduction: int | None = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock3D(out_channels, se_reduction) if se_reduction else nn.Identity()

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + identity)


class SpatialEncoder3D(nn.Module):
    """(B, 1, D, H, W) -> (B, spatial_tokens, d_model) spatial token sequence.

    `self.last_conv` exposes the final residual block so Grad-CAM3D can hook
    its output activation map.
    """

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        se = cfg.se_reduction
        self.layer1 = ResidualBlock3D(16, 32, stride=2, se_reduction=se)
        self.layer2 = ResidualBlock3D(32, 64, stride=2, se_reduction=se)
        self.layer3 = ResidualBlock3D(64, cfg.d_model, stride=2, se_reduction=se)
        self.last_conv = self.layer3

        self.pool = AdaptiveAvgPool(cfg.spatial_pool)
        self.d_model = cfg.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)  # (B, d_model, *spatial_pool)
        tokens = x.flatten(2).transpose(1, 2)  # (B, tokens, d_model)
        return tokens
