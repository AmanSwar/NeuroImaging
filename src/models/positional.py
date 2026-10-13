"""Positional encodings for the CAFN token streams.

Cross-attention correlates spatial tokens with temporal tokens, but a
token that carries no positional information cannot represent "occipital
region" or "the 90-110 ms latency window" -- it is just a feature average.
Without positions, the cross-attention heatmap is a *bag-of-features*
correlation, not a region-to-latency map, which undermines the clinical
interpretability claim of the framework.

- Spatial tokens come from a fixed adaptive-pool grid, so a learned 3D
  positional embedding parameterized by the pool grid size is exact.
- Temporal tokens come from a fixed-size adaptive pool over time; fixed
  sinusoidal 1D encodings are used so the encoding generalizes to any
  ``temporal_tokens`` count without retraining a lookup table.
"""

import math

import torch
import torch.nn as nn

from src.config import ModelConfig, DEFAULT_CONFIG


class PositionalEmbedding(nn.Module):
    """Learned additive positional embedding: PositionalEmbedding(num_tokens, d_model).

    Generic (cfg-free) form — indexed identically to the token axis it is
    added to, trained end-to-end. Used for the spatial token grid via
    :class:`SpatialPositionalEmbedding` (which validates the pool grid) and
    available directly whenever a bare sequence needs position identity.
    """

    def __init__(self, num_tokens: int, d_model: int, init_std: float = 0.02):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.pos = nn.Parameter(torch.zeros(num_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=init_std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected (B, N, d_model) tokens, got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"Token count {tokens.shape[1]} != positional embedding size {self.num_tokens}"
            )
        return tokens + self.pos.unsqueeze(0)


class SpatialPositionalEmbedding(nn.Module):
    """Learned positional embedding over the 3D adaptive-pool grid.

    Token order after ``flatten(2).transpose(1, 2)`` is (d, h, w) row-major,
    so the embedding table is indexed the same way.
    """

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        dd, hh, ww = cfg.spatial_pool
        tokens = dd * hh * ww
        if tokens != cfg.spatial_tokens:
            raise ValueError(
                f"spatial_pool {cfg.spatial_pool} yields {tokens} tokens "
                f"but spatial_tokens={cfg.spatial_tokens}"
            )
        self.grid = (dd, hh, ww)
        self.pos = nn.Parameter(torch.zeros(1, tokens, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.pos


class TemporalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding over the temporal token axis."""

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        self.d_model = cfg.d_model
        self.max_tokens = cfg.temporal_tokens

    @staticmethod
    def _sinusoidal(n_tokens: int, d_model: int, device, dtype) -> torch.Tensor:
        position = torch.arange(n_tokens, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(n_tokens, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        n_tokens = tokens.shape[1]
        if n_tokens > self.max_tokens:
            raise ValueError(f"Got {n_tokens} temporal tokens; configured max is {self.max_tokens}")
        pe = self._sinusoidal(n_tokens, self.d_model, tokens.device, tokens.dtype)
        return tokens + pe.unsqueeze(0)
