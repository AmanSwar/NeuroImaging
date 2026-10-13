"""Learned attention-pooling head.

v1 pooled the fused token sequence with a plain mean. A learned softmax
scorer lets the classifier *weight* tokens: pathology-bearing tokens (a
lesion, a delayed latency window) can contribute more than background.

Design notes:
- The scorer is an ``nn.Linear(d_model, 1)`` (attribute ``scorer``), whose
  near-zero init makes attention almost uniform, so this head starts where
  mean pooling ended and learns to sharpen from there. Zeroing the scorer
  recovers mean pooling EXACTLY (softmax over constant scores).
- ``forward(tokens)`` returns just the pooled vector; pass
  ``return_weights=True`` to also get the (B, N) token salience weights for
  explainability.
"""

import torch
import torch.nn as nn


class AttentionPool1D(nn.Module):
    """Soft attention pooling over a (B, N, d) token sequence -> (B, d)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scorer = nn.Linear(d_model, 1)
        nn.init.normal_(self.scorer.weight, std=0.01)
        nn.init.zeros_(self.scorer.bias)

    def forward(self, tokens: torch.Tensor, return_weights: bool = False):
        scores = self.scorer(tokens).squeeze(-1)  # (B, N)
        weights = torch.softmax(scores, dim=-1)  # (B, N)
        pooled = torch.einsum("bn,bnd->bd", weights, tokens)
        if return_weights:
            return pooled, weights
        return pooled


# Alias used across the CAFN surface.
AttentionPool = AttentionPool1D
