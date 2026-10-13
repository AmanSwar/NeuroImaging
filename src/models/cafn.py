"""Cross-Attention Fusion Network (CAFN): full model wiring.

Dual-stream encoders feed the fusion module; a fused classification head and
two auxiliary unimodal heads train jointly (multi-task supervision) so the
network cannot collapse onto a single dominant modality.

v2 upgrades (all configured via `ModelConfig`, defaults on):
- Bidirectional gated co-attention (`fusion="bidirectional"`, default): each
  stream queries the other; `fusion="cross"` recovers the v1 one-way design.
- Positional embeddings (`pos_encoding=True`): learned over the pooled 3D
  spatial grid, sinusoidal over the temporal latency axis.
- Learned attention pooling for the fused head (`pooling="attn"`), which
  starts at mean pooling -- `pooling="mean"` recovers v1 exactly.
- Optional train-time modality dropout (`modality_dropout`): zeroes a
  sample's VEP tokens *for the fusion path only* -- auxiliary heads always
  see clean inputs -- so the fused head cannot lean on one stream.
- Stackable fusion depth (`fusion_blocks`).
- The canonical multi-task loss lives in `src.models.losses`;
  `CAFN.multi_task_loss` delegates to it (with optional class weights).
"""

import torch
import torch.nn as nn

from src.config import ModelConfig, DEFAULT_CONFIG
from src.models.attention_pool import AttentionPool1D
from src.models.cross_attention import BidirectionalCrossAttentionFusion, CrossAttentionFusion
from src.models.losses import multi_task_ce
from src.models.positional import (
    PositionalEmbedding,
    SpatialPositionalEmbedding,
    TemporalPositionalEmbedding,
)
from src.models.spatial_encoder import SpatialEncoder3D
from src.models.temporal_encoder import TemporalEncoder1D

# Public re-exports: the pooling/positional building blocks are importable
# directly from the CAFN surface.
__all__ = [
    "CAFN",
    "AttentionPool",
    "AttentionPool1D",
    "PositionalEmbedding",
    "SpatialPositionalEmbedding",
    "TemporalPositionalEmbedding",
]
AttentionPool = AttentionPool1D
from src.models.cross_attention import BidirectionalCrossAttentionFusion, CrossAttentionFusion
from src.models.losses import multi_task_ce
from src.models.positional import (
    PositionalEmbedding,
    SpatialPositionalEmbedding,
    TemporalPositionalEmbedding,
)
from src.models.spatial_encoder import SpatialEncoder3D
from src.models.temporal_encoder import TemporalEncoder1D

FUSION_MODES = ("bidirectional", "cross")


class CAFN(nn.Module):
    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        if cfg.fusion not in FUSION_MODES:
            raise ValueError(f"Unknown fusion mode {cfg.fusion!r}; expected one of {FUSION_MODES}")
        if cfg.pooling not in ("attn", "mean"):
            raise ValueError(f"pooling must be 'attn' or 'mean', got {cfg.pooling!r}")
        if not 0.0 <= cfg.modality_dropout < 1.0:
            raise ValueError(f"modality_dropout must be in [0, 1), got {cfg.modality_dropout}")
        self.cfg = cfg
        self.spatial_encoder = SpatialEncoder3D(cfg)
        self.temporal_encoder = TemporalEncoder1D(cfg)
        self.spatial_pos = SpatialPositionalEmbedding(cfg) if cfg.pos_encoding else None
        self.temporal_pos = TemporalPositionalEmbedding(cfg) if cfg.pos_encoding else None
        if cfg.fusion == "bidirectional":
            self.fusion = BidirectionalCrossAttentionFusion(cfg)
        else:
            self.fusion = CrossAttentionFusion(cfg)

        self.fused_pool = AttentionPool1D(cfg.d_model) if cfg.pooling == "attn" else None
        self.fused_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.spatial_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.temporal_head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(self, mri_volume: torch.Tensor, vep_signal: torch.Tensor) -> dict:
        spatial_tokens = self.spatial_encoder(mri_volume)    # (B, S, d)
        temporal_tokens = self.temporal_encoder(vep_signal)  # (B, T, d)

        if self.spatial_pos is not None:
            spatial_tokens = self.spatial_pos(spatial_tokens)
        if self.temporal_pos is not None:
            temporal_tokens = self.temporal_pos(temporal_tokens)

        fusion_temporal = temporal_tokens
        modality_dropped = None
        if self.training and self.cfg.modality_dropout > 0.0:
            # Zero the VEP stream for a random subset of samples -- but only
            # inside the fusion path, so auxiliary unimodal heads still learn.
            keep = (
                torch.rand(vep_signal.shape[0], device=vep_signal.device)
                >= self.cfg.modality_dropout
            )
            modality_dropped = ~keep
            if not keep.all():
                fusion_temporal = temporal_tokens * keep.view(-1, 1, 1).to(temporal_tokens.dtype)

        fusion_out = self.fusion(spatial_tokens, fusion_temporal)
        fused_tokens = fusion_out[0]
        attn_maps = fusion_out[-1] if self.cfg.fusion == "cross" else fusion_out[2]
        fused_temporal = fusion_out[1] if self.cfg.fusion == "bidirectional" else None

        if self.fused_pool is not None:
            fused_embedding, pool_weights = self.fused_pool(fused_tokens, return_weights=True)
        else:
            fused_embedding = fused_tokens.mean(dim=1)
            pool_weights = None
        spatial_embedding = spatial_tokens.mean(dim=1)
        temporal_embedding = temporal_tokens.mean(dim=1)

        outputs = {
            "logits_fused": self.fused_head(fused_embedding),
            "logits_spatial": self.spatial_head(spatial_embedding),
            "logits_temporal": self.temporal_head(temporal_embedding),
            "attn_weights": attn_maps["forward"],  # (B, H, S, T): cortex reads latency
            "attn_weights_backward": attn_maps.get("backward"),  # (B, H, T, S) | None
            "fused_tokens": fused_tokens,
            "fused_temporal_tokens": fused_temporal,
            "spatial_tokens": spatial_tokens,
            "temporal_tokens": temporal_tokens,
            "pool_weights": pool_weights,  # (B, S) token salience from the pool head
        }
        if modality_dropped is not None:
            outputs["modality_dropped"] = modality_dropped
        return outputs

    def multi_task_loss(
        self,
        outputs: dict,
        labels: torch.Tensor,
        aux_weight: float = 0.3,
        class_weights: torch.Tensor | None = None,
    ):
        total, parts = multi_task_ce(
            outputs, labels, aux_weight=aux_weight, class_weights=class_weights
        )
        return total, parts

    @torch.no_grad()
    def embed(self, mri_volume: torch.Tensor, vep_signal: torch.Tensor) -> torch.Tensor:
        """Convenience API: fused embedding for feature extraction/probing."""
        was_training = self.training
        self.eval()
        spatial_tokens = self.spatial_encoder(mri_volume)
        temporal_tokens = self.temporal_encoder(vep_signal)
        if self.spatial_pos is not None:
            spatial_tokens = self.spatial_pos(spatial_tokens)
        if self.temporal_pos is not None:
            temporal_tokens = self.temporal_pos(temporal_tokens)
        fused_tokens = self.fusion(spatial_tokens, temporal_tokens)[0]
        if self.fused_pool is not None:
            pooled = self.fused_pool(fused_tokens)
        else:
            pooled = fused_tokens.mean(dim=1)
        if was_training:
            self.train()
        return pooled
