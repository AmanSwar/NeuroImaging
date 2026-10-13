"""Multi-Head Cross-Attention Fusion Module (CAFM).

Spatial 3D tokens are projected as Queries; temporal 1D latency tokens are
projected as Keys and Values, so each spatial brain region dynamically
queries the electrophysiological latency bins:

    Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V

v2 changes:
- **Pre-norm** block structure (LayerNorm before attention/FFN instead of
  after the residual), which is markedly more stable to train and allows
  deeper stacks without warmup gymnastics.
- Configurable fusion depth (`ModelConfig.fusion_blocks`): deeper stacks
  refine the queries over multiple rounds of cross-modal retrieval while
  K/V stay anchored to the raw temporal tokens.
- A final LayerNorm normalizes the fused token sequence before pooling.
"""

import math

import torch
import torch.nn as nn

from src.config import ModelConfig, DEFAULT_CONFIG


class MultiHeadCrossAttention(nn.Module):
    """Q from `query_tokens`, K/V from `kv_tokens`. Sequence lengths may differ."""

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        if cfg.d_model % cfg.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        self.d_model = cfg.d_model

        self.w_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Dh)

    def forward(self, query_tokens: torch.Tensor, kv_tokens: torch.Tensor):
        b = query_tokens.shape[0]
        q = self._split_heads(self.w_q(query_tokens))
        k = self._split_heads(self.w_k(kv_tokens))
        v = self._split_heads(self.w_v(kv_tokens))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)  # (B, H, Nq, Nk)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # (B, H, Nq, Dh)

        context = context.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.out_proj(context)
        return out, attn


class _CrossAttentionBlock(nn.Module):
    """One pre-norm cross-attention + FFN block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadCrossAttention(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, cfg.d_model),
        )

    def forward(self, spatial_tokens: torch.Tensor, temporal_tokens: torch.Tensor):
        attn_out, attn_weights = self.cross_attn(self.norm1(spatial_tokens), temporal_tokens)
        fused = spatial_tokens + self.dropout(attn_out)
        fused = fused + self.dropout(self.ffn(self.norm2(fused)))
        return fused, attn_weights


class CrossAttentionFusion(nn.Module):
    """CAFM: stacked pre-norm cross-attention blocks -> final LayerNorm."""

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        if cfg.fusion_blocks < 1:
            raise ValueError(f"fusion_blocks must be >= 1, got {cfg.fusion_blocks}")
        self.blocks = nn.ModuleList(_CrossAttentionBlock(cfg) for _ in range(cfg.fusion_blocks))
        self.norm_out = nn.LayerNorm(cfg.d_model)

    def forward(self, spatial_tokens: torch.Tensor, temporal_tokens: torch.Tensor):
        fused = spatial_tokens
        attn_by_direction: dict = {}
        for block in self.blocks:
            fused, block_attn = block(fused, temporal_tokens)
            if isinstance(block_attn, dict):
                for name, w in block_attn.items():
                    attn_by_direction[name] = w
            else:
                attn_by_direction["forward"] = block_attn
        # Single attention map per direction for downstream interpretability;
        # with depth > 1 later blocks refine the queries, so the final map is
        # the most representative of the decision the head actually makes.
        return self.norm_out(fused), attn_by_direction


class _BidirectionalBlock(nn.Module):
    """One gated co-attention round updating BOTH streams.

    Temporal stream updates first so the spatial read (the direction that
    feeds the fused head) consumes the *enriched* temporal representation:

        t' = t + sig_t * CA_{T<-S}(norm(t), s);  t' = t' + FFN_t(norm(t'))
        s' = s + sig_s * CA_{S<-T}(norm(s), t'); s' = s' + FFN_s(norm(s'))

    The gates live on the parent fusion module (fusion-level
    `gate_spatial`/`gate_temporal` parameters) and are shared by all blocks;
    they are zero-initialized so sigmoid(0) = 0.5 -- both directions exactly
    half-open at init -- and each direction can learn to close itself.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_s = nn.LayerNorm(cfg.d_model)
        self.norm_t = nn.LayerNorm(cfg.d_model)
        self.cross_attn_s2t = MultiHeadCrossAttention(cfg)
        self.cross_attn_t2s = MultiHeadCrossAttention(cfg)
        self.ffn_s = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, cfg.d_model),
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, cfg.d_model),
        )
        self.norm_s2 = nn.LayerNorm(cfg.d_model)
        self.norm_t2 = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        sig_s: torch.Tensor,
        sig_t: torch.Tensor,
    ):
        t_attn_out, w_t2s = self.cross_attn_t2s(self.norm_t(temporal_tokens), spatial_tokens)
        t = temporal_tokens + sig_t * t_attn_out
        t = t + self.ffn_t(self.norm_t2(t))

        s_attn_out, w_s2t = self.cross_attn_s2t(self.norm_s(spatial_tokens), t)
        s = spatial_tokens + sig_s * s_attn_out
        s = s + self.ffn_s(self.norm_s2(s))

        return s, t, {"forward": w_s2t, "backward": w_t2s}


class BidirectionalCrossAttentionFusion(nn.Module):
    """Gated bidirectional co-attention: each stream queries the other.

    Returns `(fused_spatial, fused_temporal, {"forward": W_s2t, "backward":
    W_t2s})`. "forward" is the clinically canonical spatial-reads-latency map
    (B, H, S, T); "backward" shows which spatial regions each latency bin
    drew from (B, H, T, S).

    `gate_spatial`/`gate_temporal` are zero-initialized scalar gates on this
    module: sigmoid(0) = 0.5, so both directions start exactly half-open.
    """

    def __init__(self, cfg: ModelConfig = DEFAULT_CONFIG.model):
        super().__init__()
        if cfg.fusion_blocks < 1:
            raise ValueError(f"fusion_blocks must be >= 1, got {cfg.fusion_blocks}")
        self.blocks = nn.ModuleList(_BidirectionalBlock(cfg) for _ in range(cfg.fusion_blocks))
        self.gate_spatial = nn.Parameter(torch.zeros(()))
        self.gate_temporal = nn.Parameter(torch.zeros(()))
        self.norm_out_s = nn.LayerNorm(cfg.d_model)

    def forward(self, spatial_tokens: torch.Tensor, temporal_tokens: torch.Tensor):
        sig_s = torch.sigmoid(self.gate_spatial)
        sig_t = torch.sigmoid(self.gate_temporal)
        s, t = spatial_tokens, temporal_tokens
        attn: dict = {}
        for block in self.blocks:
            s, t, attn = block(s, t, sig_s, sig_t)
        # Final norm applies to the stream the fused head consumes; the
        # output temporal stream is already normalized inside each block.
        return self.norm_out_s(s), t, attn
