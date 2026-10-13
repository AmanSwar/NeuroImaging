"""2D spatial-temporal cross-attention correlation heatmaps.

Reduces the multi-head cross-attention weights (B, H, S, T) produced by
`CrossAttentionFusion` down to a single (S, T) matrix -- spatial cortex
tokens against VEP temporal latency bins -- so clinicians can see which
brain regions the model correlated with which millisecond latency window.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def cross_attention_heatmap(attn_weights: torch.Tensor | dict) -> np.ndarray:
    """Average (B, H, S, T) attention weights over batch and heads -> (S, T).

    Accepts either the raw tensor or the direction-keyed mapping CAFN now
    emits (`{"forward": ...}`; the "forward" direction — spatial tokens
    reading latency bins — is the clinically meaningful one to visualize).
    """
    if isinstance(attn_weights, dict):
        if "forward" not in attn_weights:
            raise ValueError(
                f"attention dict must contain 'forward'; got keys {sorted(attn_weights)}"
            )
        attn_weights = attn_weights["forward"]
    if attn_weights.dim() != 4:
        raise ValueError(
            f"Expected (B, H, S, T) attention weights, got shape {tuple(attn_weights.shape)}"
        )
    reduced = attn_weights.mean(dim=(0, 1))
    return reduced.detach().cpu().numpy()


def plot_cross_attention(
    attn_weights: torch.Tensor | dict,
    save_path: str,
    xlabel: str = "Temporal latency bin",
    ylabel: str = "Spatial cortex token",
    title: str = "Spatial-Temporal Cross-Attention",
) -> np.ndarray:
    """Render and save the cross-attention heatmap; returns the plotted array.

    Accepts a (B, H, S, T) tensor, or the direction-keyed dict CAFN emits --
    dict input plots the "forward" (spatial <- latency) map.
    """
    if isinstance(attn_weights, dict):
        if "forward" not in attn_weights:
            raise ValueError(
                f"attention dict must contain 'forward'; got keys {sorted(attn_weights)}"
            )
        attn_weights = attn_weights["forward"]
    heatmap = cross_attention_heatmap(attn_weights)

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return heatmap


def plot_attention_by_head(
    attn_weights: torch.Tensor,
    save_path: str,
    xlabel: str = "Temporal latency bin",
    ylabel: str = "Spatial cortex token",
    title: str = "Cross-Attention by Head",
    max_heads: int = 8,
) -> list[np.ndarray]:
    """One heatmap per attention head -- heads often specialize (some track
    the P100 window, others spread mass over baseline), which pooling across
    heads hides. Returns the per-head (S, T) arrays."""
    if attn_weights.dim() != 4:
        raise ValueError(
            f"Expected (B, H, S, T) attention weights, got shape {tuple(attn_weights.shape)}"
        )
    per_head = attn_weights.mean(dim=0).detach().cpu().numpy()  # (H, S, T)
    num_heads = min(per_head.shape[0], max_heads)

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for i in range(num_heads):
        ax = axes[i // cols][i % cols]
        im = ax.imshow(per_head[i], aspect="auto", cmap="viridis")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f"head {i}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for j in range(num_heads, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return [per_head[i] for i in range(num_heads)]
