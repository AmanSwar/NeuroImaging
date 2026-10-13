"""Multi-task loss implementations.

The canonical fused-plus-two-auxiliary supervised loss lives here so the
training loop, the model convenience method, and tests share one
implementation.

Two weighting schemes:

- `multi_task_ce`: fixed-weight sum (v1 behavior, default). Optionally with
  label smoothing on every head for calibration-friendly training.
- `UncertaintyWeightedLoss`: learnable homoscedastic task weights
  (Kendall et al., CVPR 2018) -- each task is scaled by 1/(2*sigma_t^2) with
  learnable log-variance, so the network automatically balances how hard it
  leans on the fused vs. auxiliary objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_PART_KEYS = ("loss_fused", "loss_spatial", "loss_temporal", "loss_total")


def multi_task_ce(
    outputs: dict,
    labels: torch.Tensor,
    aux_weight: float = 0.3,
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
):
    """Fused CE + `aux_weight` * (spatial CE + temporal CE). Returns (total, parts).

    `class_weights` (optional tensor of shape (num_classes,)) up-weights
    minority classes in every head -- used with `TrainConfig.class_weights="auto"`.
    """
    loss_fused = F.cross_entropy(
        outputs["logits_fused"], labels, weight=class_weights, label_smoothing=label_smoothing
    )
    loss_spatial = F.cross_entropy(
        outputs["logits_spatial"], labels, weight=class_weights, label_smoothing=label_smoothing
    )
    loss_temporal = F.cross_entropy(
        outputs["logits_temporal"], labels, weight=class_weights, label_smoothing=label_smoothing
    )
    total = loss_fused + aux_weight * (loss_spatial + loss_temporal)
    return total, {
        "loss_fused": loss_fused.item(),
        "loss_spatial": loss_spatial.item(),
        "loss_temporal": loss_temporal.item(),
        "loss_total": total.item(),
    }


class UncertaintyWeightedLoss(nn.Module):
    """Learnable per-task weights via homoscedastic uncertainty.

    total = exp(-s_fused)*L_fused + s_fused
          + exp(-s_spatial)*L_spatial + s_spatial
          + exp(-s_temporal)*L_temporal + s_temporal

    where s_task is a learnable log-variance. The additive regularizer
    prevents the trivial "inflate all sigmas" solution. Init at s=0 means all
    three tasks start equally weighted (sigma=1).
    """

    def __init__(self, task_names: tuple[str, ...] = ("fused", "spatial", "temporal")):
        super().__init__()
        self.task_names = task_names
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in task_names}
        )

    def forward(self, outputs: dict, labels: torch.Tensor, label_smoothing: float = 0.0):
        losses = {
            "fused": F.cross_entropy(outputs["logits_fused"], labels, label_smoothing=label_smoothing),
            "spatial": F.cross_entropy(outputs["logits_spatial"], labels, label_smoothing=label_smoothing),
            "temporal": F.cross_entropy(outputs["logits_temporal"], labels, label_smoothing=label_smoothing),
        }
        parts = {}
        total = 0.0
        for name in self.task_names:
            weighted = torch.exp(-self.log_vars[name]) * losses[name] + self.log_vars[name]
            total = total + weighted
            parts[f"loss_{name}"] = losses[name].item()
            parts[f"weight_{name}"] = float(torch.exp(-self.log_vars[name]).item())
        return total, parts
