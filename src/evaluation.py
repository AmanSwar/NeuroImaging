"""Detailed evaluation utilities: research-grade metrics and figures.

``evaluate`` in :mod:`src.train` reports loss/accuracy only. This module
adds the metrics reviewers and clinicians actually ask for:

- balanced accuracy and macro/weighted F1 (robust to class imbalance),
- per-class precision/recall/F1,
- the full confusion matrix (as nested lists and as a rendered figure),
- ROC-AUC (one-vs-rest) when the loader is non-degenerate,

plus a training-history curve plot for quick visual QA of a run.

Only numpy / scikit-learn / matplotlib are used, so these helpers work on
any loader yielding the CAFN batch dict.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import CLASS_NAMES


def _collect_predictions(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                         device: torch.device, criterion=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Run the model over ``loader``.

    Returns (labels, preds, fused_probs, mean_loss) where mean_loss is
    ``None`` when no criterion is supplied.
    """
    was_training = model.training
    model.eval()
    labels, preds, probs = [], [], []
    loss_sum, n_seen = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            mri = batch["mri"].to(device)
            vep = batch["vep"].to(device)
            outputs = model(mri, vep)
            prob = torch.softmax(outputs["logits_fused"], dim=1)
            labels.append(batch["label"].cpu().numpy())
            preds.append(prob.argmax(dim=1).cpu().numpy())
            probs.append(prob.cpu().numpy())
            if criterion is not None:
                y = batch["label"].to(device)
                loss_sum += float(criterion(outputs, y)) * y.size(0)
                n_seen += y.size(0)
    if was_training:
        model.train()
    if not labels:
        return np.empty(0), np.empty(0), np.empty((0, len(CLASS_NAMES))), None
    mean_loss = loss_sum / n_seen if criterion is not None else None
    return np.concatenate(labels), np.concatenate(preds), np.concatenate(probs), mean_loss


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict:
    """Compute the full metric suite from label/prediction arrays.

    Returns a JSON-serialisable dict: accuracy, balanced_accuracy,
    macro_f1, weighted_f1, per_class report, confusion_matrix, and
    roc_auc_ovr when probabilities with all classes present are given.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty set")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"label/prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.min() < 0 or y_true.max() >= len(class_names):
        raise ValueError("labels fall outside the class-name range")

    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [class_names[i] for i in present]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for name, p, r, f, s in zip(target_names, precision, recall, f1, support)
    }

    result: dict = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
        "n_samples": int(y_true.size),
        "per_class": per_class,
        "empty": False,
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        present_labels = sorted(set(y_true.tolist()))
        # Multi-class OvR AUC is only well-defined when every class is
        # represented; degenerate subsets would yield NaN.
        if (
            y_prob.shape == (y_true.size, len(class_names))
            and present_labels == list(range(len(class_names)))
        ):
            try:
                result["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr")
                )
            except ValueError:
                pass  # degenerate probability matrix; AUC silently omitted
    return result


def evaluate_detailed(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion=None,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict:
    """Full evaluation pass: loss (optional) plus :func:`classification_metrics`."""
    y_true, y_pred, y_prob, mean_loss = _collect_predictions(model, loader, device, criterion)
    if y_true.size == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "empty": True}

    result = classification_metrics(y_true, y_pred, y_prob, class_names=class_names)
    result["empty"] = False
    if mean_loss is not None:
        result["loss"] = mean_loss
    return result


def print_classification_report(metrics: dict, class_names: tuple[str, ...] = CLASS_NAMES) -> str:
    """Human-readable console summary of :func:`classification_metrics` output."""
    lines = [
        f"n={metrics['n_samples']}  acc={metrics['accuracy']:.4f}  "
        f"bal-acc={metrics['balanced_accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}"
    ]
    if "roc_auc_ovr" in metrics:
        lines.append(f"ROC-AUC (OvR)={metrics['roc_auc_ovr']:.4f}")
    lines.append(f"{'class':<24}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
    for name in class_names:
        row = metrics["per_class"].get(name)
        if row is None:
            continue
        lines.append(
            f"{name:<24}{row['precision']:>10.4f}{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}{row['support']:>9d}"
        )
    return "\n".join(lines)


def plot_confusion_matrix(
    confusion: list[list[int]],
    save_path: str,
    class_names: tuple[str, ...] = CLASS_NAMES,
    title: str = "Confusion matrix (fused head)",
) -> np.ndarray:
    """Render a labelled confusion-matrix figure; returns the matrix array."""
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square confusion matrix, got shape {matrix.shape}")
    if matrix.shape[0] != len(class_names):
        raise ValueError(
            f"matrix has {matrix.shape[0]} classes but {len(class_names)} class names given"
        )

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    thresh = matrix.max() / 2 if matrix.max() else 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                color="white" if matrix[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return matrix


def plot_training_curves(history: dict, save_path: str, title: str = "CAFN training") -> None:
    """Plot train/val loss and val accuracy over epochs; saves a PNG."""
    if "train_loss" not in history or not history["train_loss"]:
        raise ValueError("history must contain a non-empty 'train_loss' list")

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, history["train_loss"], "o-", label="train", ms=3)
    if history.get("val_loss"):
        ax_loss.plot(epochs, history["val_loss"], "s-", label="val", ms=3)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Multi-task loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    if history.get("val_accuracy"):
        ax_acc.plot(epochs, history["val_accuracy"], "s-", color="#2E7D5B", ms=3)
        best = int(np.argmax(history["val_accuracy"]))
        ax_acc.annotate(
            f"best {history['val_accuracy'][best]:.3f} (ep {best + 1})",
            xy=(best + 1, history["val_accuracy"][best]),
            xytext=(5, -12), textcoords="offset points", fontsize=8,
        )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Validation accuracy")
    ax_acc.grid(alpha=0.3)
    if history.get("lr"):
        ax2 = ax_loss.twinx()
        ax2.plot(epochs, history["lr"], ":", color="#A8691A", lw=1)
        ax2.set_ylabel("LR", color="#A8691A", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
