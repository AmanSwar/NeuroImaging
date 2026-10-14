"""Multi-task training loop for CAFN.

Trains the fused cross-attention head jointly with the two auxiliary
unimodal heads (`CAFN.multi_task_loss`). Callable as a library function
(`train_model`) for tests/notebooks, or as a script:

    python -m src.train --epochs 20 --synthetic-samples 64
    python -m src.train --manifest data/raw/manifest.csv --amp --device auto

v2 pipeline features (config/CLI-gated, behavior-safe by default):
- AdamW (+ adam/sgd) with a linear-warmup cosine LR schedule, stepped per batch;
- gradient-norm clipping, CUDA/MPS mixed precision, early stopping with
  best-weights restore;
- resume-warm-start from `last.pt` (model + optimizer + scheduler state);
- stratified train/val split and inverse-frequency class weighting;
- rich validation metrics: accuracy, balanced accuracy, macro-F1, and the
  confusion matrix (written to `<run>/final_metrics.json`);
- per-run artifact directory `<checkpoint_dir>/<run_name>/` with best.pt,
  last.pt, history.json, final_metrics.json and training_curves.png;
- train-only augmentation via `AugmentedView` (wrapped train indices), so a
  shared underlying dataset never leaks augmentation into validation;
- deterministic seeding of python/numpy/torch and DataLoader workers.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import CLASS_NAMES, Config, DEFAULT_CONFIG

# --- defensive imports (utils live in a package OR a flat module) ----------
try:
    from src.utils import seed_everything  # type: ignore
except ImportError:
    try:
        from src.utils.setup import seed_everything  # type: ignore
    except ImportError:
        def seed_everything(seed: int, deterministic: bool = False) -> None:  # type: ignore
            random.seed(seed)
            np.random.seed(seed % (2**31))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

try:
    from src.utils import auto_device as resolve_device  # type: ignore
except ImportError:
    try:
        from src.utils.setup import resolve_device  # type: ignore
    except ImportError:
        def resolve_device(preference: str = "auto") -> torch.device:  # type: ignore
            if preference != "auto":
                return torch.device(preference)
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")


HISTORY_LIST_KEYS = (
    "train_loss", "train_accuracy", "val_loss", "val_accuracy",
    "val_balanced_accuracy", "val_macro_f1", "learning_rate", "runtime_s",
)


# ---------------------------------------------------------------------------
# Splitting & labels
# ---------------------------------------------------------------------------

def _extract_labels(dataset) -> list[int] | None:
    labels = getattr(dataset, "labels", None)
    if isinstance(labels, (list, tuple)) and len(labels):
        return [int(x) for x in labels]
    rows = getattr(dataset, "rows", None)
    if isinstance(rows, list) and rows and "label" in rows[0]:
        return [int(r["label"]) for r in rows]
    return None


def _stratified_partition(labels: list[int], val_split: float, seed: int):
    """Per-class shuffle then proportional allocation to the val split."""
    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    train_idx: list[int] = []
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)
    for label, indices in sorted(by_class.items()):
        indices = np.array(indices)
        rng.shuffle(indices)
        n_val = int(round(len(indices) * val_split))
        if len(indices) > 1:  # keep at least one training sample per class
            n_val = min(n_val, len(indices) - 1)
        val_idx.extend(indices[:n_val].tolist())
        train_idx.extend(indices[n_val:].tolist())
    return train_idx, val_idx


def train_val_split(
    dataset,
    val_split: float,
    seed: int,
    stratify: bool = False,
    group_by: str | None = None,
) -> tuple[Subset, Subset]:
    """Deterministic train/val split.

    ``group_by`` (e.g. ``"subject_id"``) assigns whole groups -- never single
    rows -- to each side, preventing within-subject leakage in longitudinal
    cohorts. ``stratify`` keeps class proportions row-wise. Group split wins
    when both are requested (it subsumes leakage-free stratification for
    balanced cohorts and falls back to stratified when no groups exist).
    """
    if group_by:
        from src.utils import group_train_val_split

        return group_train_val_split(dataset, val_split, seed, group_key=group_by)

    n = len(dataset)
    n_val = max(1, int(n * val_split)) if n > 1 else 0

    labels = _extract_labels(dataset) if stratify else None
    if stratify and labels is not None and len(labels) == n and n_val >= 1:
        train_idx, val_idx = _stratified_partition(labels, val_split, seed)
        if not train_idx:
            train_idx, val_idx = val_idx, []
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if not train_idx:
        train_idx, val_idx = perm, []
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, mean-normalized to 1.0."""
    from src.data.dataset import compute_class_weights

    return compute_class_weights(list(labels), num_classes)


def _resolve_class_weights(train_dataset, num_classes: int, mode) -> torch.Tensor | None:
    if mode in (None, False, "", "none"):
        return None
    dataset = getattr(train_dataset, "dataset", train_dataset)
    indices = getattr(train_dataset, "indices", None)
    labels = _extract_labels(dataset)
    if labels is None:
        return None
    if indices is not None:
        labels = [labels[i] for i in indices]
    return _compute_class_weights(labels, num_classes)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: float = 1.0):
    """Linear warmup over `warmup_epochs`, then cosine decay to ~0.

    Stepped once per optimizer step; `total_steps = steps_per_epoch * epochs`.
    """
    total_steps = max(1, int(steps_per_epoch * epochs))
    warmup_steps = min(total_steps, max(1, int(warmup_epochs * steps_per_epoch)))

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _class_names(num_classes: int) -> tuple[str, ...]:
    if num_classes == len(CLASS_NAMES):
        return CLASS_NAMES
    return tuple(f"class_{i}" for i in range(num_classes))


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, amp: bool = False) -> dict:
    """Validation metrics: loss, accuracy, balanced accuracy, macro-F1,
    confusion matrix, and sample count."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    autocast_device = device.type if device.type in ("cuda", "mps") else "cpu"
    num_classes = None
    for batch in loader:
        mri = batch["mri"].to(device)
        vep = batch["vep"].to(device)
        labels = batch["label"].to(device)
        with torch.autocast(device_type=autocast_device, enabled=amp):
            outputs = model(mri, vep)
            logits = {
                name: outputs[name].float()
                for name in ("logits_fused", "logits_spatial", "logits_temporal")
            }
            loss, _ = model.multi_task_loss(logits, labels)
        preds = outputs["logits_fused"].argmax(dim=1)
        num_classes = outputs["logits_fused"].shape[1]
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if total == 0:
        return {
            "loss": float("nan"), "accuracy": float("nan"),
            "balanced_accuracy": float("nan"), "macro_f1": float("nan"),
            "confusion_matrix": np.zeros((num_classes or 1, num_classes or 1), dtype=np.int64),
            "n_samples": 0,
        }

    detailed = None
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    preds_all = np.concatenate(all_preds)
    labels_all = np.concatenate(all_labels)
    np.add.at(cm, (labels_all, preds_all), 1)

    metrics = {
        "loss": loss_sum / total,
        "accuracy": float((labels_all == preds_all).mean()),
        "confusion_matrix": cm,
        "n_samples": total,
    }
    # Derive the clinical metrics from the confusion matrix directly so the
    # output contract holds regardless of which metrics helper is installed.
    cmf = cm.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(cmf.sum(axis=1) > 0, np.diag(cmf) / np.maximum(cmf.sum(axis=1), 1), 0.0)
        precision = np.where(cmf.sum(axis=0) > 0, np.diag(cmf) / np.maximum(cmf.sum(axis=0), 1), 0.0)
        f1_denom = precision + recall
        f1 = np.where(f1_denom > 0, 2 * precision * recall / np.maximum(f1_denom, 1e-12), 0.0)
    metrics["balanced_accuracy"] = float(recall.mean())
    metrics["macro_f1"] = float(f1.mean())
    metrics["per_class"] = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i, name in enumerate(_class_names(num_classes))
    }
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_loader_kwargs(cfg_train, device: torch.device) -> dict:
    workers = int(getattr(cfg_train, "num_workers", 0) or 0)
    seed = int(getattr(cfg_train, "seed", 0))
    return dict(
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=lambda wid: np.random.seed(seed + wid),
    )


def _wrap_train_view(train_dataset, use_augment: bool, seed: int):
    """Wrap TRAIN indices in an AugmentedView; validation stays untouched."""
    if not use_augment:
        return train_dataset
    try:
        from src.data.dataset import AugmentedView
    except ImportError:
        return train_dataset
    indices = getattr(train_dataset, "indices", None)
    if indices is None:  # plain dataset passed directly
        indices = list(range(len(train_dataset)))
        dataset = train_dataset
    else:
        dataset = train_dataset.dataset
    return AugmentedView(dataset, indices, mri_augment=True, vep_augment=True, seed=seed)


def _build_optimizer(model, tcfg):
    mode = str(getattr(tcfg, "optimizer", "adamw") or "adamw").lower()
    if mode == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    if mode == "adam":
        return torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    if mode == "sgd":
        return torch.optim.SGD(model.parameters(), lr=tcfg.lr,
                               weight_decay=tcfg.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer {mode!r}")


def train_model(
    model,
    train_dataset,
    val_dataset=None,
    cfg: Config = DEFAULT_CONFIG,
    device: torch.device | None = None,
    verbose: bool = False,
    run_name: str | None = None,
    resume: str | None = None,
    augment: bool | None = None,
) -> dict:
    """Runs the multi-task training loop and returns the run history.

    History contains the per-epoch list keys (train_loss, train_accuracy,
    val_loss, val_accuracy, val_balanced_accuracy, val_macro_f1,
    learning_rate, runtime_s) plus the scalars epochs_run and
    restored_best. When `run_name` is given, artifacts (best.pt, last.pt,
    history.json, final_metrics.json, training_curves.png) are written to
    `<checkpoint_dir>/<run_name>/`.

    `resume` warm-starts from a checkpoint saved by this module: model,
    optimizer and scheduler state are restored and training continues for
    `cfg.train.epochs` further epochs; the returned history covers only
    those new epochs.
    """
    tcfg = cfg.train
    if augment is None:
        augment = bool(getattr(tcfg, "augment", False))
    run_dir = None
    if run_name:
        run_dir = os.path.join(tcfg.checkpoint_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

    use_amp = bool(getattr(tcfg, "amp", False))
    if device is None:
        device = torch.device("cpu")
    if use_amp and device.type not in ("cuda", "mps"):
        use_amp = False
    autocast_device = device.type if device.type in ("cuda", "mps") else "cpu"
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    model.to(device)

    train_ds = _wrap_train_view(train_dataset, bool(augment), int(tcfg.seed) + 1)
    loader_kwargs = _make_loader_kwargs(tcfg, device)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, **loader_kwargs)
    steps_per_epoch = max(1, len(train_loader))
    val_loader = (
        DataLoader(val_dataset, batch_size=tcfg.batch_size, shuffle=False, **loader_kwargs)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    num_classes = int(getattr(getattr(model, "cfg", None), "num_classes", 3))
    class_weights = _resolve_class_weights(
        train_dataset, num_classes, getattr(tcfg, "class_weights", None)
        or (True if getattr(tcfg, "balance_classes", False) else None)
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    import inspect

    _loss_params = inspect.signature(model.multi_task_loss).parameters
    loss_kwargs: dict = {"aux_weight": tcfg.aux_loss_weight}
    if "label_smoothing" in _loss_params:
        loss_kwargs["label_smoothing"] = label_smoothing
    if "class_weights" in _loss_params:
        loss_kwargs["class_weights"] = class_weights

    optimizer = _build_optimizer(model, tcfg)
    grad_clip = getattr(tcfg, "max_grad_norm", None) or getattr(tcfg, "grad_clip_norm", None)
    label_smoothing = float(getattr(tcfg, "label_smoothing", 0.0) or 0.0)
    patience = getattr(tcfg, "patience", None)  # None = off; 0 = stop after 1 plateau epoch

    select_metric = str(getattr(tcfg, "select_metric", "val_loss") or "val_loss")
    if select_metric not in ("val_loss", "val_balanced_accuracy", "val_macro_f1"):
        raise ValueError(
            f"select_metric must be val_loss, val_balanced_accuracy or val_macro_f1; got {select_metric!r}"
        )
    best_score = float("-inf")
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        payload = load_full_checkpoint(resume)
        model.load_state_dict(payload["model_state_dict"])
        if payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        prev_val_losses = [
            float(v) for v in (payload.get("history") or {}).get("val_loss", [])
            if isinstance(v, (int, float)) and math.isfinite(v)
        ]
        if prev_val_losses:
            best_val_loss = min(prev_val_losses)
            start_epoch = int(payload.get("epoch", len(prev_val_losses)))
        if verbose:
            print(f"resumed weights from {resume}; continuing for {tcfg.epochs} epochs")

    scheduler = None
    scheduler_name = str(getattr(tcfg, "scheduler", "cosine") or "none")
    if scheduler_name == "cosine":
        warmup = float(getattr(tcfg, "warmup_epochs", None) or 0)
        if not warmup:  # prefer warmup_fraction (fraction of steps) when present
            warmup = float(getattr(tcfg, "warmup_fraction", 0.1)) * tcfg.epochs
        scheduler = _build_scheduler(optimizer, steps_per_epoch, tcfg.epochs, warmup)
        if resume:
            # Fast-forward the schedule to the resumed point. PyTorch warns
            # about scheduler.step() before optimizer.step() here; that
            # ordering concern does not apply -- this is pure state catch-up,
            # so the specific warning is suppressed.
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*lr_scheduler\.step\(\).*before.*optimizer\.step\(\).*",
                )
                for _ in range(start_epoch * steps_per_epoch):
                    scheduler.step()

    history: dict = {key: [] for key in HISTORY_LIST_KEYS}
    epochs_run = 0
    restored_best = False
    best_state: dict | None = None
    stopped_early = False
    epochs_since_best = 0

    for epochs_run in range(1, tcfg.epochs + 1):
        model.train()
        epoch_loss, n_samples, n_correct = 0.0, 0, 0
        t0 = time.time()
        for batch in train_loader:
            mri = batch["mri"].to(device)
            vep = batch["vep"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, enabled=use_amp, dtype=amp_dtype):
                outputs = model(mri, vep)
                loss, _ = model.multi_task_loss(outputs, labels, **loss_kwargs)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            n_correct += (outputs["logits_fused"].argmax(dim=1) == labels).sum().item()

        history["train_loss"].append(epoch_loss / max(n_samples, 1))
        history["train_accuracy"].append(n_correct / max(n_samples, 1))
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["runtime_s"].append(time.time() - t0)

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, amp=use_amp)
            history["val_loss"].append(metrics["loss"])
            history["val_accuracy"].append(metrics["accuracy"])
            history["val_balanced_accuracy"].append(metrics["balanced_accuracy"])
            history["val_macro_f1"].append(metrics["macro_f1"])

            if select_metric == "val_loss":
                improved = metrics["loss"] < best_val_loss - 1e-5
            else:  # higher-is-better score; ties are broken on val loss
                score = float(metrics[select_metric[len("val_"):]])
                improved = score > best_score + 1e-9 or (
                    abs(score - best_score) <= 1e-9 and metrics["loss"] < best_val_loss - 1e-5
                )
            if improved:
                best_val_loss = metrics["loss"]
                if select_metric != "val_loss":
                    best_score = score
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            if verbose:
                print(
                    f"epoch {start_epoch + epochs_run} "
                    f"train_loss={history['train_loss'][-1]:.4f} "
                    f"val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.4f} "
                    f"val_f1={metrics['macro_f1']:.4f} lr={history['learning_rate'][-1]:.2e}"
                )

            # patience=None disables; patience=k stops after k+1 plateau epochs
            if patience is not None and epochs_since_best > patience:
                if verbose:
                    print(f"early stopping: no val improvement for {epochs_since_best} epoch(s)")
                stopped_early = True
                break
        elif verbose:
            print(f"epoch {start_epoch + epochs_run} train_loss={history['train_loss'][-1]:.4f}")

    # Restore the best validation weights whenever one was captured.
    if best_state is not None:
        model.load_state_dict(best_state)
        restored_best = True

    # Deploy-safe default: after training the model is left in eval mode so
    # a forward pass right afterwards is deterministic (BatchNorm running
    # stats, dropout off) and matches `load_checkpoint` semantics.
    model.eval()

    final_metrics = None
    if val_loader is not None:
        final_metrics = evaluate(model, val_loader, device, amp=use_amp)

    # Finalize history before writing it so history.json carries the same
    # scalars as the returned dict.
    history["epochs_run"] = epochs_run
    history["restored_best"] = restored_best

    if run_dir is not None:
        if best_state is not None:
            _save_payload({"model_state_dict": best_state, "model_config": cfg.model,
                           "config": cfg}, os.path.join(run_dir, "best.pt"))
        _save_payload(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": start_epoch + epochs_run,
                "history": {k: v for k, v in history.items()},
                "model_config": cfg.model,
                "config": cfg,
            },
            os.path.join(run_dir, "last.pt"),
        )
        save_history(history, os.path.join(run_dir, "history.json"))
        if final_metrics is not None:
            save_metrics(final_metrics, os.path.join(run_dir, "final_metrics.json"))
        try:
            plot_training_curves(history, os.path.join(run_dir, "training_curves.png"))
        except Exception:
            pass  # curves are cosmetic; never fail training for a plot

    return history


# ---------------------------------------------------------------------------
# Artifacts & serialization
# ---------------------------------------------------------------------------

def _nan_to_none(obj):
    """Recursively replace NaN floats with None so JSON stays valid."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if math.isnan(val) else val
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _nan_to_none(obj.tolist())
    return obj


def _save_payload(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def save_checkpoint(
    model,
    path: str,
    cfg: Config = DEFAULT_CONFIG,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    history: dict | None = None,
    metrics: dict | None = None,
) -> None:
    """Serializes model (+ optional training state) for resume/best tracking."""
    # Prefer the model's own config: saving with a default cfg while the
    # model was built from a different one produces checkpoints that reload
    # into mismatched architectures.
    model_cfg = getattr(model, "cfg", cfg.model)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_cfg,
        "config": cfg,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if history is not None:
        payload["history"] = history
    if metrics is not None:
        payload["metrics"] = metrics
    _save_payload(payload, path)


def load_full_checkpoint(path: str) -> dict:
    """Loads the complete checkpoint payload (model/optimizer/history/...)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(path: str, cfg: Config = DEFAULT_CONFIG):
    """Loads a CAFN from a checkpoint file (v1-compatible behavior).

    The returned model is in eval mode so immediate inference matches the
    trained model's deterministic behavior.
    """
    checkpoint = load_full_checkpoint(path)
    model_cfg = checkpoint.get("model_config", cfg.model)
    from src.models.cafn import CAFN

    model = CAFN(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_history(history: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    serializable = {}
    for key, value in history.items():
        if isinstance(value, list):
            serializable[key] = _nan_to_none([
                float(v) if isinstance(v, (int, float, np.number)) else v for v in value
            ])
        else:
            serializable[key] = _nan_to_none(value)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def save_metrics(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_nan_to_none(metrics), f, indent=2, default=str)


def plot_training_curves(history: dict, path: str) -> None:
    """Loss + metric curves per epoch; cosmetic, best-effort."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    if history.get("val_macro_f1"):
        axes[1].plot(epochs, history["val_macro_f1"], label="val_macro_f1", color="tab:green")
        axes[1].set_ylim(0.0, 1.05)
    if history.get("train_accuracy"):
        axes[1].plot(epochs, history["train_accuracy"], label="train_acc", color="tab:orange")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].set_title("Accuracy / F1")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _replace_fields(dc, **kwargs):
    """dataclasses.replace, silently dropping kwargs the dataclass lacks."""
    from dataclasses import fields as dc_fields

    payload = {f.name: getattr(dc, f.name) for f in dc_fields(dc)}
    payload.update({k: v for k, v in kwargs.items() if k in payload})
    return type(dc)(**payload)


def build_config(args) -> Config:
    patience = DEFAULT_CONFIG.train.patience
    if args.patience is not None and args.patience >= 0:
        patience = args.patience
    model = _replace_fields(
        DEFAULT_CONFIG.model,
        fusion=args.fusion,
        pos_encoding=args.pos_encoding,
    )
    train = _replace_fields(
        DEFAULT_CONFIG.train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aux_loss_weight=args.aux_loss_weight,
        seed=args.seed,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_epochs=getattr(args, "warmup_epochs", None),
        patience=patience,
        amp=args.amp,
        num_workers=args.num_workers,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        balance_classes=args.balance_classes
        or getattr(args, "class_weights", None) == "auto",
    )
    return _replace_fields(DEFAULT_CONFIG, model=model, train=train)


def _verify_device_supports_model(device: torch.device, cfg: Config) -> torch.device:
    """Probe the accelerator with one forward pass; fall back to CPU on gaps.

    Some MPS builds lack ops the model needs (e.g. `adaptive_avg_pool3d`);
    better to discover that before epoch 1 than mid-training.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from src.models.cafn import CAFN

    try:
        probe = CAFN(cfg.model).to(device)
        probe.eval()
        with torch.no_grad():
            probe(
                torch.zeros(1, 1, *cfg.mri.target_shape, device=device),
                torch.zeros(1, 1, cfg.vep.length, device=device),
            )
        return device
    except (NotImplementedError, RuntimeError) as exc:
        if device.type == "cpu":
            raise
        print(f"warning: {device} cannot run this model ({exc}); falling back to CPU")
        return torch.device("cpu")


def _carve_splits(dataset, args, cfg: Config) -> dict:
    """Train/val(/test) splits honouring the CLI's stratify/group flags.

    ``--test-split`` carves the held-out test set first; the remaining
    flags then partition train/val exactly as before, so every pre-existing
    invocation keeps its split behaviour.
    """
    test_split = float(getattr(args, "test_split", 0.0) or 0.0)
    if test_split <= 0:
        train_ds, val_ds = train_val_split(
            dataset, cfg.train.val_split, cfg.train.seed,
            stratify=not args.no_stratify,
            group_by="subject_id" if args.group_by_subject else None,
        )
        return {"train": train_ds, "val": val_ds, "test": None}

    from src.data.splits import stratified_train_val_test_split

    if args.no_stratify:  # legacy-style uniform random 3-way carve
        n = len(dataset)
        n_test = max(1, int(n * test_split)) if n > 1 else 0
        g = torch.Generator().manual_seed(cfg.train.seed)
        perm = torch.randperm(n, generator=g).tolist()
        test_idx, rest = perm[:n_test], perm[n_test:]
        rest = Subset(dataset, rest)
        train_ds, val_ds = train_val_split(
            rest, cfg.train.val_split, cfg.train.seed,
            stratify=False, group_by=None,
        )
        return {"train": train_ds, "val": val_ds, "test": Subset(dataset, test_idx)}

    train_ds, val_ds, test_ds = stratified_train_val_test_split(
        dataset, cfg.train.val_split, test_split, cfg.train.seed
    )
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def _eval_loader(subset, cfg: Config, device: torch.device) -> DataLoader | None:
    if subset is None or len(subset) == 0:
        return None
    return DataLoader(subset, batch_size=cfg.train.batch_size, shuffle=False,
                      **_make_loader_kwargs(cfg.train, device))


def _final_detailed_report(model, splits: dict, cfg: Config, device: torch.device,
                           out_dir: str) -> None:
    """Confusion-matrix PNGs + console classification reports for val/test.

    Best-effort by design: reporting must never kill a finished run.
    """
    try:
        from src.evaluation import (
            evaluate_detailed,
            plot_confusion_matrix,
            print_classification_report,
        )

        os.makedirs(out_dir, exist_ok=True)
        model.to(device)
        for name in ("val", "test"):
            loader = _eval_loader(splits.get(name), cfg, device)
            if loader is None:
                continue
            metrics = evaluate_detailed(model, loader, device)
            print(f"\n== {name} (detailed) ==")
            print(print_classification_report(metrics))
            png = os.path.join(out_dir, f"final_{name}_confusion_matrix.png")
            plot_confusion_matrix(metrics["confusion_matrix"], png)
            save_metrics(metrics, os.path.join(out_dir, f"final_{name}_metrics.json"))
    except Exception as exc:  # noqa: BLE001 - reporting is non-critical
        print(f"detailed reporting skipped: {exc}")


def _run_eval_only(args, cfg: Config, splits: dict, device: torch.device) -> None:
    """--eval-only: load a checkpoint and report detailed metrics."""
    model = load_checkpoint(args.eval_only, cfg).to(device)
    out_dir = os.path.join(cfg.train.checkpoint_dir, getattr(args, "run_name", "run") or "run")
    print(f"loaded checkpoint {args.eval_only}")
    _final_detailed_report(model, splits, cfg, device, out_dir=out_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train CAFN on synthetic or manifest data")
    parser.add_argument("--manifest", type=str, default=None,
                        help="manifest CSV (subject_id,mri_path,vep_path,fs,label); omit for synthetic")
    parser.add_argument("--synthetic-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.train.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.train.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.train.lr)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.train.weight_decay)
    parser.add_argument("--aux-loss-weight", type=float, default=DEFAULT_CONFIG.train.aux_loss_weight)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--warmup-epochs", type=float,
                        default=DEFAULT_CONFIG.train.warmup_epochs,
                        help="linear LR warmup before cosine decay (scheduler=cosine)")
    parser.add_argument("--optimizer", choices=("adamw", "adam", "sgd"),
                        default=DEFAULT_CONFIG.train.optimizer,
                        help="AdamW default (decoupled weight decay)")
    parser.add_argument("--fusion", choices=("bidirectional", "cross"),
                        default=DEFAULT_CONFIG.model.fusion,
                        help="'cross' recovers the v1 one-way spatial->temporal fusion")
    parser.add_argument("--no-pos-encoding", dest="pos_encoding", action="store_false",
                        help="disable positional embeddings (v1 bag-of-features attention)")
    parser.add_argument("--patience", type=int, default=None,
                        help="early-stopping patience epochs; omit = config default (off)")
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_CONFIG.train.max_grad_norm)
    parser.add_argument("--amp", action="store_true", help="mixed precision on CUDA/MPS")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--class-weights", choices=("auto",), default=None,
                        help="inverse-frequency CE weights (alias of --balance-classes)")
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument("--group-by-subject", action="store_true",
                        help="split train/val by subject_id (prevents within-subject "
                             "leakage in longitudinal cohorts); falls back to a "
                             "stratified row split when no subject ids exist")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--augment", action="store_true", help="train-time augmentation")
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=DEFAULT_CONFIG.train.checkpoint_dir,
                        help="where per-run artifact directories are created")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test-split", type=float, default=0.0,
                        help="hold out an additional stratified test split "
                             "(reported with the detailed metrics + confusion matrix)")
    parser.add_argument("--eval-only", type=str, default=None, metavar="CKPT",
                        help="skip training; evaluate an existing checkpoint on the "
                             "val (and test, when --test-split > 0) split")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.train.seed)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    cfg = build_config(args)
    device = _verify_device_supports_model(device, cfg)
    seed_everything(args.seed, deterministic=args.deterministic)
    print(f"device: {device}")

    if args.manifest:
        from src.data.dataset import PairedNeuroDataset

        dataset = PairedNeuroDataset(args.manifest, cfg=cfg, cache=not args.no_cache)
        print(f"loaded manifest with {len(dataset)} subjects")
    else:
        from src.data.dataset import SyntheticNeuroDataset

        dataset = SyntheticNeuroDataset(num_samples=args.synthetic_samples, cfg=cfg, seed=args.seed)

    splits = _carve_splits(dataset, args, cfg)
    train_ds, val_ds = splits["train"], splits["val"]
    print(
        f"split: {len(train_ds)} train / {len(val_ds)} val"
        + (f" / {len(splits['test'])} test" if splits.get("test") else "")
    )

    from src.models.cafn import CAFN

    if args.eval_only:
        _run_eval_only(args=args, cfg=cfg, splits=splits, device=device)
        return

    model = CAFN(cfg.model)
    print(f"CAFN parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(
        model, train_ds, val_ds, cfg=cfg, device=device, verbose=True,
        augment=args.augment, resume=args.resume, run_name=args.run_name,
    )
    print(f"epochs run: {history['epochs_run']}  restored_best: {history['restored_best']}")
    if history["val_loss"]:
        print(f"final val loss: {history['val_loss'][-1]:.4f}")

    _final_detailed_report(model, splits, cfg, device,
                           out_dir=os.path.join(cfg.train.checkpoint_dir, args.run_name))
    print(f"artifacts: {os.path.join(cfg.train.checkpoint_dir, args.run_name)}")


if __name__ == "__main__":
    main()
