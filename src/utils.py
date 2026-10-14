"""Shared training utilities: reproducibility, device selection, label access."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, and torch (CPU + all accelerators) for reproducibility.

    With ``deterministic=True`` also requests deterministic cuDNN kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve a torch device string with sensible fallbacks.

    ``None`` or ``"auto"`` picks cuda -> mps -> cpu in that order. Explicit
    requests are honored as-is (falling back to CPU if the backend exists but
    is not available at runtime).
    """
    if requested is None or (isinstance(requested, str) and requested == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _enable_mps_fallback()
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(requested, torch.device):
        return requested
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count, optionally restricted to trainable tensors."""
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad or not trainable_only
    )


def extract_labels(dataset) -> list[int] | None:
    """Best-effort class-label extraction for stratified splitting.

    Understands :class:`~src.data.dataset.SyntheticNeuroDataset` (``.labels``),
    :class:`~src.data.dataset.PairedNeuroDataset` (``.rows`` manifest rows) and
    any object exposing a ``labels`` sequence. Returns ``None`` (unstratified)
    when labels cannot be recovered.
    """
    labels = getattr(dataset, "labels", None)
    if isinstance(labels, (list, tuple)) and len(labels) > 0:
        return [int(x) for x in labels]
    rows = getattr(dataset, "rows", None)
    if isinstance(rows, list) and rows and "label" in rows[0]:
        return [int(r["label"]) for r in rows]
    return None


# --- Compatibility / training-loop helpers -------------------------------

def get_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve a torch device with auto-detection (cuda -> mps -> cpu).

    Unlike :func:`auto_device`, an *explicit* request for an unavailable
    accelerator raises ``RuntimeError`` instead of silently downgrading to
    CPU -- silent fallbacks turn "it trained on GPU" into "it trained on
    CPU for 40x longer" without anyone noticing.
    """
    if isinstance(requested, str) and requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but not available: {requested}")
        if device.type == "mps":
            _enable_mps_fallback()
            if not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                raise RuntimeError(f"MPS requested but not available: {requested}")
        return device
    return auto_device(requested)


def _enable_mps_fallback() -> None:
    """Allow unsupported MPS ops to fall back to CPU (e.g. adaptive_avg_pool3d
    is not implemented for MPS in current PyTorch builds and would otherwise
    crash CAFN inference on Apple Silicon)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def collect_labels(dataset) -> list[int] | None:
    """Best-effort class-label extraction, index-correct for Subsets."""
    from torch.utils.data import Subset

    labels = None
    base = dataset
    indices = None
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = list(dataset.indices)

    if isinstance(base, dict):  # TensorDataset-style pair sequences
        return None
    raw = getattr(base, "labels", None)
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        labels = [int(x) for x in (raw if indices is None else [raw[i] for i in indices])]
        return labels
    rows = getattr(base, "rows", None)
    if isinstance(rows, list) and rows and "label" in rows[0]:
        rows = rows if indices is None else [rows[i] for i in indices]
        return [int(r["label"]) for r in rows]
    return None


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding: distinct, reproducible stream per worker."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def stratified_train_val_split(
    dataset, val_split: float, seed: int
) -> tuple[Subset, Subset]:
    """Class-balanced train/val split with a graceful random fallback.

    Stratification guarantees the validation set contains every class that
    exists in the data -- on a small cohort a plain random split can produce
    a val set missing an entire pathology class, making its accuracy/F1
    meaningless. Falls back to a plain random split when labels are not
    recoverable (e.g. bare TensorDatasets).
    """
    from torch.utils.data import Subset

    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot split an empty dataset")
    generator = torch.Generator().manual_seed(seed)

    def _subset(train_idx: list[int], val_idx: list[int]) -> tuple[Subset, Subset]:
        if not train_idx and val_idx:
            train_idx, val_idx = val_idx, []
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    if n == 1:
        return _subset([0], [])

    n_val_total = max(1, int(round(n * val_split)))
    n_val_total = min(n_val_total, n - 1)

    labels = extract_labels(dataset)
    if labels is None or len(set(labels)) < 2:
        perm = torch.randperm(n, generator=generator).tolist()
        return _subset(perm[n_val_total:], perm[:n_val_total])

    # Group indices per class, shuffle each, then allocate validation slots
    # proportionally (largest remainder method so rounding stays fair).
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)
    for class_indices in by_class.values():
        shuffled = torch.randperm(len(class_indices), generator=generator).tolist()
        class_indices[:] = [class_indices[i] for i in shuffled]

    raw = {c: len(idxs) * val_split for c, idxs in by_class.items()}
    allocations = {c: int(v) for c, v in raw.items()}
    remainder = n_val_total - sum(allocations.values())
    by_fraction = sorted(
        by_class, key=lambda c: raw[c] - allocations[c], reverse=True
    )
    for c in by_fraction[:max(0, remainder)]:
        allocations[c] += 1
    # Never take every sample of a class into val (keep >= 1 per class in
    # train) unless that guard would empty the val set entirely -- with one
    # sample per class, val must draw whole classes and train keeps the rest.
    clamped = {c: min(k, max(0, len(by_class[c]) - 1)) for c, k in allocations.items()}
    if any(clamped.values()):
        allocations = clamped

    train_idx: list[int] = []
    val_idx: list[int] = []
    c_order = sorted(by_class, key=lambda c: -len(by_class[c]))
    for c in c_order:
        k = allocations[c]
        val_idx.extend(by_class[c][:k])
        train_idx.extend(by_class[c][k:])
    return _subset(train_idx, val_idx)


def extract_groups(dataset, group_key: str = "subject_id") -> list[str] | None:
    """Best-effort per-sample group ids (e.g. subject ids), Subset-correct.

    Understands the same attribute surface as :func:`extract_labels`: a
    ``rows`` list of manifest dicts (``PairedNeuroDataset``) or a plain
    ``groups`` sequence attribute. Returns ``None`` when no grouping
    information exists.
    """
    from torch.utils.data import Subset

    base, indices = dataset, None
    if isinstance(dataset, Subset):
        base, indices = dataset.dataset, list(dataset.indices)

    raw = getattr(base, "groups", None)
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        values = [str(g) for g in (raw if indices is None else [raw[i] for i in indices])]
        return values
    rows = getattr(base, "rows", None)
    if isinstance(rows, list) and rows and group_key in rows[0]:
        rows = rows if indices is None else [rows[i] for i in indices]
        return [str(r[group_key]) for r in rows]
    items = getattr(base, "subject_id", None)  # synthetic-style attribute
    if isinstance(items, (list, tuple)) and len(items) > 0:
        return [str(g) for g in (items if indices is None else [items[i] for i in indices])]
    return None


def group_train_val_split(
    dataset,
    val_split: float,
    seed: int,
    group_key: str = "subject_id",
) -> tuple[Subset, Subset]:
    """Split by *group* (e.g. subject), never by row -- prevents leakage.

    Longitudinal cohorts often contain several scans per subject. A row-wise
    random split puts scans of the same subject in both train and val, so
    validation accuracy measures subject memorization, not generalization --
    a classic neuroimaging pitfall. Here whole groups are assigned to one
    side: shuffled group order, greedily filling the val budget (never all
    groups -- at least one group always stays in train).

    Falls back to :func:`stratified_train_val_split` when the dataset
    exposes no group information.
    """
    from torch.utils.data import Subset

    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot split an empty dataset")
    if n == 1:
        return Subset(dataset, [0]), Subset(dataset, [])

    groups = extract_groups(dataset, group_key=group_key)
    if groups is None or len(groups) != n:
        return stratified_train_val_split(dataset, val_split, seed)

    # Group sample indices by group id.
    by_group: dict[str, list[int]] = {}
    for idx, gid in enumerate(groups):
        by_group.setdefault(gid, []).append(idx)

    unique_groups = sorted(by_group)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique_groups), generator=generator).tolist()
    order = [unique_groups[i] for i in perm]

    # Greedily fill the val budget with whole groups.
    target = max(1, int(round(n * val_split)))
    val_groups: list[str] = []
    val_size = 0
    for gid in order:
        if val_size >= target:
            break
        val_groups.append(gid)
        val_size += len(by_group[gid])

    val_idx: list[int] = []
    for gid in val_groups:
        val_idx.extend(by_group[gid])
    train_idx: list[int] = []
    val_set = set(val_groups)
    for gid, indices in by_group.items():
        if gid not in val_set:
            train_idx.extend(indices)

    if not train_idx:  # single-group degenerate case: everything to train
        return Subset(dataset, groups and list(range(n))), Subset(dataset, [])
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
