"""Deterministic dataset splitting utilities.

``src.train.train_val_split`` does a plain random partition, which can put
very few (or zero) members of a minority class into validation when the
cohort is small or imbalanced. This module adds label-stratified splits:

- proportion of each class is preserved across train/val(/test),
- subject-level grouping is respected when a dataset exposes subject ids
  (all scans of one subject land in the same split -- no leakage),
- everything is seeded and therefore reproducible.

Datasets are probed for labels/groups via well-known attribute names
(``labels``, ``targets``, ``rows``) so no new interface is required.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.config import DEFAULT_CONFIG, Config


def _safe_len(obj) -> int | None:
    """len() that returns None instead of raising on unsized objects."""
    try:
        return len(obj)
    except TypeError:
        return None


def extract_labels(dataset: Dataset) -> list[int] | None:
    """Best-effort label extraction for stratification; None when unknown.

    Tolerates lightweight objects that expose labels but no ``__len__`` by
    trusting the labels' own length, and looks through ``Subset`` wrappers
    (sub-selected in index order).
    """
    if isinstance(dataset, Subset):
        inner = extract_labels(dataset.dataset)
        if inner is None:
            return None
        try:
            return [inner[i] for i in dataset.indices]
        except IndexError:
            return None
    n = _safe_len(dataset)
    labels = getattr(dataset, "labels", None)
    if n is None and isinstance(labels, (list, np.ndarray)):
        n = _safe_len(labels)
    if n is None:
        return None
    if isinstance(labels, (list, np.ndarray)) and _safe_len(labels) == n:
        return [int(x) for x in labels]
    targets = getattr(dataset, "targets", None)
    if isinstance(targets, (list, np.ndarray)) and _safe_len(targets) == n:
        return [int(x) for x in targets]
    rows = getattr(dataset, "rows", None)
    if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "label" in rows[0]:
        return [int(r["label"]) for r in rows]
    # torchvision-style TensorDataset(tensors...) has `tensors`
    tensors = getattr(dataset, "tensors", None)
    if isinstance(tensors, (list, tuple)) and len(tensors) >= 2:
        last = tensors[-1]
        try:
            if hasattr(last, "shape") and _safe_len(last) == n:
                return [int(x) for x in last.tolist()]
        except TypeError:
            pass
    return None


def extract_groups(dataset: Dataset) -> list[str] | None:
    """Best-effort subject-id extraction for leakage-free splits."""
    if isinstance(dataset, Subset):
        inner = extract_groups(dataset.dataset)
        if inner is None:
            return None
        try:
            return [inner[i] for i in dataset.indices]
        except IndexError:
            return None
    n = _safe_len(dataset)
    if n is None:
        return None
    subject_ids = getattr(dataset, "subject_ids", None)
    if isinstance(subject_ids, (list, np.ndarray)) and _safe_len(subject_ids) == n:
        return [str(s) for s in subject_ids]
    rows = getattr(dataset, "rows", None)
    if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "subject_id" in rows[0]:
        return [str(r["subject_id"]) for r in rows]
    return None


def _seeds_to_indices(
    n: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    labels: list[int] | None,
    groups: list[str] | None,
) -> dict[str, list[int]]:
    """Compute index lists for each split (shared by the public helpers)."""
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")
    rng = np.random.default_rng(seed)

    # Group the dataset into indivisible units (subject ids when available).
    if groups is not None and len(groups) == n:
        unit_indices: dict[str, list[int]] = {}
        for i, g in zip(range(n), groups):
            unit_indices.setdefault(g, []).append(i)
        units = list(unit_indices.keys())
    else:
        group_list = ["__unit_%d__" % i for i in range(n)]
        unit_indices = {g: [i] for i, g in zip(range(n), group_list)}
        units = group_list

    # Per-unit label for stratification: majority label, falling back to
    # the unit's first label (relevant for grouped multi-scan units).
    if labels is not None and len(labels) == n:
        def unit_label(unit: str) -> int:
            vals = [labels[i] for i in unit_indices[unit]]
            return int(max(set(vals), key=vals.count))
    else:
        def unit_label(unit: str) -> int:
            return 0

    n_val = int(round(len(units) * val_fraction))
    n_test = int(round(len(units) * test_fraction))
    # Always keep at least one unit for training on a non-empty dataset.
    while len(units) - n_val - n_test < 1 and units and (n_val > 0 or n_test > 0):
        if n_val >= n_test:
            n_val -= 1
        else:
            n_test -= 1

    def take_stratified(pool: list[str], take: int) -> list[str]:
        """Pick ``take`` units from ``pool`` preserving class proportions.

        Uses the largest-remainder method so fractional per-class quotas
        (e.g. a 2-sample class at 25% of 20 units) are not lost to rounding,
        with ties on the fractional remainder broken toward rarer classes --
        minorities are exactly the groups a naive round() drops.
        """
        if take <= 0 or not pool:
            return []
        by_class: dict[int, list[str]] = {}
        for u in pool:
            by_class.setdefault(unit_label(u), []).append(u)
        n_pool = len(pool)

        quotas: dict[int, int] = {}
        remainders: list[tuple[float, int, int]] = []
        allocated = 0
        for cls, cls_units in by_class.items():
            exact = take * len(cls_units) / n_pool
            base = int(np.floor(exact + 1e-12))
            quotas[cls] = base
            allocated += base
            remainders.append((exact - base, len(cls_units), cls))

        leftover = take - allocated
        remainders.sort(key=lambda t: (-t[0], t[1]))  # big remainder first, rare class first
        for _, _, cls in remainders:
            if leftover <= 0:
                break
            if quotas[cls] < len(by_class[cls]):
                quotas[cls] += 1
                leftover -= 1

        chosen: list[str] = []
        for cls, cls_units in by_class.items():
            order = rng.permutation(len(cls_units))
            chosen.extend(cls_units[i] for i in order[: quotas[cls]])
        # Safety net: exact-count adjustment via random top-up/trim.
        chosen_set = set(chosen)
        remaining = [u for u in pool if u not in chosen_set]
        if remaining:
            order = rng.permutation(len(remaining))
            while len(chosen) > take and chosen:
                chosen.pop()
            for i in order:
                if len(chosen) >= take:
                    break
                chosen.append(remaining[i])
        return chosen

    order = rng.permutation(len(units))
    shuffled = [units[i] for i in order]
    test_units = take_stratified(shuffled, n_test) if n_test else []
    remaining = [u for u in shuffled if u not in set(test_units)]
    val_units = take_stratified(remaining, n_val) if n_val else []
    train_units = [u for u in remaining if u not in set(val_units)]
    if not train_units and shuffled:
        # Degenerate request (e.g. val_split=1.0 rounded up): all train.
        train_units, val_units, test_units = shuffled, [], []

    split_map = {"train": train_units, "val": val_units, "test": test_units}
    indices: dict[str, list[int]] = {}
    for name, us in split_map.items():
        idx: list[int] = []
        for u in us:
            idx.extend(unit_indices[u])
        indices[name] = sorted(idx)
    return indices


def _subset(check, indices: list[int]) -> Subset:
    return Subset(check, indices)


def stratified_train_val_split(
    dataset: Dataset,
    val_split: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Label-stratified two-way split; falls back to a random split when
    the dataset exposes no usable labels or when stratification is
    impossible for the requested sizes."""
    n = len(dataset)
    if n <= 1:
        return Subset(dataset, list(range(n))), Subset(dataset, [])
    labels = extract_labels(dataset)
    groups = extract_groups(dataset)
    try:
        idx = _seeds_to_indices(n, val_split, 0.0, seed, labels, groups)
    except ValueError:
        idx = _seeds_to_indices(n, val_split, 0.0, seed, None, None)
    train_idx, val_idx = idx["train"], idx["val"]
    if not train_idx:  # tiny edge case: keep old all-train fallback semantics
        train_idx, val_idx = sorted(range(n)), []
    return _subset(dataset, train_idx), _subset(dataset, val_idx)


def stratified_train_val_test_split(
    dataset: Dataset,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    """Label-stratified three-way train/val/test split."""
    n = len(dataset)
    if n <= 1:
        return (Subset(dataset, list(range(n))), Subset(dataset, []), Subset(dataset, []))
    labels = extract_labels(dataset)
    groups = extract_groups(dataset)
    try:
        idx = _seeds_to_indices(n, val_split, test_split, seed, labels, groups)
    except ValueError:
        idx = _seeds_to_indices(n, val_split, test_split, seed, None, None)
    return (
        _subset(dataset, idx["train"]),
        _subset(dataset, idx["val"]),
        _subset(dataset, idx["test"]),
    )


def split_summary(split: dict[str, Subset], labels: list[int] | None = None) -> str:
    """One-line human summary of split sizes and per-class counts."""
    parts = []
    for name in ("train", "val", "test"):
        sub = split.get(name)
        if sub is None:
            continue
        desc = f"{name}={len(sub)}"
        if labels is not None:
            counts = Counter(labels[i] for i in sub.indices)
            desc += " (" + ", ".join(f"c{k}:{v}" for k, v in sorted(counts.items())) + ")"
        parts.append(desc)
    return "  ".join(parts)


def make_splits(
    dataset: Dataset,
    cfg: Config = DEFAULT_CONFIG,
    test_split: float = 0.0,
    seed: int | None = None,
) -> dict[str, Subset]:
    """Convenience: build a dict of splits from the config's val_split."""
    seed = cfg.train.seed if seed is None else seed
    if test_split > 0:
        tr, va, te = stratified_train_val_test_split(dataset, cfg.train.val_split, test_split, seed)
        return {"train": tr, "val": va, "test": te}
    tr, va = stratified_train_val_split(dataset, cfg.train.val_split, seed)
    return {"train": tr, "val": va}
