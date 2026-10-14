"""PyTorch Dataset classes for paired MRI + VEP samples.

`PairedNeuroDataset` reads a manifest CSV (subject_id, mri_path, vep_path,
fs, label) and applies the real preprocessing pipeline -- this is what a
labeled clinical cohort would plug into. `SyntheticNeuroDataset` generates
class-conditioned synthetic samples in-memory through the same pipeline,
used for fast unit tests, integration tests, and development/training-loop
validation ahead of real labeled data.

v2 additions shared by both datasets:
- ``labels`` attribute: the per-sample label list, used by stratified
  train/val splitting and class-weight estimation.
- ``class_weights`` property: inverse-frequency weights (mean-normalized)
  ready to pass as cross-entropy ``weight``.
- Optional per-item caching: preprocessing (NIfTI load, resample, mask) and
  generation are expensive; with ``cache=True`` each item is computed once
  and reused across epochs. Augmentation is applied *after* the cache so
  cached items still vary per access.
- Optional train-time augmentation (``augment=True``), see
  ``src/data/augment.py``.
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import Config, DEFAULT_CONFIG
from src.data.augment import augment_mri, augment_vep
from src.data.synthetic import synthetic_mri_volume, synthetic_vep_signal
from src.preprocessing.mri_transforms import preprocess_mri_file, preprocess_mri_volume
from src.preprocessing.signal_cleaner import (
    clean_vep_signal,
    resample_to_length,
    zscore_normalize,
)


def compute_class_weights(
    labels: list[int] | np.ndarray, num_classes: int
) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.0.

    On balanced data this returns ones (a no-op for weighted cross-entropy).
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    if not present.any():
        return torch.ones(num_classes, dtype=torch.float32)
    # Only classes actually present contribute to the normalization so a
    # class missing from the split does not silently rescale the rest.
    # (Absent-class weights are irrelevant: no target can reference them.)
    weights = np.ones(num_classes, dtype=np.float64)
    weights[present] = 1.0 / counts[present]
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


class PairedNeuroDataset(Dataset):
    """Loads (MRI volume, VEP signal, label) triples from a manifest CSV.

    Expected columns: ``subject_id, mri_path, vep_path, label`` and
    optionally ``fs`` (falls back to ``cfg.vep.sample_rate``). Paths are used
    as-is when absolute or cwd-relative; otherwise they are resolved against
    the manifest's directory, so a manifest can be moved with its data
    folder. VEP files may be CSV (single numeric column, header allowed) or
    ``.npy``. The manifest is validated at construction so broken rows fail
    fast instead of one epoch into training.
    """

    REQUIRED_COLUMNS = ("subject_id", "mri_path", "vep_path", "label")

    def __init__(
        self,
        manifest_path: str,
        cfg: Config = DEFAULT_CONFIG,
        cache: bool = True,
        augment: bool = False,
        augment_rng: np.random.Generator | None = None,
        validate: bool = True,
    ):
        self.cfg = cfg
        self.manifest_path = manifest_path
        self.base_dir = os.path.dirname(os.path.abspath(manifest_path))
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            self.columns = list(reader.fieldnames or [])
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"Manifest at {manifest_path} has no rows")
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.columns]
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} is missing required column(s): {missing}. "
                f"Found columns: {self.columns}"
            )
        self.labels = []
        num_classes = cfg.model.num_classes
        for i, row in enumerate(self.rows):
            try:
                label = int(row["label"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Manifest row {i + 2}: label {row['label']!r} is not an integer"
                ) from None
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Manifest row {i + 2}: label {label} out of range "
                    f"[0, {num_classes}) for {num_classes} classes"
                )
            if "fs" in row and row["fs"]:
                try:
                    float(row["fs"])
                except ValueError:
                    raise ValueError(
                        f"Manifest row {i + 2}: fs {row['fs']!r} is not a number"
                    ) from None
            self.labels.append(label)
        self.rows = [self._resolve_paths(row, index=i) for i, row in enumerate(self.rows)]
        if validate:
            self._validate_files()
        self.cache = cache
        self._cache: dict[int, dict] = {}
        self.augment = augment
        self.augment_rng = augment_rng or np.random.default_rng(0)

    def _resolve_paths(self, row: dict, index: int) -> dict:
        """Absolute/cwd paths pass through; bare names resolve under the manifest dir."""

        def _resolve(key: str) -> None:
            value = row.get(key)
            if not value:
                raise ValueError(f"Manifest row {index + 2}: {key} is empty")
            if os.path.isabs(value) or os.path.exists(value):
                return
            candidate = os.path.join(self.base_dir, value)
            if os.path.exists(candidate):
                row[key] = candidate

        _resolve("mri_path")
        _resolve("vep_path")
        return row

    def _validate_files(self) -> None:
        missing = [
            (i + 2, row["mri_path"])
            for i, row in enumerate(self.rows)
            if not os.path.exists(row["mri_path"])
        ]
        missing += [
            (i + 2, row["vep_path"])
            for i, row in enumerate(self.rows)
            if not os.path.exists(row["vep_path"])
        ]
        if missing:
            detail = ", ".join(f"row {row}: {path}" for row, path in missing[:5])
            raise FileNotFoundError(
                f"Manifest references {len(missing)} missing file(s); first few: {detail}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, idx: int) -> dict:
        row = self.rows[idx]
        if str(row["mri_path"]).endswith(".npy"):
            # Already standardised (1, D, H, W) volume cached by the cohort
            # preprocessing step -- skip the NIfTI pipeline.
            mri = np.load(row["mri_path"], allow_pickle=False).astype(np.float32)
        else:
            mri = preprocess_mri_file(row["mri_path"], self.cfg.mri)

        if str(row["vep_path"]).endswith(".npy"):
            raw_signal = np.load(row["vep_path"], allow_pickle=False).astype(np.float64)
            if raw_signal.ndim == 2 and 1 in raw_signal.shape:
                raw_signal = raw_signal.reshape(-1)
        else:
            raw_signal = np.loadtxt(row["vep_path"], delimiter=",", skiprows=1)
        fs = float(row.get("fs") or self.cfg.vep.sample_rate)
        if self.cfg.vep.signal_mode == "spectrum":
            vep = zscore_normalize(
                resample_to_length(raw_signal, self.cfg.vep.length)
            ).astype(np.float32)
        else:
            vep = clean_vep_signal(raw_signal, fs, self.cfg.vep)

        return {
            "mri": torch.from_numpy(mri),
            "vep": torch.from_numpy(vep).unsqueeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "subject_id": row["subject_id"],
        }

    def __getitem__(self, idx: int) -> dict:
        if self.cache and idx in self._cache:
            item = self._cache[idx]
        else:
            item = self._load(idx)
            if self.cache:
                self._cache[idx] = item

        if self.augment:
            item = dict(item)  # never mutate the cached tensors
            item["mri"] = augment_mri(item["mri"], self.augment_rng)
            item["vep"] = augment_vep(item["vep"], self.augment_rng)
        return item

    @property
    def class_weights(self) -> torch.Tensor:
        return compute_class_weights(self.labels, self.cfg.model.num_classes)

    @property
    def cohorts(self) -> list[str]:
        """Per-row cohort/site id from the optional ``dataset`` manifest column."""
        return [row.get("dataset") or "unknown" for row in self.rows]


class AugmentedView(Dataset):
    """Augmentation wrapper over a fixed set of indices of a base dataset.

    The training pipeline wraps only the *train* indices with this view so
    augmentation never leaks into validation, while both sides share the
    same underlying (possibly cached) base dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int] | range,
        mri_augment: bool = True,
        vep_augment: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.mri_augment = mri_augment
        self.vep_augment = vep_augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream for the new epoch.

        Keeps augmentations different every epoch while remaining fully
        reproducible (epoch 3 of run A == epoch 3 of run B, same seed).
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        idx = self.indices[position]
        item = self.dataset[idx]
        if not (self.mri_augment or self.vep_augment):
            return item
        rng = np.random.default_rng([self.seed, self.epoch, idx, 0x0CCA])
        item = dict(item)
        if self.mri_augment:
            item["mri"] = torch.from_numpy(augment_mri(item["mri"].numpy(), rng))
        if self.vep_augment:
            item["vep"] = torch.from_numpy(
                augment_vep(item["vep"][0].numpy(), rng)
            ).unsqueeze(0)
        return item


class SyntheticNeuroDataset(Dataset):
    """In-memory class-conditioned synthetic dataset, real preprocessing applied."""

    def __init__(
        self,
        num_samples: int,
        cfg: Config = DEFAULT_CONFIG,
        seed: int = 0,
        raw_mri_shape: tuple[int, int, int] = (80, 96, 64),
        cache: bool = True,
        augment: bool = False,
        augment_rng: np.random.Generator | None = None,
        mri_strength: float = 1.0,
        vep_strength: float = 1.0,
    ):
        """``mri_strength`` / ``vep_strength`` in [0, 1] scale how informative each
        modality is about the class (1 = default generator), enabling controlled
        studies of when fusion helps."""
        self.cfg = cfg
        self.mri_strength = float(mri_strength)
        self.vep_strength = float(vep_strength)
        self.num_samples = num_samples
        self.raw_mri_shape = raw_mri_shape
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cache = cache
        self._cache: dict[int, dict] = {}
        self.augment = augment
        self.augment_rng = augment_rng or np.random.default_rng(seed + 1)
        num_classes = cfg.model.num_classes
        # Balanced labels, deterministically shuffled.
        base_labels = [i % num_classes for i in range(num_samples)]
        self.rng.shuffle(base_labels)
        self.labels = base_labels

    def __len__(self) -> int:
        return self.num_samples

    def _generate(self, idx: int) -> dict:
        label = self.labels[idx]
        # Per-index RNG stream: sample (seed, idx) is fully deterministic no
        # matter *when* or *how often* it is drawn. A shared advancing RNG
        # made content depend on access order, which silently breaks under
        # num_workers > 0 (workers draw in different orders and each caches
        # different content for the same index).
        item_rng = np.random.default_rng([self.seed, idx])
        raw_volume = synthetic_mri_volume(
            label, shape=self.raw_mri_shape, rng=item_rng, strength=self.mri_strength
        )
        mri = preprocess_mri_volume(raw_volume, zooms=(1.0, 1.0, 1.0), cfg=self.cfg.mri)

        raw_signal = synthetic_vep_signal(
            label, fs=self.cfg.vep.sample_rate, rng=item_rng, strength=self.vep_strength
        )
        vep = clean_vep_signal(raw_signal, self.cfg.vep.sample_rate, self.cfg.vep)

        return {
            "mri": torch.from_numpy(mri),
            "vep": torch.from_numpy(vep).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": f"synthetic-{idx:04d}",
        }

    def __getitem__(self, idx: int) -> dict:
        if self.cache and idx in self._cache:
            item = self._cache[idx]
        else:
            item = self._generate(idx)
            if self.cache:
                self._cache[idx] = item

        if self.augment:
            item = dict(item)  # never mutate the cached tensors
            item["mri"] = augment_mri(item["mri"], self.augment_rng)
            item["vep"] = augment_vep(item["vep"], self.augment_rng)
        return item

    @property
    def class_weights(self) -> torch.Tensor:
        return compute_class_weights(self.labels, self.cfg.model.num_classes)
