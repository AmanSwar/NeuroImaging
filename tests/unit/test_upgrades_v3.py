"""Tests for the v3 hardening pass: subject-aware (group) splitting,
per-index deterministic synthetic generation, manifest validation, and the
Otsu/zscore MRI preprocessing paths.
"""

import csv
import os

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from src.config import MRIConfig
from src.data.dataset import PairedNeuroDataset, SyntheticNeuroDataset
from src.data.synthetic import write_synthetic_dataset
from src.preprocessing.mri_transforms import (
    foreground_mask_otsu,
    otsu_threshold,
    preprocess_mri_volume,
)
from src.train import train_val_split
from src.utils import collect_labels, extract_groups, group_train_val_split


class _FakeCohort:
    """Minimal dataset stand-in with manifest-style rows (multi-scan subjects)."""

    def __init__(self, rows):
        self.rows = rows
        self.labels = [int(r["label"]) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return {"label": self.labels[idx], "subject_id": self.rows[idx]["subject_id"]}


def _multi_scan_rows():
    """3 subjects x 2 scans + 2 singletons, deliberately imbalanced sizes."""
    rows = []
    for subject, label, scans in (("s1", 0, 2), ("s2", 1, 2), ("s3", 2, 2), ("s4", 0, 1), ("s5", 1, 1)):
        for scan in range(scans):
            rows.append(
                {"subject_id": subject, "mri_path": "x.nii.gz", "vep_path": "x.csv",
                 "fs": 250.0, "label": label, "scan": scan}
            )
    return rows


class TestGroupSplit:
    def test_no_subject_in_both_sides(self):
        ds = _FakeCohort(_multi_scan_rows())
        train_ds, val_ds = group_train_val_split(ds, val_split=0.4, seed=0)
        train_subjects = {ds.rows[i]["subject_id"] for i in train_ds.indices}
        val_subjects = {ds.rows[i]["subject_id"] for i in val_ds.indices}
        assert train_subjects.isdisjoint(val_subjects), "same subject on both sides = leakage"

    def test_whole_groups_assigned(self):
        ds = _FakeCohort(_multi_scan_rows())
        for seed in range(5):  # across seeds, never split a subject's scans
            train_ds, val_ds = group_train_val_split(ds, val_split=0.4, seed=seed)
            train_idx, val_idx = set(train_ds.indices), set(val_ds.indices)
            assert train_idx | val_idx == set(range(len(ds)))
            assert not (train_idx & val_idx)

    def test_val_fraction_roughly_honored(self):
        ds = _FakeCohort(_multi_scan_rows())
        _, val_ds = group_train_val_split(ds, val_split=0.4, seed=0)
        val_frac = len(val_ds) / len(ds)
        # Whole-group granularity (groups of 2) limits precision.
        assert 0.2 <= val_frac <= 0.8

    def test_deterministic_for_same_seed(self):
        ds = _FakeCohort(_multi_scan_rows())
        a = group_train_val_split(ds, 0.4, seed=3)
        b = group_train_val_split(ds, 0.4, seed=3)
        assert list(a[0].indices) == list(b[0].indices)
        assert list(a[1].indices) == list(b[1].indices)

    def test_falls_back_to_stratified_without_groups(self):
        ds = torch.utils.data.TensorDataset(torch.arange(10))
        train_ds, val_ds = group_train_val_split(ds, val_split=0.3, seed=0)
        assert len(train_ds) + len(val_ds) == 10
        assert len(val_ds) == 3  # random/stratified behavior, not group-based

    def test_train_val_split_group_by_passthrough(self):
        ds = _FakeCohort(_multi_scan_rows())
        train_ds, val_ds = train_val_split(ds, 0.4, seed=0, group_by="subject_id")
        train_subjects = {ds.rows[i]["subject_id"] for i in train_ds.indices}
        val_subjects = {ds.rows[i]["subject_id"] for i in val_ds.indices}
        assert train_subjects.isdisjoint(val_subjects)

    def test_extract_groups_subset_correct(self):
        ds = _FakeCohort(_multi_scan_rows())
        subset = Subset(ds, [7, 3, 1])
        expected = [ds.rows[i]["subject_id"] for i in (7, 3, 1)]
        assert extract_groups(subset) == expected

    def test_split_on_real_paired_dataset(self, tmp_path, tiny_config):
        manifest = write_synthetic_dataset(str(tmp_path / "raw"), samples_per_class=2,
                                           cfg=tiny_config, seed=0)
        ds = PairedNeuroDataset(manifest, cfg=tiny_config)
        # Collapse pairs of rows onto shared subject ids to emulate multi-scan subjects.
        for row, idx in zip(ds.rows, range(len(ds.rows))):
            row["subject_id"] = f"subject-{idx // 2}"
        train_ds, val_ds = train_val_split(ds, 0.4, seed=0, group_by="subject_id")
        train_subjects = {ds.rows[i]["subject_id"] for i in train_ds.indices}
        val_subjects = {ds.rows[i]["subject_id"] for i in val_ds.indices}
        assert train_subjects.isdisjoint(val_subjects)


class TestSyntheticPerIndexDeterminism:
    def test_same_index_same_content_without_cache(self, tiny_config):
        """cache=False forces regeneration; per-index RNG must return the
        identical sample regardless of access history (worker safety)."""
        ds = SyntheticNeuroDataset(6, cfg=tiny_config, seed=1,
                                   raw_mri_shape=(40, 48, 32), cache=False)
        first = ds[0]
        ds[3]
        ds[1]
        again = ds[0]
        assert torch.equal(first["mri"], again["mri"])
        assert torch.equal(first["vep"], again["vep"])

    def test_two_instances_same_seed_agree(self, tiny_config):
        a = SyntheticNeuroDataset(5, cfg=tiny_config, seed=7,
                                  raw_mri_shape=(40, 48, 32), cache=False)
        b = SyntheticNeuroDataset(5, cfg=tiny_config, seed=7,
                                  raw_mri_shape=(40, 48, 32), cache=False)
        a0, b0 = a[0], b[0]
        assert torch.equal(a0["mri"], b0["mri"])
        assert torch.equal(a0["vep"], b0["vep"])

    def test_different_seeds_differ(self, tiny_config):
        a = SyntheticNeuroDataset(4, cfg=tiny_config, seed=1,
                                  raw_mri_shape=(40, 48, 32), cache=False)
        b = SyntheticNeuroDataset(4, cfg=tiny_config, seed=2,
                                  raw_mri_shape=(40, 48, 32), cache=False)
        assert not torch.equal(a[1]["mri"], b[1]["mri"])


class TestManifestValidation:
    def _write_manifest(self, path, rows, fieldnames=("subject_id", "mri_path", "vep_path", "fs", "label")):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    def test_missing_column_raises(self, tmp_path, tiny_config):
        manifest = self._write_manifest(tmp_path / "manifest.csv", [
            {"subject_id": "s0", "mri_path": "a.nii.gz", "vep_path": "a.csv"},
        ], fieldnames=("subject_id", "mri_path", "vep_path"))
        with pytest.raises(ValueError, match="missing required column"):
            PairedNeuroDataset(manifest, cfg=tiny_config)

    def test_bad_label_raises(self, tmp_path, tiny_config):
        manifest = self._write_manifest(tmp_path / "manifest.csv", [
            {"subject_id": "s0", "mri_path": "a.nii.gz", "vep_path": "a.csv",
             "fs": 250.0, "label": "9"},
        ])
        with pytest.raises(ValueError, match="out of range"):
            PairedNeuroDataset(manifest, cfg=tiny_config)

    def test_non_integer_label_raises(self, tmp_path, tiny_config):
        manifest = self._write_manifest(tmp_path / "manifest.csv", [
            {"subject_id": "s0", "mri_path": "a.nii.gz", "vep_path": "a.csv",
             "fs": 250.0, "label": "glaucoma"},
        ])
        with pytest.raises(ValueError, match="not an integer"):
            PairedNeuroDataset(manifest, cfg=tiny_config)

    def test_missing_file_raises_fast(self, tmp_path, tiny_config):
        manifest = self._write_manifest(tmp_path / "manifest.csv", [
            {"subject_id": "s0", "mri_path": "nope.nii.gz", "vep_path": "nope.csv",
             "fs": 250.0, "label": "0"},
        ])
        with pytest.raises(FileNotFoundError, match="missing file"):
            PairedNeuroDataset(manifest, cfg=tiny_config)

    def test_relative_paths_resolve_against_manifest_dir(self, tmp_path, tiny_config):
        manifest = write_synthetic_dataset(str(tmp_path / "raw"), samples_per_class=1,
                                           cfg=tiny_config, seed=0)
        # Rewrite with paths relative to the *manifest's* directory.
        with open(manifest, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["mri_path"] = os.path.relpath(row["mri_path"], tmp_path / "raw")
            row["vep_path"] = os.path.relpath(row["vep_path"], tmp_path / "raw")
        rel_manifest = self._write_manifest(tmp_path / "raw" / "relative_manifest.csv", rows)
        # cwd (the repo root) does not contain mri/sub-0000_mri.nii.gz, so the
        # manifest-directory fallback must kick in.
        ds = PairedNeuroDataset(rel_manifest, cfg=tiny_config)
        assert ds[0]["label"].item() in (0, 1, 2)

    def test_npy_vep_supported(self, tmp_path, tiny_config):
        manifest = write_synthetic_dataset(str(tmp_path / "raw"), samples_per_class=1,
                                           cfg=tiny_config, seed=0)
        row = list(csv.DictReader(open(manifest)))[0]  # noqa: SIM115
        npy_path = tmp_path / "raw" / "s0.npy"
        np.save(npy_path, np.loadtxt(row["vep_path"], delimiter=",", skiprows=1))
        manifest2 = self._write_manifest(tmp_path / "raw" / "npy_manifest.csv", [{
            "subject_id": "s0", "mri_path": row["mri_path"], "vep_path": str(npy_path),
            "fs": row["fs"], "label": row["label"],
        }])
        ds = PairedNeuroDataset(manifest2, cfg=tiny_config)
        item = ds[0]
        assert item["vep"].shape == (1, tiny_config.vep.length)
        assert torch.isfinite(item["vep"]).all()


class TestOtsuAndZscorePaths:
    def test_otsu_threshold_separates_bimodal(self):
        volume = np.zeros((16, 16, 16), dtype=np.float32)
        volume[8:, :, :] = 10.0
        volume[4:8, :, :] = 200.0  # two clean modes, then air
        threshold = otsu_threshold(volume)
        assert 10.0 < threshold < 200.0

    def test_otsu_mask_removes_background_keeps_brain(self):
        volume = np.zeros((16, 16, 16), dtype=np.float32)
        volume[4:12, 4:12, 4:12] = 120.0  # a solid "brain"
        masked, mask = foreground_mask_otsu(volume, morphology=True)
        assert mask[8, 8, 8], "brain interior must survive"
        assert not mask[0, 0, 0], "air must be removed"
        np.testing.assert_array_equal(masked[~mask], 0.0)

    def test_degenerate_volume_untouched(self):
        flat = np.full((8, 8, 8), 42.0, dtype=np.float32)
        masked, mask = foreground_mask_otsu(flat)
        assert mask.all(), "single-level volume must not be destroyed"
        np.testing.assert_array_equal(masked, flat)

    def test_preprocess_zscore_normalization(self, tiny_config):
        cfg = MRIConfig(
            target_shape=(16, 16, 16),
            mask_method="otsu",
            normalization="zscore",
        )
        rng = np.random.default_rng(0)
        volume = rng.normal(loc=5.0, scale=1.0, size=(24, 24, 24)).astype(np.float32) + 100.0
        out = preprocess_mri_volume(volume, zooms=(1.0, 1.0, 1.0), cfg=cfg)
        assert out.shape == (1, 16, 16, 16)
        foreground = out[0][out[0] != 0]
        assert foreground.size > 0
        assert abs(float(foreground.mean())) < 0.5  # roughly zero-mean
        assert 0.2 < float(foreground.std()) < 5.0   # roughly unit variance

    def test_invalid_mask_method_raises(self):
        cfg = MRIConfig(target_shape=(8, 8, 8), mask_method="magic")
        with pytest.raises(ValueError, match="mask_method"):
            preprocess_mri_volume(np.ones((8, 8, 8), dtype=np.float32), (1, 1, 1), cfg)

    def test_invalid_normalization_raises(self):
        cfg = MRIConfig(target_shape=(8, 8, 8), normalization="log")
        with pytest.raises(ValueError, match="normalization"):
            preprocess_mri_volume(np.ones((8, 8, 8), dtype=np.float32), (1, 1, 1), cfg)


class TestCollectLabelsRegression:
    def test_collect_labels_still_works_after_utils_additions(self, tiny_config):
        ds = SyntheticNeuroDataset(9, cfg=tiny_config, seed=3,
                                   raw_mri_shape=(40, 48, 32))
        assert collect_labels(ds) == ds.labels


class TestCLISmoke:
    def test_main_synthetic_run_writes_artifacts(self, tmp_path):
        from src.train import main as train_main

        train_main([
            "--synthetic-samples", "12",
            "--epochs", "1",
            "--batch-size", "4",
            "--run-name", "smoke",
            "--checkpoint-dir", str(tmp_path),
        ])
        run_dir = tmp_path / "smoke"
        for name in ("best.pt", "last.pt", "history.json", "training_curves.png"):
            assert (run_dir / name).exists(), f"missing artifact {name}"

        history = __import__("json").loads((run_dir / "history.json").read_text())
        assert len(history["train_loss"]) == 1
        assert history["epochs_run"] == 1

    def test_main_with_group_by_subject(self, tmp_path):
        """The leakage-safe split flag must run end-to-end on synthetic data
        (rows carry unique synthetic-XXXX ids, so the group split degenerates
        to one subject per row -- still a valid whole-group assignation)."""
        from src.train import main as train_main

        train_main([
            "--synthetic-samples", "6",
            "--epochs", "1",
            "--batch-size", "3",
            "--group-by-subject",
            "--run-name", "grouped",
            "--checkpoint-dir", str(tmp_path),
        ])
        assert (tmp_path / "grouped" / "last.pt").exists()
