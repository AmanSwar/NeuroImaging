"""Tests for shared training utilities: seeding, device selection, label
extraction, and the stratified split that keeps small cohorts measurable."""

import numpy as np
import pytest
import torch

from src.config import Config
from src.data.dataset import SyntheticNeuroDataset
from src.utils import (
    collect_labels,
    get_device,
    seed_everything,
    stratified_train_val_split,
)


class TestSeedEverything:
    def test_same_seed_same_randomness(self):
        seed_everything(123)
        a_t = torch.randn(4)
        a_n = np.random.randn(4)

        seed_everything(123)
        b_t = torch.randn(4)
        b_n = np.random.randn(4)
        assert torch.equal(a_t, b_t)
        assert np.allclose(a_n, b_n)

    def test_does_not_crash_without_cuda(self):
        seed_everything(0)  # cuda branches must be guarded internally


class TestGetDevice:
    def test_returns_torch_device(self):
        assert isinstance(get_device(), torch.device)

    def test_explicit_cpu(self):
        assert get_device("cpu").type == "cpu"

    def test_invalid_cuda_request_raises(self):
        # On a machine without CUDA this must raise; with CUDA, skip.
        if torch.cuda.is_available():
            pytest.skip("CUDA present; cannot test the failure path")
        with pytest.raises(RuntimeError):
            get_device("cuda")


class TestCollectLabels:
    def test_synthetic_dataset_labels(self, tiny_config):
        ds = SyntheticNeuroDataset(9, cfg=tiny_config, seed=3, raw_mri_shape=(40, 48, 32))
        assert collect_labels(ds) == ds.labels

    def test_subset_labels_follow_indices(self, tiny_config):
        ds = SyntheticNeuroDataset(10, cfg=tiny_config, seed=1, raw_mri_shape=(40, 48, 32))
        from torch.utils.data import Subset

        subset = Subset(ds, [3, 1, 7])
        expected = [ds.labels[i] for i in (3, 1, 7)]
        assert collect_labels(subset) == expected

    def test_paired_rows_labels(self, tmp_path, tiny_config):
        from src.data.synthetic import write_synthetic_dataset

        manifest = write_synthetic_dataset(str(tmp_path / "raw"), samples_per_class=2,
                                           cfg=tiny_config, seed=0)
        from src.data.dataset import PairedNeuroDataset

        ds = PairedNeuroDataset(manifest, cfg=tiny_config)
        assert collect_labels(ds) == [int(r["label"]) for r in ds.rows]


class TestStratifiedSplit:
    def test_classes_present_proportionally_in_both_sides(self, tiny_config):
        ds = SyntheticNeuroDataset(30, cfg=tiny_config, seed=5, raw_mri_shape=(40, 48, 32))
        train_ds, val_ds = stratified_train_val_split(ds, val_split=0.3, seed=0)
        from src.utils import collect_labels

        train_labels = collect_labels(train_ds)
        val_labels = collect_labels(val_ds)
        assert len(train_labels) + len(val_labels) == 30
        train_counts = [train_labels.count(c) for c in range(3)]
        val_counts = [val_labels.count(c) for c in range(3)]
        # Every class represented on both sides.
        assert all(c > 0 for c in train_counts)
        assert all(c > 0 for c in val_counts)
        # Proportions roughly match the requested split.
        val_frac = len(val_labels) / 30
        assert 0.15 < val_frac < 0.45

    def test_deterministic_for_same_seed(self, tiny_config):
        ds = SyntheticNeuroDataset(24, cfg=tiny_config, seed=2, raw_mri_shape=(40, 48, 32))
        tr_a, va_a = stratified_train_val_split(ds, 0.25, seed=42)
        tr_b, va_b = stratified_train_val_split(ds, 0.25, seed=42)
        assert list(tr_a.indices) == list(tr_b.indices)
        assert list(va_a.indices) == list(va_b.indices)

    def test_disjoint_and_covering(self, tiny_config):
        ds = SyntheticNeuroDataset(20, cfg=tiny_config, seed=4, raw_mri_shape=(40, 48, 32))
        tr, va = stratified_train_val_split(ds, 0.2, seed=1)
        assert set(tr.indices).isdisjoint(va.indices)
        assert sorted(list(tr.indices) + list(va.indices)) == list(range(20))

    def test_no_class_larger_than_train_minus_one(self, tiny_config):
        """Degenerate guard: a class that fits entirely in val must keep at
        least one sample in train."""
        ds = SyntheticNeuroDataset(3, cfg=tiny_config, seed=0, raw_mri_shape=(40, 48, 32))
        tr, va = stratified_train_val_split(ds, 0.6, seed=0)
        from src.utils import collect_labels

        assert len(collect_labels(tr)) >= 1
        assert len(collect_labels(va)) >= 1
