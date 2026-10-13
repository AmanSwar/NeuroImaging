"""Tests for stratified/group-aware dataset splitting."""

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from src.data.splits import (
    extract_groups,
    extract_labels,
    make_splits,
    split_summary,
    stratified_train_val_split,
    stratified_train_val_test_split,
)


class LabeledListDataset(Dataset):
    """Minimal dataset exposing `.labels` like SyntheticNeuroDataset."""

    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(float(idx)), torch.tensor(self.labels[idx])


class SubjectDataset(LabeledListDataset):
    """Adds `.subject_ids` so group-level splitting can be exercised."""

    def __init__(self, labels, subject_ids):
        super().__init__(labels)
        self.subject_ids = list(subject_ids)


class TestExtractLabels:
    def test_reads_labels_attribute(self):
        assert extract_labels(LabeledListDataset([0, 1, 2])) == [0, 1, 2]

    def test_reads_rows_dicts(self):
        class RowDS(Dataset):
            rows = [{"label": "1"}, {"label": "0"}]

            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return 0

        assert extract_labels(RowDS()) == [1, 0]

    def test_returns_none_when_unknown(self):
        assert extract_labels(TensorDataset(torch.arange(5))) is None

    def test_tensor_dataset_last_tensor(self):
        ds = TensorDataset(torch.zeros(4), torch.tensor([0, 1, 1, 0]))
        assert extract_labels(ds) == [0, 1, 1, 0]


class TestExtractGroups:
    def test_reads_subject_ids(self):
        ds = SubjectDataset([0, 0, 1], ["a", "a", "b"])
        assert extract_groups(ds) == ["a", "a", "b"]

    def test_returns_none_when_unknown(self):
        assert extract_groups(TensorDataset(torch.arange(3))) is None


class TestStratifiedTrainValSplit:
    def test_partitions_and_preserves_class_ratios(self):
        labels = [0] * 30 + [1] * 60 + [2] * 10
        ds = LabeledListDataset(labels)
        train, val = stratified_train_val_split(ds, val_split=0.2, seed=0)
        assert len(train) + len(val) == 100
        val_labels = [labels[i] for i in val.indices]
        # 20% of each class: 6 / 12 / 2
        assert val_labels.count(0) == 6
        assert val_labels.count(1) == 12
        assert val_labels.count(2) == 2

    def test_deterministic_for_fixed_seed(self):
        ds = LabeledListDataset([0, 1, 2, 1, 0, 2, 1, 1])
        a_tr, a_va = stratified_train_val_split(ds, 0.25, seed=7)
        b_tr, b_va = stratified_train_val_split(ds, 0.25, seed=7)
        assert a_tr.indices == b_tr.indices
        assert a_va.indices == b_va.indices

    def test_groups_stay_together(self):
        # Two samples per subject; all samples of a subject must share a split.
        ds = SubjectDataset([0, 0, 1, 1, 2, 2], ["a", "a", "b", "b", "c", "c"])
        train, val = stratified_train_val_split(ds, val_split=0.5, seed=3)
        subject_of = ["a", "a", "b", "b", "c", "c"]
        in_val = {subject_of[i] for i in val.indices}
        in_train = {subject_of[i] for i in train.indices}
        assert not (in_val & in_train), "a subject leaked across splits"
        # Group granularity: val holds whole subjects (2 or 4 samples at 50%).
        assert len(val) in (2, 4)
        assert len(val) > 0

    def test_fallback_when_no_labels(self):
        ds = TensorDataset(torch.arange(10))
        train, val = stratified_train_val_split(ds, 0.2, seed=0)
        assert len(train) == 8 and len(val) == 2

    def test_single_sample_all_train(self):
        ds = LabeledListDataset([1])
        train, val = stratified_train_val_split(ds, 0.5, seed=0)
        assert len(train) == 1 and len(val) == 0

    def test_extreme_val_split_keeps_train(self):
        ds = LabeledListDataset([0, 1, 2])
        train, val = stratified_train_val_split(ds, 0.9, seed=0)
        assert len(train) >= 1


class TestStratifiedTrainValTestSplit:
    def test_three_way_disjoint_partition(self):
        labels = [0] * 20 + [1] * 20 + [2] * 20
        ds = LabeledListDataset(labels)
        train, val, test = stratified_train_val_test_split(ds, 0.2, 0.1, seed=1)
        assert len(train) + len(val) + len(test) == 60
        all_idx = set(train.indices) | set(val.indices) | set(test.indices)
        assert len(all_idx) == 60, "splits overlap"

    def test_each_class_appears_in_test(self, tiny_config=None):
        labels = [0] * 30 + [1] * 30 + [2] * 30
        ds = LabeledListDataset(labels)
        _, _, test = stratified_train_val_test_split(ds, 0.1, 0.2, seed=2)
        test_classes = {labels[i] for i in test.indices}
        assert test_classes == {0, 1, 2}

    def test_zero_test_split(self):
        ds = LabeledListDataset([0, 1, 0, 1])
        _, val, test = stratified_train_val_test_split(ds, 0.5, 0.0, seed=0)
        assert len(test) == 0
        assert len(val) == 2


class TestHelpers:
    def test_make_splits_two_way(self, tiny_config):
        ds = LabeledListDataset([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
        splits = make_splits(ds, cfg=tiny_config)
        assert set(splits.keys()) == {"train", "val"}
        assert len(splits["train"]) + len(splits["val"]) == 10

    def test_make_splits_three_way(self, tiny_config):
        ds = LabeledListDataset([0, 1, 2] * 10)
        splits = make_splits(ds, cfg=tiny_config, test_split=0.3)
        assert set(splits.keys()) == {"train", "val", "test"}
        total = sum(len(s) for s in splits.values())
        assert total == 30

    def test_split_summary_mentions_counts(self):
        labels = [0, 1, 1, 2, 2, 2]
        ds = LabeledListDataset(labels)
        tr, va = stratified_train_val_split(ds, 0.5, seed=0)
        text = split_summary({"train": tr, "val": va}, labels=labels)
        assert "train=3" in text and "val=3" in text and "c0:" in text

    def test_invalid_fractions_rejected(self):
        ds = LabeledListDataset([0, 1, 2])
        with pytest.raises(ValueError):
            stratified_train_val_test_split(ds, 0.6, 0.6, seed=0)

    def test_all_classes_present_in_val_for_imbalanced_data(self):
        # 12:6:2 imbalance -- the plain random split can zero out class 2;
        # the stratified split must not.
        labels = [0] * 12 + [1] * 6 + [2] * 2
        ds = LabeledListDataset(labels)
        _, val = stratified_train_val_split(ds, 0.25, seed=0)
        val_classes = {labels[i] for i in val.indices}
        assert 2 in val_classes
