"""Coverage round-out for src/utils.py and signal-cleaning edge paths.

These are thin, deterministic tests for branches the higher-level pipeline
tests rarely reach: device-resolution fallbacks, Subset-correct label/group
extraction, split degenerate cases, and VEP baseline-correction guards.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from src.preprocessing.signal_cleaner import baseline_correct, butterworth_bandpass
from src.utils import (
    auto_device,
    collect_labels,
    count_parameters,
    extract_groups,
    extract_labels,
    get_device,
    group_train_val_split,
    seed_everything,
    stratified_train_val_split,
    worker_init_fn,
)


class LabeledDS(Dataset):
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(float(idx)), torch.tensor(self.labels[idx])


class RowsDS(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return 0


class GroupedDS(Dataset):
    def __init__(self, labels, groups):
        self.labels = labels
        self.groups = groups

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(float(idx)), torch.tensor(self.labels[idx])


class TestDeviceResolution:
    def test_count_parameters_trainable_vs_all(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 3))
        assert count_parameters(model) == 4 * 3 + 3
        assert count_parameters(model, trainable_only=False) == count_parameters(model)
        for p in model.parameters():
            p.requires_grad_(False)
        assert count_parameters(model) == 0
        assert count_parameters(model, trainable_only=False) == 15

    def test_auto_device_returns_given_device_object(self):
        d = torch.device("cpu")
        assert auto_device(d) == d

    def test_auto_device_falls_back_when_no_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
        monkeypatch.setattr(
            torch.backends, "mps", type("M", (), {"is_available": staticmethod(lambda: False)}),
            raising=False,
        )
        assert auto_device("cuda").type == "cpu"

    def test_auto_device_mps_unavailable_falls_back(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
        monkeypatch.setattr(
            torch.backends, "mps", type("M", (), {"is_available": staticmethod(lambda: False)}),
            raising=False,
        )
        assert auto_device("mps").type == "cpu"

    def test_get_device_cpu_ok_and_bad_request_raises(self):
        assert get_device("cpu").type == "cpu"
        with pytest.raises(RuntimeError):
            get_device("cuda-not-a-real-device")

    def test_seed_everything_runs_in_both_modes(self):
        seed_everything(0)
        a = torch.randn(3).sum().item()
        seed_everything(0, deterministic=True)
        b = torch.randn(3).sum().item()
        seed_everything(0, deterministic=False)
        c = torch.randn(3).sum().item()
        assert a == b == c

    def test_worker_init_fn_seeds(self):
        import random as _random

        worker_init_fn(0)
        a = _random.random()
        worker_init_fn(0)
        assert _random.random() == a


class TestLabelExtraction:
    def test_extract_labels_rows(self):
        ds = RowsDS([{"label": "2"}, {"label": "0"}])
        assert extract_labels(ds) == [2, 0]

    def test_extract_labels_none(self):
        assert extract_labels(TensorDataset(torch.arange(3))) is None

    def test_collect_labels_subset_indexed(self):
        ds = LabeledDS([0, 1, 2, 1, 0, 2])
        sub = Subset(ds, [5, 0, 3])
        assert collect_labels(sub) == [2, 0, 1]

    def test_collect_labels_rows_path(self):
        ds = RowsDS([{"label": 1}, {"label": 2}, {"label": 0}])
        sub = Subset(ds, [2, 1])
        assert collect_labels(sub) == [0, 2]

    def test_collect_labels_dict_base_returns_none(self):
        assert collect_labels({"labels": [0, 1]}) is None


class TestGroupExtraction:
    def test_groups_attribute(self):
        ds = GroupedDS([0, 0, 1], ["a", "a", "b"])
        assert extract_groups(ds) == ["a", "a", "b"]

    def test_rows_with_custom_key(self):
        ds = RowsDS([{"scan_id": "s1"}, {"scan_id": "s2"}])
        assert extract_groups(ds, group_key="scan_id") == ["s1", "s2"]

    def test_rows_with_subject_id_key(self):
        ds = RowsDS([{"subject_id": "p1"}])
        assert extract_groups(ds) == ["p1"]

    def test_subject_id_attribute(self):
        ds = RowsDS([])
        ds.subject_id = ["u1", "u2"]
        assert extract_groups(ds) == ["u1", "u2"]

    def test_subset_corrected(self):
        ds = GroupedDS([0] * 4, ["a", "a", "b", "b"])
        sub = Subset(ds, [2, 3])
        assert extract_groups(sub) == ["b", "b"]

    def test_none_when_no_groups(self):
        assert extract_groups(TensorDataset(torch.arange(2))) is None


class TestStratifiedUtilsSplit:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stratified_train_val_split(LabeledDS([]), 0.2, 0)

    def test_single_sample_all_train(self):
        tr, va = stratified_train_val_split(LabeledDS([1]), 0.5, 0)
        assert len(tr) == 1 and len(va) == 0

    def test_single_class_falls_back_to_random(self):
        tr, va = stratified_train_val_split(LabeledDS([1] * 10), 0.3, 0)
        assert len(tr) == 7 and len(va) == 3

    def test_multi_sample_classes_keep_at_least_one_in_train(self):
        # Multi-sample classes are clamped to keep >= 1 sample in train;
        # the single-sample class stays whole on one side.
        labels = [0, 0, 1, 1, 2]
        tr, va = stratified_train_val_split(LabeledDS(labels), 0.4, 0)
        assert len(tr) > 0 and len(va) > 0
        assert len(tr) + len(va) == 5
        train_classes = {labels[i] for i in tr.indices}
        assert train_classes == {0, 1, 2}

    def test_partition_preserves_every_class_in_val(self):
        labels = [0] * 8 + [1] * 4 + [2] * 2
        ds = LabeledDS(labels)
        tr, va = stratified_train_val_split(ds, 0.25, 1)
        val_classes = {labels[i] for i in va.indices}
        assert val_classes == {0, 1, 2}


class TestGroupSplit:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            group_train_val_split(LabeledDS([]), 0.5, 0)

    def test_single_sample(self):
        tr, va = group_train_val_split(LabeledDS([0]), 0.5, 0)
        assert len(tr) == 1 and len(va) == 0

    def test_no_groups_falls_back_to_stratified(self):
        ds = LabeledDS([0] * 4 + [1] * 4)
        tr, va = group_train_val_split(ds, 0.25, 0)
        assert len(tr) + len(va) == 8

    def test_whole_subjects_stay_on_one_side(self):
        labels = [0] * 6
        groups = ["s1", "s1", "s2", "s2", "s3", "s3"]
        ds = GroupedDS(labels, groups)
        tr, va = group_train_val_split(ds, 0.33, 2)
        tr_groups = {groups[i] for i in tr.indices}
        va_groups = {groups[i] for i in va.indices}
        assert not (tr_groups & va_groups)  # no subject leaks
        assert va_groups, "val side should receive whole subjects"

    def test_single_group_degenerate(self):
        labels = [0, 0, 0]
        groups = ["s1", "s1", "s1"]
        ds = GroupedDS(labels, groups)
        tr, va = group_train_val_split(ds, 0.5, 0)
        assert len(tr) == 3 and len(va) == 0 or len(va) == 3


class TestSignalCleanerEdges:
    def test_baseline_correct_zero_window_raises(self):
        with pytest.raises(ValueError):
            baseline_correct(np.ones(64), 250.0, window_ms=0)

    def test_baseline_correct_removes_dc_offset(self):
        fs = 250.0
        t = np.arange(0, 2.0, 1 / fs)
        # Flat +3 V pre-stimulus baseline for the first 200 ms, then activity.
        signal = np.where(t[: len(t)] < 0.2, 3.0, 3.0 + np.sin(2 * np.pi * 10 * t))
        out = baseline_correct(signal, fs, window_ms=100.0)
        # The flat pre-stimulus window's mean (exactly 3.0) is removed.
        assert abs(out[:50].mean()) < 1e-9
        assert abs(signal[:50].mean() - 3.0) < 1e-9
        # Applied correction is a plain shift by that window mean.
        assert np.allclose(out, signal - 3.0)

    def test_bandpass_rejects_nonpositive_lowcut(self):
        with pytest.raises(ValueError):
            butterworth_bandpass(np.ones(500), 250.0, lowcut=0.0, highcut=100.0)
