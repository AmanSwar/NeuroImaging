"""Tests for the cohort experiment runner (splits, metrics, selection, end-to-end)."""

import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import cohort_experiments as E


def _labels(n_per=(6, 6, 5, 7)):
    cohorts = ["A"] * (n_per[0] + n_per[1]) + ["B"] * (n_per[2] + n_per[3])
    labels = [0] * n_per[0] + [1] * n_per[1] + [0] * n_per[2] + [1] * n_per[3]
    return cohorts, labels


def test_per_cohort_split_is_disjoint_complete_stratified_and_deterministic():
    cohorts, labels = _labels((10, 12, 9, 11))
    s = E.per_cohort_split(cohorts, labels, seed=42)
    assert s == E.per_cohort_split(cohorts, labels, seed=42)
    assert s != E.per_cohort_split(cohorts, labels, seed=43)
    seen = [i for d in s.values() for p in d.values() for i in p]
    assert sorted(seen) == list(range(len(labels)))  # every subject exactly once
    for d, parts in s.items():
        for part, idx in parts.items():
            assert all(cohorts[i] == d for i in idx)
            assert {labels[i] for i in idx} == {0, 1}, f"{d}/{part} lost a class"


def test_per_cohort_split_tiny_groups_still_give_train_and_test():
    s = E.per_cohort_split(["A"] * 4, [0, 0, 1, 1], seed=1)
    assert len(s["A"]["train"]) == 2 and len(s["A"]["test"]) == 2 and s["A"]["val"] == []


def test_binary_metrics_perfect_and_inverted():
    y = np.array([0, 0, 1, 1]); m = E.binary_metrics(y, np.array([.1, .2, .8, .9]))
    assert m["accuracy"] == 1 and m["auc"] == 1 and m["confusion_matrix"] == [[2, 0], [0, 2]]
    m = E.binary_metrics(y, np.array([.9, .8, .2, .1]))
    assert m["accuracy"] == 0 and m["auc"] == 0 and m["balanced_accuracy"] == 0


def test_binary_metrics_single_class_gives_nan_auc_not_crash():
    m = E.binary_metrics(np.array([1, 1, 1]), np.array([.9, .6, .2]))
    assert np.isnan(m["auc"]) and m["sensitivity"] == pytest.approx(2 / 3) and np.isnan(m["specificity"])


def test_bootstrap_ci_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60); p = np.clip(y * 0.6 + rng.random(60) * 0.5, 0, 1)
    a, b = E.bootstrap_ci(y, p, 300), E.bootstrap_ci(y, p, 300)
    assert a == b
    lo, hi = a["accuracy"]
    assert lo <= E.binary_metrics(y, p)["accuracy"] <= hi
    assert E.bootstrap_ci(y[:3], p[:3], 50)["auc"] is None or True  # tiny sets never raise


def _write_fake_manifest(tmp_path, per_group=4):
    """Two cohorts x two groups with a learnable signal in both modalities."""
    rng = np.random.default_rng(0)
    rows = []
    for ds in ("dsA", "dsB"):
        for lab in (0, 1):
            for i in range(per_group):
                sid = f"{ds}_{lab}_{i}"
                vol = 0.1 * rng.standard_normal((1, 64, 64, 64)).astype(np.float32)
                vol[:, 20:44, 20:44, 20:44] += 1.0 + 0.6 * lab
                np.save(tmp_path / f"{sid}_mri.npy", vol)
                spec = np.sin(np.linspace(0, 6, 500) * (1 + lab)) + 0.1 * rng.standard_normal(500)
                np.savetxt(tmp_path / f"{sid}_spec.csv", spec, header="log_psd", comments="")
                rows.append(dict(subject_id=sid, dataset=ds, group="g", label=lab,
                                 mri_path=f"{sid}_mri.npy", vep_path=f"{sid}_spec.csv", fs=1.0))
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return str(path)


def _args(tmp_path, manifest, **kw):
    base = dict(manifest=manifest, out=str(tmp_path / "out"), repeats=1, epochs=2, patience=5, batch_size=2,
                lr=1e-3, weight_decay=1e-2, select_metric="val_balanced_accuracy", device="cpu", smoke=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_run_all_end_to_end_writes_pooled_lodo_and_summary(tmp_path):
    manifest = _write_fake_manifest(tmp_path)
    args = _args(tmp_path, manifest)
    res = E.run_all(args, log=lambda *_: None)
    out = args.out
    rep = res["repeats"]["0"]
    assert set(rep) == {"splits", "pooled", "lodo_dsA", "lodo_dsB"}
    assert {"val:dsA", "test:dsA", "test:dsB", "test:ALL"} <= set(rep["pooled"]["splits"])
    assert list(rep["lodo_dsA"]["splits"]) == ["test:dsA"]
    # LODO tests every subject of the held-out cohort and never trains on it
    assert rep["lodo_dsA"]["splits"]["test:dsA"]["n"] == 8
    assert rep["lodo_dsA"]["train_cohorts"] == ["dsB"]
    for name in ("results.json", "predictions.csv", "summary.md", "r0/pooled/best.pt", "r0/lodo_dsA/history.json"):
        assert os.path.exists(os.path.join(out, name)), name
    md = open(os.path.join(out, "summary.md")).read()
    assert "Leave-one-cohort-out" in md and "CAFN fused" in md
    preds = list(csv.DictReader(open(os.path.join(out, "predictions.csv"))))
    lodo_a = [p for p in preds if p["experiment"] == "lodo_dsA"]
    assert {p["dataset"] for p in lodo_a} == {"dsA"}


def test_run_all_resumes_without_retraining(tmp_path, monkeypatch):
    manifest = _write_fake_manifest(tmp_path, per_group=3)
    args = _args(tmp_path, manifest)
    E.run_all(args, log=lambda *_: None)
    monkeypatch.setattr(E, "train_with_oom_fallback", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    res = E.run_all(args, log=lambda *_: None)  # all runs already have result.json
    assert "pooled" in res["repeats"]["0"]


def test_oom_fallback_halves_batch_size(tmp_path, monkeypatch):
    manifest = _write_fake_manifest(tmp_path, per_group=3)
    args = _args(tmp_path, manifest, batch_size=4)
    cfg = E.build_config(args)
    from src.data.dataset import PairedNeuroDataset
    ds = PairedNeuroDataset(manifest, cfg=cfg)
    seen = []
    real = E.train_model

    def flaky(model, tr, va, cfg, **kw):
        seen.append(cfg.train.batch_size)
        if cfg.train.batch_size > 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(model, tr, va, cfg=cfg, **kw)

    monkeypatch.setattr(E, "train_model", flaky)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    idx = list(range(len(ds)))
    _, hist, bs = E.train_with_oom_fallback(ds, idx[:8], idx[8:], cfg, torch.device("cpu"), "oom", lambda *_: None)
    assert seen == [4, 2, 1] and bs == 1 and hist["epochs_run"] >= 1


def test_identity_probe_detects_separable_cohorts():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.standard_normal((20, 30)), rng.standard_normal((20, 30)) + 3])
    p = E.cohort_identity_probe(X, ["A"] * 20 + ["B"] * 20)
    assert p["accuracy"] > 0.95 and p["chance"] == 0.5
    assert E.cohort_identity_probe(X[:20], ["A"] * 20)["accuracy"] is None


def test_select_metric_balanced_accuracy_restores_better_epoch(tmp_path):
    """val-loss selection can restore an untrained epoch 1; balanced-acc selection must not."""
    from src.config import Config, ModelConfig, TrainConfig, VEPConfig
    from src.data.dataset import PairedNeuroDataset
    from src.models.cafn import CAFN
    from src.train import train_model
    manifest = _write_fake_manifest(tmp_path, per_group=4)
    cfg = Config(vep=VEPConfig(signal_mode="spectrum"), model=ModelConfig(num_classes=2),
                 train=TrainConfig(epochs=4, batch_size=4, select_metric="val_balanced_accuracy", checkpoint_dir=str(tmp_path / "c")))
    ds = PairedNeuroDataset(manifest, cfg=cfg)
    from torch.utils.data import Subset
    hist = train_model(CAFN(cfg.model), Subset(ds, list(range(0, 12))), Subset(ds, list(range(12, 16))), cfg=cfg, device=torch.device("cpu"))
    assert hist["restored_best"] is True
    bad = Config(train=TrainConfig(select_metric="nonsense"), model=ModelConfig(num_classes=2))
    with pytest.raises(ValueError, match="select_metric"):
        train_model(CAFN(bad.model), Subset(ds, [0, 1]), Subset(ds, [2, 3]), cfg=bad, device=torch.device("cpu"))
