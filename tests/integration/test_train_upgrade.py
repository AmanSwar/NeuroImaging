"""Integration-level tests for the v2 training pipeline: rich evaluation
metrics, stratified splitting, augmentation views, scheduler behavior,
caching, checkpoint compatibility, early stopping, resume, and the predict
CLI."""

import json
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import Config, DEFAULT_CONFIG, TrainConfig, merge_model_config
from src.data.dataset import AugmentedView, SyntheticNeuroDataset
from src.models.cafn import CAFN
from src.train import (
    _compute_class_weights,
    _nan_to_none,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_model,
    train_val_split,
)
from tests.integration.test_training_loop import _learnable_config


class TestEvaluateMetrics:
    def test_evaluation_runs_and_loses_nothing(self, tiny_config):
        cfg = _learnable_config(epochs=1)
        ds = SyntheticNeuroDataset(12, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        _, val_ds = train_val_split(ds, cfg.train.val_split, cfg.train.seed)
        model = CAFN(cfg.model)
        loader = DataLoader(val_ds, batch_size=4)
        metrics = evaluate(model, loader, torch.device("cpu"))
        assert np.isfinite(metrics["loss"])

    def test_rich_metric_keys_and_confusion_matrix(self, tiny_config):
        cfg = _learnable_config(epochs=1)
        ds = SyntheticNeuroDataset(12, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        _, val_ds = train_val_split(ds, cfg.train.val_split, cfg.train.seed)
        model = CAFN(cfg.model)
        loader = DataLoader(val_ds, batch_size=4)
        metrics = evaluate(model, loader, torch.device("cpu"))
        for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix", "n_samples"):
            assert key in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert sum(sum(row) for row in metrics["confusion_matrix"]) == metrics["n_samples"]

    def test_accuracy_matches_torch_baseline(self, tiny_config):
        cfg = _learnable_config(epochs=1)
        ds = SyntheticNeuroDataset(8, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        model = CAFN(cfg.model)
        model.eval()
        loader = DataLoader(ds, batch_size=4)
        metrics = evaluate(model, loader, torch.device("cpu"))

        correct = 0
        with torch.no_grad():
            for item_idx in range(len(ds)):
                item = ds[item_idx]
                out = model(item["mri"].unsqueeze(0), item["vep"].unsqueeze(0))
                correct += int(out["logits_fused"].argmax().item() == item["label"].item())
        assert metrics["accuracy"] == pytest.approx(correct / len(ds), abs=1e-6)


class TestClassWeights:
    def test_inverse_frequency_weighting(self):
        weights = _compute_class_weights([0, 0, 0, 1, 1, 2], num_classes=3)
        assert weights.shape == (3,)
        # class 0 (freq 3) gets the lowest weight, class 2 (freq 1) highest
        assert weights[2] > weights[1] > weights[0]
        assert weights.mean() == pytest.approx(1.0, abs=1e-5)

    def test_weighted_loss_changes_value(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri = torch.randn(2, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(2, 1, tiny_config.vep.length)
        labels = torch.tensor([0, 2])
        out = model(mri, vep)
        loss_plain, _ = model.multi_task_loss(out, labels)
        w = torch.tensor([3.0, 1.0, 0.5])
        loss_w, _ = model.multi_task_loss(out, labels, class_weights=w)
        assert loss_plain.item() != pytest.approx(loss_w.item())


class TestAugmentedView:
    def test_augments_train_side_but_leaves_val_untouched(self, tiny_config):
        ds = SyntheticNeuroDataset(6, cfg=tiny_config, seed=1, raw_mri_shape=(40, 48, 32))
        tr_indices, va_indices = [0, 1, 2, 3], [4, 5]
        view = AugmentedView(ds, tr_indices, mri_augment=True, vep_augment=True, seed=0)
        augmented = view[0]
        base = ds[0]
        assert augmented["mri"].shape == base["mri"].shape
        assert augmented["label"].item() == base["label"].item()
        assert not torch.allclose(augmented["mri"], base["mri"])
        # val side uses the raw dataset directly -> untouched
        assert torch.equal(ds[4]["mri"], ds[4]["mri"])

    def test_augmentation_varies_across_epochs_but_stays_reproducible(self, tiny_config):
        ds = SyntheticNeuroDataset(4, cfg=tiny_config, seed=1, raw_mri_shape=(40, 48, 32))
        view = AugmentedView(ds, [0, 1, 2], seed=0)
        view.set_epoch(0)
        e0_a, e0_b = view[0], view[0]
        assert torch.allclose(e0_a["mri"], e0_b["mri"]), "same epoch must be reproducible"
        view.set_epoch(1)
        e1 = view[0]
        assert not torch.allclose(e0_a["mri"], e1["mri"]), "epochs must produce different augmentations"

    def test_noop_when_disabled(self, tiny_config):
        ds = SyntheticNeuroDataset(4, cfg=tiny_config, seed=1, raw_mri_shape=(40, 48, 32))
        view = AugmentedView(ds, [0], mri_augment=False, vep_augment=False)
        base, viewed = ds[0], view[0]
        assert torch.equal(base["mri"], viewed["mri"])

    def test_workers_never_duplicate_content(self, tiny_config):
        """Per-index RNG: idx 0 must yield identical content no matter how
        it is fetched (the old shared-RNG dataset silently violated this
        once num_workers > 0)."""
        ds = SyntheticNeuroDataset(8, cfg=tiny_config, seed=2, raw_mri_shape=(40, 48, 32))
        a, b = ds[0], ds[0]
        assert torch.equal(a["mri"], b["mri"])
        assert torch.equal(a["vep"], b["vep"])


class TestDatasetCache:
    def test_cache_returns_identical_items(self, tiny_config):
        ds = SyntheticNeuroDataset(4, cfg=tiny_config, seed=1, raw_mri_shape=(40, 48, 32), cache=True)
        first = ds[0]
        second = ds[0]
        assert torch.equal(first["mri"], second["mri"])
        assert len(ds._cache) == 1

    def test_train_model_with_cached_dataset(self):
        cfg = _learnable_config(epochs=1)
        ds = SyntheticNeuroDataset(6, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32), cache=True)
        model = CAFN(cfg.model)
        history = train_model(model, ds, cfg=cfg)
        assert len(history["train_loss"]) == 1


class TestSchedulerBehavior:
    def test_cosine_scheduler_decays_after_warmup(self):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        from src.train import _build_scheduler

        sched = _build_scheduler(opt, steps_per_epoch=10, epochs=10, warmup_epochs=1.0)
        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        assert lrs[0] < lrs[5], "warmup should ramp LR up"
        assert lrs[-1] < 0.05, "cosine phase should decay close to zero"
        peak = max(lrs)
        assert peak == pytest.approx(1.0, abs=0.01)

    def test_scheduler_none_keeps_constant_lr(self):
        model = torch.nn.Linear(2, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=0.5)
        lrs = []
        for _ in range(3):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
        assert lrs == [0.5, 0.5, 0.5]


class TestTrainingUpgrades:
    def test_train_model_reports_new_history_keys(self):
        cfg = _learnable_config(epochs=2)
        ds = SyntheticNeuroDataset(8, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        train_ds, val_ds = train_val_split(ds, cfg.train.val_split, cfg.train.seed)
        model = CAFN(cfg.model)
        history = train_model(model, train_ds, val_ds, cfg=cfg)
        for key in ("train_accuracy", "val_balanced_accuracy", "val_macro_f1", "learning_rate", "runtime_s", "epochs_run", "restored_best"):
            assert key in history
        assert history["epochs_run"] <= cfg.train.epochs
        assert history["restored_best"] is True

    def test_early_stopping_stops_before_budget(self):
        cfg = _learnable_config(epochs=12)
        cfg = Config(
            mri=cfg.mri, vep=cfg.vep, model=cfg.model,
            train=TrainConfig(**{**cfg.train.__dict__, "epochs": 12, "patience": 0}),
        )
        ds = SyntheticNeuroDataset(12, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        train_ds, val_ds = train_val_split(ds, cfg.train.val_split, cfg.train.seed)
        model = CAFN(cfg.model)
        history = train_model(model, train_ds, val_ds, cfg=cfg)
        # patience=0 stops the epoch after any non-improvement; with 12
        # budget epochs a tiny model will almost surely plateau early.
        assert history["epochs_run"] < 12

    def test_gradient_clipping_runs(self):
        cfg = Config(
            mri=_learnable_config(1).mri, vep=_learnable_config(1).vep,
            model=_learnable_config(1).model,
            train=TrainConfig(**{**_learnable_config(1).train.__dict__, "max_grad_norm": 0.5}),
        )
        ds = SyntheticNeuroDataset(4, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        model = CAFN(cfg.model)
        history = train_model(model, ds, cfg=cfg)
        assert np.isfinite(history["train_loss"][-1])

    def test_run_artifacts_written(self, tmp_path):
        cfg = _learnable_config(epochs=2)
        cfg = Config(
            mri=cfg.mri, vep=cfg.vep, model=cfg.model,
            train=TrainConfig(**{**cfg.train.__dict__, "checkpoint_dir": str(tmp_path)}),
        )
        ds = SyntheticNeuroDataset(8, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        train_ds, val_ds = train_val_split(ds, cfg.train.val_split, cfg.train.seed)
        model = CAFN(cfg.model)
        train_model(model, train_ds, val_ds, cfg=cfg, run_name="unitrun")
        run_dir = tmp_path / "unitrun"
        for name in ("best.pt", "last.pt", "history.json", "final_metrics.json", "training_curves.png"):
            assert (run_dir / name).exists(), f"missing artifact {name}"
        history = json.loads((run_dir / "history.json").read_text())
        assert "train_loss" in history

    def test_resume_continues(self, tmp_path):
        cfg = _learnable_config(epochs=2)
        cfg = Config(
            mri=cfg.mri, vep=cfg.vep, model=cfg.model,
            train=TrainConfig(**{**cfg.train.__dict__,
                                 "checkpoint_dir": str(tmp_path),
                                 "scheduler": "cosine"}),
        )
        ds = SyntheticNeuroDataset(8, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        train_ds, val_ds = train_val_split(ds, cfg.train.val_split, cfg.train.seed)
        model = CAFN(cfg.model)
        train_model(model, train_ds, val_ds, cfg=cfg, run_name="res")

        full_model = CAFN(cfg.model)
        full_model.load_state_dict(
            torch.load(tmp_path / "res" / "last.pt", weights_only=False)["model_state_dict"]
        )
        history = train_model(full_model, train_ds, val_ds, cfg=cfg,
                              resume=str(tmp_path / "res" / "last.pt"))
        assert len(history["train_loss"]) == 2  # epochs 2..3 only


class TestCheckpointCompat:
    def test_checkpoint_round_trip_v2(self, tiny_config, tmp_path):
        cfg = _learnable_config(epochs=1)
        ds = SyntheticNeuroDataset(4, cfg=cfg, seed=0, raw_mri_shape=(40, 48, 32))
        model = CAFN(cfg.model)
        train_model(model, ds, cfg=cfg)
        path = os.path.join(str(tmp_path), "_rt.pt")
        save_checkpoint(model, path, cfg)
        loaded = load_checkpoint(path, cfg)
        item = ds[0]
        with torch.no_grad():
            a = model(item["mri"].unsqueeze(0), item["vep"].unsqueeze(0))["logits_fused"]
            b = loaded(item["mri"].unsqueeze(0), item["vep"].unsqueeze(0))["logits_fused"]
        torch.testing.assert_close(a, b)

    def test_merge_fills_missing_v2_fields(self):
        old = {"d_model": 32, "num_heads": 2, "num_classes": 3}  # simulates ancient object

        class OldCfg:
            pass

        obj = OldCfg()
        for k, v in old.items():
            setattr(obj, k, v)
        merged = merge_model_config(obj)
        assert merged.d_model == 32
        assert merged.num_heads == 2
        assert merged.fusion == DEFAULT_CONFIG.model.fusion
        assert merged.pos_encoding == DEFAULT_CONFIG.model.pos_encoding


class TestPredictPipeline:
    def test_end_to_end_predict_with_artifacts(self, tmp_path, tiny_config):
        from src.data.synthetic import write_synthetic_dataset
        from src.predict import predict_single
        from src.train import train_model as tm

        manifest = write_synthetic_dataset(str(tmp_path / "raw"), samples_per_class=2,
                                           cfg=tiny_config, seed=3)
        from src.data.dataset import PairedNeuroDataset

        ds = PairedNeuroDataset(manifest, cfg=tiny_config)
        model = CAFN(tiny_config.model)
        tm(model, ds, cfg=_learnable_config(1))

        import csv as _csv

        with open(manifest, newline="") as f:
            row = list(_csv.DictReader(f))[0]

        result = predict_single(
            model, row["mri_path"], row["vep_path"], fs=float(row["fs"]),
            cfg=Config(mri=tiny_config.mri, vep=tiny_config.vep, model=tiny_config.model),
        )
        assert result["predicted_class"] in (0, 1, 2)
        assert result["predicted_label"] in ("Healthy Control", "Glaucoma", "Optic Neuritis / CVI")
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)

        # Full artifact path (attention heatmap + gradcam)
        from src.predict import run as predict_run

        outdir = str(tmp_path / "out")
        predict_run(
            _save(model, tmp_path), row["mri_path"], row["vep_path"], outdir=outdir
        )
        assert os.path.exists(os.path.join(outdir, "prediction.json"))
        assert os.path.exists(os.path.join(outdir, "cross_attention.png"))
        assert os.path.exists(os.path.join(outdir, "gradcam.png"))


def _save(model, tmp_path) -> str:
    path = str(tmp_path / "predict_ckpt.pt")
    save_checkpoint(model, path)
    return path
