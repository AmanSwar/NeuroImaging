"""Tests for the detailed evaluation utilities."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import CLASS_NAMES
from src.evaluation import (
    classification_metrics,
    evaluate_detailed,
    plot_confusion_matrix,
    plot_training_curves,
    print_classification_report,
)
from src.models.cafn import CAFN


class DictBatchDataset(TensorDataset):
    """TensorDataset that yields the dict batches CAFN-style loaders use."""

    def __getitem__(self, idx):
        mri, y = super().__getitem__(idx)
        return {"mri": torch.zeros(1), "vep": mri.unsqueeze(0), "label": y}


def _perfect_problem(n_per_class=8):
    """A trivially separable problem: label == one-hot feature value."""
    xs, ys = [], []
    for c in range(3):
        for i in range(n_per_class):
            feat = torch.zeros(4)
            feat[c] = 10.0 + i
            xs.append(feat)
            ys.append(c)
    return DictBatchDataset(torch.stack(xs), torch.tensor(ys))


class LinearProbe(torch.nn.Module):
    """Stand-in model exposing the CAFN batch/outputs contract."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 3)

    def forward(self, mri, vep):
        logits = self.net(vep.squeeze(1))
        return {"logits_fused": logits}


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        metrics = classification_metrics(y, y.copy())
        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert metrics["n_samples"] == 6
        assert metrics["empty"] is False

    def test_always_wrong_predictions(self):
        y = np.array([0, 1, 2, 0])
        wrong = np.array([1, 2, 0, 1])
        metrics = classification_metrics(y, wrong)
        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] == 0.0

    def test_balanced_accuracy_robust_to_imbalance(self):
        # Majority class always right; minority never predicted.
        y = np.array([0, 0, 0, 0, 1])
        pred = np.array([0, 0, 0, 0, 0])
        metrics = classification_metrics(y, pred)
        assert metrics["accuracy"] == 0.8
        assert metrics["balanced_accuracy"] == 0.5  # recall 1.0 and 0.0

    def test_roc_auc_included_with_probs(self):
        # All three classes present; valid probability rows that sum to 1.
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        prob = np.zeros((9, 3))
        for i, c in enumerate(y):
            main = 0.9 - 0.02 * (i % 3)
            prob[i, c] = main
            prob[i, (c + 1) % 3] = (1 - main) / 2
            prob[i, (c + 2) % 3] = (1 - main) / 2
        metrics = classification_metrics(y, prob.argmax(1), prob)
        assert 0.9 <= metrics["roc_auc_ovr"] <= 1.0

    def test_roc_auc_omitted_when_classes_missing(self):
        y = np.array([0, 0, 1, 1])  # class 2 absent -> AUC would be NaN
        prob = np.full((4, 3), 1 / 3)
        metrics = classification_metrics(y, y.copy(), prob)
        assert "roc_auc_ovr" not in metrics

    def test_per_class_support_counts(self):
        y = np.array([0, 0, 0, 1, 2])
        pred = np.array([0, 1, 0, 1, 2])
        metrics = classification_metrics(y, pred)
        assert metrics["per_class"]["Healthy Control"]["support"] == 3
        assert metrics["per_class"]["Healthy Control"]["recall"] == pytest.approx(2 / 3)

    def test_rejects_empty_and_shape_mismatch(self):
        with pytest.raises(ValueError):
            classification_metrics(np.empty(0), np.empty(0))
        with pytest.raises(ValueError):
            classification_metrics(np.zeros(3), np.zeros(4))

    def test_rejects_out_of_range_labels(self):
        with pytest.raises(ValueError):
            classification_metrics(np.array([0, 5]), np.array([0, 1]))


class TestEvaluateDetailed:
    def test_perfect_probe_scores(self):
        model = LinearProbe()
        # Train briefly so the probe separates the classes.
        ds = _perfect_problem()
        loader = DataLoader(ds, batch_size=6)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(60):
            for batch in loader:
                opt.zero_grad()
                out = model(batch["mri"], batch["vep"])
                loss = torch.nn.functional.cross_entropy(out["logits_fused"], batch["label"])
                loss.backward()
                opt.step()
        model.eval()
        criterion = lambda out, y: torch.nn.functional.cross_entropy(  # noqa: E731
            out["logits_fused"], y
        )
        metrics = evaluate_detailed(model, loader, torch.device("cpu"), criterion=criterion)
        assert metrics["accuracy"] > 0.99
        assert metrics["macro_f1"] > 0.99
        assert metrics["loss"] < 0.2

    def test_empty_loader_marks_empty(self, tiny_config):
        model = CAFN(tiny_config.model)
        loader = DataLoader(TensorDataset(torch.empty(0)), batch_size=2)
        metrics = evaluate_detailed(model, loader, torch.device("cpu"))
        assert metrics["empty"] is True
        assert np.isnan(metrics["loss"])

    def test_confusion_matrix_shape_matches_classes(self):
        model = LinearProbe()
        loader = DataLoader(_perfect_problem(), batch_size=6)
        metrics = evaluate_detailed(model, loader, torch.device("cpu"))
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (len(CLASS_NAMES), len(CLASS_NAMES))
        assert cm.sum() == len(_perfect_problem())


class TestReportAndFigures:
    def test_print_classification_report_text(self):
        metrics = classification_metrics(np.array([0, 1, 2, 1]), np.array([0, 1, 2, 1]))
        text = print_classification_report(metrics)
        assert "acc=1.0000" in text
        for name in CLASS_NAMES:
            assert name in text

    def test_plot_confusion_matrix_saves(self, tmp_path):
        cm = [[3, 1, 0], [0, 4, 0], [0, 0, 2]]
        path = tmp_path / "cm.png"
        out = plot_confusion_matrix(cm, str(path))
        assert np.array_equal(out, np.array(cm))
        assert path.exists() and path.stat().st_size > 0

    def test_plot_confusion_matrix_rejects_nonsquare(self, tmp_path):
        with pytest.raises(ValueError):
            plot_confusion_matrix([[1, 2, 3], [4, 5, 6]], str(tmp_path / "x.png"))

    def test_plot_training_curves_saves(self, tmp_path):
        history = {
            "train_loss": [2.0, 1.5, 1.0],
            "val_loss": [1.9, 1.6, 1.1],
            "val_accuracy": [0.3, 0.5, 0.7],
            "lr": [1e-3, 8e-4, 5e-4],
        }
        path = tmp_path / "curves.png"
        plot_training_curves(history, str(path))
        assert path.exists() and path.stat().st_size > 0

    def test_plot_training_curves_requires_history(self, tmp_path):
        with pytest.raises(ValueError):
            plot_training_curves({"train_loss": []}, str(tmp_path / "y.png"))
