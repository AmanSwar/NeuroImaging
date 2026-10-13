"""End-to-end CLI tests for the v2 add-ons: --test-split and --eval-only.

These exercise ``src.train.main(argv=...)`` in-process against tiny
synthetic datasets and assert on the per-run artifact directory.
"""

import json

import pytest

from src import train as train_module
from src.data.synthetic import write_synthetic_dataset


@pytest.fixture
def run_cli(tmp_path, monkeypatch, capsys):
    """Run main(argv) with cwd=tmp_path so relative artifact dirs land there."""
    monkeypatch.chdir(tmp_path)

    def _run(*args):
        train_module.main(list(args))
        return capsys.readouterr().out

    return _run


BASE = ("--epochs", "1", "--synthetic-samples", "24", "--batch-size", "8", "--seed", "3")


class TestTrainCliV2:
    def test_test_split_writes_test_report_and_pngs(self, run_cli, tmp_path):
        out = run_cli(*BASE, "--test-split", "0.2", "--run-name", "ts",
                      "--checkpoint-dir", "ck")
        assert "train / " in out and "test" in out
        assert "== val (detailed) ==" in out
        assert "== test (detailed) ==" in out
        art = tmp_path / "ck" / "ts"
        assert (art / "best.pt").exists()
        assert (art / "final_val_confusion_matrix.png").exists()
        assert (art / "final_test_confusion_matrix.png").exists()
        assert (art / "final_test_metrics.json").exists()
        payload = json.loads((art / "final_test_metrics.json").read_text())
        assert payload["n_samples"] == 5  # round(24 * 0.2), stratified carve

    def test_no_test_split_skips_test_report(self, run_cli, tmp_path):
        out = run_cli(*BASE, "--run-name", "nt", "--checkpoint-dir", "ck")
        assert "== val (detailed) ==" in out
        assert "== test (detailed) ==" not in out
        art = tmp_path / "ck" / "nt"
        assert (art / "final_val_confusion_matrix.png").exists()
        assert not (art / "final_test_confusion_matrix.png").exists()

    def test_eval_only_reports_without_training(self, run_cli, tmp_path):
        run_cli(*BASE, "--test-split", "0.2", "--run-name", "src_run",
                "--checkpoint-dir", "ck")
        out = run_cli("--eval-only", "ck/src_run/best.pt", "--synthetic-samples", "24",
                      "--test-split", "0.2", "--seed", "3", "--run-name", "replay",
                      "--checkpoint-dir", "ck")
        assert "loaded checkpoint" in out
        assert "== val (detailed) ==" in out
        assert "== test (detailed) ==" in out
        assert (tmp_path / "ck" / "replay" / "final_val_confusion_matrix.png").exists()

    def test_manifest_mode_with_test_split(self, run_cli, tmp_path):
        manifest = write_synthetic_dataset(str(tmp_path / "raw"), samples_per_class=4, seed=1)
        out = run_cli("--epochs", "1", "--manifest", manifest, "--test-split", "0.2",
                      "--run-name", "man", "--checkpoint-dir", "ck", "--seed", "3")
        assert "loaded manifest with 12 subjects" in out
        assert "== test (detailed) ==" in out

    def test_no_stratify_test_split_falls_back_to_random_carve(self, run_cli):
        out = run_cli(*BASE, "--test-split", "0.2", "--no-stratify",
                      "--run-name", "rnd", "--checkpoint-dir", "ck")
        assert "== test (detailed) ==" in out

    def test_detailed_reporting_failure_never_kills_run(self, run_cli, monkeypatch):
        # Sabotage the plotting helper; training must still complete.
        def boom(*a, **k):
            raise RuntimeError("plt exploded")

        monkeypatch.setattr("src.evaluation.plot_confusion_matrix", boom)
        out = run_cli(*BASE, "--run-name", "boom", "--checkpoint-dir", "ck")
        assert "detailed reporting skipped" in out
        assert "artifacts:" in out
