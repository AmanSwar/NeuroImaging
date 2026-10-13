"""End-to-end system test: raw files on disk -> preprocessing -> model ->
loss -> backward -> optimizer step -> explainability, all wired together
exactly as a real training/inference run would use them.
"""

import numpy as np
import torch

from src.config import Config
from src.data.dataset import PairedNeuroDataset
from src.data.synthetic import write_synthetic_dataset
from src.explainability.grad_cam_3d import compute_gradcam, plot_gradcam_slices
from src.explainability.plot_attention import plot_cross_attention
from src.models.cafn import CAFN
from src.train import save_checkpoint


class TestFullPipelineFromRawFiles:
    def test_raw_files_to_trained_step(self, tmp_path, tiny_config):
        manifest = write_synthetic_dataset(
            str(tmp_path / "raw"), samples_per_class=2, cfg=tiny_config, seed=7
        )
        dataset = PairedNeuroDataset(manifest, cfg=tiny_config)
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        batch = next(iter(loader))

        model = CAFN(tiny_config.model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        before = {n: p.clone() for n, p in model.named_parameters()}

        outputs = model(batch["mri"], batch["vep"])
        loss, parts = model.multi_task_loss(outputs, batch["label"], aux_weight=tiny_config.train.aux_loss_weight)
        assert torch.isfinite(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = any(not torch.equal(before[n], p) for n, p in model.named_parameters())
        assert changed, "optimizer step should update at least one parameter"

    def test_preprocessing_output_feeds_model_without_shape_errors(self, tmp_path, tiny_config):
        manifest = write_synthetic_dataset(
            str(tmp_path / "raw"), samples_per_class=1, cfg=tiny_config, seed=11
        )
        dataset = PairedNeuroDataset(manifest, cfg=tiny_config)
        item = dataset[0]

        model = CAFN(tiny_config.model)
        model.eval()
        with torch.no_grad():
            out = model(item["mri"].unsqueeze(0), item["vep"].unsqueeze(0))
        assert out["logits_fused"].shape == (1, tiny_config.model.num_classes)


class TestExplainabilityOnRealPipelineOutput:
    def test_gradcam_and_attention_plot_from_dataset_sample(self, tmp_path, tiny_config):
        manifest = write_synthetic_dataset(
            str(tmp_path / "raw"), samples_per_class=1, cfg=tiny_config, seed=3
        )
        dataset = PairedNeuroDataset(manifest, cfg=tiny_config)
        item = dataset[0]
        mri = item["mri"].unsqueeze(0)
        vep = item["vep"].unsqueeze(0)

        model = CAFN(tiny_config.model)

        cam, target_class = compute_gradcam(model, mri, vep)
        assert cam.shape == (1, *tiny_config.mri.target_shape)

        gradcam_path = tmp_path / "gradcam.png"
        plot_gradcam_slices(mri[0, 0].numpy(), cam[0].numpy(), str(gradcam_path))
        assert gradcam_path.exists() and gradcam_path.stat().st_size > 0

        model.eval()
        with torch.no_grad():
            out = model(mri, vep)
        attn_path = tmp_path / "attn.png"
        heatmap = plot_cross_attention(out["attn_weights"], str(attn_path))
        assert attn_path.exists() and attn_path.stat().st_size > 0
        assert heatmap.shape == (tiny_config.model.spatial_tokens, tiny_config.model.temporal_tokens)


class TestPredictCLISmoke:
    def test_main_entrypoint_exists_and_runs(self, tmp_path, tiny_config):
        """Regression: the v2 refactor dropped `main()`/`__main__` from
        src/predict.py, so `python -m src.predict` silently did nothing
        (exit 0, no artifacts). This pins the CLI shell in place."""
        from src.data.synthetic import write_synthetic_dataset
        from src.models.cafn import CAFN
        from src.train import save_checkpoint
        from src.predict import main as predict_main

        manifest = write_synthetic_dataset(str(tmp_path / "raw"), samples_per_class=1,
                                           cfg=tiny_config, seed=3)
        import csv as _csv
        row = list(_csv.DictReader(open(manifest)))[0]  # noqa: SIM115
        ckpt = str(tmp_path / "cli.pt")
        save_checkpoint(CAFN(tiny_config.model), ckpt,
                        Config(mri=tiny_config.mri, vep=tiny_config.vep, model=tiny_config.model))

        outdir = tmp_path / "cli_out"
        predict_main([
            "--checkpoint", ckpt,
            "--mri", row["mri_path"],
            "--vep", row["vep_path"],
            "--fs", str(row["fs"]),
            "--outdir", str(outdir),
        ])
        for name in ("prediction.json", "cross_attention.png", "gradcam.png"):
            assert (outdir / name).exists() and (outdir / name).stat().st_size > 0
        import json
        result = json.loads((outdir / "prediction.json").read_text())
        assert result["predicted_class"] in (0, 1, 2)
        assert result["predicted_label"] in ("Healthy Control", "Glaucoma", "Optic Neuritis / CVI")


class TestClassConditionedSignalIsLearnable:
    def test_glaucoma_vs_neuritis_vep_signals_are_distinguishable(self, tiny_config):
        """Sanity check on the synthetic data generator itself: distinct
        classes must actually produce distinguishable preprocessed signals,
        otherwise no downstream model could ever learn from them."""
        from src.data.synthetic import synthetic_vep_signal
        from src.preprocessing.signal_cleaner import clean_vep_signal

        rng = np.random.default_rng(0)
        sig_glaucoma = synthetic_vep_signal(1, fs=tiny_config.vep.sample_rate, rng=rng)
        sig_neuritis = synthetic_vep_signal(2, fs=tiny_config.vep.sample_rate, rng=rng)

        cleaned_glaucoma = clean_vep_signal(sig_glaucoma, tiny_config.vep.sample_rate, tiny_config.vep)
        cleaned_neuritis = clean_vep_signal(sig_neuritis, tiny_config.vep.sample_rate, tiny_config.vep)

        peak_idx_glaucoma = np.argmax(cleaned_glaucoma)
        peak_idx_neuritis = np.argmax(cleaned_neuritis)
        assert peak_idx_glaucoma != peak_idx_neuritis
