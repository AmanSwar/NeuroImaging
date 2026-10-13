"""Tests for explainability v2: Grad-CAM++, direction-aware attention plots."""

import numpy as np
import pytest
import torch

from src.explainability.grad_cam_3d import GradCAM3D, compute_gradcam
from src.explainability.plot_attention import (
    cross_attention_heatmap,
    plot_attention_by_head,
    plot_cross_attention,
)
from src.models.cafn import CAFN


def _tiny_cfg():
    from src.config import ModelConfig

    return ModelConfig(
        d_model=32,
        num_heads=2,
        spatial_tokens=8,
        temporal_tokens=8,
        spatial_pool=(2, 2, 2),
        ffn_hidden=64,
        dropout=0.1,
        num_classes=3,
    )


def _model_and_batch():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = CAFN(cfg)
    mri = torch.rand(1, 1, 32, 32, 32)  # positive so gamma/pow paths are sane
    vep = torch.randn(1, 1, 200)
    return model, mri, vep, cfg


class TestGradCAMPlusPlus:
    def test_gradcam_plus_plus_runs_and_shapes_match(self):
        model, mri, vep, cfg = _model_and_batch()
        cam, target = compute_gradcam(model, mri, vep, method="gradcam++")
        assert cam.shape == (1, 32, 32, 32)
        assert cam.min() >= 0.0 and cam.max() <= 1.0
        assert int(target[0]) in range(3)

    def test_gradcam_and_pp_maps_differ_on_a_peaked_input(self):
        """A single bright lesion block should make the two variants finite
        and positively correlated but not identical (++ re-weights pixels)."""
        model, mri, vep, cfg = _model_and_batch()
        mri = mri.clone()
        mri[0, 0, 8:12, 8:12, 8:12] += 3.0
        cam_a, _ = compute_gradcam(model, mri, vep, method="gradcam")
        cam_b, _ = compute_gradcam(model, mri, vep, method="gradcam++")
        assert torch.all(torch.isfinite(cam_a)) and torch.all(torch.isfinite(cam_b))
        assert not torch.allclose(cam_a, cam_b)

    def test_fallback_reports_method_used(self):
        model, mri, vep, _ = _model_and_batch()
        engine = GradCAM3D(model, method="gradcam++")
        try:
            cam, _, used = engine.generate_with_fallback(mri, vep)
            assert cam.dim() == 4
            assert used in ("gradcam", "gradcam++")
        finally:
            engine.remove_hooks()

    def test_invalid_method_rejected(self):
        model, _, _, _ = _model_and_batch()
        with pytest.raises(ValueError, match="method"):
            GradCAM3D(model, method="lime")


class TestDictAwareAttentionPlots:
    def test_heatmap_accepts_direction_dict(self):
        weights = torch.softmax(torch.randn(2, 4, 10, 6), dim=-1)
        as_dict = {"forward": weights, "backward": torch.softmax(torch.randn(2, 4, 6, 10), dim=-1)}
        from_dict = cross_attention_heatmap(as_dict)
        from_tensor = cross_attention_heatmap(weights)
        np.testing.assert_allclose(from_dict, from_tensor)

    def test_heatmap_rejects_dict_without_forward(self):
        with pytest.raises(ValueError, match="forward"):
            cross_attention_heatmap({"backward": torch.rand(2, 4, 4, 4)})

    def test_plot_cross_attention_accepts_dict(self, tmp_path):
        weights = torch.softmax(torch.randn(2, 4, 10, 6), dim=-1)
        path = tmp_path / "attn.png"
        heatmap = plot_cross_attention({"forward": weights}, str(path))
        assert path.exists() and heatmap.shape == (10, 6)

    def test_per_head_plot_shapes(self, tmp_path):
        weights = torch.softmax(torch.randn(2, 4, 10, 6), dim=-1)
        path = tmp_path / "heads.png"
        head_maps = plot_attention_by_head(weights, str(path))
        assert len(head_maps) == 4
        assert all(m.shape == (10, 6) for m in head_maps)
        assert path.exists() and path.stat().st_size > 0
