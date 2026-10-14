"""Fusion baselines, unimodal references and the controllable synthetic informativeness."""

import numpy as np
import pytest
import torch

from src.config import ModelConfig
from src.data.synthetic import synthetic_mri_volume, synthetic_vep_signal
from src.models.cafn import CAFN, FUSION_MODES


@pytest.mark.parametrize("mode", FUSION_MODES)
def test_every_fusion_mode_forward_backward(mode):
    cfg = ModelConfig(d_model=32, num_heads=2, ffn_hidden=64, spatial_tokens=8, temporal_tokens=8,
                      spatial_pool=(2, 2, 2), fusion=mode, num_classes=2)
    m = CAFN(cfg)
    out = m(torch.randn(2, 1, 16, 16, 16), torch.randn(2, 1, 200))
    assert out["logits_fused"].shape == (2, 2)
    loss, _ = m.multi_task_loss(out, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)


def test_unimodal_references_ignore_the_other_modality():
    cfg = ModelConfig(d_model=32, num_heads=2, ffn_hidden=64, spatial_tokens=8, temporal_tokens=8,
                      spatial_pool=(2, 2, 2), dropout=0.0, num_classes=2)
    mri, vep = torch.randn(2, 1, 16, 16, 16), torch.randn(2, 1, 200)
    for mode, flip in (("mri_only", "vep"), ("vep_only", "mri")):
        m = CAFN(ModelConfig(**{**cfg.__dict__, "fusion": mode})).eval()
        base = m(mri, vep)["logits_fused"]
        other = m(mri, torch.randn_like(vep))["logits_fused"] if flip == "vep" else m(torch.randn_like(mri), vep)["logits_fused"]
        torch.testing.assert_close(base, other)


def test_unknown_fusion_still_rejected():
    with pytest.raises(ValueError, match="fusion"):
        CAFN(ModelConfig(fusion="nonsense"))


def test_mri_strength_zero_removes_class_signal_and_one_is_default():
    a = synthetic_mri_volume(2, shape=(40, 40, 40), rng=np.random.default_rng(3))
    b = synthetic_mri_volume(2, shape=(40, 40, 40), rng=np.random.default_rng(3), strength=1.0)
    np.testing.assert_array_equal(a, b)
    lesion = synthetic_mri_volume(2, shape=(40, 40, 40), rng=np.random.default_rng(3), strength=0.0)
    healthy = synthetic_mri_volume(0, shape=(40, 40, 40), rng=np.random.default_rng(3))
    # with no lesion contrast the label-2 volume matches the lesion-free statistics (same rng stream up to the lesion)
    assert abs(float(lesion.mean()) - float(healthy.mean())) < 1.0


def test_vep_strength_moves_class_latency_towards_healthy():
    def peak_ms(label, strength):
        sig = synthetic_vep_signal(label, rng=np.random.default_rng(11), strength=strength)
        t = np.arange(sig.size) / 250.0 * 1000
        win = (t > 60) & (t < 220)
        return float(t[win][np.argmax(sig[win])])
    full, weak = peak_ms(2, 1.0), peak_ms(2, 0.0)
    assert full > weak + 20          # delayed P100 at full strength, healthy-like at 0
    assert abs(weak - peak_ms(0, 1.0)) < 12
