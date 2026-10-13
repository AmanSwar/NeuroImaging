"""Coverage round-out, part 2: guard branches and convenience APIs that the
higher-level pipeline tests rarely reach (config validation, legacy-checkpoint
config merging, CAFN.embed, attention-plot input variants, P100 edge shape).
"""

import numpy as np
import pytest
import torch

from src.config import AugmentConfig, ModelConfig, TrainConfig, VEPConfig, merge_model_config
from src.data.synthetic import synthetic_vep_signal
from src.data.splits import (
    extract_groups,
    extract_labels,
    stratified_train_val_test_split,
    stratified_train_val_split,
)
from src.explainability.p100 import estimate_p100
from src.explainability.plot_attention import cross_attention_heatmap
from src.models.cafn import CAFN
from src.models.cross_attention import MultiHeadCrossAttention
from src.models.spatial_encoder import SpatialEncoder3D
from src.models.temporal_encoder import TemporalEncoder1D


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        d_model=32, num_heads=2, spatial_tokens=8, temporal_tokens=8,
        spatial_pool=(2, 2, 2), ffn_hidden=64, dropout=0.0,
    )


class TestConfigValidation:
    def test_augment_invalid_ranges(self):
        with pytest.raises(ValueError):
            AugmentConfig(mri_flip_prob=1.5)
        with pytest.raises(ValueError):
            AugmentConfig(mri_intensity_scale=(1.0, 0.0))
        with pytest.raises(ValueError):
            AugmentConfig(mri_noise_std=-0.1)
        with pytest.raises(ValueError):
            AugmentConfig(vep_noise_std=-0.1)
        with pytest.raises(ValueError):
            AugmentConfig(vep_amplitude_scale=(1.2, 0.9))

    def test_backcompat_aliases(self):
        cfg = ModelConfig(
            d_model=32, num_heads=2, spatial_tokens=8, temporal_tokens=8,
            spatial_pool=(2, 2, 2),
            fusion="bidirectional",
        )
        assert cfg.bidirectional is True
        assert TrainConfig(max_grad_norm=2.0).grad_clip_norm == 2.0


class TestMergeModelConfig:
    def test_legacy_dict_bidirectional_mapping(self):
        merged = merge_model_config({"bidirectional": True, "d_model": 16})
        assert merged.fusion == "bidirectional"
        assert merged.d_model == 16

    def test_legacy_positional_and_fusion_blocks(self):
        merged = merge_model_config(
            {"use_positional_encoding": False, "num_fusion_layers": 3}
        )
        assert merged.pos_encoding is False
        assert merged.fusion_blocks == 3

    def test_legacy_dataclass_and_plain_object(self):
        old = ModelConfig(d_model=16)
        assert merge_model_config(old).d_model == 16

        class LegacyStub:  # old checkpoints froze a plain object
            pass

        stub = LegacyStub()
        stub.d_model = 24
        stub.dropout = 0.2
        stub.bidirectional = False
        assert merge_model_config(stub).d_model == 24
        assert merge_model_config(stub).fusion == "cross"

    def test_unknown_keys_dropped_from_legacy_but_rejected_in_overrides(self):
        # Legacy checkpoints carry extra fields -- they are dropped silently;
        # explicit override typos, however, must fail loudly.
        merged = merge_model_config({"not_a_field": 1, "d_model": 16})
        assert merged.d_model == 16
        with pytest.raises(ValueError, match="Unknown ModelConfig field"):
            merge_model_config({"d_model": 16}, nope=1)
        assert merge_model_config({"d_model": 16}, d_model=48).d_model == 48


class TestCAFNEmbedAndAttentionExtras:
    def test_embed_eval_mode_returns_pooled(self, tiny_config):
        model = CAFN(tiny_config.model)
        model.train()
        mri = torch.randn(1, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(1, 1, tiny_config.vep.length)
        emb = model.embed(mri, vep)
        assert emb.shape == (1, tiny_config.model.d_model)
        assert model.training  # embed always restores the previous mode

    def test_cross_attention_non_divisible_heads(self):
        bad = ModelConfig(d_model=30, num_heads=4)
        with pytest.raises(ValueError):
            MultiHeadCrossAttention(bad)

    def test_cross_attention_heatmap_dict_input(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri = torch.randn(2, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(2, 1, tiny_config.vep.length)
        out = model(mri, vep)
        # CAFN returns direction-keyed attention mappings in v3; plotting
        # accepts either the bare (B,H,S,T) tensor or the mapping.
        attn = out["attn_weights"]
        if isinstance(attn, dict):
            heat = cross_attention_heatmap(attn["forward"])
        else:
            heat = cross_attention_heatmap(attn)
        assert heat.shape == (tiny_config.model.spatial_tokens, tiny_config.model.temporal_tokens)


class TestP100EdgeShape:
    def test_peak_at_window_start_still_measured(self):
        fs = 250.0
        t = np.arange(0, 2.0, 1 / fs)
        # Sharp rise that peaks immediately at the 70 ms window edge.
        signal = np.where(t >= 0.070, np.exp(-((t - 0.070) ** 2) / (2 * 0.02**2)), 0.0)
        meas = estimate_p100(signal, fs)
        assert meas.latency_ms == pytest.approx(70.0, abs=4.5)


class TestSplitsGuards:
    def test_extract_labels_targets_attribute(self):
        class TargetDS(torch.utils.data.Dataset):
            targets = [0, 1, 2]

            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return idx

        assert extract_labels(TargetDS()) == [0, 1, 2]

    def test_extract_groups_missing_returns_none(self):
        ds = torch.utils.data.TensorDataset(torch.arange(3))
        assert extract_groups(ds) is None

    def test_subset_with_missing_inner_labels_returns_none(self):
        from torch.utils.data import Subset

        ds = torch.utils.data.TensorDataset(torch.arange(3))
        sub = Subset(ds, [0, 1])
        assert extract_labels(sub) is None
        assert extract_groups(sub) is None

    def test_degenerate_splits_never_crash(self):
        ds = _Labeled([0, 1])
        tr, va, te = stratified_train_val_test_split(ds, 0.4, 0.4, seed=0)
        assert len(tr) + len(va) + len(te) == 2
        tr, va = stratified_train_val_split(_Labeled([0] * 6 + [1] * 6), 0.01, 0)
        assert len(tr) + len(va) == 12

    def test_fraction_sum_ge_one_rejected(self):
        with pytest.raises(ValueError):
            stratified_train_val_test_split(_Labeled([0, 1, 2]), 0.5, 0.5, seed=0)


class _Labeled(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(float(idx)), torch.tensor(self.labels[idx])
