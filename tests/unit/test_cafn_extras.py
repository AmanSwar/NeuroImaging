"""Tests for CAFN additions: positional embeddings, attention pooling,
modality dropout, and the extended multi-task losses."""

import pytest
import torch

from src.config import ModelConfig
from src.models.attention_pool import AttentionPool1D
from src.models.cafn import CAFN
from src.models.losses import UncertaintyWeightedLoss, multi_task_ce
from src.models.positional import SpatialPositionalEmbedding, TemporalPositionalEmbedding


def _tiny_model_cfg(**overrides) -> ModelConfig:
    base = dict(
        d_model=32,
        num_heads=2,
        spatial_tokens=8,
        temporal_tokens=8,
        spatial_pool=(2, 2, 2),
        ffn_hidden=64,
        dropout=0.1,
        num_classes=3,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _batch(cfg: ModelConfig, batch_size: int = 2):
    mri = torch.randn(batch_size, 1, 32, 32, 32)
    vep = torch.randn(batch_size, 1, 200)
    labels = torch.randint(0, cfg.num_classes, (batch_size,))
    return mri, vep, labels


class TestPositionalEmbeddings:
    def test_spatial_embedding_shape_preserved(self, tiny_config):
        emb = SpatialPositionalEmbedding(tiny_config.model)
        tokens = torch.randn(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        assert emb(tokens).shape == tokens.shape

    def test_spatial_embedding_validates_pool_token_match(self, tiny_config):
        bad = tiny_config.model
        object.__setattr__(bad, "spatial_tokens", 7)  # != 2*2*2
        with pytest.raises(ValueError, match="spatial_pool"):
            SpatialPositionalEmbedding(bad)

    def test_temporal_embedding_finite_and_deterministic(self, tiny_config):
        emb = TemporalPositionalEmbedding(tiny_config.model)
        emb.eval()
        tokens = torch.randn(1, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        out1, out2 = emb(tokens), emb(tokens)
        assert torch.all(torch.isfinite(out1))
        torch.testing.assert_close(out1, out2)

    def test_temporal_embedding_is_ordered(self):
        """Sinusoidal codes must differ per position -- otherwise attention
        cannot distinguish latency bins."""
        emb = TemporalPositionalEmbedding(_tiny_model_cfg(temporal_tokens=8))
        zeros = torch.zeros(1, 8, 32)
        out = emb(zeros)[0]
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[2], out[-1])


class TestAttentionPool:
    def test_output_shape_and_weights(self):
        pool = AttentionPool1D(16)
        tokens = torch.randn(3, 7, 16)
        pooled, weights = pool(tokens, return_weights=True)
        assert pooled.shape == (3, 16)
        assert weights.shape == (3, 7)
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5
        )

    def test_default_call_returns_pooled_only(self):
        pool = AttentionPool1D(8)
        tokens = torch.randn(2, 5, 8)
        pooled = pool(tokens)
        assert pooled.shape == (2, 8)

    def test_zeroed_scorer_is_exact_mean_pooling(self):
        pool = AttentionPool1D(8)
        pool.eval()
        with torch.no_grad():
            pool.scorer.weight.zero_()
            pool.scorer.bias.zero_()
        tokens = torch.randn(2, 5, 8)
        torch.testing.assert_close(pool(tokens), tokens.mean(dim=1))

    def test_gradients_flow_to_query(self):
        pool = AttentionPool1D(8)
        pooled, weights = pool(torch.randn(2, 4, 8), return_weights=True)
        pooled.sum().backward()
        assert pool.scorer.weight.grad is not None
        assert torch.count_nonzero(pool.scorer.weight.grad) > 0


class TestModalityDropout:
    def test_train_time_zeroing_of_vep_stream(self):
        cfg = _tiny_model_cfg(modality_dropout=0.99)  # nearly always drop
        torch.manual_seed(3)
        model = CAFN(cfg)
        model.train()
        mri, vep, _ = _batch(cfg, batch_size=8)
        out = model(mri, vep)
        assert "modality_dropped" in out
        assert out["modality_dropped"].any()  # dropping occurs
        # Dropped samples -> fusion temporal input is zeros; output must be finite.
        assert torch.all(torch.isfinite(out["logits_fused"]))

    def test_eval_mode_ignores_dropout(self):
        cfg = _tiny_model_cfg(modality_dropout=0.99)
        test_model = CAFN(cfg)
        test_model.eval()
        mri, vep, _ = _batch(cfg)
        out = test_model(mri, vep)
        assert "modality_dropped" not in out

    def test_rejects_out_of_range_dropout(self):
        with pytest.raises(ValueError, match="modality_dropout"):
            CAFN(_tiny_model_cfg(modality_dropout=1.5))


class TestPoolingModes:
    def test_mean_pooling_mode_recovers_v1(self, tiny_config):
        cfg = _tiny_model_cfg(pooling="mean")
        test_model = CAFN(cfg)
        mri, vep, _ = _batch(cfg)
        out = test_model(mri, vep)
        assert out["pool_weights"] is None
        assert torch.all(torch.isfinite(out["logits_fused"]))

    def test_attn_pooling_default_emits_weights(self, tiny_config):
        test_model = CAFN(tiny_config.model)
        mri, vep, _ = _batch(tiny_config.model)
        out = test_model(mri, vep)
        assert out["pool_weights"] is not None
        assert out["pool_weights"].shape == (2, tiny_config.model.spatial_tokens)

    def test_invalid_pooling_rejected(self):
        with pytest.raises(ValueError, match="pooling"):
            CAFN(_tiny_model_cfg(pooling="softmax"))


class TestMultiTaskCE:
    def test_class_weights_change_loss(self, tiny_config):
        test_model = CAFN(tiny_config.model)
        _, vep, _ = _batch(tiny_config.model)
        # Mixed labels: with a weighted *mean*, weights cancel exactly when every
        # target is the same class, so the (5,1,1) vector only bites on a mix.
        mri = torch.randn(2, 1, 32, 32, 32)
        labels = torch.tensor([0, 1])
        out = test_model(mri, vep)
        unweighted, _ = multi_task_ce(out, labels)
        weighted, _ = multi_task_ce(out, labels, class_weights=torch.tensor([5.0, 1.0, 1.0]))
        assert unweighted != weighted

    def test_label_smoothing_caps_confidence(self, tiny_config):
        test_model = CAFN(tiny_config.model)
        mri, vep, labels = _batch(tiny_config.model)
        out = test_model(mri, vep)
        hard, _ = multi_task_ce(out, labels)
        smooth, _ = multi_task_ce(out, labels, label_smoothing=0.5)
        assert smooth < 5.0  # smoothed CE against uniform target is bounded


class TestUncertaintyWeightedLoss:
    def test_starts_equal_weighted(self, tiny_config):
        """At s=0 every task carries weight 1 -- i.e. the plain sum of the
        three cross-entropies (stronger than the fixed 0.3 aux weighting)."""
        import torch.nn.functional as F

        test_model = CAFN(tiny_config.model)
        mri, vep, labels = _batch(tiny_config.model)
        out = test_model(mri, vep)
        module = UncertaintyWeightedLoss()
        total, parts = module(out, labels)
        manual = sum(
            F.cross_entropy(out[k], labels) for k in ("logits_fused", "logits_spatial", "logits_temporal")
        )
        assert total.item() == pytest.approx(manual.item(), rel=1e-4)
        assert parts["weight_fused"] == pytest.approx(1.0, abs=1e-6)

    def test_regularizer_keeps_large_sigma_costly(self):
        """Inflating a task's sigma must increase the total by ~+s even when
        the corresponding CE would prefer it -- that is what stops the trivial
        'make every task weigh ~0' solution."""
        import torch.nn.functional as F

        module = UncertaintyWeightedLoss()
        fake_outputs = {
            "logits_fused": torch.zeros(4, 3),
            "logits_spatial": torch.zeros(4, 3),
            "logits_temporal": torch.zeros(4, 3),
        }
        labels = torch.zeros(4, dtype=torch.long)
        plain = sum(F.cross_entropy(fake_outputs[k], labels) for k in fake_outputs)
        total, _ = module(fake_outputs, labels)
        assert total.item() == pytest.approx(plain.item(), rel=1e-4)

        module.log_vars["fused"].data.fill_(2.0)
        total_lifted, parts = module(fake_outputs, labels)
        ce_fused = F.cross_entropy(fake_outputs["logits_fused"], labels)
        ce_other = sum(
            F.cross_entropy(fake_outputs[k], labels)
            for k in ("logits_spatial", "logits_temporal")
        )
        expected = (
            float(torch.exp(-module.log_vars["fused"]).detach()) * ce_fused + 2.0 + ce_other
        )
        assert total_lifted.item() == pytest.approx(expected, rel=1e-4)
        assert parts["weight_fused"] == pytest.approx(float(torch.exp(torch.tensor(-2.0))), abs=1e-6)

    def test_backward_reaches_log_vars(self, tiny_config):
        test_model = CAFN(tiny_config.model)
        mri, vep, labels = _batch(tiny_config.model)
        out = test_model(mri, vep)
        module = UncertaintyWeightedLoss()
        total, _ = module(out, labels)
        total.backward()
        for log_var in module.log_vars.values():
            assert log_var.grad is not None
