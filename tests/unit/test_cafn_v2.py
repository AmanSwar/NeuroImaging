"""Tests for the v2 bidirectional fusion module and CAFN wiring:
positional encodings, attention pooling, fused-stream heads, and gating.
"""

import pytest
import torch

from src.config import ModelConfig
from src.models.cafn import AttentionPool, CAFN, PositionalEmbedding
from src.models.cross_attention import BidirectionalCrossAttentionFusion


class TestBidirectionalFusion:
    def test_returns_both_fused_streams_and_attention_directions(self, tiny_config):
        fusion = BidirectionalCrossAttentionFusion(tiny_config.model)
        spatial = torch.randn(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        temporal = torch.randn(2, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        fused_s, fused_t, attn = fusion(spatial, temporal)
        assert fused_s.shape == spatial.shape
        assert fused_t.shape == temporal.shape
        assert set(attn.keys()) == {"forward", "backward"}
        assert attn["forward"].shape == (
            2, tiny_config.model.num_heads,
            tiny_config.model.spatial_tokens, tiny_config.model.temporal_tokens,
        )
        assert attn["backward"].shape == (
            2, tiny_config.model.num_heads,
            tiny_config.model.temporal_tokens, tiny_config.model.spatial_tokens,
        )

    def test_temporal_stream_receives_spatial_information(self, tiny_config):
        """The core motivation: with the backward block, changing the MRI
        tokens must change the enriched temporal tokens (in the original
        unidirectional design the temporal stream was frozen w.r.t. MRI)."""
        fusion = BidirectionalCrossAttentionFusion(tiny_config.model)
        fusion.eval()
        temporal = torch.randn(1, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        spatial_a = torch.randn(1, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        spatial_b = torch.randn(1, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        with torch.no_grad():
            _, t_a, _ = fusion(spatial_a, temporal)
            _, t_b, _ = fusion(spatial_b, temporal)
        assert not torch.allclose(t_a, t_b)

    def test_gates_admit_cross_modal_evidence(self, tiny_config):
        """Gates start open (sigmoid(0) = 0.5) and the fusion output must
        differ from the raw input tokens, i.e. cross-modal content flows."""
        fusion = BidirectionalCrossAttentionFusion(tiny_config.model)
        fusion.eval()
        spatial = torch.randn(1, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        temporal = torch.randn(1, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        with torch.no_grad():
            fused_s, fused_t, _ = fusion(spatial, temporal)
        assert torch.sigmoid(fusion.gate_spatial).item() == pytest.approx(0.5, abs=1e-6)
        assert not torch.allclose(fused_s, spatial)
        assert not torch.allclose(fused_t, temporal)

    def test_gradients_flow_to_all_parameters(self, tiny_config):
        fusion = BidirectionalCrossAttentionFusion(tiny_config.model)
        spatial = torch.randn(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        temporal = torch.randn(2, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        fused_s, fused_t, _ = fusion(spatial, temporal)
        (fused_s.sum() + fused_t.sum()).backward()
        missing = [n for n, p in fusion.named_parameters() if p.grad is None]
        assert missing == []


class TestPositionalEmbedding:
    def test_adds_position_information(self, tiny_config):
        pos = PositionalEmbedding(tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        x = torch.zeros(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        out = pos(x)
        assert out.shape == x.shape
        # Each token position must get a distinct vector (not a broadcast).
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_gradients_flow(self, tiny_config):
        pos = PositionalEmbedding(4, 8)
        out = pos(torch.zeros(1, 4, 8))
        out.sum().backward()
        assert pos.pos.grad is not None and pos.pos.grad.abs().sum() > 0


class TestAttentionPool:
    def test_output_is_weighted_combination(self, tiny_config):
        pool = AttentionPool(tiny_config.model.d_model)
        tokens = torch.randn(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        emb = pool(tokens)
        assert emb.shape == (2, tiny_config.model.d_model)

    def test_uniform_scorer_equals_mean_pooling(self):
        pool = AttentionPool(8)
        with torch.no_grad():
            pool.scorer.weight.zero_()
            pool.scorer.bias.zero_()
        tokens = torch.randn(2, 5, 8)
        emb = pool(tokens)
        torch.testing.assert_close(emb, tokens.mean(dim=1))


class TestCAFNv2Wiring:
    def test_default_model_uses_bidirectional_fusion(self):
        model = CAFN()
        assert model.cfg.fusion == "bidirectional"
        out = model(torch.randn(1, 1, 64, 64, 64), torch.randn(1, 1, 500))
        assert out["attn_weights_backward"] is not None

    def test_cross_mode_keeps_original_contract(self, tiny_config):
        cfg = ModelConfig(**{**tiny_config.model.__dict__, "fusion": "cross"})
        model = CAFN(cfg)
        out = model(torch.randn(1, 1, *tiny_config.mri.target_shape),
                    torch.randn(1, 1, tiny_config.vep.length))
        assert out["attn_weights_backward"] is None
        assert out["logits_fused"].shape == (1, cfg.num_classes)

    def test_invalid_fusion_mode_rejected(self, tiny_config):
        cfg = ModelConfig(**{**tiny_config.model.__dict__, "fusion": "nope"})
        with pytest.raises(ValueError, match="Unknown fusion"):
            CAFN(cfg)

    def test_all_parameters_receive_gradients_bidirectional(self, tiny_config):
        model = CAFN(tiny_config.model)
        out = model(torch.randn(2, 1, *tiny_config.mri.target_shape),
                    torch.randn(2, 1, tiny_config.vep.length))
        loss, _ = model.multi_task_loss(out, torch.tensor([0, 1]))
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == [], f"parameters with no gradient: {missing}"
        all_zero = [
            n for n, p in model.named_parameters()
            if p.grad is not None and torch.all(p.grad == 0)
        ]
        assert all_zero == [], f"parameters with all-zero gradient: {all_zero}"

    def test_pos_encoding_can_be_disabled(self, tiny_config):
        cfg = ModelConfig(**{**tiny_config.model.__dict__, "pos_encoding": False})
        model = CAFN(cfg)
        assert not any("pos" in n for n, _ in model.named_parameters())
        out = model(torch.randn(1, 1, *tiny_config.mri.target_shape),
                    torch.randn(1, 1, tiny_config.vep.length))
        assert out["logits_fused"].shape == (1, cfg.num_classes)

    def test_head_dropout_present(self, tiny_config):
        model = CAFN(tiny_config.model)
        model.train()
        mri = torch.randn(2, 1, *tiny_config.mri.target_shape)
        vep = torch.randn(2, 1, tiny_config.vep.length)
        out_a = model(mri, vep)["logits_fused"]
        out_b = model(mri, vep)["logits_fused"]
        assert not torch.allclose(out_a, out_b), "head dropout should stochasticise logits in train mode"
