import pytest
import torch

from src.config import ModelConfig
from src.models.cross_attention import CrossAttentionFusion, MultiHeadCrossAttention


class TestMultiHeadCrossAttention:
    def test_output_and_attention_shapes(self, tiny_config):
        attn = MultiHeadCrossAttention(tiny_config.model)
        q = torch.randn(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        kv = torch.randn(2, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        out, weights = attn(q, kv)
        assert out.shape == q.shape
        assert weights.shape == (
            2,
            tiny_config.model.num_heads,
            tiny_config.model.spatial_tokens,
            tiny_config.model.temporal_tokens,
        )

    def test_attention_weights_sum_to_one(self, tiny_config):
        attn = MultiHeadCrossAttention(tiny_config.model)
        attn.eval()  # disable dropout so softmax rows sum exactly to 1
        q = torch.randn(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        kv = torch.randn(2, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        _, weights = attn(q, kv)
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)

    def test_handles_mismatched_query_key_lengths(self, tiny_config):
        attn = MultiHeadCrossAttention(tiny_config.model)
        q = torch.randn(1, 64, tiny_config.model.d_model)  # many spatial tokens
        kv = torch.randn(1, 5, tiny_config.model.d_model)  # few temporal tokens
        out, weights = attn(q, kv)
        assert out.shape == (1, 64, tiny_config.model.d_model)
        assert weights.shape == (1, tiny_config.model.num_heads, 64, 5)

    def test_rejects_non_divisible_heads(self):
        bad_cfg = ModelConfig(d_model=10, num_heads=3)
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadCrossAttention(bad_cfg)


class TestCrossAttentionFusion:
    def test_output_shape_matches_query_tokens(self, tiny_config):
        fusion = CrossAttentionFusion(tiny_config.model)
        spatial = torch.randn(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        temporal = torch.randn(2, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        fused, weights = fusion(spatial, temporal)
        assert fused.shape == spatial.shape
        assert set(weights.keys()) == {"forward"}
        assert weights["forward"].shape[:2] == (2, tiny_config.model.num_heads)

    def test_output_is_finite(self, tiny_config):
        fusion = CrossAttentionFusion(tiny_config.model)
        spatial = torch.randn(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        temporal = torch.randn(2, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        fused, _ = fusion(spatial, temporal)
        assert torch.all(torch.isfinite(fused))

    def test_gradients_flow_to_all_parameters(self, tiny_config):
        fusion = CrossAttentionFusion(tiny_config.model)
        spatial = torch.randn(2, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        temporal = torch.randn(2, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        fused, _ = fusion(spatial, temporal)
        fused.sum().backward()
        for name, p in fusion.named_parameters():
            assert p.grad is not None, f"no grad for {name}"

    def test_temporal_signal_actually_influences_output(self, tiny_config):
        """Sanity check that fusion is not silently ignoring the K/V stream."""
        fusion = CrossAttentionFusion(tiny_config.model)
        fusion.eval()
        spatial = torch.randn(1, tiny_config.model.spatial_tokens, tiny_config.model.d_model)
        temporal_a = torch.randn(1, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        temporal_b = torch.randn(1, tiny_config.model.temporal_tokens, tiny_config.model.d_model)
        with torch.no_grad():
            out_a, _ = fusion(spatial, temporal_a)
            out_b, _ = fusion(spatial, temporal_b)
        assert not torch.allclose(out_a, out_b)
