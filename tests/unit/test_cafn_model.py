import pytest
import torch

from src.models.cafn import CAFN


def _make_batch(cfg, batch_size=2):
    mri = torch.randn(batch_size, 1, *cfg.mri.target_shape)
    vep = torch.randn(batch_size, 1, cfg.vep.length)
    labels = torch.randint(0, cfg.model.num_classes, (batch_size,))
    return mri, vep, labels


class TestCAFNForward:
    def test_output_keys_and_shapes(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri, vep, _ = _make_batch(tiny_config)
        out = model(mri, vep)

        expected_keys = {
            "logits_fused",
            "logits_spatial",
            "logits_temporal",
            "attn_weights",
            "fused_tokens",
            "spatial_tokens",
            "temporal_tokens",
        }
        assert expected_keys.issubset(out.keys())
        assert out["logits_fused"].shape == (2, tiny_config.model.num_classes)
        assert out["logits_spatial"].shape == (2, tiny_config.model.num_classes)
        assert out["logits_temporal"].shape == (2, tiny_config.model.num_classes)
        assert out["attn_weights"].shape == (
            2,
            tiny_config.model.num_heads,
            tiny_config.model.spatial_tokens,
            tiny_config.model.temporal_tokens,
        )
        # v2: the backward (latency reads cortex) map exists whenever the
        # bidirectional fusion is active.
        assert "attn_weights_backward" in out

    def test_batch_size_one(self, tiny_config):
        model = CAFN(tiny_config.model)
        model.eval()
        mri, vep, _ = _make_batch(tiny_config, batch_size=1)
        out = model(mri, vep)
        assert out["logits_fused"].shape == (1, tiny_config.model.num_classes)

    def test_logits_are_finite(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri, vep, _ = _make_batch(tiny_config)
        out = model(mri, vep)
        for key in ("logits_fused", "logits_spatial", "logits_temporal"):
            assert torch.all(torch.isfinite(out[key]))


class TestCAFNMultiTaskLoss:
    def test_loss_is_finite_scalar(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri, vep, labels = _make_batch(tiny_config)
        out = model(mri, vep)
        loss, parts = model.multi_task_loss(out, labels)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert set(parts.keys()) == {"loss_fused", "loss_spatial", "loss_temporal", "loss_total"}

    def test_loss_equals_weighted_sum_of_parts(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri, vep, labels = _make_batch(tiny_config)
        out = model(mri, vep)
        loss, parts = model.multi_task_loss(out, labels, aux_weight=0.5)
        expected = parts["loss_fused"] + 0.5 * (parts["loss_spatial"] + parts["loss_temporal"])
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_backward_reaches_every_parameter(self, tiny_config):
        model = CAFN(tiny_config.model)
        mri, vep, labels = _make_batch(tiny_config)
        out = model(mri, vep)
        loss, _ = model.multi_task_loss(out, labels)
        loss.backward()

        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == [], f"parameters with no gradient: {missing}"

        all_zero = [
            n for n, p in model.named_parameters() if p.grad is not None and torch.all(p.grad == 0)
        ]
        assert all_zero == [], f"parameters with all-zero gradient: {all_zero}"

    def test_optimizer_step_changes_weights(self, tiny_config):
        model = CAFN(tiny_config.model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        before = {n: p.clone() for n, p in model.named_parameters()}

        mri, vep, labels = _make_batch(tiny_config)
        out = model(mri, vep)
        loss, _ = model.multi_task_loss(out, labels)
        loss.backward()
        optimizer.step()

        changed = any(not torch.equal(before[n], p) for n, p in model.named_parameters())
        assert changed
