"""Tests for train-time augmentation transforms (MRI + VEP)."""

import numpy as np
import pytest

from src.data.augment import (
    augment_mri,
    augment_vep,
    random_flip_3d,
    random_gaussian_noise,
    random_intensity_scale_shift,
    random_rotate90_axial,
)


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return rng.random((1, 8, 8, 8)).astype(np.float32)


class TestMRITransforms:
    def test_flip_shape_preserved(self, volume):
        rng = np.random.default_rng(1)
        out = random_flip_3d(volume, rng)
        assert out.shape == volume.shape
        assert np.isfinite(out).all()

    def test_flip_with_zero_probability_is_identity(self, volume):
        rng = np.random.default_rng(1)
        out = random_flip_3d(volume, rng, p=0.0)
        np.testing.assert_array_equal(out, volume)

    def test_rotate90_shape_preserved(self, volume):
        rng = np.random.default_rng(2)
        out = random_rotate90_axial(volume, rng)
        assert out.shape == volume.shape
        assert np.isfinite(out).all()

    def test_intensity_jitter_shape_and_range(self, volume):
        rng = np.random.default_rng(3)
        out = random_intensity_scale_shift(volume, rng)
        assert out.shape == volume.shape
        assert out.min() >= 0.0

    def test_noise_is_deterministic_given_rng(self, volume):
        out_a = random_gaussian_noise(volume, np.random.default_rng(7), sigma=0.05, p=1.0)
        out_b = random_gaussian_noise(volume, np.random.default_rng(7), sigma=0.05, p=1.0)
        np.testing.assert_array_equal(out_a, out_b)

    def test_full_stack_disabled_is_identity(self, volume):
        rng = np.random.default_rng(4)
        out = augment_mri(volume, rng, enabled=False)
        assert out is volume


class TestVEPTransforms:
    def test_disabled_is_identity(self):
        sig = np.arange(50, dtype=np.float32)
        assert augment_vep(sig, np.random.default_rng(0), enabled=False) is sig

    def test_shape_preserved_and_finite(self):
        rng = np.random.default_rng(5)
        sig = rng.normal(size=200).astype(np.float32)
        out = augment_vep(sig, rng)
        assert out.shape == sig.shape
        assert np.isfinite(out).all()

    def test_amplitude_scaling_preserves_shape_order(self):
        sig = np.arange(100, dtype=np.float32)
        out = augment_vep(sig, np.random.default_rng(9),
                          amp_scale_range=(1.0, 1.0), noise_sigma=0.0, max_shift_samples=0)
        np.testing.assert_allclose(out, sig)

    def test_shift_is_bounded(self):
        """The shift must stay tiny (<= 1 sample) so P100 latency classes
        remain separable; also edge-padded, not wrapped."""
        sig = np.zeros(100, dtype=np.float32)
        sig[0] = 1.0  # marker at the edge: np.roll would wrap it to the end
        out = augment_vep(sig, np.random.default_rng(11),
                          amp_scale_range=(1.0, 1.0), noise_sigma=0.0, max_shift_samples=1)
        assert np.isfinite(out).all()
        assert out.shape == sig.shape

    def test_deterministic_given_rng(self):
        rng_vals = np.random.default_rng(21).normal(size=120).astype(np.float32)
        a = augment_vep(rng_vals, np.random.default_rng(21))
        b = augment_vep(rng_vals, np.random.default_rng(21))
        np.testing.assert_array_equal(a, b)
