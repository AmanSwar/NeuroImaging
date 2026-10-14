import nibabel as nib
import numpy as np
import pytest

from src.config import DEFAULT_CONFIG, MRIConfig
from src.preprocessing.mri_transforms import (
    foreground_mask_threshold,
    load_nifti,
    preprocess_mri_file,
    preprocess_mri_volume,
    resample_isotropic,
)


class TestLoadNifti:
    def test_returns_ras_oriented_data(self, synthetic_nifti_path):
        path = synthetic_nifti_path(shape=(40, 50, 30), affine=np.diag([-1.5, 1.5, 2.0, 1.0]))
        data, affine, zooms = load_nifti(path)
        assert data.ndim == 3
        ornt = nib.orientations.io_orientation(affine)
        codes = nib.orientations.ornt2axcodes(ornt)
        assert codes == ("R", "A", "S")

    def test_returns_voxel_zooms(self, synthetic_nifti_path):
        path = synthetic_nifti_path(affine=np.diag([1.5, 1.5, 2.0, 1.0]))
        _, _, zooms = load_nifti(path)
        assert len(zooms) == 3
        assert all(z > 0 for z in zooms)

    def test_rejects_non_3d_volume(self, tmp_path):
        data = np.random.rand(10, 10, 10, 3).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "bad.nii.gz"
        nib.save(img, str(path))
        with pytest.raises(ValueError, match="3D"):
            load_nifti(str(path))


class TestForegroundMaskThreshold:
    def test_zeroes_low_intensity_voxels(self):
        # Nonzero voxels span two intensity levels so the percentile threshold
        # has something to split; a single uniform level would remove nothing.
        volume = np.concatenate(
            [np.zeros(50), np.full(25, 10.0), np.full(25, 100.0)]
        ).reshape(10, 10, 1)
        masked = foreground_mask_threshold(volume, percentile=50.0)
        assert masked.min() == 0.0
        assert (masked > 0).sum() < (volume > 0).sum()

    def test_all_zero_volume_is_noop(self):
        volume = np.zeros((5, 5, 5))
        masked = foreground_mask_threshold(volume, percentile=60.0)
        np.testing.assert_array_equal(masked, volume)


class TestResampleIsotropic:
    def test_anisotropic_spacing_upsamples_correct_axis(self):
        volume = np.random.rand(10, 10, 10).astype(np.float32)
        out = resample_isotropic(volume, zooms=(2.0, 1.0, 1.0), target_spacing=(1.0, 1.0, 1.0))
        # 2mm spacing over 10 voxels -> 20mm extent -> ~20 voxels at 1mm
        assert out.shape[0] == 20
        assert out.shape[1] == 10
        assert out.shape[2] == 10

    def test_isotropic_input_shape_unchanged(self):
        volume = np.random.rand(12, 12, 12).astype(np.float32)
        out = resample_isotropic(volume, zooms=(1.0, 1.0, 1.0))
        assert out.shape == (12, 12, 12)


class TestPreprocessMriVolume:
    def test_output_shape_and_range(self, rng):
        cfg = MRIConfig(target_shape=(32, 32, 32))
        volume = rng.random((50, 60, 40)).astype(np.float32) * 255
        out = preprocess_mri_volume(volume, zooms=(1.2, 1.2, 1.5), cfg=cfg)
        assert out.shape == (1, 32, 32, 32)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5

    def test_output_shape_default_config(self, rng):
        volume = rng.random((70, 80, 55)).astype(np.float32) * 255
        out = preprocess_mri_volume(volume, zooms=(1.0, 1.0, 1.0))
        assert out.shape == (1, *DEFAULT_CONFIG.mri.target_shape)

    def test_constant_volume_does_not_produce_nans(self):
        cfg = MRIConfig(target_shape=(16, 16, 16))
        volume = np.full((20, 20, 20), 7.0, dtype=np.float32)
        out = preprocess_mri_volume(volume, zooms=(1.0, 1.0, 1.0), cfg=cfg)
        assert np.all(np.isfinite(out))

    def test_rejects_non_3d_volume(self):
        with pytest.raises(ValueError, match="3D"):
            preprocess_mri_volume(np.zeros((10, 10)), zooms=(1.0, 1.0, 1.0))


class TestPreprocessMriFile:
    def test_full_file_pipeline(self, synthetic_nifti_path):
        path = synthetic_nifti_path(shape=(80, 96, 60), affine=np.diag([1.5, 1.5, 2.0, 1.0]))
        out = preprocess_mri_file(path)
        assert out.shape == (1, *DEFAULT_CONFIG.mri.target_shape)
        assert np.all(np.isfinite(out))


def test_bbox_resize_fits_whole_head_instead_of_centre_crop():
    """A 150^3 'head' must be rescaled into 64^3, not centre-cropped to a 64^3 sliver."""
    import numpy as np
    from src.config import MRIConfig
    from src.preprocessing.mri_transforms import preprocess_mri_volume

    g = np.indices((150, 150, 150)).astype(float) - 75
    head = np.where(np.sqrt((g ** 2).sum(0)) < 60, 100.0, 0.0).astype(np.float32)
    crop = preprocess_mri_volume(head, (1, 1, 1), MRIConfig(mask_method="otsu", normalization="zscore", fit="crop"))
    boxed = preprocess_mri_volume(head, (1, 1, 1), MRIConfig(mask_method="otsu", normalization="zscore", fit="bbox_resize"))
    assert crop.shape == boxed.shape == (1, 64, 64, 64)
    # the crop sees only head interior (no background); the resized volume contains the whole object edge
    assert (crop[0] == 0).mean() < 0.01
    assert (boxed[0] == 0).mean() > 0.05 and np.isfinite(boxed).all()


def test_unknown_fit_is_rejected():
    import numpy as np
    import pytest
    from src.config import MRIConfig
    from src.preprocessing.mri_transforms import preprocess_mri_volume

    with pytest.raises(ValueError, match="fit"):
        preprocess_mri_volume(np.random.rand(20, 20, 20).astype(np.float32), (1, 1, 1), MRIConfig(fit="stretch"))
