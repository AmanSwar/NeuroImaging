"""3D sMRI spatial standardization: RAS reorientation, 1mm isotropic
resampling, foreground masking, and intensity normalization.

Foreground ("skull-strip") masking
----------------------------------
True skull-stripping (FSL BET / HD-BET / SynthStrip) requires an external
binary or a pretrained segmentation network that is not pip-installable.
Two self-contained approximations ship here:

- ``otsu`` (default): Otsu's automatic threshold separates brain tissue
  from background/skull on the bimodal intensity histogram -- far less
  destructive than a raw percentile, which on real MRI zeroes large
  portions of genuinely useful tissue. An optional morphological cleanup
  (hole filling + keep-largest-connected-component, via ``scipy.ndimage``)
  removes stray voxels and produces a solid brain mask.
- ``percentile``: the v1 intensity-percentile mask, kept for reproducing
  earlier experiments.

Both are drop-in placeholders -- swap in HD-BET or SynthStrip for
clinical-grade deployment; the rest of the pipeline is agnostic to which
method produced the mask.
"""

import nibabel as nib
import numpy as np
from monai.transforms import Resize, ResizeWithPadOrCrop, ScaleIntensity
from scipy import ndimage

from src.config import MRIConfig, DEFAULT_CONFIG


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI volume, reoriented to RAS (nibabel's closest-canonical).

    Returns (data, affine, voxel_zooms).
    """
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if any(z <= 0 for z in zooms):
        raise ValueError(f"Invalid voxel spacing in header: {zooms}")
    return data, img.affine, zooms


def otsu_threshold(volume: np.ndarray, nbins: int = 128) -> float:
    """Otsu's threshold over strictly-positive voxels.

    Zero-valued (air) voxels are excluded so an already-masked or
    zero-padded volume does not skew the histogram toward background.
    Returns the intensity that best separates the two modes; a volume with
    a single (or degenerate) intensity level returns its minimum value so
    masking becomes a no-op.
    """
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return 0.0
    hist, edges = np.histogram(nonzero, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    weight_left = np.cumsum(hist).astype(np.float64)
    weight_right = weight_left[-1] - weight_left
    if weight_left[-1] == 0:
        return float(nonzero.min())

    mean_left = np.cumsum(hist * centers) / np.maximum(weight_left, 1e-12)
    total_mean = np.cumsum(hist * centers)[-1] / weight_left[-1]
    mean_right = (total_mean * weight_left[-1] - np.cumsum(hist * centers)) / np.maximum(
        weight_right, 1e-12
    )

    valid = (weight_left > 0) & (weight_right > 0)
    if not np.any(valid):
        return float(nonzero.min())
    variance_between = np.where(
        valid, weight_left * weight_right * (mean_left - mean_right) ** 2, -1.0
    )
    return float(centers[int(np.argmax(variance_between))])


def _largest_component_with_holes_filled(mask: np.ndarray) -> np.ndarray:
    """Binary morphology: fill interior holes, keep the largest component."""
    mask = ndimage.binary_fill_holes(mask)
    labels, num_features = ndimage.label(mask)
    if num_features <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    return labels == int(np.argmax(sizes))


def foreground_mask_threshold(volume: np.ndarray, percentile: float) -> np.ndarray:
    """Legacy v1 mask: zero out voxels below an intensity percentile.

    Percentile is computed over strictly-positive voxels so that an
    already-large air/background region does not collapse the threshold to
    zero. Kept for reproducibility of v1 experiments; ``foreground_mask_otsu``
    is the recommended default in ``preprocess_mri_volume``.
    """
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return volume
    threshold = np.percentile(nonzero, percentile)
    masked = volume.copy()
    masked[masked < threshold] = 0.0
    return masked


def foreground_mask_otsu(
    volume: np.ndarray, morphology: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Otsu-threshold foreground mask with optional morphological cleanup.

    Returns (masked_volume, mask) where masked_volume zeroes everything
    below the threshold. A volume with no positive voxels is returned
    unchanged with an all-False mask; a degenerate single-level volume is
    kept intact (mask selects everything positive) rather than destroyed.
    """
    if volume.size == 0 or not np.any(volume > 0):
        return volume.copy(), np.zeros(volume.shape, dtype=bool)

    threshold = otsu_threshold(volume)
    mask = volume > 0 if threshold <= 0 else volume >= threshold

    # Guard against a pathological threshold that keeps ~nothing.
    if mask.sum() < max(8, volume.size // 10000):
        positive = volume > 0
        if positive.any():
            mask = positive

    if morphology and mask.any() and not mask.all():
        mask = _largest_component_with_holes_filled(mask)

    masked = np.where(mask, volume, 0.0).astype(volume.dtype, copy=False)
    return masked, mask


def resample_isotropic(
    volume: np.ndarray,
    zooms: tuple[float, float, float],
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Resample a volume to isotropic spacing by scaling voxel counts.

    MONAI's shape-based `Resize` combined with a zoom-derived target shape
    is mathematically equivalent to physical-spacing resampling and avoids
    MetaTensor/affine bookkeeping for a standalone array pipeline.
    """
    new_shape = tuple(
        max(1, round(dim * zoom / target))
        for dim, zoom, target in zip(volume.shape, zooms, target_spacing)
    )
    resize = Resize(spatial_size=new_shape, mode="trilinear", align_corners=False)
    resized = resize(volume[np.newaxis, ...])  # add channel dim
    return np.asarray(resized[0])


def _zscore_normalize(volume: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero mean / unit variance over foreground voxels; background stays 0."""
    values = volume[mask] if mask.any() else volume.ravel()
    std = float(values.std())
    mean = float(values.mean())
    normalized = (volume - mean) / (std + eps)
    if mask.any() and not mask.all():
        normalized[~mask] = 0.0
    return normalized.astype(np.float32, copy=False)


def preprocess_mri_volume(
    volume: np.ndarray,
    zooms: tuple[float, float, float],
    cfg: MRIConfig = DEFAULT_CONFIG.mri,
) -> np.ndarray:
    """Full spatial standardization pipeline for an already-RAS volume.

    Returns a float32 array of shape (1, *cfg.target_shape). With the
    default ``normalization="minmax"`` values are scaled to [0, 1]; with
    ``"zscore"`` foreground voxels are standardized to ~N(0, 1) and the
    background is zeroed.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    isotropic = resample_isotropic(volume, zooms, cfg.target_spacing)

    if cfg.mask_method == "otsu":
        masked, mask = foreground_mask_otsu(isotropic, morphology=cfg.mask_morphology)
    elif cfg.mask_method == "percentile":
        masked = foreground_mask_threshold(isotropic, cfg.foreground_percentile)
        mask = masked > 0
    else:
        raise ValueError(
            f"Unknown mask_method {cfg.mask_method!r}; expected 'otsu' or 'percentile'"
        )

    if cfg.normalization == "zscore":
        normalized = _zscore_normalize(masked, mask)[np.newaxis, ...]
    elif cfg.normalization == "minmax":
        if masked.max() > masked.min():
            scaler = ScaleIntensity(minv=0.0, maxv=1.0)
            normalized = np.asarray(scaler(masked[np.newaxis, ...]))
        else:
            normalized = masked[np.newaxis, ...].astype(np.float32)
    else:
        raise ValueError(
            f"Unknown normalization {cfg.normalization!r}; expected 'minmax' or 'zscore'"
        )

    if cfg.fit == "bbox_resize" and mask.any():
        idx = np.argwhere(mask)
        lo = np.maximum(idx.min(axis=0) - 2, 0)
        hi = np.minimum(idx.max(axis=0) + 3, mask.shape)
        cropped = normalized[(slice(None),) + tuple(slice(a, b) for a, b in zip(lo, hi))]
        resize = Resize(spatial_size=cfg.target_shape, mode="trilinear", align_corners=False)
        standardized = np.asarray(resize(cropped))
    elif cfg.fit in ("crop", "bbox_resize"):
        pad_crop = ResizeWithPadOrCrop(spatial_size=cfg.target_shape)
        standardized = np.asarray(pad_crop(normalized))
    else:
        raise ValueError(f"Unknown fit {cfg.fit!r}; expected 'crop' or 'bbox_resize'")
    return standardized.astype(np.float32)


def preprocess_mri_file(path: str, cfg: MRIConfig = DEFAULT_CONFIG.mri) -> np.ndarray:
    """Convenience wrapper: load a NIfTI file and run full standardization."""
    data, _, zooms = load_nifti(path)
    return preprocess_mri_volume(data, zooms, cfg)
