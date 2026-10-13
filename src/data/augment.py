"""Train-time augmentation transforms for MRI volumes and VEP waveforms.

Design constraint: augmentation must *never* destroy the class-defining
pathology markers.

- MRI ops (all shape-preserving, all deterministic given the RNG):
  flips, axial 90-degree rotations, intensity scale/shift, Gaussian noise.
- VEP ops: amplitude scaling, additive noise, and a *tiny* bounded,
  edge-padded time shift (<= max_shift_samples, default 1). The waveform is
  never rolled (wrapping would move an edge marker to the other end) and
  never shifted more than a sample -- P100 latency is the clinical biomarker
  separating optic neuritis from glaucoma.

Every primitive accepts a `numpy.random.Generator` and is therefore fully
reproducible. `enabled=False` bypasses everything and returns the input
object untouched (identity, not a copy).
"""

import numpy as np


def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x, False
    return np.asarray(x), True  # torch tensors & friends -> numpy view


def _wrap_like(original, out):
    if isinstance(original, np.ndarray):
        return out
    import torch

    return torch.from_numpy(out)


def random_flip_3d(volume: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Randomly flip each leading axis independently with probability `p`."""
    out = volume
    for axis in range(out.ndim):
        if rng.random() < p:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out).astype(np.float32, copy=False)


def random_rotate90_axial(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate the last two (in-plane) axes by a random multiple of 90 degrees."""
    k = int(rng.integers(0, 4))
    if k == 0:
        return np.ascontiguousarray(volume).astype(np.float32, copy=False)
    out = np.rot90(volume, k=k, axes=(-2, -1))
    return np.ascontiguousarray(out).astype(np.float32, copy=False)


def random_intensity_scale_shift(
    volume: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.95, 1.05),
    shift_range: float = 0.02,
) -> np.ndarray:
    """Multiplicative gain + additive offset, clipped back to [0, 1]."""
    out = volume.astype(np.float32, copy=True)
    scale = float(rng.uniform(*scale_range))
    shift = float(rng.uniform(-shift_range, shift_range))
    out = out * scale + shift
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def random_gaussian_noise(
    volume: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.01,
    p: float = 1.0,
) -> np.ndarray:
    """Additive Gaussian scanner noise, applied with probability `p`."""
    if p <= 0 or sigma <= 0 or rng.random() >= p:
        return volume
    out = volume.astype(np.float32, copy=True)
    out = out + rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
    return out.astype(np.float32)


def augment_mri(
    volume,
    rng: np.random.Generator,
    enabled: bool = True,
    flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    scale_range: tuple[float, float] = (0.95, 1.05),
    shift_range: float = 0.02,
    noise_sigma: float = 0.01,
) -> np.ndarray:
    """Full MRI augmentation stack. Returns the input object when disabled."""
    if not enabled:
        return volume
    data, _ = _as_numpy(volume)
    out = data
    out = random_flip_3d(out, rng, p=flip_prob)
    if rng.random() < rotate_prob:
        out = random_rotate90_axial(out, rng)
    out = random_intensity_scale_shift(out, rng, scale_range=scale_range, shift_range=shift_range)
    out = random_gaussian_noise(out, rng, sigma=noise_sigma, p=1.0)
    return _wrap_like(volume, out)


def augment_vep(
    signal,
    rng: np.random.Generator,
    enabled: bool = True,
    amp_scale_range: tuple[float, float] = (0.9, 1.1),
    noise_sigma: float = 0.02,
    max_shift_samples: int = 1,
) -> np.ndarray:
    """VEP augmentation: amplitude scale, bounded edge-padded shift, noise.

    The shift is applied with probability 0.5 and bounded to
    `max_shift_samples` (<= ~4 ms at 250 Hz) so class-defining P100 latency
    differences (tens of ms) are never shuffled across classes.
    """
    if not enabled:
        return signal
    data, _ = _as_numpy(signal)
    out = data.astype(np.float32, copy=True)

    out = out * float(rng.uniform(*amp_scale_range))

    if max_shift_samples > 0 and rng.random() < 0.5:
        shift = int(rng.integers(-max_shift_samples, max_shift_samples + 1))
        if shift:
            if shift > 0:  # shift right: pad the head with the first sample
                out = np.pad(out, (shift, 0), mode="edge")[..., : out.shape[-1]]
            else:          # shift left: pad the tail with the last sample
                s = -shift
                out = np.pad(out, (0, s), mode="edge")[..., s:]

    if noise_sigma > 0:
        out = out + rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32)
    return _wrap_like(signal, np.ascontiguousarray(out).astype(np.float32))


def augment_sample(mri, vep, rng: np.random.Generator, **kwargs):
    """Augment both modalities of one sample with a shared RNG stream."""
    return augment_mri(mri, rng), augment_vep(vep, rng)
