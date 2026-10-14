"""Synthetic MRI + VEP generators.

The datasets named in Project.md (OpenNeuro, HCP Retinotopy, OASIS, MNE
sample, PhysioNet) provide structural/functional scans and generic
electrophysiology, but none carry the clinical Healthy/Glaucoma/Optic
Neuritis-CVI labels this model targets -- that requires a curated clinical
cohort. This module generates class-conditioned synthetic MRI volumes and
VEP waveforms with the same interface real data would have, so the rest of
the system (preprocessing, model, training, explainability) can be built,
exercised, and tested end-to-end while a labeled cohort is sourced
separately.

v2 realism upgrades (why: v1 samples were trivially separable -- every
sample of a shared the same lesion quadrant and every VEP had identical
class parameters, letting models score high on template matching rather
than on the biomarkers the project claims to detect):

- Anatomical MRI proxy: an elliptical brain with smoothed tissue texture,
  darker internal ventricles, and a bright rim -- so foreground masking and
  the encoder must actually work instead of exploiting global statistics.
- Lesions are placed at *random* locations/varying radii with per-sample
  contrast (glaucoma: focal hypointensity/atrophy proxy; optic neuritis /
  CVI: focal hyperintensity/lesion proxy). The model must localize and
  characterize pathology, not remember its position.
- VEP waveforms keep the class-conditioned N75-P100-N145 complex but with
  per-sample latency/amplitude jitter drawn from class-specific
  distributions (healthy ~100ms, glaucoma reduced-amplitude ~104ms,
  neuritis markedly delayed ~138ms), mimicking inter-subject variability.
"""

import csv
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from src.config import Config, DEFAULT_CONFIG

NUM_CLASSES = DEFAULT_CONFIG.model.num_classes

# Class-conditioned VEP parameter distributions: (latency_ms, latency_sd,
# amplitude_uV, amplitude_sd). Glaucoma keeps near-normal latency with
# reduced amplitude; optic neuritis / CVI shows the clinical hallmark --
# markedly delayed P100 (Odom et al., 2016 ISCEV standard).
VEP_CLASS_PARAMS = {
    0: {"latency_ms": 100.0, "latency_sd": 3.0, "amplitude": 6.0, "amplitude_sd": 0.8},
    1: {"latency_ms": 104.0, "latency_sd": 4.0, "amplitude": 2.5, "amplitude_sd": 0.6},
    2: {"latency_ms": 138.0, "latency_sd": 6.0, "amplitude": 4.0, "amplitude_sd": 1.0},
}


def synthetic_mri_volume(
    label: int, shape: tuple[int, int, int] = (80, 96, 64), rng: np.random.Generator | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Generate a class-conditioned synthetic 3D MRI-like volume.

    An elliptical "brain" sits in a noisy background; label 0 (Healthy) is
    lesion-free, label 1 (Glaucoma) carries a random focal hypointensity,
    label 2 (Optic Neuritis / CVI) a random focal hyperintensity.

    ``strength`` in [0, 1] scales the lesion contrast (1 = default; 0 = no
    visible class signal), which lets experiments control how informative
    the MRI modality is independently of the VEP.
    """
    rng = rng or np.random.default_rng()
    mri_strength = float(strength)
    d, h, w = shape

    # --- background (air/skull): low-intensity correlated noise ---
    volume = rng.normal(loc=6.0, scale=3.0, size=shape)
    volume = gaussian_filter(volume, sigma=1.5)

    # --- brain: random ellipsoid center/radii (anatomical variability) ---
    center = np.array([d / 2, h / 2, w / 2]) + rng.uniform(-3, 3, size=3)
    radii = np.array([d, h, w]) * rng.uniform(0.33, 0.40, size=3)

    grids = np.ogrid[:d, :h, :w]
    dist = np.sqrt(
        sum(((g - c) / r) ** 2 for g, c, r in zip(grids, center, radii))
    )
    brain = dist <= 1.0
    rim = (dist > 0.82) & brain  # bright cortex-like shell

    tissue = rng.normal(loc=120.0, scale=8.0, size=shape)
    tissue = gaussian_filter(tissue, sigma=2.0)  # smooth, texture-like
    volume[brain] = tissue[brain] + rng.normal(0.0, 6.0, size=int(brain.sum()))
    volume[rim] += 35.0
    volume = np.clip(volume, 0.0, None)

    # --- ventricles: dark internal cavities ---
    vcenter = center + rng.uniform(-2, 2, size=3)
    vradii = radii * rng.uniform(0.10, 0.16, size=3)
    vdist = np.sqrt(sum(((g - c) / r) ** 2 for g, c, r in zip(grids, vcenter, vradii)))
    volume[vdist <= 1.0] *= 0.25

    # --- class-conditioned focal lesions at randomized locations ---
    if label in (1, 2):
        # Place the lesion inside the brain, away from its rim: sample a
        # direction and a radius no larger than 0.6 of the local extent.
        offset = rng.normal(size=3)
        offset /= np.linalg.norm(offset)
        lesion_center = center + offset * radii * rng.uniform(0.15, 0.55)
        lesion_radius = float(rng.integers(3, 7))
        ldist = np.sqrt(
            sum(((g - c) / lesion_radius) ** 2 for g, c in zip(grids, lesion_center))
        )
        sphere = (ldist <= 1.0) & brain
        falloff = np.exp(-(ldist**2) * 2.0)

        contrast = rng.uniform(0.35, 0.55) if label == 1 else rng.uniform(1.6, 2.2)
        # Blended contrast keeps edges soft and MRI-plausible.
        factor = 1.0 + (contrast - 1.0) * falloff * mri_strength
        volume = np.where(sphere, volume * factor, volume)

    return volume.astype(np.float32)


def synthetic_vep_signal(
    label: int, fs: float = 250.0, duration_s: float = 2.0, rng: np.random.Generator | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Generate a class-conditioned synthetic VEP waveform (N75-P100-N145).

    Per-sample latency and amplitude are drawn from class-specific
    distributions (see ``VEP_CLASS_PARAMS``); physiological artefacts
    (baseline drift, 50 Hz mains hum, EMG noise) are added on top.
    """
    rng = rng or np.random.default_rng()
    params = dict(VEP_CLASS_PARAMS[label])
    if strength != 1.0:
        # Shrink the class offsets (latency, amplitude) towards the healthy
        # class: strength 0 makes every class look healthy on the VEP.
        healthy = VEP_CLASS_PARAMS[0]
        for k in ("latency_ms", "amplitude"):
            params[k] = healthy[k] + strength * (params[k] - healthy[k])
    latency = max(
        0.05, rng.normal(params["latency_ms"] / 1000.0, params["latency_sd"] / 1000.0)
    )
    # Amplitude floor: a real clinical P100 is only measurable on *averaged*
    # sweeps; for single-sweep synthesis keep enough amplitude that the true
    # peak survives EMG noise after z-scoring (SNR >> noise spikes), or
    # downstream P100-latency estimation becomes unstable. Class amplitude
    # RATIOS are preserved (z-normalization cancels absolute scale).
    amplitude = max(1.5, rng.normal(params["amplitude"], params["amplitude_sd"]))

    t = np.arange(0.0, duration_s, 1.0 / fs)

    p100 = amplitude * np.exp(-((t - latency) ** 2) / (2 * 0.012**2))
    n75 = -0.5 * amplitude * np.exp(-((t - (latency - 0.025)) ** 2) / (2 * 0.010**2))
    n145 = -0.4 * amplitude * np.exp(-((t - (latency + 0.045)) ** 2) / (2 * 0.015**2))
    waveform = p100 + n75 + n145

    drift = 0.3 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
    mains = 0.15 * np.sin(2 * np.pi * 50.0 * t + rng.uniform(0, 2 * np.pi))
    emg_noise = rng.normal(0.0, 0.3, size=t.shape)

    return (waveform + drift + mains + emg_noise).astype(np.float64)


def write_synthetic_dataset(
    root: str,
    samples_per_class: int = 4,
    cfg: Config = DEFAULT_CONFIG,
    seed: int = 0,
) -> str:
    """Write synthetic NIfTI + VEP CSV files and a manifest.csv under `root`.

    Mirrors the data/raw/{mri,vep} layout described in Project.md. Returns
    the manifest CSV path, with columns: subject_id, mri_path, vep_path, fs, label.
    """
    rng = np.random.default_rng(seed)
    mri_dir = os.path.join(root, "mri")
    vep_dir = os.path.join(root, "vep")
    os.makedirs(mri_dir, exist_ok=True)
    os.makedirs(vep_dir, exist_ok=True)

    manifest_path = os.path.join(root, "manifest.csv")
    rows = []
    subject_idx = 0
    for label in range(NUM_CLASSES):
        for _ in range(samples_per_class):
            subject_id = f"sub-{subject_idx:04d}"
            volume = synthetic_mri_volume(label, rng=rng)
            affine = np.diag([1.0, 1.0, 1.0, 1.0])
            img = nib.Nifti1Image(volume, affine)
            mri_path = os.path.join(mri_dir, f"{subject_id}_mri.nii.gz")
            nib.save(img, mri_path)

            signal = synthetic_vep_signal(label, fs=cfg.vep.sample_rate, rng=rng)
            vep_path = os.path.join(vep_dir, f"{subject_id}_vep.csv")
            with open(vep_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["amplitude"])
                for value in signal:
                    writer.writerow([value])

            rows.append(
                {
                    "subject_id": subject_id,
                    "mri_path": mri_path,
                    "vep_path": vep_path,
                    "fs": cfg.vep.sample_rate,
                    "label": label,
                }
            )
            subject_idx += 1

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "mri_path", "vep_path", "fs", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path
