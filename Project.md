# Project Blueprint: A Cross-Attention Deep Learning Framework for Multimodal Neuroimaging Fusion in Vision Defect Detection and Classification

---

## 1. Problem Statement

Automated detection and classification of vision defects (such as Optic Neuritis, Glaucoma, and Cortical Visual Impairments) remain limited in clinical practice due to three primary bottlenecks:

- **Unimodal Isolation:** Clinical diagnosis requires assessing both anatomical structure and electrical transmission latency. Current computational pipelines analyze either 3D structural MRI (sMRI) or 1D electrophysiology (Visual Evoked Potentials - VEP) in isolation. sMRI captures spatial lesions and structural atrophy but lacks millisecond temporal latency. VEP captures conduction delays along the optic nerve but lacks localized spatial resolution.
- **Cross-Modal Heterogeneity:** Merging a high-dimensional 3D volumetric matrix $(1, D, H, W)$ with a 1D temporal waveform $(1, L)$ introduces severe spatial-temporal misalignment. Standard fusion approaches (such as early feature concatenation or late average pooling) treat all feature dimensions equally. This introduces substantial feature noise, overlooks spatial-temporal cross-correlations, and dilutes subtle pathology markers.
- **Lack of Clinical Interpretability:** Conventional deep neural networks operate as black-box decision systems. They do not demonstrate to clinicians _which_ anatomical visual cortex zones (V1, V2, V3) correlate directly with specific millisecond signal delays (such as the $P100$ latency peak).

---

## 2. Proposed Solution

The proposed solution implements an end-to-end **Cross-Attention Fusion Network (CAFN)** designed to bridge 3D structural neuroimaging and 1D electrophysiological latency:

- **Dual-Stream Signal Conditioning:**
- **1D Temporal Stream:** Zero-phase 4th-order Butterworth bandpass filtering ($1.0\text{ Hz} - 100.0\text{ Hz}$) combined with a $50\text{ Hz}$ IIR notch filter to remove electromyographic (EMG) jitter, baseline drift, and AC power-line hum without altering critical $P100$ peak timing.
- **3D Spatial Stream:** Automated NIfTI skull-stripping, RAS orientation alignment, $1.0\text{ mm}$ isotropic voxel resampling, and foreground intensity normalization.

- **Cross-Attention Fusion Module (CAFM):** Rather than concatenating features, spatial 3D volume tokens are projected as Queries ($Q$), while temporal 1D latency tokens are projected as Keys ($K$) and Values ($V$). This enables spatial brain regions to dynamically query electrophysiological latency bins:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

**v2/v3 architecture & methodology upgrades (all active by default):**

- **Bidirectional gated co-attention:** temporal tokens also query spatial
  context (latency bins ask *which cortical zone* drives them), and each
  direction is scaled by a learnable sigmoid gate; `"cross"` mode recovers
  the original one-way fusion.
- **Positional information:** learned 3D positional embeddings on the pooled
  spatial grid and fixed sinusoidal encodings on the temporal axis -- without
  them the cross-attention heatmap is a bag-of-features correlation, not a
  region-to-latency map.
- **Backbone upgrades:** Squeeze-and-Excitation channel attention in the 3D
  residual blocks; a multi-scale 1D temporal encoder (parallel 3/9/19-sample
  kernels per stage) so both the ~30 ms P100 peak and the ~80 ms N75-P100-N145
  complex are visible to the encoder.
- **Learned attention pooling** for the fused classification embedding
  (initialized near-uniform, i.e. it starts at v1 mean-pool behavior), plus
  per-head token salience for inspection.
- **Otsu foreground masking** (histogram-optimal, with morphological hole
  filling / largest-component cleanup) instead of the destructive v1
  intensity-percentile "skull strip"; optional foreground z-score
  normalization for real MRI intensity ranges.
- **Multi-task loss with options:** fixed-weight fused+auxiliary CE
  (label smoothing, inverse-frequency class weights for imbalanced cohorts)
  or Kendall-style learnable homoscedastic task weighting
  (`src/models/losses.UncertaintyWeightedLoss`).
- **Train-time augmentation that respects the biomarkers:** MRI flips and
  intensity jitter; VEP amplitude/noise jitter only -- the waveform is never
  shifted in time because P100 latency carries the class signal.
- **Leakage-safe evaluation splits:** class-stratified row splits by
  default, plus `--group-by-subject` whole-subject splitting for
  longitudinal cohorts (several scans per subject) so validation accuracy
  measures generalization, not subject memorization.
- **Deterministic data:** synthetic samples are keyed by `(seed, index)`, so
  the same index yields identical content across epochs, workers, and runs.

- **Multi-Task Auxiliary Supervision:** Unimodal baseline heads train concurrently with the fused attention head, preventing the network from over-relying on a single dominant modality.
- **Clinical Interpretability Engine:** Generates 2D spatial-temporal cross-attention matrices and 3D visual cortex activation maps, giving clinicians verifiable evidence for classification decisions.

---

## 3. System Architecture & Data Flow

```
========================================================================================
                                PIPELINE FLOW
========================================================================================

   [ 3D sMRI Volume ]                           [ 1D VEP Waveform ]
  (Raw NIfTI: .nii.gz)                          (Raw Electrophysiology)
           │                                               │
           ▼                                               ▼
 [ MONAI Preprocessing ]                        [ Butterworth & Notch ]
   - RAS Reorientation                            - 50 Hz Notch (Mains)
   - 1mm Isotropic Resampling                     - 1-100 Hz Bandpass
   - Intensity Scale [0, 1]                       - Z-Score Normalization
           │                                               │
     (1, 64, 64, 64)                                    (1, 500)
           │                                               │
           ▼                                               ▼
  [ 3D-CNN / ResNet ]                            [ 1D-CNN Latency Net ]
   - 3D Convolutions                              - 1D Temporal Convolutions
   - Adaptive Pool (4x4x4)                        - Adaptive Pool (16)
           │                                               │
    Spatial Tokens (F_s)                           Temporal Tokens (F_t)
       (B, 64, 128)                                   (B, 16, 128)
           │                                               │
           ├──────────────────────────┐                    │
           ▼                          ▼                    ▼
     [ Linear: W_Q ]            [ Linear: W_K ]      [ Linear: W_V ]
           │                                  \      /
       Query (Q)                               Key (K), Value (V)
           │                                      │
           └─────────────────┬────────────────────┘
                             ▼
              [ Multi-Head Cross-Attention ]
               Score = Softmax( Q * K^T / sqrt(d_k) )
               Fused_Tokens = Score * V + Q (Residual)
               LayerNorm & Feed-Forward Block
                             │
                     Fused Tensor (B, 64, 128)
                             │
                             ▼
                    [ Global Mean Pool ]
                             │
                     Embedding (B, 128)
                             │
                             ▼
              [ Softmax Classification Head ]
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
      Healthy Control   Glaucoma     Optic Neuritis / CVI

```

---

## 4. Dataset Repositories & Access Links

| Modality           | Dataset Name                                     | Description / Utility                                                                                     | Access Link                                                                                                           |
| ------------------ | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **3D sMRI / fMRI** | **OpenNeuro (ds004496)**                         | High-resolution visual cortex mapping under retinotopic stimulation paradigms.                            | [OpenNeuro ds004496](https://www.google.com/search?q=https://openneuro.org/datasets/ds004496)                         |
| **3D sMRI / fMRI** | **OpenNeuro (ds003787)**                         | Multimodal structural and functional visual pathway stimulation benchmarks.                               | [OpenNeuro ds003787](https://openneuro.org/datasets/ds003787)                                                         |
| **3D sMRI / fMRI** | **Human Connectome Project (HCP) 7T Retinotopy** | Gold-standard 7T structural and retinotopic functional scans of cortical visual areas (V1, V2, V3).       | [HCP Retinotopy Benchmark](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) |
| **3D sMRI**        | **OASIS Brains Database**                        | Large-scale public neuroimaging data for structural normalization and gray matter segmentation baselines. | [OASIS Brains Portal](https://www.oasis-brains.org/)                                                                  |
| **1D VEP / EEG**   | **MNE Sample Electrophysiology**                 | Visual and auditory stimulus responses across occipital sensor lines (e.g., channel EEG 057).             | [MNE Sample Dataset](https://www.google.com/search?q=https://mne.tools/stable/overview/datasets_index.html%23sample)  |
| **1D VEP / ERG**   | **PhysioNet Clinical Electrophysiology**         | Standardized clinical wave files following ISCEV standards for optic pathway conduction analysis.         | [PhysioNet Repositories](https://physionet.org/)                                                                      |

> **Labeling gap:** none of the repositories above ship the clinical
> Healthy / Glaucoma / Optic Neuritis-CVI diagnosis labels this model
> targets -- they provide structural/retinotopic-functional scans and
> generic electrophysiology. A labeled clinical cohort (or a clinician
> partner to label a subset of the above) is still required before real
> training can happen. Until then, `src/data/synthetic.py` generates
> class-conditioned synthetic MRI volumes and VEP waveforms through the
> same interface, so the full pipeline -- preprocessing, model, training,
> explainability -- is built, exercised, and tested end-to-end. Point
> `PairedNeuroDataset` at a real manifest CSV (`subject_id, mri_path,
> vep_path, fs, label`) as soon as labeled data exists; no other code
> changes are required.

---

## 5. Base Papers & Key References

- **Vaswani, A., et al. (2017).** _Attention Is All You Need._ Advances in Neural Information Processing Systems (NeurIPS).
  _Foundational paper introducing the Query-Key-Value Multi-Head Attention and Transformer architecture._
  Link: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Calhoun, V. D., & Sui, J. (2016).** _Multimodal Fusion of Brain Imaging Data: A Key to Finding the Missing Link(s) in Complex Mental Illness._ Biological Psychiatry: Cognitive Neuroscience and Neuroimaging, 1(3), 230–244.
  _Defines foundational principles for cross-modal neuroimaging integration and information complementarity._
  Link: [DOI: 10.1016/j.bpsc.2015.12.005](https://www.google.com/search?q=https://doi.org/10.1016/j.bpsc.2015.12.005)
- **Zhang, Y., et al. (2024).** _CAMF: Cross-Attention Multimodal Fusion for Neuroimaging-Based Diagnosis of Neurological Disorders._ IEEE Transactions on Medical Imaging.
  _Direct benchmark for cross-attention mechanisms applied to heterogeneous medical imaging modalities._
  Link: [IEEE Xplore / PubMed: 38416629](https://www.google.com/search?q=https://pubmed.ncbi.nlm.nih.gov/38416629/)
- **Hao, R., et al. (2023).** _Cross-Spatial Attention Guided Network for Multimodal Brain Tumor Segmentation and Structural Feature Linking._ Computerized Medical Imaging and Graphics, 108, 102263.
  _Formulates cross-attention modules targeting 3D spatial feature localization._
  Link: [DOI: 10.1016/j.compmedimag.2023.102263](https://www.google.com/search?q=https://doi.org/10.1016/j.compmedimag.2023.102263)
- **Guo, Z., et al. (2025).** _DRIFA-Net: Deformable Cross-Attention and Iterative Feature Alignment for Multimodal 3D Medical Volume Fusion._ IEEE Winter Conference on Applications of Computer Vision (WACV).
  _State-of-the-art methodology for 3D alignment and cross-attention feature aggregation._
  Link: [WACV 2025 Open Access / arXiv:2411.08324](https://arxiv.org/abs/2411.08324)
- **Odom, J. V., et al. (2016).** _ISCEV Standard for Clinical Visual Evoked Potentials (2016 Edition)._ Documenta Ophthalmologica, 133(1), 1–9.
  _Clinical protocol defining standard VEP waveform biomarkers, calibration, and P100 peak latency metrics._
  Link: [DOI: 10.1007/s10633-016-9553-y](https://www.google.com/search?q=https://doi.org/10.1007/s10633-016-9553-y)

---

## 6. Implementation Summary & Deliverables

```
├── data/
│   ├── raw/
│   │   ├── mri/                     # Raw 3D NIfTI volumes (.nii.gz)
│   │   └── vep/                     # Raw 1D electrophysiological CSVs (.csv/.npy)
│   └── processed/                   # Resampled, masked, and normalized arrays
├── src/
│   ├── config.py                    # Central shape/hyperparameter config (single source of truth)
│   ├── utils.py                     # Seeding, device resolution, stratified & subject-group splits
│   ├── train.py                     # Multi-task training loop (CLI + library entry points)
│   ├── predict.py                   # Single-pair inference CLI (prediction + evidence artifacts)
│   ├── preprocessing/
│   │   ├── signal_cleaner.py        # Butterworth & notch filtering + baseline correction
│   │   └── mri_transforms.py        # MONAI spatial standardizer (Otsu/percentile masks)
│   ├── data/
│   │   ├── synthetic.py             # Class-conditioned synthetic MRI/VEP generator (see Sec. 4 note)
│   │   ├── augment.py               # Biomarker-preserving train-time augmentation
│   │   └── dataset.py               # PairedNeuroDataset (validated manifest) & SyntheticNeuroDataset
│   ├── models/
│   │   ├── spatial_encoder.py       # 3D ResNet + Squeeze-Excitation backbone
│   │   ├── temporal_encoder.py      # Multi-scale 1D residual latency backbone
│   │   ├── positional.py            # 3D learned / 1D sinusoidal positional encodings
│   │   ├── attention_pool.py        # Learned attention pooling head
│   │   ├── cross_attention.py       # Multi-Head CAFM (pre-norm, stacked, gated bidirectional)
│   │   ├── losses.py                # Multi-task CE (smoothing/class weights) + uncertainty weighting
│   │   └── cafn.py                  # Full CAFN: encoders + fusion + fused/aux heads
│   └── explainability/
│       ├── grad_cam_3d.py           # 3D spatial activation maps (Grad-CAM on fused logit)
│       └── plot_attention.py        # 2D cross-attention correlation heatmaps (both directions)
├── tests/
│   ├── unit/                        # Per-module tests (preprocessing, models, data, explainability)
│   └── integration/                 # Full raw-file-to-trained-step pipeline & training-loop tests
├── checkpoints/                     # Serialized PyTorch models (.pt); per-run artifact dirs
├── pytest.ini                       # pythonpath + test discovery config
├── requirements.txt                 # torch, monai, nibabel, nilearn, mne, scipy,
│                                    # matplotlib, scikit-learn, pytest, pytest-cov
└── Project.md                       # This blueprint
```

**Typical runs:**

```
python -m src.train --synthetic-samples 64 --epochs 20                     # synthetic end-to-end
python -m src.train --manifest data/raw/manifest.csv --group-by-subject \
    --augment --balance-classes --run-name cohort1                         # real cohort (when labeled)
python -m src.train --test-split 0.15 --run-name cohort1                   # + stratified hold-out test set
python -m src.predict --checkpoint checkpoints/cohort1/best.pt \
    --mri subject.nii.gz --vep subject_vep.csv --outdir out/               # inference + evidence
```

**Evaluation-v2 layer (CLI + module):**

- `src/evaluation.py` computes balanced accuracy, macro/weighted F1,
  per-class precision/recall, ROC-AUC (one-vs-rest) and renders labelled
  confusion-matrix PNGs; every training run finishing with a val/test
  split writes `final_<split>_confusion_matrix.png` +
  `final_<split>_metrics.json` alongside the standard artifacts, and
  `--eval-only checkpoints/<run>/best.pt` replays the same report on a
  fresh process without training.
- `src/data/splits.py` is the reusable splitter behind the CLI:
  label-stratified with largest-remainder quotas (minority classes are
  never rounded out of validation), subject-group aware, and
  `Subset`-transparent; it powers `--test-split` under all stratify/
  group flags. `python -m src.train --eval-only ...` reports the exact
  figures quoted in reports — no manual metric juggling.

**Design deviations from the original blueprint, and why:**

- **Skull-stripping** is implemented as a self-contained foreground mask
  rather than an external tool (FSL BET / HD-BET / SynthStrip), since those
  require a separate binary or pretrained network not installable via pip.
  Default is an **Otsu threshold with morphological cleanup**
  (`mri_transforms.foreground_mask_otsu`: hole filling + largest connected
  component), which preserves far more genuine tissue than the legacy
  intensity-percentile mask (`foreground_mask_threshold`, kept behind
  ``MRIConfig.mask_method="percentile"`` for reproducing v1 experiments).
  Both are drop-in placeholders -- swap in HD-BET/SynthStrip for
  clinical-grade deployment; nothing downstream depends on which method
  produced the mask.
- **`src/config.py`** was added (not in the original tree) as the single
  source of truth for tensor shapes and hyperparameters (`d_model=128`,
  `num_heads=4`, 64 spatial / 16 temporal tokens, etc.), so the
  preprocessing stream, both encoders, fusion module, and training loop
  cannot silently disagree on shapes.
- **`src/data/`** was added (not in the original tree) to hold the dataset
  loaders and the synthetic-data generator required to close the labeling
  gap noted in Section 4.
- **v2/v3 additions** beyond the original scope: `src/models/positional.py`,
  `src/models/attention_pool.py`, `src/models/losses.py`,
  `src/data/augment.py`, `src/utils.py` (splits/seeding/device), and
  `src/predict.py` (inference CLI). Training artifacts (per-run `best.pt`,
  `last.pt`, `history.json`, `final_metrics.json`, `training_curves.png`)
  land under `checkpoints/<run_name>/`.
- **Test suite**: 300 tests (`pytest.ini`, `tests/unit/`,
  `tests/integration/`) cover every module in isolation plus full-system
  integration tests -- raw synthetic files through preprocessing, model,
  loss, backward, and optimizer step; a multi-epoch training run asserting
  the loss actually drops and the model separates classes; checkpoint
  round-trips, resume, early stopping, and train-artifact generation; CLI
  end-to-end runs (`--test-split`, `--eval-only`, manifest mode) in temp
  dirs. Run with `pytest` (or `pytest --cov=src` for coverage; currently
  ~92%).
- **MPS note**: recent torch builds do not implement
  `adaptive_avg_pool3d` on Apple Silicon's MPS backend, so training
  auto-falls back to CPU with a printed warning when the spatial encoder's
  pool is hit; behaviour and results are identical.

---

## v2/v3 engineering notes (fusion, losses, explainability)

Companion details to the sections above; quickstart lives in `README.md`.

**Model / fusion**

- `AttentionPool1D` (`src/models/attention_pool.py`): the scorer is an
  `nn.Linear(d_model, 1)` with near-zero init, so the pooled head begins
  exactly at mean-pooling and learns to sharpen; zeroing the scorer
  recovers mean pooling bit-for-bit. It also emits per-token salience
  weights (`pool_weights`) usable as a training-free token importance map.
- `PositionalEmbedding` (`src/models/positional.py`) is a plain learned
  table `(num_tokens, d_model)`; the spatial wrapper validates that the
  pooling grid product equals `spatial_tokens` so the two can never
  silently disagree.
- The gates on the bidirectional fusion live on the fusion module itself
  (`fusion.gate_spatial` / `fusion.gate_temporal`), zero-initialized:
  `sigmoid(0) = 0.5` means both cross-modal directions start exactly
  half-open.
- Fusion depth is honored by both fusion modes (`ModelConfig.fusion_blocks`);
  the fusion layer returns the last block's attention maps, and in the
  bidirectional mode the temporal stream updates *before* the spatial read,
  so the fused representation has consumed the enriched latencies within a
  single block (this also guarantees gradients reach every fusion
  parameter through a single-layer stack).

**Losses** (`src/models/losses.py`)

- `multi_task_ce` is the canonical fused + auxiliary CE, supporting label
  smoothing and optional class weights threaded into every head.
- `UncertaintyWeightedLoss` implements learnable homoscedastic task
  weighting (Kendall et al., 2018): each task scales by
  `exp(-log_sigma)` with an additive `+log_sigma` regularizer that provably
  prevents the degenerate "inflate every sigma" solution. At init
  (`log_sigma = 0`) it reduces exactly to the plain sum of the three CEs.
  Wire it by constructing `UncertaintyWeightedLoss()` and calling it
  instead of `model.multi_task_loss` in custom loops.

**Explainability** (`src/explainability/`)

- `GradCAM3D(method="gradcam++")`: second-order pixel re-weighting
  (Selvaraju et al., 2017) for classes that activate multiple regions --
  exactly the lesion-plus-atrophy picture here. Differentiation targets the
  captured (non-leaf) activation tensor directly, so no leaf-grad juggling;
  `generate_with_fallback()` degrades gracefully to vanilla Grad-CAM on
  backends without double-backward support.
- `plot_cross_attention` accepts either the raw `(B, H, S, T)` tensor or
  CAFN's direction-keyed mapping, and `plot_attention_by_head` renders one
  heatmap per head -- heads specialize, and cross-head pooling hides it.
- `p100.py` turns the VEP stream into a clinician-checkable number: P100
  latency/amplitude with an ISCEV-aligned search window, plus
  `estimate_p100_after_cleaning`, which corrects the effective sample rate
  whenever the cleaning pipeline resampled the sweep (a naive estimate
  otherwise reports 2x latencies for 500 Hz sources).

**Checkpoint contract**

- `save_checkpoint` now prefers the model's own `cfg` (`getattr(model,
  "cfg", cfg.model)`) when serializing: saving a model built from a custom
  config with a default cfg silently produced checkpoints that reloaded
  into mismatched architectures (the predict-CLI regression). Legacy
  checkpoints still load via `merge_model_config`, which maps renamed
  fields forward (`bidirectional -> fusion`,
  `use_positional_encoding -> pos_encoding`, `num_fusion_layers ->
  fusion_blocks`) and drops unknown keys.


---

## Real-data multi-cohort pipeline (v4)

`./run_all.sh` runs download -> preprocess -> train -> validate -> test end to end (see `README.md`).
Cohorts: OpenNeuro ds003568 (MDD vs healthy, T1w + resting MEG), ds007181 (zoster neuralgia vs healthy, T1w +
PSG EEG), optional ds008894. The 1-D stream is the channel-averaged 1-45 Hz log-PSD (`VEPConfig.signal_mode="spectrum"`),
MRI is Otsu-masked, z-scored and resized to 64^3 from the foreground bounding box (`MRIConfig.fit="bbox_resize"`;
a centre crop of a whole-head scan sees only a sliver). Protocol: per-cohort 70/15/15 splits, a pooled model reported
per cohort, and leave-one-cohort-out; checkpoints are selected on validation balanced accuracy
(`TrainConfig.select_metric`), which fixes short runs restoring an untrained epoch 1.

Other changes made while validating on a Linux/CUDA machine:

- `src/models/pooling.py`: deterministic adaptive average pooling (per-axis matmul) replaces
  `nn.AdaptiveAvgPool{1,3}d`, whose CUDA backward is non-deterministic and which is unimplemented on MPS.
  Outputs match the PyTorch op; no parameters, so existing checkpoints load unchanged.
- `src/utils.py`: `from __future__ import annotations` -- an un-imported `Subset` annotation raised `NameError` on
  Python 3.10-3.13 (it only worked on 3.14, which defers annotation evaluation).
- `--eval-only` moved the loaded checkpoint to the compute device (it silently skipped the detailed report on CUDA).
- `test_class_weights_change_loss` used all-identical labels, for which a weighted mean is mathematically unchanged.

Known limitations: the label "patient" means a different disorder in each cohort; cohorts differ in sensor
(MEG vs EEG) so a cohort-identity shortcut is possible (the summary reports a probe for it); skull-stripping is
still an Otsu foreground mask rather than HD-BET/SynthStrip.
