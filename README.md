# NeuroImaging — CAFN v2

A Cross-Attention Fusion Network (CAFN) that classifies vision defects
(Healthy Control / Glaucoma / Optic Neuritis–CVI) by fusing **3D structural
MRI** volumes with **1D visual evoked potential (VEP)** waveforms. The full
project blueprint lives in [`Project.md`](Project.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt

pytest                      # full test suite (unit + integration)
python -m src.train --epochs 20 --synthetic-samples 128 --run-name base
python -m src.train --epochs 30 --manifest data/raw/manifest.csv --run-name real
python -m src.train --eval-only checkpoints/<run>/best.pt --synthetic-samples 64

# One-sample prediction + explainability bundle (JSON + gradcam/attention PNGs)
python -c "from src.predict import run; run('checkpoints/<run>/best.pt', 'mri.nii.gz', 'vep.csv', outdir='out')"
```

## Architecture (v2)

```
MRI (B,1,D,H,W) ──► 3D ResNet + Squeeze-Excitation ──► spatial tokens (B, 64, d)
                                     │                     + learned 3D positional embedding
VEP (B,1,L)     ──► multi-scale 1D CNN ──► temporal tokens (B, 16, d)
                                     │                     + sinusoidal positional embedding
                                     ▼
        Gated BIDIRECTIONAL co-attention (fusion_blocks deep, pre-norm):
          t' = t + ½·CA(t→s),  s' = s + ½·CA(s→t')     # both streams read each other
                                     ▼
        learned attention pooling ──► fused head
        + unimodal spatial/temporal auxiliary heads (multi-task)
```

- **Bidirectional fusion** — in v1 only spatial tokens could query latency
  bins; temporal tokens now also query cortex regions, with zero-init
  learned gates (`sigmoid(0)=0.5`) letting each direction open or close.
- **Attention pooling** replaces mean pooling (zero-init scorer ⇒ exactly
  mean pooling at start of training) and emits token-salience weights.
- **Grad-CAM++** joins vanilla Grad-CAM for the 3D spatial stream, with an
  automatic fallback when a backend cannot do the higher-order backward.
- **Per-head attention plots** — heads specialize; pooling across heads
  hides it (see `src/explainability/plot_attention.py`).

## Training pipeline

AdamW + linear-warmup/cosine LR, gradient clipping, stratified train/val
split (class-balanced, with a degenerate-cohort guard), optional class
weighting, label smoothing, early stopping with best-weight restore, AMP on
CUDA, automatic device selection (CUDA → MPS → CPU with a fallback shim for
MPS-unsupported ops such as `adaptive_avg_pool3d`), rich evaluation
(balanced accuracy / macro-F1 / confusion matrix), per-run artifacts
(`best.pt`, `last.pt`, `history.json`, `final_metrics.json`,
`training_curves.png`), and resume support.

## Data

No data is downloaded. `SyntheticNeuroDataset` generates class-conditioned
MRI/VEP pairs (atrophy proxies, lesion proxies, P100 latency shifts) through
the exact preprocessing pipeline real data will use. Point
`PairedNeuroDataset` at a manifest CSV (`subject_id, mri_path, vep_path, fs,
label`) when labeled data exists — no other code changes required.

Preprocessing: RAS reorientation, isotropic resampling, Otsu **or**
percentile foreground masking (with connected-component cleanup), min-max
**or** foreground-z-score normalization; VEP: optional baseline correction,
zero-phase Butterworth bandpass, mains notch, resample, z-score.

## Layout

```
src/
├── config.py               # single source of truth (+ merge_model_config)
├── preprocessing/          # mri_transforms.py, signal_cleaner.py
├── data/                   # dataset.py, synthetic.py, augment.py
├── models/                 # spatial/temporal encoders, cross_attention, cafn,
│                           # attention_pool, positional, losses
├── explainability/         # grad_cam_3d.py (CAM & CAM++), plot_attention.py
├── utils.py                # seeding, device, stratified/group splits, label access
├── train.py                # training loop + CLI (torch-native balanced acc /
│                           # macro-F1 / confusion matrix in evaluate())
└── predict.py              # checkpoint → prediction + explainability bundle
tests/unit/  tests/integration/
```

See `Project.md` §v2 for the rationale behind each change.
