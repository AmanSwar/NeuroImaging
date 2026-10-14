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

## Real-data pipeline: multi-cohort experiment (one command)

```bash
./run_all.sh --smoke   # 1 subject per group, 2 epochs: checks the plumbing (~minutes + a small download)
./run_all.sh           # full run: download -> preprocess -> train -> validate -> test (leave it overnight)
```

`run_all.sh` creates the virtualenv, runs pre-flight tests, downloads the public
cohorts from OpenNeuro, preprocesses them, trains CAFN and writes
`results/cohorts/summary.md`. Everything is resumable (re-run the same command)
and seeded by one constant (`SEED = 42`; repeat *k* uses `42 + k`). Every run logs to
`logs/` and records the exact environment in `results/cohorts/environment.txt`.

| cohort (OpenNeuro) | patients vs controls | paired data | downloaded |
|---|---|---|---|
| [ds003568](https://openneuro.org/datasets/ds003568) | 29 MDD vs 22 healthy adolescents | T1w MRI + resting MEG | 43.7 GB |
| [ds007181](https://openneuro.org/datasets/ds007181) | 32 zoster neuralgia vs 34 healthy | T1w MRI + whole-night PSG EEG | 24.5 GB |
| [ds008894](https://openneuro.org/datasets/ds008894) (`--tier all`, optional) | 25 mood/suicide-risk vs 18 healthy | T1w MRI + task MEG | 53.9 GB |

**What the model sees.** MRI: RAS reorientation, Otsu mask, foreground z-score, 64^3. 1-D stream: the
channel-averaged log power spectrum (1-45 Hz, 500 points) of the resting MEG/EEG recording
(`VEPConfig.signal_mode="spectrum"`; no time-domain filtering). Label: 0 = control, 1 = patient
(pan-disorder; the disorder differs per cohort -- a deliberate, documented limitation).

**Protocol** (`src/cohort_experiments.py`): each cohort is split 70/15/15 (train/val/test, stratified);
(1) a **pooled** model is trained on all cohorts and validated/tested *per cohort*; (2) **leave-one-cohort-out**
trains on the other cohort(s) and tests on every subject of the held-out one. The checkpoint is chosen by
validation balanced accuracy (`TrainConfig.select_metric`), never on test data. Reports include bootstrap 95%
CIs, unimodal-head and spectrum-logistic-regression baselines, and a cohort-identity probe.

**Hardware.** The model has ~2 M parameters; training memory is ~0.3-0.6 GB of activations at batch 8 (AMP on CUDA),
so a 4 GB RTX 2050 is enough; the runner halves the batch size automatically on a CUDA OOM. No GPU?
It still runs on CPU, just slower. **Windows**: run inside WSL2 (or Git Bash) and disable sleep for overnight runs.

Download-only / manual steps:

```bash
python scripts/download_datasets.py                 # dry run: file counts + exact sizes
python scripts/download_datasets.py --yes           # download the core tier
python -m src.data.cohorts --raw data/raw --out data/processed
python -m src.cohort_experiments --manifest data/processed/manifest.csv --out results/cohorts
```

## Controlled synthetic benchmark and real-cohort cross-validation

```bash
# Study A: when does fusion help? 12 configurations x seeds, held-out test set (MRI 1.0 : VEP 0.3 informativeness)
python -m src.ablation_synthetic --settings 1.0:0.3 --seeds 0 1 2 3 4 --out results/synth_main
python -m src.ablation_synthetic --settings 1.0:0.15 1.0:0.3 1.0:0.5 1.0:1.0 --seeds 0 1 \
    --variants "CAFN (full)" "Concatenation" "MRI only" "VEP only" --out results/synth_sweep
# Study B: repeated CV on the 95 real subjects, baselines, bootstrap CIs, permutation tests
python -m src.cv_real --manifest data/processed/manifest.csv --out results/real_cv
```

`ModelConfig.fusion` also accepts `concat`, `sum`, `mri_only`, `vep_only` (literature fusion baselines and unimodal
references sharing the same encoders and recipe), and `SyntheticNeuroDataset(mri_strength=, vep_strength=)` sets how much
class information each modality carries. Findings (see `docs/` for the paper, report and slides): fusion beats the
better single modality by ~10.7 points in the synthetic benchmark, but cross-attention is statistically indistinguishable
from concatenation/sum and no single ablation is reliable; on the two real cohorts neural fusion is at chance (AUC 0.51) and
only a weak spectral signal (AUC 0.56-0.62) is detectable.

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

See `Project.md` ("v2/v3 engineering notes") for the rationale behind each
change.
