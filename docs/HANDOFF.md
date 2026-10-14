# Handoff: updating the project documents (report .docx, slides .pptx, paper)

Repository: `/Users/aman/code/faltu/mish/NeuroImaging` (docs live in `docs/`). Author on all documents: **Mishthi Jain**,
School of Computer Science and Engineering, VIT Chennai, BCSE497J project. Keep names, university template, headers, styles.

## Why the documents are out of date
They were written on 13 Oct 2026, **before any model was trained**. They describe the first model ("v1": one-way
cross-attention, 1,055,081 parameters, 86 tests, 95.1 % coverage) and state that "no training or test run was executed"
and "no accuracy is claimed". Since then the repository gained a v2/v3 architecture, real training and evaluation, a
controlled synthetic benchmark with ablations, a real-data pipeline on two public cohorts, and a 340-test suite.
**Every statement that is now false must be corrected; content that is still true (VEP conditioning results, P100
latency, filter responses, MRI mask behaviour on synthetic data, clinical background, literature on glaucoma/VEP) stays.**

## Ground rules (non-negotiable)
1. **No invented or "improved" numbers.** Every number must come from `docs/results_assets/facts.json`,
   `docs/results_assets/*.tex`, `docs/results_raw/**`, or the new paper sections in `docs/journal_ieee_access/parts/*.tex`
   (those are the authoritative wording). If a number you need is not there, leave it out or say it is not available.
2. **Label every result** as *synthetic* or *real-cohort*. Synthetic numbers are a controlled benchmark; they are not
   clinical performance. Real-cohort results are modest/near chance and must be reported as such. Do **not** soften them.
3. The real cohorts are **not** eye-disease data: ds003568 = MDD vs healthy adolescents (MEG), ds007181 = zoster-associated
   neuralgia vs healthy (PSG EEG). The label is pan-disorder "patient vs control". No VEP was available publicly; the 1-D
   stream is the log power spectrum of the resting recording.
4. Citations: only those verified. CAMF = Zhou, Orlichenko, Qu, Fu, Calhoun, Ding, Wang, arXiv:2404.00144 (2024; schizophrenia,
   sMRI+fMRI). DRIFA-Net = Dhar, Zaidi, Haghighat, Goyal, Roy, Alavi, Kumar, WACV 2025, arXiv:2412.01248. (Project.md's older
   attributions "Zhang et al. 2024 CAMF / IEEE TMI / PubMed 38416629", "Guo et al. 2025 DRIFA-Net / arXiv:2411.08324" and
   "Hao et al. 2023" are wrong or unverified: do not use them.)
5. Do not touch files outside `docs/` (except to read). Work on copies and keep the originals as `*_v1_backup.*`.

## What is new (facts that are final)
**Architecture (CAFN v2/v3)** – `src/models/`: gated *bidirectional* cross-attention (temporal tokens also query spatial tokens;
learned scalar gates init 0, sigma(0)=0.5; pre-norm blocks, FFN, `fusion_blocks` stackable); learned 3-D positional embedding on the
4x4x4 token grid + sinusoidal temporal encoding; learned attention pooling (near-uniform init = mean pooling at start);
squeeze-and-excitation (reduction 4) in the 3-D residual blocks; multi-scale 1-D temporal blocks (kernels 3/9/19);
auxiliary unimodal heads (lambda = 0.3); optional modality dropout, class weights, label smoothing, Kendall-style uncertainty
loss; Grad-CAM and Grad-CAM++; per-head attention plots; P100 estimator. Deterministic adaptive pooling (`src/models/pooling.py`,
matmul-based, equals PyTorch's op, deterministic on CUDA, works on MPS).
**New baselines for like-for-like comparison** (`ModelConfig.fusion`): `concat`, `sum` (element-wise), `mri_only`, `vep_only`,
and `cross` (= original one-way v1 fusion).
**Size:** 1,955,140 parameters (3 classes): spatial encoder 894,504; temporal encoder 785,936; fusion 265,218; positional
embeddings 8,192; attention pool 129; heads 1,161. MACs per sample: 2.60 G at 64^3, 0.39 G at 32^3 (spatial encoder 97 % of MACs).
**Tests:** 340 pass (macOS, Python 3.14; Ubuntu 22.04 WSL2, Python 3.12, CUDA: 1 skipped). Three latent bugs found by the second
platform were fixed (Python <= 3.13 annotation NameError in `utils.py`; `--eval-only` device mismatch on CUDA; a vacuous test).
**Reproducibility:** one-command pipeline `./run_all.sh` (download -> preprocess -> train -> validate -> test, resumable),
constant seed `SEED = 42` (repeat k uses 42+k), deterministic algorithms, environment snapshot, size-verified resumable downloads.
Hardware used: NVIDIA RTX 3050 Laptop GPU (4 GB, thermally throttled during long runs) + Apple M3 Max; model needs < 1 GB at batch 8.
**Preprocessing additions:** Otsu mask + hole filling + largest component; foreground z-score; `fit="bbox_resize"` (crop to the
foreground box then resample to 64^3 / 32^3; the old centre crop saw only a sliver of a real head); `signal_mode="spectrum"`
(channel-averaged log10 Welch PSD, 1-45 Hz, 500 points, z-scored; no filtering). `select_metric` (checkpoint chosen by validation
balanced accuracy) fixes a bug where val-loss selection could restore an untrained epoch-1 model on short runs (smoke/demo runs
ended at 33 % accuracy for that reason).

## Experiments (see the paper sections for exact protocols)
* **Study A – controlled synthetic benchmark** (`src/ablation_synthetic.py`): 3-class (Healthy/Glaucoma/ON-CVI); generator knobs
  `mri_strength`, `vep_strength` in [0,1]; 144/48/120 train/val/test, 32^3 MRI, 500-pt VEP; checkpoint by val balanced accuracy,
  all numbers on the held-out test set; main setting MRI 1.0 : VEP 0.3 (chosen from a pilot on the unimodal models only), 12
  configurations x seeds; sweep over VEP informativeness for MRI-only / VEP-only / concatenation / CAFN.
* **Study B – real cohorts** (`src/cohort_experiments.py`, `src/cv_real.py`, `run_all.sh`): ds003568 (47 subjects with both
  modalities: 18 healthy, 29 MDD) and ds007181 (48: 27 healthy, 21 zoster neuralgia); pooled model + leave-one-cohort-out, 5-fold
  repeated cross-validation, bootstrap CIs, within-cohort label-permutation tests, classical baselines.
* Earlier, smaller synthetic runs (6 runs; flagship 97.9 % val accuracy, held-out 99.0 % on 96 samples) used the default
  (unrestricted) generator where the VEP alone is almost perfectly separable (VEP-only head 99.0 %, MRI-only head 70.8 %), i.e. they
  demonstrate that the pipeline trains, **not** that fusion helps. That observation motivated Study A.
* An earlier v1-vs-v2 comparison (30 epochs, 192 synthetic subjects): both reach 98-100 % after best-checkpoint restore; v1 is
  unstable (validation accuracy ends at 48 %), v2 stays at 94-100 %.

## Real-cohort results already final
Classical 5x5-fold CV on all 95 subjects (AUC mean over repeats; permutation p within cohort, repeat 0; bootstrap CI repeat 0):
spectrum LR 0.563 (CI 0.45-0.68, p=0.025); band-power LR 0.567 (0.45-0.68, p=0.005); band-power RF 0.518 (0.36-0.60, p=0.73);
MRI(8^3->PCA20) LR 0.587 (0.43-0.67, p=0.24); MRI+band-power LR 0.588 (0.43-0.67, p=0.23). Leave-one-cohort-out for classical
models: AUC 0.40-0.63 (no consistent transfer). CAFN pooled/LODO (3 repeats, 64^3 MRI): all test accuracies 0.45-0.52, balanced
accuracy 0.46-0.54: **at chance**. Cohort-identity probe: the spectrum alone identifies the cohort with 100 % accuracy.
Everything else is in `docs/results_assets/facts.json` once the experiments finish (a file `docs/results_assets/FINAL` appears then).

## The honest story (what the documents should argue)
Novelty/contribution = (1) a gated bidirectional cross-attention design for heterogeneous 3-D imaging + 1-D signal with
deterministic, tested implementation; (2) a **controlled benchmark that shows when fusion helps** and a component ablation, against
the fusion baselines used in the literature (concatenation, sum, unimodal); (3) a fully reproducible real-data protocol with
leave-one-cohort-out and permutation tests; (4) candid negative/weak real-data findings and the pitfalls discovered.
Not claimed: clinical accuracy, glaucoma/optic-neuritis diagnosis, generalisation across cohorts, state-of-the-art.
