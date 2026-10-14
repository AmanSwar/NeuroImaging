# FINAL RESULTS (authoritative; generated from docs/results_raw by make_results.py / build_p2b.py)

## Study A - controlled synthetic benchmark (3 classes; MRI strength 1.0, VEP strength 0.3; test set of 120; chance 33.3 %)

| configuration | seeds | params (M) | accuracy % (mean ± sd) | macro-F1 % | Δ acc vs full CAFN (pts) |
|---|---|---|---|---|---|
| CAFN (full) | 7 | 1.96 | 78.1 ± 2.3 | 78.0 ± 2.4 | -- |
| w/o bidirectional | 5 | 1.82 | 77.0 ± 5.5 | 76.7 ± 5.1 | -1.7 |
| w/o positional enc. | 5 | 1.95 | 78.7 ± 3.9 | 78.5 ± 3.9 | -0.0 |
| w/o attention pooling | 5 | 1.96 | 77.7 ± 2.1 | 77.2 ± 2.2 | -1.0 |
| w/o SE blocks | 5 | 1.94 | 77.5 ± 5.9 | 76.8 ± 6.8 | -1.2 |
| w/o auxiliary loss | 5 | 1.96 | 77.3 ± 5.7 | 77.2 ± 5.8 | -1.3 |
| 2 fusion blocks | 5 | 2.22 | 77.0 ± 1.6 | 76.5 ± 2.1 | -1.7 |
| v1-style (all off) | 5 | 1.80 | 77.2 ± 7.0 | 76.7 ± 7.1 | -1.5 |
| Concatenation | 6 | 1.72 | 77.5 ± 2.3 | 77.1 ± 2.4 | -0.7 |
| Element-wise sum | 6 | 1.69 | 79.6 ± 3.4 | 79.4 ± 3.3 | +1.4 |
| MRI only | 5 | 1.69 | 68.0 ± 4.7 | 66.0 ± 5.9 | -10.7 |
| VEP only | 4 | 1.69 | 64.0 ± 4.3 | 63.5 ± 4.4 | -14.6 |

Key statistics (paired over common seeds): accFull=78.1; sdFull=2.3; nFull=7; accMRI=68.0; accVEP=64.0; accCat=77.5; accSum=79.6; gainUni=10.7; gainUniP=<0.001; nGainUni=5; dCat=+0.7; pCat==0.629; dSum=-1.4; pSum==0.465; ablMin=-1.7; ablMax=+0.0; ablSig=0; ablN=7; sdAll=4.3; accVone=77.2; sdVone=7.0; nSeedsMax=7

## Study A sweep - test accuracy % vs VEP informativeness (MRI fixed at 1.0), 2 seeds per point

| method | VEP 0.15 | VEP 0.3 | VEP 0.5 | VEP 1.0 |
|---|---|---|---|---|
| MRI only | 62.9 ± 4.1 | 65.0 ± 5.9 | 70.0 ± 4.7 | 62.9 ± 1.8 |
| VEP only | 45.0 ± 0.0 | 64.2 ± 1.2 | 85.4 ± 0.6 | 96.7 ± 2.4 |
| Concatenation | 69.6 ± 0.6 | 77.5 ± 7.1 | 90.8 ± 1.2 | 98.3 ± 0.0 |
| CAFN (full) | 67.5 ± 7.1 | 77.9 ± 11.2 | 90.0 ± 2.4 | 97.5 ± 2.4 |

## Study B - real cohorts (95 subjects: ds003568 18 healthy + 29 MDD; ds007181 27 healthy + 21 zoster neuralgia)

Out-of-fold ROC-AUC (mean over repeats; classical 5x5-fold CV, neural 1x5-fold), bootstrap 95 % CI (repeat 0), balanced accuracy, within-cohort permutation p (200 perms, classical only):

| model | AUC | 95 % CI | bal. acc. | perm. p |
|---|---|---|---|---|
| Spectrum (500-pt) · LogReg | 0.563 | [0.45, 0.68] | 0.590 | 0.025 |
| Band powers (5) · LogReg | 0.567 | [0.45, 0.68] | 0.590 | 0.005 |
| Band powers (5) · Random forest | 0.518 | [0.36, 0.60] | 0.539 | 0.726 |
| MRI (8³ → PCA-20) · LogReg | 0.587 | [0.43, 0.67] | 0.584 | 0.239 |
| MRI + band powers · LogReg | 0.588 | [0.43, 0.67] | 0.580 | 0.234 |
| CAFN (ours) | 0.510 | [0.40, 0.62] | 0.518 | -- |
| Concatenation fusion | 0.533 | [0.42, 0.66] | 0.553 | -- |
| MRI only | 0.479 | [0.36, 0.60] | 0.503 | -- |
| Spectrum only | 0.618 | [0.50, 0.72] | 0.561 | -- |

CAFN (64^3 MRI) on the cohorts, mean ± sd over 3 repeats (test subjects only):

| setting | accuracy | balanced acc. | AUC |
|---|---|---|---|
| pooled|test:ALL | 0.52 ± 0.04 | 0.52 ± 0.04 | 0.61 ± 0.09 |
| pooled|test:ds003568 | 0.52 ± 0.08 | 0.46 ± 0.07 | 0.61 ± 0.17 |
| pooled|test:ds007181 | 0.52 ± 0.08 | 0.54 ± 0.07 | 0.53 ± 0.17 |
| lodo_ds003568|test:ds003568 | 0.48 ± 0.09 | 0.46 ± 0.04 | 0.57 ± 0.08 |
| lodo_ds007181|test:ds007181 | 0.45 ± 0.02 | 0.49 ± 0.05 | 0.44 ± 0.03 |

Classical leave-one-cohort-out, AUC / balanced accuracy on the named test cohort (trained on the other):

| model | test ds003568 | test ds007181 |
|---|---|---|
| Spectrum (500-pt) · LogReg | 0.63 / 0.50 | 0.57 / 0.50 |
| Band powers (5) · LogReg | 0.50 / 0.50 | 0.48 / 0.50 |
| Band powers (5) · Random forest | 0.50 / 0.50 | 0.50 / 0.50 |
| MRI (8³ → PCA-20) · LogReg | 0.40 / 0.43 | 0.44 / 0.47 |
| MRI + band powers · LogReg | 0.41 / 0.43 | 0.46 / 0.51 |

Cohort-identity probe: a linear model on the spectrum alone identifies the cohort with 100 % accuracy (majority baseline 50.5 %).

## Figures (PNG, ready to insert): docs/results_assets/fig_ablation.png (Study A bars), fig_sweep.png (sweep), fig_real_cv.png (real-cohort AUC forest plot), fig_real_curves.png (pooled CAFN train vs val curves), docs/journal_ieee_access/figs/fig_arch_v2.png (architecture).

## Conclusions the documents must carry
1. Fusion beats the better single modality by ~10.7 points (paired, 5 seeds, p<0.001) in the controlled synthetic benchmark.
2. CAFN (78.1 %) is statistically indistinguishable from concatenation (77.5 %) and element-wise sum (79.6 %); none of 7 component ablations is reliably different (range -1.7 to 0.0 points).
3. Sweep: fusion gain over the best single modality is largest at intermediate informativeness and vanishes when one modality dominates; CAFN tracks concatenation.
4. Real cohorts: neural fusion at chance (CAFN AUC 0.51; concatenation 0.53; MRI-only 0.48); spectrum-only network best (AUC 0.62, CI 0.50-0.72); classical band-power logistic regression AUC 0.57 (perm p 0.005; Bonferroni-adjusted 0.025; CI includes 0.5); MRI features not significant; no transfer across cohorts.
5. NOT claimed: clinical accuracy, diagnosis of glaucoma/optic neuritis/CVI, cross-cohort generalisation, that cross-attention beats simpler fusion.
