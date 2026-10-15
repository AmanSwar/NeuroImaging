You are helping finish a university project (BCSE497J, VIT Chennai; author Mishthi Jain). You created the
original project report (.docx), slides (.pptx) and IEEE-style paper in /Users/aman/code/faltu/mish/NeuroImaging/docs/
on 13 Oct 2026, BEFORE any model had been trained. Since then a lot has changed: the model was redesigned, real
experiments were run, and the documents were rewritten by another Claude (Claude Code) using python-docx / python-pptx / LaTeX
without Word or PowerPoint available. I now need YOU, with Word/PowerPoint/PDF tools on the Mac, to do the finishing work
that could not be done headlessly. Deadline: finish within ~1.5 hours. Read first: docs/HANDOFF.md and
docs/results_assets/FINAL_RESULTS.md (authoritative numbers). Work on COPIES (backups *_v1_backup.* already exist) and
tell me exactly what you changed.

== WHAT CHANGED SINCE YOU WROTE THE ORIGINALS ==
1. Model "CAFN v2/v3": gated BIDIRECTIONAL cross-attention (temporal tokens also query spatial tokens; learned scalar gates
   init 0), learned 3-D + sinusoidal positional encodings, learned attention pooling, squeeze-and-excitation blocks, multi-scale
   1-D temporal blocks (kernels 3/9/19), auxiliary unimodal heads, deterministic adaptive pooling (matmul-based). 1,955,140
   parameters (was 1,055,081), 2.60 GMACs at 64^3 (was 2.53 G). New fusion baselines: concatenation, element-wise sum, MRI-only,
   VEP/spectrum-only. 340 automated tests (was 86), pass on macOS (Py 3.14) and Ubuntu 22.04 (Py 3.12, CUDA). Three latent bugs
   found and fixed on the second platform. Everything is in GitHub PR https://github.com/Mishthijainn/NeuroImaging/pull/2.
2. Real training and evaluation now exist (the old documents said "no training was run" - that is no longer true):
   - Study A, controlled SYNTHETIC benchmark (3 classes; generator knobs set how informative each modality is; MRI 1.0 / VEP 0.3;
     144/48/120 train/val/test; 12 configurations x >=5 seeds + an informativeness sweep with 2 seeds per point).
   - Study B, two REAL public OpenNeuro cohorts, 95 subjects with both modalities: ds003568 (MDD vs healthy, T1w + resting MEG,
     18 healthy + 29 patients) and ds007181 (zoster-associated neuralgia vs healthy, T1w + whole-night PSG EEG, 27 + 21).
     The label is pan-disorder "patient vs control"; the 1-D stream is the resting log-power spectrum (no VEP is publicly
     available with MRI + diagnoses). Protocols: per-cohort splits, pooled model, leave-one-cohort-out, repeated cross-validation,
     bootstrap CIs, within-cohort permutation tests, classical baselines.
3. One-command reproducible pipeline: ./run_all.sh (download -> preprocess -> train -> validate -> test), SEED=42, run on an
   RTX 3050 4 GB laptop GPU.

== FINAL RESULTS (the only numbers you may use) ==
- Synthetic: fusion beats the better single modality by +10.7 points (paired, 5 seeds, p<0.001): CAFN 78.1% (n=7 seeds) vs
  MRI-only 68.0% and VEP-only 64.0%. CAFN 78.1% vs concatenation 77.5% vs element-wise sum 79.6%: statistically
  indistinguishable (p=0.63 / 0.47). None of 7 single-component ablations is reliable (range -1.7..0.0 points; seed SD ~4).
  => We do NOT claim cross-attention beats simpler fusion.
- Sweep (2 seeds/point, trends only): fusion gain over best single modality largest at intermediate informativeness, vanishes
  when one modality dominates; CAFN tracks concatenation.
- Real cohorts: neural fusion at CHANCE (CAFN AUC 0.51, concatenation 0.53, MRI-only 0.48); best = spectrum-only network AUC 0.62
  (95% CI 0.50-0.72); classical band-power logistic regression AUC 0.567, permutation p=0.005 (Bonferroni-adjusted 0.025, CI
  includes 0.5); MRI features not significant; CAFN pooled/leave-one-cohort-out accuracy 0.45-0.52 (chance); classical
  leave-one-cohort-out AUC 0.40-0.63; the spectrum alone identifies the cohort with 100% accuracy (shortcut possible).
- NOT claimed anywhere: clinical accuracy, glaucoma/optic-neuritis diagnosis, cross-cohort generalisation, or that cross-attention
  beats simpler fusion. Keep this honest tone; never soften the real-cohort results.

== CURRENT STATE OF THE FILES (docs/) ==
- NeuroImaging_BCSE497J_Project_Report.docx : already rewritten (Ch.1-7 + appendices, 42 tables, 42 images). Backup: *_v1_backup.docx.
- NeuroImaging_CAFN_Presentation.pptx : 32 slides, placeholders filled. Backup: *_v1_backup.pptx.
- NeuroImaging_IEEE_Access_Paper.pdf + journal_ieee_access/ (main.tex, parts/, refs.tex) + journal_ieee_access_latex_source.zip.
  Rebuild: .venv/bin/python docs/make_results.py && docs/build_p2b.py && docs/build_paper.py (uses tectonic).
- /Users/aman/code/faltu/mish/NeuroImaging/report.html : standalone web report with a new "Final results" section on top.
- Raw results: docs/results_raw/, figures + tables: docs/results_assets/ (fig_ablation, fig_sweep, fig_real_cv, fig_real_curves,
  facts.json, FINAL_RESULTS.md). Architecture figure: docs/journal_ieee_access/figs/fig_arch_v2.png.

== YOUR TASKS ==
A. Word report (.docx)
 1. Right-click the table of contents -> Update field (13 new heading entries have blank page numbers).
 2. The List of Tables / List of Figures are MANUAL text: type correct page numbers for new/renumbered entries
    (Tables 2.4, 3.4, 3.9, 4.1, 4.4, 5.9-5.13; Figures 4.1, 5.7-5.10) and re-check every shifted one. Tables 3.4-3.7 became
    3.5-3.8, old 4.1-4.5 became 4.2,4.3,4.5,4.6,4.7, old Figures 4.1-4.6 became 4.2-4.7; check in-text references match.
 3. Figures 3.1, 4.2, 4.4, 4.7 and 5.5 still show the ORIGINAL v1 model (their captions say so). Redraw them for the v2/v3 model
    (or keep with the "original" caption, but tell me which). Figure 5.5 is a v1-only parameter chart: regenerate from
    docs/results_raw/model_complexity_v2.json (spatial 894,504; temporal 785,936; fusion 265,218; pos-emb 8,192; pool 129; heads 1,161).
 4. The new equations (gated updates, attention pool) are inserted as PNG images; convert to native Word equations if easy.
 5. Sections 3.5.x / 5.2-5.3 describe the ORIGINAL synthetic generator (latencies 100/108/140 ms, block lesions) and 3.5.4 / 5.9
    describe the CURRENT generator (100+-3, 104+-4, 138+-6 ms; random lesions). Make sure this is clearly labelled everywhere.
 6. Check the university template compliance against "BCE497J PROJECT REPORT TEMPLATE_ UPDATED.docx" (fonts, margins, headings,
    page numbering, cover/certificate/declaration pages, abstract length), spelling, consistency of terminology.
 7. References: the 5 new entries (Zhou 2024 CAMF, Dhar 2025 DRIFA-Net, Hu 2018 SE-Net, ds003568 Liuzzi et al., ds007181 Li et al.)
    show "n.d." for the datasets; look up the correct years/DOIs (OpenNeuro pages) and fix. Verify in-text citations match the list.
 8. Re-export docs/NeuroImaging_BCSE497J_Project_Report_preview.pdf (the old one is stale).
B. Slides (.pptx)
 1. Flip through all 32 slides in PowerPoint: fix overflow/overlap (tightest: slides 21, 22, 25 banners and slide 28 bullets),
    fonts (originals use Calibri/Cambria), image distortion, slide numbers, speaker notes.
 2. Slide order: motivation -> method -> data -> engineering -> results -> discussion -> conclusion. Suggest cuts for a 15-20 minute talk
    (e.g. which of slides 13, 17-20 can become backup/appendix) but do not delete content without telling me.
 3. Make sure slide 30 (final conclusions) and slide 31 (closing) still say no diagnostic claim.
C. IEEE paper
 1. Fill the yellow placeholders: author e-mail, funding statement, guide name/designation (acknowledgement), short biography.
 2. For real submission move the body into the official IEEE Access template (ieeeaccess.cls); here a local approximation is used.
 3. Spot-check every reference (volume/pages/DOI), especially the 5 new ones; check page count (currently 8-9 pages) and that tables fit columns.
D. Cross-document consistency pass: the same numbers (above), same terminology, "synthetic" vs "real-cohort" always labelled, same
   hardware statement (RTX 3050 4 GB laptop GPU + Apple M3 Max for some synthetic runs), same test count (340), same parameter count.
   Produce a short table "number -> where it appears in each document" and flag any mismatch.

== RULES ==
- Do not invent or round numbers differently; if a number is not in FINAL_RESULTS.md / facts.json / the paper, ask me.
- Do not modify anything under src/ or tests/. Do not touch results_raw/. Keep the *_v1_backup.* files.
- If something in a document contradicts FINAL_RESULTS.md, FINAL_RESULTS.md wins - fix the document and tell me.
- Report at the end: what you changed per file, what you could not do, and anything that needs my decision.
