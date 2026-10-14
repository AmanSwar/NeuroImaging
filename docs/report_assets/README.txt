Supporting files for the BCSE497J project report (NeuroImaging / CAFN)

scripts/report_experiments.py   Runs the repository's own VEP-conditioning, synthetic-data and mask
                                functions on seeded synthetic data (50 trials per class) and writes the
                                Chapter 5 figures and results.json.
                                  NEURO_REPO=<path to NeuroImaging> python report_experiments.py <out_dir>
scripts/model_complexity.py     Analytic parameter / multiply-accumulate count from the layer definitions
                                (no PyTorch needed). Writes figs/complexity.json.
scripts/report_diagrams.py      Generates the architecture / pipeline diagrams and bar charts
                                (uses new_blocks.py).  python report_diagrams.py <out_dir>
figures/                        All figures used in the report (PNG).
data/results.json               Measurements reported in Chapter 5 (signal conditioning, synthetic MRI).
data/complexity.json            Parameter and MAC counts (Table 5.5).
data/coverage.json              Per-module statement coverage reconstructed from the repository's
                                recorded .coverage file (Table 5.6).
data/tests_inventory.json       The 86 collected test functions (Appendix 3).

Note: PyTorch/MONAI/NiBabel could not be installed where the report was prepared, so no training run
or test run was executed; see Section 6.6 and Appendix 4 of the report.
