"""Repeated cross-validation on the pooled real cohorts, with baselines and permutation tests.

The 7-subject per-cohort test splits of ``cohort_experiments`` are too small to
separate skill from luck. Here every one of the N subjects is predicted
out-of-fold exactly once per repeat (5-fold, stratified on cohort x label), so
AUC/balanced accuracy are estimated on all N subjects, with:

* classical baselines (logistic regression / random forest on band powers, the
  full spectrum, and PCA-reduced MRI) and four neural variants (CAFN,
  concatenation, MRI-only, spectrum-only);
* a fixed-seed bootstrap CI over subjects and a label-permutation p-value
  (permuting labels *within cohort*), so "no better than chance" is a tested
  statement, not an impression;
* leave-one-cohort-out for the classical models.

    python -m src.cv_real --manifest data/processed/manifest.csv --out results/real_cv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset

from src.config import Config, ModelConfig, TrainConfig
from src.models.cafn import CAFN
from src.train import _verify_device_supports_model, train_model
from src.utils import auto_device, seed_everything

SEED = 42
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_arrays(manifest: str) -> dict:
    base = os.path.dirname(os.path.abspath(manifest))
    rows = list(csv.DictReader(open(manifest)))
    mri64 = np.stack([np.load(os.path.join(base, r["mri_path"])) for r in rows]).astype(np.float32)  # (N,1,64,64,64)
    spec = np.stack([np.loadtxt(os.path.join(base, r["vep_path"]), skiprows=1) for r in rows])
    n = len(rows)
    return dict(
        ids=[r["subject_id"] for r in rows], cohort=np.array([r["dataset"] for r in rows]),
        y=np.array([int(r["label"]) for r in rows]),
        mri32=mri64.reshape(n, 1, 32, 2, 32, 2, 32, 2).mean(axis=(3, 5, 7)),
        mri8=mri64.reshape(n, 8, 8, 8, 8, 8, 8).mean(axis=(2, 4, 6)).reshape(n, -1),
        spec=spec,
    )


def band_power(spec: np.ndarray) -> np.ndarray:
    """Mean log-PSD in the canonical EEG/MEG bands (spec is the 500-pt 1-45 Hz vector)."""
    f = np.linspace(1, 45, spec.shape[1])
    return np.stack([spec[:, (f >= lo) & (f < hi + (hi == 45))].mean(axis=1) for lo, hi in BANDS.values()], axis=1)


class TensorPairDataset(Dataset):
    def __init__(self, mri32, spec, y, ids):
        spec = (spec - spec.mean(1, keepdims=True)) / (spec.std(1, keepdims=True) + 1e-8)
        self.mri = torch.from_numpy(mri32.astype(np.float32))
        self.vep = torch.from_numpy(spec.astype(np.float32))[:, None, :]
        self.labels = [int(v) for v in y]
        self.ids = list(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"mri": self.mri[i], "vep": self.vep[i], "label": torch.tensor(self.labels[i]), "subject_id": self.ids[i]}


# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------

def classical_models(d: dict) -> dict:
    bp = band_power(d["spec"])
    feats = {"spec": d["spec"], "bp": bp, "mri": d["mri8"], "mri+bp": np.hstack([d["mri8"], bp])}
    mk = {
        "Spectrum (500-pt) · LogReg": ("spec", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))),
        "Band powers (5) · LogReg": ("bp", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))),
        "Band powers (5) · Random forest": ("bp", lambda: RandomForestClassifier(100, max_depth=3, min_samples_leaf=3, random_state=SEED, n_jobs=1)),
        "MRI (8³ → PCA-20) · LogReg": ("mri", lambda: make_pipeline(StandardScaler(), PCA(20, random_state=SEED), LogisticRegression(C=0.05, max_iter=3000))),
        "MRI + band powers · LogReg": ("mri+bp", lambda: make_pipeline(StandardScaler(), PCA(20, random_state=SEED), LogisticRegression(C=0.05, max_iter=3000))),
    }
    return {name: (feats[k], f) for name, (k, f) in mk.items()}


def strata(d) -> np.ndarray:
    return np.array([f"{c}|{y}" for c, y in zip(d["cohort"], d["y"])])


def folds(d, repeat: int, k: int = 5):
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED + repeat)
    return list(skf.split(np.zeros(len(d["y"])), strata(d)))


def oof_classical(X, make, y, splits) -> np.ndarray:
    p = np.zeros(len(y))
    for tr, te in splits:
        m = make().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    return dict(auc=float(roc_auc_score(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                accuracy=float((pred == y).mean()))


def bootstrap_ci(y, p, n=1000, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        a.append(roc_auc_score(y[i], p[i])); b.append(balanced_accuracy_score(y[i], (p[i] >= .5).astype(int)))
    return dict(auc=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                balanced_accuracy=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])


def permutation_p(X, make, y, cohort, splits, observed_auc, n_perm, seed=SEED) -> float:
    """P(AUC >= observed) when labels are shuffled within each cohort (keeps cohort base rates)."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for c in np.unique(cohort):
            idx = np.where(cohort == c)[0]
            yp[idx] = y[rng.permutation(idx)]
        null.append(roc_auc_score(yp, oof_classical(X, make, yp, splits)))
    return float((1 + sum(v >= observed_auc for v in null)) / (1 + n_perm))


def lodo_classical(X, make, y, cohort) -> dict:
    out = {}
    for c in np.unique(cohort):
        te = np.where(cohort == c)[0]; tr = np.where(cohort != c)[0]
        m = make().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        out[str(c)] = metrics(y[te], p)
    return out


# ---------------------------------------------------------------------------
# Neural variants
# ---------------------------------------------------------------------------

NEURAL = {
    "CAFN (ours)": "bidirectional",
    "Concatenation fusion": "concat",
    "MRI only": "mri_only",
    "Spectrum only": "vep_only",
}


def neural_cfg(fusion: str, args) -> Config:
    return Config(
        model=ModelConfig(d_model=64, num_heads=4, ffn_hidden=128, dropout=0.3, num_classes=2, fusion=fusion),
        train=TrainConfig(batch_size=8, epochs=args.epochs, lr=5e-4, weight_decay=1e-2, aux_loss_weight=0.3,
                          patience=15, seed=SEED, balance_classes=True, augment=True,
                          select_metric="val_balanced_accuracy", amp=True, deterministic=True,
                          checkpoint_dir=args.out, num_workers=0),
    )


@torch.no_grad()
def predict_pos(model, ds, idx, device) -> np.ndarray:
    model.eval()
    out = []
    for i in idx:
        b = ds[i]
        o = model(b["mri"][None].to(device), b["vep"][None].to(device))
        out.append(float(torch.softmax(o["logits_fused"].float(), 1)[0, 1]))
    return np.array(out)


def oof_neural(ds, d, fusion, splits, device, args, rep) -> np.ndarray:
    p = np.zeros(len(d["y"]))
    for k, (tr, te) in enumerate(splits):
        # inner validation split (15% of train, stratified) for checkpoint selection
        inner = StratifiedKFold(6, shuffle=True, random_state=SEED + 100 * rep + k)
        a, b = next(inner.split(np.zeros(len(tr)), strata(d)[tr]))
        tr_idx, va_idx = tr[a], tr[b]
        cfg = neural_cfg(fusion, args)
        seed_everything(SEED + 100 * rep + k, deterministic=True)
        model = CAFN(cfg.model)
        train_model(model, Subset(ds, list(tr_idx)), Subset(ds, list(va_idx)), cfg=cfg, device=device, augment=True)
        p[te] = predict_pos(model, ds, te, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return p


# ---------------------------------------------------------------------------

def run(args, log=print) -> dict:
    d = load_arrays(args.manifest)
    y, cohort = d["y"], d["cohort"]
    os.makedirs(args.out, exist_ok=True)
    res: dict = dict(n=int(len(y)), cohorts={c: int((cohort == c).sum()) for c in np.unique(cohort)},
                     n_patient=int(y.sum()), repeats=args.repeats, models={}, lodo={})
    preds: list[dict] = []

    for name, (X, make) in ([] if args.skip_classical else classical_models(d).items()):
        per = []
        for r in range(args.repeats):
            sp = folds(d, r)
            p = oof_classical(X, make, y, sp)
            per.append(metrics(y, p))
            if r == 0:
                ci = bootstrap_ci(y, p)
                pval = permutation_p(X, make, y, cohort, sp, per[0]["auc"], args.perms)
                preds += [dict(model=name, subject_id=s, label=int(l), p=float(q)) for s, l, q in zip(d["ids"], y, p)]
        res["models"][name] = dict(per_repeat=per, ci95_repeat0=ci, perm_p_repeat0=pval)
        res["lodo"][name] = lodo_classical(X, make, y, cohort)
        log(f"{name:<36} AUC {np.mean([m['auc'] for m in per]):.3f}  bAcc {np.mean([m['balanced_accuracy'] for m in per]):.3f}  perm-p {pval:.3f}")

    if not args.skip_neural:
        device = _verify_device_supports_model(auto_device(args.device), neural_cfg("bidirectional", args))
        ds = TensorPairDataset(d["mri32"], d["spec"], y, d["ids"])
        for name, fusion in NEURAL.items():
            per = []
            t0 = time.time()
            for r in range(args.neural_repeats):
                p = oof_neural(ds, d, fusion, folds(d, r), device, args, r)
                per.append(metrics(y, p))
                if r == 0:
                    ci = bootstrap_ci(y, p)
                    preds += [dict(model=name, subject_id=s, label=int(l), p=float(q)) for s, l, q in zip(d["ids"], y, p)]
            res["models"][name] = dict(per_repeat=per, ci95_repeat0=ci, minutes=round((time.time() - t0) / 60, 1))
            log(f"{name:<36} AUC {np.mean([m['auc'] for m in per]):.3f}  bAcc {np.mean([m['balanced_accuracy'] for m in per]):.3f}  ({res['models'][name]['minutes']} min)")
            json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "subject_id", "label", "p"]); w.writeheader(); w.writerows(preds)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--out", default="results/real_cv")
    ap.add_argument("--repeats", type=int, default=5, help="CV repeats for classical models")
    ap.add_argument("--neural-repeats", type=int, default=2)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-neural", action="store_true")
    ap.add_argument("--skip-classical", action="store_true")
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
