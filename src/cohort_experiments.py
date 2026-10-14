"""Train / validate / test CAFN on the pooled real cohorts, plus leave-one-cohort-out.

Protocol (all randomness derives from ``SEED``; repeat ``k`` uses ``SEED + k``):

1. **Per-cohort split.** Inside every cohort, subjects are split 70/15/15 into
   train/val/test, stratified by label, so each cohort contributes to every split.
2. **Pooled run** -- train on the union of the cohorts' train splits, select the
   checkpoint on the union of their val splits, then report **val and test
   separately for every cohort** and for the pool (within-cohort performance).
3. **Leave-one-cohort-out (LODO)** -- for each cohort D: train on the *other*
   cohorts (their train splits; their val splits select the checkpoint) and test
   on **all** subjects of D. This is the cross-cohort generalisation claim.
4. **Context baselines** on the same splits: logistic regression on the 1-D
   spectrum, the unimodal auxiliary heads of the trained model, chance, and a
   cohort-identity probe (can the spectrum alone tell which cohort a subject is from?).

Everything is written under ``--out``; finished runs are skipped on re-launch,
so an interrupted overnight job resumes where it stopped.

    python -m src.cohort_experiments --manifest data/processed/manifest.csv --out results/cohorts
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import math
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (must precede CUDA init)
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from src.config import Config, ModelConfig, TrainConfig, VEPConfig
from src.data.cohorts import CLASS_NAMES_BINARY, MRI_CONFIG, SPECTRUM_LENGTH
from src.data.dataset import PairedNeuroDataset
from src.models.cafn import CAFN
from src.train import _verify_device_supports_model, train_model
from src.utils import auto_device, count_parameters, seed_everything

SEED = 42
HEADS = ("fused", "spatial", "temporal")

DEFAULTS = dict(
    epochs=120, patience=25, batch_size=8, lr=5e-4, weight_decay=1e-2,
    select_metric="val_balanced_accuracy", repeats=3,
    fractions=(0.70, 0.15, 0.15),
)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def per_cohort_split(
    cohorts: list[str], labels: list[int], seed: int,
    fractions: tuple[float, float, float] = DEFAULTS["fractions"],
) -> dict[str, dict[str, list[int]]]:
    """{cohort: {"train","val","test": [row indices]}}, stratified by label, deterministic."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, list[int]]] = {}
    for ds in sorted(set(cohorts)):
        parts = {"train": [], "val": [], "test": []}
        for lab in sorted(set(labels)):
            idx = [i for i, (c, y) in enumerate(zip(cohorts, labels)) if c == ds and y == lab]
            idx = [idx[j] for j in rng.permutation(len(idx))]
            n = len(idx)
            n_test = max(1, round(fractions[2] * n)) if n >= 2 else 0
            n_val = max(1, round(fractions[1] * n)) if n >= 3 else 0
            parts["test"] += idx[:n_test]
            parts["val"] += idx[n_test:n_test + n_val]
            parts["train"] += idx[n_test + n_val:]
        out[ds] = {k: sorted(v) for k, v in parts.items()}
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan")


def binary_metrics(y: np.ndarray, p1: np.ndarray) -> dict:
    """Metrics for label array ``y`` and P(patient) ``p1`` at the 0.5 threshold."""
    y = np.asarray(y).astype(int); p1 = np.asarray(p1, dtype=float)
    pred = (p1 >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    parts = [v for v in (sens, spec) if not math.isnan(v)]
    return dict(
        n=int(y.size), n_patient=int((y == 1).sum()), accuracy=float((pred == y).mean()),
        balanced_accuracy=float(np.mean(parts)) if parts else float("nan"),
        sensitivity=sens, specificity=spec, auc=_auc(y, p1),
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        confusion_matrix=[[tn, fp], [fn, tp]],
    )


def bootstrap_ci(y, p1, n_boot: int = 1000, seed: int = SEED) -> dict:
    """95% percentile CIs for accuracy / balanced accuracy / AUC (fixed-seed bootstrap)."""
    y = np.asarray(y).astype(int); p1 = np.asarray(p1, dtype=float)
    rng = np.random.default_rng(seed)
    vals = {"accuracy": [], "balanced_accuracy": [], "auc": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        m = binary_metrics(y[i], p1[i])
        for k in vals:
            if not math.isnan(m[k]):
                vals[k].append(m[k])
    return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None)
            for k, v in vals.items()}


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

def build_config(args) -> Config:
    return Config(
        mri=MRI_CONFIG,
        vep=VEPConfig(signal_mode="spectrum", length=SPECTRUM_LENGTH),
        model=ModelConfig(num_classes=2),
        train=TrainConfig(
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            patience=args.patience, seed=SEED, balance_classes=True, augment=True,
            select_metric=args.select_metric, amp=True, deterministic=True,
            checkpoint_dir=args.out, num_workers=0,
        ),
    )


@torch.no_grad()
def predict_heads(model, subset, device, batch_size: int = 4) -> dict:
    """{'y': labels, 'fused'|'spatial'|'temporal': P(patient)} for every sample of ``subset``."""
    model.eval()
    out = {"y": [], **{h: [] for h in HEADS}}
    for batch in DataLoader(subset, batch_size=batch_size, shuffle=False):
        o = model(batch["mri"].to(device), batch["vep"].to(device))
        for h in HEADS:
            out[h].append(torch.softmax(o[f"logits_{h}"].float(), dim=1)[:, 1].cpu().numpy())
        out["y"].append(batch["label"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()} if out["y"] else {}


def train_with_oom_fallback(ds, train_idx, val_idx, cfg: Config, device, run_name: str, log=print):
    """Train; on CUDA OOM halve the batch size and retry (4 GB laptop GPUs)."""
    bs = cfg.train.batch_size
    while True:
        seed_everything(cfg.train.seed, deterministic=True)
        model = CAFN(cfg.model)
        run_cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, batch_size=bs))
        try:
            hist = train_model(
                model, Subset(ds, train_idx), Subset(ds, val_idx) if len(val_idx) else None,
                cfg=run_cfg, device=device, verbose=True, run_name=run_name, augment=True,
            )
            return model, hist, bs
        except torch.cuda.OutOfMemoryError:
            del model
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            log(f"!! CUDA out of memory -> retrying with batch size {bs}")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def load_spectra(ds: PairedNeuroDataset) -> np.ndarray:
    return np.stack([np.loadtxt(r["vep_path"], skiprows=1) for r in ds.rows])


def spectrum_logreg(X, y, tr, te) -> np.ndarray | None:
    if len(set(y[tr].tolist())) < 2 or len(te) == 0:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X[te])[:, 1]


def cohort_identity_probe(X, cohorts: list[str], seed: int = SEED) -> dict:
    """5-fold CV accuracy of predicting *which cohort* from the spectrum alone."""
    names = sorted(set(cohorts))
    if len(names) < 2:
        return {"accuracy": None, "chance": None}
    y = np.array([names.index(c) for c in cohorts])
    k = min(5, int(np.bincount(y).min()))
    if k < 2:
        return {"accuracy": None, "chance": None}
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), X, y, cv=skf)
    return {"accuracy": float(acc.mean()), "chance": float(np.bincount(y).max() / len(y)), "cohorts": names}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def _evaluate_split(name: str, idx: list[int], model, ds, device, X, y_all, tr_idx, bs, rows_out, exp, rep):
    if len(idx) == 0:
        return None
    pred = predict_heads(model, Subset(ds, idx), device, bs)
    res = {"n": len(idx)}
    for h in HEADS:
        m = binary_metrics(pred["y"], pred[h])
        if h == "fused":
            m["ci95"] = bootstrap_ci(pred["y"], pred[h])
        res[h] = m
    lr = spectrum_logreg(X, y_all, np.array(tr_idx), np.array(idx))
    if lr is not None:
        res["spectrum_logreg"] = binary_metrics(pred["y"], lr)
    ids = [ds.rows[i]["subject_id"] for i in idx]
    for sid, i, yy, *ps in zip(ids, idx, pred["y"], pred["fused"], pred["spatial"], pred["temporal"]):
        rows_out.append(dict(experiment=exp, repeat=rep, split=name, subject_id=sid,
                             dataset=ds.rows[i].get("dataset", ""), label=int(yy),
                             p_fused=float(ps[0]), p_spatial=float(ps[1]), p_temporal=float(ps[2])))
    return res


def run_all(args, log=print) -> dict:
    cfg = build_config(args)
    device = _verify_device_supports_model(auto_device(args.device), cfg)
    os.makedirs(args.out, exist_ok=True)
    ds = PairedNeuroDataset(args.manifest, cfg=cfg, cache=True)
    cohorts = ds.cohorts
    labels = list(ds.labels)
    y_all = np.array(labels)
    X = load_spectra(ds)
    names = sorted(set(cohorts))
    log(f"device={device}  subjects={len(ds)}  cohorts={names}  params={count_parameters(CAFN(cfg.model)):,}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}  "
            f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    results = {"seed": SEED, "config": dataclasses.asdict(cfg.train), "cohorts": names,
               "n_subjects": len(ds), "identity_probe": cohort_identity_probe(X, cohorts), "repeats": {}}
    pred_rows: list[dict] = []
    for rep in range(args.repeats):
        seed = SEED + rep
        cfg_r = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=seed))
        splits = per_cohort_split(cohorts, labels, seed)
        union = lambda part, which=names: sorted(i for d in which for i in splits[d][part])
        rep_res: dict = {"splits": {d: {k: len(v) for k, v in s.items()} for d, s in splits.items()}}
        experiments = [("pooled", names, None)] + [(f"lodo_{d}", [n for n in names if n != d], d) for d in names if len(names) > 1]
        for exp, train_cohorts, held in experiments:
            run_name = f"r{rep}/{exp}"
            done = os.path.join(args.out, run_name, "result.json")
            if os.path.exists(done):
                log(f"[skip] {run_name} (already finished)")
                saved = json.load(open(done))
                rep_res[exp] = saved["result"]; pred_rows += saved["predictions"]
                continue
            tr, va = union("train", train_cohorts), union("val", train_cohorts)
            log(f"\n=== {run_name}: train {len(tr)} / val {len(va)} subjects ({', '.join(train_cohorts)}) ===")
            t0 = time.time()
            model, hist, bs = train_with_oom_fallback(ds, tr, va, cfg_r, device, run_name, log)
            ebs = min(bs, 4)
            local: list[dict] = []
            result = {"train_cohorts": train_cohorts, "epochs_run": hist["epochs_run"], "batch_size": bs,
                      "minutes": round((time.time() - t0) / 60, 2), "splits": {}}
            if held is None:
                for d in names + ["ALL"]:
                    which = names if d == "ALL" else [d]
                    for part in ("val", "test"):
                        r = _evaluate_split(f"{part}:{d}", union(part, which), model, ds, device, X, y_all, tr, ebs, local, exp, rep)
                        if r:
                            result["splits"][f"{part}:{d}"] = r
            else:
                every = [i for i, c in enumerate(cohorts) if c == held]
                r = _evaluate_split(f"test:{held}", every, model, ds, device, X, y_all, tr, ebs, local, exp, rep)
                if r:
                    result["splits"][f"test:{held}"] = r
            rep_res[exp] = result
            pred_rows += local
            with open(done, "w") as f:
                json.dump({"result": result, "predictions": local}, f)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results["repeats"][str(rep)] = rep_res
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=1)
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys())); w.writeheader(); w.writerows(pred_rows)
    summarize(args.out)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _agg(vals: list[float]) -> str:
    v = [x for x in vals if x is not None and not math.isnan(x)]
    if not v:
        return "n/a"
    return f"{np.mean(v):.3f}" + (f" ± {np.std(v):.3f}" if len(v) > 1 else "")


def summarize(out: str) -> str:
    """Write ``summary.md`` (+ ROC figure) from ``results.json`` / ``predictions.csv``."""
    res = json.load(open(os.path.join(out, "results.json")))
    reps = list(res["repeats"].values())
    L = [f"# Cohort experiment summary",
         f"\nSeed {res['seed']} (repeat *k* uses seed+*k*) · {res['n_subjects']} subjects · cohorts: {', '.join(res['cohorts'])} · "
         f"{len(reps)} repeat(s). Mean ± sd over repeats. Labels: 0 = control, 1 = patient.\n",
         "> **How to read this.** Test sets contain only tens of subjects, so single numbers are noisy; "
         "chance balanced accuracy is 0.50. \"patient\" means a different disorder in each cohort, so "
         "cross-cohort transfer (LODO) is a hard, honest test.\n"]
    probe = res.get("identity_probe", {})
    if probe.get("accuracy") is not None:
        L.append(f"**Cohort-identity probe:** a linear model on the 1-D spectrum alone identifies the cohort with "
                 f"{probe['accuracy']:.1%} accuracy (majority baseline {probe['chance']:.1%}). If this is near 100%, the "
                 "model *could* use cohort identity as a shortcut; it only helps if label balance differs between cohorts.\n")

    def table(title, exp, split_keys):
        L.append(f"\n## {title}\n")
        L.append("| split | n | head | accuracy | balanced acc | AUC | sensitivity | specificity |")
        L.append("|---|---|---|---|---|---|---|---|")
        for sk in split_keys:
            rs = [r[exp]["splits"].get(sk) for r in reps if exp in r and r[exp]["splits"].get(sk)]
            if not rs:
                continue
            for h, label in (("fused", "**CAFN fused**"), ("spatial", "MRI-only head"), ("temporal", "spectrum-only head"), ("spectrum_logreg", "spectrum logistic-reg baseline")):
                ms = [r[h] for r in rs if h in r]
                if not ms:
                    continue
                L.append(f"| {sk} | {rs[0]['n']} | {label} | " + " | ".join(_agg([m[k] for m in ms]) for k in
                         ("accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity")) + " |")

    names = res["cohorts"]
    table("Pooled model — within-cohort results", "pooled",
          [f"{p}:{d}" for d in names + ["ALL"] for p in ("val", "test")])
    for d in names:
        if len(names) > 1:
            table(f"Leave-one-cohort-out — train on the other cohort(s), test on **{d}**", f"lodo_{d}", [f"test:{d}"])
    first = reps[0]
    L.append("\n## Runs\n")
    L.append("| repeat | experiment | train cohorts | epochs run | batch size | minutes |")
    L.append("|---|---|---|---|---|---|")
    for k, r in res["repeats"].items():
        for exp, v in r.items():
            if exp == "splits":
                continue
            L.append(f"| {k} | {exp} | {', '.join(v['train_cohorts'])} | {v['epochs_run']} | {v['batch_size']} | {v['minutes']} |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(text)
    _plot_roc(out)
    return text


def _plot_roc(out: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception:
        return
    rows = list(csv.DictReader(open(os.path.join(out, "predictions.csv"))))
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["split"].startswith("test:") and r["split"] != "test:ALL" or r["experiment"].startswith("lodo"):
            groups.setdefault((r["experiment"], r["split"]), []).append(r)
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (exp, split), rs in sorted(groups.items()):
        y = np.array([int(r["label"]) for r in rs]); p = np.array([float(r["p_fused"]) for r in rs])
        if len(set(y.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{exp} · {split.split(':')[1]} (AUC {_auc(y, p):.2f}, n={len(y)})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("Test ROC (fused head, all repeats pooled)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "roc_test.png"), dpi=140); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--out", default="results/cohorts")
    ap.add_argument("--repeats", type=int, default=DEFAULTS["repeats"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--select-metric", default=DEFAULTS["select_metric"],
                    choices=("val_loss", "val_balanced_accuracy", "val_macro_f1"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 1 repeat: plumbing check only")
    args = ap.parse_args(argv)
    if args.smoke:
        args.epochs, args.repeats, args.patience = 2, 1, 1
    run_all(args)
    print(f"\nsummary: {os.path.join(args.out, 'summary.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
