"""Aggregate every experiment into figures, LaTeX tables and one facts.json.

Inputs (copied from the GPU box into docs/results_raw/):
  synth_main/   12 variants x 5 seeds at the preset difficulty (MRI 1.0 : VEP 0.3)
  synth_sweep/  4 methods x 5 VEP strengths x 3 seeds
  real_cv_classical/results.json   classical baselines, 5x5-fold CV + permutation tests + LODO
  real_cv_neural/results.json      neural variants, 3x5-fold CV
  cohorts/                         pooled / leave-one-cohort-out CAFN runs (results.json, r0/*/history.json)

Outputs go to docs/results_assets/. Nothing here invents numbers: every value is read from those files.
"""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results_raw")
OUT = os.path.join(HERE, "results_assets")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
COL = {"CAFN (full)": "#1d4ed8", "Concatenation": "#d97706", "MRI only": "#6b7280", "VEP only": "#059669"}
facts: dict = {}


def load_runs(d):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, d, "*.json")))]


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ---------------------------------------------------------------- synthetic ablation
def synthetic_main():
    runs = load_runs("synth_main")
    if not runs:
        return
    order = ["CAFN (full)", "w/o bidirectional", "w/o positional enc.", "w/o attention pooling", "w/o SE blocks",
             "w/o auxiliary loss", "2 fusion blocks", "v1-style (all off)", "Concatenation", "Element-wise sum", "MRI only", "VEP only"]
    by = {n: {} for n in order}
    for r in runs:
        by[r["variant"]][r["seed"]] = r
    seeds = sorted(by["CAFN (full)"])
    rows, tex = [], []
    full = np.array([by["CAFN (full)"][s]["test"]["fused"]["accuracy"] for s in seeds])
    for n in order:
        have = [s for s in seeds if s in by[n]]
        if not have:
            continue
        acc = np.array([by[n][s]["test"]["fused"]["accuracy"] for s in have])
        f1 = np.array([by[n][s]["test"]["fused"]["macro_f1"] for s in have])
        params = by[n][have[0]]["params"]
        diff = acc - np.array([by["CAFN (full)"][s]["test"]["fused"]["accuracy"] for s in have])
        rows.append(dict(variant=n, n_seeds=len(have), acc_mean=ms(acc)[0], acc_sd=ms(acc)[1], f1_mean=ms(f1)[0], f1_sd=ms(f1)[1],
                         params=params, diff_vs_full_mean=float(diff.mean()), wins_vs_full=int((diff > 0).sum()), per_seed=acc.tolist()))
        d = "--" if n == "CAFN (full)" else f"{100*diff.mean():+.1f}"
        tex.append(f"{n} & {params/1e6:.2f} & ${100*ms(acc)[0]:.1f}\\pm{100*ms(acc)[1]:.1f}$ & ${100*ms(f1)[0]:.1f}\\pm{100*ms(f1)[1]:.1f}$ & {d} \\\\")
    facts["synthetic_main"] = rows
    open(os.path.join(OUT, "tab_ablation.tex"), "w").write("\n".join(tex))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(rows))[::-1]
    colors = ["#1d4ed8" if r["variant"] == "CAFN (full)" else "#9ca3af" if r["variant"] in ("MRI only", "VEP only") else "#60a5fa" for r in rows]
    ax.barh(y, [100 * r["acc_mean"] for r in rows], xerr=[100 * r["acc_sd"] for r in rows], color=colors, capsize=2)
    ax.set_yticks(y, [r["variant"] for r in rows]); ax.set_xlabel("Test accuracy (%), mean ± sd over seeds"); ax.axvline(100 / 3, ls=":", c="k", lw=.8)
    ax.set_xlim(20, 100)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ablation.png"), dpi=200); plt.close(fig)


def synthetic_sweep():
    runs = load_runs("synth_sweep")
    if not runs:
        return
    vs = sorted({r["vep_strength"] for r in runs})
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    table = {}
    for m in ["MRI only", "VEP only", "Concatenation", "CAFN (full)"]:
        mu, sd = [], []
        for v in vs:
            a = [r["test"]["fused"]["accuracy"] for r in runs if r["variant"] == m and r["vep_strength"] == v]
            mu.append(100 * np.mean(a) if a else np.nan); sd.append(100 * np.std(a, ddof=1) if len(a) > 1 else 0)
        table[m] = dict(mean=mu, sd=sd, n=[sum(1 for r in runs if r["variant"] == m and r["vep_strength"] == v) for v in vs])
        ax.errorbar(vs, mu, yerr=sd, label=m, color=COL[m], marker="o", ms=3, lw=1.5 if m == "CAFN (full)" else 1, capsize=2,
                    ls="--" if m in ("MRI only", "VEP only") else "-")
    ax.set_xlabel("VEP informativeness (MRI fixed at 1.0)"); ax.set_ylabel("Test accuracy (%)"); ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sweep.png"), dpi=220); plt.close(fig)
    facts["synthetic_sweep"] = dict(vep_strengths=vs, methods=table)


# ---------------------------------------------------------------- real data
def real():
    cl = os.path.join(RAW, "real_cv_classical", "results.json")
    nn = os.path.join(RAW, "real_cv_neural", "results.json")
    models = {}
    lodo = {}
    info = {}
    if os.path.exists(cl):
        r = json.load(open(cl)); models.update(r["models"]); lodo = r["lodo"]; info = {k: r[k] for k in ("n", "cohorts", "n_patient")}
    if os.path.exists(nn):
        r = json.load(open(nn)); models.update(r["models"])
    if not models:
        return
    rows, tex = [], []
    for name, m in models.items():
        auc = [p["auc"] for p in m["per_repeat"]]; ba = [p["balanced_accuracy"] for p in m["per_repeat"]]
        ci = m.get("ci95_repeat0", {})
        rows.append(dict(model=name, auc_mean=float(np.mean(auc)), auc_sd=float(np.std(auc, ddof=1)) if len(auc) > 1 else 0.0,
                         bacc_mean=float(np.mean(ba)), bacc_sd=float(np.std(ba, ddof=1)) if len(ba) > 1 else 0.0,
                         auc_ci=ci.get("auc"), bacc_ci=ci.get("balanced_accuracy"), perm_p=m.get("perm_p_repeat0"), repeats=len(auc)))
        p = m.get("perm_p_repeat0")
        c = ci.get("auc")
        tex.append(f"{name.replace('&', '\\&')} & ${np.mean(auc):.3f}$ & " + (f"[{c[0]:.2f}, {c[1]:.2f}]" if c else "--") +
                   f" & ${np.mean(ba):.3f}$ & " + (f"{p:.3f}" if p is not None else "--") + " \\\\")
    facts["real_cv"] = dict(info=info, models=rows)
    open(os.path.join(OUT, "tab_real_cv.tex"), "w").write("\n".join(tex).replace("·", "$\\cdot$").replace("³", "$^3$").replace("→", "$\\rightarrow$"))
    # forest plot
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        c = r["auc_ci"]
        neural = r["model"] in ("CAFN (ours)", "Concatenation fusion", "MRI only", "Spectrum only")
        ax.plot([c[0], c[1]] if c else [r["auc_mean"]] * 2, [yi, yi], c="#1d4ed8" if neural else "#6b7280", lw=2)
        ax.plot(r["auc_mean"], yi, "o", c="#1d4ed8" if neural else "#6b7280")
    ax.axvline(0.5, ls=":", c="k"); ax.set_yticks(y, [r["model"] for r in rows]); ax.set_xlabel("Cross-validated ROC-AUC (dot: mean, bar: 95% CI)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_real_cv.png"), dpi=200); plt.close(fig)
    # LODO table
    if lodo:
        names = sorted({c for v in lodo.values() for c in v})
        lt = []
        for n, v in lodo.items():
            lt.append(f"{n.replace('·','$\\cdot$').replace('³','$^3$')} & " + " & ".join(f"{v[c]['auc']:.2f} / {v[c]['balanced_accuracy']:.2f}" for c in names) + " \\\\")
        open(os.path.join(OUT, "tab_lodo.tex"), "w").write("\n".join(lt).replace("→", "$\\rightarrow$"))
        facts["real_lodo_classical"] = lodo


def cohorts():
    f = os.path.join(RAW, "cohorts", "results.json")
    if not os.path.exists(f):
        return
    r = json.load(open(f))
    facts["cohort_cafn"] = r
    h = os.path.join(RAW, "cohorts", "r0", "pooled", "history.json")
    if os.path.exists(h):
        hh = json.load(open(h))
        fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
        e = np.arange(1, len(hh["train_loss"]) + 1)
        ax[0].plot(e, hh["train_loss"], label="train"); ax[0].plot(e, hh["val_loss"], label="val"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
        ax[1].plot(e, hh["train_accuracy"], label="train"); ax[1].plot(e, hh["val_accuracy"], label="val"); ax[1].axhline(.5, ls=":", c="k"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_real_curves.png"), dpi=200); plt.close(fig)
    # pooled / LODO summary table (mean over repeats)
    rows = []
    reps = list(r["repeats"].values())
    def agg(exp, split, head="fused"):
        v = [x[exp]["splits"][split][head] for x in reps if exp in x and split in x[exp]["splits"]]
        if not v:
            return None
        return {k: (float(np.mean([m[k] for m in v])), float(np.std([m[k] for m in v], ddof=1)) if len(v) > 1 else 0.0) for k in ("accuracy", "balanced_accuracy", "auc")}
    out = {}
    for exp, split in [("pooled", "test:ALL"), ("pooled", "test:ds003568"), ("pooled", "test:ds007181"), ("lodo_ds003568", "test:ds003568"), ("lodo_ds007181", "test:ds007181")]:
        out[f"{exp}|{split}"] = agg(exp, split)
    facts["cohort_cafn_summary"] = out
    lines = []
    for label, key in [("Pooled, test (both cohorts, $n{=}14$)", "pooled|test:ALL"), ("Pooled, test ds003568 ($n{=}7$)", "pooled|test:ds003568"),
                       ("Pooled, test ds007181 ($n{=}7$)", "pooled|test:ds007181"), ("Leave ds003568 out ($n{=}47$)", "lodo_ds003568|test:ds003568"),
                       ("Leave ds007181 out ($n{=}48$)", "lodo_ds007181|test:ds007181")]:
        a = out[key]
        if a:
            lines.append(f"{label} & ${a['accuracy'][0]:.2f}\\pm{a['accuracy'][1]:.2f}$ & ${a['balanced_accuracy'][0]:.2f}\\pm{a['balanced_accuracy'][1]:.2f}$ & ${a['auc'][0]:.2f}\\pm{a['auc'][1]:.2f}$ \\\\")
    open(os.path.join(OUT, "tab_cohort_cafn.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    synthetic_main(); synthetic_sweep(); real(); cohorts()
    json.dump(facts, open(os.path.join(OUT, "facts.json"), "w"), indent=1)
    print("wrote", OUT, sorted(os.listdir(OUT)))
