"""Generate docs/journal_ieee_access/parts/p2b_synthetic.tex from the raw synthetic results.

Every number in the generated prose is computed here from docs/results_raw/synth_main and synth_sweep,
so the paper text cannot drift from the data. Re-run after any new runs.
"""

import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results_raw")
OUT = os.path.join(HERE, "journal_ieee_access", "parts", "p2b_synthetic.tex")


def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, d, "*.json")))]


main, sweep = load("synth_main"), load("synth_sweep")
acc = {}
for r in main:
    acc.setdefault(r["variant"], {})[r["seed"]] = r["test"]["fused"]["accuracy"] * 100


def ms(name):
    v = np.array(list(acc[name].values()))
    return v.mean(), v.std(ddof=1), len(v)


def paired(a, b):
    common = sorted(set(acc[a]) & set(acc[b]))
    d = np.array([acc[a][s] - acc[b][s] for s in common])
    p = stats.ttest_rel([acc[a][s] for s in common], [acc[b][s] for s in common]).pvalue if len(common) >= 3 else float("nan")
    return d.mean(), d.std(ddof=1) if len(d) > 1 else 0.0, len(common), p


full = ms("CAFN (full)")
mri, vep = ms("MRI only"), ms("VEP only")
catn, summ = ms("Concatenation"), ms("Element-wise sum")
best_uni = "MRI only" if mri[0] >= vep[0] else "VEP only"
g_uni = paired("CAFN (full)", best_uni)
g_cat, g_sum = paired("CAFN (full)", "Concatenation"), paired("CAFN (full)", "Element-wise sum")
g_v1 = paired("v1-style (all off)", "CAFN (full)")
abl_names = ["w/o bidirectional", "w/o positional enc.", "w/o attention pooling", "w/o SE blocks", "w/o auxiliary loss", "2 fusion blocks", "v1-style (all off)"]
abl = {n: paired(n, "CAFN (full)") for n in abl_names if n in acc}   # variant minus full
worst = min(abl.items(), key=lambda kv: kv[1][0])
n_sig = sum(1 for v in abl.values() if v[3] < 0.05)
sd_all = np.mean([ms(n)[1] for n in abl_names if n in acc] + [full[1]])


def pq(p):
    return "n/a" if np.isnan(p) else ("<0.001" if p < 0.001 else f"={p:.3f}")


def pv(p):
    return "n/a" if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}".lstrip("0") if False else f"{p:.3f}")


# sweep
vs = sorted({r["vep_strength"] for r in sweep})
sw = {m: {v: [r["test"]["fused"]["accuracy"] * 100 for r in sweep if r["variant"] == m and r["vep_strength"] == v] for v in vs}
      for m in ("MRI only", "VEP only", "Concatenation", "CAFN (full)")}
def sm(m, v):
    a = np.array(sw[m][v]); return a.mean()
gain_rows = []
for v in vs:
    best = max(sm("MRI only", v), sm("VEP only", v))
    gain_rows.append((v, sm("CAFN (full)", v) - best, sm("Concatenation", v) - best, best))
n_sw = len(next(iter(sw["CAFN (full)"].values())))

tex = rf"""\subsection{{Study A: when does fusion help? (synthetic benchmark)}}\label{{sec:studyA}}
Table~\ref{{tab:ablation}} and Fig.~\ref{{fig:ablation}} compare twelve configurations at the preset difficulty ($\mu_{{\mathrm{{MRI}}}}{{=}}1.0$, $\mu_{{\mathrm{{VEP}}}}{{=}}0.3$; chance is 33.3\%). Two results are clear and one is not.

\emph{{Fusion helps, a lot.}} The two unimodal references reach only {mri[0]:.1f}\% (MRI only, $n{{=}}{mri[2]}$ seeds) and {vep[0]:.1f}\% (VEP only, $n{{=}}{vep[2]}$), whereas every fusion model is at {min(full[0], catn[0], summ[0]):.1f}--{max(full[0], catn[0], summ[0]):.1f}\%. The full CAFN improves on the better unimodal model ({best_uni}) by {g_uni[0]:.1f} percentage points (paired over {g_uni[2]} seeds, SD {g_uni[1]:.1f}, paired $t$-test $p{pq(g_uni[3])}$). This confirms the premise of multimodal fusion in the setting where the evidence in the two modalities is complementary.

\emph{{The specific fusion mechanism does not matter at this scale.}} CAFN ({full[0]:.1f}$\pm${full[1]:.1f}\%, $n{{=}}{full[2]}$) is statistically indistinguishable from concatenation ({catn[0]:.1f}$\pm${catn[1]:.1f}\%; paired difference {g_cat[0]:+.1f}$\pm${g_cat[1]:.1f}, $p{pq(g_cat[3])}$) and from element-wise sum ({summ[0]:.1f}$\pm${summ[1]:.1f}\%; {g_sum[0]:+.1f}$\pm${g_sum[1]:.1f}, $p{pq(g_sum[3])}$). Removing any single design choice (bidirectionality, positional encoding, attention pooling, SE blocks, the auxiliary loss, or adding a second fusion block) changes the mean accuracy by between {min(v[0] for v in abl.values()):+.1f} and {max(v[0] for v in abl.values()):+.1f} points, whose magnitude is smaller than the seed-to-seed standard deviation (about {sd_all:.1f} points); {n_sig} of {len(abl)} of these differences is nominally significant ($p<0.05$, uncorrected). The v1-style configuration with everything switched off scores {ms('v1-style (all off)')[0]:.1f}\% ($\Delta={g_v1[0]:+.1f}$, $p{pq(g_v1[3])}$), but with a much larger spread ({ms('v1-style (all off)')[1]:.1f} versus {full[1]:.1f} points), which is in line with the training instability we observed earlier for the v1 recipe. We therefore make \emph{{no}} claim that gated bidirectional cross-attention improves accuracy over simpler fusion on this benchmark.

\begin{{table}}[!t]
\caption{{Study A (synthetic, three classes, $\mu_{{\mathrm{{MRI}}}}{{=}}1.0$, $\mu_{{\mathrm{{VEP}}}}{{=}}0.3$): test performance of the fused head (mean $\pm$ SD over seeds). $\Delta$: mean paired difference in accuracy to the full CAFN, in percentage points.}}
\label{{tab:ablation}}
\centering\scriptsize\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
Configuration & Params (M) & Acc.\ (\%) & Macro-F1 (\%) & $\Delta$\\
\midrule
\input{{results_assets/tab_ablation}}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[!t]
\centering
\includegraphics[width=\columnwidth]{{results_assets/fig_ablation.png}}
\caption{{Study A: test accuracy by configuration (bars: mean, whiskers: SD over seeds; dotted line: chance). Grey: unimodal references.}}
\label{{fig:ablation}}
\end{{figure}}

\subsection{{Study A: how informative must each modality be?}}
Fig.~\ref{{fig:sweep}} varies $\mu_{{\mathrm{{VEP}}}}$ with the MRI fixed ($n{{=}}{n_sw}$ seeds per point; four methods). The VEP-only model rises from chance ({sm('VEP only', vs[0]):.0f}\%) to {sm('VEP only', vs[-1]):.0f}\% while the MRI-only model stays at {min(sm('MRI only', v) for v in vs):.0f}--{max(sm('MRI only', v) for v in vs):.0f}\%. The gain of the fused models over the better single modality is largest at intermediate informativeness and vanishes when one modality dominates: {', '.join(f'{g[1]:+.0f} points (CAFN) and {g[2]:+.0f} (concatenation) at $\\mu_{{\\mathrm{{VEP}}}}{{=}}{g[0]:g}$' for g in gain_rows)}. Fusion therefore never hurt here, helps when both streams carry independent evidence, and converges to the dominant stream otherwise; CAFN and concatenation track each other throughout. With only {n_sw} seeds per point the intervals are wide and the curves should be read as a trend, not as precise estimates.

\begin{{figure}}[!t]
\centering
\includegraphics[width=0.78\columnwidth]{{results_assets/fig_sweep.png}}
\caption{{Study A sweep: test accuracy versus VEP informativeness (MRI fixed at 1.0). Dashed: unimodal models; solid: fusion models.}}
\label{{fig:sweep}}
\end{{figure}}
"""
open(OUT, "w").write(tex)
print("wrote", OUT, "| full", full, "uni", mri, vep, "| gains", g_uni, g_cat, g_sum, "| abl range",
      min(v[0] for v in abl.values()), max(v[0] for v in abl.values()), "sig", n_sig)

# ---- macros for the abstract / discussion / conclusion -------------------------------------------
def f1(x):
    return f"{x:.1f}"
macros = {
    "accFull": f1(full[0]), "sdFull": f1(full[1]), "nFull": str(full[2]),
    "accMRI": f1(mri[0]), "accVEP": f1(vep[0]), "accCat": f1(catn[0]), "accSum": f1(summ[0]),
    "gainUni": f1(g_uni[0]), "gainUniP": pq(g_uni[3]), "nGainUni": str(g_uni[2]),
    "dCat": f"{g_cat[0]:+.1f}", "pCat": pq(g_cat[3]), "dSum": f"{g_sum[0]:+.1f}", "pSum": pq(g_sum[3]),
    "ablMin": f"{min(v[0] for v in abl.values()):+.1f}", "ablMax": f"{max(v[0] for v in abl.values()):+.1f}",
    "ablSig": str(n_sig), "ablN": str(len(abl)), "sdAll": f1(sd_all),
    "accVone": f1(ms("v1-style (all off)")[0]), "sdVone": f1(ms("v1-style (all off)")[1]),
    "nSeedsMax": str(max(len(v) for v in acc.values())),
}
open(os.path.join(HERE, "journal_ieee_access", "numbers_results.tex"), "w").write(
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
print("macros", macros)
