"""Reproducible measurements for the project report.

Runs the project's OWN signal-conditioning and synthetic-data code
(src/preprocessing/signal_cleaner.py, src/data/synthetic.py,
src/preprocessing/mri_transforms.foreground_mask_threshold) and writes
figures + results.json.  Needs only numpy, scipy, matplotlib: when torch /
monai / nibabel are not installed, light stub modules are injected so the
pure-numpy functions can be imported unchanged.

    NEURO_REPO=/path/to/NeuroImaging python report_experiments.py <out_dir>
"""
import json
import os
import sys
import types

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sps

REPO = os.environ.get("NEURO_REPO", "/home/claude/mishthijainn/neuroimaging")
OUT = sys.argv[1] if len(sys.argv) > 1 else "report_assets"
os.makedirs(OUT, exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, REPO)


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


try:
    import nibabel  # noqa: F401
except Exception:
    _stub("nibabel", Nifti1Image=object)
try:
    import monai.transforms  # noqa: F401
except Exception:
    _stub("monai")
    _stub("monai.transforms", Resize=object, ResizeWithPadOrCrop=object, ScaleIntensity=object)

from src.config import DEFAULT_CONFIG, CLASS_NAMES  # noqa: E402
from src.data.synthetic import synthetic_mri_volume, synthetic_vep_signal  # noqa: E402
from src.preprocessing.mri_transforms import foreground_mask_threshold  # noqa: E402
from src.preprocessing.signal_cleaner import (  # noqa: E402
    butterworth_bandpass,
    clean_vep_signal,
    notch_filter,
)

plt.rcParams.update(
    {
        "font.family": "Liberation Serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 100,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)
COL = {0: "#009E73", 1: "#E69F00", 2: "#0072B2"}  # Okabe-Ito: healthy / glaucoma / neuritis
LS = {0: "-", 1: "--", 2: "-."}
SHORT = {0: "Healthy", 1: "Glaucoma", 2: "Optic neuritis / CVI"}
TRUE_LAT = {0: 100.0, 1: 108.0, 2: 140.0}  # ms, from src/data/synthetic.py
FS = DEFAULT_CONFIG.vep.sample_rate
N = 500
T_MS = np.arange(N) / FS * 1000.0
res = {}


def p100_ms(x, lo=60.0, hi=220.0):
    m = (T_MS >= lo) & (T_MS <= hi)
    return float(T_MS[m][np.argmax(x[m])])


def ls_amp(x, f):
    """Amplitude of the best-fit sinusoid at frequency f (least squares)."""
    t = np.arange(len(x)) / FS
    A = np.c_[np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)]
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return float(np.hypot(c[0], c[1]))


# ------------------------------------------------------------------ VEP
NTR = 50
sos = sps.butter(4, [1.0, 100.0], btype="bandpass", fs=FS, output="sos")
bn, an = sps.iirnotch(50.0, 30.0, FS)
rows = {c: {"raw": [], "zp": [], "causal": [], "peak_z": [], "d50": [], "a50_before": [], "a50_after": [], "a47_after": []} for c in range(3)}
example = {}
for c in range(3):
    for k in range(NTR):
        rng = np.random.default_rng(1000 * c + k)
        x = synthetic_vep_signal(c, fs=FS, rng=rng)
        bp = butterworth_bandpass(x, FS, 1.0, 100.0)
        zp = notch_filter(bp, FS, 50.0, 30.0)  # zero-phase pipeline before resample/z-score
        causal = sps.lfilter(bn, an, sps.sosfilt(sos, x))  # ordinary one-pass filtering
        cleaned = clean_vep_signal(x, FS, DEFAULT_CONFIG.vep)
        rows[c]["raw"].append(p100_ms(x))
        rows[c]["zp"].append(p100_ms(zp))
        rows[c]["causal"].append(p100_ms(causal))
        m = (T_MS >= 60) & (T_MS <= 220)
        rows[c]["peak_z"].append(float(cleaned[m].max()))
        rows[c]["d50"].append(20 * np.log10(ls_amp(zp, 50.0) / ls_amp(x, 50.0)))
        rows[c]["a50_before"].append(ls_amp(x, 50.0))
        rows[c]["a50_after"].append(ls_amp(zp, 50.0))
        rows[c]["a47_after"].append(ls_amp(zp, 47.0))
        if k == 0:
            example[c] = (x, causal, cleaned)

summ = {}
for c in range(3):
    r = rows[c]
    summ[c] = {
        "true_ms": TRUE_LAT[c],
        "raw_mean": np.mean(r["raw"]), "raw_sd": np.std(r["raw"]),
        "zp_mean": np.mean(r["zp"]), "zp_sd": np.std(r["zp"]),
        "causal_mean": np.mean(r["causal"]), "causal_sd": np.std(r["causal"]),
        "zp_err": np.mean(np.array(r["zp"]) - TRUE_LAT[c]),
        "causal_err": np.mean(np.array(r["causal"]) - TRUE_LAT[c]),
        "zp_maxabs": float(np.max(np.abs(np.array(r["zp"]) - TRUE_LAT[c]))),
        "causal_maxabs": float(np.max(np.abs(np.array(r["causal"]) - TRUE_LAT[c]))),
        "raw_maxabs": float(np.max(np.abs(np.array(r["raw"]) - TRUE_LAT[c]))),
        "zp_within4": float(np.mean(np.abs(np.array(r["zp"]) - TRUE_LAT[c]) <= 4.0)),
        "causal_within4": float(np.mean(np.abs(np.array(r["causal"]) - TRUE_LAT[c]) <= 4.0)),
        "peak_z_mean": np.mean(r["peak_z"]), "peak_z_sd": np.std(r["peak_z"]),
        "d50_mean": np.mean(r["d50"]), "a50_before": np.mean(r["a50_before"]), "a50_after": np.mean(r["a50_after"]), "a47_after": np.mean(r["a47_after"]),
    }
res["vep"] = {"n_trials_per_class": NTR, "fs": FS, "samples": N, "per_class": summ}
res["vep"]["all_d50_mean"] = float(np.mean([v for c in range(3) for v in rows[c]["d50"]]))
for k in ("a50_before", "a50_after", "a47_after"):
    res["vep"]["all_" + k] = float(np.mean([v for c in range(3) for v in rows[c][k]]))

# frequency response (forward-backward => squared magnitude)
w, h = sps.sosfreqz(sos, worN=8192, fs=FS)
wn, hn = sps.freqz(bn, an, worN=8192, fs=FS)
bp_db = 20 * np.log10(np.abs(h) ** 2 + 1e-12)
nt_db = 20 * np.log10(np.abs(hn) ** 2 + 1e-12)
tot_db = bp_db + nt_db


def at(f, arr):
    return float(arr[np.argmin(np.abs(w - f))])


res["freq_response_db"] = {str(f): {"bandpass": at(f, bp_db), "notch": at(f, nt_db), "combined": at(f, tot_db)}
                           for f in (0.3, 1.0, 10.0, 49.0, 50.0, 51.0, 100.0)}
# -6 dB (i.e. -3 dB per pass) bandwidth of the notch
ix = np.where((w > 40) & (w < 60) & (tot_db < -6.0))[0]
res["notch_minus6dB_band_hz"] = [float(w[ix[0]]), float(w[ix[-1]])]

fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.semilogx(w[1:], tot_db[1:], color="#1b1b1b", lw=1.6, label="Combined (bandpass + notch, zero-phase)")
ax.semilogx(w[1:], bp_db[1:], color="#0072B2", lw=1.0, ls="--", label="4th-order Butterworth 1–100 Hz")
ax.semilogx(w[1:], nt_db[1:], color="#D55E00", lw=1.0, ls=":", label="IIR notch, 50 Hz, Q = 30")
for f in (1, 50, 100):
    ax.axvline(f, color="#999999", lw=0.5)
ax.set_ylim(-80, 5)
ax.set_xlim(0.2, 125)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.savefig(f"{OUT}/fig_filter_response.png")
plt.close(fig)

# waveform figure: raw vs cleaned per class
fig, axes = plt.subplots(3, 2, figsize=(6.6, 5.6), sharex="col")
for c in range(3):
    x, causal, cleaned = example[c]
    a0, a1 = axes[c]
    a0.plot(T_MS, x, color="#777777", lw=0.7)
    a0.set_ylabel(SHORT[c], fontsize=8.5)
    a1.plot(T_MS, cleaned, color=COL[c], lw=1.1, ls="-")
    a1.axvline(TRUE_LAT[c], color="#444444", lw=0.6, ls=":")
    a1.annotate("P100", (TRUE_LAT[c], cleaned[int(TRUE_LAT[c] / 1000 * FS)]), xytext=(TRUE_LAT[c] + 90, cleaned[int(TRUE_LAT[c] / 1000 * FS)] * 0.9),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444"))
    a0.set_xlim(0, 500)
axes[0, 0].set_title("Raw synthetic VEP (µV, arbitrary)", fontsize=9)
axes[0, 1].set_title("After clean_vep_signal (z-score)", fontsize=9)
axes[2, 0].set_xlabel("Time after stimulus (ms)")
axes[2, 1].set_xlabel("Time after stimulus (ms)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_vep_examples.png")
plt.close(fig)

# P100 latency: group delay of the single-pass filter vs zero-phase (0 ms)
wg = np.logspace(np.log10(0.5), np.log10(120), 600)
_, h_bp = sps.sosfreqz(sos, worN=wg, fs=FS)
_, h_nt = sps.freqz(bn, an, worN=wg, fs=FS)
phase = np.unwrap(np.angle(h_bp * h_nt))
gd_ms = -np.gradient(phase, 2 * np.pi * wg) * 1000.0
res["group_delay_ms"] = {str(f): float(np.interp(f, wg, gd_ms)) for f in (1, 2, 5, 10, 20, 30, 40, 60)}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})
a = axes[0]
a.semilogx(wg, gd_ms, color="#D55E00", lw=1.4, ls="--", label="Single-pass (causal)")
a.axhline(0, color="#0072B2", lw=1.4, label="Zero-phase (project)")
a.axhspan(-4, 4, color="#cccccc", alpha=0.45, lw=0)
a.text(1.7, 5.5, "±1 sample (4 ms)", fontsize=7.5, color="#555555")
a.set_ylim(-10, 60)
a.set_xlim(1.5, 120)
a.set_xlabel("Frequency (Hz)")
a.set_ylabel("Group delay (ms)")
a.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.30, 1.0))
b = axes[1]
pos = np.arange(3)
wd = 0.27
b.bar(pos - wd, [summ[c]["raw_mean"] - TRUE_LAT[c] for c in range(3)], wd, color="#999999", label="Raw")
b.bar(pos, [summ[c]["zp_err"] for c in range(3)], wd, color="#0072B2", label="Zero-phase")
b.bar(pos + wd, [summ[c]["causal_err"] for c in range(3)], wd, color="#D55E00", label="Causal")
b.axhline(0, color="#333333", lw=0.6)
b.axhspan(-4, 4, color="#cccccc", alpha=0.45, lw=0)
b.set_ylim(-5, 5)
b.set_xticks(pos, ["Healthy", "Glauc.", "ON/CVI"], fontsize=8)
b.set_ylabel("Mean P100 latency error (ms)")
b.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_p100_shift.png")
plt.close(fig)

# spectrum before / after
fig, ax = plt.subplots(figsize=(6.4, 2.8))
x = example[0][0]
zp0 = notch_filter(butterworth_bandpass(x, FS, 1.0, 100.0), FS, 50.0, 30.0)
fr = np.fft.rfftfreq(N, 1 / FS)
ax.semilogy(fr, np.abs(np.fft.rfft(x - x.mean())) / N * 2, color="#999999", lw=0.9, label="Raw")
ax.semilogy(fr, np.abs(np.fft.rfft(zp0 - zp0.mean())) / N * 2 + 1e-6, color="#0072B2", lw=1.0, label="Conditioned")
ax.axvline(50, color="#D55E00", lw=0.7, ls=":")
ax.text(51, 1.2, "50 Hz mains", fontsize=8, color="#D55E00")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude spectrum")
ax.set_xlim(0, 125)
ax.set_ylim(1e-4, 5)
ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
fig.savefig(f"{OUT}/fig_spectrum.png")
plt.close(fig)

# per-class distributions (latency + peak z)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
for c in range(3):
    axes[0].hist(rows[c]["zp"], bins=np.arange(90, 160, 2), color=COL[c], alpha=0.8, label=SHORT[c], histtype="stepfilled", edgecolor="white", lw=0.4)
    axes[1].hist(rows[c]["peak_z"], bins=14, color=COL[c], alpha=0.8, histtype="stepfilled", edgecolor="white", lw=0.4)
axes[0].set_xlabel("Detected P100 latency (ms)")
axes[0].set_ylabel("Trials")
axes[1].set_xlabel("Peak value after z-scoring")
axes[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_class_distributions.png")
plt.close(fig)

# ------------------------------------------------------------------ MRI
shape = (80, 96, 64)
vols = {c: synthetic_mri_volume(c, shape=shape, rng=np.random.default_rng(7 + c)) for c in range(3)}
d, hh, ww = shape
roi_atrophy = (slice(d // 2, None), slice(None, hh // 2), slice(None, ww // 2))
roi_lesion = (slice(None, d // 2), slice(hh // 2, None), slice(ww // 2, None))
mask_all = np.ones(shape, bool)
mask_all[roi_atrophy] = False
mask_all[roi_lesion] = False
mri = {}
for c in range(3):
    v = vols[c]
    masked = foreground_mask_threshold(v, DEFAULT_CONFIG.mri.foreground_percentile)
    mri[c] = {
        "mean_background": float(v[mask_all].mean()),
        "mean_atrophy_roi": float(v[roi_atrophy].mean()),
        "mean_lesion_roi": float(v[roi_lesion].mean()),
        "fraction_zeroed_by_mask": float((masked == 0).mean()),
        "retained_atrophy_roi": float((masked[roi_atrophy] > 0).mean()),
        "retained_lesion_roi": float((masked[roi_lesion] > 0).mean()),
        "retained_background": float((masked[mask_all] > 0).mean()),
    }
res["mri"] = {"raw_shape": shape, "per_class": mri}

fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4))
for c in range(3):
    for r, (di, lab) in enumerate([(20, "axial index 20"), (60, "axial index 60")]):
        ax = axes[r, c]
        ax.imshow(vols[c][di], cmap="gray", vmin=0, vmax=260)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
        if r == 0:
            ax.set_title(SHORT[c], fontsize=9)
        if c == 0:
            ax.set_ylabel(lab, fontsize=8.5)
# mark ROIs
axes[0, 2].add_patch(plt.Rectangle((ww // 2 - 0.5, hh // 2 - 0.5), ww // 2, hh // 2, fill=False, ec="#E69F00", lw=1.4, ls="--"))
axes[1, 1].add_patch(plt.Rectangle((-0.5, -0.5), ww // 2, hh // 2, fill=False, ec="#E69F00", lw=1.4, ls="--"))
fig.tight_layout()
fig.savefig(f"{OUT}/fig_synthetic_mri.png")
plt.close(fig)

# token <-> time mapping schematic (nominal)
fig, ax = plt.subplots(figsize=(6.6, 2.3))
x = example[0][2]
ax.plot(T_MS, x, color="#1b1b1b", lw=0.9)
ntok = DEFAULT_CONFIG.model.temporal_tokens
binw = 2000.0 / ntok
for i in range(ntok + 1):
    ax.axvline(i * binw, color="#bbbbbb", lw=0.5)
for i in range(ntok):
    ax.text(i * binw + binw / 2, 9.3, str(i), ha="center", va="top", fontsize=7)
ax.set_ylim(-3.5, 9.5)
ax.set_xlim(0, 2000)
ax.set_xlabel("Time after stimulus (ms)")
ax.set_ylabel("z-score")
ax.axvspan(0, binw, color="#0072B2", alpha=0.12)
fig.savefig(f"{OUT}/fig_token_time.png")
plt.close(fig)

res["token_time"] = {"tokens": ntok, "nominal_bin_ms": binw, "p100_token_idx": {SHORT[c]: int(TRUE_LAT[c] // binw) for c in range(3)}}
res["config"] = {"classes": list(CLASS_NAMES)}

# JSON-safe
def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


with open(f"{OUT}/results.json", "w") as f:
    json.dump(conv(res), f, indent=1)
print(json.dumps(conv(res), indent=1))
