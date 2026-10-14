"""Diagrams and computed-data charts for the project report (matplotlib only)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8.5, "savefig.dpi": 220, "savefig.bbox": "tight", "axes.linewidth": 0.8})

C = dict(
    sp=("#DCE9F5", "#2B5C8A"),   # spatial / MRI
    tp=("#FBE9CC", "#A8691A"),   # temporal / VEP
    fu=("#DDF0E6", "#2E7D5B"),   # fusion
    hd=("#EDEDED", "#555555"),   # heads / neutral
    ex=("#ECE0F3", "#6B4C85"),   # explainability
    io=("#FFFFFF", "#333333"),   # io
)


def canvas(w, h, wi=6.4):
    fig, ax = plt.subplots(figsize=(wi, wi * h / w))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind="io", fs=8, bold=False, ls="-"):
    fc, ec = C[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=1.0, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.15)
    return (x, y, w, h)


def arrow(ax, pts, label=None, lpos=0.5, fs=7, off=(0, 1.3), color="#333333", ls="-", ha="center"):
    """Polyline with arrow head on the last segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.0, ls=ls, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=9))
    if label:
        a, b = pts[0], pts[1]
        mx, my = a[0] + (b[0] - a[0]) * lpos, a[1] + (b[1] - a[1]) * lpos
        ax.text(mx + off[0], my + off[1], label, fontsize=fs, ha=ha, va="bottom", color="#222222", style="italic")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)


# ----------------------------------------------------------- 1. pipeline
fig, ax = canvas(100, 62)
ax.set_ylim(-2, 60)
box(ax, 0, 38, 16.5, 13, "3D structural\nMRI\n(NIfTI)", "sp", fs=7.4)
box(ax, 0, 8, 16.5, 13, "1D visual evoked\npotential\n(CSV, 250 Hz)", "tp", fs=7.4)
box(ax, 19, 35, 22, 19, "MRI preprocessing\nRAS · 1 mm · mask\nscale [0,1] · crop/pad\n→ (1, 64, 64, 64)", "sp", fs=7.4)
box(ax, 19, 5, 22, 19, "VEP preprocessing\n1–100 Hz bandpass\n50 Hz notch · z-score\n→ (1, 500)", "tp", fs=7.4)
box(ax, 46, 38, 14, 13, "3D residual\nCNN encoder", "sp")
box(ax, 46, 8, 14, 13, "1D CNN\nencoder", "tp")
box(ax, 66, 20, 14, 18, "Cross-\nattention\nfusion\n(CAFM)", "fu", bold=True)
box(ax, 85, 42, 15, 9, "Spatial head\n(auxiliary)", "hd")
box(ax, 85, 25, 15, 9, "Fused head\n(primary)", "hd", bold=True)
box(ax, 85, 8, 15, 9, "Temporal head\n(auxiliary)", "hd")
arrow(ax, [(16.5, 44.5), (19, 44.5)])
arrow(ax, [(16.5, 14.5), (19, 14.5)])
arrow(ax, [(41, 44.5), (46, 44.5)])
arrow(ax, [(41, 14.5), (46, 14.5)])
arrow(ax, [(60, 44.5), (63, 44.5), (63, 34), (66, 34)])
arrow(ax, [(60, 14.5), (63, 14.5), (63, 24), (66, 24)])
ax.text(62.2, 40.5, "Q", fontsize=7.5, ha="right", style="italic", fontweight="bold")
ax.text(62.2, 17.5, "K, V", fontsize=7.5, ha="right", style="italic", fontweight="bold")
arrow(ax, [(80, 29.5), (85, 29.5)])
arrow(ax, [(53, 51), (53, 56), (92.5, 56), (92.5, 51)])
arrow(ax, [(53, 8), (53, 2), (92.5, 2), (92.5, 8)])
ax.text(72, 56.8, "64 spatial tokens (B, 64, 128)", fontsize=6.8, ha="center", style="italic")
ax.text(72, -1.6, "16 temporal tokens (B, 16, 128)", fontsize=6.8, ha="center", style="italic")
box(ax, 64, 42, 18, 9, "Grad-CAM +\nattention heatmap", "ex", fs=7.4)
arrow(ax, [(73, 38), (73, 42)], ls="--", color="#6B4C85")
save(fig, "fig_pipeline")

# ----------------------------------------------------------- 2. architecture
fig, ax = canvas(100, 128, wi=5.6)
ax.text(24, 125, "Spatial stream (MRI)", ha="center", fontsize=9, fontweight="bold", color=C["sp"][1])
ax.text(76, 125, "Temporal stream (VEP)", ha="center", fontsize=9, fontweight="bold", color=C["tp"][1])
sp = ["Input MRI\n(B, 1, 64, 64, 64)", "Stem: Conv3d 1→16 + BN + ReLU\n(B, 16, 64, 64, 64)", "ResBlock 16→32, stride 2\n(B, 32, 32, 32, 32)",
      "ResBlock 32→64, stride 2\n(B, 64, 16, 16, 16)", "ResBlock 64→128, stride 2 [Grad-CAM layer]\n(B, 128, 8, 8, 8)", "AdaptiveAvgPool3d(4,4,4) + flatten\n(B, 64, 128)"]
tp_ = ["Input VEP\n(B, 1, 500)", "Conv1d 1→16, k=7, s=1 + BN + ReLU\n(B, 16, 500)", "Conv1d 16→32, k=5, s=2\n(B, 32, 250)",
       "Conv1d 32→64, k=5, s=2\n(B, 64, 125)", "Conv1d 64→128, k=3, s=2\n(B, 128, 63)", "AdaptiveAvgPool1d(16) + transpose\n(B, 16, 128)"]
ys = [110, 94, 78, 62, 46, 30]
H = 11
for t, y in zip(sp, ys):
    box(ax, 1, y, 46, H, t, "sp", fs=7)
for t, y in zip(tp_, ys):
    box(ax, 53, y, 46, H, t, "tp", fs=7)
for y_hi, y_lo in zip(ys[:-1], ys[1:]):
    for xc in (24, 76):
        arrow(ax, [(xc, y_hi), (xc, y_lo + H)])
box(ax, 14, 7, 72, 14, "Cross-attention fusion (CAFM)\nQ = spatial tokens · K, V = temporal tokens\n4 heads · residual + LayerNorm · FFN 128→256→128", "fu", fs=7.2)
arrow(ax, [(24, 30), (24, 26), (34, 26), (34, 21)])
arrow(ax, [(76, 30), (76, 26), (66, 26), (66, 21)])
ax.text(21.5, 26.7, "Q", fontsize=7.5, style="italic", fontweight="bold", ha="right")
ax.text(78.5, 26.7, "K, V", fontsize=7.5, style="italic", fontweight="bold", ha="left")
ax.text(52, 2.6, "fused tokens (B, 64, 128) → mean-pool → fused head", fontsize=6.8, style="italic", ha="left")
ax.set_ylim(-14, 128)
arrow(ax, [(50, 7), (50, 3.5)])
box(ax, 1, -13, 30, 9, "Spatial head\nLinear 128→3", "hd", fs=7)
box(ax, 35, -13, 30, 9, "Fused head\nLinear 128→3", "hd", fs=7, bold=True)
box(ax, 69, -13, 30, 9, "Temporal head\nLinear 128→3", "hd", fs=7)
arrow(ax, [(50, 3.5), (50, -4)])
arrow(ax, [(16, 30), (16, 27), (4, 27), (4, 3), (16, 3), (16, -4)], ls="--", color="#2B5C8A") if False else None
save(fig, "fig_architecture")

# ----------------------------------------------------------- 3. cross attention block
fig, ax = canvas(100, 46)
box(ax, 0, 33, 16, 9, "Spatial tokens\n(B, 64, 128)", "sp", fs=7.5)
box(ax, 0, 3, 16, 9, "Temporal tokens\n(B, 16, 128)", "tp", fs=7.5)
box(ax, 22, 33, 11, 9, "W_Q", "fu", fs=8)
box(ax, 22, 17, 11, 9, "W_K", "fu", fs=8)
box(ax, 22, 3, 11, 9, "W_V", "fu", fs=8)
arrow(ax, [(16, 37.5), (22, 37.5)])
arrow(ax, [(16, 10.5), (19, 10.5), (19, 21.5), (22, 21.5)])
arrow(ax, [(16, 6.5), (22, 6.5)])
box(ax, 39, 22, 20, 20, "Scaled dot-product\nscores = Q Kᵀ / √d_k\nsoftmax over the\n16 temporal tokens\n→ (B, 4, 64, 16)", "fu", fs=7.2)
arrow(ax, [(33, 37.5), (39, 37.5)])
arrow(ax, [(33, 21.5), (36, 21.5), (36, 27), (39, 27)])
box(ax, 39, 3, 20, 12, "Weighted sum\nattn · V\n(B, 4, 64, 32)", "fu", fs=7.2)
arrow(ax, [(49, 22), (49, 15)])
arrow(ax, [(33, 7.5), (39, 7.5)])
box(ax, 64, 22, 15, 20, "Concat heads\n+ W_O\n(B, 64, 128)", "fu", fs=7.2)
arrow(ax, [(59, 9), (61.5, 9), (61.5, 28), (64, 28)])
box(ax, 83, 31, 17, 11, "Add & LayerNorm\n(residual from\nspatial tokens)", "hd", fs=6.6)
box(ax, 83, 14, 17, 11, "FFN 128→256→128\n+ Add & LayerNorm", "hd", fs=6.2)
arrow(ax, [(79, 36.5), (83, 36.5)])
arrow(ax, [(91.5, 31), (91.5, 25)])
box(ax, 83, 0.5, 17, 9, "Fused tokens\n(B, 64, 128)", "fu", fs=7.2, bold=True)
arrow(ax, [(91.5, 14), (91.5, 9.5)])
arrow(ax, [(8, 42), (8, 45), (91.5, 45), (91.5, 42)], ls="--", color="#666666")
ax.text(50, 45.6, "residual connection", fontsize=6.8, ha="center", style="italic", color="#444444")
save(fig, "fig_cross_attention")

# ----------------------------------------------------------- 4. preprocessing chains
fig, ax = canvas(100, 41)
ax.text(0, 39, "MRI stream (src/preprocessing/mri_transforms.py)", fontsize=8.5, fontweight="bold", color=C["sp"][1], va="top")
mri_steps = ["1. Load NIfTI,\nreorient to RAS\n(nibabel)", "2. Resample to\n1 mm isotropic\n(MONAI Resize)", "3. Foreground mask\n(60th percentile\nof positive voxels)", "4. Scale to [0, 1]\n(MONAI\nScaleIntensity)", "5. Pad / crop to\n64×64×64, add\nchannel axis"]
w = 17.5
for i, t in enumerate(mri_steps):
    x = i * (w + 3)
    box(ax, x, 22, w, 12, t, "sp", fs=7)
    if i:
        arrow(ax, [(x - 3, 28), (x, 28)])
ax.text(0, 17, "VEP stream (src/preprocessing/signal_cleaner.py)", fontsize=8.5, fontweight="bold", color=C["tp"][1], va="top")
vep_steps = ["1. Validate\n(1-D, finite,\nnon-empty)", "2. Zero-phase\nbandpass 1–100 Hz\n(Butterworth, 4th)", "3. Zero-phase\n50 Hz notch\n(Q = 30)", "4. Linear resample\nto 500 samples", "5. Z-score\nnormalise,\nadd channel axis"]
for i, t in enumerate(vep_steps):
    x = i * (w + 3)
    box(ax, x, 0, w, 12, t, "tp", fs=7)
    if i:
        arrow(ax, [(x - 3, 6), (x, 6)])
save(fig, "fig_preprocessing")

# ----------------------------------------------------------- 5. training flow
fig, ax = canvas(100, 36)
steps = [("Dataset\n(paired or\nsynthetic)", "io"), ("Seeded split\n80 % / 20 %\n(seed 42)", "io"), ("DataLoader\nbatch = 8", "io"),
         ("CAFN forward\nfused + 2 aux\nlogits", "fu"), ("Multi-task loss\nL_f + 0.3\n(L_s + L_t)", "fu"), ("Backward +\nAdam update\nlr 1e-3, wd 1e-4", "hd")]
w = 14.5
for i, (t, k) in enumerate(steps):
    x = i * (w + 2.6)
    box(ax, x, 18, w, 13, t, k, fs=7)
    if i:
        arrow(ax, [(x - 2.6, 24.5), (x, 24.5)])
box(ax, 0, 0, 30, 10, "Per-epoch validation\n(fused loss, fused accuracy)", "hd", fs=7.2)
box(ax, 36, 0, 28, 10, "save_checkpoint\n(state dict + ModelConfig)", "hd", fs=7.2)
box(ax, 70, 0, 30, 10, "Explainability\nGrad-CAM + attention map", "ex", fs=7.2)
arrow(ax, [(91.5, 18), (91.5, 14), (15, 14), (15, 10)], ls="--", color="#666666")
ax.text(50, 14.8, "repeat for each of N epochs", fontsize=6.8, style="italic", ha="center", color="#444444")
arrow(ax, [(30, 5), (36, 5)])
arrow(ax, [(64, 5), (70, 5)])
save(fig, "fig_training_flow")

# ----------------------------------------------------------- 6. explainability
fig, ax = canvas(100, 44)
ax.text(0, 42.5, "A. 3-D Grad-CAM (fused logit)", fontsize=8.5, fontweight="bold", color=C["ex"][1], va="top")
g = ["Predicted class\nfrom fused head", "Backpropagate\nfused logit to\nlast residual block", "Channel weights =\nmean of gradients\nover 3-D space", "ReLU(Σ weight ×\nactivation)", "Upsample to\n64³, min–max\nnormalise"]
w = 17.5
for i, t in enumerate(g):
    x = i * (w + 3)
    box(ax, x, 22, w, 13, t, "ex", fs=7)
    if i:
        arrow(ax, [(x - 3, 28.5), (x, 28.5)])
ax.text(0, 17, "B. Cross-attention heatmap", fontsize=8.5, fontweight="bold", color=C["ex"][1], va="top")
b = ["attn_weights\n(B, 4, 64, 16)", "Mean over batch\nand heads", "Matrix (64, 16):\nspatial token ×\ntemporal token", "Heatmap\n(viridis)"]
for i, t in enumerate(b):
    x = i * (w + 3)
    box(ax, x, 0, w, 12, t, "ex", fs=7)
    if i:
        arrow(ax, [(x - 3, 6), (x, 6)])
save(fig, "fig_explainability")

# ----------------------------------------------------------- 7. software structure
fig, ax = canvas(100, 60)
ax.set_ylim(-8, 52)
box(ax, 30, 44, 40, 7, "config.py  (shapes, hyper-parameters)", "hd", fs=7.2)
box(ax, 0, 28, 28, 12, "preprocessing/\nmri_transforms.py\nsignal_cleaner.py", "sp", fs=7.2)
box(ax, 36, 28, 28, 12, "data/\nsynthetic.py\ndataset.py", "tp", fs=7.2)
box(ax, 66, 28, 34, 12, "models/\nspatial_ · temporal_encoder.py\ncross_attention.py · cafn.py", "fu", fs=6.8)
box(ax, 18, 8, 28, 12, "train.py\ntrain · evaluate · checkpoints", "hd", fs=7.2)
box(ax, 54, 8, 28, 12, "explainability/\ngrad_cam_3d.py\nplot_attention.py", "ex", fs=7.2)
box(ax, 0, -7, 100, 8, "tests/unit (78 tests)   ·   tests/integration (8 tests)   —   86 test functions in total", "io", fs=7.4)
for x in (14, 50, 86):
    arrow(ax, [(x, 44) if x == 50 else (50, 44), (x, 40)]) if False else None
arrow(ax, [(32, 44), (14, 44), (14, 40)])
arrow(ax, [(50, 44), (50, 40)])
arrow(ax, [(68, 44), (84, 44), (84, 40)])
arrow(ax, [(14, 28), (14, 24), (28, 24), (28, 20)])
arrow(ax, [(50, 28), (50, 24), (36, 24), (36, 20)])
arrow(ax, [(84, 28), (84, 24), (60, 24), (60, 20)], ls="-")
arrow(ax, [(84, 28), (84, 24), (70, 24), (70, 20)], ls="-")
arrow(ax, [(46, 14), (54, 14)])
save(fig, "fig_software_structure")

# ----------------------------------------------------------- 8. motivation
fig, ax = canvas(100, 36)
box(ax, 0, 20, 30, 15, "Structural MRI\nWHERE: lesions, atrophy,\nregional tissue change\nNo timing information", "sp", fs=7.2)
box(ax, 0, 0, 30, 15, "Visual evoked potential\nWHEN: P100 latency,\namplitude, conduction delay\nNo spatial localisation", "tp", fs=7.2)
box(ax, 40, 7, 28, 21, "Cross-attention fusion\nEach spatial token asks:\nwhich temporal features\nare relevant to me?", "fu", fs=7.4, bold=False)
box(ax, 77, 20, 23, 15, "Class prediction\nHealthy · Glaucoma ·\nON / CVI", "hd", fs=7.2)
box(ax, 77, 0, 23, 15, "Evidence for review\nGrad-CAM map +\nattention heatmap", "ex", fs=7.2)
arrow(ax, [(30, 27.5), (35, 27.5), (35, 21), (40, 21)])
arrow(ax, [(30, 7.5), (35, 7.5), (35, 14), (40, 14)])
arrow(ax, [(68, 21), (72, 21), (72, 27.5), (77, 27.5)])
arrow(ax, [(68, 14), (72, 14), (72, 7.5), (77, 7.5)])
save(fig, "fig_motivation")

# ----------------------------------------------------------- 9. data-source path
fig, ax = canvas(100, 36)
box(ax, 0, 22, 30, 12, "manifest.csv\nsubject_id, mri_path, vep_path,\nfs, label", "io", fs=7)
box(ax, 0, 2, 30, 12, "Synthetic generators\nsynthetic_mri_volume\nsynthetic_vep_signal", "io", fs=7)
box(ax, 40, 22, 28, 12, "PairedNeuroDataset\n(real files on disk;\nfuture labelled cohort)", "tp", fs=7.4)
box(ax, 40, 2, 28, 12, "SyntheticNeuroDataset\n(in-memory;\ncurrent prototype)", "tp", fs=7.4)
box(ax, 76, 10, 24, 16, "Same preprocessing\nMRI → (1,64,64,64)\nVEP → (1,500)\n→ DataLoader", "fu", fs=7.2)
arrow(ax, [(30, 28), (40, 28)])
arrow(ax, [(30, 8), (40, 8)])
arrow(ax, [(68, 28), (72, 28), (72, 21), (76, 21)])
arrow(ax, [(68, 8), (72, 8), (72, 15), (76, 15)])
save(fig, "fig_data_paths")

# ----------------------------------------------------------- 10. params chart
cx = json.load(open(f"{OUT}/complexity.json"))
labels = ["Spatial encoder (3-D ResNet)", "Fusion (CAFM)", "Temporal encoder (1-D CNN)", "Classification heads (×3)"]
vals = [cx["spatial_encoder_params"], cx["fusion_params"], cx["temporal_encoder_params"], cx["head_params"]]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
a = axes[0]
cols = [C["sp"][1], C["fu"][1], C["tp"][1], "#777777"]
a.barh(labels[::-1], vals[::-1], color=cols[::-1], height=0.55)
for i, v in enumerate(vals[::-1]):
    a.text(v + 8000, i, f"{v:,}", va="center", fontsize=8)
a.set_xlim(0, 1.15e6)
from matplotlib.ticker import FuncFormatter
a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v else "0"))
a.set_xlabel("Trainable parameters")
a.spines[["top", "right"]].set_visible(False)
a.tick_params(axis="y", labelsize=8)
mm = cx["macs"]
sp_m = sum(v for k, v in mm.items() if k.startswith("Stem") or k.startswith("Residual"))
parts = {"Spatial encoder": sp_m, "Temporal encoder": mm["Temporal encoder (4 × Conv1d+BN)"],
         "Fusion + heads": mm["Cross-attention (W_Q, W_K, W_V, W_O)"] + mm["Feed-forward block (128→256→128)"] + mm["Classification heads ×3"]}
b = axes[1]
b.bar(["Spatial\nencoder", "Temporal\nencoder", "Fusion\n+ heads"], [v / 1e9 for v in parts.values()], color=[C["sp"][1], C["tp"][1], C["fu"][1]], width=0.55)
for i, v in enumerate(parts.values()):
    b.text(i, v / 1e9 + 0.05, f"{v / 1e9:.3f}", ha="center", fontsize=8)
b.set_ylabel("Multiply–accumulates per sample (G)")
b.set_ylim(0, 2.9)
b.tick_params(axis="x", labelsize=7.2)
b.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig_params")

# ----------------------------------------------------------- 11. tests / coverage
tests = {"test_signal_cleaner": 18, "test_mri_transforms": 12, "test_cross_attention": 8, "test_grad_cam_3d": 8, "test_cafn_model": 7, "test_dataset": 7,
         "test_spatial_encoder": 5, "test_plot_attention": 5, "test_temporal_encoder": 4, "test_train_utils": 4, "integration/pipeline_end_to_end": 4, "integration/training_loop": 4}
cov = json.load(open(f"{OUT}/coverage.json"))
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
a = axes[0]
names = list(tests.keys())[::-1]
vals = [tests[n] for n in names]
colr = ["#2E7D5B" if n.startswith("integration") else "#2B5C8A" for n in names]
a.barh(names, vals, color=colr, height=0.6)
for i, v in enumerate(vals):
    a.text(v + 0.3, i, str(v), va="center", fontsize=7.5)
a.set_xlabel("Test functions (total 86)")
a.tick_params(axis="y", labelsize=7)
a.set_xlim(0, 21)
a.spines[["top", "right"]].set_visible(False)
b = axes[1]
mods = sorted(cov.items(), key=lambda kv: kv[1]["hit"] / kv[1]["stmts"])
nm = [k.replace("src/", "") for k, _ in mods]
pc = [100 * v["hit"] / v["stmts"] for _, v in mods]
b.barh(nm, pc, color=["#B5543C" if p < 90 else "#2B5C8A" for p in pc], height=0.6)
for i, p in enumerate(pc):
    b.text(min(p + 1, 102), i, f"{p:.0f}%", va="center", fontsize=7.5)
b.set_xlim(0, 118)
b.set_xlabel("Statement coverage (%)")
b.tick_params(axis="y", labelsize=7)
b.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig_tests_coverage")
print("ok")
