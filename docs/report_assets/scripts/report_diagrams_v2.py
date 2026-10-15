"""Final-model (CAFN v2/v3) versions of five report figures (matplotlib only).

Copy of the helper style of report_diagrams.py (colour dict C, Liberation Serif,
canvas/box/arrow) but every canvas is laid out in POINTS at the Word insertion
width (14.5 cm = 411 pt), so a font size of 8 here is 8 pt in the document.
Each figure runs an automatic text-fit / text-overlap check (see check()).

Usage: python report_diagrams_v2.py <out_dir> [fig names...]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures_v2"
ONLY = set(sys.argv[2:])
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8.5, "savefig.dpi": 220,
                     "axes.linewidth": 0.8, "mathtext.fontset": "stix"})

C = dict(
    sp=("#DCE9F5", "#2B5C8A"),   # spatial / MRI
    tp=("#FBE9CC", "#A8691A"),   # temporal / signal
    fu=("#DDF0E6", "#2E7D5B"),   # fusion
    hd=("#EDEDED", "#555555"),   # heads / neutral
    ex=("#ECE0F3", "#6B4C85"),   # explainability
    io=("#FFFFFF", "#333333"),   # io
)
W_PT = 411.0  # 14.5 cm
_REG = []     # (name, rect, text artist) for the fit check
_FREE = []    # free text artists


def canvas(aspect, h=None):
    """Full-bleed canvas, 1 data unit = 1 pt at 14.5 cm width. aspect = width/height of the original."""
    H = h if h else W_PT / aspect
    fig = plt.figure(figsize=(W_PT / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT)
    ax.set_ylim(0, H)
    ax.axis("off")
    _REG.clear(); _FREE.clear()
    return fig, ax, H


def box(ax, x, y, w, h, text, kind="io", fs=8, bold=False, ls="-", lw=1.0, ec=None, fc=None, ha="center"):
    f, e = C[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", fc=fc or f, ec=ec or e, lw=lw, ls=ls)
    ax.add_patch(p)
    tx = x + w / 2 if ha == "center" else x + 5
    t = ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.15)
    _REG.append((text.replace("\n", " ")[:30], (x, y, w, h), t))
    return (x, y, w, h)


def label(ax, x, y, s, fs=8, ha="center", va="center", **kw):
    t = ax.text(x, y, s, fontsize=fs, ha=ha, va=va, **kw)
    _FREE.append(t)
    return t


def arrow(ax, pts, color="#333333", ls="-", lw=1.0):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                shrinkA=0, shrinkB=0, mutation_scale=8))


def check(fig, ax, name, pad=2.0):
    """Report text overflowing its box, text-text overlaps and text outside the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def ext(t):
        bb = t.get_window_extent(r)
        (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        return x0, y0, x1, y1
    bad = 0
    W, H = ax.get_xlim()[1], ax.get_ylim()[1]
    allt = []
    for nm, (x, y, w, h), t in _REG:
        x0, y0, x1, y1 = ext(t)
        if x0 < x + pad or x1 > x + w - pad or y0 < y + 1 or y1 > y + h - 1:
            print(f"  [{name}] OVERFLOW box '{nm}': text {x0:.0f}-{x1:.0f} x {y0:.0f}-{y1:.0f} vs box {x:.0f}-{x+w:.0f} x {y:.0f}-{y+h:.0f}")
            bad += 1
        allt.append((nm, ext(t)))
    for t in _FREE:
        allt.append((t.get_text()[:30].replace("\n", " "), ext(t)))
    for nm, (x0, y0, x1, y1) in allt:
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            print(f"  [{name}] OUTSIDE canvas '{nm}': {x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}")
            bad += 1
    for i in range(len(allt)):
        for j in range(i + 1, len(allt)):
            a, b = allt[i][1], allt[j][1]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                print(f"  [{name}] TEXT OVERLAP '{allt[i][0]}' x '{allt[j][0]}'")
                bad += 1
    print(f"  [{name}] check: {'OK' if not bad else str(bad) + ' problem(s)'}")


def save(fig, ax, name):
    check(fig, ax, name)
    fig.savefig(f"{OUT}/{name}.png", dpi=220)
    plt.close(fig)
    print("saved", name)


def want(n):
    return not ONLY or n in ONLY


# ======================================================= 1. pipeline (Fig 3.1)
def fig_pipeline():
    fig, ax, H = canvas(1135 / 716)          # H ~ 259
    # --- MRI row (y centre 205) and signal row (y centre 61)
    box(ax, 3, 180, 60, 50, "3D structural\nMRI (NIfTI)", "sp")
    box(ax, 3, 36, 60, 50, "1-D signal\n(VEP or\nspectrum)", "tp")
    box(ax, 71, 175, 78, 60, "MRI preprocessing\nRAS · 1 mm · mask\nscale · crop/pad\n→ (1, 64, 64, 64)", "sp")
    box(ax, 71, 31, 78, 60, "Signal conditioning\nbandpass · notch\nresample · z-score\n→ (1, 500)", "tp")
    box(ax, 157, 180, 58, 50, "3D ResNet\n+ SE blocks\nencoder", "sp")
    box(ax, 157, 36, 58, 50, "Multi-scale\n1-D CNN\nencoder", "tp")
    arrow(ax, [(63, 205), (71, 205)]); arrow(ax, [(63, 61), (71, 61)])
    arrow(ax, [(149, 205), (157, 205)]); arrow(ax, [(149, 61), (157, 61)])
    # --- fusion
    box(ax, 232, 98, 96, 62, "Gated bidirectional\ncross-attention\nt ← s  (Q = t)\ns ← t′ (Q = s)", "fu", bold=False)
    arrow(ax, [(215, 205), (223, 205), (223, 146), (232, 146)])
    arrow(ax, [(215, 61), (223, 61), (223, 112), (232, 112)])
    # --- explainability (dashed)
    box(ax, 240, 192, 80, 34, "Grad-CAM\n(MRI)", "ex")
    box(ax, 240, 34, 80, 34, "Cross-attention\nheatmaps", "ex")
    arrow(ax, [(280, 160), (280, 192)], ls="--", color="#6B4C85")
    arrow(ax, [(280, 98), (280, 68)], ls="--", color="#6B4C85")
    # --- right column
    box(ax, 336, 208, 72, 32, "MRI head\n(auxiliary)", "hd")
    box(ax, 336, 140, 72, 32, "Attention\npool", "fu")
    box(ax, 336, 98, 72, 32, "Fused head\n(primary)", "hd", bold=True)
    box(ax, 336, 58, 72, 28, "Prediction\n(B, 3) logits", "io")
    box(ax, 336, 22, 72, 28, "Signal head\n(auxiliary)", "hd")
    arrow(ax, [(328, 150), (336, 150)])
    arrow(ax, [(372, 140), (372, 130)])
    arrow(ax, [(372, 98), (372, 86)])
    # --- auxiliary token paths (unimodal tokens, mean-pooled)
    arrow(ax, [(186, 230), (186, 247), (372, 247), (372, 240)])
    label(ax, 280, 253, "spatial tokens + 3-D pos-emb, (B, 64, 128)", fs=8, style="italic")
    arrow(ax, [(186, 36), (186, 14), (372, 14), (372, 22)])
    label(ax, 280, 7, "temporal tokens + sinusoidal pos-enc, (B, 16, 128)", fs=8, style="italic")
    # --- dashed legend
    save(fig, ax, "fig_pipeline_v2")


if want("pipeline"):
    fig_pipeline()


# ======================================================= 2. architecture (Fig 4.2)
def fig_architecture():
    fig, ax, H = canvas(998 / 1258)          # H ~ 518
    XL, XR, CW = 12, 211, 188                # column x, width
    label(ax, XL + CW / 2, 508, "Spatial stream (MRI)", fs=9.5, fontweight="bold", color=C["sp"][1])
    label(ax, XR + CW / 2, 508, "Temporal stream (signal)", fs=9.5, fontweight="bold", color=C["tp"][1])
    sp = ["Input MRI\n(B, 1, 64, 64, 64)",
          "Stem: Conv3d 1→16 + BN + ReLU\n(B, 16, 64, 64, 64)",
          "SE-ResBlock 16→32, stride 2\n(B, 32, 32, 32, 32)",
          "SE-ResBlock 32→64, stride 2\n(B, 64, 16, 16, 16)",
          "SE-ResBlock 64→128, stride 2\n[Grad-CAM layer]  (B, 128, 8, 8, 8)",
          "Adaptive avg-pool 4×4×4 + flatten\n(B, 64, 128)",
          "+ learned 3-D positional embedding\nspatial tokens (B, 64, 128)"]
    tp = ["Input signal\n(B, 1, 500)",
          "Stem: Conv1d 1→16, k=7 + BN + GELU\n(B, 16, 500)",
          "Conv1d 16→32 (k=5, s=2) + BN + GELU\n+ multi-scale residual (k = 3 / 9 / 19)\n(B, 32, 250)",
          "Conv1d 32→64 (k=5, s=2) + BN + GELU\n+ multi-scale residual (k = 3 / 9 / 19)\n(B, 64, 125)",
          "Conv1d 64→128 (k=5, s=2) + BN + GELU\n+ multi-scale residual (k = 3 / 9 / 19)\n(B, 128, 63)",
          "Adaptive avg-pool to 16 + transpose\n(B, 16, 128)",
          "+ sinusoidal positional encoding (fixed)\ntemporal tokens (B, 16, 128)"]
    RH, GAP, TOP = 32, 6, 498
    ys = [TOP - (RH + GAP) * i - RH for i in range(7)]
    for t, y in zip(sp, ys):
        box(ax, XL, y, CW, RH, t, "sp", fs=8)
    for t, y in zip(tp, ys):
        box(ax, XR, y, CW, RH, t, "tp", fs=8)
    for a, b in zip(ys[:-1], ys[1:]):
        for xc in (XL + CW / 2, XR + CW / 2):
            arrow(ax, [(xc, a), (xc, b + RH)])
    y7 = ys[-1]                               # bottom of last row
    # --- fusion block
    FY, FH = 156, 56
    box(ax, 40, FY, 331, FH,
        "Gated bidirectional cross-attention  (N stacked blocks, default N = 1)\n"
        "t′ = t + σ(g_t)·CA(Q = LN(t); K, V = s) → FFN     weights (B, 4, 16, 64)\n"
        "s′ = s + σ(g_s)·CA(Q = LN(s); K, V = t′) → FFN     weights (B, 4, 64, 16)\n"
        "4 heads · pre-norm · FFN 128→256→128 · gates g_t, g_s initialised at 0\n"
        "the fused head reads s′ only; t′ (B, 16, 128) is returned for inspection", "fu", fs=8)
    arrow(ax, [(XL + CW / 2, y7), (XL + CW / 2, FY + FH)])
    arrow(ax, [(XR + CW / 2, y7), (XR + CW / 2, FY + FH)])
    label(ax, XL + CW / 2 + 5, (y7 + FY + FH) / 2, "s", fs=9, ha="left", style="italic", fontweight="bold")
    label(ax, XR + CW / 2 + 5, (y7 + FY + FH) / 2, "t", fs=9, ha="left", style="italic", fontweight="bold")
    # --- pool + heads
    box(ax, 95, 108, 221, 28, "final LayerNorm → learned attention pool\nα_i = softmax_i(w$^{\mathsf{T}}$f_i + b),  f = Σ α_i f_i", "fu", fs=8)
    arrow(ax, [(205, FY), (205, 136)])
    label(ax, 211, 146, "fused spatial tokens s′  (B, 64, 128)", fs=8, ha="left", style="italic")
    HY, HH = 40, 34
    box(ax, XL, HY, 124, HH, "MRI head (auxiliary)\nmean-pool → Linear 128→3\nlogits (B, 3)", "hd", fs=8)
    box(ax, 150, HY, 111, HH, "Fused head (primary)\nLinear 128→3\nlogits (B, 3)", "hd", fs=8, bold=False)
    box(ax, 275, HY, 124, HH, "Signal head (auxiliary)\nmean-pool → Linear 128→3\nlogits (B, 3)", "hd", fs=8)
    arrow(ax, [(205, 108), (205, HY + HH)])
    label(ax, 211, 91, "pooled embedding (B, 128)", fs=8, ha="left", style="italic")
    # auxiliary heads read the pre-fusion tokens (dashed, outer margin)
    ym = y7 + RH / 2
    arrow(ax, [(XL, ym), (5, ym), (5, HY + HH / 2), (XL, HY + HH / 2)], ls="--", color=C["sp"][1])
    arrow(ax, [(XR + CW, ym), (406, ym), (406, HY + HH / 2), (399, HY + HH / 2)], ls="--", color=C["tp"][1])
    label(ax, 205, 18, "L = L_fused + 0.3 (L_MRI + L_signal)     (3 classes in the synthetic benchmark)", fs=8, style="italic")
    save(fig, ax, "fig_architecture_v2")


if want("architecture"):
    fig_architecture()


# ======================================================= 3. one gated bidirectional block (Fig 4.4)
def _plus(ax, x, y, r=7):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fc="white", ec="#333333", lw=1.0))
    ax.plot([x - r * 0.6, x + r * 0.6], [y, y], color="#333333", lw=1.0)
    ax.plot([x, x], [y - r * 0.6, y + r * 0.6], color="#333333", lw=1.0)


def _lane(ax, yc, kind, tok, nm, other_nm, w_shape, gate, in_lab, out_lab, in_shape, up):
    """One update lane. up=True: residual dashed line above, False: below."""
    bh = 50
    y0 = yc - bh / 2
    box(ax, 3, y0, 54, bh, f"{in_lab}\n{in_shape}", kind, fs=8)
    box(ax, 63, yc - 12, 26, 24, "LN", "hd", fs=8)
    box(ax, 95, y0, 100, bh,
        f"Multi-head\ncross-attention\nQ = LN({nm}); K, V = {other_nm}\nweights {w_shape}", "fu", fs=8)
    box(ax, 202, yc - 17, 44, 34, f"× σ(g_{gate})\ng init 0", "io", fs=8, ec="#2E7D5B", lw=1.4)
    _plus(ax, 260, yc)
    box(ax, 274, yc - 21, 66, 42, "LN → FFN\n128→256→128\n+ residual", "hd", fs=8)
    box(ax, 348, y0, 60, bh, f"{out_lab}\n{out_lab[:1]}′ {in_shape}" if False else f"{out_lab}\n{in_shape}", kind, fs=8)
    arrow(ax, [(57, yc), (63, yc)]); arrow(ax, [(89, yc), (95, yc)])
    arrow(ax, [(195, yc), (202, yc)]); arrow(ax, [(246, yc), (253, yc)])
    arrow(ax, [(267, yc), (274, yc)]); arrow(ax, [(340, yc), (348, yc)])
    ytop = yc + bh / 2 if up else yc - bh / 2
    yline = ytop + 18 if up else ytop - 12
    arrow(ax, [(30, ytop), (30, yline), (260, yline), (260, yc + (7 if up else -7))], ls="--", color="#666666")
    label(ax, 145, yline - 7 if up else yline + 6, "residual connection", fs=8, style="italic", color="#444444")


def fig_cross_attention():
    fig, ax, H = canvas(1135 / 552, h=204)
    TOPC, BOTC = 153, 51
    _lane(ax, TOPC, "tp", "t", "t", "s", "(B, 4, 16, 64)", "t", "temporal\ntokens t", "t′ (updated)", "(B, 16, 128)", True)
    _lane(ax, BOTC, "sp", "s", "s", "t′", "(B, 4, 64, 16)", "s", "spatial\ntokens s", "s′ (fused)", "(B, 64, 128)", False)
    # cross links: spatial tokens -> K,V of the temporal update; updated t' -> K,V of the spatial update
    arrow(ax, [(30, BOTC + 25), (30, 111), (145, 111), (145, TOPC - 25)])
    label(ax, 88, 117, "s → K, V   (B, 64, 128)", fs=8, style="italic")
    arrow(ax, [(378, TOPC - 25), (378, 95), (145, 95), (145, BOTC + 25)])
    label(ax, 262, 101, "updated t′ → K, V   (B, 16, 128)", fs=8, style="italic")
    label(ax, 205, 5, "Both attention maps are returned: forward (B, 4, 64, 16) and backward (B, 4, 16, 64)", fs=8, style="italic")
    save(fig, ax, "fig_cross_attention_v2")


if want("cross_attention"):
    fig_cross_attention()


# ======================================================= 4. parameters / MACs (Fig 5.5)
def fig_params():
    cx = json.load(open(os.path.join(HERE, "..", "..", "results_raw", "model_complexity_v2.json")))
    H = W_PT / (1395 / 537)                          # ~158 pt
    fig = plt.figure(figsize=(W_PT / 72, H / 72))
    # parameters (left)
    short = {"Spatial encoder (3D ResNet + SE)": "Spatial encoder",
             "Temporal encoder (multi-scale 1D CNN)": "Temporal encoder",
             "Gated bidirectional fusion": "Gated fusion",
             "Positional embeddings": "Positional emb.",
             "Attention pool": "Attention pool",
             "Heads (3)": "Heads (×3)"}
    colour = {"Spatial encoder": C["sp"][1], "Temporal encoder": C["tp"][1], "Gated fusion": C["fu"][1]}
    names = [short[k] for k in cx["params"]]
    vals = list(cx["params"].values())
    assert sum(vals) == cx["total"]
    a = fig.add_axes([0.19, 0.20, 0.35, 0.775])
    ys = list(range(len(vals)))[::-1]
    a.barh(ys, vals, color=[colour.get(n, "#777777") for n in names], height=0.62)
    for y, v in zip(ys, vals):
        a.text(v + 22000, y, f"{v:,}", va="center", ha="left", fontsize=8)
    a.set_yticks(ys)
    a.set_yticklabels(names, fontsize=8)
    a.set_xlim(0, 1.25e6)
    a.set_ylim(-0.6, len(vals) - 0.4)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v / 1000)}k" if v else "0"))
    a.set_xticks([0, 250000, 500000, 750000, 1000000])
    a.tick_params(axis="x", labelsize=8, length=3, pad=2)
    a.tick_params(axis="y", length=3, pad=2)
    a.set_xlabel("Trainable parameters", fontsize=8, labelpad=2)
    a.spines[["top", "right"]].set_visible(False)
    a.text(0.99, 0.30, f"Total: {cx['total']:,}", transform=a.transAxes, ha="right", va="center", fontsize=8, fontweight="bold")
    # MACs (right)
    m = cx["component_macs_64_M"]
    mshort = {"Spatial encoder (3D ResNet + SE)": "Spatial\nencoder", "Temporal encoder (multi-scale 1D CNN)": "Temporal\nencoder",
              "Gated bidirectional fusion": "Gated\nfusion"}
    mv = list(m.values())
    b = fig.add_axes([0.685, 0.20, 0.305, 0.775])
    mcol = [C["sp"][1], C["tp"][1], C["fu"][1]]
    b.bar(range(3), mv, color=mcol, width=0.62)
    for i, v in enumerate(mv):
        b.text(i, v + 45, f"{v:,.2f}", ha="center", va="bottom", fontsize=8)
    b.set_xticks(range(3))
    b.set_xticklabels([mshort[k] for k in m], fontsize=8)
    b.set_xlim(-0.55, 2.55)
    b.set_ylim(0, 3100)
    b.set_yticks([0, 1000, 2000, 3000])
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    b.tick_params(axis="y", labelsize=8, length=3, pad=2)
    b.tick_params(axis="x", length=3, pad=2)
    b.set_ylabel("MACs per sample (millions)", fontsize=8, labelpad=2)
    b.spines[["top", "right"]].set_visible(False)
    b.text(0.98, 0.80, f"Total: {cx['macs_64_M']:,.2f} M\n(2.60 G, 64³ input)", transform=b.transAxes, ha="right", va="center", fontsize=8, fontweight="bold", linespacing=1.15)
    fig.savefig(f"{OUT}/fig_params_v2.png", dpi=220)
    plt.close(fig)
    print("saved fig_params_v2")


if want("params"):
    fig_params()


# ======================================================= 5. software structure (Fig 4.7)
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
# modules present in the first model commit (6567f3d, "v1"); everything else is new
V1 = {"config", "preprocessing.mri_transforms", "preprocessing.signal_cleaner", "data.synthetic", "data.dataset",
      "models.spatial_encoder", "models.temporal_encoder", "models.cross_attention", "models.cafn",
      "explainability.grad_cam_3d", "explainability.plot_attention", "train"}


def import_graph():
    """Intra-package import graph of src/**/*.py via ast (module -> sorted list of src modules it imports)."""
    import ast
    import glob
    mods = {}
    for p in sorted(glob.glob(os.path.join(SRC, "**", "*.py"), recursive=True)):
        rel = os.path.relpath(p, os.path.dirname(SRC))[:-3].replace(os.sep, ".")
        if rel.endswith(".__init__"):
            rel = rel[:-9]
        mods[rel] = p

    def resolve(name):
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            if ".".join(parts[:i]) in mods:
                return ".".join(parts[:i])
        return None
    graph = {}
    for m, p in mods.items():
        if os.path.getsize(p) == 0:        # empty package __init__
            continue
        deps = set()
        for n in ast.walk(ast.parse(open(p, encoding="utf8").read())):
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    base = m.split(".")
                    if not p.endswith("__init__.py"):
                        base = base[:-1]
                    base = base[:len(base) - (n.level - 1)]
                    full = ".".join(base + ([n.module] if n.module else []))
                else:
                    full = n.module or ""
                names = [full] + [full + "." + a.name for a in n.names]
            else:
                continue
            for nm in names:
                if nm.startswith("src"):
                    r = resolve(nm)
                    if r and r != m and r != "src":
                        deps.add(r)
        # drop a package/parent hit when a more specific submodule was found
        graph[m[4:] if m.startswith("src.") else m] = sorted(d[4:] for d in deps if d != "src" and d.count(".") >= 1 or d == "src.config")
    return graph


def _mix(hexc, t=0.35):
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(255 - (255 - v) * t) for v in (r, g, b))


def _group(ax, x, ytop, w, h, title, kind, chips, ncol, cw, ch=14, gap=3):
    """Container with a title and a grid of module chips (column-major). chips = [(name, is_new)]."""
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec=ec, lw=1.0))
    label(ax, x + 5, ytop - 7, title, fs=8, ha="left", fontweight="bold", color=ec)
    nrow = -(-len(chips) // ncol)
    for i, (nm, new) in enumerate(chips):
        c, r = divmod(i, nrow)
        cx = x + 5 + c * (cw + 4)
        cy = ytop - 14 - (r + 1) * ch - r * gap
        box(ax, cx, cy, cw, ch, nm, kind, fs=8, fc=_mix(ec, 0.30) if new else "white",
            ls=(0, (3, 1.5)) if new else "-", lw=1.5 if new else 0.8)
    return (x, ytop - h, w, h)


def fig_software_structure():
    g = import_graph()
    allm = sorted(g)
    new = {m for m in allm if m not in V1}
    # persist the graph used for the figure
    out_json = os.path.join(HERE, "..", "data", "import_graph_v2.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump({"modules": allm, "new_since_v1": sorted(new), "imports": g}, open(out_json, "w"), indent=1)
    print(f"  {len(allm)} modules, {len(new)} new since v1")
    fig, ax, H = canvas(1135 / 694, h=251)
    nw = lambda n: n in new
    # A: config
    box(ax, 3, 230, 405, 18, "config.py: shapes, hyper-parameters, class names (imported by every layer)", "hd", fs=8)
    # B row
    BT = 209
    _group(ax, 3, BT, 75, 82, "preprocessing/", "sp", [("mri_transforms", nw("preprocessing.mri_transforms")), ("signal_cleaner", nw("preprocessing.signal_cleaner"))], 1, 65)
    _group(ax, 90, BT, 106, 82, "data/", "tp", [(n, nw("data." + n)) for n in ("synthetic", "dataset", "augment", "cohorts", "splits")], 2, 46)
    _group(ax, 214, BT, 194, 82, "models/", "fu", [(n, nw("models." + n)) for n in
           ("spatial_encoder", "temporal_encoder", "positional", "pooling", "cross_attention", "attention_pool", "losses", "cafn")], 2, 88)
    # C row
    CT = 113
    _group(ax, 3, CT, 107, 65, "explainability/", "ex", [(n, nw("explainability." + n)) for n in ("grad_cam_3d", "plot_attention", "p100")], 1, 97)
    _group(ax, 122, CT, 134, 48, "training / inference", "hd", [("train", nw("train")), ("evaluation", nw("evaluation")), ("predict", nw("predict")), ("utils", nw("utils"))], 2, 60)
    _group(ax, 268, CT, 140, 65, "experiments", "io", [(n, nw(n)) for n in ("ablation_synthetic", "cohort_experiments", "cv_real")], 1, 130)
    # tests
    box(ax, 3, 24, 405, 16, "tests/unit · tests/integration   (pytest suite covering the modules above)", "io", fs=8)
    # arrows (provider -> importer)
    for x in (40, 143, 311):
        arrow(ax, [(x, 230), (x, BT)])
    arrow(ax, [(78, 165), (90, 165)])                              # preprocessing -> data
    arrow(ax, [(143, 127), (143, CT)])                             # data -> training
    arrow(ax, [(235, 127), (235, CT)])                             # models -> training
    arrow(ax, [(40, 127), (40, 120), (130, 120), (130, CT)])       # preprocessing -> predict
    arrow(ax, [(330, 127), (330, CT)])                             # models -> experiments
    arrow(ax, [(185, 127), (185, 120), (290, 120), (290, CT)])     # data -> experiments (crosses models -> training)
    arrow(ax, [(110, 89), (122, 89)])                              # explainability -> predict
    arrow(ax, [(256, 89), (268, 89)])                              # training -> experiments
    # legend
    box(ax, 3, 3, 40, 14, "module", "io", fs=8, fc="white", lw=0.8)
    label(ax, 48, 10, "in the v1 code base", fs=8, ha="left")
    box(ax, 150, 3, 40, 14, "module", "io", fs=8, fc=_mix("#333333", 0.25), ls=(0, (3, 1.5)), lw=1.5)
    label(ax, 195, 10, "added since v1", fs=8, ha="left")
    arrow(ax, [(283, 10), (303, 10)])
    label(ax, 308, 10, "provides code to", fs=8, ha="left")
    save(fig, ax, "fig_software_structure_v2")


if want("software_structure"):
    fig_software_structure()
