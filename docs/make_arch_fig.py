import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.4, 3.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
def box(x, y, w, h, t, fc="#eef2ff", ec="#1d4ed8", fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)
def arrow(a, b, c="#374151", st="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=st, mutation_scale=8, lw=1, color=c, ls=ls))
box(1, 38, 22, 8, "3D T1w MRI\n(1, 64³)", "#f3f4f6", "#6b7280")
box(77, 38, 22, 8, "1D signal: VEP / spectrum\n(1, 500)", "#f3f4f6", "#6b7280")
box(1, 26, 22, 8, "3D ResNet + SE blocks\npool 4×4×4")
box(77, 26, 22, 8, "Multi-scale 1D CNN\nkernels 3 / 9 / 19")
box(1, 14.5, 22, 8, "Spatial tokens (64×d)\n+ learned 3D pos-emb")
box(77, 14.5, 22, 8, "Temporal tokens (16×d)\n+ sinusoidal pos-enc")
box(30, 12, 40, 24, "", "#dbeafe", "#1d4ed8")
ax.text(50, 33.3, "Gated bidirectional cross-attention", ha="center", fontsize=7.5, fontweight="bold")
ax.text(50, 29.6, "t′ = t + σ(g_t)·CA(Q=t, K,V=s)", ha="center", fontsize=7)
ax.text(50, 26.3, "s′ = s + σ(g_s)·CA(Q=s, K,V=t′)", ha="center", fontsize=7)
ax.text(50, 22.6, "pre-norm · FFN · N blocks · 4 heads", ha="center", fontsize=6.5, color="#374151")
ax.text(50, 19.4, "gates g init 0  →  σ(0)=0.5", ha="center", fontsize=6.5, color="#374151")
ax.text(50, 15.6, "returns both attention maps", ha="center", fontsize=6.5, color="#374151")
box(30, 1, 40, 7.5, "Learned attention pool → fused head\n(Healthy / Glaucoma / ON–CVI  or  control / patient)", "#ecfdf5", "#059669", 7)
box(1, 1, 22, 7.5, "Auxiliary MRI head", "#fff7ed", "#d97706", 6.8)
box(77, 1, 22, 7.5, "Auxiliary signal head", "#fff7ed", "#d97706", 6.8)
for x in (12, 88):
    arrow((x, 38), (x, 34.2)); arrow((x, 26), (x, 22.7))
arrow((23.3, 18.5), (30, 22), "#1d4ed8"); arrow((76.7, 18.5), (70, 22), "#1d4ed8")
arrow((50, 12), (50, 8.7), "#059669")
arrow((12, 14.5), (12, 8.7), "#d97706", ls="--"); arrow((88, 14.5), (88, 8.7), "#d97706", ls="--")
ax.text(50, 44.5, "L = L_fused + λ (L_MRI + L_signal),  λ = 0.3", ha="center", fontsize=7.5)
fig.tight_layout(pad=0.2); fig.savefig("journal_ieee_access/figs/fig_arch_v2.png", dpi=240)
print("ok")
