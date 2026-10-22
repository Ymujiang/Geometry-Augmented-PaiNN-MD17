"""
绘制 fig1_rbf_and_gate_mechanism：
  (A) Gaussian+Bessel 混合径向基函数 (RBF) 响应曲线，叠加真实 Uracil 测试集原子对
      距离分布直方图与 CosineCutoff 包络函数；
  (B) 3-body 几何内积门控机制示意图（s1=<mu_i,dir_ij>, s2=<mu_j,dir_ij>, s3=<mu_i,mu_j>
      随键角 theta 的解析扫描）；
  (C) Tanh 双极性门控 vs Sigmoid 单极性门控的函数形状对比（纯数学函数）；
  (D) 真实训练后模型在测试集推理时三层 PaiNNInteraction.geo_gate_net 的
      tanh 激活值经验分布（真实前向钩子捕获，非合成数据）。

数据来源：
  - papers/painn_sci/data/fig1_rbf_gate_data.json
      (scripts/painn_sci_scratch/extract_rbf_gate_mechanism.py 生成，权重无关的
       解析函数 + 受保护 npz 中的真实原子坐标统计)
  - papers/painn_sci/data/raw/gate_activation_final.json
      (scripts/painn_sci_scratch/train_and_extract.py --variant final 生成，
       真实训练 80 epoch 后在测试集推理时通过 register_forward_hook 捕获的
       geo_gate_net 真实输出)

必须使用 .autosota/venv/bin/python3.11 执行，输出 300 DPI PDF+PNG。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/ymujdb/.dsh/skills/figure-style")
from kernel import apply_figure_style
apply_figure_style()

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MECH_PATH = "/home/ymujdb/my_sota_project/papers/painn_sci/data/fig1_rbf_gate_data.json"
GATE_PATH = "/home/ymujdb/my_sota_project/papers/painn_sci/data/raw/gate_activation_final.json"
GATE_SIGMOID_PATH = "/home/ymujdb/my_sota_project/papers/painn_sci/data/raw/gate_activation_sigmoid.json"
FIG_DIR = "/home/ymujdb/my_sota_project/papers/painn_sci/figures"

with open(MECH_PATH) as f:
    mech = json.load(f)
with open(GATE_PATH) as f:
    gate_real = json.load(f)
with open(GATE_SIGMOID_PATH) as f:
    gate_sigmoid = json.load(f)

r = np.array(mech["r"])
gauss_curves = np.array(mech["gaussian_curves"])   # (1000, 20)
bessel_curves = np.array(mech["bessel_curves"])    # (1000, 20)
fcut = np.array(mech["cosine_cutoff_fn"])
all_dists = np.array(mech["all_pair_dists_sampled"])
cutoff = mech["cutoff"]

theta = np.array(mech["theta_scan"]["theta"])
s1 = np.array(mech["theta_scan"]["s1"])
s2 = np.array(mech["theta_scan"]["s2"])
s3 = np.array(mech["theta_scan"]["s3"])

gate_input = np.array(mech["gate_shape"]["gate_input"])
tanh_vals = np.array(mech["gate_shape"]["tanh"])
sigmoid_vals = np.array(mech["gate_shape"]["sigmoid"])

fig = plt.figure(figsize=(12.5, 10))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

# ---------------------------------------------------------------------------
# Panel A: Hybrid RBF response curves
# ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax2 = ax.twinx()
ax2.hist(all_dists, bins=80, range=(0, cutoff), color="#d9d9d9", alpha=0.7, zorder=0,
         label="Real Uracil pairwise\ndistance distribution")
ax2.set_ylabel("Count (sampled test-set\natom pairs)", fontsize=8.5, color="#808080")
ax2.tick_params(axis="y", labelcolor="#808080", labelsize=8)
ax2.set_zorder(0)
ax.set_zorder(1)
ax.patch.set_visible(False)

n_show = 6
gauss_idx = np.linspace(0, gauss_curves.shape[1] - 1, n_show).astype(int)
bessel_idx = np.linspace(0, bessel_curves.shape[1] - 1, n_show).astype(int)
cmap_g = plt.cm.Blues(np.linspace(0.4, 0.95, n_show))
cmap_b = plt.cm.Oranges(np.linspace(0.4, 0.95, n_show))

for k, idx in enumerate(gauss_idx):
    ax.plot(r, gauss_curves[:, idx], color=cmap_g[k], linewidth=1.3, linestyle="-")
for k, idx in enumerate(bessel_idx):
    ax.plot(r, bessel_curves[:, idx], color=cmap_b[k], linewidth=1.1, linestyle="--")

ax.plot(r, fcut, color="black", linewidth=1.6, linestyle=":", label="Cosine cutoff envelope")

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], color=cmap_g[3], lw=1.5, label="Gaussian RBF ($n=20$)"),
    Line2D([0], [0], color=cmap_b[3], lw=1.5, linestyle="--", label="Bessel RBF ($n=20$)"),
    Line2D([0], [0], color="black", lw=1.6, linestyle=":", label="Cosine cutoff $f_c(r)$"),
    Line2D([0], [0], color="#d9d9d9", lw=6, alpha=0.7, label="Real pairwise $r$ (Uracil)"),
]
ax.legend(handles=legend_elems, frameon=False, fontsize=7.8, loc="upper right")

ax.set_xlabel(r"Interatomic distance $r$ (Å)")
ax.set_ylabel("RBF response amplitude")
ax.set_title("(A) Hybrid Gaussian+Bessel RBF, $r \\in [0, r_{cut}]$", fontsize=11, fontweight="bold", loc="left")
ax.set_xlim(0, cutoff)
ax.spines["top"].set_visible(False)

# ---------------------------------------------------------------------------
# Panel B: 3-body geometric inner-product gating schematic
# ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.plot(np.degrees(theta), s1, color="#1f77b4", linewidth=1.8, label=r"$s_1=\langle\mu_i,\hat{r}_{ij}\rangle$")
ax.plot(np.degrees(theta), s2, color="#2ca02c", linewidth=1.8, linestyle="--", label=r"$s_2=\langle\mu_j,\hat{r}_{ij}\rangle$")
ax.plot(np.degrees(theta), s3, color="#d62728", linewidth=1.8, linestyle="-.", label=r"$s_3=\langle\mu_i,\mu_j\rangle$")
ax.axhline(0, color="gray", linewidth=0.6)
ax.set_xlabel(r"Bond angle $\theta$ between $\mu_i$ and bond direction (deg)")
ax.set_ylabel("Geometric inner-product value")
ax.set_title("(B) 3-Body Geometric Inner-Product Features", fontsize=11, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 180)
ax.text(0.98, 0.95, r"$\mathbf{geo\_ctx} = [s_1; s_2; s_3]$" + "\n" + r"$\mathrm{gate} = \tanh(W\cdot\mathbf{geo\_ctx}+b)$",
        transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="#fff7e6", edgecolor="#d9a441", alpha=0.9))

# ---------------------------------------------------------------------------
# Panel C: Tanh bipolar gate vs Sigmoid unipolar gate
# ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.plot(gate_input, tanh_vals, color="#d62728", linewidth=2.0, label=r"$\tanh(x)$  (bipolar, final model)")
ax.plot(gate_input, sigmoid_vals, color="#1f77b4", linewidth=2.0, linestyle="--", label=r"$\sigma(x)$  (unipolar, iter-2/4)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axhline(1, color="gray", linewidth=0.4, linestyle=":")
ax.axhline(-1, color="gray", linewidth=0.4, linestyle=":")
ax.set_xlabel(r"Gate pre-activation (logit) $x = W\cdot\mathbf{geo\_ctx}+b$")
ax.set_ylabel("Gate activation value")
ax.set_title("(C) Bipolar (Tanh) vs Unipolar (Sigmoid) Gating", fontsize=11, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(-1.15, 1.15)
ax.text(0.03, 0.95,
        "Residual multiplier: 1 + activation\nTanh: attenuation or amplification\nSigmoid: amplification only\nNeither reverses the message sign.",
        transform=ax.transAxes, fontsize=7.8, va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))

# ---------------------------------------------------------------------------
# Panel D: Real empirical gate-activation distributions (from trained model)
# ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
layer_colors = ["#9ecae1", "#fdae6b", "#a1d99b"]
layer_names = ["Interaction 1", "Interaction 2", "Interaction 3"]
for i, lname in enumerate(["layer_0", "layer_1", "layer_2"]):
    acts = np.array(gate_real["layers"][lname]["activated_sample"])
    ax.hist(acts, bins=60, range=(-1, 1), histtype="stepfilled", alpha=0.55,
            color=layer_colors[i], edgecolor="black", linewidth=0.3,
            label=f"{layer_names[i]} ($\\mu$={acts.mean():.3f}, $\\sigma$={acts.std():.3f})")
# Real sigmoid-gated control model (iter-4 architecture), overlaid as outlines.
sig_acts = np.concatenate([
    np.array(gate_sigmoid["layers"][l]["activated_sample"])
    for l in ["layer_0", "layer_1", "layer_2"]
])
ax.hist(sig_acts, bins=60, range=(-1, 1), histtype="step", alpha=0.95,
        color="#d62728", linewidth=1.4,
        label=f"$\\sigma$ control, all layers ($\\mu$={sig_acts.mean():.3f})")
ax.set_xlabel(r"Real gate activation value (test-set inference)")
ax.set_ylabel("Count (all atom pairs × test frames)")
ax.set_title("(D) Empirical Gate-Activation Distribution (Trained Models)", fontsize=11, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=7.2, loc="upper center")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(-1, 1)
ax.text(0.02, 0.40,
        "Negative tanh activation attenuates\nthe residual message; the multiplier\n1 + tanh remains positive.",
        transform=ax.transAxes, fontsize=7.0, va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))

plt.savefig(f"{FIG_DIR}/fig1_rbf_and_gate_mechanism.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig1_rbf_and_gate_mechanism.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig1_rbf_and_gate_mechanism.{pdf,png}")
