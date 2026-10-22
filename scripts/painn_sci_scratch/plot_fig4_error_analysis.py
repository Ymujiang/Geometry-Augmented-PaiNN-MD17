"""
绘制 fig4_uracil_error_distribution：Uracil 测试集逐原子类型/化学微环境误差分解 +
Parity Plot + 误差直方图。

数据来源: papers/painn_sci/data/raw/predictions_final.npz
  (真实训练 80 epoch 后在 best_state 上对 1000 个测试构型做的一次性前向推理，
   未经任何修改的 optimized_code/painn/eval_benchmark.py 协议复现)

Uracil 原子序号与类型 (z = [6,6,7,6,7,6,8,8,1,1,1,1]):
  idx 0,1,3,5 -> C (ring/vinylic carbon)
  idx 2,4     -> N (ring amino nitrogen)
  idx 6,7     -> O (carbonyl oxygen, C=O)
  idx 8,9,10,11 -> H

化学微环境细分:
  - carbonyl C (C=O 碳):      idx {3, 5}
  - vinylic C (C5=C6 碳):     idx {0, 1}
  - ring N (N-H 氨基氮):      idx {2, 4}
  - carbonyl O (C=O 氧):      idx {6, 7}
  - amino H (N-H 氢):         idx {9, 10}
  - vinylic H (C-H 氢):       idx {8, 11}

必须使用 .autosota/venv/bin/python3.11 执行,输出 300 DPI PDF+PNG。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATA_PATH = "/home/ymujdb/my_sota_project/papers/painn_sci/data/raw/predictions_final.npz"
FIG_DIR = "/home/ymujdb/my_sota_project/papers/painn_sci/figures"

d = np.load(DATA_PATH)
pred_f = d["pred_forces"]   # (1000, 12, 3)
true_f = d["true_forces"]
pred_e = d["pred_energy"]   # (1000,)
true_e = d["true_energy"]
z = d["z"]                  # (12,)

print("z =", z.tolist())
print("test_f_mae (global) =", float(d["test_f_mae"]))
print("test_e_mae (global) =", float(d["test_e_mae"]))

# per-atom force MAE (across all 3 components and all test frames): shape (12,)
abs_err = np.abs(pred_f - true_f)  # (1000, 12, 3)
per_atom_mae = abs_err.mean(axis=(0, 2))  # (12,)

atom_type_groups = {"C": [0, 1, 3, 5], "N": [2, 4], "O": [6, 7], "H": [8, 9, 10, 11]}
microenv_groups = {
    "Carbonyl C\n(C=O)": [3, 5],
    "Vinylic C\n(C=C)": [0, 1],
    "Ring N\n(N-H)": [2, 4],
    "Carbonyl O\n(C=O)": [6, 7],
    "Amino H\n(N-H)": [9, 10],
    "Vinylic H\n(C-H)": [8, 11],
}

atom_type_mae = {k: float(per_atom_mae[v].mean()) for k, v in atom_type_groups.items()}
microenv_mae = {k: float(per_atom_mae[v].mean()) for k, v in microenv_groups.items()}

print("\nper-atom-type MAE:", atom_type_mae)
print("\nmicroenvironment MAE:", microenv_mae)

with open("/home/ymujdb/my_sota_project/papers/painn_sci/data/fig4_error_summary.json", "w") as f:
    json.dump({
        "per_atom_mae": per_atom_mae.tolist(),
        "atom_type_mae": atom_type_mae,
        "microenv_mae": microenv_mae,
        "global_test_f_mae": float(d["test_f_mae"]),
        "global_test_e_mae": float(d["test_e_mae"]),
    }, f, indent=2)

# ---------------------------------------------------------------------------
# 构建 2x2 figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))

# --- Panel A: 逐元素类型误差条形图 ---
ax = axes[0, 0]
types = list(atom_type_groups.keys())
mae_vals = [atom_type_mae[t] for t in types]
colors_type = {"C": "#404040", "N": "#1f77b4", "O": "#d62728", "H": "#aaaaaa"}
bar_colors = [colors_type[t] for t in types]
bars = ax.bar(types, mae_vals, color=bar_colors, edgecolor="black", linewidth=0.7, width=0.55)
for b, v in zip(bars, mae_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel(r"Force MAE (kcal mol$^{-1}$ Å$^{-1}$)")
ax.set_title("(A) Per-Element-Type Force Error", fontsize=11, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel B: 化学微环境误差条形图 ---
ax = axes[0, 1]
menv_labels = list(microenv_groups.keys())
menv_vals = [microenv_mae[k] for k in menv_labels]
menv_colors = ["#404040", "#808080", "#1f77b4", "#d62728", "#f4a582", "#cccccc"]
bars = ax.bar(range(len(menv_labels)), menv_vals, color=menv_colors, edgecolor="black", linewidth=0.7, width=0.6)
for i, (b, v) in enumerate(zip(bars, menv_vals)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(menv_labels)))
ax.set_xticklabels(menv_labels, fontsize=8)
ax.set_ylabel(r"Force MAE (kcal mol$^{-1}$ Å$^{-1}$)")
ax.set_title("(B) Per-Microenvironment Force Error", fontsize=11, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel C: Parity plot (力分量, 随机下采样避免过密) ---
ax = axes[1, 0]
rng = np.random.default_rng(42)
flat_pred = pred_f.reshape(-1)
flat_true = true_f.reshape(-1)
n_sample = min(15000, flat_pred.shape[0])
idx = rng.choice(flat_pred.shape[0], size=n_sample, replace=False)
ax.scatter(flat_true[idx], flat_pred[idx], s=2.5, alpha=0.25, color="#1f77b4", linewidths=0)
lims = [min(flat_true.min(), flat_pred.min()), max(flat_true.max(), flat_pred.max())]
ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.0, zorder=5)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel(r"DFT Reference Force Component (kcal mol$^{-1}$ Å$^{-1}$)")
ax.set_ylabel(r"PaiNN Predicted Force Component (kcal mol$^{-1}$ Å$^{-1}$)")
ax.set_title("(C) Force-Component Parity Plot", fontsize=11, fontweight="bold", loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_aspect("equal")

r2 = 1 - np.sum((flat_pred - flat_true) ** 2) / np.sum((flat_true - flat_true.mean()) ** 2)
ax.text(0.05, 0.92, f"$R^2$ = {r2:.4f}\nMAE = {float(d['test_f_mae']):.4f}", transform=ax.transAxes,
        fontsize=9.5, va="top", bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85))

# --- Panel D: 力误差直方图 (全体原子, per-component abs error) ---
ax = axes[1, 1]
all_abs_err = np.abs(flat_pred - flat_true)
ax.hist(all_abs_err, bins=60, color="#6baed6", edgecolor="black", linewidth=0.3)
ax.axvline(float(d["test_f_mae"]), color="#d62728", linestyle="--", linewidth=1.3,
           label=f"MAE = {float(d['test_f_mae']):.4f}")
ax.set_xlabel(r"Absolute Force Error (kcal mol$^{-1}$ Å$^{-1}$)")
ax.set_ylabel("Count (per force component, 1000 test frames × 12 atoms × 3)")
ax.set_title("(D) Force Error Distribution", fontsize=11, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, np.percentile(all_abs_err, 99.5))

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_uracil_error_distribution.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig4_uracil_error_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved fig4_uracil_error_distribution.{pdf,png}")
