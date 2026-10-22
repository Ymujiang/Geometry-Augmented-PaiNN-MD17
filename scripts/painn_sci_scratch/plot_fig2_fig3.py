"""
绘制 fig2 (收敛曲线 + 消融瀑布图) 和 fig3 (负面结果对比图)。
数据来源: papers/painn_sci/data/painn_log_parsed.json (真实日志解析结果,
见 scripts/painn_sci_scratch/parse_painn_log.py)。
必须使用 .autosota/venv/bin/python3.11 执行,输出 300+ DPI 的 PDF+PNG。
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/ymujdb/.dsh/skills/figure-style")
from kernel import apply_figure_style
apply_figure_style()
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATA_PATH = "/home/ymujdb/my_sota_project/papers/painn_sci/data/painn_log_parsed.json"
FIG_DIR = "/home/ymujdb/my_sota_project/papers/painn_sci/figures"

with open(DATA_PATH) as f:
    iterations = json.load(f)

by_tag = {it["iter_tag"]: it for it in iterations}

# ---------------------------------------------------------------------------
# FIG 2: 收敛曲线 (左) + 消融瀑布图 (右)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Panel A: 训练收敛曲线 (Val Force MAE vs Epoch), 选取关键迭代 ---
ax = axes[0]
curves_to_plot = [
    ("iter-0", "Baseline (iter-0)", "#999999", "--"),
    ("iter-2", "+Sigmoid gate (iter-2)", "#1f77b4", "-"),
    ("iter-7", "+Tanh gate (iter-7)", "#d62728", "-"),
    ("iter-final", "Final model", "#2ca02c", "-"),
]
for tag, label, color, ls in curves_to_plot:
    it = by_tag[tag]
    epochs = [e["epoch"] for e in it["epochs"]]
    force_mae = [e["val_force_mae"] for e in it["epochs"]]
    ax.plot(epochs, force_mae, marker="o", markersize=3.5, linewidth=1.6,
            color=color, linestyle=ls, label=label)

ax.set_xlabel("Training Epoch")
ax.set_ylabel(r"Validation Force MAE (kcal mol$^{-1}$ Å$^{-1}$)")
ax.set_title("(A) Training Convergence Curves", fontsize=11, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 82)

# --- Panel B: 消融瀑布图 ---
ax2 = axes[1]
# 只保留成功迭代的累积序列 (baseline -> iter1 -> iter2 -> iter4 -> iter7 -> final)
waterfall_order = ["iter-0", "iter-1", "iter-2", "iter-4", "iter-7", "iter-final"]
waterfall_labels = [
    "Baseline",
    "+Gaussian\n+Bessel RBF",
    "+Sigmoid\ngeo-gate",
    "+Bidirectional\nmessage",
    "+Tanh\ngeo-gate",
    "Restored\n(same model)",
]
values = [by_tag[t]["final_test_mae"] for t in waterfall_order]

n = len(values)
x = np.arange(n)
colors = ["#888888"] + ["#1f77b4"] * (n - 2) + ["#2ca02c"]

bar_bottoms = []
bar_heights = []
for i in range(n):
    if i == 0:
        bar_bottoms.append(0)
        bar_heights.append(values[0])
    else:
        top = max(values[i], values[i - 1])
        bottom = min(values[i], values[i - 1])
        bar_bottoms.append(bottom)
        bar_heights.append(top - bottom)

for i in range(n):
    ax2.bar(x[i], bar_heights[i], bottom=bar_bottoms[i], width=0.6,
            color=colors[i], edgecolor="black", linewidth=0.6, zorder=3)
    ax2.text(x[i], values[i] + 0.006, f"{values[i]:.4f}",
              ha="center", va="bottom", fontsize=8.0, fontweight="bold")

for i in range(1, n):
    ax2.plot([x[i - 1] + 0.3, x[i] - 0.3], [values[i - 1], values[i - 1]],
             color="gray", linestyle=":", linewidth=0.8, zorder=2)

ax2.set_xticks(x)
ax2.set_xticklabels(waterfall_labels, fontsize=8.0)
ax2.set_ylabel("Test force MAE (kcal mol$^{-1}$ Å$^{-1}$)")
ax2.set_title("(B) Stepwise Ablation Waterfall", fontsize=11, fontweight="bold", loc="left")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, 0.52)

total_improvement = (values[0] - values[-1]) / values[0] * 100
ax2.annotate(
    f"Total improvement:\n{values[0]:.4f} → {values[-1]:.4f}\n(−{total_improvement:.1f}%)",
    xy=(n - 1, values[-1]), xytext=(n - 2.6, 0.40),
    fontsize=8.5, ha="left",
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_convergence_and_ablation.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig2_convergence_and_ablation.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig2_convergence_and_ablation.{pdf,png}")

# ---------------------------------------------------------------------------
# FIG 3: 负面结果分析 (3 个失败尝试 vs 对应的成功基准)
# ---------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(8.5, 5.5))

# 失败尝试与其对比基准 (紧邻的前一个成功 best state)
comparisons = [
    ("iter-8", "iter-7", "Bessel basis\n20 to 32", "vs iter-7 best\n(0.3021)"),
    ("iter-3", "iter-2", "Multi-Scale JK +\nVector Readout", "vs iter-2 best\n(0.3430)"),
    ("iter-5", "iter-4", "Deeper 2-layer\nMLP Filter", "vs iter-4 best\n(0.3375)"),
    ("iter-6", "iter-4", "LayerNorm in\nInteraction/Mixing", "vs iter-4 best\n(0.3375)"),
]

labels = []
failed_vals = []
baseline_vals = []
for failed_tag, baseline_tag, label, sub in comparisons:
    labels.append(label)
    failed_vals.append(by_tag[failed_tag]["final_test_mae"])
    baseline_vals.append(by_tag[baseline_tag]["final_test_mae"])

x = np.arange(len(labels))
width = 0.35

bars1 = ax3.bar(x - width / 2, baseline_vals, width, label="Preceding best (kept)",
                 color="#2ca02c", edgecolor="black", linewidth=0.6)
bars2 = ax3.bar(x + width / 2, failed_vals, width, label="Failed attempt (rolled back)",
                 color="#d62728", edgecolor="black", linewidth=0.6)

for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, h + 0.004, f"{h:.4f}",
                  ha="center", va="bottom", fontsize=8.0)

for i, (failed_tag, baseline_tag, label, sub) in enumerate(comparisons):
    fv = failed_vals[i]
    bv = baseline_vals[i]
    pct = (fv - bv) / bv * 100
    ax3.annotate(f"+{pct:.1f}%", xy=(x[i] + width / 2, fv + 0.018),
                 ha="center", fontsize=8.5, color="#d62728", fontweight="bold")

ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylabel("Test force MAE (kcal mol$^{-1}$ Å$^{-1}$)")
ax3.set_title("Negative Results: Four Failed Architectural Modifications", fontsize=11.5, fontweight="bold")
ax3.legend(frameon=False, fontsize=9, loc="upper left")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_ylim(0, max(failed_vals) * 1.25)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_negative_results_analysis.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{FIG_DIR}/fig3_negative_results_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig3_negative_results_analysis.{pdf,png}")
