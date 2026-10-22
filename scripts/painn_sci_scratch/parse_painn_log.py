"""
从 logs/sota/painn.log 中解析每一次迭代的 epoch 级验证曲线、最终测试指标、
commit hash 与成功/失败标记,生成结构化 JSON,供 fig2(收敛曲线+消融瀑布图)
和 fig3(负面结果对比图)绘图脚本使用。全部数据来自真实日志文本,无任何推测
或编造数值。
"""
import json
import re

LOG_PATH = "/home/ymujdb/my_sota_project/logs/sota/painn.log"

with open(LOG_PATH, "r", errors="ignore") as f:
    lines = f.readlines()

# 迭代边界标记 (commit 行) 用于切分日志
commit_pattern = re.compile(
    r"\[record_score\] git commit:\s*(\S+)\s*—\s*(iter-\S+):\s*(.*?)\s*\[(success|failed)\]"
)
epoch_pattern = re.compile(
    r"Epoch (\d+)/(\d+) \| Val_Force_MAE: ([\d.]+) \| Val_Energy_MAE: ([\d.]+) \| Time: ([\d.]+)s"
)
final_mae_pattern = re.compile(r"Final_Test_MAE:\s*([\d.]+)")

# 先找到所有 commit 记录行号及其内容
commit_hits = []
for i, line in enumerate(lines):
    m = commit_pattern.search(line)
    if m:
        commit_hits.append({
            "line_no": i,
            "commit": m.group(1),
            "iter_tag": m.group(2),
            "desc": m.group(3).strip(),
            "status": m.group(4),
        })

print(f"Found {len(commit_hits)} commit records")
for c in commit_hits:
    print(f"  line={c['line_no']:5d} {c['commit']} {c['iter_tag']:12s} [{c['status']:7s}] {c['desc']}")

# 对每个 commit 记录,向前搜索最近的一段 Epoch 曲线 + Final_Test_MAE
iterations = []
search_start = 0
for idx, c in enumerate(commit_hits):
    # 在 [search_start, c.line_no] 区间内找 epoch 曲线和 Final_Test_MAE
    window = lines[search_start:c["line_no"] + 1]
    epochs = []
    for wl in window:
        em = epoch_pattern.search(wl)
        if em:
            epochs.append({
                "epoch": int(em.group(1)),
                "total_epochs": int(em.group(2)),
                "val_force_mae": float(em.group(3)),
                "val_energy_mae": float(em.group(4)),
                "time_s": float(em.group(5)),
            })
    final_mae = None
    for wl in reversed(window):
        fm = final_mae_pattern.search(wl)
        if fm:
            final_mae = float(fm.group(1))
            break
    iterations.append({
        "commit": c["commit"],
        "iter_tag": c["iter_tag"],
        "desc": c["desc"],
        "status": c["status"],
        "final_test_mae": final_mae,
        "epochs": epochs,
    })
    search_start = c["line_no"] + 1

print(f"\nParsed {len(iterations)} iterations with epoch curves")
for it in iterations:
    print(f"  {it['iter_tag']:12s} [{it['status']:7s}] final_mae={it['final_test_mae']}  n_epochs_logged={len(it['epochs'])}  desc={it['desc']}")

with open("/home/ymujdb/my_sota_project/papers/painn_sci/data/painn_log_parsed.json", "w") as f:
    json.dump(iterations, f, indent=2)

print("\nWritten to papers/painn_sci/data/painn_log_parsed.json")
