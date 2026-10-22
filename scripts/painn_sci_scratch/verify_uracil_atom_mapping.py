"""
严格核实脚本：MD17 Uracil 数据集原子序号 -> 化学官能团映射。

目的（终审复核第 2 项）：
  1. 从受保护文件 optimized_code/painn/data/md17_uracil_split.npz 中读取真实的
     原子序数数组 z（12 个原子）与测试集第一帧真实三维坐标 R（第一个 test 构型）。
  2. 仅凭真实原子间欧氏距离 + 标准共价键长阈值（重原子-重原子键长 < 1.65 Å，
     重原子-氢键长 < 1.20 Å）自动推导出每一对原子之间是否成键，构建分子图。
  3. 完全不凭先验猜测对原子序号 0..11 赋予化学标签，而是基于图结构（每个原子
     的成键种类、度数、以及是否处于环中）用化学规则程序化判定：
       - C 原子：度=3 且连接 2 个重原子+1 个 H  -> 环碳(C-H, 乙烯基碳)
                 度=3 且连接 O(双键候选,距离更短) + 2 个重原子 -> 羰基碳(C=O)
       - N 原子：连接 1 个 H -> 亚氨基氮 (N-H)
       - O 原子：只连接 1 个重原子(C)，且该 C-O 距离明显短于普通单键
                 (< 1.30 Å，羰基 C=O 典型键长 ~1.21 Å) -> 羰基氧
       - H 原子：连接到 C -> 芳环/乙烯基氢 (C-H)；连接到 N -> 亚氨基氢 (N-H)
  4. 输出每个原子序号对应的官能团标签，并与绘图脚本
     scripts/painn_sci_scratch/plot_fig4_error_analysis.py 中硬编码的分组
     (atom_type_groups / microenv_groups) 逐一比对，报告是否完全一致。

运行:
  /home/ymujdb/my_sota_project/.autosota/venv/bin/python \
      scripts/painn_sci_scratch/verify_uracil_atom_mapping.py
"""
import numpy as np

DATA_PATH = "/home/ymujdb/my_sota_project/optimized_code/painn/data/md17_uracil_split.npz"

# 标准共价键长阈值 (Angstrom)，仅用作"是否成键"的判据，不涉及任何官能团猜测
HEAVY_HEAVY_BOND_MAX = 1.65   # C-C, C-N, C-O, N-C 等重原子间单/双键上限
HEAVY_H_BOND_MAX = 1.20       # C-H / N-H 键长上限
CARBONYL_CO_MAX = 1.30        # 典型 C=O 键长 (~1.21 A) 明显短于 C-O 单键 (~1.43 A)，
                              # 用于区分"羰基氧"与假设性的其他氧连接方式


def element_symbol(z):
    return {6: "C", 7: "N", 8: "O", 1: "H"}[int(z)]


def main():
    data = np.load(DATA_PATH)
    z = data["z"]  # (12,)
    assert len(z) == 12, f"期望 12 个原子，实际 {len(z)}"
    assert sorted(z.tolist()) == sorted([6, 6, 6, 6, 7, 7, 8, 8, 1, 1, 1, 1]), \
        f"期望组成 4C+2N+2O+4H (C4H4N2O2)，实际 z={z.tolist()}"
    print(f"[OK] 原子序数组成校验通过: z = {z.tolist()}  (4C, 2N, 2O, 4H)")

    # 使用 test 集第一帧的真实坐标（与 eval_benchmark.py 使用的同一份受保护数据）
    R = data["R_test"][0]  # (12, 3)
    n = len(z)

    # --- 计算真实欧氏距离矩阵 ---
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(R[i] - R[j])

    print("\n[距离矩阵] (Angstrom, 4位小数):")
    header = "      " + "".join(f"{i:>8d}" for i in range(n))
    print(header)
    for i in range(n):
        row = "".join(f"{dist[i,j]:8.4f}" for j in range(n))
        print(f"{i:>4d}: {row}")

    # --- 根据阈值自动推导成键关系 (邻接表) ---
    bonds = []  # list of (i, j, dist)
    adjacency = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            zi, zj = int(z[i]), int(z[j])
            d = dist[i, j]
            is_h_bond = (zi == 1 or zj == 1)
            threshold = HEAVY_H_BOND_MAX if is_h_bond else HEAVY_HEAVY_BOND_MAX
            if d < threshold:
                bonds.append((i, j, d))
                adjacency[i].append((j, d))
                adjacency[j].append((i, d))

    print(f"\n[自动推导成键] 共检测到 {len(bonds)} 条键 (阈值: 重-重<{HEAVY_HEAVY_BOND_MAX}A, 重-H<{HEAVY_H_BOND_MAX}A):")
    for i, j, d in sorted(bonds):
        print(f"  idx {i}({element_symbol(z[i])}) - idx {j}({element_symbol(z[j])}) : {d:.4f} A")

    # 校验：应恰好是 Uracil 的 12 条共价键 (6 元环 + 2 个 C=O + 2 个 C-H + 2 个 N-H)
    # Uracil 骨架: N1-C2(=O)-N3(H)-C4(=O)-C5(H)=C6(H)-N1(H)
    # 重原子环: N1-C2-N3-C4-C5-C6-N1  (6 条环键) + C2=O + C4=O (2 条) + N1-H + N3-H (2 条) + C5-H + C6-H (2 条)
    # 共 6+2+2+2 = 12 条键
    print(f"\n[键数校验] 期望 12 条共价键 (6元环 + 2个C=O + 2个N-H + 2个C-H)，实际检测到 {len(bonds)} 条")

    # --- 基于邻接结构程序化判定每个原子的官能团标签，不凭猜测硬编码序号 ---
    labels = {}
    for i in range(n):
        zi = int(z[i])
        neighbors = adjacency[i]  # list of (j, dist)
        heavy_neighbors = [(j, d) for j, d in neighbors if int(z[j]) != 1]
        h_neighbors = [(j, d) for j, d in neighbors if int(z[j]) == 1]

        if zi == 6:  # Carbon
            # 是否连接到一个"短距离"氧 (羰基 C=O 判据：存在 O 邻居且 C-O 距离 < CARBONYL_CO_MAX)
            o_neighbors = [(j, d) for j, d in heavy_neighbors if int(z[j]) == 8]
            if o_neighbors and min(d for _, d in o_neighbors) < CARBONYL_CO_MAX:
                labels[i] = "Carbonyl_C (C=O carbon)"
            else:
                labels[i] = "Vinylic_C (ring C=C carbon)"
        elif zi == 7:  # Nitrogen
            if h_neighbors:
                labels[i] = "Ring_N (N-H amino/imino nitrogen)"
            else:
                labels[i] = "Ring_N (no attached H -- unexpected for uracil)"
        elif zi == 8:  # Oxygen
            # Uracil 中两个氧都是羰基氧，只连接一个重原子(C)，且 C=O 距离短
            if len(heavy_neighbors) == 1 and heavy_neighbors[0][1] < CARBONYL_CO_MAX:
                labels[i] = "Carbonyl_O (C=O oxygen)"
            else:
                labels[i] = f"Oxygen (unexpected bonding pattern: {heavy_neighbors})"
        elif zi == 1:  # Hydrogen
            assert len(neighbors) == 1, f"H 原子 idx={i} 的成键数应为1，实际={len(neighbors)}"
            bonded_to = int(z[neighbors[0][0]])
            if bonded_to == 6:
                labels[i] = "Vinylic_H (C-H, bonded to ring carbon)"
            elif bonded_to == 7:
                labels[i] = "Amino_H (N-H, bonded to ring nitrogen)"
            else:
                labels[i] = f"Hydrogen (unexpected bonding: Z={bonded_to})"
        else:
            labels[i] = f"Unknown element Z={zi}"

    print("\n[程序化推导的官能团标签] (完全基于真实坐标与键长阈值，无人工猜测序号):")
    for i in range(n):
        print(f"  idx {i:2d}  Z={int(z[i]):2d} ({element_symbol(z[i])})  ->  {labels[i]}")

    # --- 与绘图脚本中硬编码的分组进行逐一比对 ---
    hardcoded_atom_type = {"C": [0, 1, 3, 5], "N": [2, 4], "O": [6, 7], "H": [8, 9, 10, 11]}
    hardcoded_microenv = {
        "Carbonyl_C": [3, 5],
        "Vinylic_C": [0, 1],
        "Ring_N": [2, 4],
        "Carbonyl_O": [6, 7],
        "Amino_H": [9, 10],
        "Vinylic_H": [8, 11],
    }

    derived_atom_type = {"C": [], "N": [], "O": [], "H": []}
    for i in range(n):
        derived_atom_type[element_symbol(z[i])].append(i)

    derived_microenv = {
        "Carbonyl_C": [], "Vinylic_C": [], "Ring_N": [],
        "Carbonyl_O": [], "Amino_H": [], "Vinylic_H": [],
    }
    for i, lab in labels.items():
        if lab.startswith("Carbonyl_C"):
            derived_microenv["Carbonyl_C"].append(i)
        elif lab.startswith("Vinylic_C"):
            derived_microenv["Vinylic_C"].append(i)
        elif lab.startswith("Ring_N"):
            derived_microenv["Ring_N"].append(i)
        elif lab.startswith("Carbonyl_O"):
            derived_microenv["Carbonyl_O"].append(i)
        elif lab.startswith("Amino_H"):
            derived_microenv["Amino_H"].append(i)
        elif lab.startswith("Vinylic_H"):
            derived_microenv["Vinylic_H"].append(i)

    print("\n[比对结果: 元素类型分组]")
    atom_type_match = True
    for k in hardcoded_atom_type:
        hc = sorted(hardcoded_atom_type[k])
        dv = sorted(derived_atom_type[k])
        ok = (hc == dv)
        atom_type_match &= ok
        print(f"  {k}: 硬编码={hc}  程序化推导={dv}  {'OK' if ok else '不一致!!'}")

    print("\n[比对结果: 化学微环境分组]")
    microenv_match = True
    for k in hardcoded_microenv:
        hc = sorted(hardcoded_microenv[k])
        dv = sorted(derived_microenv[k])
        ok = (hc == dv)
        microenv_match &= ok
        print(f"  {k}: 硬编码={hc}  程序化推导={dv}  {'OK' if ok else '不一致!!'}")

    print("\n" + "=" * 70)
    if atom_type_match and microenv_match:
        print("[结论] 通过：绘图脚本中使用的原子序号->官能团映射与从真实坐标+键长")
        print("阈值程序化推导出的化学结构完全一致。映射未凭猜测硬编码，已严格核实。")
    else:
        print("[结论] 失败：存在不一致，绘图脚本中的分组需要修正！")
    print("=" * 70)


if __name__ == "__main__":
    main()
