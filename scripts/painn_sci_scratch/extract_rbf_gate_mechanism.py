"""
提取图1所需的"权重无关"物理机制数据：
  1. GaussianRBF(n_rbf=20, cutoff=5.0) 与 BesselRBF(n_rbf=20, cutoff=5.0) 在 r∈[0,cutoff] 上的
     响应曲线（纯函数形式，来自 optimized_code/painn/src/schnetpack/nn/radial.py，与训练权重无关）。
  2. CosineCutoff(5.0) 包络函数曲线。
  3. 3-body 门控机制示意所需的解析几何关系（s1=<mu_i,dir_ij>, s2=<mu_j,dir_ij>, s3=<mu_i,mu_j>）
     用一个合成的三体（最近邻键角可变）toy case 展示：固定 bond length，扫描键角 theta，
     观察 tanh(gate) 的理论饱和特性（纯数学函数形状，不依赖训练权重）。
  4. uracil 测试集第一帧的真实原子间距分布，标出各 RBF 中心覆盖情况（展示为何需要双重基组）。

本脚本只读取 optimized_code/painn 下的源码与受保护的 npz 数据（只读 np.load，不做任何写入），
不依赖 torch，可用 .autosota/venv 的纯 numpy 环境运行。
"""
import os
import sys
import json
import math
import numpy as np

PAINN_REPO = "/home/ymujdb/my_sota_project/optimized_code/painn"


def gaussian_rbf(r, n_rbf=20, cutoff=5.0, start=0.0):
    offset = np.linspace(start, cutoff, n_rbf)
    widths = np.abs(offset[1] - offset[0]) * np.ones_like(offset)
    diff = r[..., None] - offset[None, :]
    coeff = -0.5 / (widths ** 2)
    return np.exp(coeff[None, :] * diff ** 2), offset


def bessel_rbf(r, n_rbf=20, cutoff=5.0):
    freqs = np.arange(1, n_rbf + 1) * math.pi / cutoff
    ax = r[..., None] * freqs[None, :]
    sinax = np.sin(ax)
    norm = np.where(r == 0, 1.0, r)
    y = sinax / norm[..., None]
    return y, freqs


def cosine_cutoff(r, cutoff=5.0):
    f = 0.5 * (np.cos(r * math.pi / cutoff) + 1.0)
    f = f * (r < cutoff).astype(float)
    return f


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "papers/painn_sci/data"
    os.makedirs(out_dir, exist_ok=True)

    cutoff = 5.0
    n_gauss = 20
    n_bessel = 20
    r = np.linspace(0.001, cutoff, 1000)

    gauss_curves, gauss_offsets = gaussian_rbf(r, n_rbf=n_gauss, cutoff=cutoff)
    bessel_curves, bessel_freqs = bessel_rbf(r, n_rbf=n_bessel, cutoff=cutoff)
    fcut = cosine_cutoff(r, cutoff=cutoff)

    # 频域特性粗略刻画：每个基函数的"有效波数"（Bessel 为 freqs/pi 本身即波数；
    # Gaussian 的有效带宽正比于 1/width）
    gauss_width = float(np.abs(gauss_offsets[1] - gauss_offsets[0]))
    gauss_eff_wavenumber = 1.0 / gauss_width  # 粗略：高斯函数局部特征波数尺度

    # --- 真实 Uracil 测试集第一帧的成键原子间距（仅用于标注 RBF 覆盖范围的真实物理背景） ---
    data_path = os.path.join(PAINN_REPO, "data", "md17_uracil_split.npz")
    data = np.load(data_path)
    z = data["z"]
    R0 = data["R_test"][0]  # (12,3)
    n_atoms = len(z)
    bond_dists = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = float(np.linalg.norm(R0[i] - R0[j]))
            if d < cutoff:
                bond_dists.append(d)
    bond_dists = sorted(bond_dists)

    # 全测试集（1000帧）中所有在 cutoff 内的原子对距离，用于更稳健的直方图统计
    R_test = data["R_test"]
    all_dists = []
    for f in range(0, R_test.shape[0], 20):  # 抽样每隔20帧，降低计算量，仍覆盖全测试集分布
        Rf = R_test[f]
        diff = Rf[:, None, :] - Rf[None, :, :]
        dmat = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(n_atoms, k=1)
        dvals = dmat[iu]
        dvals = dvals[dvals < cutoff]
        all_dists.extend(dvals.tolist())
    all_dists = np.array(all_dists)

    # --- 3-body 几何门控：解析扫描 bond-angle theta，展示 s1/s2/s3 的几何依赖关系 ---
    # 固定 |mu_i| = |mu_j| = 1（单位向量场的示意情形），dir_ij 固定在 x 轴，
    # mu_i 在 xy 平面内随 theta 旋转，mu_j 固定沿 dir_ij。
    thetas = np.linspace(0, math.pi, 200)
    dir_ij = np.array([1.0, 0.0])
    mu_j = np.array([1.0, 0.0])  # 沿键方向
    s1_list, s2_list, s3_list = [], [], []
    for th in thetas:
        mu_i = np.array([math.cos(th), math.sin(th)])
        s1 = float(np.dot(mu_i, dir_ij))
        s2 = float(np.dot(mu_j, dir_ij))
        s3 = float(np.dot(mu_i, mu_j))
        s1_list.append(s1)
        s2_list.append(s2)
        s3_list.append(s3)

    # tanh 和 sigmoid 门控函数形状对比（标量输入到激活值的映射，纯数学函数，不依赖训练权重）
    gate_input = np.linspace(-6, 6, 400)
    tanh_vals = np.tanh(gate_input)
    sigmoid_vals = 1.0 / (1.0 + np.exp(-gate_input))
    scale_tanh = 1.0 + tanh_vals        # PaiNN 实现中的最终标量: (1+gate) in (0,2)
    scale_sigmoid = 1.0 + sigmoid_vals  # 对照: sigmoid 版本的 (1+gate) in (1,2)

    out = {
        "cutoff": cutoff,
        "n_gauss": n_gauss,
        "n_bessel": n_bessel,
        "r": r.tolist(),
        "gaussian_curves": gauss_curves.tolist(),   # shape (len(r), n_gauss)
        "gaussian_offsets": gauss_offsets.tolist(),
        "gaussian_width": gauss_width,
        "bessel_curves": bessel_curves.tolist(),    # shape (len(r), n_bessel)
        "bessel_freqs": bessel_freqs.tolist(),
        "cosine_cutoff_fn": fcut.tolist(),
        "bond_dists_frame0": bond_dists,
        "all_pair_dists_sampled": all_dists.tolist(),
        "theta_scan": {
            "theta": thetas.tolist(),
            "s1": s1_list,
            "s2": s2_list,
            "s3": s3_list,
        },
        "gate_shape": {
            "gate_input": gate_input.tolist(),
            "tanh": tanh_vals.tolist(),
            "sigmoid": sigmoid_vals.tolist(),
            "scale_tanh": scale_tanh.tolist(),
            "scale_sigmoid": scale_sigmoid.tolist(),
        },
    }

    out_path = os.path.join(out_dir, "fig1_rbf_gate_data.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"[done] wrote {out_path}")
    print(f"n_bond_dists(frame0)={len(bond_dists)}  n_all_pair_dists(sampled)={len(all_dists)}")
    print(f"gaussian_width={gauss_width:.4f}  bessel_freqs[:3]={bessel_freqs[:3]}")


if __name__ == "__main__":
    main()
