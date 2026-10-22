"""
PaiNN SCI 深度数据提取脚本（一次性训练 + 推理 + 仪表化提取）。

严格遵循 optimized_code/painn/eval_benchmark.py 的训练协议（受保护文件，未做任何修改）：
  - set_seed(42)
  - cutoff=5.0, n_atom_basis=128, n_interactions=3
  - AdamW(lr=1e-3, wd=0) + CosineAnnealingLR(T_max=80, eta_min=1e-5)
  - loss = 0.01*MSE_E + 0.99*MSE_F, grad_clip=10.0
  - 每 5 epoch 用验证集 force MAE 挑选 best_state（仅保存在内存，不落盘 —— 与原脚本一致）
  - 80 epoch 结束后在 best_state 上做一次测试集评估

本脚本在上述协议基础上【仅新增只读式仪表化】，不修改 optimized_code/ 下任何文件：
  1. 通过 sys.path 注入指定的 src 树（与 eval_benchmark.py 完全相同的技巧）来选择
     使用"最终 tanh 门控版本"还是"iter-4 sigmoid 门控版本"的 painn.py。
  2. 用 register_forward_hook 挂载在每个 PaiNNInteraction.geo_gate_net（Dense 线性层）
     上，捕获其输出的"门控原始 logit"（线性层输出，位于 tanh/sigmoid 之前）。
     —— 这是非侵入式监听，不修改任何 forward() 源码逻辑。
  3. 测试集推理时，对每个样本的 12 个原子分别保存预测力 / 真实力 / 预测能量 / 真实能量，
     用于逐原子类型与逐化学微环境的误差分解。

用法：
  /home/ymujdb/miniconda3/envs/cgcnn_env/bin/python train_and_extract.py --variant final --out-dir <dir>
  /home/ymujdb/miniconda3/envs/cgcnn_env/bin/python train_and_extract.py --variant sigmoid --out-dir <dir>
"""
import os
import sys
import time
import random
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAINN_REPO = "/home/ymujdb/my_sota_project/optimized_code/painn"
SIGMOID_SRC = "/tmp/painn_sci_src_sigmoid/src"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(R_batch, Z_single, cutoff, device, properties):
    B, N_atoms, _ = R_batch.shape
    R_b = R_batch.reshape(B * N_atoms, 3)
    Z_b = Z_single.unsqueeze(0).expand(B, N_atoms).reshape(B * N_atoms)
    idx_m = torch.arange(B, device=device).unsqueeze(1).expand(B, N_atoms).reshape(B * N_atoms)

    diff = R_batch.unsqueeze(2) - R_batch.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    mask = (dist > 0.0) & (dist <= cutoff)
    b_idx, i_local, j_local = torch.where(mask)
    idx_i = b_idx * N_atoms + i_local
    idx_j = b_idx * N_atoms + j_local
    offsets = torch.zeros((len(idx_i), 3), dtype=torch.float32, device=device)

    return {
        properties.Z: Z_b,
        properties.R: R_b,
        properties.idx_m: idx_m,
        properties.idx_i: idx_i,
        properties.idx_j: idx_j,
        properties.offsets: offsets,
        properties.n_atoms: torch.full((B,), N_atoms, dtype=torch.long, device=device),
    }


def evaluate(model, R_data, E_data, F_data, Z_single, cutoff, batch_size, device, properties):
    # 注意：不能用 torch.no_grad() 包裹 —— PaiNN 的 forces 是通过
    # autograd 对能量求 -dE/dR 得到的，需要保留计算图。这与受保护的
    # eval_benchmark.py 中 evaluate() 的实现完全一致（它也不使用 no_grad）。
    model.eval()
    total_f_abs = 0.0
    total_f_count = 0
    total_e_abs = 0.0
    total_e_count = 0
    for start in range(0, len(R_data), batch_size):
        R_b = R_data[start:start + batch_size].detach().clone()
        E_b = E_data[start:start + batch_size]
        F_b = F_data[start:start + batch_size]
        B, N_atoms, _ = R_b.shape
        inputs = make_batch(R_b, Z_single, cutoff, device, properties)
        out = model(inputs)
        pred_e = out[properties.energy].detach()
        pred_f = out[properties.forces].detach().reshape(B, N_atoms, 3)
        total_e_abs += torch.sum(torch.abs(pred_e - E_b)).item()
        total_e_count += B
        total_f_abs += torch.sum(torch.abs(pred_f - F_b)).item()
        total_f_count += F_b.numel()
    return total_f_abs / total_f_count, total_e_abs / total_e_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["final", "sigmoid"], required=True,
                     help="final = 当前 _best (tanh geometric gate); sigmoid = iter-4 (sigmoid geometric gate) 对照组")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- sys.path 注入：与 eval_benchmark.py 完全相同的手法，选择不同的 src 树 ---
    # 必须保证 schnetpack 包只从目标 src 树导入一次（python 模块缓存问题），
    # 因此清掉可能已导入的 schnetpack* 模块缓存。
    for mod_name in list(sys.modules.keys()):
        if mod_name == "schnetpack" or mod_name.startswith("schnetpack."):
            del sys.modules[mod_name]

    if args.variant == "final":
        src_path = os.path.join(PAINN_REPO, "src")
        gate_activation_name = "tanh"
    else:
        src_path = SIGMOID_SRC
        gate_activation_name = "sigmoid"

    sys.path.insert(0, src_path)

    import schnetpack as spk
    import schnetpack.properties as properties
    from schnetpack.representation.painn import PaiNNInteraction

    print(f"[info] variant={args.variant}  schnetpack from: {spk.__file__}")
    print(f"[info] gate activation (known from source checkout) = {gate_activation_name}")

    set_seed(42)
    device = torch.device("cpu")

    data_path = os.path.join(PAINN_REPO, "data", "md17_uracil_split.npz")
    data = np.load(data_path)

    Z_single = torch.tensor(data["z"], dtype=torch.long, device=device)
    R_train = torch.tensor(data["R_train"], dtype=torch.float32, device=device)
    E_train = torch.tensor(data["E_train"], dtype=torch.float32, device=device)
    F_train = torch.tensor(data["F_train"], dtype=torch.float32, device=device)

    R_val = torch.tensor(data["R_val"], dtype=torch.float32, device=device)
    E_val = torch.tensor(data["E_val"], dtype=torch.float32, device=device)
    F_val = torch.tensor(data["F_val"], dtype=torch.float32, device=device)

    R_test = torch.tensor(data["R_test"], dtype=torch.float32, device=device)
    E_test = torch.tensor(data["E_test"], dtype=torch.float32, device=device)
    F_test = torch.tensor(data["F_test"], dtype=torch.float32, device=device)

    E_mean = E_train.mean()
    E_train = E_train - E_mean
    E_val = E_val - E_mean
    E_test = E_test - E_mean

    cutoff = 5.0
    n_atom_basis = 128
    n_interactions = 3
    batch_size = 32
    epochs = 80

    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    painn = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )
    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=properties.energy)
    pred_forces = spk.atomistic.Forces(energy_key=properties.energy, force_key=properties.forces)

    model = spk.model.NeuralNetworkPotential(
        representation=painn,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[pred_energy, pred_forces],
        do_postprocessing=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f_mae = float("inf")
    best_state = None
    t0 = time.time()

    num_train = len(R_train)
    N_atoms = R_train.shape[1]

    epoch_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train, device=device)
        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size]
            B = len(idx)
            R_b = R_train[idx]
            E_b = E_train[idx]
            F_b = F_train[idx].reshape(B * N_atoms, 3)

            inputs = make_batch(R_b, Z_single, cutoff, device, properties)
            out = model(inputs)

            loss_e = F.mse_loss(out[properties.energy], E_b)
            loss_f = F.mse_loss(out[properties.forces], F_b)
            loss = 0.01 * loss_e + 0.99 * loss_f

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            val_f_mae, val_e_mae = evaluate(model, R_val, E_val, F_val, Z_single, cutoff, 50, device, properties)
            if val_f_mae < best_val_f_mae:
                best_val_f_mae = val_f_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epoch_log.append({"epoch": epoch, "val_force_mae": val_f_mae, "val_energy_mae": val_e_mae,
                               "time_s": time.time() - t0})
            print(f"Epoch {epoch:03d}/{epochs:03d} | Val_Force_MAE: {val_f_mae:.4f} | "
                  f"Val_Energy_MAE: {val_e_mae:.4f} | Time: {time.time()-t0:.1f}s")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_f_mae, test_e_mae = evaluate(model, R_test, E_test, F_test, Z_single, cutoff, 100, device, properties)
    print(f"Test_Energy_MAE: {test_e_mae:.4f}")
    print(f"Final_Test_MAE: {test_f_mae:.4f}")

    # ---------------------------------------------------------------
    # 仪表化推理：在 best_state 下，对整个测试集做一次前向传播，
    # 同时 (a) 收集逐原子力预测 / 真值，(b) 通过 forward hook 捕获
    # 每个 PaiNNInteraction.geo_gate_net 的原始 logit（未过激活函数）。
    # ---------------------------------------------------------------
    gate_logits_per_layer = [[] for _ in range(n_interactions)]

    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            gate_logits_per_layer[layer_idx].append(out.detach().cpu().numpy())
        return hook

    for li, interaction in enumerate(model.representation.interactions):
        h = interaction.geo_gate_net.register_forward_hook(make_hook(li))
        hooks.append(h)

    # 同样不能用 no_grad() —— forces 来自对 energy 的 autograd 求导。
    model.eval()
    all_pred_f = []
    all_true_f = []
    all_pred_e = []
    all_true_e = []
    batch_size_test = 100
    for start in range(0, len(R_test), batch_size_test):
        R_b = R_test[start:start + batch_size_test].detach().clone()
        E_b = E_test[start:start + batch_size_test]
        F_b = F_test[start:start + batch_size_test]
        B, N_atoms_, _ = R_b.shape
        inputs = make_batch(R_b, Z_single, cutoff, device, properties)
        out = model(inputs)
        pred_e = out[properties.energy].detach().cpu().numpy()
        pred_f = out[properties.forces].detach().reshape(B, N_atoms_, 3).cpu().numpy()
        all_pred_f.append(pred_f)
        all_true_f.append(F_b.cpu().numpy())
        all_pred_e.append(pred_e)
        all_true_e.append(E_b.cpu().numpy())

    for h in hooks:
        h.remove()

    all_pred_f = np.concatenate(all_pred_f, axis=0)   # (N_test, 12, 3)
    all_true_f = np.concatenate(all_true_f, axis=0)
    all_pred_e = np.concatenate(all_pred_e, axis=0)
    all_true_e = np.concatenate(all_true_e, axis=0)

    # 拼接每层门控 logits: list of (n_pairs_in_batch, 1, 3*n_atom_basis)
    gate_logits_concat = [np.concatenate(layer_list, axis=0) for layer_list in gate_logits_per_layer]

    np.savez_compressed(
        os.path.join(args.out_dir, f"predictions_{args.variant}.npz"),
        pred_forces=all_pred_f,
        true_forces=all_true_f,
        pred_energy=all_pred_e,
        true_energy=all_true_e,
        z=data["z"],
        test_f_mae=test_f_mae,
        test_e_mae=test_e_mae,
    )

    # 门控 logits 体积可能很大 (n_pairs x 384)，只保存统计摘要 + 下采样样本用于绘图
    gate_summary = {}
    rng = np.random.default_rng(42)
    for li, arr in enumerate(gate_logits_concat):
        flat = arr.reshape(-1)  # 展平所有 pair x 3*n_atom_basis 维度
        if gate_activation_name == "tanh":
            activated = np.tanh(flat)
        else:
            activated = 1.0 / (1.0 + np.exp(-flat))
        n_sample = min(50000, flat.shape[0])
        sample_idx = rng.choice(flat.shape[0], size=n_sample, replace=False)
        gate_summary[f"layer_{li}"] = {
            "logit_mean": float(flat.mean()),
            "logit_std": float(flat.std()),
            "activated_mean": float(activated.mean()),
            "activated_std": float(activated.std()),
            "activated_sample": activated[sample_idx].astype(np.float32).tolist(),
        }

    with open(os.path.join(args.out_dir, f"gate_activation_{args.variant}.json"), "w") as f:
        json.dump({
            "variant": args.variant,
            "gate_activation_name": gate_activation_name,
            "n_interactions": n_interactions,
            "layers": gate_summary,
        }, f)

    with open(os.path.join(args.out_dir, f"training_curve_{args.variant}.json"), "w") as f:
        json.dump({
            "variant": args.variant,
            "epoch_log": epoch_log,
            "test_f_mae": test_f_mae,
            "test_e_mae": test_e_mae,
            "best_val_f_mae": best_val_f_mae,
        }, f, indent=2)

    print(f"[done] variant={args.variant} 已写出 predictions_/gate_activation_/training_curve_ 到 {args.out_dir}")


if __name__ == "__main__":
    main()
