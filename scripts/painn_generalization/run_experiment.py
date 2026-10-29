"""
去偏与跨分子泛化补充实验 — 统一训练/评估运行脚本。

设计目标（对应终审建议第5项）：
  阶段 A：Uracil 独立多随机种子验证 —— 架构与超参数完全固定（与
          optimized_code/painn/eval_benchmark.py 协议逐项相同），仅更换
          模型初始化 / 数据打乱所使用的随机种子，消除单一种子偶然性。
  阶段 B：跨分子零调参泛化测试 —— 不修改任何一行模型代码或任何超参数，
          直接将 baseline PaiNN 与改进版 PaiNN 应用到未参与架构选择的
          新分子（Ethanol、Aspirin）。

本脚本是 eval_benchmark.py 训练/评估协议的参数化复制，严格保持以下
受保护设定不变（与 eval_benchmark.py 源码逐行比对确认一致）：
  - cutoff = 5.0, n_atom_basis = 128, n_interactions = 3
  - batch_size = 32, epochs = 80
  - AdamW(lr=1e-3, weight_decay=0.0)
  - CosineAnnealingLR(T_max=epochs, eta_min=1e-5)
  - loss = 0.01 * MSE_energy + 0.99 * MSE_force
  - grad_clip max_norm = 10.0
  - 每 5 个 epoch（含最后一个 epoch）在验证集上评估一次，按验证集
    Force MAE 保留 best_state（仅内存保留，不落盘，与原脚本一致）
  - 训练结束后用 best_state 在测试集上做唯一一次最终评估

与 eval_benchmark.py 的唯一差异（均为新增的只读式参数化，不改变任何
训练/评估逻辑本身）：
  1. 随机种子可通过 --seed 指定（原脚本固定写死 set_seed(42)）。
  2. 分子与数据集 split 文件可通过 --molecule / --split-npz 指定
     （原脚本固定写死 uracil 的 split 路径）。
  3. 模型代码来源（baseline 或 improved 架构）可通过 --src-path 指定
     （原脚本固定 import 改进版 src/schnetpack）。
  4. 日志以追加模式写入 --log-file，同时将结构化结果写入 --out-json。

Uracil 的 protected split 文件（data/md17_uracil_split.npz）本身保持
byte-for-byte 不变 —— 阶段 A 只改变 --seed，不改变 --split-npz。
"""
import os
import sys
import time
import random
import argparse
import json
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int):
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
    ap.add_argument("--molecule", required=True, help="molecule tag, e.g. uracil/ethanol/aspirin")
    ap.add_argument("--arch", required=True, choices=["baseline", "improved"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split-npz", required=True, help="path to <molecule>_split.npz")
    ap.add_argument("--src-path", required=True, help="path to the schnetpack src tree to import")
    ap.add_argument("--log-file", required=True, help="append-mode shared log file")
    ap.add_argument("--out-json", required=True, help="per-run structured result json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--save-checkpoint", default=None,
                     help="optional path to save the best-validation-epoch model "
                          "state dict (torch.save) after the run completes. Purely "
                          "additive instrumentation: saving happens after the fixed "
                          "80-epoch training/evaluation protocol is finished and does "
                          "not alter training, selection, or evaluation logic in any "
                          "way. Previously run campaigns (eval_benchmark.py and this "
                          "script's own supplementary runs) did not save a checkpoint "
                          "and their numbers are unaffected retroactively; this flag "
                          "only affects runs launched with it from now on.")
    ap.add_argument("--epochs-override", type=int, default=None,
                     help="SMOKE-TEST ONLY: override the protocol's 80 epochs. "
                          "Must never be used for a reported result.")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    # Ensure the requested src tree is imported first (same sys.path trick as
    # eval_benchmark.py, generalized to point at baseline or improved code).
    sys.path.insert(0, args.src_path)
    import schnetpack as spk
    import schnetpack.properties as properties

    run_tag = f"[{args.molecule}|{args.arch}|seed={args.seed}]"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {run_tag} {msg}"
        print(line, flush=True)
        with open(args.log_file, "a") as f:
            f.write(line + "\n")

    log(f"RUN START host={platform.node()} src_path={args.src_path} split={args.split_npz} "
        f"threads={args.threads} painn_module_file={spk.representation.__file__ if hasattr(spk.representation,'__file__') else 'n/a'}")
    # Record the exact painn.py file resolved by this interpreter, to make the
    # baseline-vs-improved source selection independently auditable.
    import schnetpack.representation.painn as painn_mod
    log(f"resolved painn.py = {painn_mod.__file__}")

    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f"device = {device}")

    data = np.load(args.split_npz)

    Z_single = torch.tensor(data['z'], dtype=torch.long, device=device)
    R_train = torch.tensor(data['R_train'], dtype=torch.float32, device=device)
    E_train = torch.tensor(data['E_train'], dtype=torch.float32, device=device)
    F_train = torch.tensor(data['F_train'], dtype=torch.float32, device=device)

    R_val = torch.tensor(data['R_val'], dtype=torch.float32, device=device)
    E_val = torch.tensor(data['E_val'], dtype=torch.float32, device=device)
    F_val = torch.tensor(data['F_val'], dtype=torch.float32, device=device)

    R_test = torch.tensor(data['R_test'], dtype=torch.float32, device=device)
    E_test = torch.tensor(data['E_test'], dtype=torch.float32, device=device)
    F_test = torch.tensor(data['F_test'], dtype=torch.float32, device=device)

    log(f"n_atoms={R_train.shape[1]} train={len(R_train)} val={len(R_val)} test={len(R_test)} z={data['z'].tolist()}")

    # Remove training energy mean offset (standard SchNetPack RemoveOffsets),
    # identical to eval_benchmark.py.
    E_mean = E_train.mean()
    E_train = E_train - E_mean
    E_val = E_val - E_mean
    E_test = E_test - E_mean

    cutoff = 5.0
    n_atom_basis = 128
    n_interactions = 3
    batch_size = 32
    epochs = 80 if args.epochs_override is None else args.epochs_override
    if args.epochs_override is not None:
        log(f"WARNING: epochs overridden to {epochs} for smoke-test only; "
            f"this run's test_f_mae/test_e_mae MUST NOT be reported as a protocol result.")

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

    n_params = sum(p.numel() for p in model.parameters())
    log(f"n_params={n_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f_mae = float('inf')
    best_val_e_mae_at_best = None
    best_epoch = None
    best_state = None
    t0 = time.time()

    num_train = len(R_train)
    N_atoms = R_train.shape[1]

    val_history = []

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

        if epoch % args.val_every == 0 or epoch == epochs:
            val_f_mae, val_e_mae = evaluate(model, R_val, E_val, F_val, Z_single, cutoff, 50, device, properties)
            val_history.append({"epoch": epoch, "val_f_mae": val_f_mae, "val_e_mae": val_e_mae})
            improved = val_f_mae < best_val_f_mae
            if improved:
                best_val_f_mae = val_f_mae
                best_val_e_mae_at_best = val_e_mae
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(f"Epoch {epoch:03d}/{epochs:03d} | Val_Force_MAE: {val_f_mae:.4f} | "
                f"Val_Energy_MAE: {val_e_mae:.4f} | Time: {time.time()-t0:.1f}s"
                f"{'  <- new best' if improved else ''}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_f_mae, test_e_mae = evaluate(model, R_test, E_test, F_test, Z_single, cutoff, 100, device, properties)
    total_time = time.time() - t0

    # Optional checkpoint save: additive only, runs after the
    # best-validation-epoch state has already been selected and evaluated
    # above by the unchanged protocol. Writing to disk here cannot influence
    # best_epoch, best_val_f_mae, test_f_mae, or test_e_mae, which are already
    # computed. Absent --save-checkpoint, behaviour is identical to before this
    # option was added (no file is written, matching every run already reported
    # in the paper).
    checkpoint_path = None
    if args.save_checkpoint is not None and best_state is not None:
        checkpoint_path = os.path.abspath(args.save_checkpoint)
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_val_f_mae": best_val_f_mae,
                "best_val_e_mae": best_val_e_mae_at_best,
                "test_f_mae": test_f_mae,
                "test_e_mae": test_e_mae,
                "molecule": args.molecule,
                "arch": args.arch,
                "seed": args.seed,
                "src_path": os.path.abspath(args.src_path),
            },
            checkpoint_path,
        )
        log(f"checkpoint saved to {checkpoint_path} (best_epoch={best_epoch})")

    log(f"Test_Energy_MAE: {test_e_mae:.4f}")
    log(f"Final_Test_MAE: {test_f_mae:.4f}")
    log(f"RUN END best_epoch={best_epoch} best_val_f_mae={best_val_f_mae:.4f} "
        f"test_f_mae={test_f_mae:.4f} test_e_mae={test_e_mae:.4f} total_time_s={total_time:.1f}")

    result = {
        "molecule": args.molecule,
        "arch": args.arch,
        "seed": args.seed,
        "split_npz": os.path.abspath(args.split_npz),
        "src_path": os.path.abspath(args.src_path),
        "painn_py_file": painn_mod.__file__,
        "n_atoms": int(R_train.shape[1]),
        "n_train": int(len(R_train)),
        "n_val": int(len(R_val)),
        "n_test": int(len(R_test)),
        "n_params": int(n_params),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_f_mae": best_val_f_mae,
        "best_val_e_mae": best_val_e_mae_at_best,
        "test_f_mae": test_f_mae,
        "test_e_mae": test_e_mae,
        "total_time_s": total_time,
        "device": str(device),
        "val_history": val_history,
        "checkpoint_path": checkpoint_path,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    log(f"result written to {args.out_json}")


if __name__ == "__main__":
    main()
