import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local editable src/schnetpack is always imported first
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

import schnetpack as spk
import schnetpack.properties as properties


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(R_batch, Z_single, cutoff: float, device: torch.device):
    B, N_atoms, _ = R_batch.shape
    R_b = R_batch.reshape(B * N_atoms, 3)
    Z_b = Z_single.unsqueeze(0).expand(B, N_atoms).reshape(B * N_atoms)
    idx_m = torch.arange(B, device=device).unsqueeze(1).expand(B, N_atoms).reshape(B * N_atoms)

    diff = R_batch.unsqueeze(2) - R_batch.unsqueeze(1)  # (B, N, N, 3)
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


def evaluate(model, R_data, E_data, F_data, Z_single, cutoff, batch_size, device):
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

        inputs = make_batch(R_b, Z_single, cutoff, device)
        out = model(inputs)

        pred_e = out[properties.energy].detach()
        pred_f = out[properties.forces].detach().reshape(B, N_atoms, 3)

        total_e_abs += torch.sum(torch.abs(pred_e - E_b)).item()
        total_e_count += B
        total_f_abs += torch.sum(torch.abs(pred_f - F_b)).item()
        total_f_count += F_b.numel()

    return total_f_abs / total_f_count, total_e_abs / total_e_count


def main():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_path = os.path.join(REPO_ROOT, 'data', 'md17_uracil_split.npz')
    data = np.load(data_path)

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

    # Remove training energy mean offset (standard SchNetPack RemoveOffsets)
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

    best_val_f_mae = float('inf')
    best_state = None
    t0 = time.time()

    num_train = len(R_train)
    N_atoms = R_train.shape[1]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train, device=device)
        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size]
            B = len(idx)
            R_b = R_train[idx]
            E_b = E_train[idx]
            F_b = F_train[idx].reshape(B * N_atoms, 3)

            inputs = make_batch(R_b, Z_single, cutoff, device)
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
            val_f_mae, val_e_mae = evaluate(model, R_val, E_val, F_val, Z_single, cutoff, 50, device)
            if val_f_mae < best_val_f_mae:
                best_val_f_mae = val_f_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f'Epoch {epoch:03d}/{epochs:03d} | Val_Force_MAE: {val_f_mae:.4f} | Val_Energy_MAE: {val_e_mae:.4f} | Time: {time.time()-t0:.1f}s')

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_f_mae, test_e_mae = evaluate(model, R_test, E_test, F_test, Z_single, cutoff, 100, device)
    print(f'Test_Energy_MAE: {test_e_mae:.4f}')
    print(f'Final_Test_MAE: {test_f_mae:.4f}')


if __name__ == '__main__':
    main()
