"""Measure real parameter counts / MACs / latency per optimisation iteration.

For each real git commit in optimized_code/painn, extract that commit's source
tree read-only with `git archive` into /tmp, build the exact model
configuration eval_benchmark.py uses, and measure:

  * total trainable parameters (real state_dict sizes)
  * forward-pass multiply-accumulate count, obtained analytically from the
    *actual* tensor shapes observed by forward hooks on the real model
  * wall-clock forward+backward latency

Nothing in the repo is modified; all writes go to /tmp and papers/painn_sci/data.
"""
import os
import subprocess
import shutil
import sys
import json
import statistics

REPO = "/home/ymujdb/my_sota_project/optimized_code/painn"
SCRATCH = "/tmp/painn_cost_src"
PY = sys.executable

COMMITS = [
    ("iter-0 baseline", "f4683fcc"),
    ("iter-1 +Bessel RBF", "62f3adcf"),
    ("iter-2 +3-body gate", "464ffbef"),
    ("iter-4 +bidirectional", "1d9f0f0e"),
    ("iter-7 tanh gate (final)", "c5c11994"),
]

PROBE = r'''
import os, sys, time, json, statistics
SRC = sys.argv[1]
sys.path.insert(0, os.path.join(SRC, "src"))
import torch, numpy as np
import schnetpack as spk
import schnetpack.properties as properties

torch.manual_seed(42); np.random.seed(42)
dev = torch.device("cpu")

data = np.load("/home/ymujdb/my_sota_project/optimized_code/painn/data/md17_uracil_split.npz")
Z_single = torch.tensor(data["z"], dtype=torch.long)
R = torch.tensor(data["R_train"][:32], dtype=torch.float32)
E = torch.tensor(data["E_train"][:32], dtype=torch.float32)
F = torch.tensor(data["F_train"][:32], dtype=torch.float32)

cutoff, nab, nint = 5.0, 128, 3
painn = spk.representation.PaiNN(
    n_atom_basis=nab, n_interactions=nint,
    radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
    cutoff_fn=spk.nn.CosineCutoff(cutoff),
)
model = spk.model.NeuralNetworkPotential(
    representation=painn,
    input_modules=[spk.atomistic.PairwiseDistances()],
    output_modules=[
        spk.atomistic.Atomwise(n_in=nab, output_key=properties.energy),
        spk.atomistic.Forces(energy_key=properties.energy, force_key=properties.forces),
    ],
    do_postprocessing=False,
).to(dev)

n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_param_rep = sum(p.numel() for p in painn.parameters() if p.requires_grad)
n_basis = painn.radial_basis.n_rbf

def make_batch(R_b):
    B, N, _ = R_b.shape
    R_b = R_b.reshape(B * N, 3)
    Z_b = Z_single.unsqueeze(0).expand(B, N).reshape(B * N)
    idx_m = torch.arange(B).unsqueeze(1).expand(B, N).reshape(B * N)
    diff = R_b.reshape(B, N, 1, 3) - R_b.reshape(B, 1, N, 3)
    dist = torch.norm(diff, dim=-1)
    mask = (dist > 0.0) & (dist <= cutoff)
    b, i, j = torch.where(mask)
    return {
        properties.Z: Z_b, properties.R: R_b, properties.idx_m: idx_m,
        properties.idx_i: b * N + i, properties.idx_j: b * N + j,
        properties.offsets: torch.zeros((len(b), 3), dtype=torch.float32),
        properties.n_atoms: torch.full((B,), N, dtype=torch.long),
    }

inputs = make_batch(R)
N_atoms = R.shape[1]
n_pairs = int(inputs[properties.idx_i].shape[0])

# ---- analytic forward MAC, from real observed tensor shapes ----
mac = {"linear": 0, "n_linear_calls": 0, "rbf_elements": 0}
handles = []

def lin_hook(mod, inp, out):
    # MAC for a Linear/Dense applied to a token tensor: tokens * in * out
    x = inp[0]
    tokens = x.numel() // x.shape[-1] if x.dim() >= 1 else 1
    mac["linear"] += tokens * mod.in_features * mod.out_features
    mac["n_linear_calls"] += 1

for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        handles.append(mod.register_forward_hook(lin_hook))

# Radial basis expansion: every pair evaluated against n_rbf centres.
orig_fwd = {}
def rbf_hook(mod, inp, out):
    mac["rbf_elements"] += out.numel()
for name, mod in model.named_modules():
    if type(mod).__name__ in ("GaussianRBF", "BesselRBF"):
        handles.append(mod.register_forward_hook(rbf_hook))

model.train()
out = model(make_batch(R))
out[properties.energy].sum().backward()
for h in handles:
    h.remove()

# ---- latency: forward + backward (PaiNN forces need autograd) ----
def fwd_bwd():
    o = model(make_batch(R))
    le = torch.nn.functional.mse_loss(o[properties.energy], E)
    lf = torch.nn.functional.mse_loss(o[properties.forces], F.reshape(-1, 3))
    loss = 0.01 * le + 0.99 * lf
    model.zero_grad(set_to_none=True)
    loss.backward()

for _ in range(2):
    fwd_bwd()
ts = []
for _ in range(7):
    t = time.perf_counter(); fwd_bwd(); ts.append((time.perf_counter() - t) * 1e3)

print(json.dumps({
    "n_params_total": n_param,
    "n_params_representation": n_param_rep,
    "n_rbf": int(n_basis),
    "n_pairs": n_pairs,
    "linear_macs": int(mac["linear"]),
    "n_linear_calls": int(mac["n_linear_calls"]),
    "rbf_elements": int(mac["rbf_elements"]),
    "forward_macs_total": int(mac["linear"] + mac["rbf_elements"]),
    "fwd_bwd_ms_median": round(statistics.median(ts), 2),
}))
'''


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    probe_path = os.path.join(SCRATCH, "_probe.py")
    with open(probe_path, "w") as f:
        f.write(PROBE)

    results = {}
    for label, sha in COMMITS:
        tree = os.path.join(SCRATCH, sha)
        if os.path.isdir(tree):
            shutil.rmtree(tree)
        os.makedirs(tree, exist_ok=True)
        p1 = subprocess.run(["git", "archive", sha], cwd=REPO,
                            stdout=subprocess.PIPE, check=True)
        subprocess.run(["tar", "-x", "-C", tree], input=p1.stdout, check=True)
        out = subprocess.run([PY, probe_path, tree], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        if out.returncode != 0:
            print(f"[FAIL] {label} ({sha})\n{out.stderr[-1500:]}", file=sys.stderr)
            continue
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        rec["commit"] = sha
        rec["label"] = label
        results[sha] = rec
        print(f"[ok] {label:26s} {sha}  params={rec['n_params_total']:>9,}  "
              f"rbf={rec['n_rbf']:>3}  MACs={rec['forward_macs_total']:>11,}  "
              f"fwd+bwd={rec['fwd_bwd_ms_median']:>6.2f} ms")

    with open("/home/ymujdb/my_sota_project/papers/painn_sci/data/cost_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten -> papers/painn_sci/data/cost_analysis.json")


if __name__ == "__main__":
    main()
