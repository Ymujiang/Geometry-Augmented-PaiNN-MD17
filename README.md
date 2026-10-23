# Geometry-Augmented Equivariant Message Passing for Molecular Force Fields

Code, data splits, trained-run logs, and figure-generation scripts for the paper:

> **Geometry-Augmented Equivariant Message Passing: Signed Gating, Hybrid Radial Bases and Bidirectional Coupling for Molecular Force Fields**

Equivariant message-passing neural networks define the state of the art for
machine-learned interatomic potentials, but the Cartesian equivariant
architecture PaiNN caps its own expressiveness by conditioning each
interaction filter on interatomic distance alone. This repository contains
a faithful PaiNN reimplementation on the MD17 uracil benchmark together with
four architectural modifications — a hybrid Gaussian–Bessel radial basis, an
invariant inner-product geometric gate, a bidirectional center–neighbor
message, and a signed `tanh` gate activation — that remain strictly within
the scalar–vector channel structure and reduce test force MAE on uracil from
0.4755 to 0.3013 kcal/mol/Å (−36.6%) and energy MAE from 0.1973 to 0.1211
kcal/mol (−38.6%). The repository also contains the supplementary multi-seed
(uracil, 3 seeds) and cross-molecule (ethanol, aspirin) generalization
experiments, the stepwise ablation and rolled-back negative results, the
parameter/MACs cost analysis, and the O(3)-equivariance verification used in
the paper.

## Result overview

**Table 1 — Headline comparison (MD17 uracil, seed 42, 80-epoch protocol).**

| Configuration | Force MAE (kcal/mol/Å) | Energy MAE (kcal/mol) | Δ Force | Δ Energy | Params |
|---|---|---|---|---|---|
| PaiNN baseline (reproduced) | 0.4755 | 0.1973 | — | — | 589,057 |
| **This work (final)** | **0.3013** | **0.1211** | **−36.6%** | **−38.6%** | 1,104,769 |

**Table 2 — Multi-seed replication (uracil, seeds 1001/1002/1003, never used during architecture search).**

| Configuration | Force MAE (mean ± sd) | Energy MAE (mean ± sd) |
|---|---|---|
| PaiNN baseline (reproduced) | 0.4754 ± 0.0047 | 0.1466 ± 0.0145 |
| This work (final) | 0.3086 ± 0.0048 | 0.1188 ± 0.0016 |

**Table 3 — Zero-tuning cross-molecule generalization (seed 42, code/hyperparameters unchanged from uracil).**

| Molecule (N atoms) | Configuration | Force MAE | Energy MAE | Δ Force |
|---|---|---|---|---|
| Uracil (12) | baseline / final | 0.4755 / **0.3013** | 0.1973 / **0.1211** | −36.6% |
| Ethanol (9) | baseline / final | 0.4861 / **0.3494** | 0.1138 / **0.0815** | −28.1% |
| Aspirin (21) | baseline / final | 0.8202 / **0.5550** | 0.4372 / **0.2525** | −32.3% |

**Table 4 — Stepwise ablation (each row cumulative; A–D are rolled-back negative results).**

| Step | Modification | Force MAE | Δ vs. prev. | Energy MAE | Outcome |
|---|---|---|---|---|---|
| 0 | PaiNN baseline (reproduced) | 0.4755 | — | 0.1973 | reference |
| 1 | + Hybrid Gaussian–Bessel RBF | 0.4082 | −14.2% | 0.1382 | accepted |
| 2 | + Three-body inner-product gate | 0.3430 | −16.0% | 0.1317 | accepted |
| 3 | + Bidirectional center–neighbor | 0.3375 | −1.6% | 0.1216 | accepted |
| 4 | + Signed tanh gate activation | 0.3021 | −10.5% | 0.1182 | accepted |
| — | Fresh retrain of step-4 architecture (headline result) | 0.3013 | — | 0.1211 | final |
| A | Multi-scale JK pooling + vector readout | 0.3809 | +11.1% | 0.1268 | rolled back |
| B | Deeper 2-layer MLP filter (40–80–384) | 0.3403 | +0.8% | 0.1204 | rolled back |
| C | LayerNorm in interaction and mixing | 0.3392 | +0.5% | 0.1274 | rolled back |
| D | Bessel basis widened to n_bessel=32 | 0.3984 | +31.9% | 0.1236 | rolled back |

**Table 5 — Parameter and inference cost (batch of 32 uracil frames, CPU).**

| Configuration | Parameters | Forward MACs | Latency (ms) | MACs/param |
|---|---|---|---|---|
| Step 0: baseline | 589,057 | 381,385,264 | 79.21 | 647 |
| Step 1: + hybrid RBF | 612,097 | 476,484,704 | 77.65 | 779 |
| Step 2: + geometric gate | 1,055,617 | 2,300,810,336 | 137.73 | 2179 |
| Step 3: + bidirectional | 1,104,769 | 3,238,827,104 | 150.50 | 2932 |
| Step 4: + signed tanh gate | 1,104,769 | 3,238,827,104 | 159.16 | 2932 |
| **Ratio final / baseline** | **1.88×** | **8.49×** | **2.01×** | **4.53×** |

## Directory structure

```
.
├── reproducibility_artifact/   # Self-contained artifact matching the paper's
│                                # "Data and code availability" section verbatim:
│                                # painn.py, final_patch.diff, pinned requirement
│                                # lockfiles, checksums.sha256, README_REPRODUCE.md
├── src/
│   ├── final_architecture/     # Full schnetpack-based source tree, HEAD = tag
│   │                            # _best (commit c5c11994): hybrid RBF + geometric
│   │                            # gate + bidirectional message + signed tanh gate
│   ├── baseline_architecture/  # Full source tree at tag _baseline (faithful
│   │                            # PaiNN reimplementation, pre-modification)
│   ├── eval_benchmark.py       # Protected, unmodified training/evaluation script
│   └── final_patch.diff        # Unified diff, baseline -> final architecture
├── scripts/
│   ├── verify_painn_physics.py            # O(3) equivariance check (double AND
│   │                                        # single precision, incl. improper
│   │                                        # rotations / reflections)
│   └── painn_sci_scratch/
│       ├── verify_uracil_atom_mapping.py  # 12x12 covalent-bond-length functional
│       │                                    # group auto-derivation for uracil
│       │                                    # (no hard-coded chemical labels)
│       ├── measure_cost.py                # Forward-hook based parameter/MACs
│       │                                    # counting (produces Table 5)
│       ├── plot_fig1_rbf_gate.py          # Figure 1 (RBF + gate mechanism)
│       ├── plot_fig2_fig3.py              # Figures 2-3 (convergence, ablation,
│       │                                    # negative results)
│       ├── plot_fig4_error_analysis.py    # Figure 4 (uracil error decomposition)
│       ├── extract_rbf_gate_mechanism.py  # Mechanism data extraction for Fig. 1
│       ├── train_and_extract.py           # Training + per-atom prediction export
│       └── parse_painn_log.py             # Log parsing utility
│   └── painn_generalization/
│       └── run_experiment.py              # Multi-seed / cross-molecule runner
├── predictions_and_logs/
│   ├── raw_predictions/         # Per-atom force/energy prediction arrays
│   │                             # (predictions_final.npz, predictions_sigmoid.npz)
│   │                             # and gate-activation / training-curve data
│   │                             # underlying Figures 1(D) and 4
│   ├── training_logs/           # Main experiment logs (painn_optimizer_detail.log,
│   │                             # painn_sota_summary.log) and the Fig.1/Fig.4
│   │                             # source training logs (train_final.log,
│   │                             # train_sigmoid.log)
│   └── multiseed_crossmol_logs/ # 3-seed (1001/1002/1003) uracil reproduction
│                                 # results/logs and ethanol/aspirin cross-molecule
│                                 # generalization results/logs (Tables 2-3)
├── data/
│   └── splits/                  # Uracil/Ethanol/Aspirin 950/50/1000 split .npz
│                                 # files, each checksum-verified against
│                                 # reproducibility_artifact/checksums.sha256
├── figures/                      # Figures 1-4 (PDF + PNG, 300 DPI+)
└── LICENSE                       # MIT (inherited from the schnetpack codebase
                                   # this project is built on)
```

## Environment

Two environments are used, pinned in `reproducibility_artifact/`:

- **Training/evaluation** (`requirements_lock_training_env.txt`): Python
  3.10.21, PyTorch 2.14.1+cu130. The supplementary multi-seed, cross-molecule
  and gate-comparison runs hardcode `torch.device("cpu")` and were executed
  CPU-only. The original ten-step architecture-search campaign did not
  hardcode a device and was written to use a GPU automatically via
  `torch.cuda.is_available()` when present; we cannot now determine with
  certainty whether a GPU was used for some or all of those earlier runs
  (see `reproducibility_artifact/README_REPRODUCE.md` for the full
  disclosure). This does not affect the reported accuracy comparisons, since
  relative force and energy MAE do not depend on the compute device.
- **Plotting/verification** (`requirements_lock.txt`): a lighter
  matplotlib/numpy-based environment used only for figure generation and the
  atom-mapping / equivariance verification scripts, never for training.

```bash
conda create -n painn_repro python=3.10
conda activate painn_repro
pip install -r reproducibility_artifact/requirements_lock_training_env.txt
```

## One-command reproduction

Run from the repository root. All relative paths below assume this layout.

```bash
# 1. Verify every protected input and data split is byte-identical to what was
#    used to produce the reported numbers.
sha256sum -c reproducibility_artifact/checksums.sha256 --ignore-missing
sha256sum -c data/splits/checksums.sha256

# 2. Activate the training environment (see above).
conda activate painn_repro

# 3. Headline comparison (Table 1, uracil, seed 42): run the protected,
#    unmodified evaluation script against the final architecture. See
#    reproducibility_artifact/README_REPRODUCE.md section 4 for the exact
#    invocation of src/eval_benchmark.py against src/final_architecture/.
#    Expected: test force MAE ~ 0.3013, test energy MAE ~ 0.1211.

# 4. Multi-seed replication (Table 2) and cross-molecule generalization
#    (Table 3): see scripts/painn_generalization/run_experiment.py, invoked
#    once per (architecture, molecule, seed) combination exactly as
#    documented in reproducibility_artifact/README_REPRODUCE.md §5.

# 5. O(3) equivariance check (double and single precision, including
#    improper rotations):
python scripts/verify_painn_physics.py

# 6. Forward-hook MACs / parameter cost accounting (Table 5):
python scripts/painn_sci_scratch/measure_cost.py

# 7. Regenerate Figure 4 and its independent atom-mapping verification:
python scripts/painn_sci_scratch/verify_uracil_atom_mapping.py
python scripts/painn_sci_scratch/plot_fig4_error_analysis.py
```

For the complete, byte-exact reproduction protocol — including the precise
optimizer/scheduler/loss settings, the full multi-seed and cross-molecule
command set, and what is deliberately **not** redistributed (raw MD17
datasets, intermediate checkpoints between `_baseline` and `_best`) — see
[`reproducibility_artifact/README_REPRODUCE.md`](reproducibility_artifact/README_REPRODUCE.md).

## Data

Three MD17 splits are used, each a 950/50/1000 (train/val/test) split with no
geometry overlap between subsets, verified by exact-coordinate hashing:
Uracil (12 atoms), Ethanol (9 atoms), Aspirin (21 atoms). Only the fixed
split indices and the corresponding sliced coordinates/energies/forces are
redistributed in `data/splits/`; the full MD17/revised-MD17 raw datasets are
publicly available from their original source (Chmiela et al.; Christensen &
von Lilienfeld) and are not redistributed here.

No trained model checkpoints (`.pt`/`.pth` state dicts) are retained for any
run: `eval_benchmark.py` evaluates the best-validation-epoch state in memory
and never checkpoints to disk, so only scalar logs and the one-time exported
per-atom prediction arrays in `predictions_and_logs/` are available for the
runs that produced Figures 1(D) and 4.

## License

MIT, inherited from the schnetpack codebase this project builds on. See
[`LICENSE`](LICENSE).
