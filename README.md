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
the scalar–vector channel structure. The headline comparison is reported as
a multi-seed replication rather than a single training run: retraining both
the reproduced PaiNN baseline and the combined final configuration from
three seeds never used during architecture search gives a mean ± population
standard deviation test force MAE of 0.4754(47) falling to 0.3086(48), a
**35.1%** reduction, and a test energy MAE of 0.1466(145) falling to
0.1188(16), a **19.0%** reduction. For reference, the single search seed
(42) used throughout architecture search gives a force-MAE reduction of
36.6% (from 0.4755 to 0.3013) and an energy-MAE reduction of 38.6% (from
0.1973 to 0.1211), consistent with, but not identical to, the multi-seed
means. The repository also contains the cross-molecule (ethanol, aspirin)
generalization experiments, the stepwise ablation and rolled-back negative
results, the parameter/MACs cost analysis, and the O(3)-equivariance
verification used in the paper.

## Result overview

**Table 1 — Main comparison on MD17 uracil: multi-seed replication (headline result).**
Three seeds (1001/1002/1003) never used at any stage of architecture search,
every architectural and optimisation setting identical to the single-seed
search protocol. Mean ± population standard deviation over the three seeds;
these means are the paper's headline comparison.

| Configuration | Force MAE (mean ± sd) | Energy MAE (mean ± sd) |
|---|---|---|
| PaiNN baseline (reproduced) | 0.4754 ± 0.0047 | 0.1466 ± 0.0145 |
| **This work (final)** | **0.3086 ± 0.0048** | **0.1188 ± 0.0016** |
| **Δ** | **−35.1%** | **−19.0%** |

For reference, and because it is the number used throughout the
architecture-search campaign below, the single search-seed (42) comparison
gives a test force MAE of 0.4755 falling to 0.3013 (−36.6%) and a test
energy MAE of 0.1973 falling to 0.1211 (−38.6%). Both single-seed values
fall inside the range spanned by the corresponding three-seed distribution
above; this single-seed reference is retained for the ablation table (Table
3) and the stepwise search record because that is the exploratory search
record as it was actually run, not because it is the paper's headline
result.

**Table 2 — Zero-tuning cross-molecule generalization (seed 42, code/hyperparameters unchanged from uracil).**

| Molecule (N atoms) | Configuration | Force MAE | Energy MAE | Δ Force |
|---|---|---|---|---|
| Uracil (12) | baseline / final | 0.4755 / **0.3013** | 0.1973 / **0.1211** | −36.6% |
| Ethanol (9) | baseline / final | 0.4861 / **0.3494** | 0.1138 / **0.0815** | −28.1% |
| Aspirin (21) | baseline / final | 0.8202 / **0.5550** | 0.4372 / **0.2525** | −32.3% |

**Table 3 — Stepwise ablation (development-log record; each row cumulative; A–D are rolled-back negative results).**

| Step | Modification | Force MAE | Δ vs. prev. | Energy MAE | Outcome |
|---|---|---|---|---|---|
| 0 | PaiNN baseline (reproduced) | 0.4755 | — | 0.1973 | reference |
| 1 | + Hybrid Gaussian–Bessel RBF | 0.4082 | −14.2% | 0.1382 | accepted |
| 2 | + Three-body inner-product gate | 0.3430 | −16.0% | 0.1317 | accepted |
| 3 | + Bidirectional center–neighbor | 0.3375 | −1.6% | 0.1216 | accepted |
| 4 | + Signed tanh gate activation | 0.3021 | −10.5% | 0.1182 | accepted |
| — | Fresh retrain of step-4 architecture (single-seed reference above) | 0.3013 | — | 0.1211 | final |
| A | Multi-scale JK pooling + vector readout | 0.3809 | +11.1% | 0.1268 | rolled back |
| B | Deeper 2-layer MLP filter (40–80–384) | 0.3403 | +0.8% | 0.1204 | rolled back |
| C | LayerNorm in interaction and mixing | 0.3392 | +0.5% | 0.1274 | rolled back |
| D | Bessel basis widened to n_bessel=32 | 0.3984 | +31.9% | 0.1236 | rolled back |

**Table 4 — Parameter and inference cost (batch of 32 uracil frames, CPU).**

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
│       │                                    # counting (produces Table 4)
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
#    used to produce the reported numbers. data/splits/checksums.sha256 lists
#    bare filenames, so it must be run with that directory as cwd.
sha256sum -c reproducibility_artifact/checksums.sha256
(cd data/splits && sha256sum -c checksums.sha256)

# 2. Activate the training environment (see above).
conda activate painn_repro

# 3. Headline comparison (Table 1, multi-seed, uracil): see
#    scripts/painn_generalization/run_experiment.py, invoked once per
#    (architecture, seed) combination exactly as documented in
#    reproducibility_artifact/README_REPRODUCE.md §5. For the single
#    search-seed (42) reference number instead, run the protected,
#    unmodified evaluation script against the final architecture; see
#    reproducibility_artifact/README_REPRODUCE.md section 4 for the exact
#    invocation of src/eval_benchmark.py against src/final_architecture/.
#    Expected: test force MAE ~ 0.3013, test energy MAE ~ 0.1211.

# 4. Cross-molecule generalization (Table 2): see
#    scripts/painn_generalization/run_experiment.py, invoked once per
#    (architecture, molecule) combination at seed 42 exactly as documented
#    in reproducibility_artifact/README_REPRODUCE.md §5.

# 5. O(3) equivariance check (double and single precision, including
#    improper rotations). The script expects a directory whose `src/`
#    subdirectory contains `schnetpack/`; point it at the final architecture
#    via a one-time symlink (this repo keeps `schnetpack/` directly under
#    `src/final_architecture/`, not nested one level deeper):
mkdir -p /tmp/painn_src_final && ln -sf "$(pwd)/src/final_architecture" /tmp/painn_src_final/src
python scripts/verify_painn_physics.py --src /tmp/painn_src_final \
  --data data/splits/md17_uracil_split.npz --label final

# 6. Forward-hook MACs / parameter cost accounting (Table 4). This script
#    checks out five specific historical commits by SHA via `git archive`
#    against the original AutoSOTA project repository's git history (not
#    this one, which was exported as a single-commit snapshot without that
#    intermediate-step commit history), so it cannot be re-run as-is from
#    this repository alone. See `scripts/painn_sci_scratch/measure_cost.py`
#    for the exact forward-hook MAC-counting methodology (same approach
#    applies to `src/final_architecture/` and `src/baseline_architecture/`
#    individually); the already-committed logs reflect the real measured
#    numbers reported in Table 4.

# 7. Regenerate Figure 4 and its independent atom-mapping verification.
#    These scripts, like measure_cost.py above, hardcode absolute input/
#    output paths from the original AutoSOTA run
#    (/home/ymujdb/my_sota_project/...); run them unmodified only on that
#    original machine/checkout, or edit the path constants at the top of
#    each script before running elsewhere. See
#    reproducibility_artifact/README_REPRODUCE.md §6 for the full data
#    provenance and the exact expected values.
/home/ymujdb/my_sota_project/.autosota/venv/bin/python scripts/painn_sci_scratch/verify_uracil_atom_mapping.py
/home/ymujdb/my_sota_project/.autosota/venv/bin/python scripts/painn_sci_scratch/plot_fig4_error_analysis.py
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

No trained model checkpoints (`.pt`/`.pth` state dicts) were retained for any
run reported in the paper: `eval_benchmark.py` evaluates the
best-validation-epoch state in memory and never checkpoints to disk, so only
scalar logs and the one-time exported per-atom prediction arrays in
`predictions_and_logs/` are available for the runs that produced Figures
1(D) and 4. `scripts/painn_generalization/run_experiment.py` additionally
supports an opt-in `--save-checkpoint <path>` flag that writes the
best-validation-epoch state dict to disk purely as additive instrumentation,
after the fixed 80-epoch training/evaluation protocol and its test-set
metrics have already been computed; it cannot influence `best_epoch`,
`best_val_f_mae`, `test_f_mae`, or `test_e_mae`. This flag is non-retroactive:
none of the runs that produced the numbers reported in the paper used it.

## License

MIT, inherited from the schnetpack codebase this project builds on. See
[`LICENSE`](LICENSE).
