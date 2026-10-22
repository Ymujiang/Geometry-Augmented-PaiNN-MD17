# Reproducibility Artifact — Geometry-Augmented PaiNN on MD17

This directory packages everything needed to reproduce the numerical results
reported in `main.tex`, independent of the rest of the project repository.
Every number cited below is read directly from the committed training logs
and result JSON files; nothing in this artifact or in the paper is simulated
or hand-entered.

## 1. Contents of this directory

| File | Description |
|---|---|
| `painn.py` | The exact, final `PaiNNInteraction`/representation module used for every "this work (final)" result in the paper. Copied verbatim from `optimized_code/painn/src/schnetpack/representation/painn.py` at git commit `c5c11994` (tag `_best`, "iter-7: Tanh-based Geometric Gate (replaces sigmoid) [success]"). |
| `final_patch.diff` | The full unified diff of all architectural changes relative to the faithful PaiNN reimplementation (`_baseline` tag), covering: (1) hybrid Gaussian+Bessel radial basis, (2) invariant inner-product geometric gate, (3) bidirectional center–neighbor message, (4) signed tanh gate activation. Copied verbatim from `optimized_code/painn/final_patch.diff`. |
| `requirements_lock.txt` | `pip freeze` of `/home/ymujdb/my_sota_project/.autosota/venv` (the AutoSOTA orchestration/plotting environment used to generate all figures and the atom-mapping verification script in this project). |
| `requirements_lock_training_env.txt` | `pip freeze` of the actual conda environment used to run every training/evaluation job reported in the paper: `cgcnn_env` (`/home/ymujdb/miniconda3/envs/cgcnn_env`). This is the environment to use if you want to reproduce the reported MAE numbers, not `requirements_lock.txt`. |
| `checksums.sha256` | SHA-256 checksums of the two protected inputs (uracil split, `eval_benchmark.py`), the two cross-molecule splits (ethanol, aspirin), the final patch, the final and baseline `painn.py` source files, and the generalization-experiment runner script. |
| `README_REPRODUCE.md` | This file. |

## 2. Environment

- OS: Ubuntu 22.04.5 LTS (observed on WSL2, kernel 6.6.87.2-microsoft-standard-WSL2; the code has no WSL-specific dependency).
- Python: 3.10.21 (training environment, `cgcnn_env`).
- PyTorch: 2.14.1+cu130, **run CPU-only** (`torch.cuda.is_available() == False` in the environment used for every reported run; no GPU was used or available). This is disclosed in `main.tex` §Experimental protocol and does not affect correctness, since every architecture compared is trained and evaluated on the same hardware.
- Full training-environment dependency list: `requirements_lock_training_env.txt` (112 packages, includes `schnetpack`-derived code, `ase`, `pytorch-lightning`, etc., frozen from the exact environment that produced the numbers in the paper).
- Plotting/verification-environment dependency list: `requirements_lock.txt` (26 packages, matplotlib/numpy-based, used only for figure generation and the atom-mapping verification script, never for training).

To recreate the training environment from scratch (approximate; exact pinned
versions are in `requirements_lock_training_env.txt`):
```bash
conda create -n cgcnn_env python=3.10
conda activate cgcnn_env
pip install -r requirements_lock_training_env.txt
```

## 3. Data provenance and integrity

Three MD17 splits are used, each a 950/50/1000 (train/val/test) split with
no geometry overlap between the three subsets (checked by exact-coordinate
hashing before use):

| Molecule | Split file (relative to project root) | SHA-256 |
|---|---|---|
| Uracil (12 atoms, protected file, never modified) | `optimized_code/painn/data/md17_uracil_split.npz` | see `checksums.sha256` |
| Ethanol (9 atoms) | `data_cache/generalization_splits/md17_ethanol_split.npz` | see `checksums.sha256` |
| Aspirin (21 atoms) | `data_cache/generalization_splits/md17_aspirin_split.npz` | see `checksums.sha256` |

`optimized_code/painn/eval_benchmark.py` is the protected reference training/
evaluation script; it was never modified at any point in this project, and its
checksum here matches the value recorded independently in
`optimized_code/painn/.autosota_protected_hashes.json`, which the AutoSOTA
tooling writes once at the start of optimization and checks on every run.

Before using any of the files in this directory, verify them against
`checksums.sha256`:
```bash
cd /home/ymujdb/my_sota_project
sha256sum -c papers/painn_sci/reproducibility_artifact/checksums.sha256
```
(Run from the project root `/home/ymujdb/my_sota_project` so the relative
paths in the checksum file resolve.)

## 4. One-click reproduction of the headline comparison (Table 1 / uracil, seed 42)

```bash
# 1. Verify the two protected inputs and all split files are byte-identical
#    to what was used to produce the reported numbers.
cd /home/ymujdb/my_sota_project
sha256sum -c papers/painn_sci/reproducibility_artifact/checksums.sha256

# 2. Install the exact training environment (or activate an existing
#    equivalent one, e.g. the project's cgcnn_env).
conda activate cgcnn_env   # or: pip install -r reproducibility_artifact/requirements_lock_training_env.txt

# 3. Run the protected, unmodified evaluation script. This trains for the
#    full 80-epoch protocol (set_seed(42), AdamW lr=1e-3 wd=0,
#    CosineAnnealingLR T_max=80 eta_min=1e-5, loss = 0.01*MSE_E + 0.99*MSE_F,
#    grad_clip=10.0) and reports the test-set force/energy MAE of the
#    best-validation-epoch checkpoint.
cd optimized_code/painn
python eval_benchmark.py
# Expected (final / improved architecture, HEAD = tag _best = commit c5c11994):
#   test force MAE ≈ 0.3013   test energy MAE ≈ 0.1211
```

To reproduce the **reproduced-baseline** numbers instead, point the same
protocol at the `_baseline`-tagged source tree
(`scripts/painn_generalization/baseline_src/`), e.g. via
`scripts/painn_generalization/run_experiment.py --src-path
scripts/painn_generalization/baseline_src --molecule uracil --seed 42`.
Expected: test force MAE ≈ 0.4755, test energy MAE ≈ 0.1973.

## 5. One-click reproduction of the supplementary multi-seed and cross-molecule results (§3.2)

```bash
cd /home/ymujdb/my_sota_project
conda activate cgcnn_env

# Multi-seed replication on uracil (Table 2): 3 seeds never used during
# architecture search, for both the reproduced baseline and the final
# architecture, all other settings identical to the protocol above.
for seed in 1001 1002 1003; do
  for variant in baseline improved; do
    src="optimized_code/painn/src"
    [ "$variant" = "baseline" ] && src="scripts/painn_generalization/baseline_src"
    python scripts/painn_generalization/run_experiment.py \
      --src-path "$src" --molecule uracil --seed "$seed" \
      --out scripts/painn_generalization/results/uracil_${variant}_seed${seed}.json
  done
done

# Cross-molecule zero-tuning transfer (Table 3): same seed (42) used
# throughout architecture search, applied unchanged to ethanol and aspirin.
for molecule in ethanol aspirin; do
  for variant in baseline improved; do
    src="optimized_code/painn/src"
    [ "$variant" = "baseline" ] && src="scripts/painn_generalization/baseline_src"
    python scripts/painn_generalization/run_experiment.py \
      --src-path "$src" --molecule "$molecule" --seed 42 \
      --out scripts/painn_generalization/results/${molecule}_${variant}_seed42.json
  done
done
```

Each run reproduces one row of Table 2 or Table 3 in `main.tex`. No
architecture search, hyperparameter tuning, or epoch-budget change was
performed for any seed or molecule beyond the fixed protocol above; every
run uses the same 80-epoch budget as the headline uracil comparison.

## 6. Reproducing Figure 4 (error decomposition)

Figure 4 (`figures/fig4_uracil_error_distribution.pdf`) is built from a single
real forward-inference pass of a trained final-architecture model over the
full uracil test set (1000 configurations), recorded in
`papers/painn_sci/data/raw/predictions_final.npz` (per-atom predicted and
reference forces/energies) and `papers/painn_sci/data/raw/train_final.log`
(the training run that produced it: `Final_Test_MAE: 0.3024`,
`Test_Energy_MAE: 0.1185`). The per-atom-index assignment to element and
chemical microenvironment used in that figure is not hard-coded from prior
knowledge of uracil's structure: it is derived programmatically from the raw
atomic numbers and first test-frame coordinates in
`optimized_code/painn/data/md17_uracil_split.npz` using covalent bond-length
thresholds (see `scripts/painn_sci_scratch/verify_uracil_atom_mapping.py`,
which independently reconstructs the molecular graph from geometry alone and
confirms it matches the grouping used for plotting).

To regenerate the figure and the independent atom-mapping verification:
```bash
cd /home/ymujdb/my_sota_project
.autosota/venv/bin/python scripts/painn_sci_scratch/verify_uracil_atom_mapping.py
.autosota/venv/bin/python scripts/painn_sci_scratch/plot_fig4_error_analysis.py
```

No trained model checkpoint (`.pt`/`.pth` state dict) for the uracil runs was
retained on disk after training completed; only the scalar logs
(`train_final.log`) and the one-time exported per-atom prediction array
(`predictions_final.npz`) were kept, consistent with the project's
`optimized_code/painn/eval_benchmark.py` protocol, which evaluates the
best-validation-epoch in-memory state and does not checkpoint to disk.
Re-running Figure 4 from the raw `.npz` therefore reproduces the figure
exactly from already-recorded real predictions; regenerating the predictions
themselves from scratch requires re-running the training protocol in
§4 above and re-exporting per-atom predictions with
`scripts/painn_sci_scratch/train_and_extract.py --variant final`.

## 7. What is NOT included in this artifact

- Full MD17 raw datasets (`rmd17_uracil.npz` etc.) are not redistributed here;
  they are the standard, publicly available MD17/revised-MD17 dataset
  (Chmiela et al., Christensen & von Lilienfeld) and must be obtained from
  the original source. Only the fixed 950/50/1000 split indices and the
  corresponding sliced coordinates/energies/forces are included via the
  `.npz` files referenced above.
- Intermediate-architecture checkpoints (the 7 iterative steps between
  `_baseline` and `_best`) are not retained as binary weights; their
  per-step scalar metrics are recorded in `logs/` and summarized in Table 4
  of the paper. Only the final architecture's source code and patch are
  packaged here, per §1.
