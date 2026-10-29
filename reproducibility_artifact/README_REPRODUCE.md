# Reproducibility Artifact — Geometry-Augmented PaiNN on MD17

This directory packages everything needed to reproduce the numerical results
reported in the paper, independent of the rest of this repository. Every
number cited below is read directly from the committed training logs and
result JSON files in `predictions_and_logs/`; nothing in this artifact or in
the paper is simulated or hand-entered.

## 1. Contents of this directory

| File | Description |
|---|---|
| `painn.py` | The exact, final `PaiNNInteraction`/representation module used for every "this work (final)" result in the paper. Byte-identical to `src/final_architecture/schnetpack/representation/painn.py`, from git commit `c5c11994` (tag `_best`, "iter-7: Tanh-based Geometric Gate (replaces sigmoid) [success]"). |
| `final_patch.diff` | The full unified diff of all architectural changes relative to the faithful PaiNN reimplementation (`_baseline` tag), covering: (1) hybrid Gaussian+Bessel radial basis, (2) invariant inner-product geometric gate, (3) bidirectional center-neighbor message, (4) signed tanh gate activation. |
| `requirements_lock.txt` | `pip freeze` of the plotting/verification environment used to generate all figures and the atom-mapping / equivariance verification scripts in this project. |
| `requirements_lock_training_env.txt` | `pip freeze` of the actual conda environment used to run every training/evaluation job reported in the paper (`cgcnn_env`). This is the environment to use if you want to reproduce the reported MAE numbers, not `requirements_lock.txt`. |
| `checksums.sha256` | SHA-256 checksums of the protected uracil split and `src/eval_benchmark.py`, the two cross-molecule splits (ethanol, aspirin), the final patch, the final and baseline `painn.py` source files, and the generalization-experiment runner script. |
| `DATA_MANIFEST.md` | File-by-file provenance of every artifact in `predictions_and_logs/`: which script produced it, its checksum, and exactly which table/figure/number it supports. Also states explicitly that no model checkpoint (`.pt`/`.pth`) is retained for any run and why. |
| `README_REPRODUCE.md` | This file. |

## 2. Environment

- OS: Ubuntu 22.04.5 LTS (observed on WSL2, kernel 6.6.87.2-microsoft-standard-WSL2; the code has no WSL-specific dependency).
- Python: 3.10.21 (training environment, `cgcnn_env`).
- PyTorch: 2.14.1+cu130. The supplementary multi-seed, cross-molecule and
  gate-comparison runs (`run_experiment.py`, `train_and_extract.py`) hardcode
  `torch.device("cpu")` and were executed CPU-only, with no GPU present or
  used, as recorded in their logs. The original ten-step architecture-search
  campaign (`eval_benchmark.py`) did not hardcode a device and was written to
  use a GPU automatically via `torch.cuda.is_available()` when present; we
  cannot now determine with certainty whether a GPU was used for some or all
  of those earlier runs. This does not affect the reported accuracy
  comparisons, since relative force and energy MAE do not depend on the
  compute device, but absolute wall-clock timings should not be assumed
  uniform across every stage of the project (see the paper's Experimental
  protocol section for the full disclosure).
- Full training-environment dependency list: `requirements_lock_training_env.txt` (112 packages, includes `schnetpack`-derived code, `ase`, `pytorch-lightning`, etc., frozen from the exact environment that produced the numbers in the paper).
- Plotting/verification-environment dependency list: `requirements_lock.txt` (26 packages, matplotlib/numpy-based, used only for figure generation and the atom-mapping verification script, never for training).

To recreate the training environment from scratch (approximate; exact pinned
versions are in `requirements_lock_training_env.txt`):
```bash
conda create -n cgcnn_env python=3.10
conda activate cgcnn_env
pip install -r requirements_lock_training_env.txt
```

**Working-directory independence.** `scripts/painn_generalization/run_experiment.py`
and `scripts/painn_sci_scratch/train_and_extract.py` take every input path
(`--split-npz`, `--src-path`, `--log-file`, `--out-json`, `--out-dir`,
`--save-checkpoint`) as a command-line argument rather than assuming a fixed
relative location, and `run_experiment.py` records `os.path.abspath()` of its
split and source paths in the output JSON for traceability. `train_and_extract.py`
reads the uracil split from a hardcoded absolute path on the machine the
paper's experiments were run on, since that file's identity as the one
protected input is itself part of the experiment's integrity guarantee,
rather than a path relative to the invocation directory or this repository's
checkout location; see the comment at the top of `train_and_extract.py`
before re-running it in a different environment. `run_experiment.py` has no
such hardcoded path and can be pointed at the split files in `data/splits/`
of this repository directly. Both scripts were verified to run correctly
from an arbitrary current working directory; only the checksum-verification
command in §3 below is written to assume this directory's parent (the
repository root) as the working directory, and that assumption is stated
explicitly next to it.

## 3. Data provenance and integrity

Three MD17 splits are used, each a 950/50/1000 (train/val/test) split with
no geometry overlap between the three subsets (checked by exact-coordinate
hashing before use):

| Molecule | Split file (relative to repository root) | SHA-256 |
|---|---|---|
| Uracil (12 atoms, protected file, never modified) | `data/splits/md17_uracil_split.npz` | see `checksums.sha256` |
| Ethanol (9 atoms) | `data/splits/md17_ethanol_split.npz` | see `checksums.sha256` |
| Aspirin (21 atoms) | `data/splits/md17_aspirin_split.npz` | see `checksums.sha256` |

`src/eval_benchmark.py` is the protected reference training/evaluation
script; it was never modified at any point in this project, and its checksum
here matches the value recorded independently by the AutoSOTA tooling that
produced this project, which writes a protected-file hash once at the start
of optimization and checks it on every run.

Before using any of the files in this directory, verify them against
`checksums.sha256`:
```bash
cd /path/to/Geometry-Augmented-PaiNN-MD17
sha256sum -c reproducibility_artifact/checksums.sha256
```
(Run from the repository root so the relative paths in the checksum file
resolve.)

## 4. One-click reproduction of the headline comparison (Table 1 / uracil, multi-seed; the primary search seed 42 run below)

```bash
# 1. Verify the two protected inputs and all split files are byte-identical
#    to what was used to produce the reported numbers.
cd /path/to/Geometry-Augmented-PaiNN-MD17
sha256sum -c reproducibility_artifact/checksums.sha256

# 2. Install the exact training environment (or activate an existing
#    equivalent one, e.g. the project's cgcnn_env).
conda activate cgcnn_env   # or: pip install -r reproducibility_artifact/requirements_lock_training_env.txt

# 3. Run the protected, unmodified evaluation script. This trains for the
#    full 80-epoch protocol (set_seed(42), AdamW lr=1e-3 wd=0,
#    CosineAnnealingLR T_max=80 eta_min=1e-5, loss = 0.01*MSE_E + 0.99*MSE_F,
#    grad_clip=10.0) and reports the test-set force/energy MAE of the
#    best-validation-epoch checkpoint.
#
#    `src/eval_benchmark.py` resolves its inputs relative to its own file
#    location (`REPO_ROOT = os.path.dirname(os.path.abspath(__file__))`),
#    expecting `<that dir>/src/schnetpack/` and
#    `<that dir>/data/md17_uracil_split.npz` — a flat layout from a prior
#    version of this project, not this repository's current directory
#    structure. It is the protected file and was intentionally left
#    byte-for-byte unmodified (see `checksums.sha256`), so assemble its
#    expected layout in a scratch directory instead of editing it:
mkdir -p /tmp/painn_eval/src /tmp/painn_eval/data
cp -r src/final_architecture/schnetpack /tmp/painn_eval/src/
ln -s "$(pwd)/data/splits/md17_uracil_split.npz" /tmp/painn_eval/data/md17_uracil_split.npz
cp src/eval_benchmark.py /tmp/painn_eval/eval_benchmark.py
cd /tmp/painn_eval
python eval_benchmark.py
# Expected (final / improved architecture, HEAD = tag _best = commit c5c11994):
#   test force MAE ≈ 0.3013   test energy MAE ≈ 0.1211
#
# To evaluate the baseline architecture instead, replace the `cp -r
# src/final_architecture/schnetpack ...` line above with `src/baseline_architecture/schnetpack`.
```

To reproduce the **reproduced-baseline** numbers instead, point the same
protocol at the `_baseline`-tagged source tree (`src/baseline_architecture/`)
using `scripts/painn_generalization/run_experiment.py`, e.g.:
```bash
cd /path/to/Geometry-Augmented-PaiNN-MD17
python scripts/painn_generalization/run_experiment.py \
  --molecule uracil --arch baseline --seed 42 \
  --split-npz data/splits/md17_uracil_split.npz \
  --src-path src/baseline_architecture \
  --log-file /tmp/run.log \
  --out-json /tmp/uracil_baseline_seed42.json
```
`--molecule`, `--arch`, `--seed`, `--split-npz`, `--src-path`, `--log-file`
and `--out-json` are all required by the script; omitting any of them exits
with an argparse error rather than silently falling back to a default.

This exact invocation (`run_experiment.py --molecule uracil --arch baseline
--seed 42`) has not itself been executed to produce a result file in this
artifact; `0.4755`/`0.1973` is the measured baseline reported throughout the
paper and was obtained from the protected `eval_benchmark.py` campaign, not
from an independent run of this script. Do not treat `0.4755`/`0.1973` as a
tight reproduction target for *this* script: an independently executed
seed=42 baseline training run of the byte-identical architecture (verified by
`diff` against `src/baseline_architecture`'s `_baseline`-tag source; see
`scripts/painn_sci_scratch/train_and_extract.py --variant baseline`, recorded
in `predictions_and_logs/training_logs/train_baseline.log` and
`predictions_and_logs/raw_predictions/training_curve_baseline.json`) produced
test force MAE `0.4927` and test energy MAE `0.2425`, a non-trivial spread
from the `eval_benchmark.py` value despite identical source code,
hyperparameters, and seed. The spread is consistent with CPU-only execution
(`eval_benchmark.py` auto-selects CUDA when available, while the
supplementary verification scripts are hardcoded to CPU) and with the known
lack of run-to-run determinism of scatter-add / autograd-based force
evaluation across devices and thread counts; it has not been further
decomposed. Both numbers are genuine measured results of the unmodified
baseline architecture and are reported honestly at their respective points of
use: `0.4755`/`0.1973` as the primary campaign baseline denominator (Table 1
and the headline reductions), and `0.4927`/`0.2425` as the baseline reference
in the Figure 4 error-decomposition comparison, which pairs it with a
baseline-free final-model number produced by the same supplementary script
family so that the comparison is internally consistent even though neither
supplementary number is expected to exactly match the primary campaign.

## 5. One-click reproduction of the supplementary multi-seed and cross-molecule results

```bash
cd /path/to/Geometry-Augmented-PaiNN-MD17
conda activate cgcnn_env
mkdir -p /tmp/painn_results

# Multi-seed replication on uracil (Table 1): 3 seeds never used during
# architecture search, for both the reproduced baseline and the final
# architecture, all other settings identical to the protocol above.
for seed in 1001 1002 1003; do
  for variant in baseline improved; do
    src="src/final_architecture"
    [ "$variant" = "baseline" ] && src="src/baseline_architecture"
    python scripts/painn_generalization/run_experiment.py \
      --molecule uracil --arch "$variant" --seed "$seed" \
      --split-npz data/splits/md17_uracil_split.npz \
      --src-path "$src" \
      --log-file /tmp/painn_results/run.log \
      --out-json /tmp/painn_results/uracil_${variant}_seed${seed}.json
  done
done

# Cross-molecule zero-tuning transfer (Table 2): same seed (42) used
# throughout architecture search, applied unchanged to ethanol and aspirin.
for molecule in ethanol aspirin; do
  for variant in baseline improved; do
    src="src/final_architecture"
    [ "$variant" = "baseline" ] && src="src/baseline_architecture"
    python scripts/painn_generalization/run_experiment.py \
      --molecule "$molecule" --arch "$variant" --seed 42 \
      --split-npz "data/splits/md17_${molecule}_split.npz" \
      --src-path "$src" \
      --log-file /tmp/painn_results/run.log \
      --out-json /tmp/painn_results/${molecule}_${variant}_seed42.json
  done
done
```
`--molecule`, `--arch`, `--split-npz`, `--src-path`, `--log-file` and
`--out-json` must all be passed explicitly on every invocation; the script
has no default for any of them.

Each run reproduces one row of Table 1 or Table 2 in the paper. No
architecture search, hyperparameter tuning, or epoch-budget change was
performed for any seed or molecule beyond the fixed protocol above; every
run uses the same 80-epoch budget as the headline uracil comparison. The
already-committed results of exactly these twenty runs (ten `(molecule,
variant, seed)` combinations, each with a `.json` result and a `.stdout`
capture) are in `predictions_and_logs/multiseed_crossmol_logs/`.

## 6. Reproducing Figure 4 (error decomposition)

Figure 4 (`figures/fig4_uracil_error_distribution.pdf`) is built from two
single real forward-inference passes over the full uracil test set (1000
configurations): a trained final-architecture ($\tanh$-gated) model, recorded
in `predictions_and_logs/raw_predictions/predictions_final.npz` and
`predictions_and_logs/training_logs/train_final.log` (`Final_Test_MAE: 0.3024`,
`Test_Energy_MAE: 0.1185`), and a trained baseline (unmodified, un-gated
`_baseline`/f4683fcc PaiNN) model, recorded in
`predictions_and_logs/raw_predictions/predictions_baseline.npz` and
`predictions_and_logs/training_logs/train_baseline.log` (`Final_Test_MAE: 0.4927`,
`Test_Energy_MAE: 0.2425`; see §4 above for why this baseline number differs
from the primary-campaign `0.4755`/`0.1973`). Both runs share the same data
split, seed (42), and 80-epoch protocol, differing only in which `src` tree
(hence which `painn.py`) is imported. The per-atom-index assignment to
element and chemical microenvironment used in that figure is not hard-coded
from prior knowledge of uracil's structure: it is derived programmatically
from the raw atomic numbers and first test-frame coordinates in
`data/splits/md17_uracil_split.npz` using covalent bond-length thresholds
(see `scripts/painn_sci_scratch/verify_uracil_atom_mapping.py`, which
independently reconstructs the molecular graph from geometry alone and
confirms it matches the grouping used for plotting).

To regenerate the figure and the independent atom-mapping verification,
run from this repository checked out on the machine the original project
ran on (see the path note in §2 above), or adapt the hardcoded input/output
paths at the top of each script to your own layout:
```bash
/home/ymujdb/my_sota_project/.autosota/venv/bin/python scripts/painn_sci_scratch/verify_uracil_atom_mapping.py
/home/ymujdb/my_sota_project/.autosota/venv/bin/python scripts/painn_sci_scratch/plot_fig4_error_analysis.py
```

No trained model checkpoint (`.pt`/`.pth` state dict) for any run already
reported in this paper (every number in Tables 1-4, Fig. 1D, and Fig. 4) was
retained on disk after training completed; only the scalar logs
(`train_final.log`, `train_baseline.log`) and the one-time exported per-atom
prediction arrays (`predictions_final.npz`, `predictions_baseline.npz`) were
kept, consistent with the project's `eval_benchmark.py` protocol, which
evaluates the best-validation-epoch in-memory state and does not checkpoint
to disk. This applies retroactively to every already-reported number; none
of them can be re-derived from a saved checkpoint, only from re-running the
fixed protocol.

For runs launched *from now on*, `scripts/painn_generalization/run_experiment.py`
accepts an additive, opt-in `--save-checkpoint <path>` argument that writes the
best-validation-epoch `model.state_dict()` (plus `best_epoch`,
`best_val_f_mae`/`e_mae`, `test_f_mae`/`e_mae`, and run metadata) to disk with
`torch.save` after the unmodified 80-epoch training/selection/evaluation
protocol has already produced its test-set numbers. Saving happens strictly
after `test_f_mae`/`test_e_mae` are computed, so it cannot influence them; it
is new instrumentation for future reproducibility, not a retroactive source
for the numbers already in the paper. Omitting the flag reproduces the exact
prior behaviour (no file written), so every run already reported here remains
accurately described by the paragraph above.

Re-running Figure 4 from the raw `.npz` files therefore reproduces the figure
exactly from already-recorded real predictions; regenerating the predictions
themselves from scratch requires re-running the training protocol in §4
above and re-exporting per-atom predictions with:
```bash
/home/ymujdb/my_sota_project/.autosota/venv/bin/python scripts/painn_sci_scratch/train_and_extract.py \
  --variant final --out-dir <output-directory>
/home/ymujdb/my_sota_project/.autosota/venv/bin/python scripts/painn_sci_scratch/train_and_extract.py \
  --variant baseline --out-dir <output-directory>
```
(`--variant sigmoid` instead reproduces the sigmoid-gated control run used
in Figure 1D; `--variant` and `--out-dir` are required arguments for all
three variants. Note that `train_and_extract.py` hardcodes CPU execution
regardless of GPU availability, unlike `eval_benchmark.py`; re-running these
commands is therefore expected to take substantially longer than the
original GPU-capable campaign and, per §4 above, is not guaranteed to
reproduce the exact decimal MAE values recorded here, only the same
architecture, protocol, and qualitative baseline-vs-final ordering.)

## 7. What is NOT included in this artifact

- Full MD17 raw datasets (`rmd17_uracil.npz` etc.) are not redistributed here;
  they are the standard, publicly available MD17/revised-MD17 dataset
  (Chmiela et al., Christensen & von Lilienfeld) and must be obtained from
  the original source. Only the fixed 950/50/1000 split indices and the
  corresponding sliced coordinates/energies/forces are included via the
  `.npz` files referenced above.
- Intermediate-architecture checkpoints (the 7 iterative steps between
  `_baseline` and `_best`) are not retained as binary weights; their
  per-step scalar metrics are recorded in `predictions_and_logs/training_logs/`
  and summarized in Table 3 of the paper. Only the final architecture's
  source code and patch are packaged here, per §1.
