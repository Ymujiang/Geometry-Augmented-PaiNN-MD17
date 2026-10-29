# Data Manifest — `predictions_and_logs/`

This manifest lists every real, measured artifact file in this repository
that feeds a reported number or figure in the paper, its SHA-256 checksum
(first 16 hex) at the time of this manifest, its producing script, and which
exact paper location (table, figure, or in-text number) it supports. No file
listed here is synthetic, interpolated, or hand-edited after generation.

## 1. Checkpoint-saving policy (explicit statement)

**No `.pt`/`.pth` model weight files are retained anywhere in this project for
any of the uracil/ethanol/aspirin training runs reported in the paper.**
This is a deliberate consequence of the evaluation protocol in the protected
`eval_benchmark.py` (reference copy: `src/eval_benchmark.py`; and its
parameterized clones `scripts/painn_generalization/run_experiment.py` and
`scripts/painn_sci_scratch/train_and_extract.py`), which:
1. trains in-process for the full fixed epoch budget,
2. tracks the best validation-force-MAE epoch's model state **in memory**
   (`best_state = copy.deepcopy(model.state_dict())`-style snapshotting, never
   written to disk mid-training),
3. loads that in-memory best state into the model for a single final test-set
   evaluation pass, and
4. exits, discarding the in-memory state.

Consequently, every MAE number already reported in this paper is reproducible
by re-running the fixed protocol end-to-end (see `README_REPRODUCE.md` §4),
but no pretrained checkpoint can be distributed or loaded directly for any of
them; reproduction of the reported numbers always means re-training, not
re-loading weights. The only persisted artifacts for those runs are (a) scalar
text logs, (b) one-time exported per-atom/per-epoch JSON and `.npz` arrays
produced by the extraction scripts below, and (c) the figures/tables derived
from those arrays. This is consistent across all three training-script
variants used for the paper's reported numbers (`eval_benchmark.py`,
`run_experiment.py` without `--save-checkpoint`, `train_and_extract.py`) —
none of them was called with checkpoint saving enabled for any run cited in
this paper.

**Going forward**, `run_experiment.py` additionally accepts an opt-in
`--save-checkpoint <path>` flag that `torch.save`s the best-validation-epoch
state dict after the fixed train/select/evaluate protocol has already
produced its test MAEs, strictly as additive post-hoc instrumentation that
cannot influence those MAEs. It defaults to off, so every run already
reported in this paper remains exactly as described above; the flag only
changes behaviour for runs launched with it from this point on.
`eval_benchmark.py` (protected) and `train_and_extract.py` are unchanged and
still never call `torch.save` on a model state dict.

## 2. `predictions_and_logs/raw_predictions/` and `predictions_and_logs/training_logs/` — Figure 1D and Figure 4 source data

| File | SHA-256 (first 16 hex) | Producing script | Supports |
|---|---|---|---|
| `raw_predictions/predictions_final.npz` | `212517e87ef6d62d` | `train_and_extract.py --variant final` | Fig. 4 (A)-(D), final-model series |
| `raw_predictions/training_curve_final.json` | `7b10116957b0f00f` | same | Fig. 4 caption "force MAE 0.3024, energy MAE 0.1185"; Fig. 2 supplementary curve |
| `training_logs/train_final.log` | `08594be4af94cba0` | same | cross-check of `training_curve_final.json` scalars |
| `raw_predictions/predictions_baseline.npz` | `4f40345c0f573903`† | `train_and_extract.py --variant baseline` | Fig. 4 (A)-(D), baseline series |
| `raw_predictions/training_curve_baseline.json` | `015763eb7bb3ca10` | same | Fig. 4 caption "baseline: force MAE 0.4927, energy MAE 0.2425" |
| `training_logs/train_baseline.log` | `fa2415b046ac2549` | same | cross-check of `training_curve_baseline.json` scalars |
| `raw_predictions/predictions_sigmoid.npz` | `179e415e17551a53` | `train_and_extract.py --variant sigmoid` | Fig. 1D sigmoid-gate control series |
| `raw_predictions/training_curve_sigmoid.json` | `b5e227600c365f15` | same | Fig. 1D caption "sigmoid force MAE 0.3442" |
| `training_logs/train_sigmoid.log` | `62b3ef50a27a014a` | same | cross-check |
| `raw_predictions/gate_activation_final.json` | `c53d6d880cc49bdd` | same (forward-hook on `geo_gate_net`) | Fig. 1D gate-activation histogram, tanh variant |
| `raw_predictions/gate_activation_sigmoid.json` | `a35f39b8dcf94a39` | same | Fig. 1D gate-activation histogram, sigmoid variant |

† `predictions_baseline.npz` has no corresponding `gate_activation_baseline.json`
by design: the unmodified `_baseline` (f4683fcc) architecture has no
`geo_gate_net` module to hook, as documented in `train_and_extract.py`'s
docstring and in `README_REPRODUCE.md` §6.

Regeneration: `/home/ymujdb/my_sota_project/.autosota/venv/bin/python
scripts/painn_sci_scratch/plot_fig4_error_analysis.py` reads
`predictions_final.npz` and `predictions_baseline.npz` and writes
`figures/fig4_uracil_error_distribution.{pdf,png}` plus a JSON provenance
summary alongside them (note: this script, like the other plotting scripts
in `scripts/painn_sci_scratch/`, reads its inputs from and writes its outputs
to the absolute paths under `/home/ymujdb/my_sota_project/papers/painn_sci/`
used during the original AutoSOTA run; see the comment at the top of the
script before re-running it in a different environment).

## 3. `predictions_and_logs/multiseed_crossmol_logs/` — Table 1/2 multi-seed and cross-molecule data

| File pattern | Count | Producing script | Supports |
|---|---|---|---|
| `uracil_{baseline,improved}_seed{1001,1002,1003}.json` (+ matching `.stdout`) | 6 (+6) | `scripts/painn_generalization/run_experiment.py` | Table 1 multi-seed means/SDs; paper abstract's headline 35.1%/19.0% reductions |
| `{ethanol,aspirin}_{baseline,improved}_seed42.json` (+ matching `.stdout`) | 4 (+4) | `scripts/painn_generalization/run_experiment.py` | Table 2 cross-molecule zero-tuning transfer |
| `painn_generalization.log` | 1 | same (aggregate run log) | audit trail of all ten `run_experiment.py` invocations above |

No `uracil_baseline_seed42.json` exists in this directory: the single-seed
`0.4755`/`0.1973` baseline number used as the primary denominator in the
headline reduction percentages comes from the protected `eval_benchmark.py`
campaign log (`training_logs/painn_sota_summary.log` /
`training_logs/painn_optimizer_detail.log`), not from this directory or from
`run_experiment.py`. See `README_REPRODUCE.md` §4 for the full disclosure of
this distinction and of the measured spread between that number and an
independently executed `--arch baseline --seed 42` run.

## 4. Provenance chain summary

```
eval_benchmark.py  (protected, unmodified; reference copy src/eval_benchmark.py)
  -> predictions_and_logs/training_logs/{painn_sota_summary.log,painn_optimizer_detail.log}
       (10-step architecture-search campaign, incl. seed=42 baseline & final)
       -> Table 3 (ablation), Table 4 (cost), headline single-run numbers

scripts/painn_generalization/run_experiment.py  (parameterized clone, cwd-independent)
  -> predictions_and_logs/multiseed_crossmol_logs/*.json
       -> Table 1 (multi-seed), Table 2 (cross-molecule), headline multi-seed stats

scripts/painn_sci_scratch/train_and_extract.py  (parameterized clone, cwd-independent, CPU-only)
  -> predictions_and_logs/{raw_predictions,training_logs}/*_{final,sigmoid,baseline}.{npz,json,log}
       -> Figure 1D (gate-activation + sigmoid control), Figure 4 (error decomposition, final vs. baseline)
```

All three scripts train the exact same architecture code referenced by git
tag/commit (`_best`=c5c11994 for "final"/"improved", `_baseline`=f4683fcc for
"baseline"; see `src/final_architecture/` and `src/baseline_architecture/`
respectively) and the same fixed optimizer/scheduler/loss/grad-clip protocol;
only which `src` tree is imported, which molecule/split is loaded, and which
seed is set differ between runs. No number in this manifest or in the paper
was fabricated, hand-edited, or extrapolated.
