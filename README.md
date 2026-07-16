# TriGuard: reference-distribution robustness for Integrated Gradients

TriGuard studies how Integrated Gradients changes when its reference changes. It separates the constant component forced by completeness from the remaining allocation drift, then trains against the upper tail of drift over a reference distribution.

This repository is a frontier candidate, not a finished empirical result. The method, theory, and protocol are implemented; the protocol-v2.3 experiments still need to be run. The paper deliberately contains no numerical claims from old or simulated outputs.

TriGuard does not produce a safety score. Zero, blur, noise, uniform, and midpoint references are a synthetic stress test. Optional reference banks use real images selected for neutral predictions by a frozen calibration model, but that still does not establish domain-semantic missingness.

## Setup

This repo is locked to Python 3.10.x and pinned to 3.10.12 via `.python-version`. Run one command to install the pinned `uv`-managed Python, create the local `.triguard` environment, install dependencies, and start an activated shell:

```bash
bash scripts/00_install.sh
```

After setup, follow the commands in the `Suggested workflow` section. The workflow scripts automatically use `.triguard`, even if your current shell still points at another Python.
The installer defaults to CUDA 12.1 PyTorch wheels. For a CPU-only machine, run
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash scripts/00_install.sh`.

## Validation

From the repository root with `.triguard` activated:

```bash
python -m ruff check run_triguard.py triguard tests
python -m compileall -q run_triguard.py triguard tests
python -m unittest discover -s tests -v
bash -n scripts/*.sh
```

## Main outputs

The current paper workflow centers on:

- clean accuracy
- five-restart PGD error or standard AutoAttack in pixel coordinates
- empirical random-probe violation rate and worst margin (not a certificate)
- unconditional CROWN proven rate, conditional valid-call rate, and error taxonomy
- normalized pixel-level attribution entropy
- fixed-reference allocation ADS: signed L1 distance after absolute-mass normalization
- allocation WADS: the per-sample worst ADS over the finite reference family
- raw RMS drift, completeness-orthogonal RMS drift, and baseline-output gaps
- relative attribution-mass ratio tails and floor-violation rate, plus raw-mass diagnostics
- absolute and relative Integrated Gradients completeness error over every reference
- held-out reference-bank WADS and completeness-orthogonal drift
- held-out relative mass-ratio tails and floor-violation rate
- prediction-preserving attribution stability under small image transforms
- same-checkpoint deletion/insertion faithfulness AUC with a random-ranking control
- train, evaluation, and total runtime

TriGuard-Train supports:

- mean, CVaR, or maximum risk over reference pairs
- calibration-relative in-distribution reference banks and disjoint held-out banks
- a scale-invariant attribution-mass floor against near-zero-map collapse
- RAR-like fixed-baseline adversarial attribution regularization
- FAR-like fixed-baseline local attribution regularization
- local gradient-variation regularization
- one-step adversarial consistency regularization
- entropy-only and CE-only controls
- sampled reference pairs for larger banks
- pair-first reference selection so unused references are never differentiated
- vectorized multi-reference IG for higher accelerator utilization
- effective-batch-preserving regularizer microbatching
- optional activation checkpointing for higher-order IG
- sampled mass-floor estimation over the active reference subset
- optional accelerator-resident reference banks
- batched SmoothGrad and deletion/insertion curve evaluation

Additional workflows also support:

- baseline sensitivity for Attribution Drift Score
- faithfulness via deletion and insertion AUC
- SmoothGrad squared comparison

## Supported datasets and models

The workshop grid is kept for reproducibility.

| Dataset      | Models                          |
|--------------|---------------------------------|
| MNIST        | SimpleCNN, ResNet50             |
| FashionMNIST | SimpleCNN, ResNet50             |
| CIFAR10      | ResNet50, DenseNet121, ViT B 16 |
| CIFAR100     | ResNet50, DenseNet121, ViT B 16 |

The upgraded grid adds ImageNet initialized models for the CIFAR settings:

| Dataset  | Models |
|----------|--------|
| CIFAR10  | ResNet50-ImageNet, DenseNet121-ImageNet, ViT-B/16-ImageNet, ConvNeXt-Tiny-ImageNet, Swin-T-ImageNet |
| CIFAR100 | ResNet50-ImageNet, DenseNet121-ImageNet, ViT-B/16-ImageNet, ConvNeXt-Tiny-ImageNet, Swin-T-ImageNet |

## Suggested workflow

The primary protocol uses fixed epochs, not training-loss early stopping. MNIST and FashionMNIST use 5 epochs by default, CIFAR-10 uses 10, and CIFAR-100 uses 15 unless `--epochs` overrides the budget.

Run the main benchmark with multiple seeds:

```bash
bash scripts/01_run_main_table.sh
```

Run the lambda ablation:

```bash
bash scripts/02_run_lambda_ablation.sh
```

Run the baseline sensitivity appendix workflow:

```bash
bash scripts/03_run_baseline_sensitivity.sh
```

Run the faithfulness appendix workflow:

```bash
bash scripts/04_run_faithfulness.sh
```

Generate LaTeX tables:

```bash
bash scripts/05_make_tables.sh
```

## Upgraded paper workflow

Run the pretrained architecture grid as a secondary breadth analysis:

```bash
bash scripts/06_run_pretrained_grid.sh
```

Run a separate certification sweep on a CROWN-friendly MNIST/SimpleCNN setting with smaller pixel-space radii:

```bash
bash scripts/07_run_certification_sweep.sh
```

First reserve a model-independent candidate pool. The same pool is excluded
from calibration training and every compared classifier:

```bash
bash scripts/11_reserve_reference_candidates.sh
```

Then build disjoint reference banks from a frozen target-task checkpoint that
was trained with `--exclude_train_indices_file` pointing to that reservation:

```bash
CHECKPOINT=outputs/icml2026_calibration/checkpoints/main_cifar10_resnet50_imagenet_seed0_lam0.000.pt \
  CANDIDATE_INDICES=reference_banks/mainconf/cifar10_resnet50_imagenet_candidates.pt \
  bash scripts/11_build_reference_banks.sh
```

Run the focused twenty-seed primary TriGuard-Train comparison. Set `REFERENCE_BANK` and
`HELDOUT_REFERENCE_BANK` to include the calibration-relative bank protocol. Bank draws
are without replacement, with four references per training example and sixteen
per evaluation example:

```bash
bash scripts/09_run_triguard_train.sh
```

Run a ten-seed secondary TriGuard-Train ablation. By default this runs CIFAR-10 with `resnet50_imagenet` over:

- CE only
- entropy only
- RAR-like fixed-baseline attribution regularization
- FAR-like fixed-baseline attribution regularization
- small WADS
- mean reference-pair risk
- WADS only
- WADS + curvature
- WADS + curvature + robust consistency
- CVaR reference risk + curvature + robust consistency
- CVaR reference risk + curvature + robust consistency + mass floor

```bash
bash scripts/10_run_triguard_train_ablation.sh
```

Generate statistical summaries, paired tests, per-dataset correlations, accuracy--risk tradeoff plots, and LaTeX tables:

```bash
bash scripts/08_run_stats_and_figures.sh
```

For ablations, the statistics workflow writes `regularizer_paired_tests.csv`. The focused primary run also isolates the two prespecified held-out-WADS contrasts in `primary_paired_tests.csv`; accuracy, adversarial-error, held-out mass-floor, and same-checkpoint faithfulness guardrails are evaluated in `primary_decisions.csv`. `design_sensitivity.csv` records the planning approximation for the matched-seed count actually present. Architecture comparisons remain secondary and are written to `mannwhitney_tests.csv`.

Generate artifacts for the baseline, TriGuard-Train, and TriGuard-Train ablation output directories:

```bash
bash scripts/11_make_triguard_train_artifacts.sh
```

Run the full main-conference rerun sequence:

```bash
bash scripts/12_run_mainconf_workflow.sh
```

This creates the reservation, trains the leakage-free calibration checkpoint,
runs the pretrained grid and certification sweep, builds the reference banks,
then runs the focused training comparison, ablation, and artifact generation.
To resume a partial run, set any block flag to `0`, for example:

```bash
RUN_RESERVATION=0 RUN_CALIBRATION=0 RUN_PRETRAINED_GRID=0 \
  RUN_CERT_SWEEP=0 RUN_REFERENCE_BANK=0 \
  bash scripts/12_run_mainconf_workflow.sh
```

On a multi-GPU machine, the full workflow can shard the focused primary and
ablation seeds automatically:

```bash
PARALLEL_GPU_IDS=0,1,2,3,4,5,6,7 \
  bash scripts/12_run_mainconf_workflow.sh
```

All TriGuard-Train scripts accept environment overrides. For example:

```bash
SEEDS=0,1 DATASET=cifar100 MODEL=convnext_tiny_imagenet \
  bash scripts/10_run_triguard_train_ablation.sh
```

The focused frontier workflows default to exact all-pair risk over every
reference instantiated for that training example (the five synthetic
references plus the configured bank draw) and full mass-floor evaluation.
They still use regularizer microbatches of one, vectorized reference IG,
accelerator-resident reference banks, and activation checkpointing. These can
be tuned without changing the cross-entropy batch or optimizer-step count:

```bash
REGULARIZER_MICROBATCH=2 PRELOAD_REFERENCE_BANKS=1 \
CHECKPOINT_REGULARIZER_IG=0 \
  bash scripts/09_run_triguard_train.sh
```

Pair subsampling and sampled mass-floor estimation are opt-in approximations:

```bash
REFERENCE_PAIR_SAMPLES=4 SAMPLED_MASS_PENALTY=1 \
  bash scripts/09_run_triguard_train.sh
```

Do not use those approximations for the frozen frontier protocol until their
checkpoint-level bias and ranking fidelity have been measured:

```bash
CHECKPOINT=outputs/.../checkpoints/...pt \
  bash scripts/15_audit_pair_sampling.sh
```

The audit computes every reference-pair loss once on fixed test examples, then
replays pair subsampling cheaply for mean, max, and CVaR. It writes absolute and
relative bias, MAE/RMSE, rank correlation, and full-tail pair coverage. Set
`FAIL_ON_THRESHOLD=1` to enforce the default 5% relative-bias and 0.90
rank-correlation gates.

Run seed shards concurrently on several GPUs with concurrency-safe CSV writes:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 WORKFLOW=primary \
  bash scripts/13_run_parallel_training.sh

GPU_IDS=0,1,2,3,4,5,6,7 WORKFLOW=ablation \
  bash scripts/13_run_parallel_training.sh

GPU_IDS=0,1,2,3,4 WORKFLOW=pretrained \
  bash scripts/13_run_parallel_training.sh

GPU_IDS=0,1,2,3,4 WORKFLOW=certification \
  bash scripts/13_run_parallel_training.sh
```

Each seed stays on one GPU across all treatments, preserving paired hardware
conditions. Statistical artifacts are generated once after every worker exits.

Before renting a multi-GPU fleet, benchmark the most expensive primary setting
for one epoch and run the full primary evaluation once. The projection includes
both repeated training and per-checkpoint AutoAttack/attribution/faithfulness
evaluation, and reports peak CUDA memory:

```bash
bash scripts/14_benchmark_primary.sh
```

Set `RUN_FULL_EVAL_BENCHMARK=0` only for a quick training-throughput check; the
resulting projection will explicitly warn that evaluation time is excluded.

The training scripts write to `outputs/icml2026_triguard_train_mainconf` and
`outputs/icml2026_triguard_train_ablation_mainconf` by default. Reuse an old output
directory only if its CSV headers match the current schema. Each row has a
unique run hash, a seed-independent condition hash, and a treatment-independent
comparison hash. Duplicate run identities are rejected, and `manifests/`
records arguments, artifact hashes, package versions, hardware, and Git state.
Every saved checkpoint also receives a content-hashed `.meta.json` sidecar.

Protocol-v2.3 outputs are intentionally incompatible with earlier CSVs, including protocol-v2.2 and the older raw-L2-only files under `outputs/`. The analysis and table builders reject mixed or stale protocol versions, and each run identity includes a scientific source-tree hash, so use a fresh output directory. Attribution evaluation explains the predicted class by default; pass `--target_mode truth` only for an explicitly label-conditioned analysis.

The old notebook and legacy manuscript figures live under `archive/` and
`paper/archive/`. They contain simulated or obsolete material and are not
experimental evidence.

To generate a saliency panel for a trained checkpoint:

```bash
python -m triguard.make_figures \
  --mode saliency \
  --dataset cifar10 \
  --model vit_b_16_imagenet \
  --checkpoint outputs/icml2026/checkpoints/<checkpoint-with-config-hash>.pt
```

## Acknowledgement

This project uses the external [`auto_LiRPA`](https://github.com/Verified-Intelligence/auto_LiRPA) library for certification-aware verification components. A copy of that library is included in this repository under `auto_LiRPA/`.

