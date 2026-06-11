# TriGuard: A Multi-Axis Benchmark for Robustness and Attribution Stability

TriGuard is a practical evaluation toolkit for studying when prediction robustness, certification aware checks, and explanation stability disagree in image classifiers.

## Setup

This repo is locked to Python 3.10.x and pinned to 3.10.12 via `.python-version`. Create and activate a Python 3.10 virtual environment first, then run the install script:

```bash
python3.10 -m venv .triguard
source .triguard/bin/activate
```

After creating virtual environment, install the required dependencies then follow the instructions given in `Suggested workflow` section.

```
bash scripts/00_install.sh
```

## Main outputs

The workshop aligned version of the repo centers on:

- clean accuracy
- PGD error
- bound check rate
- CROWN rate
- attribution entropy
- Attribution Drift Score

Appendix only workflows also support:

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

The default training policy is dataset-aware: MNIST and FashionMNIST use 5 epochs by default, CIFAR10 uses 10, and CIFAR100 uses 15 unless you override `--epochs`.

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

For a stronger paper, run the pretrained grid with five seeds:

```bash
bash scripts/06_run_pretrained_grid.sh
```

Run a separate certification sweep on a CROWN-friendly MNIST/SimpleCNN setting with smaller pixel-space radii:

```bash
bash scripts/07_run_certification_sweep.sh
```

Generate statistical summaries, Mann-Whitney tests, metric correlations, radar plots, and updated LaTeX tables:

```bash
bash scripts/08_run_stats_and_figures.sh
```

To generate a saliency panel for a trained checkpoint:

```bash
python -m triguard.make_figures \
  --mode saliency \
  --dataset cifar10 \
  --model vit_b_16_imagenet \
  --checkpoint outputs/icml2026/checkpoints/main_cifar10_vit_b_16_imagenet_seed0_lam0.000.pt
```

## Acknowledgement

This project uses the external [`auto_LiRPA`](https://github.com/Verified-Intelligence/auto_LiRPA) library for certification-aware verification components. A copy of that library is included in this repository under `auto_LiRPA/`.

## Notes on framing

TriGuard is a diagnostic benchmark, not a safety certificate. The upgraded workflow is designed to address the main reviewer risks: weak ViT training, all-zero certification output, missing statistical tests, and the need to check whether ADS is being measured on informative attribution maps.
