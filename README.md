# TriGuard: Multi Axis Evaluation of Robustness and Explanation Stability

TriGuard is a practical evaluation toolkit for studying when prediction robustness, certification aware checks, and explanation stability disagree in image classifiers.

## Setup

This repo is locked to Python 3.10.x and pinned to 3.10.12 via `.python-version`. Create and activate a Python 3.10 virtual environment first, then run the install script:

```bash
python3.10 -m venv .triguard
source .triguard/bin/activate

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

| Dataset      | Models                          |
|--------------|---------------------------------|
| MNIST        | SimpleCNN, ResNet50             |
| FashionMNIST | SimpleCNN, ResNet50             |
| CIFAR10      | ResNet50, DenseNet121, ViT B 16 |
| CIFAR100     | ResNet50, DenseNet121, ViT B 16 |

## Suggested workflow

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

## Acknowledgement

This project uses the external [`auto_LiRPA`](https://github.com/Verified-Intelligence/auto_LiRPA) library for certification-aware verification components. A copy of that library is included in this repository under `auto_LiRPA/`.

## Notes on framing

This codebase is aligned with the workshop version of the paper, where TriGuard is presented as a multi axis diagnostic framework rather than a complete safety certificate. The main paper path focuses on robustness and explanation stability. Appendix workflows remain available for baseline sensitivity and faithfulness checks.
