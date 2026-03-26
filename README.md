# TriGuard: Multi Axis Evaluation of Robustness and Explanation Stability

TriGuard is a practical evaluation toolkit for studying when prediction robustness, certification aware checks, and explanation stability disagree in image classifiers.

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
````

Run the lambda ablation:

```bash
bash scripts/02_run_lambda_ablation.sh
```

Generate LaTeX tables:

```bash
bash scripts/05_make_tables.sh
```

## Notes on framing

This codebase is aligned with the workshop version of the paper, where TriGuard is presented as a multi axis diagnostic framework rather than a complete safety certificate. The main paper path focuses on robustness and explanation stability. Appendix workflows remain available for baseline sensitivity and faithfulness checks.
