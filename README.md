# 🛡️ TriGuard: Testing Model Safety with Attribution Entropy, Verification, and Drift

TriGuard is a diagnostic toolkit for evaluating the **safety** of image classifiers across three critical dimensions:

- ✅ **Robustness**: Adversarial accuracy and certified verification
- 🧠 **Interpretability**: Attribution entropy and drift
- 📊 **Faithfulness**: Saliency effectiveness under input perturbations

We introduce **Attribution Drift Score (ADS)** and demonstrate how entropy-regularized training improves explanation stability even in models with high adversarial performance. For methodology, analysis, and results, see the full paper: [![Paper](https://img.shields.io/badge/Paper-red)](https://arxiv.org/abs/2506.14217)

---

## 🚀 Features

- 📈 Multi-axis safety metrics: Accuracy, adversarial error, drift, entropy, SmoothGrad², and CROWN-IBP
- 🔬 Saliency faithfulness via **Deletion/Insertion AUC** curves
- 🔁 Evaluation with and without entropy-regularized training
- ✅ Supports multiple models (SimpleCNN, ResNet, DenseNet, MobileNetV3)
- 📦 Datasets: MNIST, FashionMNIST, CIFAR-10

---

## 🔖 Citation

If you use this codebase or find our work helpful, please cite:

```bibtex
@article{mahato2025triguard,
  title={TriGuard: Testing Model Safety with Attribution Entropy, Verification, and Drift},
  author={Mahato, Dipesh Tharu and Poudel, Rohan and Dhungana, Pramod},
  journal={arXiv preprint arXiv:2506.14217},
  year={2025}
}
```
