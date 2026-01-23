import os
import csv
import numpy as np
import torch

from .attacks import pgd_linf
from .attributions import (
    integrated_gradients, attribution_entropy,
    blurred_baseline, ads_baseline, ads_adv,
    smoothgrad_squared
)
from .verify import crown_ibp_certify
from .faithfulness import faithfulness_auc


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def pgd_accuracy(model, loader, device, eps, alpha, steps, clamp_min, clamp_max, max_batches=10):
    model.eval()
    correct = 0
    total = 0
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_linf(model, x, y, eps, alpha, steps, clamp_min, clamp_max)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        seen += 1
        if seen >= max_batches:
            break
    return correct / total


def evaluate_attribution_metrics(model, test_set, device, eps, alpha, pgd_steps, clamp_min, clamp_max,
                                 ig_steps=50, K=100, seed=0, target_mode="truth"):
    """
    Computes mean metrics over K test samples:
      entropy(IG), ADS (zero vs blurred), ADS-Adv (clean vs PGD)
      crown_ibp certification rate
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(K, len(test_set)), replace=False)

    ent_list = []
    ads_list = []
    adsA_list = []
    crown_list = []

    for idx in idxs:
        x, y = test_set[idx]
        y = int(y)
        x = x.to(device).unsqueeze(0)

        # choose consistent target
        if target_mode == "pred":
            with torch.no_grad():
                target = int(model(x).argmax(dim=1).item())
        else:
            target = y

        b0 = torch.zeros_like(x)
        b_blur = blurred_baseline(x)

        ig = integrated_gradients(model, x, target, b0, steps=ig_steps)
        ent_list.append(attribution_entropy(ig))
        ads_list.append(ads_baseline(
            model, x, target, b0, b_blur, steps=ig_steps))

        y_tensor = torch.tensor([y], device=device, dtype=torch.long)
        x_adv = pgd_linf(model, x, y_tensor, eps, alpha,
                         pgd_steps, clamp_min, clamp_max)
        adsA_list.append(ads_adv(model, x, x_adv, target, b0, steps=ig_steps))

        crown_list.append(int(crown_ibp_certify(
            model, x.squeeze(0), y, eps, device)))

    return {
        "entropy_mean": float(np.mean(ent_list)),
        "ads_mean": float(np.mean(ads_list)),
        "ads_adv_mean": float(np.mean(adsA_list)),
        "crown_rate": float(np.mean(crown_list)),
    }


def evaluate_faithfulness(model, test_set, device, K=50, ig_steps=50, delins_steps=50,
                          seed=0, baseline_mode="zero", target_mode="truth",
                          smoothgrad_noise=0.1, smoothgrad_samples=50):
    """
    Computes mean deletion/insertion AUC for IG and SmoothGrad^2 over K samples.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(K, len(test_set)), replace=False)

    ig_del, ig_ins = [], []
    sg_del, sg_ins = [], []

    curves = []  # store a couple curves for plotting
    keep_curves = 3

    for t, idx in enumerate(idxs):
        x, y = test_set[idx]
        y = int(y)
        x = x.to(device).unsqueeze(0)

        if target_mode == "pred":
            with torch.no_grad():
                target = int(model(x).argmax(dim=1).item())
        else:
            target = y

        if baseline_mode == "zero":
            baseline = torch.zeros_like(x)
        else:
            baseline = torch.zeros_like(x)

        ig_attr = integrated_gradients(
            model, x, target, baseline, steps=ig_steps)
        ig_d, ig_i, del_curve, ins_curve = faithfulness_auc(
            model, x, target, ig_attr, steps=delins_steps, baseline=baseline)
        ig_del.append(ig_d)
        ig_ins.append(ig_i)

        sg_attr = smoothgrad_squared(
            model, x, target, noise_level=smoothgrad_noise, n_samples=smoothgrad_samples)
        sg_d, sg_i, del_curve2, ins_curve2 = faithfulness_auc(
            model, x, target, sg_attr, steps=delins_steps, baseline=baseline)
        sg_del.append(sg_d)
        sg_ins.append(sg_i)

        if t < keep_curves:
            curves.append(("IG", del_curve, ins_curve))
            curves.append(("SmoothGrad2", del_curve2, ins_curve2))

    return {
        "ig_del_auc_mean": float(np.mean(ig_del)),
        "ig_ins_auc_mean": float(np.mean(ig_ins)),
        "sg2_del_auc_mean": float(np.mean(sg_del)),
        "sg2_ins_auc_mean": float(np.mean(sg_ins)),
        "curves": curves,
    }


def append_csv(path, row, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)
