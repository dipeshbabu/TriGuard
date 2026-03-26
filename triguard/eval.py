import csv
import os

import numpy as np
import torch

from .attacks import pgd_linf
from .attributions import (
    ads_adv,
    ads_baseline,
    attribution_entropy,
    blurred_baseline,
    integrated_gradients,
    noise_baseline,
    smoothgrad_squared,
    uniform_baseline,
)
from .faithfulness import faithfulness_auc
from .verify import crown_ibp_certify, empirical_probe


def _append_finite(values: list[float], value: float | None):
    if value is None:
        return
    if np.isfinite(value):
        values.append(float(value))


def _safe_mean(values: list[float], default: float = float("nan")) -> float:
    if not values:
        return default
    return float(np.mean(values))


def _report_skipped(metric_name: str, valid_count: int, total_count: int):
    skipped = total_count - valid_count
    if skipped > 0:
        print(
            f"[Metric Warning] Skipped {skipped}/{total_count} invalid samples while computing {metric_name}."
        )


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
    return correct / max(total, 1)


def pgd_accuracy(model, loader, device, eps, alpha, steps, clamp_min, clamp_max, max_batches=10):
    model.eval()
    correct = 0
    total = 0
    seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_linf(model, x, y, eps, alpha, steps, clamp_min, clamp_max, random_start=True)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        seen += 1
        if seen >= max_batches:
            break
    return correct / max(total, 1)


def evaluate_main_metrics(
    model,
    test_set,
    device,
    eps,
    alpha,
    pgd_steps,
    clamp_min,
    clamp_max,
    ig_steps=50,
    k=100,
    seed=0,
    target_mode="truth",
    bound_probe_samples=16,
    do_crown=True,
    baseline_min=0.0,
    baseline_max=1.0,
):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(k, len(test_set)), replace=False)

    ent_list = []
    ads_list = []
    bound_list = []
    crown_list = []

    for idx in idxs:
        x, y = test_set[idx]
        y = int(y)
        x = x.to(device).unsqueeze(0)

        if target_mode == "pred":
            with torch.no_grad():
                target = int(model(x).argmax(dim=1).item())
        else:
            target = y

        b0 = torch.zeros_like(x)
        b_blur = blurred_baseline(x)

        ig = integrated_gradients(model, x, target, b0, steps=ig_steps)
        _append_finite(ent_list, attribution_entropy(ig))
        _append_finite(ads_list, ads_baseline(model, x, target, b0, b_blur, steps=ig_steps))

        bound_ok = empirical_probe(model, x.squeeze(0), bound_probe_samples, eps, clamp_min, clamp_max, device)
        bound_list.append(int(bound_ok))
        crown_list.append(int(crown_ibp_certify(model, x.squeeze(0), y, eps, device)) if do_crown else 0)

    _report_skipped("entropy_mean", len(ent_list), len(idxs))
    _report_skipped("ads_mean", len(ads_list), len(idxs))
    return {
        "entropy_mean": _safe_mean(ent_list),
        "ads_mean": _safe_mean(ads_list),
        "entropy_valid_n": int(len(ent_list)),
        "ads_valid_n": int(len(ads_list)),
        "bound_check_rate": _safe_mean(bound_list),
        "crown_rate": _safe_mean(crown_list),
    }


def evaluate_appendix_metrics(
    model,
    test_set,
    device,
    eps,
    alpha,
    pgd_steps,
    clamp_min,
    clamp_max,
    ig_steps=50,
    k=100,
    seed=0,
    target_mode="truth",
    baseline_min=0.0,
    baseline_max=1.0,
):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(k, len(test_set)), replace=False)

    ads_adv_list = []
    base_rows = []

    for idx in idxs:
        x, y = test_set[idx]
        y = int(y)
        x = x.to(device).unsqueeze(0)

        if target_mode == "pred":
            with torch.no_grad():
                target = int(model(x).argmax(dim=1).item())
        else:
            target = y

        b0 = torch.zeros_like(x)
        b_blur = blurred_baseline(x)
        b_noise = noise_baseline(x)
        b_uniform = uniform_baseline(x, low=baseline_min, high=baseline_max)

        y_tensor = torch.tensor([y], device=device, dtype=torch.long)
        x_adv = pgd_linf(model, x, y_tensor, eps, alpha, pgd_steps, clamp_min, clamp_max, random_start=True)
        ads_adv_value = ads_adv(model, x, x_adv, target, b0, steps=ig_steps)
        if ads_adv_value is not None and np.isfinite(ads_adv_value):
            ads_adv_list.append(float(ads_adv_value))

        row = {
            "ads_zero_blur": ads_baseline(model, x, target, b0, b_blur, steps=ig_steps),
            "ads_zero_noise": ads_baseline(model, x, target, b0, b_noise, steps=ig_steps),
            "ads_zero_uniform": ads_baseline(model, x, target, b0, b_uniform, steps=ig_steps),
            "ads_blur_noise": ads_baseline(model, x, target, b_blur, b_noise, steps=ig_steps),
            "ads_blur_uniform": ads_baseline(model, x, target, b_blur, b_uniform, steps=ig_steps),
            "ads_noise_uniform": ads_baseline(model, x, target, b_noise, b_uniform, steps=ig_steps),
        }
        if all(value is not None and np.isfinite(value) for value in row.values()):
            base_rows.append({key: float(value) for key, value in row.items()})

    _report_skipped("ads_adv_mean", len(ads_adv_list), len(idxs))
    _report_skipped("baseline sensitivity metrics", len(base_rows), len(idxs))
    result = {"ads_adv_mean": _safe_mean(ads_adv_list)}
    baseline_keys = [
        "ads_zero_blur",
        "ads_zero_noise",
        "ads_zero_uniform",
        "ads_blur_noise",
        "ads_blur_uniform",
        "ads_noise_uniform",
    ]
    for key in baseline_keys:
        result[key] = _safe_mean([row[key] for row in base_rows])
    return result


def evaluate_faithfulness(model, test_set, device, k=50, ig_steps=50, delins_steps=50,
                          seed=0, baseline_mode="zero", target_mode="truth",
                          smoothgrad_noise=0.1, smoothgrad_samples=50):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(k, len(test_set)), replace=False)

    ig_del, ig_ins = [], []
    sg_del, sg_ins = [], []
    curves = []
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

        baseline = torch.zeros_like(x)
        ig_attr = integrated_gradients(model, x, target, baseline, steps=ig_steps)
        ig_d, ig_i, del_curve, ins_curve = faithfulness_auc(model, x, target, ig_attr, steps=delins_steps, baseline=baseline)
        _append_finite(ig_del, ig_d)
        _append_finite(ig_ins, ig_i)

        sg_attr = smoothgrad_squared(model, x, target, noise_level=smoothgrad_noise, n_samples=smoothgrad_samples)
        sg_d, sg_i, del_curve2, ins_curve2 = faithfulness_auc(model, x, target, sg_attr, steps=delins_steps, baseline=baseline)
        _append_finite(sg_del, sg_d)
        _append_finite(sg_ins, sg_i)

        if t < keep_curves:
            curves.append(("IG", del_curve, ins_curve))
            curves.append(("SmoothGrad2", del_curve2, ins_curve2))

    _report_skipped("ig_del_auc_mean", len(ig_del), len(idxs))
    _report_skipped("ig_ins_auc_mean", len(ig_ins), len(idxs))
    _report_skipped("sg2_del_auc_mean", len(sg_del), len(idxs))
    _report_skipped("sg2_ins_auc_mean", len(sg_ins), len(idxs))

    return {
        "ig_del_auc_mean": _safe_mean(ig_del),
        "ig_ins_auc_mean": _safe_mean(ig_ins),
        "sg2_del_auc_mean": _safe_mean(sg_del),
        "sg2_ins_auc_mean": _safe_mean(sg_ins),
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
