import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

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


def _baseline_family(x, modes: str, baseline_min: float, baseline_max: float):
    family = {}
    for mode in [m.strip().lower() for m in modes.split(",") if m.strip()]:
        if mode == "zero":
            family[mode] = torch.zeros_like(x)
        elif mode == "blur":
            family[mode] = blurred_baseline(x)
        elif mode == "noise":
            family[mode] = noise_baseline(x, low=baseline_min, high=baseline_max)
        elif mode == "uniform":
            family[mode] = uniform_baseline(x, low=baseline_min, high=baseline_max)
        elif mode == "mean":
            family[mode] = torch.full_like(x, (baseline_min + baseline_max) / 2.0)
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")
    return family


def _worst_ads_baseline(model, x, target, modes, baseline_min, baseline_max, steps):
    family = _baseline_family(x, modes, baseline_min, baseline_max)
    if len(family) < 2:
        return None
    attrs = {
        name: integrated_gradients(model, x, target, baseline, steps=steps)
        for name, baseline in family.items()
    }
    best = None
    names = list(attrs)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            dist = torch.norm((attrs[left] - attrs[right]).flatten(), p=2)
            if torch.isfinite(dist):
                value = float(dist.item())
                best = value if best is None else max(best, value)
    return best


def _attribution_similarity(clean_attr, pert_attr, topk: int):
    clean = clean_attr.detach().flatten()
    pert = pert_attr.detach().flatten()
    if clean.numel() == 0 or pert.numel() == 0:
        return None
    l2 = torch.norm(clean - pert, p=2)
    cosine = F.cosine_similarity(clean.view(1, -1), pert.view(1, -1), dim=1)[0]
    k = min(max(int(topk), 1), clean.numel())
    clean_idx = torch.topk(clean.abs(), k=k).indices
    pert_idx = torch.topk(pert.abs(), k=k).indices
    clean_set = set(clean_idx.detach().cpu().tolist())
    pert_set = set(pert_idx.detach().cpu().tolist())
    union = clean_set | pert_set
    jaccard = len(clean_set & pert_set) / max(len(union), 1)
    if not torch.isfinite(l2) or not torch.isfinite(cosine):
        return None
    return float(l2.item()), float(cosine.item()), float(jaccard)


def _stability_perturbations(x, modes: str, clamp_min: float, clamp_max: float, noise_std: float):
    perturbed = []
    for mode in [m.strip().lower() for m in modes.split(",") if m.strip()]:
        if mode == "noise":
            x_p = x + noise_std * torch.randn_like(x)
        elif mode == "brightness":
            x_p = x + noise_std
        elif mode == "contrast":
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x_p = (x - mean) * (1.0 + noise_std) + mean
        elif mode == "blur":
            x_p = blurred_baseline(x)
        elif mode == "shift":
            x_p = torch.roll(x, shifts=(1, 1), dims=(-2, -1))
        else:
            raise ValueError(f"Unknown stability perturbation mode: {mode}")
        perturbed.append((mode, x_p.clamp(clamp_min, clamp_max).detach()))
    return perturbed


def _prediction_preserving_stability(
    model,
    x,
    target,
    baseline,
    clamp_min,
    clamp_max,
    ig_steps,
    modes,
    topk,
    noise_std,
):
    with torch.no_grad():
        clean_pred = int(model(x).argmax(dim=1).item())
    clean_attr = integrated_gradients(model, x, target, baseline, steps=ig_steps)
    total = 0
    kept = 0
    l2_values = []
    cosine_values = []
    jaccard_values = []
    for _, x_pert in _stability_perturbations(x, modes, clamp_min, clamp_max, noise_std):
        total += 1
        with torch.no_grad():
            pert_pred = int(model(x_pert).argmax(dim=1).item())
        if pert_pred != clean_pred:
            continue
        kept += 1
        pert_attr = integrated_gradients(model, x_pert, target, baseline, steps=ig_steps)
        scores = _attribution_similarity(clean_attr, pert_attr, topk=topk)
        if scores is None:
            continue
        l2, cosine, jaccard = scores
        l2_values.append(l2)
        cosine_values.append(cosine)
        jaccard_values.append(jaccard)
    return {
        "l2": _safe_mean(l2_values),
        "cosine": _safe_mean(cosine_values),
        "topk_jaccard": _safe_mean(jaccard_values),
        "kept": kept,
        "total": total,
        "valid": len(l2_values),
    }


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
    cert_eps=None,
    baseline_min=0.0,
    baseline_max=1.0,
    baseline_modes="zero,blur,noise,uniform,mean",
    stability_modes="noise,brightness,contrast,blur,shift",
    stability_topk=50,
    stability_noise_std=0.03,
):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(k, len(test_set)), replace=False)

    ent_list = []
    ads_list = []
    wads_list = []
    pp_l2_list = []
    pp_cosine_list = []
    pp_jaccard_list = []
    pp_kept = 0
    pp_total = 0
    pp_valid = 0
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
        _append_finite(
            wads_list,
            _worst_ads_baseline(
                model,
                x,
                target,
                modes=baseline_modes,
                baseline_min=baseline_min,
                baseline_max=baseline_max,
                steps=ig_steps,
            ),
        )
        pp = _prediction_preserving_stability(
            model,
            x,
            target,
            b0,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            ig_steps=ig_steps,
            modes=stability_modes,
            topk=stability_topk,
            noise_std=stability_noise_std,
        )
        _append_finite(pp_l2_list, pp["l2"])
        _append_finite(pp_cosine_list, pp["cosine"])
        _append_finite(pp_jaccard_list, pp["topk_jaccard"])
        pp_kept += int(pp["kept"])
        pp_total += int(pp["total"])
        pp_valid += int(pp["valid"])

        bound_ok = empirical_probe(model, x.squeeze(0), bound_probe_samples, eps, clamp_min, clamp_max, device)
        bound_list.append(int(bound_ok))
        active_cert_eps = eps if cert_eps is None else cert_eps
        if do_crown:
            crown_list.append(
                int(crown_ibp_certify(model, x.squeeze(0), y, active_cert_eps, device))
            )

    _report_skipped("entropy_mean", len(ent_list), len(idxs))
    _report_skipped("ads_mean", len(ads_list), len(idxs))
    _report_skipped("wads_mean", len(wads_list), len(idxs))
    _report_skipped("pp_stability_l2_mean", len(pp_l2_list), len(idxs))
    return {
        "entropy_mean": _safe_mean(ent_list),
        "ads_mean": _safe_mean(ads_list),
        "wads_mean": _safe_mean(wads_list),
        "pp_stability_l2_mean": _safe_mean(pp_l2_list),
        "pp_stability_cosine_mean": _safe_mean(pp_cosine_list),
        "pp_stability_topk_jaccard_mean": _safe_mean(pp_jaccard_list),
        "pp_stability_keep_rate": float(pp_kept / max(pp_total, 1)),
        "entropy_valid_n": int(len(ent_list)),
        "ads_valid_n": int(len(ads_list)),
        "wads_valid_n": int(len(wads_list)),
        "pp_stability_valid_n": int(pp_valid),
        "bound_check_rate": _safe_mean(bound_list),
        "crown_rate": _safe_mean(crown_list),
    }


def evaluate_certification(
    model,
    test_set,
    device,
    cert_eps,
    k=100,
    seed=0,
):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(k, len(test_set)), replace=False)
    crown_list = []

    for idx in idxs:
        x, y = test_set[idx]
        y = int(y)
        x = x.to(device).unsqueeze(0)
        crown_list.append(int(crown_ibp_certify(model, x.squeeze(0), y, cert_eps, device)))

    return {
        "cert_eps": float(cert_eps),
        "crown_rate": _safe_mean(crown_list),
        "cert_valid_n": int(len(crown_list)),
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
        b_noise = noise_baseline(x, low=baseline_min, high=baseline_max)
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
    if exists:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header != list(header):
            raise ValueError(
                "CSV schema mismatch for "
                f"{path}. Existing columns do not match this run. "
                "Use a new --out directory or remove the old CSV file."
            )
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)
