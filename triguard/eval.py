import hashlib
import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .attacks import clamp_input, pgd_linf
from .attributions import (
    allocation_ads_baseline,
    attribution_allocation_distance,
    attribution_entropy,
    blurred_baseline,
    completeness_orthogonal_distance,
    integrated_gradients,
    noise_baseline,
    smoothgrad_squared,
    uniform_baseline,
)
from .faithfulness import faithfulness_auc
from .references import sample_reference_baselines
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


def _stable_seed(seed: int, *parts) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _torch_generator(reference: torch.Tensor, seed: int, *parts) -> torch.Generator:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(_stable_seed(seed, *parts))
    return generator


def _midpoint_baseline(x, baseline_min, baseline_max):
    lower = torch.as_tensor(baseline_min, device=x.device, dtype=x.dtype)
    upper = torch.as_tensor(baseline_max, device=x.device, dtype=x.dtype)
    if lower.ndim == 1:
        lower = lower.view(1, -1, 1, 1)
        upper = upper.view(1, -1, 1, 1)
    return torch.zeros_like(x) + (lower + upper) / 2.0


def _baseline_family(
    x,
    modes: str,
    baseline_min,
    baseline_max,
    generator=None,
    reference_bank=None,
    reference_bank_samples=4,
):
    family = {}
    for mode in [m.strip().lower() for m in modes.split(",") if m.strip()]:
        if mode == "zero":
            family[mode] = torch.zeros_like(x)
        elif mode == "blur":
            family[mode] = blurred_baseline(x)
        elif mode == "noise":
            family[mode] = noise_baseline(
                x, low=baseline_min, high=baseline_max, generator=generator
            )
        elif mode == "uniform":
            family[mode] = uniform_baseline(
                x, low=baseline_min, high=baseline_max, generator=generator
            )
        elif mode in {"midpoint", "mean"}:
            key = "midpoint" if mode == "mean" else mode
            family[key] = _midpoint_baseline(x, baseline_min, baseline_max)
        elif mode == "bank":
            if reference_bank is None:
                raise ValueError("baseline mode 'bank' requires a reference bank.")
            family.update(
                sample_reference_baselines(
                    x,
                    reference_bank,
                    reference_bank_samples,
                    generator=generator,
                )
            )
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")
    return family


def _baseline_drift_metrics(
    model,
    x,
    target,
    modes,
    baseline_min,
    baseline_max,
    steps,
    generator=None,
    reference_bank=None,
    reference_bank_samples=4,
):
    family = _baseline_family(
        x,
        modes,
        baseline_min,
        baseline_max,
        generator=generator,
        reference_bank=reference_bank,
        reference_bank_samples=reference_bank_samples,
    )
    if len(family) < 2:
        return None
    attrs = {
        name: integrated_gradients(model, x, target, baseline, steps=steps)
        for name, baseline in family.items()
    }
    with torch.no_grad():
        baselines = torch.cat(list(family.values()), dim=0)
        scores = model(baselines)[:, int(target)]

    allocation_values = []
    raw_l2_values = []
    raw_rms_values = []
    orthogonal_rms_values = []
    output_gap_values = []
    names = list(attrs)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            difference = (attrs[left] - attrs[right]).flatten()
            raw_l2_values.append(torch.norm(difference, p=2))
            raw_rms_values.append(difference.square().mean().sqrt())
            orthogonal_rms_values.append(
                completeness_orthogonal_distance(attrs[left], attrs[right])[0]
            )
            allocation_values.append(
                attribution_allocation_distance(attrs[left], attrs[right])[0]
            )
            left_idx = names.index(left)
            right_idx = names.index(right)
            output_gap_values.append((scores[left_idx] - scores[right_idx]).abs())

    stacked = torch.stack(
        allocation_values
        + raw_l2_values
        + raw_rms_values
        + orthogonal_rms_values
        + output_gap_values
    )
    if not torch.isfinite(stacked).all():
        return None
    attribution_masses = torch.stack(
        [attrs[name].flatten().abs().sum() for name in names]
    )
    with torch.no_grad():
        input_score = model(x)[0, int(target)]
    output_changes = (input_score - scores).abs()
    active_mass_ratios = attribution_masses[output_changes > 1e-6] / output_changes[
        output_changes > 1e-6
    ].clamp_min(1e-8)
    completeness_abs = torch.stack(
        [
            (attrs[name].sum() - (input_score - scores[index])).abs()
            for index, name in enumerate(names)
        ]
    )
    completeness_relative = torch.stack(
        [
            completeness_abs[index]
            / (input_score - scores[index]).abs().clamp_min(1e-8)
            for index in range(len(names))
        ]
    )
    return {
        "wads": float(torch.stack(allocation_values).max().item()),
        "raw_wads_l2": float(torch.stack(raw_l2_values).max().item()),
        "raw_wads_rms": float(torch.stack(raw_rms_values).max().item()),
        "orthogonal_wads_rms": float(
            torch.stack(orthogonal_rms_values).max().item()
        ),
        "baseline_output_gap": float(torch.stack(output_gap_values).max().item()),
        "baseline_attr_mass_min": float(attribution_masses.min().item()),
        "baseline_attr_mass_ratio_min": (
            float(active_mass_ratios.min().item())
            if active_mass_ratios.numel()
            else float("nan")
        ),
        "completeness_abs_max": float(completeness_abs.max().item()),
        "completeness_relative_max": float(completeness_relative.max().item()),
    }


def _attribution_similarity(clean_attr, pert_attr, topk_fraction: float):
    clean = clean_attr.detach().flatten()
    pert = pert_attr.detach().flatten()
    if clean.numel() == 0 or pert.numel() == 0:
        return None
    l2 = torch.norm(clean - pert, p=2)
    rms = (clean - pert).square().mean().sqrt()
    cosine = F.cosine_similarity(clean.view(1, -1), pert.view(1, -1), dim=1)[0]
    clean_pixels = clean_attr.detach().abs().sum(dim=1).flatten()
    pert_pixels = pert_attr.detach().abs().sum(dim=1).flatten()
    k = min(
        max(int(round(float(topk_fraction) * clean_pixels.numel())), 1),
        clean_pixels.numel(),
    )
    clean_idx = torch.topk(clean_pixels, k=k).indices
    pert_idx = torch.topk(pert_pixels, k=k).indices
    clean_set = set(clean_idx.detach().cpu().tolist())
    pert_set = set(pert_idx.detach().cpu().tolist())
    union = clean_set | pert_set
    jaccard = len(clean_set & pert_set) / max(len(union), 1)
    if not torch.isfinite(l2) or not torch.isfinite(cosine):
        return None
    return float(l2.item()), float(rms.item()), float(cosine.item()), float(jaccard)


def _stability_perturbations(
    x,
    modes: str,
    clamp_min,
    clamp_max,
    noise_std: float,
    generator=None,
):
    perturbed = []
    for mode in [m.strip().lower() for m in modes.split(",") if m.strip()]:
        if mode == "noise":
            noise = torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
            x_p = x + noise_std * noise
        elif mode == "brightness":
            x_p = x + noise_std
        elif mode == "contrast":
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x_p = (x - mean) * (1.0 + noise_std) + mean
        elif mode == "blur":
            x_p = blurred_baseline(x)
        elif mode == "shift":
            height, width = x.shape[-2:]
            x_p = F.pad(x, (1, 0, 1, 0), mode="replicate")[
                ..., :height, :width
            ]
        else:
            raise ValueError(f"Unknown stability perturbation mode: {mode}")
        perturbed.append((mode, clamp_input(x_p, clamp_min, clamp_max).detach()))
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
    topk_fraction,
    noise_std,
    generator=None,
):
    with torch.no_grad():
        clean_pred = int(model(x).argmax(dim=1).item())
    clean_attr = integrated_gradients(model, x, target, baseline, steps=ig_steps)
    total = 0
    kept = 0
    l2_values = []
    rms_values = []
    cosine_values = []
    jaccard_values = []
    for _, x_pert in _stability_perturbations(
        x, modes, clamp_min, clamp_max, noise_std, generator=generator
    ):
        total += 1
        with torch.no_grad():
            pert_pred = int(model(x_pert).argmax(dim=1).item())
        if pert_pred != clean_pred:
            continue
        kept += 1
        pert_attr = integrated_gradients(model, x_pert, target, baseline, steps=ig_steps)
        scores = _attribution_similarity(
            clean_attr, pert_attr, topk_fraction=topk_fraction
        )
        if scores is None:
            continue
        l2, rms, cosine, jaccard = scores
        l2_values.append(l2)
        rms_values.append(rms)
        cosine_values.append(cosine)
        jaccard_values.append(jaccard)
    return {
        "l2": _safe_mean(l2_values),
        "rms": _safe_mean(rms_values),
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


class _PixelSpaceModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, pixels):
        return self.model((pixels - self.mean) / self.std)


def autoattack_accuracy(
    model,
    loader,
    device,
    pixel_eps,
    normalization_mean,
    normalization_std,
    max_samples=1000,
    batch_size=128,
    seed=0,
):
    """Run official standard AutoAttack in [0,1] pixel coordinates."""
    try:
        from autoattack import AutoAttack
    except ImportError as exc:
        raise RuntimeError(
            "AutoAttack is required for --attack_suite autoattack. "
            "Install requirements-autoattack.txt."
        ) from exc

    model.eval()
    mean = torch.as_tensor(
        normalization_mean, device=device, dtype=torch.float32
    ).view(1, -1, 1, 1)
    std = torch.as_tensor(
        normalization_std, device=device, dtype=torch.float32
    ).view(1, -1, 1, 1)
    images = []
    labels = []
    remaining = max(int(max_samples), 1)
    for x, y in loader:
        take = min(remaining, y.numel())
        x = x[:take].to(device)
        images.append((x * std + mean).clamp(0.0, 1.0))
        labels.append(y[:take].to(device))
        remaining -= take
        if remaining <= 0:
            break
    x_pixel = torch.cat(images, dim=0)
    y_all = torch.cat(labels, dim=0)
    pixel_model = _PixelSpaceModel(
        model, normalization_mean, normalization_std
    ).to(device).eval()
    adversary = AutoAttack(
        pixel_model,
        norm="Linf",
        eps=float(pixel_eps),
        seed=int(seed),
        version="standard",
        device=device,
    )
    x_adv = adversary.run_standard_evaluation(
        x_pixel, y_all, bs=min(int(batch_size), y_all.numel())
    )
    with torch.no_grad():
        correct = pixel_model(x_adv).argmax(dim=1).eq(y_all).sum().item()
    return float(correct / max(y_all.numel(), 1)), int(y_all.numel())


def pgd_accuracy(
    model,
    loader,
    device,
    eps,
    alpha,
    steps,
    clamp_min,
    clamp_max,
    max_batches=10,
    restarts=5,
    seed=0,
):
    model.eval()
    correct = 0
    total = 0
    seen = 0
    for batch_index, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            robust = model(x).argmax(dim=1).eq(y)
        for restart in range(max(int(restarts), 1)):
            generator = _torch_generator(
                x, seed, "pgd_accuracy", batch_index, restart
            )
            x_adv = pgd_linf(
                model,
                x,
                y,
                eps,
                alpha,
                steps,
                clamp_min,
                clamp_max,
                random_start=True,
                generator=generator,
            )
            with torch.no_grad():
                robust &= model(x_adv).argmax(dim=1).eq(y)
        correct += robust.sum().item()
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
    target_mode="pred",
    empirical_probe_samples=16,
    do_crown=True,
    cert_eps=None,
    baseline_min=0.0,
    baseline_max=1.0,
    baseline_modes="zero,blur,noise,uniform,midpoint",
    stability_modes="noise,brightness,contrast,blur,shift",
    stability_topk_fraction=0.05,
    stability_noise_std=0.03,
    reference_bank=None,
    heldout_reference_bank=None,
    reference_bank_samples=4,
    attr_mass_floor=0.9,
):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(k, len(test_set)), replace=False)

    ent_list = []
    ent_raw_list = []
    ads_list = []
    ads_raw_l2_list = []
    ads_raw_rms_list = []
    ads_orthogonal_rms_list = []
    ads_output_gap_list = []
    wads_list = []
    raw_wads_l2_list = []
    raw_wads_rms_list = []
    orthogonal_wads_rms_list = []
    baseline_output_gap_list = []
    baseline_attr_mass_min_list = []
    baseline_attr_mass_ratio_min_list = []
    heldout_wads_list = []
    heldout_orthogonal_rms_list = []
    heldout_attr_mass_list = []
    heldout_attr_mass_ratio_list = []
    completeness_error_list = []
    completeness_relative_list = []
    completeness_abs_max_list = []
    completeness_relative_max_list = []
    pp_l2_list = []
    pp_rms_list = []
    pp_cosine_list = []
    pp_jaccard_list = []
    pp_kept = 0
    pp_total = 0
    pp_valid = 0
    probe_list = []
    probe_violation_list = []
    probe_margin_list = []
    crown_list = []
    crown_error_count = 0
    crown_error_types = Counter()

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
        ig_blur = integrated_gradients(model, x, target, b_blur, steps=ig_steps)
        _append_finite(
            ent_list,
            attribution_entropy(ig, normalized=True, pixel_level=True),
        )
        _append_finite(
            ent_raw_list,
            attribution_entropy(ig, normalized=False, pixel_level=True),
        )
        fixed_allocation = attribution_allocation_distance(ig, ig_blur)[0]
        fixed_difference = (ig - ig_blur).flatten()
        fixed_raw_l2 = torch.norm(fixed_difference, p=2)
        fixed_raw_rms = fixed_difference.square().mean().sqrt()
        fixed_orthogonal_rms = completeness_orthogonal_distance(ig, ig_blur)[0]
        with torch.no_grad():
            input_score = model(x)[0, target]
            zero_score = model(b0)[0, target]
            blur_score = model(b_blur)[0, target]
        _append_finite(ads_list, float(fixed_allocation.item()))
        _append_finite(ads_raw_l2_list, float(fixed_raw_l2.item()))
        _append_finite(ads_raw_rms_list, float(fixed_raw_rms.item()))
        _append_finite(
            ads_orthogonal_rms_list, float(fixed_orthogonal_rms.item())
        )
        _append_finite(ads_output_gap_list, float((zero_score - blur_score).abs().item()))
        _append_finite(
            completeness_error_list,
            float((ig.sum() - (input_score - zero_score)).abs().item()),
        )
        _append_finite(
            completeness_relative_list,
            float(
                (
                    (ig.sum() - (input_score - zero_score)).abs()
                    / (input_score - zero_score).abs().clamp_min(1e-8)
                ).item()
            ),
        )
        baseline_drift = _baseline_drift_metrics(
            model,
            x,
            target,
            modes=baseline_modes,
            baseline_min=baseline_min,
            baseline_max=baseline_max,
            steps=ig_steps,
            generator=_torch_generator(x, seed, "baseline_family", int(idx)),
            reference_bank=reference_bank,
            reference_bank_samples=reference_bank_samples,
        )
        if baseline_drift is not None:
            _append_finite(wads_list, baseline_drift["wads"])
            _append_finite(raw_wads_l2_list, baseline_drift["raw_wads_l2"])
            _append_finite(raw_wads_rms_list, baseline_drift["raw_wads_rms"])
            _append_finite(
                orthogonal_wads_rms_list,
                baseline_drift["orthogonal_wads_rms"],
            )
            _append_finite(
                baseline_output_gap_list, baseline_drift["baseline_output_gap"]
            )
            _append_finite(
                baseline_attr_mass_min_list,
                baseline_drift["baseline_attr_mass_min"],
            )
            _append_finite(
                baseline_attr_mass_ratio_min_list,
                baseline_drift["baseline_attr_mass_ratio_min"],
            )
            _append_finite(
                completeness_abs_max_list,
                baseline_drift["completeness_abs_max"],
            )
            _append_finite(
                completeness_relative_max_list,
                baseline_drift["completeness_relative_max"],
            )
        if heldout_reference_bank is not None:
            heldout_drift = _baseline_drift_metrics(
                model,
                x,
                target,
                modes="bank",
                baseline_min=baseline_min,
                baseline_max=baseline_max,
                steps=ig_steps,
                generator=_torch_generator(
                    x, seed, "heldout_reference_bank", int(idx)
                ),
                reference_bank=heldout_reference_bank,
                reference_bank_samples=reference_bank_samples,
            )
            if heldout_drift is not None:
                _append_finite(heldout_wads_list, heldout_drift["wads"])
                _append_finite(
                    heldout_orthogonal_rms_list,
                    heldout_drift["orthogonal_wads_rms"],
                )
                _append_finite(
                    heldout_attr_mass_list,
                    heldout_drift["baseline_attr_mass_min"],
                )
                _append_finite(
                    heldout_attr_mass_ratio_list,
                    heldout_drift["baseline_attr_mass_ratio_min"],
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
            topk_fraction=stability_topk_fraction,
            noise_std=stability_noise_std,
            generator=_torch_generator(x, seed, "stability", int(idx)),
        )
        _append_finite(pp_l2_list, pp["l2"])
        _append_finite(pp_rms_list, pp["rms"])
        _append_finite(pp_cosine_list, pp["cosine"])
        _append_finite(pp_jaccard_list, pp["topk_jaccard"])
        pp_kept += int(pp["kept"])
        pp_total += int(pp["total"])
        pp_valid += int(pp["valid"])

        probe = empirical_probe(
            model,
            x.squeeze(0),
            empirical_probe_samples,
            eps,
            clamp_min,
            clamp_max,
            device,
            generator=_torch_generator(x, seed, "empirical_probe", int(idx)),
            return_details=True,
        )
        probe_list.append(int(probe["passed"]))
        _append_finite(probe_violation_list, probe["violation_fraction"])
        _append_finite(probe_margin_list, probe["minimum_margin"])
        active_cert_eps = eps if cert_eps is None else cert_eps
        if do_crown:
            certified, status = crown_ibp_certify(
                model,
                x.squeeze(0),
                y,
                active_cert_eps,
                device,
                return_status=True,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
            if status == "ok":
                crown_list.append(int(certified))
            else:
                crown_error_count += 1
                crown_error_types[status] += 1

    _report_skipped("entropy_mean", len(ent_list), len(idxs))
    _report_skipped("ads_mean", len(ads_list), len(idxs))
    _report_skipped("wads_mean", len(wads_list), len(idxs))
    _report_skipped("pp_stability_l2_mean", len(pp_l2_list), len(idxs))
    if crown_error_count:
        print(
            f"[Verifier Warning] CROWN failed on {crown_error_count}/{len(idxs)} samples; "
            "failures count as not proven in crown_proven_rate and are itemized."
        )
    return {
        "entropy_mean": _safe_mean(ent_list),
        "entropy_raw_mean": _safe_mean(ent_raw_list),
        "ads_mean": _safe_mean(ads_list),
        "ads_raw_l2_mean": _safe_mean(ads_raw_l2_list),
        "ads_raw_rms_mean": _safe_mean(ads_raw_rms_list),
        "ads_orthogonal_rms_mean": _safe_mean(ads_orthogonal_rms_list),
        "ads_output_gap_mean": _safe_mean(ads_output_gap_list),
        "wads_mean": _safe_mean(wads_list),
        "raw_wads_l2_mean": _safe_mean(raw_wads_l2_list),
        "raw_wads_rms_mean": _safe_mean(raw_wads_rms_list),
        "orthogonal_wads_rms_mean": _safe_mean(orthogonal_wads_rms_list),
        "baseline_output_gap_mean": _safe_mean(baseline_output_gap_list),
        "baseline_attr_mass_min_mean": _safe_mean(baseline_attr_mass_min_list),
        "baseline_attr_mass_q05": (
            float(np.quantile(baseline_attr_mass_min_list, 0.05))
            if baseline_attr_mass_min_list
            else float("nan")
        ),
        "baseline_attr_mass_near_zero_rate": (
            float(np.mean(np.asarray(baseline_attr_mass_min_list) <= 1e-10))
            if baseline_attr_mass_min_list
            else float("nan")
        ),
        "baseline_attr_mass_ratio_min_mean": _safe_mean(
            baseline_attr_mass_ratio_min_list
        ),
        "baseline_attr_mass_ratio_q05": (
            float(np.quantile(baseline_attr_mass_ratio_min_list, 0.05))
            if baseline_attr_mass_ratio_min_list
            else float("nan")
        ),
        "baseline_attr_mass_ratio_below_floor_rate": (
            float(
                np.mean(
                    np.asarray(baseline_attr_mass_ratio_min_list)
                    < float(attr_mass_floor)
                )
            )
            if baseline_attr_mass_ratio_min_list
            else float("nan")
        ),
        "heldout_wads_mean": _safe_mean(heldout_wads_list),
        "heldout_orthogonal_wads_rms_mean": _safe_mean(
            heldout_orthogonal_rms_list
        ),
        "heldout_attr_mass_min_mean": _safe_mean(heldout_attr_mass_list),
        "heldout_attr_mass_ratio_min_mean": _safe_mean(
            heldout_attr_mass_ratio_list
        ),
        "heldout_attr_mass_ratio_q05": (
            float(np.quantile(heldout_attr_mass_ratio_list, 0.05))
            if heldout_attr_mass_ratio_list
            else float("nan")
        ),
        "heldout_attr_mass_ratio_below_floor_rate": (
            float(
                np.mean(
                    np.asarray(heldout_attr_mass_ratio_list)
                    < float(attr_mass_floor)
                )
            )
            if heldout_attr_mass_ratio_list
            else float("nan")
        ),
        "heldout_reference_valid_n": int(len(heldout_wads_list)),
        "ig_completeness_error_mean": _safe_mean(completeness_error_list),
        "ig_completeness_relative_error_mean": _safe_mean(
            completeness_relative_list
        ),
        "ig_completeness_abs_max_mean": _safe_mean(completeness_abs_max_list),
        "ig_completeness_relative_max_mean": _safe_mean(
            completeness_relative_max_list
        ),
        "pp_stability_l2_mean": _safe_mean(pp_l2_list),
        "pp_stability_rms_mean": _safe_mean(pp_rms_list),
        "pp_stability_cosine_mean": _safe_mean(pp_cosine_list),
        "pp_stability_topk_jaccard_mean": _safe_mean(pp_jaccard_list),
        "pp_stability_keep_rate": float(pp_kept / max(pp_total, 1)),
        "entropy_valid_n": int(len(ent_list)),
        "ads_valid_n": int(len(ads_list)),
        "wads_valid_n": int(len(wads_list)),
        "pp_stability_valid_n": int(pp_valid),
        "empirical_probe_rate": _safe_mean(probe_list),
        "empirical_probe_violation_rate": _safe_mean(probe_violation_list),
        "empirical_probe_min_margin_mean": _safe_mean(probe_margin_list),
        "crown_rate": (
            float(sum(crown_list) / len(idxs)) if do_crown and len(idxs) else float("nan")
        ),
        "crown_proven_rate": (
            float(sum(crown_list) / len(idxs)) if do_crown and len(idxs) else float("nan")
        ),
        "crown_conditional_rate": _safe_mean(crown_list),
        "crown_attempted_n": int(len(idxs) if do_crown else 0),
        "crown_certified_n": int(sum(crown_list)),
        "crown_valid_n": int(len(crown_list)),
        "crown_error_n": int(crown_error_count),
        "crown_error_types": json.dumps(crown_error_types, sort_keys=True),
    }


def evaluate_certification(
    model,
    test_set,
    device,
    cert_eps,
    clamp_min,
    clamp_max,
    k=100,
    seed=0,
):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(k, len(test_set)), replace=False)
    crown_list = []
    crown_error_count = 0
    crown_error_types = Counter()

    for idx in idxs:
        x, y = test_set[idx]
        y = int(y)
        x = x.to(device).unsqueeze(0)
        certified, status = crown_ibp_certify(
            model,
            x.squeeze(0),
            y,
            cert_eps,
            device,
            return_status=True,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        if status == "ok":
            crown_list.append(int(certified))
        else:
            crown_error_count += 1
            crown_error_types[status] += 1

    if crown_error_count:
        print(
            f"[Verifier Warning] CROWN failed on {crown_error_count}/{len(idxs)} samples; "
            "failures count as not proven in crown_proven_rate and are itemized."
        )

    proven_rate = float(sum(crown_list) / len(idxs)) if len(idxs) else float("nan")
    return {
        "cert_eps": cert_eps,
        "crown_rate": proven_rate,
        "crown_proven_rate": proven_rate,
        "crown_conditional_rate": _safe_mean(crown_list),
        "cert_attempted_n": int(len(idxs)),
        "cert_certified_n": int(sum(crown_list)),
        "cert_valid_n": int(len(crown_list)),
        "cert_error_n": int(crown_error_count),
        "cert_error_types": json.dumps(crown_error_types, sort_keys=True),
    }


def evaluate_appendix_metrics(
    model,
    test_set,
    device,
    ig_steps=50,
    k=100,
    seed=0,
    target_mode="pred",
    baseline_min=0.0,
    baseline_max=1.0,
):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(k, len(test_set)), replace=False)

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
        generator = _torch_generator(x, seed, "appendix_baselines", int(idx))
        b_noise = noise_baseline(
            x, low=baseline_min, high=baseline_max, generator=generator
        )
        b_uniform = uniform_baseline(
            x, low=baseline_min, high=baseline_max, generator=generator
        )

        row = {
            "ads_zero_blur": allocation_ads_baseline(model, x, target, b0, b_blur, steps=ig_steps),
            "ads_zero_noise": allocation_ads_baseline(model, x, target, b0, b_noise, steps=ig_steps),
            "ads_zero_uniform": allocation_ads_baseline(model, x, target, b0, b_uniform, steps=ig_steps),
            "ads_blur_noise": allocation_ads_baseline(model, x, target, b_blur, b_noise, steps=ig_steps),
            "ads_blur_uniform": allocation_ads_baseline(model, x, target, b_blur, b_uniform, steps=ig_steps),
            "ads_noise_uniform": allocation_ads_baseline(model, x, target, b_noise, b_uniform, steps=ig_steps),
        }
        if all(value is not None and np.isfinite(value) for value in row.values()):
            base_rows.append({key: float(value) for key, value in row.items()})

    _report_skipped("baseline sensitivity metrics", len(base_rows), len(idxs))
    result = {}
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
                          seed=0, baseline_mode="blur", target_mode="pred",
                          smoothgrad_noise=0.1, smoothgrad_samples=50,
                          clamp_min=0.0, clamp_max=1.0,
                          baseline_min=0.0, baseline_max=1.0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_set), size=min(k, len(test_set)), replace=False)

    ig_del, ig_ins = [], []
    sg_del, sg_ins = [], []
    random_del, random_ins = [], []
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

        baseline = next(
            iter(
                _baseline_family(
                    x,
                    baseline_mode,
                    baseline_min,
                    baseline_max,
                    generator=_torch_generator(
                        x, seed, "faithfulness_baseline", int(idx)
                    ),
                ).values()
            )
        )
        ig_attr = integrated_gradients(model, x, target, baseline, steps=ig_steps)
        ig_d, ig_i, del_curve, ins_curve = faithfulness_auc(model, x, target, ig_attr, steps=delins_steps, baseline=baseline)
        _append_finite(ig_del, ig_d)
        _append_finite(ig_ins, ig_i)

        sg_attr = smoothgrad_squared(
            model,
            x,
            target,
            noise_level=smoothgrad_noise,
            n_samples=smoothgrad_samples,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            generator=_torch_generator(x, seed, "smoothgrad", int(idx)),
        )
        sg_d, sg_i, del_curve2, ins_curve2 = faithfulness_auc(model, x, target, sg_attr, steps=delins_steps, baseline=baseline)
        _append_finite(sg_del, sg_d)
        _append_finite(sg_ins, sg_i)

        random_attr = torch.rand(
            x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=_torch_generator(x, seed, "random_ranking", int(idx)),
        )
        random_d, random_i, _, _ = faithfulness_auc(
            model,
            x,
            target,
            random_attr,
            steps=delins_steps,
            baseline=baseline,
        )
        _append_finite(random_del, random_d)
        _append_finite(random_ins, random_i)

        if t < keep_curves:
            curves.append(("IG", del_curve, ins_curve))
            curves.append(("SmoothGrad2", del_curve2, ins_curve2))

    _report_skipped("ig_del_auc_mean", len(ig_del), len(idxs))
    _report_skipped("ig_ins_auc_mean", len(ig_ins), len(idxs))
    _report_skipped("sg2_del_auc_mean", len(sg_del), len(idxs))
    _report_skipped("sg2_ins_auc_mean", len(sg_ins), len(idxs))
    _report_skipped("random_del_auc_mean", len(random_del), len(idxs))
    _report_skipped("random_ins_auc_mean", len(random_ins), len(idxs))

    return {
        "ig_del_auc_mean": _safe_mean(ig_del),
        "ig_ins_auc_mean": _safe_mean(ig_ins),
        "sg2_del_auc_mean": _safe_mean(sg_del),
        "sg2_ins_auc_mean": _safe_mean(sg_ins),
        "random_del_auc_mean": _safe_mean(random_del),
        "random_ins_auc_mean": _safe_mean(random_ins),
        "ig_valid_n": int(min(len(ig_del), len(ig_ins))),
        "sg2_valid_n": int(min(len(sg_del), len(sg_ins))),
        "random_valid_n": int(min(len(random_del), len(random_ins))),
        "curves": curves,
    }
