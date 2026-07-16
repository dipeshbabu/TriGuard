from contextlib import contextmanager, nullcontext
from itertools import combinations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attacks import _as_channel_tensor, _clamp_delta, clamp_input, uniform_like
from .attributions import (
    attribution_allocation_distance,
    completeness_orthogonal_distance,
)
from .references import sample_reference_baselines


@contextmanager
def _inference_mode_layers(module):
    """Disable dropout/BatchNorm updates while preserving parameter gradients."""
    states = [(submodule, submodule.training) for submodule in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for submodule, training in states:
            submodule.training = training


def _sdp_math_context(device, require_higher_order: bool):
    if device.type != "cuda" or not require_higher_order:
        return nullcontext()

    attention_mod = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attention_mod, "sdpa_kernel", None)
    sdp_backend = getattr(attention_mod, "SDPBackend", None)
    if sdpa_kernel is not None and sdp_backend is not None:
        return sdpa_kernel(backends=[sdp_backend.MATH])

    cuda_backends = getattr(torch.backends, "cuda", None)
    legacy_sdp_kernel = getattr(cuda_backends, "sdp_kernel", None)
    if legacy_sdp_kernel is not None:
        return legacy_sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

    return nullcontext()


def entropy_reg_term(model, x, y, eps=1e-10):
    x = x.detach().requires_grad_(True)
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(ce, x, create_graph=True, retain_graph=True)[0]
    g = grad.abs().reshape(grad.size(0), -1)
    p = g / (g.sum(dim=1, keepdim=True) + eps)
    H = -(p * torch.log(p + eps)).sum(dim=1).mean()
    return H


def _rms_flat(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(x.flatten(1).pow(2).mean(dim=1) + eps)


def _smooth_baseline(x: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
    if kernel_size <= 1:
        return x.detach()
    height, width = x.shape[-2:]
    kernel_size = min(int(kernel_size), height, width)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size <= 1:
        return x.detach()
    pad = kernel_size // 2
    mode = "reflect" if pad < min(height, width) else "replicate"
    padded = F.pad(x.detach(), (pad,) * 4, mode=mode)
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def make_baseline_family(
    x: torch.Tensor,
    modes: str,
    baseline_min: float,
    baseline_max: float,
    reference_bank: torch.Tensor | None = None,
    reference_bank_samples: int = 4,
):
    family = {}
    for mode in [m.strip().lower() for m in modes.split(",") if m.strip()]:
        if mode == "zero":
            family[mode] = torch.zeros_like(x)
        elif mode == "blur":
            family[mode] = _smooth_baseline(x)
        elif mode == "noise":
            family[mode] = clamp_input(
                torch.randn_like(x) * 0.1, baseline_min, baseline_max
            )
        elif mode == "uniform":
            family[mode] = uniform_like(x, baseline_min, baseline_max)
        elif mode in {"midpoint", "mean"}:
            lower = torch.as_tensor(baseline_min, device=x.device, dtype=x.dtype)
            upper = torch.as_tensor(baseline_max, device=x.device, dtype=x.dtype)
            if lower.ndim == 1:
                lower = lower.view(1, -1, 1, 1)
                upper = upper.view(1, -1, 1, 1)
            key = "midpoint" if mode == "mean" else mode
            family[key] = torch.zeros_like(x) + (lower + upper) / 2.0
        elif mode == "bank":
            if reference_bank is None:
                raise ValueError("baseline mode 'bank' requires --reference_bank.")
            family.update(
                sample_reference_baselines(
                    x, reference_bank, reference_bank_samples
                )
            )
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")
    if len(family) < 2:
        raise ValueError("TriGuard-Train requires at least two baseline modes.")
    return family


def differentiable_integrated_gradients(
    model,
    x: torch.Tensor,
    targets: torch.Tensor,
    baseline: torch.Tensor,
    steps: int = 8,
    checkpoint_activations: bool = False,
):
    steps = max(int(steps), 1)
    batch = x.size(0)
    alphas = torch.linspace(
        0.0, 1.0, steps + 1, device=x.device, dtype=x.dtype
    )[1:].view(steps, 1, 1, 1, 1)
    path = baseline.unsqueeze(0) + alphas * (x.unsqueeze(0) - baseline.unsqueeze(0))
    path = path.reshape(steps * batch, *x.shape[1:]).requires_grad_(True)

    repeated_targets = targets.repeat(steps)
    logits = (
        checkpoint(model, path, use_reentrant=False)
        if checkpoint_activations
        else model(path)
    )
    score = logits.gather(1, repeated_targets.view(-1, 1)).sum()
    grad = torch.autograd.grad(score, path, create_graph=True, retain_graph=True)[0]
    grad = grad.reshape(steps, batch, *x.shape[1:])
    avg_grad = grad.mean(dim=0)
    return (x - baseline) * avg_grad


def differentiable_integrated_gradients_many(
    model,
    x: torch.Tensor,
    targets: torch.Tensor,
    baselines: dict[str, torch.Tensor],
    steps: int = 8,
    checkpoint_activations: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute IG for several references in one model traversal.

    Regularizer microbatching keeps the combined path tensor small, while
    vectorizing references substantially improves accelerator utilization.
    """
    if not baselines:
        raise ValueError("At least one baseline is required.")
    steps = max(int(steps), 1)
    names = list(baselines)
    stacked = torch.stack([baselines[name] for name in names], dim=0)
    reference_count, batch = stacked.shape[:2]
    alphas = torch.linspace(
        0.0, 1.0, steps + 1, device=x.device, dtype=x.dtype
    )[1:].view(steps, 1, 1, 1, 1, 1)
    path = stacked.unsqueeze(0) + alphas * (
        x.unsqueeze(0).unsqueeze(0) - stacked.unsqueeze(0)
    )
    path = path.reshape(
        steps * reference_count * batch, *x.shape[1:]
    ).requires_grad_(True)
    repeated_targets = targets.repeat(steps * reference_count)
    logits = (
        checkpoint(model, path, use_reentrant=False)
        if checkpoint_activations
        else model(path)
    )
    score = logits.gather(1, repeated_targets.view(-1, 1)).sum()
    grad = torch.autograd.grad(
        score, path, create_graph=True, retain_graph=True
    )[0]
    avg_grad = grad.reshape(
        steps, reference_count, batch, *x.shape[1:]
    ).mean(dim=0)
    attrs = (x.unsqueeze(0) - stacked) * avg_grad
    return {name: attrs[index] for index, name in enumerate(names)}


def _aggregate_reference_risk(
    pair_losses: torch.Tensor,
    risk: str,
    cvar_alpha: float,
) -> torch.Tensor:
    risk = risk.lower()
    if risk == "mean":
        per_sample = pair_losses.mean(dim=0)
    elif risk == "max":
        per_sample = pair_losses.max(dim=0).values
    elif risk == "cvar":
        if not 0.0 <= cvar_alpha < 1.0:
            raise ValueError("reference_cvar_alpha must lie in [0, 1).")
        tail_count = max(
            int(torch.ceil(torch.tensor((1.0 - cvar_alpha) * pair_losses.size(0))).item()),
            1,
        )
        per_sample = pair_losses.topk(tail_count, dim=0).values.mean(dim=0)
    else:
        raise ValueError(f"Unknown reference risk: {risk}")
    return per_sample.mean()


def baseline_regularization_terms(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    baseline_modes: str,
    baseline_min: float,
    baseline_max: float,
    ig_steps: int = 8,
    reference_risk: str = "max",
    reference_cvar_alpha: float = 0.75,
    reference_pair_samples: int = 0,
    reference_distance: str = "allocation",
    reference_bank: torch.Tensor | None = None,
    reference_bank_samples: int = 4,
    attr_mass_floor: float = 0.9,
    compute_mass_penalty: bool = True,
    sampled_mass_penalty: bool = False,
    vectorized_reference_ig: bool = False,
    checkpoint_activations: bool = False,
):
    family = make_baseline_family(
        x,
        baseline_modes,
        baseline_min,
        baseline_max,
        reference_bank=reference_bank,
        reference_bank_samples=reference_bank_samples,
    )
    names = list(family)
    pairs = list(combinations(names, 2))
    if 0 < reference_pair_samples < len(pairs):
        # Pair selection is control flow, so keep it on CPU. Converting CUDA
        # scalars to Python integers here would synchronize the accelerator
        # once per selected pair and per regularizer microbatch.
        selected = torch.randperm(len(pairs))[:reference_pair_samples].tolist()
        pairs = [pairs[index] for index in selected]
    active_names = {name for pair in pairs for name in pair}
    if compute_mass_penalty and not sampled_mass_penalty:
        active_names.update(names)
    active_family = {
        name: family[name] for name in names if name in active_names
    }
    if vectorized_reference_ig:
        attributions = differentiable_integrated_gradients_many(
            model,
            x,
            y,
            active_family,
            steps=ig_steps,
            checkpoint_activations=checkpoint_activations,
        )
    else:
        attributions = {
            name: differentiable_integrated_gradients(
                model,
                x,
                y,
                baseline,
                steps=ig_steps,
                checkpoint_activations=checkpoint_activations,
            )
            for name, baseline in active_family.items()
        }
    pair_losses = []
    for left, right in pairs:
        if reference_distance == "allocation":
            pair_loss = attribution_allocation_distance(
                attributions[left], attributions[right]
            )
        elif reference_distance == "orthogonal_rms":
            pair_loss = completeness_orthogonal_distance(
                attributions[left], attributions[right]
            )
        else:
            raise ValueError(f"Unknown reference distance: {reference_distance}")
        pair_losses.append(pair_loss)
    reference_loss = _aggregate_reference_risk(
        torch.stack(pair_losses, dim=0), reference_risk, reference_cvar_alpha
    )

    if not compute_mass_penalty:
        return reference_loss, x.new_zeros(())

    mass_names = (
        [name for name in names if name in active_names]
        if sampled_mass_penalty
        else names
    )
    mass_inputs = torch.cat(
        [x, *(family[name] for name in mass_names)],
        dim=0,
    )
    mass_logits = (
        checkpoint(model, mass_inputs, use_reentrant=False)
        if checkpoint_activations
        else model(mass_inputs)
    )
    mass_scores = mass_logits.gather(
        1,
        y.repeat(len(mass_names) + 1).view(-1, 1),
    ).reshape(len(mass_names) + 1, x.size(0))
    output_changes = (mass_scores[0].unsqueeze(0) - mass_scores[1:]).abs()
    attribution_masses = torch.stack(
        [
            attributions[name].flatten(1).abs().sum(dim=1)
            for name in mass_names
        ],
        dim=0,
    )
    relative_masses = attribution_masses / output_changes.clamp_min(1e-8)
    active = output_changes.detach() > 1e-6
    penalties = (
        F.relu(float(attr_mass_floor) - relative_masses)
        * active.to(relative_masses.dtype)
    )
    # Average within each example before averaging the batch. This gives every
    # example equal weight and makes regularizer microbatching exactly preserve
    # the full-batch objective when the sampled references are held fixed.
    per_sample_penalty = penalties.sum(dim=0) / active.sum(dim=0).clamp_min(1)
    mass_penalty = per_sample_penalty.mean()
    return reference_loss, mass_penalty


def worst_baseline_drift_term(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    baseline_modes: str,
    baseline_min: float,
    baseline_max: float,
    ig_steps: int = 8,
    **kwargs,
):
    """Backward-compatible scalar reference-risk objective."""
    reference_loss, _ = baseline_regularization_terms(
        model,
        x,
        y,
        baseline_modes,
        baseline_min,
        baseline_max,
        ig_steps=ig_steps,
        compute_mass_penalty=False,
        **kwargs,
    )
    return reference_loss


def _fixed_baseline(
    x: torch.Tensor,
    mode: str,
    baseline_min: float,
    baseline_max: float,
) -> torch.Tensor:
    mode = mode.strip().lower()
    if mode == "zero":
        return torch.zeros_like(x)
    if mode == "blur":
        return _smooth_baseline(x)
    if mode == "noise":
        return clamp_input(torch.randn_like(x) * 0.1, baseline_min, baseline_max)
    if mode == "uniform":
        return uniform_like(x, baseline_min, baseline_max)
    if mode in {"midpoint", "mean"}:
        lower = torch.as_tensor(baseline_min, device=x.device, dtype=x.dtype)
        upper = torch.as_tensor(baseline_max, device=x.device, dtype=x.dtype)
        if lower.ndim == 1:
            lower = lower.view(1, -1, 1, 1)
            upper = upper.view(1, -1, 1, 1)
        return torch.zeros_like(x) + (lower + upper) / 2.0
    raise ValueError(f"Unknown baseline mode: {mode}")


def _one_step_linf_perturb(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps,
    alpha,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> torch.Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    ce = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(ce, x_adv, retain_graph=False, create_graph=False)[0]
    active_min = x.min().item() if clamp_min is None else clamp_min
    active_max = x.max().item() if clamp_max is None else clamp_max
    eps_tensor = _as_channel_tensor(eps, x)
    alpha_tensor = _as_channel_tensor(alpha, x)
    x_adv = clamp_input(
        x_adv.detach() + alpha_tensor * grad.sign(), active_min, active_max
    )
    delta = _clamp_delta(x_adv - x.detach(), eps_tensor)
    return clamp_input(x.detach() + delta, active_min, active_max).detach()


def rar_attribution_term(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    baseline_mode: str,
    baseline_min: float,
    baseline_max: float,
    ig_steps: int = 8,
    eps=0.01,
    alpha=0.01,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    checkpoint_activations: bool = False,
) -> torch.Tensor:
    baseline = _fixed_baseline(x, baseline_mode, baseline_min, baseline_max)
    x_adv = _one_step_linf_perturb(
        model,
        x,
        y,
        eps=eps,
        alpha=alpha,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    clean_attr = differentiable_integrated_gradients(
        model,
        x,
        y,
        baseline,
        steps=ig_steps,
        checkpoint_activations=checkpoint_activations,
    )
    adv_attr = differentiable_integrated_gradients(
        model,
        x_adv,
        y,
        baseline.detach(),
        steps=ig_steps,
        checkpoint_activations=checkpoint_activations,
    )
    return attribution_allocation_distance(clean_attr, adv_attr).mean()


def far_attribution_term(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    baseline_mode: str,
    baseline_min: float,
    baseline_max: float,
    ig_steps: int = 8,
    eps=0.01,
    samples: int = 2,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    checkpoint_activations: bool = False,
) -> torch.Tensor:
    samples = max(int(samples), 1)
    active_min = x.min().item() if clamp_min is None else clamp_min
    active_max = x.max().item() if clamp_max is None else clamp_max
    baseline = _fixed_baseline(x, baseline_mode, baseline_min, baseline_max)
    clean_attr = differentiable_integrated_gradients(
        model,
        x,
        y,
        baseline,
        steps=ig_steps,
        checkpoint_activations=checkpoint_activations,
    )
    losses = []
    eps_tensor = _as_channel_tensor(eps, x)
    for _ in range(samples):
        delta = (2.0 * torch.rand_like(x) - 1.0) * eps_tensor
        x_pert = clamp_input(x.detach() + delta, active_min, active_max)
        pert_attr = differentiable_integrated_gradients(
            model,
            x_pert,
            y,
            baseline.detach(),
            steps=ig_steps,
            checkpoint_activations=checkpoint_activations,
        )
        losses.append(attribution_allocation_distance(clean_attr, pert_attr))
    return torch.stack(losses, dim=0).max(dim=0).values.mean()


def curvature_reg_term(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    noise_std: float = 0.01,
):
    x0 = x.detach().requires_grad_(True)
    ce0 = F.cross_entropy(model(x0), y)
    grad0 = torch.autograd.grad(ce0, x0, create_graph=True, retain_graph=True)[0]

    x1 = (x.detach() + noise_std * torch.randn_like(x)).requires_grad_(True)
    ce1 = F.cross_entropy(model(x1), y)
    grad1 = torch.autograd.grad(ce1, x1, create_graph=True, retain_graph=True)[0]
    return _rms_flat(grad0 - grad1).mean()


def robust_consistency_term(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    eps=0.01,
    alpha=0.01,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
):
    x_adv = _one_step_linf_perturb(
        model,
        x,
        y,
        eps=eps,
        alpha=alpha,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )

    with torch.no_grad():
        clean_probs = F.softmax(model(x), dim=1)
    adv_log_probs = F.log_softmax(model(x_adv), dim=1)
    return F.kl_div(adv_log_probs, clean_probs, reduction="batchmean")


def train_one_epoch(
    model,
    loader,
    opt,
    device,
    lambda_entropy: float,
    scaler=None,
    entropy_model=None,
    grad_clip: float | None = None,
    lambda_wads: float = 0.0,
    lambda_rar: float = 0.0,
    lambda_far: float = 0.0,
    lambda_curvature: float = 0.0,
    lambda_robust: float = 0.0,
    lambda_attr_mass: float = 0.0,
    triguard_ig_steps: int = 8,
    baseline_modes: str = "zero,blur,noise,uniform,midpoint",
    reference_risk: str = "max",
    reference_cvar_alpha: float = 0.75,
    reference_pair_samples: int = 0,
    reference_distance: str = "allocation",
    reference_bank: torch.Tensor | None = None,
    reference_bank_samples: int = 4,
    attr_mass_floor: float = 0.9,
    attr_robust_baseline: str = "zero",
    far_samples: int = 2,
    baseline_min: float = 0.0,
    baseline_max: float = 1.0,
    curvature_noise_std: float = 0.01,
    attr_robust_eps: float = 0.01,
    attr_robust_alpha: float = 0.01,
    robust_eps: float = 0.01,
    robust_alpha: float = 0.01,
    robust_clamp_min: float | None = None,
    robust_clamp_max: float | None = None,
    regularizer_microbatch: int = 0,
    vectorized_reference_ig: bool = False,
    checkpoint_regularizer_ig: bool = False,
    sampled_mass_penalty: bool = False,
):
    model.train()
    total = 0.0
    n = 0

    use_amp = (device.type == "cuda") and (scaler is not None)
    entropy_model = entropy_model or model
    has_regularizers = any(
        weight > 0
        for weight in [
            lambda_entropy,
            lambda_wads,
            lambda_rar,
            lambda_far,
            lambda_curvature,
            lambda_robust,
            lambda_attr_mass,
        ]
    )
    needs_higher_order = any(
        weight > 0
        for weight in [
            lambda_entropy,
            lambda_wads,
            lambda_attr_mass,
            lambda_rar,
            lambda_far,
            lambda_curvature,
        ]
    )

    def regularization_loss(x_active, y_active):
        loss = x_active.new_zeros(())
        if lambda_entropy > 0:
            with _inference_mode_layers(entropy_model):
                loss = loss + lambda_entropy * entropy_reg_term(
                    entropy_model, x_active, y_active
                )
        if lambda_wads > 0 or lambda_attr_mass > 0:
            with _inference_mode_layers(entropy_model):
                reference_loss, mass_penalty = baseline_regularization_terms(
                    entropy_model,
                    x_active,
                    y_active,
                    baseline_modes=baseline_modes,
                    baseline_min=baseline_min,
                    baseline_max=baseline_max,
                    ig_steps=triguard_ig_steps,
                    reference_risk=reference_risk,
                    reference_cvar_alpha=reference_cvar_alpha,
                    reference_pair_samples=reference_pair_samples,
                    reference_distance=reference_distance,
                    reference_bank=reference_bank,
                    reference_bank_samples=reference_bank_samples,
                    attr_mass_floor=attr_mass_floor,
                    compute_mass_penalty=lambda_attr_mass > 0,
                    sampled_mass_penalty=sampled_mass_penalty,
                    vectorized_reference_ig=vectorized_reference_ig,
                    checkpoint_activations=checkpoint_regularizer_ig,
                )
            loss = loss + lambda_wads * reference_loss
            loss = loss + lambda_attr_mass * mass_penalty
        if lambda_curvature > 0:
            with _inference_mode_layers(entropy_model):
                loss = loss + lambda_curvature * curvature_reg_term(
                    entropy_model,
                    x_active,
                    y_active,
                    noise_std=curvature_noise_std,
                )
        if lambda_rar > 0:
            with _inference_mode_layers(entropy_model):
                loss = loss + lambda_rar * rar_attribution_term(
                    entropy_model,
                    x_active,
                    y_active,
                    baseline_mode=attr_robust_baseline,
                    baseline_min=baseline_min,
                    baseline_max=baseline_max,
                    ig_steps=triguard_ig_steps,
                    eps=attr_robust_eps,
                    alpha=attr_robust_alpha,
                    clamp_min=robust_clamp_min,
                    clamp_max=robust_clamp_max,
                    checkpoint_activations=checkpoint_regularizer_ig,
                )
        if lambda_far > 0:
            with _inference_mode_layers(entropy_model):
                loss = loss + lambda_far * far_attribution_term(
                    entropy_model,
                    x_active,
                    y_active,
                    baseline_mode=attr_robust_baseline,
                    baseline_min=baseline_min,
                    baseline_max=baseline_max,
                    ig_steps=triguard_ig_steps,
                    eps=attr_robust_eps,
                    samples=far_samples,
                    clamp_min=robust_clamp_min,
                    clamp_max=robust_clamp_max,
                    checkpoint_activations=checkpoint_regularizer_ig,
                )
        if lambda_robust > 0:
            with _inference_mode_layers(model):
                loss = loss + lambda_robust * robust_consistency_term(
                    model,
                    x_active,
                    y_active,
                    eps=robust_eps,
                    alpha=robust_alpha,
                    clamp_min=robust_clamp_min,
                    clamp_max=robust_clamp_max,
                )
        return loss

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)

        opt.zero_grad(set_to_none=True)
        batch_size = x.size(0)
        batch_loss = 0.0
        with _sdp_math_context(
            device, require_higher_order=needs_higher_order
        ):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    ce = F.cross_entropy(model(x), y)
                ce_for_backward = ce.float()
                if not torch.isfinite(ce_for_backward):
                    raise FloatingPointError("Cross-entropy loss became non-finite.")
                scaler.scale(ce_for_backward).backward()
            else:
                ce_for_backward = F.cross_entropy(model(x), y)
                if not torch.isfinite(ce_for_backward):
                    raise FloatingPointError("Cross-entropy loss became non-finite.")
                ce_for_backward.backward()
            batch_loss += float(ce_for_backward.detach().item())

            if has_regularizers:
                microbatch = (
                    batch_size
                    if regularizer_microbatch <= 0
                    else min(int(regularizer_microbatch), batch_size)
                )
                for start in range(0, batch_size, microbatch):
                    stop = min(start + microbatch, batch_size)
                    x_active = x[start:stop].float() if use_amp else x[start:stop]
                    y_active = y[start:stop]
                    reg = regularization_loss(x_active, y_active)
                    weight = (stop - start) / batch_size
                    weighted_reg = reg * weight
                    if not torch.isfinite(weighted_reg):
                        raise FloatingPointError(
                            "Attribution regularizer became non-finite."
                        )
                    if use_amp:
                        scaler.scale(weighted_reg).backward()
                    else:
                        weighted_reg.backward()
                    batch_loss += float(weighted_reg.detach().item())

        if use_amp:
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip
                )
            scaler.step(opt)
            scaler.update()
        else:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip
                )
            opt.step()

        total += batch_loss * batch_size
        n += batch_size

    return total / max(n, 1)
