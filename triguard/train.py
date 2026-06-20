from contextlib import nullcontext
from itertools import combinations

import torch
import torch.nn.functional as F


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
    pad = kernel_size // 2
    return F.avg_pool2d(x.detach(), kernel_size=kernel_size, stride=1, padding=pad)


def make_baseline_family(
    x: torch.Tensor,
    modes: str,
    baseline_min: float,
    baseline_max: float,
):
    family = {}
    for mode in [m.strip().lower() for m in modes.split(",") if m.strip()]:
        if mode == "zero":
            family[mode] = torch.zeros_like(x)
        elif mode == "blur":
            family[mode] = _smooth_baseline(x)
        elif mode == "noise":
            family[mode] = torch.randn_like(x) * 0.1
        elif mode == "uniform":
            family[mode] = torch.empty_like(x).uniform_(baseline_min, baseline_max)
        elif mode == "mean":
            family[mode] = torch.full_like(x, (baseline_min + baseline_max) / 2.0)
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
):
    steps = max(int(steps), 1)
    batch = x.size(0)
    alphas = torch.linspace(
        0.0, 1.0, steps + 1, device=x.device, dtype=x.dtype
    ).view(-1, 1, 1, 1, 1)
    path = baseline.unsqueeze(0) + alphas * (x.unsqueeze(0) - baseline.unsqueeze(0))
    path = path.reshape((steps + 1) * batch, *x.shape[1:]).requires_grad_(True)

    repeated_targets = targets.repeat(steps + 1)
    logits = model(path)
    score = logits.gather(1, repeated_targets.view(-1, 1)).sum()
    grad = torch.autograd.grad(score, path, create_graph=True, retain_graph=True)[0]
    grad = grad.reshape(steps + 1, batch, *x.shape[1:])
    avg_grad = grad[:-1].mean(dim=0)
    return (x - baseline) * avg_grad


def worst_baseline_drift_term(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    baseline_modes: str,
    baseline_min: float,
    baseline_max: float,
    ig_steps: int = 8,
):
    family = make_baseline_family(x, baseline_modes, baseline_min, baseline_max)
    attributions = {
        name: differentiable_integrated_gradients(model, x, y, baseline, steps=ig_steps)
        for name, baseline in family.items()
    }
    pair_losses = []
    for left, right in combinations(attributions.keys(), 2):
        pair_losses.append(_rms_flat(attributions[left] - attributions[right]).mean())
    return torch.stack(pair_losses).max()


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
    eps: float = 0.01,
    alpha: float = 0.01,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
):
    x_adv = x.detach().clone().requires_grad_(True)
    ce = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(ce, x_adv, retain_graph=False, create_graph=False)[0]
    active_min = x.min().item() if clamp_min is None else clamp_min
    active_max = x.max().item() if clamp_max is None else clamp_max
    x_adv = (x_adv.detach() + alpha * grad.sign()).clamp(active_min, active_max)
    delta = torch.clamp(x_adv - x.detach(), min=-eps, max=eps)
    x_adv = (x.detach() + delta).clamp(active_min, active_max).detach()

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
    lambda_curvature: float = 0.0,
    lambda_robust: float = 0.0,
    triguard_ig_steps: int = 8,
    baseline_modes: str = "zero,blur,noise,uniform,mean",
    baseline_min: float = 0.0,
    baseline_max: float = 1.0,
    curvature_noise_std: float = 0.01,
    robust_eps: float = 0.01,
    robust_alpha: float = 0.01,
    robust_clamp_min: float | None = None,
    robust_clamp_max: float | None = None,
):
    model.train()
    total = 0.0
    n = 0

    use_amp = (device.type == "cuda") and (scaler is not None)
    entropy_model = entropy_model or model

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)

        opt.zero_grad(set_to_none=True)
        needs_higher_order = (
            lambda_entropy > 0
            or lambda_wads > 0
            or lambda_curvature > 0
        )
        sdp_ctx = _sdp_math_context(device, require_higher_order=needs_higher_order)

        with sdp_ctx:
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    ce = F.cross_entropy(logits, y)

                loss = ce.float()
                if lambda_entropy > 0:
                    # Keep entropy term in fp32 for stability.
                    H = entropy_reg_term(entropy_model, x.float(), y)
                    loss = loss + lambda_entropy * H
                if lambda_wads > 0:
                    loss = loss + lambda_wads * worst_baseline_drift_term(
                        entropy_model,
                        x.float(),
                        y,
                        baseline_modes=baseline_modes,
                        baseline_min=baseline_min,
                        baseline_max=baseline_max,
                        ig_steps=triguard_ig_steps,
                    )
                if lambda_curvature > 0:
                    loss = loss + lambda_curvature * curvature_reg_term(
                        entropy_model,
                        x.float(),
                        y,
                        noise_std=curvature_noise_std,
                    )
                if lambda_robust > 0:
                    loss = loss + lambda_robust * robust_consistency_term(
                        model,
                        x,
                        y,
                        eps=robust_eps,
                        alpha=robust_alpha,
                        clamp_min=robust_clamp_min,
                        clamp_max=robust_clamp_max,
                    )
            else:
                logits = model(x)
                ce = F.cross_entropy(logits, y)
                loss = ce
                if lambda_entropy > 0:
                    H = entropy_reg_term(entropy_model, x, y)
                    loss = loss + lambda_entropy * H
                if lambda_wads > 0:
                    loss = loss + lambda_wads * worst_baseline_drift_term(
                        entropy_model,
                        x,
                        y,
                        baseline_modes=baseline_modes,
                        baseline_min=baseline_min,
                        baseline_max=baseline_max,
                        ig_steps=triguard_ig_steps,
                    )
                if lambda_curvature > 0:
                    loss = loss + lambda_curvature * curvature_reg_term(
                        entropy_model,
                        x,
                        y,
                        noise_std=curvature_noise_std,
                    )
                if lambda_robust > 0:
                    loss = loss + lambda_robust * robust_consistency_term(
                        model,
                        x,
                        y,
                        eps=robust_eps,
                        alpha=robust_alpha,
                        clamp_min=robust_clamp_min,
                        clamp_max=robust_clamp_max,
                    )

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

        total += float(loss.item()) * x.size(0)
        n += x.size(0)

    return total / max(n, 1)
