import torch
import torch.nn.functional as F

from .attacks import _as_channel_tensor, clamp_input, uniform_like


def sanitize_attribution(attr: torch.Tensor) -> torch.Tensor:
    """Detach a finite attribution tensor; never silently turn failures into zeros."""
    detached = attr.detach()
    if not torch.isfinite(detached).all():
        raise FloatingPointError("Attribution contains NaN or infinite values.")
    return detached


def attribution_entropy(
    attr: torch.Tensor,
    eps: float = 1e-10,
    *,
    normalized: bool = False,
    pixel_level: bool = False,
) -> float | None:
    finite = sanitize_attribution(attr).abs()
    if pixel_level and finite.dim() == 4:
        finite = finite.sum(dim=1)
    v = finite.flatten()
    s = v.sum()
    if s.item() <= eps:
        return None
    p = v / (s + eps)
    h_val = -(p * torch.log(p + eps)).sum()
    if normalized and v.numel() > 1:
        h_val = h_val / torch.log(
            torch.tensor(float(v.numel()), device=v.device, dtype=v.dtype)
        )
    if not torch.isfinite(h_val):
        return None
    return float(h_val.item())


def normalize_attribution_allocation(
    attr: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Normalize signed attributions by absolute mass for scale-free comparison."""
    if attr.dim() < 2:
        raise ValueError("Attributions must include a batch dimension.")
    mass = attr.flatten(1).abs().sum(dim=1)
    shape = (attr.size(0),) + (1,) * (attr.dim() - 1)
    return attr / (mass + eps).view(shape)


def attribution_allocation_distance(
    left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Per-sample signed L1 distance between normalized attribution allocations.

    Values lie in [0, 2] when both maps have nonzero attribution mass. Unlike
    raw L2 distance, this quantity is invariant to a common rescaling of logits
    and does not force baselines with different output magnitudes to agree.
    """
    left_norm = normalize_attribution_allocation(left, eps=eps)
    right_norm = normalize_attribution_allocation(right, eps=eps)
    return (left_norm - right_norm).flatten(1).abs().sum(dim=1)


def completeness_orthogonal_distance(
    left: torch.Tensor, right: torch.Tensor, *, root_mean_square: bool = True
) -> torch.Tensor:
    """Distance after removing the constant component forced by sum mismatch.

    The difference decomposes exactly into a constant vector, which contains
    the pair's completeness/output-gap effect, and an orthogonal residual that
    captures allocation variation.
    """
    if left.shape != right.shape or left.dim() < 2:
        raise ValueError("Attribution tensors must have the same batched shape.")
    difference = (left - right).flatten(1)
    residual = difference - difference.mean(dim=1, keepdim=True)
    if root_mean_square:
        return residual.square().mean(dim=1).sqrt()
    return residual.norm(p=2, dim=1)


def integrated_gradients(model, x, target, baseline, steps=50):
    model.eval()
    steps = max(int(steps), 1)
    batch = x.size(0)
    alphas = torch.linspace(
        0.0, 1.0, steps + 1, device=x.device, dtype=x.dtype
    )[1:].view(steps, 1, 1, 1, 1)
    path = baseline.unsqueeze(0) + alphas * (x.unsqueeze(0) - baseline.unsqueeze(0))
    path = path.reshape(steps * batch, *x.shape[1:]).requires_grad_(True)
    logits = model(path)
    if isinstance(target, torch.Tensor):
        targets = target.to(device=x.device, dtype=torch.long).reshape(-1)
        if targets.numel() == 1 and batch > 1:
            targets = targets.expand(batch)
        repeated_targets = targets.repeat(steps)
        score = logits.gather(1, repeated_targets.view(-1, 1)).sum()
    else:
        score = logits[:, int(target)].sum()
    grad = torch.autograd.grad(score, path, retain_graph=False, create_graph=False)[0]
    grad = sanitize_attribution(grad)
    avg_grad = grad.reshape(steps, batch, *x.shape[1:]).mean(dim=0)
    ig = (x - baseline) * avg_grad
    return sanitize_attribution(ig)


def blurred_baseline(x, kernel=11):
    if kernel <= 1:
        return x.detach()
    height, width = x.shape[-2:]
    kernel = min(int(kernel), height, width)
    if kernel % 2 == 0:
        kernel -= 1
    if kernel <= 1:
        return x.detach()
    padding = kernel // 2
    mode = "reflect" if padding < min(height, width) else "replicate"
    padded = F.pad(x.detach(), (padding,) * 4, mode=mode)
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1)


def uniform_baseline(x, low: float, high: float, generator=None):
    return uniform_like(x, low, high, generator=generator)


def noise_baseline(
    x,
    std: float = 0.1,
    low: float | None = None,
    high: float | None = None,
    generator=None,
):
    baseline = torch.randn(
        x.shape, device=x.device, dtype=x.dtype, generator=generator
    ) * std
    if low is not None and high is not None:
        baseline = clamp_input(baseline, low, high)
    return baseline


def ads_baseline(model, x, target, b1, b2, steps=50) -> float | None:
    a1 = integrated_gradients(model, x, target, b1, steps=steps)
    a2 = integrated_gradients(model, x, target, b2, steps=steps)
    dist = torch.norm((a1 - a2).flatten(), p=2)
    if not torch.isfinite(dist):
        return None
    return float(dist.item())


def allocation_ads_baseline(model, x, target, b1, b2, steps=50) -> float | None:
    a1 = integrated_gradients(model, x, target, b1, steps=steps)
    a2 = integrated_gradients(model, x, target, b2, steps=steps)
    dist = attribution_allocation_distance(a1, a2)
    if dist.numel() != 1 or not torch.isfinite(dist).all():
        return None
    return float(dist.item())


def smoothgrad_squared(
    model,
    x,
    target,
    noise_level=0.1,
    n_samples=50,
    clamp_min=None,
    clamp_max=None,
    generator=None,
    sample_batch_size=16,
):
    model.eval()
    accumulated = torch.zeros_like(x)
    noise_scale = _as_channel_tensor(noise_level, x)
    completed = 0
    active_batch = max(int(sample_batch_size), 1)
    while completed < n_samples:
        count = min(active_batch, n_samples - completed)
        # Draw samples one at a time so changing only the model-evaluation
        # batch size does not change the seeded SmoothGrad perturbations.
        noise = torch.stack(
            [
                torch.randn(
                    x.shape,
                    device=x.device,
                    dtype=x.dtype,
                    generator=generator,
                )
                for _ in range(count)
            ],
            dim=0,
        )
        x_noisy = x.unsqueeze(0) + noise_scale.unsqueeze(0) * noise
        x_noisy = x_noisy.reshape(count * x.size(0), *x.shape[1:])
        if clamp_min is not None and clamp_max is not None:
            x_noisy = clamp_input(x_noisy, clamp_min, clamp_max)
        x_noisy = x_noisy.detach().requires_grad_(True)
        logits = model(x_noisy)
        score = logits[:, target].sum()
        grad = torch.autograd.grad(score, x_noisy, retain_graph=False, create_graph=False)[0]
        grad = sanitize_attribution(grad)
        accumulated = accumulated + grad.reshape(
            count, x.size(0), *x.shape[1:]
        ).pow(2).sum(dim=0)
        completed += count
    return sanitize_attribution(accumulated / max(int(n_samples), 1))
