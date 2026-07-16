import torch
import torch.nn.functional as F


def _as_channel_tensor(value, ref: torch.Tensor):
    if isinstance(value, torch.Tensor):
        value = value.to(device=ref.device, dtype=ref.dtype)
    elif isinstance(value, (list, tuple)):
        value = torch.tensor(value, device=ref.device, dtype=ref.dtype)
    else:
        value = torch.tensor(float(value), device=ref.device, dtype=ref.dtype)

    if value.ndim == 1:
        value = value.view(1, -1, 1, 1)
    return value


def clamp_input(x: torch.Tensor, lower, upper) -> torch.Tensor:
    """Clamp an image tensor with scalar or channel-wise normalized bounds."""
    lower = _as_channel_tensor(lower, x)
    upper = _as_channel_tensor(upper, x)
    return torch.maximum(torch.minimum(x, upper), lower)


def uniform_like(
    x: torch.Tensor, lower, upper, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample uniformly inside scalar or channel-wise normalized bounds."""
    lower = _as_channel_tensor(lower, x)
    upper = _as_channel_tensor(upper, x)
    sample = torch.rand(
        x.shape, device=x.device, dtype=x.dtype, generator=generator
    )
    return lower + sample * (upper - lower)


def _clamp_delta(delta: torch.Tensor, eps):
    if isinstance(eps, torch.Tensor):
        return torch.maximum(torch.minimum(delta, eps), -eps)
    return torch.clamp(delta, min=-eps, max=eps)


def pgd_linf(
    model,
    x,
    y,
    eps,
    alpha,
    steps,
    clamp_min,
    clamp_max,
    random_start: bool = True,
    generator: torch.Generator | None = None,
):
    model.eval()
    x0 = x.detach()
    eps = _as_channel_tensor(eps, x0)
    alpha = _as_channel_tensor(alpha, x0)
    if random_start:
        noise = (
            2.0
            * torch.rand(
                x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
            )
            - 1.0
        ) * eps
        x_adv = x0 + noise
        x_adv = clamp_input(x_adv, clamp_min, clamp_max).detach()
    else:
        x_adv = x0.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        eta = _clamp_delta(x_adv - x0, eps)
        x_adv = clamp_input(x0 + eta, clamp_min, clamp_max).detach()

    return x_adv
