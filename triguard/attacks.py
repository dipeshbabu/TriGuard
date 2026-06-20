import torch
import torch.nn.functional as F


def _as_channel_tensor(value, ref: torch.Tensor):
    if isinstance(value, torch.Tensor):
        value = value.to(device=ref.device, dtype=ref.dtype)
    elif isinstance(value, (list, tuple)):
        value = torch.tensor(value, device=ref.device, dtype=ref.dtype)
    else:
        return float(value)

    if value.ndim == 1:
        value = value.view(1, -1, 1, 1)
    return value


def _clamp_delta(delta: torch.Tensor, eps):
    if isinstance(eps, torch.Tensor):
        return torch.maximum(torch.minimum(delta, eps), -eps)
    return torch.clamp(delta, min=-eps, max=eps)


def pgd_linf(model, x, y, eps, alpha, steps, clamp_min, clamp_max, random_start: bool = True):
    model.eval()
    x0 = x.detach()
    eps = _as_channel_tensor(eps, x0)
    alpha = _as_channel_tensor(alpha, x0)
    if random_start:
        noise = (2.0 * torch.rand_like(x0) - 1.0) * eps
        x_adv = x0 + noise
        x_adv = torch.clamp(x_adv, min=clamp_min, max=clamp_max).detach()
    else:
        x_adv = x0.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        eta = _clamp_delta(x_adv - x0, eps)
        x_adv = torch.clamp(x0 + eta, min=clamp_min, max=clamp_max).detach()

    return x_adv
