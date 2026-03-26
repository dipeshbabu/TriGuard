import torch
import torchvision.transforms as T


def sanitize_attribution(attr: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(attr.detach(), nan=0.0, posinf=0.0, neginf=0.0)


def attribution_entropy(attr: torch.Tensor, eps: float = 1e-10) -> float | None:
    v = sanitize_attribution(attr).abs().flatten()
    s = v.sum()
    if s.item() <= eps:
        return None
    p = v / (s + eps)
    h_val = -(p * torch.log(p + eps)).sum()
    if not torch.isfinite(h_val):
        return None
    return float(h_val.item())


def integrated_gradients(model, x, target, baseline, steps=50):
    model.eval()
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device).view(-1, 1, 1, 1)
    x_interp = baseline + alphas * (x - baseline)
    x_interp.requires_grad_(True)
    logits = model(x_interp)
    score = logits[:, target].sum()
    grad = torch.autograd.grad(score, x_interp, retain_graph=False, create_graph=False)[0]
    grad = sanitize_attribution(grad)
    avg_grad = grad[:-1].mean(dim=0, keepdim=True)
    ig = (x - baseline) * avg_grad
    return sanitize_attribution(ig)


def blurred_baseline(x, kernel=11, sigma=5.0):
    blur = T.GaussianBlur(kernel_size=kernel, sigma=sigma)
    return blur(x).detach()


def uniform_baseline(x, low: float, high: float):
    return torch.empty_like(x).uniform_(low, high)


def noise_baseline(x, std: float = 0.1):
    return torch.randn_like(x) * std


def ads_baseline(model, x, target, b1, b2, steps=50) -> float | None:
    a1 = integrated_gradients(model, x, target, b1, steps=steps)
    a2 = integrated_gradients(model, x, target, b2, steps=steps)
    dist = torch.norm((a1 - a2).flatten(), p=2)
    if not torch.isfinite(dist):
        return None
    return float(dist.item())


def ads_adv(model, x, x_adv, target, baseline, steps=50) -> float | None:
    a = integrated_gradients(model, x, target, baseline, steps=steps)
    b = integrated_gradients(model, x_adv, target, baseline, steps=steps)
    dist = torch.norm((a - b).flatten(), p=2)
    if not torch.isfinite(dist):
        return None
    return float(dist.item())


def smoothgrad_squared(model, x, target, noise_level=0.1, n_samples=50):
    model.eval()
    grads_sq = []
    for _ in range(n_samples):
        x_noisy = (x + noise_level * torch.randn_like(x)).detach().requires_grad_(True)
        logits = model(x_noisy)
        score = logits[:, target].sum()
        grad = torch.autograd.grad(score, x_noisy, retain_graph=False, create_graph=False)[0]
        grad = sanitize_attribution(grad)
        grads_sq.append(grad.pow(2))
    return sanitize_attribution(torch.stack(grads_sq, dim=0).mean(dim=0))
