import torch
import torchvision.transforms as T


def attribution_entropy(attr: torch.Tensor, eps: float = 1e-10):
    v = attr.abs().flatten()
    s = v.sum()
    if s.item() == 0.0 or torch.isnan(s):
        return float("nan")
    p = v / (s + eps)
    h_val = -(p * torch.log(p + eps)).sum()
    return h_val.item()


def integrated_gradients(model, x, target, baseline, steps=50):
    model.eval()
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device).view(-1, 1, 1, 1)
    x_interp = baseline + alphas * (x - baseline)
    x_interp.requires_grad_(True)
    logits = model(x_interp)
    score = logits[:, target].sum()
    grad = torch.autograd.grad(score, x_interp, retain_graph=False, create_graph=False)[0]
    avg_grad = grad[:-1].mean(dim=0, keepdim=True)
    ig = (x - baseline) * avg_grad
    return ig


def blurred_baseline(x, kernel=11, sigma=5.0):
    blur = T.GaussianBlur(kernel_size=kernel, sigma=sigma)
    return blur(x).detach()


def uniform_baseline(x, low: float, high: float):
    return torch.empty_like(x).uniform_(low, high)


def noise_baseline(x, std: float = 0.1):
    return torch.randn_like(x) * std


def ads_baseline(model, x, target, b1, b2, steps=50):
    a1 = integrated_gradients(model, x, target, b1, steps=steps)
    a2 = integrated_gradients(model, x, target, b2, steps=steps)
    return torch.norm((a1 - a2).flatten(), p=2).item()


def ads_adv(model, x, x_adv, target, baseline, steps=50):
    a = integrated_gradients(model, x, target, baseline, steps=steps)
    b = integrated_gradients(model, x_adv, target, baseline, steps=steps)
    return torch.norm((a - b).flatten(), p=2).item()


def smoothgrad_squared(model, x, target, noise_level=0.1, n_samples=50):
    model.eval()
    grads_sq = []
    for _ in range(n_samples):
        x_noisy = (x + noise_level * torch.randn_like(x)).detach().requires_grad_(True)
        logits = model(x_noisy)
        score = logits[:, target].sum()
        grad = torch.autograd.grad(score, x_noisy, retain_graph=False, create_graph=False)[0]
        grads_sq.append(grad.pow(2))
    return torch.stack(grads_sq, dim=0).mean(dim=0)
