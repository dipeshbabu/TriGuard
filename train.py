import torch
import torch.nn.functional as F


def entropy_reg_term(model, x, y, eps=1e-10):
    """
    Input-gradient entropy regularization (matches your paper).
    """
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(ce, x, create_graph=True, retain_graph=True)[0]
    g = grad.abs().view(grad.size(0), -1)
    p = g / (g.sum(dim=1, keepdim=True) + eps)
    H = -(p * torch.log(p + eps)).sum(dim=1).mean()
    return H


def train_one_epoch(model, loader, opt, device, lambda_entropy: float):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        ce = F.cross_entropy(logits, y)
        if lambda_entropy > 0:
            H = entropy_reg_term(model, x, y)
            loss = ce + lambda_entropy * H
        else:
            loss = ce
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)
