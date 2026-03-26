import torch
import torch.nn.functional as F


def pgd_linf(model, x, y, eps, alpha, steps, clamp_min, clamp_max, random_start: bool = True):
    model.eval()
    x0 = x.detach()
    if random_start:
        x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, min=clamp_min, max=clamp_max).detach()
    else:
        x_adv = x0.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        eta = torch.clamp(x_adv - x0, min=-eps, max=eps)
        x_adv = torch.clamp(x0 + eta, min=clamp_min, max=clamp_max).detach()

    return x_adv
