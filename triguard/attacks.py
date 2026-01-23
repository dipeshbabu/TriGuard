import torch
import torch.nn.functional as F


def pgd_linf(model, x, y, eps, alpha, steps, clamp_min, clamp_max):
    """
    PGD in the same space the model consumes (i.e., normalized if data is normalized).
    """
    model.eval()
    x0 = x.detach()
    x_adv = x.detach().clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(
            loss, x_adv, retain_graph=False, create_graph=False)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        eta = torch.clamp(x_adv - x0, min=-eps, max=eps)
        x_adv = torch.clamp(x0 + eta, min=clamp_min, max=clamp_max).detach()

    return x_adv
