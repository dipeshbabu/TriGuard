import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


def crown_ibp_certify(model, x, y, eps, device):
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    try:
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        bx = BoundedTensor(x, ptb)
        lirpa = BoundedModule(model, x, device=device, verbose=False)
        lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN-IBP")
        y = int(y)
        correct_lb = lb[0, y]
        others = torch.cat([lb[0, :y], lb[0, y + 1:]])
        return bool((correct_lb > others.max()).item())
    except Exception:
        return False


def empirical_probe(model, x, num_samples, eps, clamp_min, clamp_max, device):
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        base = model(x).argmax(dim=1).item()
    for _ in range(num_samples):
        noise = torch.empty_like(x).uniform_(-eps, eps)
        x2 = torch.clamp(x + noise, clamp_min, clamp_max)
        with torch.no_grad():
            if model(x2).argmax(dim=1).item() != base:
                return False
    return True
