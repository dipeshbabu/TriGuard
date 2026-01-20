import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

def crown_ibp_certify(model, x, y, eps, device):
    """
    Certified verification via CROWN-IBP: checks if correct class lower bound
    exceeds all other class lower bounds (a simple certified margin check).
    """
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
    bx = BoundedTensor(x, ptb)

    lirpa = BoundedModule(model, x, device=device, verbose=False)
    lb, ub = lirpa.compute_bounds(x=(bx,), method="CROWN-IBP")

    y = int(y)
    correct_lb = lb[0, y]
    others = torch.cat([lb[0, :y], lb[0, y+1:]])
    certified = (correct_lb > others.max()).item()
    return bool(certified)

def empirical_probe(model, x, num_samples, eps, clamp_min, clamp_max, device):
    """
    Randomized diagnostic only (NOT formal verification).
    """
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    base = model(x).argmax(dim=1).item()
    for _ in range(num_samples):
        noise = torch.empty_like(x).uniform_(-eps, eps)
        x2 = torch.clamp(x + noise, clamp_min, clamp_max)
        if model(x2).argmax(dim=1).item() != base:
            return False
    return True
