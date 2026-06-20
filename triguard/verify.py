import numpy as np
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .attacks import _as_channel_tensor


class MarginModel(nn.Module):
    def __init__(self, model: nn.Module, y: int):
        super().__init__()
        self.model = model
        self.y = int(y)

    def forward(self, x):
        logits = self.model(x)
        correct = logits[:, self.y:self.y + 1]
        others = torch.cat([logits[:, :self.y], logits[:, self.y + 1:]], dim=1)
        return correct - others


def crown_ibp_certify(model, x, y, eps, device):
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    try:
        y = int(y)
        margin_model = MarginModel(model, y).to(device).eval()
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        bx = BoundedTensor(x, ptb)
        lirpa = BoundedModule(margin_model, x, device=device, verbose=False)
        lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN-IBP")
        return bool((lb[0] > 0).all().item())
    except Exception:
        return False


def empirical_probe(model, x, num_samples, eps, clamp_min, clamp_max, device):
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    eps = _as_channel_tensor(eps, x)
    with torch.no_grad():
        base = model(x).argmax(dim=1).item()
    for _ in range(num_samples):
        noise = (2.0 * torch.rand_like(x) - 1.0) * eps
        x2 = torch.clamp(x + noise, clamp_min, clamp_max)
        with torch.no_grad():
            if model(x2).argmax(dim=1).item() != base:
                return False
    return True
