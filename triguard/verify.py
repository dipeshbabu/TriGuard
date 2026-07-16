import numpy as np
import torch
import torch.nn as nn

try:
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    _AUTO_LIRPA_IMPORT_ERROR = None
except Exception as exc:
    BoundedModule = BoundedTensor = PerturbationLpNorm = None
    _AUTO_LIRPA_IMPORT_ERROR = exc

from .attacks import _as_channel_tensor, clamp_input


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


def crown_ibp_certify(
    model,
    x,
    y,
    eps,
    device,
    return_status: bool = False,
    clamp_min=None,
    clamp_max=None,
):
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    if BoundedModule is None:
        if return_status:
            return None, "dependency:auto_lirpa_unavailable"
        raise RuntimeError(
            "auto-LiRPA is required for CROWN certification."
        ) from _AUTO_LIRPA_IMPORT_ERROR

    try:
        y = int(y)
        margin_model = MarginModel(model, y).to(device).eval()
        eps_tensor = _as_channel_tensor(eps, x)
        x_lower = x - eps_tensor
        x_upper = x + eps_tensor
        if clamp_min is not None and clamp_max is not None:
            x_lower = clamp_input(x_lower, clamp_min, clamp_max)
            x_upper = clamp_input(x_upper, clamp_min, clamp_max)
        ptb = PerturbationLpNorm(norm=np.inf, x_L=x_lower, x_U=x_upper)
        bx = BoundedTensor(x, ptb)
        lirpa = BoundedModule(margin_model, x, device=device, verbose=False)
        lb, _ = lirpa.compute_bounds(x=(bx,), method="CROWN-IBP")
        certified = bool((lb[0] > 0).all().item())
        if return_status:
            return certified, "ok"
        return certified
    except Exception as exc:
        if return_status:
            return None, f"error:{type(exc).__name__}"
        raise


def empirical_probe(
    model,
    x,
    num_samples,
    eps,
    clamp_min,
    clamp_max,
    device,
    generator=None,
    return_details: bool = False,
):
    model.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    eps = _as_channel_tensor(eps, x)
    violations = 0
    minimum_margin = float("inf")
    with torch.no_grad():
        base = int(model(x).argmax(dim=1).item())
    for _ in range(num_samples):
        noise = (
            2.0
            * torch.rand(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
            - 1.0
        ) * eps
        x2 = clamp_input(x + noise, clamp_min, clamp_max)
        with torch.no_grad():
            logits = model(x2)[0]
            other = torch.cat([logits[:base], logits[base + 1:]]).max()
            margin = float((logits[base] - other).item())
            minimum_margin = min(minimum_margin, margin)
            violations += int(margin <= 0.0)
    details = {
        "passed": violations == 0,
        "violation_fraction": float(violations / max(int(num_samples), 1)),
        "minimum_margin": minimum_margin if num_samples > 0 else float("nan"),
    }
    return details if return_details else details["passed"]
