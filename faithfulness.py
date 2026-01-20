import torch
import numpy as np


def _rank_pixels(attr: torch.Tensor):
    # attr: (1,C,H,W) -> importance per pixel (H,W)
    if attr.dim() == 4 and attr.size(1) > 1:
        imp = attr.abs().mean(dim=1)[0]  # (H,W)
    else:
        imp = attr.abs()[0, 0]
    order = torch.argsort(imp.flatten(), descending=True)
    return order


def deletion_insertion_curve(model, x, target, attr, mode, steps=50, baseline=None):
    """
    mode: 'deletion' or 'insertion'
    returns curve length steps+1 of p(target | x_t)
    """
    model.eval()
    x0 = x.detach().clone()
    if baseline is None:
        baseline = torch.zeros_like(x0)

    order = _rank_pixels(attr)
    H = x0.size(-2)
    W = x0.size(-1)
    total = H * W
    k_per = max(total // steps, 1)

    if mode == "deletion":
        cur = x0.clone()
    elif mode == "insertion":
        cur = baseline.clone()
    else:
        raise ValueError("mode must be deletion or insertion")

    curve = []
    for t in range(steps+1):
        with torch.no_grad():
            p = torch.softmax(model(cur), dim=1)[0, target].item()
        curve.append(p)

        if t == steps:
            break

        idx = order[t*k_per: min((t+1)*k_per, total)]
        h = (idx // W).long()
        w = (idx % W).long()

        if mode == "deletion":
            cur[..., h, w] = baseline[..., h, w]
        else:
            cur[..., h, w] = x0[..., h, w]

    return np.array(curve, dtype=np.float64)


def auc(curve: np.ndarray):
    x = np.linspace(0.0, 1.0, len(curve))
    return float(np.trapz(curve, x))


def faithfulness_auc(model, x, target, attr, steps=50, baseline=None):
    del_curve = deletion_insertion_curve(
        model, x, target, attr, "deletion", steps=steps, baseline=baseline)
    ins_curve = deletion_insertion_curve(
        model, x, target, attr, "insertion", steps=steps, baseline=baseline)
    return auc(del_curve), auc(ins_curve), del_curve, ins_curve
