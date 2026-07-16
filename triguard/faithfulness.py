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


def deletion_insertion_curve(
    model,
    x,
    target,
    attr,
    mode,
    steps=50,
    baseline=None,
    batch_size=64,
):
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
    steps = max(int(steps), 1)
    boundaries = torch.linspace(
        0, total, steps + 1, device=order.device
    ).round().to(dtype=torch.long)

    if mode == "deletion":
        cur = x0.clone()
    elif mode == "insertion":
        cur = baseline.clone()
    else:
        raise ValueError("mode must be deletion or insertion")

    states = []
    for t in range(steps+1):
        states.append(cur.clone())

        if t == steps:
            break

        idx = order[boundaries[t]:boundaries[t + 1]]
        h = (idx // W).long()
        w = (idx % W).long()

        if mode == "deletion":
            cur[..., h, w] = baseline[..., h, w]
        else:
            cur[..., h, w] = x0[..., h, w]

    active_batch = max(int(batch_size), 1)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(states), active_batch):
            state_batch = torch.cat(states[start:start + active_batch], dim=0)
            probabilities.extend(
                torch.softmax(model(state_batch), dim=1)[:, int(target)]
                .detach()
                .cpu()
                .tolist()
            )
    return np.asarray(probabilities, dtype=np.float64)


def auc(curve: np.ndarray):
    x = np.linspace(0.0, 1.0, len(curve))
    return float(np.trapz(curve, x))


def faithfulness_auc(
    model,
    x,
    target,
    attr,
    steps=50,
    baseline=None,
    curve_batch_size=64,
):
    del_curve = deletion_insertion_curve(
        model,
        x,
        target,
        attr,
        "deletion",
        steps=steps,
        baseline=baseline,
        batch_size=curve_batch_size,
    )
    ins_curve = deletion_insertion_curve(
        model,
        x,
        target,
        attr,
        "insertion",
        steps=steps,
        baseline=baseline,
        batch_size=curve_batch_size,
    )
    return auc(del_curve), auc(ins_curve), del_curve, ins_curve
