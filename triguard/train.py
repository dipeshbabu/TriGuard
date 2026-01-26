import torch
import torch.nn.functional as F


def entropy_reg_term(model, x, y, eps=1e-10):
    # Detach from prior graph; we only need grads wrt input for the entropy term.
    # Dropped clone() for speed (detach() is enough here).
    x = x.detach().requires_grad_(True)
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(ce, x, create_graph=True, retain_graph=True)[0]
    g = grad.abs().view(grad.size(0), -1)
    p = g / (g.sum(dim=1, keepdim=True) + eps)
    H = -(p * torch.log(p + eps)).sum(dim=1).mean()
    return H


def train_one_epoch(model, loader, opt, device, lambda_entropy: float, scaler=None):
    model.train()
    total = 0.0
    n = 0

    use_amp = (device.type == "cuda") and (scaler is not None)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # A100 likes channels_last for NCHW images
        if x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)

        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                ce = F.cross_entropy(logits, y)

            # entropy term needs input grads; keep it in fp32 to avoid instability
            if lambda_entropy > 0:
                # Higher-order autograd + torch.compile can be finicky; disable compile around this block.
                if hasattr(torch, "_dynamo"):
                    with torch._dynamo.disable():
                        H = entropy_reg_term(model, x, y)
                else:
                    H = entropy_reg_term(model, x, y)
                loss = ce.float() + lambda_entropy * H
            else:
                loss = ce.float()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            ce = F.cross_entropy(logits, y)
            if lambda_entropy > 0:
                H = entropy_reg_term(model, x, y)
                loss = ce + lambda_entropy * H
            else:
                loss = ce
            loss.backward()
            opt.step()

        total += float(loss.item()) * x.size(0)
        n += x.size(0)

    return total / max(n, 1)
