from contextlib import nullcontext

import torch
import torch.nn.functional as F


def _sdp_math_context(device, require_higher_order: bool):
    if device.type != "cuda" or not require_higher_order:
        return nullcontext()

    attention_mod = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attention_mod, "sdpa_kernel", None)
    sdp_backend = getattr(attention_mod, "SDPBackend", None)
    if sdpa_kernel is not None and sdp_backend is not None:
        return sdpa_kernel(backends=[sdp_backend.MATH])

    cuda_backends = getattr(torch.backends, "cuda", None)
    legacy_sdp_kernel = getattr(cuda_backends, "sdp_kernel", None)
    if legacy_sdp_kernel is not None:
        return legacy_sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

    return nullcontext()


def entropy_reg_term(model, x, y, eps=1e-10):
    x = x.detach().requires_grad_(True)
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(ce, x, create_graph=True, retain_graph=True)[0]
    g = grad.abs().reshape(grad.size(0), -1)
    p = g / (g.sum(dim=1, keepdim=True) + eps)
    H = -(p * torch.log(p + eps)).sum(dim=1).mean()
    return H


def train_one_epoch(
    model,
    loader,
    opt,
    device,
    lambda_entropy: float,
    scaler=None,
    entropy_model=None,
    grad_clip: float | None = None,
):
    model.train()
    total = 0.0
    n = 0

    use_amp = (device.type == "cuda") and (scaler is not None)
    entropy_model = entropy_model or model

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)

        opt.zero_grad(set_to_none=True)
        sdp_ctx = _sdp_math_context(device, require_higher_order=(lambda_entropy > 0))

        with sdp_ctx:
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    ce = F.cross_entropy(logits, y)

                if lambda_entropy > 0:
                    # Keep entropy term in fp32 for stability.
                    H = entropy_reg_term(entropy_model, x.float(), y)
                    loss = ce.float() + lambda_entropy * H
                else:
                    loss = ce.float()
            else:
                logits = model(x)
                ce = F.cross_entropy(logits, y)
                if lambda_entropy > 0:
                    H = entropy_reg_term(entropy_model, x, y)
                    loss = ce + lambda_entropy * H
                else:
                    loss = ce

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

        total += float(loss.item()) * x.size(0)
        n += x.size(0)

    return total / max(n, 1)
