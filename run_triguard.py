import os
import argparse
import random

import numpy as np
import torch
from torch import optim

from triguard.data import get_loaders
from triguard.models import get_model, remove_dropout_layers
from triguard.train import train_one_epoch
from triguard.eval import (
    accuracy,
    pgd_accuracy,
    evaluate_attribution_metrics,
    evaluate_faithfulness,
    append_csv,
)
from triguard.plots import save_curve_plot


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_compile(model: torch.nn.Module, device: torch.device):
    # A100 / CUDA speedups: channels_last + torch.compile (PyTorch 2.x)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        if hasattr(torch, "compile"):
            model = torch.compile(model)
    return model


def train_with_early_stopping(
    model,
    train_loader,
    opt,
    device,
    epochs: int,
    lambda_entropy: float,
    scaler=None,
    entropy_model=None,
    patience: int = 3,
    min_delta: float = 1e-4,
):
    best = float("inf")
    bad = 0

    for ep in range(epochs):
        loss = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            lambda_entropy=lambda_entropy,
            scaler=scaler,
            entropy_model=entropy_model,
        )

        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        if loss is None:
            continue

        loss = float(loss)

        if loss < best - min_delta:
            best = loss
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/icml2026")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--lambda_entropy", type=float, default=0.05)

    p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--pgd_steps", type=int, default=40)
    p.add_argument("--eval_batches_adv", type=int, default=10)

    # for entropy/ADS/ADS-Adv/cert
    p.add_argument("--K_attr", type=int, default=100)
    # for deletion/insertion AUC
    p.add_argument("--K_faith", type=int, default=50)
    p.add_argument("--delins_steps", type=int, default=50)

    p.add_argument("--target_mode", type=str,
                   default="truth", choices=["truth", "pred"])

    # which experiment(s)
    p.add_argument(
        "--mode",
        type=str,
        default="main",
        choices=["main", "lambda", "baseline", "faithfulness"],
    )
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--lambda_list", type=str, default="0.0,0.01,0.05,0.1")

    # early stopping
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-4)

    # speed / iteration mode
    p.add_argument("--fast", action="store_true")
    p.add_argument("--deterministic", action="store_true")

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A100 / CUDA speedups
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    # Optional TF32 toggles (usually safe + faster on A100). Uncomment if desired.
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    if args.fast:
        args.epochs = min(args.epochs, 3)
        args.eval_batches_adv = min(args.eval_batches_adv, 3)
        args.K_attr = min(args.K_attr, 50)
        args.K_faith = min(args.K_faith, 20)
        args.ig_steps = min(args.ig_steps, 25)

    datasets = ["mnist", "fashionmnist",
                "cifar10"] if args.dataset is None else [args.dataset]
    models = (
        ["simplecnn", "resnet50", "resnet101", "mobilenetv3", "densenet121"]
        if args.model is None
        else [args.model]
    )

    # Output CSV paths
    table1_csv = os.path.join(args.out, "table1_main.csv")
    table2_csv = os.path.join(args.out, "table2_baseline_sensitivity.csv")
    table4_csv = os.path.join(args.out, "table4_lambda_ablation.csv")
    faith_csv = os.path.join(args.out, "tables5_6_faithfulness.csv")

    if args.mode == "main":
        header = [
            "dataset",
            "model",
            "seed",
            "lambda_entropy",
            "clean_acc",
            "adv_error",
            "entropy_mean",
            "ads_mean",
            "ads_adv_mean",
            "crown_rate",
        ]

        for ds in datasets:
            train_loader, test_loader, test_set, clamp_min, clamp_max, eps = get_loaders(
                ds, args.batch)

            for m in models:
                base_model = get_model(m, ds).to(device)
                model = maybe_compile(base_model, device)
                entropy_model = base_model if model is not base_model else model
                eval_model = base_model if model is not base_model else model

                opt = optim.Adam(model.parameters(), lr=args.lr)

                train_with_early_stopping(
                    model,
                    train_loader,
                    opt,
                    device,
                    epochs=args.epochs,
                    lambda_entropy=args.lambda_entropy,
                    scaler=scaler,
                    entropy_model=entropy_model,
                    patience=args.patience,
                    min_delta=args.min_delta,
                )

                alpha = eps / 8.0
                remove_dropout_layers(eval_model)
                clean = accuracy(eval_model, test_loader, device)
                adv_acc = pgd_accuracy(
                    eval_model,
                    test_loader,
                    device,
                    eps,
                    alpha,
                    args.pgd_steps,
                    clamp_min,
                    clamp_max,
                    max_batches=args.eval_batches_adv,
                )
                adv_error = 1.0 - adv_acc

                attr = evaluate_attribution_metrics(
                    eval_model,
                    test_set,
                    device,
                    eps=eps,
                    alpha=alpha,
                    pgd_steps=args.pgd_steps,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
                    ig_steps=args.ig_steps,
                    K=args.K_attr,
                    seed=args.seed,
                    target_mode=args.target_mode,
                )

                row = {
                    "dataset": ds,
                    "model": m,
                    "seed": args.seed,
                    "lambda_entropy": args.lambda_entropy,
                    "clean_acc": clean,
                    "adv_error": adv_error,
                    **attr,
                }
                append_csv(table1_csv, row, header)
                print("Wrote row:", row)

        print("DONE. CSV:", table1_csv)

    elif args.mode == "lambda":
        header = [
            "dataset",
            "model",
            "seed",
            "lambda_entropy",
            "clean_acc",
            "adv_error",
            "entropy_mean",
            "ads_mean",
            "ads_adv_mean",
            "crown_rate",
        ]
        lam_list = [float(x.strip()) for x in args.lambda_list.split(",")]

        for lam in lam_list:
            for ds in datasets:
                train_loader, test_loader, test_set, clamp_min, clamp_max, eps = get_loaders(
                    ds, args.batch)

                for m in models:
                    base_model = get_model(m, ds).to(device)
                    model = maybe_compile(base_model, device)
                    entropy_model = base_model if model is not base_model else model
                    eval_model = base_model if model is not base_model else model

                    opt = optim.Adam(model.parameters(), lr=args.lr)

                    train_with_early_stopping(
                        model,
                        train_loader,
                        opt,
                        device,
                        epochs=args.epochs,
                        lambda_entropy=lam,
                        scaler=scaler,
                        entropy_model=entropy_model,
                        patience=args.patience,
                        min_delta=args.min_delta,
                    )

                    alpha = eps / 8.0
                    remove_dropout_layers(eval_model)
                    clean = accuracy(eval_model, test_loader, device)
                    adv_acc = pgd_accuracy(
                        eval_model,
                        test_loader,
                        device,
                        eps,
                        alpha,
                        args.pgd_steps,
                        clamp_min,
                        clamp_max,
                        max_batches=args.eval_batches_adv,
                    )
                    adv_error = 1.0 - adv_acc

                    attr = evaluate_attribution_metrics(
                        eval_model,
                        test_set,
                        device,
                        eps=eps,
                        alpha=alpha,
                        pgd_steps=args.pgd_steps,
                        clamp_min=clamp_min,
                        clamp_max=clamp_max,
                        ig_steps=args.ig_steps,
                        K=args.K_attr,
                        seed=args.seed,
                        target_mode=args.target_mode,
                    )

                    row = {
                        "dataset": ds,
                        "model": m,
                        "seed": args.seed,
                        "lambda_entropy": lam,
                        "clean_acc": clean,
                        "adv_error": adv_error,
                        **attr,
                    }
                    append_csv(table4_csv, row, header)
                    print("Wrote row:", row)

        print("DONE. CSV:", table4_csv)

    elif args.mode == "baseline":
        from triguard.attributions import blurred_baseline, ads_baseline

        header = [
            "dataset",
            "model",
            "seed",
            "K",
            "ads_zero_blur",
            "ads_zero_noise",
            "ads_zero_uniform",
            "ads_blur_noise",
            "ads_blur_uniform",
            "ads_noise_uniform",
        ]

        for ds in datasets:
            train_loader, test_loader, test_set, clamp_min, clamp_max, eps = get_loaders(
                ds, args.batch)

            for m in models:
                base_model = get_model(m, ds).to(device)
                model = maybe_compile(base_model, device)
                entropy_model = base_model if model is not base_model else model
                eval_model = base_model if model is not base_model else model

                opt = optim.Adam(model.parameters(), lr=args.lr)

                train_with_early_stopping(
                    model,
                    train_loader,
                    opt,
                    device,
                    epochs=args.epochs,
                    lambda_entropy=args.lambda_entropy,
                    scaler=scaler,
                    entropy_model=entropy_model,
                    patience=args.patience,
                    min_delta=args.min_delta,
                )

                remove_dropout_layers(eval_model)

                rng = np.random.default_rng(args.seed)
                idxs = rng.choice(len(test_set), size=min(
                    args.K_attr, len(test_set)), replace=False)

                vals = {k: [] for k in header if k not in [
                    "dataset", "model", "seed", "K"]}

                for idx in idxs:
                    x, y = test_set[idx]
                    y = int(y)
                    x = x.to(device).unsqueeze(0)

                    target = y if args.target_mode == "truth" else int(
                        eval_model(x).argmax(dim=1).item())

                    b0 = torch.zeros_like(x)
                    b_blur = blurred_baseline(x)
                    b_noise = torch.randn_like(x) * 0.1
                    b_uniform = torch.rand_like(x)

                    vals["ads_zero_blur"].append(ads_baseline(
                        eval_model, x, target, b0, b_blur, steps=args.ig_steps))
                    vals["ads_zero_noise"].append(ads_baseline(
                        eval_model, x, target, b0, b_noise, steps=args.ig_steps))
                    vals["ads_zero_uniform"].append(ads_baseline(
                        eval_model, x, target, b0, b_uniform, steps=args.ig_steps))
                    vals["ads_blur_noise"].append(ads_baseline(
                        eval_model, x, target, b_blur, b_noise, steps=args.ig_steps))
                    vals["ads_blur_uniform"].append(
                        ads_baseline(eval_model, x, target, b_blur,
                                     b_uniform, steps=args.ig_steps)
                    )
                    vals["ads_noise_uniform"].append(
                        ads_baseline(eval_model, x, target, b_noise,
                                     b_uniform, steps=args.ig_steps)
                    )

                row = {
                    "dataset": ds,
                    "model": m,
                    "seed": args.seed,
                    "K": len(idxs),
                    **{k: float(np.mean(v)) for k, v in vals.items()},
                }
                append_csv(table2_csv, row, header)
                print("Wrote row:", row)

        print("DONE. CSV:", table2_csv)

    elif args.mode == "faithfulness":
        header = [
            "dataset",
            "model",
            "seed",
            "K",
            "ig_del_auc_mean",
            "ig_ins_auc_mean",
            "sg2_del_auc_mean",
            "sg2_ins_auc_mean",
        ]

        for ds in datasets:
            train_loader, test_loader, test_set, clamp_min, clamp_max, eps = get_loaders(
                ds, args.batch)

            for m in models:
                base_model = get_model(m, ds).to(device)
                model = maybe_compile(base_model, device)
                entropy_model = base_model if model is not base_model else model
                eval_model = base_model if model is not base_model else model

                opt = optim.Adam(model.parameters(), lr=args.lr)

                train_with_early_stopping(
                    model,
                    train_loader,
                    opt,
                    device,
                    epochs=args.epochs,
                    lambda_entropy=args.lambda_entropy,
                    scaler=scaler,
                    entropy_model=entropy_model,
                    patience=args.patience,
                    min_delta=args.min_delta,
                )

                remove_dropout_layers(eval_model)

                res = evaluate_faithfulness(
                    eval_model,
                    test_set,
                    device,
                    K=args.K_faith,
                    ig_steps=args.ig_steps,
                    delins_steps=args.delins_steps,
                    seed=args.seed,
                    baseline_mode="zero",
                    target_mode=args.target_mode,
                    smoothgrad_noise=0.1,
                    smoothgrad_samples=50,
                )

                row = {
                    "dataset": ds,
                    "model": m,
                    "seed": args.seed,
                    "K": args.K_faith,
                    "ig_del_auc_mean": res["ig_del_auc_mean"],
                    "ig_ins_auc_mean": res["ig_ins_auc_mean"],
                    "sg2_del_auc_mean": res["sg2_del_auc_mean"],
                    "sg2_ins_auc_mean": res["sg2_ins_auc_mean"],
                }
                append_csv(faith_csv, row, header)
                print("Wrote row:", row)

                fig_dir = os.path.join(
                    args.out, "figures", "faithfulness", ds, m)
                for i, (tag, del_curve, ins_curve) in enumerate(res["curves"][:4]):
                    outpath = os.path.join(fig_dir, f"{tag}_curve_{i}.png")
                    save_curve_plot(del_curve, ins_curve,
                                    f"{ds}/{m} {tag}", outpath)

        print("DONE. CSV:", faith_csv)


if __name__ == "__main__":
    main()
