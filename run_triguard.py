import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import optim

from triguard.data import get_loaders
from triguard.eval import (
    accuracy,
    append_csv,
    evaluate_appendix_metrics,
    evaluate_faithfulness,
    evaluate_main_metrics,
    pgd_accuracy,
)
from triguard.models import get_model
from triguard.plots import save_curve_plot
from triguard.train import train_one_epoch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_compile(model: torch.nn.Module, device: torch.device):
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        if hasattr(torch, "compile"):
            model = torch.compile(model)
    return model


def parse_seeds(seed: int, seeds_arg: str | None):
    if not seeds_arg:
        return [seed]
    return [int(x.strip()) for x in seeds_arg.split(",") if x.strip()]


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
    ckpt_path: str | None = None,
):
    best = float("inf")
    bad = 0
    best_state = None

    for _ in range(epochs):
        loss = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            lambda_entropy=lambda_entropy,
            scaler=scaler,
            entropy_model=entropy_model,
        )
        loss = float(loss)
        if loss < best - min_delta:
            best = loss
            bad = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_path is not None:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(best_state, ckpt_path)
    return best


def default_models(dataset: str | None):
    if dataset and dataset.lower() == "cifar100":
        return ["resnet50", "vit_b_16"]
    return ["simplecnn", "resnet50", "resnet101", "mobilenetv3", "densenet121"]


def run_one_setting(args, seed: int, ds: str, model_name: str, lam: float, device: torch.device, mode: str):
    set_seed(seed)
    train_loader, test_loader, test_set, clamp_min, clamp_max, eps, meta = get_loaders(
        ds,
        args.batch,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )

    num_classes = int(meta["num_classes"])
    baseline_min = float(meta["baseline_min"])
    baseline_max = float(meta["baseline_max"])

    base_model = get_model(model_name, ds, num_classes=num_classes).to(device)
    model = maybe_compile(base_model, device)
    entropy_model = base_model if model is not base_model else model
    eval_model = base_model if model is not base_model else model
    opt = optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = None
    if args.save_ckpt:
        ckpt_path = os.path.join(args.out, "checkpoints", f"{mode}_{ds}_{model_name}_seed{seed}_lam{lam:.3f}.pt")

    train_with_early_stopping(
        model,
        train_loader,
        opt,
        device,
        epochs=args.epochs,
        lambda_entropy=lam,
        scaler=(torch.amp.GradScaler("cuda") if device.type == "cuda" else None),
        entropy_model=entropy_model,
        patience=args.patience,
        min_delta=args.min_delta,
        ckpt_path=ckpt_path,
    )

    eval_model.eval()
    alpha = eps / 8.0
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

    if mode in ["main", "lambda"]:
        main_metrics = evaluate_main_metrics(
            eval_model,
            test_set,
            device,
            eps=eps,
            alpha=alpha,
            pgd_steps=args.pgd_steps,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            ig_steps=args.ig_steps,
            k=args.K_attr,
            seed=seed,
            target_mode=args.target_mode,
            bound_probe_samples=args.bound_probe_samples,
            do_crown=(not args.skip_crown),
            baseline_min=baseline_min,
            baseline_max=baseline_max,
        )
        row = {
            "dataset": ds,
            "model": model_name,
            "seed": seed,
            "lambda_entropy": lam,
            "clean_acc": clean,
            "adv_error": adv_error,
            **main_metrics,
        }
        return row

    if mode == "baseline":
        appendix = evaluate_appendix_metrics(
            eval_model,
            test_set,
            device,
            eps=eps,
            alpha=alpha,
            pgd_steps=args.pgd_steps,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            ig_steps=args.ig_steps,
            k=args.K_attr,
            seed=seed,
            target_mode=args.target_mode,
            baseline_min=baseline_min,
            baseline_max=baseline_max,
        )
        return {
            "dataset": ds,
            "model": model_name,
            "seed": seed,
            "K": args.K_attr,
            **{k: appendix[k] for k in appendix if k.startswith("ads_") and k != "ads_adv_mean"},
        }

    if mode == "faithfulness":
        res = evaluate_faithfulness(
            eval_model,
            test_set,
            device,
            k=args.K_faith,
            ig_steps=args.ig_steps,
            delins_steps=args.delins_steps,
            seed=seed,
            baseline_mode="zero",
            target_mode=args.target_mode,
            smoothgrad_noise=0.1,
            smoothgrad_samples=50,
        )
        fig_dir = os.path.join(args.out, "figures", "faithfulness", ds, model_name, f"seed_{seed}")
        for i, (tag, del_curve, ins_curve) in enumerate(res["curves"][:4]):
            save_curve_plot(del_curve, ins_curve, f"{ds}/{model_name} {tag}", os.path.join(fig_dir, f"{tag}_curve_{i}.png"))
        return {
            "dataset": ds,
            "model": model_name,
            "seed": seed,
            "K": args.K_faith,
            "ig_del_auc_mean": res["ig_del_auc_mean"],
            "ig_ins_auc_mean": res["ig_ins_auc_mean"],
            "sg2_del_auc_mean": res["sg2_del_auc_mean"],
            "sg2_ins_auc_mean": res["sg2_ins_auc_mean"],
        }

    raise ValueError(f"Unsupported mode: {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/icml2026")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_entropy", type=float, default=0.05)
    p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--pgd_steps", type=int, default=40)
    p.add_argument("--eval_batches_adv", type=int, default=10)
    p.add_argument("--K_attr", type=int, default=100)
    p.add_argument("--K_faith", type=int, default=50)
    p.add_argument("--delins_steps", type=int, default=50)
    p.add_argument("--target_mode", type=str, default="truth", choices=["truth", "pred"])
    p.add_argument("--mode", type=str, default="main", choices=["main", "lambda", "baseline", "faithfulness"])
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--lambda_list", type=str, default="0.0,0.01,0.05,0.1")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--skip_crown", action="store_true")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--bound_probe_samples", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if args.fast:
        args.epochs = min(args.epochs, 3)
        args.eval_batches_adv = min(args.eval_batches_adv, 3)
        args.K_attr = min(args.K_attr, 50)
        args.K_faith = min(args.K_faith, 20)
        args.ig_steps = min(args.ig_steps, 25)

    seeds = parse_seeds(args.seed, args.seeds)
    datasets = [args.dataset] if args.dataset else ["mnist", "fashionmnist", "cifar10"]
    models = [args.model] if args.model else default_models(args.dataset)

    if args.mode == "main":
        out_csv = os.path.join(args.out, "table1_main.csv")
        header = ["dataset", "model", "seed", "lambda_entropy", "clean_acc", "adv_error", "bound_check_rate", "crown_rate", "entropy_mean", "ads_mean"]
        for seed in seeds:
            for ds in datasets:
                active_models = [args.model] if args.model else default_models(ds)
                for model_name in active_models:
                    row = run_one_setting(args, seed, ds, model_name, args.lambda_entropy, device, mode="main")
                    append_csv(out_csv, row, header)
                    print("Wrote row:", row)

    elif args.mode == "lambda":
        out_csv = os.path.join(args.out, "table4_lambda_ablation.csv")
        header = ["dataset", "model", "seed", "lambda_entropy", "clean_acc", "adv_error", "bound_check_rate", "crown_rate", "entropy_mean", "ads_mean"]
        lam_list = [float(x.strip()) for x in args.lambda_list.split(",") if x.strip()]
        active_datasets = [args.dataset or "mnist"]
        active_models = [args.model or "simplecnn"]
        for seed in seeds:
            for lam in lam_list:
                for ds in active_datasets:
                    for model_name in active_models:
                        row = run_one_setting(args, seed, ds, model_name, lam, device, mode="lambda")
                        append_csv(out_csv, row, header)
                        print("Wrote row:", row)

    elif args.mode == "baseline":
        out_csv = os.path.join(args.out, "table2_baseline_sensitivity.csv")
        header = ["dataset", "model", "seed", "K", "ads_zero_blur", "ads_zero_noise", "ads_zero_uniform", "ads_blur_noise", "ads_blur_uniform", "ads_noise_uniform"]
        active_datasets = [args.dataset or "mnist"]
        active_models = [args.model or "simplecnn"]
        for seed in seeds:
            for ds in active_datasets:
                for model_name in active_models:
                    row = run_one_setting(args, seed, ds, model_name, args.lambda_entropy, device, mode="baseline")
                    append_csv(out_csv, row, header)
                    print("Wrote row:", row)

    elif args.mode == "faithfulness":
        out_csv = os.path.join(args.out, "tables5_6_faithfulness.csv")
        header = ["dataset", "model", "seed", "K", "ig_del_auc_mean", "ig_ins_auc_mean", "sg2_del_auc_mean", "sg2_ins_auc_mean"]
        for seed in seeds:
            for ds in datasets:
                active_models = [args.model] if args.model else default_models(ds)
                for model_name in active_models:
                    row = run_one_setting(args, seed, ds, model_name, args.lambda_entropy, device, mode="faithfulness")
                    append_csv(out_csv, row, header)
                    print("Wrote row:", row)


if __name__ == "__main__":
    main()
