import argparse
import os
import random

import numpy as np
import torch
from torch import optim

from triguard.data import get_loaders
from triguard.eval import (
    accuracy,
    append_csv,
    evaluate_certification,
    evaluate_appendix_metrics,
    evaluate_faithfulness,
    evaluate_main_metrics,
    pgd_accuracy,
)
from triguard.models import get_model
from triguard.models import uses_imagenet_preprocessing
from triguard.plots import save_curve_plot
from triguard.train import train_one_epoch


WORKSHOP_2026_GRID = {
    "mnist": ["simplecnn", "resnet50"],
    "fashionmnist": ["simplecnn", "resnet50"],
    "cifar10": ["resnet50", "densenet121", "vit_b_16"],
    "cifar100": ["resnet50", "densenet121", "vit_b_16"],
}

PRETRAINED_GRID = {
    "cifar10": [
        "resnet50_imagenet",
        "densenet121_imagenet",
        "vit_b_16_imagenet",
        "convnext_tiny_imagenet",
        "swin_t_imagenet",
    ],
    "cifar100": [
        "resnet50_imagenet",
        "densenet121_imagenet",
        "vit_b_16_imagenet",
        "convnext_tiny_imagenet",
        "swin_t_imagenet",
    ],
}

DATASET_ORDER = ["mnist", "fashionmnist", "cifar10", "cifar100"]
MODEL_ORDER = [
    "simplecnn",
    "resnet50",
    "densenet121",
    "vit_b_16",
    "resnet50_imagenet",
    "densenet121_imagenet",
    "vit_b_16_imagenet",
    "convnext_tiny_imagenet",
    "swin_t_imagenet",
]
DEFAULT_EPOCHS = {
    "mnist": 5,
    "fashionmnist": 5,
    "cifar10": 10,
    "cifar100": 15,
}
PRETRAINED_DEFAULT_EPOCHS = {
    "cifar10": 20,
    "cifar100": 30,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_compile(
    model: torch.nn.Module,
    device: torch.device,
    model_name: str,
    enable_compile: bool = False,
):
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        # torch.compile can improve throughput, but it has been less reliable
        # than eager mode across our benchmark settings.
        if enable_compile and hasattr(torch, "compile") and model_name.lower() not in {"vit_b_16", "vit"}:
            model = torch.compile(model)
    return model


def parse_seeds(seed: int, seeds_arg: str | None):
    if not seeds_arg:
        return [seed]
    return [int(x.strip()) for x in seeds_arg.split(",") if x.strip()]


def resolve_epochs(args, dataset: str, model_name: str) -> int:
    if args.epochs is not None:
        return args.epochs
    if uses_imagenet_preprocessing(model_name):
        return PRETRAINED_DEFAULT_EPOCHS.get(dataset.lower(), 20)
    return DEFAULT_EPOCHS.get(dataset.lower(), 5)


def resolve_cert_eps(args, attack_eps: float, pixel_eps: float) -> float:
    if args.cert_eps is not None:
        return float(args.cert_eps)
    if args.cert_pixel_eps is not None:
        return float(attack_eps * (args.cert_pixel_eps / pixel_eps))
    return float(attack_eps * args.cert_eps_scale)


def build_training_setup(args, dataset: str, model_name: str, model, epochs: int):
    dataset = dataset.lower()
    model_name = model_name.lower()

    if uses_imagenet_preprocessing(model_name):
        lr = args.lr if args.lr is not None else 5e-5
        opt = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs - args.warmup_epochs, 1)
        )
        if args.warmup_epochs > 0 and epochs > args.warmup_epochs:
            scheduler = optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[
                    optim.lr_scheduler.LinearLR(
                        opt, start_factor=0.1, total_iters=args.warmup_epochs
                    ),
                    main_scheduler,
                ],
                milestones=[args.warmup_epochs],
            )
        else:
            scheduler = main_scheduler
        grad_clip = args.grad_clip if args.grad_clip is not None else 1.0
        return {
            "optimizer": opt,
            "scheduler": scheduler,
            "grad_clip": grad_clip,
            "optimizer_name": "adamw_imagenet",
            "lr": lr,
        }

    if dataset in {"cifar10", "cifar100"} and model_name in {"resnet50", "densenet121"}:
        lr = args.lr if args.lr is not None else 0.05
        opt = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        grad_clip = args.grad_clip if args.grad_clip is not None else 1.0
        return {
            "optimizer": opt,
            "scheduler": scheduler,
            "grad_clip": grad_clip,
            "optimizer_name": "sgd",
            "lr": lr,
        }

    if dataset in {"cifar10", "cifar100"} and model_name in {"vit_b_16", "vit"}:
        lr = args.lr if args.lr is not None else 3e-4
        opt = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.05,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        grad_clip = args.grad_clip if args.grad_clip is not None else 1.0
        return {
            "optimizer": opt,
            "scheduler": scheduler,
            "grad_clip": grad_clip,
            "optimizer_name": "adamw",
            "lr": lr,
        }

    lr = args.lr if args.lr is not None else 1e-3
    opt = optim.Adam(model.parameters(), lr=lr)
    return {
        "optimizer": opt,
        "scheduler": None,
        "grad_clip": args.grad_clip,
        "optimizer_name": "adam",
        "lr": lr,
    }


def train_with_early_stopping(
    model,
    train_loader,
    opt,
    device,
    epochs: int,
    lambda_entropy: float,
    scaler=None,
    entropy_model=None,
    scheduler=None,
    grad_clip: float | None = None,
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
            grad_clip=grad_clip,
        )
        if scheduler is not None:
            scheduler.step()
        loss = float(loss)
        if loss < best - min_delta:
            best = loss
            bad = 0
            best_state = {k: v.detach().cpu()
                          for k, v in model.state_dict().items()}
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


def merged_grid(grid_name: str):
    grid_name = grid_name.lower()
    if grid_name == "workshop":
        return WORKSHOP_2026_GRID
    if grid_name == "pretrained":
        return PRETRAINED_GRID
    if grid_name == "all":
        merged = {ds: list(models) for ds, models in WORKSHOP_2026_GRID.items()}
        for ds, models in PRETRAINED_GRID.items():
            merged.setdefault(ds, [])
            for model in models:
                if model not in merged[ds]:
                    merged[ds].append(model)
        return merged
    raise ValueError(f"Unknown grid: {grid_name}")


def get_experiment_grid(dataset: str | None, model: str | None, grid_name: str):
    grid = merged_grid(grid_name)
    if dataset and model:
        return [(dataset.lower(), model.lower())]

    if dataset:
        ds = dataset.lower()
        if ds not in grid:
            raise ValueError(f"Unknown dataset: {ds}")
        return [(ds, m) for m in grid[ds]]

    if model:
        m = model.lower()
        pairs = []
        for ds in DATASET_ORDER:
            if ds in grid and m in grid[ds]:
                pairs.append((ds, m))
        if not pairs:
            raise ValueError(f"Model {m} is not used in the {grid_name} grid.")
        return pairs

    pairs = []
    for ds in DATASET_ORDER:
        for m in grid.get(ds, []):
            pairs.append((ds, m))
    return pairs


def run_one_setting(args, seed: int, ds: str, model_name: str, lam: float, device: torch.device, mode: str):
    set_seed(seed)
    epochs = resolve_epochs(args, ds, model_name)
    input_profile = "imagenet" if uses_imagenet_preprocessing(model_name) else "native"
    train_loader, test_loader, test_set, clamp_min, clamp_max, eps, meta = get_loaders(
        ds,
        args.batch,
        num_workers=args.num_workers,
        data_root=args.data_root,
        input_profile=input_profile,
    )

    num_classes = int(meta["num_classes"])
    baseline_min = float(meta["baseline_min"])
    baseline_max = float(meta["baseline_max"])
    pixel_eps = float(meta["pixel_eps"])

    base_model = get_model(model_name, ds, num_classes=num_classes).to(device)
    model = maybe_compile(base_model, device, model_name, enable_compile=args.compile)
    entropy_model = base_model if model is not base_model else model
    eval_model = base_model if model is not base_model else model
    training_setup = build_training_setup(args, ds, model_name, model, epochs)
    opt = training_setup["optimizer"]
    scheduler = training_setup["scheduler"]
    grad_clip = training_setup["grad_clip"]

    print(
        f"[Train Config] {ds}/{model_name}: "
        f"epochs={epochs}, input_profile={input_profile}, optimizer={training_setup['optimizer_name']}, "
        f"lr={training_setup['lr']}, scheduler={'cosine' if scheduler is not None else 'none'}, "
        f"grad_clip={grad_clip}, compile={args.compile}"
    )

    ckpt_path = None
    if args.save_ckpt:
        ckpt_path = os.path.join(
            args.out,
            "checkpoints",
            f"{mode}_{ds}_{model_name}_seed{seed}_lam{lam:.3f}.pt",
        )

    if args.load_ckpt:
        state = torch.load(args.load_ckpt, map_location=device)
        eval_model.load_state_dict(state)

    if not args.eval_only:
        train_with_early_stopping(
            model,
            train_loader,
            opt,
            device,
            epochs=epochs,
            lambda_entropy=lam,
            scaler=(torch.amp.GradScaler("cuda")
                    if device.type == "cuda" else None),
            entropy_model=entropy_model,
            scheduler=scheduler,
            grad_clip=grad_clip,
            patience=args.patience,
            min_delta=args.min_delta,
            ckpt_path=ckpt_path,
        )

    eval_model.eval()
    alpha = eps / 8.0
    cert_eps = resolve_cert_eps(args, eps, pixel_eps)

    if mode in ["main", "lambda"]:
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
            cert_eps=cert_eps,
            baseline_min=baseline_min,
            baseline_max=baseline_max,
        )
        return {
            "dataset": ds,
            "model": model_name,
            "seed": seed,
            "input_profile": input_profile,
            "lambda_entropy": lam,
            "cert_eps": cert_eps,
            "clean_acc": clean,
            "adv_error": adv_error,
            **main_metrics,
        }

    if mode == "cert_sweep":
        rows = []
        cert_pixel_values = [
            float(x.strip()) for x in args.cert_eps_list.split(",") if x.strip()
        ]
        for pixel_value in cert_pixel_values:
            active_cert_eps = eps * (pixel_value / pixel_eps)
            cert_metrics = evaluate_certification(
                eval_model,
                test_set,
                device,
                cert_eps=active_cert_eps,
                k=args.K_attr,
                seed=seed,
            )
            rows.append(
                {
                    "dataset": ds,
                    "model": model_name,
                    "seed": seed,
                    "input_profile": input_profile,
                    "lambda_entropy": lam,
                    "cert_pixel_eps": pixel_value,
                    **cert_metrics,
                }
            )
        return rows

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
            "input_profile": input_profile,
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
        fig_dir = os.path.join(args.out, "figures",
                               "faithfulness", ds, model_name, f"seed_{seed}")
        for i, (tag, del_curve, ins_curve) in enumerate(res["curves"][:4]):
            save_curve_plot(
                del_curve,
                ins_curve,
                f"{ds}/{model_name} {tag}",
                os.path.join(fig_dir, f"{tag}_curve_{i}.png"),
            )
        return {
            "dataset": ds,
            "model": model_name,
            "seed": seed,
            "input_profile": input_profile,
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
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda_entropy", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--pgd_steps", type=int, default=40)
    p.add_argument("--cert_eps", type=float, default=None)
    p.add_argument("--cert_pixel_eps", type=float, default=None)
    p.add_argument("--cert_eps_scale", type=float, default=1.0)
    p.add_argument("--cert_eps_list", type=str, default="0.0,0.0039215686,0.0078431373,0.0156862745,0.031372549")
    p.add_argument("--eval_batches_adv", type=int, default=10)
    p.add_argument("--K_attr", type=int, default=100)
    p.add_argument("--K_faith", type=int, default=50)
    p.add_argument("--delins_steps", type=int, default=50)
    p.add_argument("--target_mode", type=str,
                   default="truth", choices=["truth", "pred"])
    p.add_argument("--mode", type=str, default="main",
                   choices=["main", "lambda", "baseline", "faithfulness", "cert_sweep"])
    p.add_argument("--grid", type=str, default="workshop",
                   choices=["workshop", "pretrained", "all"])
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--lambda_list", type=str, default="0.0,0.01,0.05,0.1")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--skip_crown", action="store_true")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--load_ckpt", type=str, default="")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--bound_probe_samples", type=int, default=16)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--compile", action="store_true")
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
        args.epochs = min(args.epochs, 3) if args.epochs is not None else 3
        args.eval_batches_adv = min(args.eval_batches_adv, 3)
        args.K_attr = min(args.K_attr, 50)
        args.K_faith = min(args.K_faith, 20)
        args.ig_steps = min(args.ig_steps, 25)

    seeds = parse_seeds(args.seed, args.seeds)
    experiment_pairs = get_experiment_grid(args.dataset, args.model, args.grid)
    if args.eval_only and not args.load_ckpt:
        p.error("--eval_only requires --load_ckpt.")
    if args.load_ckpt and len(experiment_pairs) != 1:
        p.error("--load_ckpt can only be used with a single --dataset/--model setting.")

    if args.mode == "main":
        out_csv = os.path.join(args.out, "table1_main.csv")
        header = [
            "dataset",
            "model",
            "seed",
            "input_profile",
            "lambda_entropy",
            "cert_eps",
            "clean_acc",
            "adv_error",
            "bound_check_rate",
            "crown_rate",
            "entropy_mean",
            "ads_mean",
            "entropy_valid_n",
            "ads_valid_n",
        ]
        for seed in seeds:
            for ds, model_name in experiment_pairs:
                row = run_one_setting(
                    args, seed, ds, model_name, args.lambda_entropy, device, mode="main")
                append_csv(out_csv, row, header)
                print("Wrote row:", row)

    elif args.mode == "lambda":
        out_csv = os.path.join(args.out, "table4_lambda_ablation.csv")
        header = [
            "dataset",
            "model",
            "seed",
            "input_profile",
            "lambda_entropy",
            "cert_eps",
            "clean_acc",
            "adv_error",
            "bound_check_rate",
            "crown_rate",
            "entropy_mean",
            "ads_mean",
            "entropy_valid_n",
            "ads_valid_n",
        ]
        lam_list = [float(x.strip())
                    for x in args.lambda_list.split(",") if x.strip()]
        active_dataset = args.dataset or "mnist"
        active_model = args.model or "simplecnn"
        for seed in seeds:
            for lam in lam_list:
                row = run_one_setting(
                    args, seed, active_dataset, active_model, lam, device, mode="lambda")
                append_csv(out_csv, row, header)
                print("Wrote row:", row)

    elif args.mode == "baseline":
        out_csv = os.path.join(args.out, "table2_baseline_sensitivity.csv")
        header = [
            "dataset",
            "model",
            "seed",
            "input_profile",
            "K",
            "ads_zero_blur",
            "ads_zero_noise",
            "ads_zero_uniform",
            "ads_blur_noise",
            "ads_blur_uniform",
            "ads_noise_uniform",
        ]
        active_dataset = args.dataset or "mnist"
        active_model = args.model or "simplecnn"
        for seed in seeds:
            row = run_one_setting(args, seed, active_dataset, active_model,
                                  args.lambda_entropy, device, mode="baseline")
            append_csv(out_csv, row, header)
            print("Wrote row:", row)

    elif args.mode == "faithfulness":
        out_csv = os.path.join(args.out, "tables5_6_faithfulness.csv")
        header = [
            "dataset",
            "model",
            "seed",
            "input_profile",
            "K",
            "ig_del_auc_mean",
            "ig_ins_auc_mean",
            "sg2_del_auc_mean",
            "sg2_ins_auc_mean",
        ]
        for seed in seeds:
            for ds, model_name in experiment_pairs:
                row = run_one_setting(
                    args, seed, ds, model_name, args.lambda_entropy, device, mode="faithfulness")
                append_csv(out_csv, row, header)
                print("Wrote row:", row)

    elif args.mode == "cert_sweep":
        out_csv = os.path.join(args.out, "table_certification_sweep.csv")
        header = [
            "dataset",
            "model",
            "seed",
            "input_profile",
            "lambda_entropy",
            "cert_pixel_eps",
            "cert_eps",
            "crown_rate",
            "cert_valid_n",
        ]
        for seed in seeds:
            for ds, model_name in experiment_pairs:
                rows = run_one_setting(
                    args, seed, ds, model_name, args.lambda_entropy, device, mode="cert_sweep")
                for row in rows:
                    append_csv(out_csv, row, header)
                    print("Wrote row:", row)


if __name__ == "__main__":
    main()
