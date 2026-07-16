import argparse
import os

import pandas as pd
import torch

from .attacks import pgd_linf
from .attributions import blurred_baseline, integrated_gradients
from .data import get_dataset
from .models import get_model, uses_imagenet_preprocessing
from .plots import save_correlation_heatmap, save_saliency_panel, save_tradeoff_plot
from .protocol import validate_protocol_values


CONFIG_COLUMNS = [
    "condition_hash",
    "code_hash",
    "lambda_entropy",
    "lambda_wads",
    "lambda_rar",
    "lambda_far",
    "lambda_curvature",
    "lambda_robust",
    "lambda_attr_mass",
    "reference_risk",
    "reference_cvar_alpha",
    "reference_distance",
    "triguard_ig_steps",
    "reference_pair_samples",
    "reference_bank_samples",
    "eval_reference_bank_samples",
    "regularizer_microbatch",
    "vectorized_reference_ig",
    "checkpoint_regularizer_ig",
    "sampled_mass_penalty",
    "preload_reference_banks",
    "training_reservation_hash",
    "reference_bank_hash",
    "heldout_reference_bank_hash",
    "attack_suite",
]


def _scale_eps(eps, scale: float):
    if isinstance(eps, (list, tuple)):
        return tuple(float(v) * scale for v in eps)
    return float(eps) * scale


def make_aggregate_figures(out_dir):
    main_csv = os.path.join(out_dir, "table1_main.csv")
    corr_csv = os.path.join(out_dir, "metric_correlations.csv")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if os.path.exists(corr_csv):
        corr_df = pd.read_csv(corr_csv)
        if "dataset" in corr_df.columns:
            for dataset, group in corr_df.groupby("dataset", dropna=False):
                save_correlation_heatmap(
                    group,
                    f"Spearman correlation: {dataset}",
                    os.path.join(fig_dir, f"metric_correlation_{dataset}.png"),
                )
        else:
            save_correlation_heatmap(
                corr_df,
                "Spearman correlation among TriGuard metrics",
                os.path.join(fig_dir, "metric_correlation_heatmap.png"),
            )

    if os.path.exists(main_csv):
        main_df = pd.read_csv(main_csv)
        validate_protocol_values(main_df["protocol_version"], main_csv)
        config = [column for column in CONFIG_COLUMNS if column in main_df.columns]
        group_cols = ["dataset", "model", *config]
        metrics = [
            metric
            for metric in ["clean_acc", "wads_mean", "heldout_wads_mean"]
            if metric in main_df.columns
        ]
        group = main_df.groupby(group_cols, dropna=False, as_index=False)[
            metrics
        ].mean(numeric_only=True)
        group["series"] = group.apply(
            lambda row: (
                f"{row['model']}|risk={row.get('reference_risk', 'n/a')}"
                f"|w={row.get('lambda_wads', 0):.3g}"
                f"|curv={row.get('lambda_curvature', 0):.3g}"
                f"|rob={row.get('lambda_robust', 0):.3g}"
                f"|mass={row.get('lambda_attr_mass', 0):.3g}"
            ),
            axis=1,
        )
        for dataset, dataset_group in group.groupby("dataset", dropna=False):
            save_tradeoff_plot(
                dataset_group,
                "clean_acc",
                "wads_mean",
                f"{dataset}: accuracy versus reference risk",
                os.path.join(fig_dir, f"tradeoff_wads_{dataset}.png"),
            )
            if "heldout_wads_mean" in dataset_group.columns:
                save_tradeoff_plot(
                    dataset_group,
                    "clean_acc",
                    "heldout_wads_mean",
                    f"{dataset}: accuracy versus held-out reference risk",
                    os.path.join(
                        fig_dir, f"tradeoff_heldout_wads_{dataset}.png"
                    ),
                )


def _collapse_attr(attr):
    attr = attr.detach().abs().sum(dim=1).squeeze(0)
    return attr.cpu().numpy()


def make_saliency_panel(args):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for saliency panels.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    input_profile = "imagenet" if uses_imagenet_preprocessing(args.model) else "native"
    _, test_set, clamp_min, clamp_max, eps, meta = get_dataset(
        args.dataset,
        data_root=args.data_root,
        input_profile=input_profile,
    )
    num_classes = int(meta["num_classes"])
    model = get_model(args.model, args.dataset, num_classes=num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    chosen = None
    for idx in range(min(args.search_k, len(test_set))):
        x, y = test_set[idx]
        y = int(y)
        x_batch = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x_batch).argmax(dim=1).item())
        if pred == y:
            chosen = (x, y)
            break
    if chosen is None:
        chosen = test_set[0]

    x, y = chosen
    y = int(y)
    x_batch = x.unsqueeze(0).to(device)
    with torch.no_grad():
        target = int(model(x_batch).argmax(dim=1).item())

    b0 = torch.zeros_like(x_batch)
    b_blur = blurred_baseline(x_batch)
    ig_zero = integrated_gradients(model, x_batch, target, b0, steps=args.ig_steps)
    ig_blur = integrated_gradients(model, x_batch, target, b_blur, steps=args.ig_steps)
    diff = (ig_zero - ig_blur).detach().abs().sum(dim=1).squeeze(0).cpu().numpy()

    y_tensor = torch.tensor([y], device=device, dtype=torch.long)
    x_adv = pgd_linf(
        model,
        x_batch,
        y_tensor,
        eps=eps,
        alpha=_scale_eps(eps, 1.0 / 8.0),
        steps=args.pgd_steps,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        random_start=True,
    )

    outpath = os.path.join(
        args.out,
        "figures",
        "saliency_panels",
        f"{args.dataset}_{args.model}.png",
    )
    save_saliency_panel(
        x.cpu().numpy(),
        _collapse_attr(ig_zero),
        _collapse_attr(ig_blur),
        diff,
        x_adv.detach().squeeze(0).cpu().numpy(),
        f"{args.dataset}/{args.model}",
        outpath,
    )
    print(f"Wrote: {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/icml2026")
    parser.add_argument("--mode", choices=["aggregate", "saliency"], default="aggregate")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet50_imagenet")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--pgd_steps", type=int, default=40)
    parser.add_argument("--search_k", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.mode == "aggregate":
        make_aggregate_figures(args.out)
        print(f"Wrote aggregate figures to: {os.path.join(args.out, 'figures')}")
    else:
        make_saliency_panel(args)


if __name__ == "__main__":
    main()
