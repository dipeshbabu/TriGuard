import argparse
import os

import pandas as pd
import torch

from .attacks import pgd_linf
from .attributions import blurred_baseline, integrated_gradients
from .data import get_dataset
from .models import get_model, uses_imagenet_preprocessing
from .plots import save_correlation_heatmap, save_radar_plot, save_saliency_panel


RADAR_METRICS = [
    "clean_acc",
    "adv_error",
    "bound_check_rate",
    "entropy_mean",
    "ads_mean",
    "wads_mean",
    "pp_stability_l2_mean",
    "pp_stability_topk_jaccard_mean",
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
        save_correlation_heatmap(
            corr_df,
            "Spearman correlation among TriGuard metrics",
            os.path.join(fig_dir, "metric_correlation_heatmap.png"),
        )

    if os.path.exists(main_csv):
        main_df = pd.read_csv(main_csv)
        metrics = [m for m in RADAR_METRICS if m in main_df.columns]
        group = main_df.groupby(["dataset", "model"], as_index=False)[metrics].mean(numeric_only=True)
        for dataset in sorted(group["dataset"].unique()):
            save_radar_plot(
                group,
                dataset,
                metrics,
                f"{dataset} metric profile",
                os.path.join(fig_dir, f"radar_{dataset}.png"),
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
