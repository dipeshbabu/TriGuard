import argparse
import os

import pandas as pd


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

REGULARIZER_COLUMNS = [
    "lambda_wads",
    "lambda_rar",
    "lambda_far",
    "lambda_curvature",
    "lambda_robust",
]


def fmt_mean_std(mean_series, std_series, pct: bool = False):
    vals = []
    for m, s in zip(mean_series, std_series):
        if pd.isna(s):
            s = 0.0
        if pct:
            vals.append(f"{100 * m:.2f} $\\pm$ {100 * s:.2f}")
        else:
            vals.append(f"{m:.3f} $\\pm$ {s:.3f}")
    return vals


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=False)
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            body,
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
            "",
        ]
    )


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" in df.columns:
        df["dataset"] = pd.Categorical(
            df["dataset"], categories=DATASET_ORDER, ordered=True)
    if "model" in df.columns:
        df["model"] = pd.Categorical(
            df["model"], categories=MODEL_ORDER, ordered=True)
    sort_cols = [c for c in ["dataset", "model"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def add_existing(cols, df):
    return cols + [c for c in REGULARIZER_COLUMNS if c in df.columns]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/icml2026")
    p.add_argument("--label_prefix", type=str, default="")
    args = p.parse_args()

    main_csv = os.path.join(args.out, "table1_main.csv")
    lambda_csv = os.path.join(args.out, "table4_lambda_ablation.csv")
    seed_csv = os.path.join(args.out, "table_seed_variance.csv")
    base_csv = os.path.join(args.out, "table2_baseline_sensitivity.csv")
    faith_csv = os.path.join(args.out, "tables5_6_faithfulness.csv")
    cert_csv = os.path.join(args.out, "table_certification_sweep.csv")
    tex_path = os.path.join(args.out, "paper_tables.tex")

    pieces: list[str] = []

    def label(name: str) -> str:
        if args.label_prefix:
            return f"tab:{args.label_prefix}_{name}"
        return f"tab:{name}"

    if os.path.exists(main_csv):
        df = pd.read_csv(main_csv)
        group_cols = ["dataset", "model", "lambda_entropy"]
        if "input_profile" in df.columns:
            group_cols.append("input_profile")
        group_cols = add_existing(group_cols, df)
        agg_map = {
            "clean_acc": ["mean", "std"],
            "adv_error": ["mean", "std"],
            "bound_check_rate": ["mean", "std"],
            "crown_rate": ["mean", "std"],
            "entropy_mean": ["mean", "std"],
            "ads_mean": ["mean", "std"],
        }
        if "wads_mean" in df.columns:
            agg_map["wads_mean"] = ["mean", "std"]
        for metric in [
            "pp_stability_l2_mean",
            "pp_stability_cosine_mean",
            "pp_stability_topk_jaccard_mean",
            "pp_stability_keep_rate",
            "train_seconds",
            "eval_seconds",
            "total_seconds",
        ]:
            if metric in df.columns:
                agg_map[metric] = ["mean", "std"]
        grp = df.groupby(group_cols, as_index=False).agg(agg_map)
        grp.columns = ["_".join(c).strip("_")
                       for c in grp.columns.to_flat_index()]
        table_data = {
            "dataset": grp["dataset"],
            "model": grp["model"],
            "Clean Acc": fmt_mean_std(grp["clean_acc_mean"], grp["clean_acc_std"], pct=True),
            "PGD Err": fmt_mean_std(grp["adv_error_mean"], grp["adv_error_std"], pct=True),
            "Bound Check": fmt_mean_std(grp["bound_check_rate_mean"], grp["bound_check_rate_std"]),
            "Entropy": fmt_mean_std(grp["entropy_mean_mean"], grp["entropy_mean_std"]),
            "ADS": fmt_mean_std(grp["ads_mean_mean"], grp["ads_mean_std"]),
        }
        if "wads_mean_mean" in grp.columns:
            table_data["WADS"] = fmt_mean_std(grp["wads_mean_mean"], grp["wads_mean_std"])
        if "pp_stability_l2_mean_mean" in grp.columns:
            table_data["PP L2"] = fmt_mean_std(
                grp["pp_stability_l2_mean_mean"],
                grp["pp_stability_l2_mean_std"],
            )
        if "pp_stability_topk_jaccard_mean_mean" in grp.columns:
            table_data["PP top-k J"] = fmt_mean_std(
                grp["pp_stability_topk_jaccard_mean_mean"],
                grp["pp_stability_topk_jaccard_mean_std"],
            )
        if "total_seconds_mean" in grp.columns:
            table_data["Time (s)"] = fmt_mean_std(
                grp["total_seconds_mean"],
                grp["total_seconds_std"],
            )
        if not grp["crown_rate_mean"].isna().all():
            table_data["CROWN"] = fmt_mean_std(grp["crown_rate_mean"], grp["crown_rate_std"])
        out = pd.DataFrame(table_data)
        out = sort_df(out)
        pieces.append(
            latex_table(
                out,
                "Main results with mean and standard deviation across seeds. We report clean accuracy, PGD error, bound check rate, attribution entropy, and Attribution Drift Score. Certification results are reported separately when the CROWN sweep is run.",
                label("main_results"),
            )
        )

        seed_cols = add_existing(["dataset", "model"], df)
        seed_agg = {
            "clean_acc": ["std"],
            "adv_error": ["std"],
            "entropy_mean": ["std"],
            "ads_mean": ["std"],
        }
        if "wads_mean" in df.columns:
            seed_agg["wads_mean"] = ["std"]
        if "pp_stability_l2_mean" in df.columns:
            seed_agg["pp_stability_l2_mean"] = ["std"]
        seed_only = df.groupby(seed_cols, as_index=False).agg(seed_agg)
        seed_only.columns = ["_".join(c).strip("_")
                             for c in seed_only.columns.to_flat_index()]
        seed_data = {
            "dataset": seed_only["dataset"],
            "model": seed_only["model"],
            "Acc std": seed_only["clean_acc_std"].map(lambda x: f"{100 * x:.2f}"),
            "PGD err std": seed_only["adv_error_std"].map(lambda x: f"{100 * x:.2f}"),
            "Entropy std": seed_only["entropy_mean_std"].map(lambda x: f"{x:.3f}"),
            "ADS std": seed_only["ads_mean_std"].map(lambda x: f"{x:.3f}"),
        }
        if "wads_mean_std" in seed_only.columns:
            seed_data["WADS std"] = seed_only["wads_mean_std"].map(lambda x: f"{x:.3f}")
        if "pp_stability_l2_mean_std" in seed_only.columns:
            seed_data["PP L2 std"] = seed_only["pp_stability_l2_mean_std"].map(lambda x: f"{x:.3f}")
        seed_out = pd.DataFrame(seed_data)
        seed_out = sort_df(seed_out)
        seed_out.to_csv(seed_csv, index=False)
        pieces.append(
            latex_table(
                seed_out,
                "Seed variance summary across the main benchmark settings.",
                label("seed_variance"),
            )
        )

    if os.path.exists(lambda_csv):
        df = pd.read_csv(lambda_csv)
        group_cols = add_existing(["dataset", "model", "lambda_entropy"], df)
        agg_map = {
            "clean_acc": "mean",
            "adv_error": "mean",
            "entropy_mean": "mean",
            "ads_mean": "mean",
        }
        if "wads_mean" in df.columns:
            agg_map["wads_mean"] = "mean"
        if "pp_stability_l2_mean" in df.columns:
            agg_map["pp_stability_l2_mean"] = "mean"
        grp = df.groupby(group_cols, as_index=False).agg(agg_map)
        out = pd.DataFrame(
            {
                "dataset": grp["dataset"],
                "model": grp["model"],
                "$\\lambda$": grp["lambda_entropy"].map(lambda x: f"{x:.2f}"),
                "Clean Acc": grp["clean_acc"].map(lambda x: f"{100 * x:.2f}"),
                "PGD Err": grp["adv_error"].map(lambda x: f"{100 * x:.2f}"),
                "Entropy": grp["entropy_mean"].map(lambda x: f"{x:.3f}"),
                "ADS": grp["ads_mean"].map(lambda x: f"{x:.3f}"),
            }
        )
        if "wads_mean" in grp.columns:
            out["WADS"] = grp["wads_mean"].map(lambda x: f"{x:.3f}")
        if "pp_stability_l2_mean" in grp.columns:
            out["PP L2"] = grp["pp_stability_l2_mean"].map(lambda x: f"{x:.3f}")
        out = sort_df(out)
        pieces.append(
            latex_table(
                out,
                "Effect of gradient entropy regularization strength on performance and explanation stability.",
                label("lambda_ablation"),
            )
        )

    if os.path.exists(base_csv):
        df = pd.read_csv(base_csv)
        df = sort_df(df)
        pieces.append(
            latex_table(
                df,
                "Baseline sensitivity of Attribution Drift Score across baseline pairs.",
                label("baseline_sensitivity"),
            )
        )

    if os.path.exists(faith_csv):
        df = pd.read_csv(faith_csv)
        group_cols = add_existing(["dataset", "model"], df)
        grp = df.groupby(group_cols, as_index=False).agg(
            {
                "ig_del_auc_mean": "mean",
                "ig_ins_auc_mean": "mean",
                "sg2_del_auc_mean": "mean",
                "sg2_ins_auc_mean": "mean",
            }
        )
        out = pd.DataFrame(
            {
                "dataset": grp["dataset"],
                "model": grp["model"],
                "IG deletion": grp["ig_del_auc_mean"].map(lambda x: f"{x:.3f}"),
                "IG insertion": grp["ig_ins_auc_mean"].map(lambda x: f"{x:.3f}"),
                "SG2 deletion": grp["sg2_del_auc_mean"].map(lambda x: f"{x:.3f}"),
                "SG2 insertion": grp["sg2_ins_auc_mean"].map(lambda x: f"{x:.3f}"),
            }
        )
        out = sort_df(out)
        pieces.append(
            latex_table(
                out,
                "Appendix faithfulness comparison for Integrated Gradients and SmoothGrad squared.",
                label("faithfulness_auc"),
            )
        )

    if os.path.exists(cert_csv):
        df = pd.read_csv(cert_csv)
        group_cols = add_existing(["dataset", "model", "cert_pixel_eps"], df)
        grp = df.groupby(group_cols, as_index=False).agg(
            {"crown_rate": ["mean", "std"], "cert_valid_n": "mean"}
        )
        grp.columns = ["_".join(c).strip("_")
                       for c in grp.columns.to_flat_index()]
        out = pd.DataFrame(
            {
                "dataset": grp["dataset"],
                "model": grp["model"],
                "$\\epsilon_{cert}$": grp["cert_pixel_eps"].map(lambda x: f"{x:.4f}"),
                "CROWN rate": fmt_mean_std(grp["crown_rate_mean"], grp["crown_rate_std"]),
                "K": grp["cert_valid_n_mean"].map(lambda x: f"{int(round(x))}"),
            }
        )
        out = sort_df(out)
        pieces.append(
            latex_table(
                out,
                "Certification sweep across smaller pixel-space perturbation radii. This table separates certification radius from the PGD attack radius.",
                label("certification_sweep"),
            )
        )

    os.makedirs(args.out, exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pieces))
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
