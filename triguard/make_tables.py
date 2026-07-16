import argparse
import os

import pandas as pd

from .protocol import validate_protocol_values


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
    "condition_hash",
    "code_hash",
    "attack_suite",
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


def read_protocol_csv(path):
    frame = pd.read_csv(path)
    validate_protocol_values(frame["protocol_version"], path)
    return frame


def method_label(row) -> str:
    terms = []
    labels = [
        ("lambda_entropy", "Entropy"),
        ("lambda_wads", f"Ref-{row.get('reference_risk', 'max')}"),
        ("lambda_rar", "RAR-like"),
        ("lambda_far", "FAR-like"),
        ("lambda_curvature", "Curvature"),
        ("lambda_robust", "Adv-consistency"),
        ("lambda_attr_mass", "Mass-floor"),
    ]
    for column, label in labels:
        value = float(row.get(column, 0.0))
        if value > 0.0:
            terms.append(f"{label}({value:.3g})")
    return "+".join(terms) if terms else "CE"


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
        df = read_protocol_csv(main_csv)
        probe_metric = (
            "empirical_probe_violation_rate"
            if "empirical_probe_violation_rate" in df.columns
            else "empirical_probe_rate"
        )
        crown_metric = (
            "crown_proven_rate" if "crown_proven_rate" in df.columns else "crown_rate"
        )
        group_cols = ["dataset", "model", "lambda_entropy"]
        if "input_profile" in df.columns:
            group_cols.append("input_profile")
        group_cols = add_existing(group_cols, df)
        agg_map = {
            "clean_acc": ["mean", "std"],
            "adv_error": ["mean", "std"],
            probe_metric: ["mean", "std"],
            crown_metric: ["mean", "std"],
            "entropy_mean": ["mean", "std"],
            "ads_mean": ["mean", "std"],
        }
        if "wads_mean" in df.columns:
            agg_map["wads_mean"] = ["mean", "std"]
        for metric in [
            "ads_raw_rms_mean",
            "ads_orthogonal_rms_mean",
            "ads_output_gap_mean",
            "raw_wads_rms_mean",
            "orthogonal_wads_rms_mean",
            "baseline_output_gap_mean",
            "baseline_attr_mass_min_mean",
            "baseline_attr_mass_ratio_q05",
            "ig_completeness_error_mean",
            "heldout_wads_mean",
            "heldout_orthogonal_wads_rms_mean",
            "heldout_attr_mass_ratio_min_mean",
            "heldout_attr_mass_ratio_q05",
            "ig_del_auc_mean",
            "ig_ins_auc_mean",
        ]:
            if metric in df.columns:
                agg_map[metric] = ["mean", "std"]
        for metric in [
            "pp_stability_rms_mean",
            "pp_stability_cosine_mean",
            "pp_stability_topk_jaccard_mean",
            "pp_stability_keep_rate",
            "train_seconds",
            "eval_seconds",
            "total_seconds",
            "peak_cuda_memory_mb",
        ]:
            if metric in df.columns:
                agg_map[metric] = ["mean", "std"]
        grp = df.groupby(group_cols, as_index=False).agg(agg_map)
        grp.columns = ["_".join(c).strip("_")
                       for c in grp.columns.to_flat_index()]
        table_data = {
            "dataset": grp["dataset"],
            "model": grp["model"],
            "Method": grp.apply(method_label, axis=1),
            "Clean Acc": fmt_mean_std(grp["clean_acc_mean"], grp["clean_acc_std"], pct=True),
            "Adversarial Err": fmt_mean_std(grp["adv_error_mean"], grp["adv_error_std"], pct=True),
            "Probe violation": fmt_mean_std(
                grp[f"{probe_metric}_mean"], grp[f"{probe_metric}_std"]
            ),
            "Entropy": fmt_mean_std(grp["entropy_mean_mean"], grp["entropy_mean_std"]),
            "Allocation ADS": fmt_mean_std(grp["ads_mean_mean"], grp["ads_mean_std"]),
        }
        if "attack_suite" in grp.columns:
            table_data["Attack"] = grp["attack_suite"]
        if "wads_mean_mean" in grp.columns:
            table_data["Allocation WADS"] = fmt_mean_std(
                grp["wads_mean_mean"], grp["wads_mean_std"]
            )
        if "baseline_output_gap_mean_mean" in grp.columns:
            table_data["Baseline output gap"] = fmt_mean_std(
                grp["baseline_output_gap_mean_mean"],
                grp["baseline_output_gap_mean_std"],
            )
        if "orthogonal_wads_rms_mean_mean" in grp.columns:
            table_data["COD RMS"] = fmt_mean_std(
                grp["orthogonal_wads_rms_mean_mean"],
                grp["orthogonal_wads_rms_mean_std"],
            )
        if (
            "heldout_wads_mean_mean" in grp.columns
            and not grp["heldout_wads_mean_mean"].isna().all()
        ):
            table_data["Held-out WADS"] = fmt_mean_std(
                grp["heldout_wads_mean_mean"], grp["heldout_wads_mean_std"]
            )
        if "baseline_attr_mass_ratio_q05_mean" in grp.columns:
            table_data["Mass-ratio q05"] = fmt_mean_std(
                grp["baseline_attr_mass_ratio_q05_mean"],
                grp["baseline_attr_mass_ratio_q05_std"],
            )
        if (
            "heldout_attr_mass_ratio_q05_mean" in grp.columns
            and not grp["heldout_attr_mass_ratio_q05_mean"].isna().all()
        ):
            table_data["Held-out mass q05"] = fmt_mean_std(
                grp["heldout_attr_mass_ratio_q05_mean"],
                grp["heldout_attr_mass_ratio_q05_std"],
            )
        if (
            "ig_del_auc_mean_mean" in grp.columns
            and not grp["ig_del_auc_mean_mean"].isna().all()
        ):
            table_data["IG deletion"] = fmt_mean_std(
                grp["ig_del_auc_mean_mean"],
                grp["ig_del_auc_mean_std"],
            )
            table_data["IG insertion"] = fmt_mean_std(
                grp["ig_ins_auc_mean_mean"],
                grp["ig_ins_auc_mean_std"],
            )
        if "pp_stability_rms_mean_mean" in grp.columns:
            table_data["PP RMS"] = fmt_mean_std(
                grp["pp_stability_rms_mean_mean"],
                grp["pp_stability_rms_mean_std"],
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
        if "peak_cuda_memory_mb_mean" in grp.columns:
            table_data["Peak CUDA MB"] = fmt_mean_std(
                grp["peak_cuda_memory_mb_mean"],
                grp["peak_cuda_memory_mb_std"],
            )
        crown_mean = f"{crown_metric}_mean"
        crown_std = f"{crown_metric}_std"
        if not grp[crown_mean].isna().all():
            table_data["CROWN proven"] = fmt_mean_std(grp[crown_mean], grp[crown_std])
        out = pd.DataFrame(table_data)
        out = sort_df(out)
        pieces.append(
            latex_table(
                out,
                "Main results across seeds. Allocation ADS/WADS are scale-free; completeness-orthogonal drift, output gaps, and held-out reference risk are reported separately.",
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
        if "pp_stability_rms_mean" in df.columns:
            seed_agg["pp_stability_rms_mean"] = ["std"]
        seed_only = df.groupby(seed_cols, as_index=False).agg(seed_agg)
        seed_only.columns = ["_".join(c).strip("_")
                             for c in seed_only.columns.to_flat_index()]
        seed_data = {
            "dataset": seed_only["dataset"],
            "model": seed_only["model"],
            "Acc std": seed_only["clean_acc_std"].map(lambda x: f"{100 * x:.2f}"),
            "Adversarial err std": seed_only["adv_error_std"].map(lambda x: f"{100 * x:.2f}"),
            "Entropy std": seed_only["entropy_mean_std"].map(lambda x: f"{x:.3f}"),
            "Allocation ADS std": seed_only["ads_mean_std"].map(lambda x: f"{x:.3f}"),
        }
        if "wads_mean_std" in seed_only.columns:
            seed_data["Allocation WADS std"] = seed_only["wads_mean_std"].map(lambda x: f"{x:.3f}")
        if "pp_stability_rms_mean_std" in seed_only.columns:
            seed_data["PP RMS std"] = seed_only["pp_stability_rms_mean_std"].map(lambda x: f"{x:.3f}")
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
        df = read_protocol_csv(lambda_csv)
        group_cols = add_existing(["dataset", "model", "lambda_entropy"], df)
        agg_map = {
            "clean_acc": "mean",
            "adv_error": "mean",
            "entropy_mean": "mean",
            "ads_mean": "mean",
        }
        if "wads_mean" in df.columns:
            agg_map["wads_mean"] = "mean"
        if "pp_stability_rms_mean" in df.columns:
            agg_map["pp_stability_rms_mean"] = "mean"
        grp = df.groupby(group_cols, as_index=False).agg(agg_map)
        out = pd.DataFrame(
            {
                "dataset": grp["dataset"],
                "model": grp["model"],
                "$\\lambda$": grp["lambda_entropy"].map(lambda x: f"{x:.2f}"),
                "Clean Acc": grp["clean_acc"].map(lambda x: f"{100 * x:.2f}"),
                "Adversarial Err": grp["adv_error"].map(lambda x: f"{100 * x:.2f}"),
                "Entropy": grp["entropy_mean"].map(lambda x: f"{x:.3f}"),
                "ADS": grp["ads_mean"].map(lambda x: f"{x:.3f}"),
            }
        )
        if "wads_mean" in grp.columns:
            out["WADS"] = grp["wads_mean"].map(lambda x: f"{x:.3f}")
        if "pp_stability_rms_mean" in grp.columns:
            out["PP RMS"] = grp["pp_stability_rms_mean"].map(lambda x: f"{x:.3f}")
        out = sort_df(out)
        pieces.append(
            latex_table(
                out,
                "Effect of gradient entropy regularization strength on performance and explanation stability.",
                label("lambda_ablation"),
            )
        )

    if os.path.exists(base_csv):
        df = read_protocol_csv(base_csv)
        df = sort_df(df)
        pieces.append(
            latex_table(
                df,
                "Baseline sensitivity of signed attribution allocation across reference pairs. Each score is an L1 distance after absolute-mass normalization and is bounded by two for nonzero maps.",
                label("baseline_sensitivity"),
            )
        )

    if os.path.exists(faith_csv):
        df = read_protocol_csv(faith_csv)
        group_cols = add_existing(["dataset", "model"], df)
        faith_agg = {
            "ig_del_auc_mean": "mean",
            "ig_ins_auc_mean": "mean",
            "sg2_del_auc_mean": "mean",
            "sg2_ins_auc_mean": "mean",
        }
        for metric in ["random_del_auc_mean", "random_ins_auc_mean"]:
            if metric in df.columns:
                faith_agg[metric] = "mean"
        grp = df.groupby(group_cols, as_index=False).agg(faith_agg)
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
        if "random_del_auc_mean" in grp.columns:
            out["Random deletion"] = grp["random_del_auc_mean"].map(
                lambda x: f"{x:.3f}"
            )
            out["Random insertion"] = grp["random_ins_auc_mean"].map(
                lambda x: f"{x:.3f}"
            )
        out = sort_df(out)
        pieces.append(
            latex_table(
                out,
                "Blur-trajectory faithfulness controls for Integrated Gradients, SmoothGrad squared, and a paired random ranking.",
                label("faithfulness_auc"),
            )
        )

    if os.path.exists(cert_csv):
        df = read_protocol_csv(cert_csv)
        group_cols = add_existing(["dataset", "model", "cert_pixel_eps"], df)
        crown_metric = (
            "crown_proven_rate"
            if "crown_proven_rate" in df.columns
            else "crown_rate"
        )
        grp = df.groupby(group_cols, as_index=False).agg(
            {
                crown_metric: ["mean", "std"],
                "cert_valid_n": "mean",
                **({"cert_error_n": "mean"} if "cert_error_n" in df.columns else {}),
            }
        )
        grp.columns = ["_".join(c).strip("_")
                       for c in grp.columns.to_flat_index()]
        out = pd.DataFrame(
            {
                "dataset": grp["dataset"],
                "model": grp["model"],
                "$\\epsilon_{cert}$": grp["cert_pixel_eps"].map(lambda x: f"{x:.4f}"),
                "CROWN proven": fmt_mean_std(
                    grp[f"{crown_metric}_mean"], grp[f"{crown_metric}_std"]
                ),
                "Valid": grp["cert_valid_n_mean"].map(lambda x: f"{int(round(x))}"),
            }
        )
        if "cert_error_n_mean" in grp.columns:
            out["Errors"] = grp["cert_error_n_mean"].map(lambda x: f"{int(round(x))}")
        out = sort_df(out)
        pieces.append(
            latex_table(
                out,
                "Certification sweep across pixel-space radii. Errors count as unproven in the reported rate and are itemized separately.",
                label("certification_sweep"),
            )
        )

    os.makedirs(args.out, exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pieces))
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
