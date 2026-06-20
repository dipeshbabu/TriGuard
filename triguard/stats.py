import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


METRICS = [
    "clean_acc",
    "adv_error",
    "bound_check_rate",
    "crown_rate",
    "entropy_mean",
    "ads_mean",
    "wads_mean",
    "pp_stability_l2_mean",
    "pp_stability_cosine_mean",
    "pp_stability_topk_jaccard_mean",
    "pp_stability_keep_rate",
    "train_seconds",
    "eval_seconds",
    "total_seconds",
]
REGULARIZER_COLUMNS = [
    "lambda_wads",
    "lambda_rar",
    "lambda_far",
    "lambda_curvature",
    "lambda_robust",
]


def _group_cols(df, base_cols):
    return base_cols + [c for c in REGULARIZER_COLUMNS if c in df.columns]


def _bootstrap_ci(values, rng, n_boot=5000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def summarize_with_ci(df, out_dir):
    rng = np.random.default_rng(0)
    rows = []
    group_cols = _group_cols(df, ["dataset", "model"])
    if "input_profile" in df.columns:
        group_cols.append("input_profile")

    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            lo, hi = _bootstrap_ci(values, rng)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)) if values.size else np.nan,
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "summary_with_ci.csv"), index=False)
    return out


def mann_whitney_grid(df, out_dir):
    rows = []
    group_cols = _group_cols(df, ["dataset"])
    for keys, ds_group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        dataset = base["dataset"]
        models = sorted(ds_group["model"].dropna().unique())
        for metric in METRICS:
            if metric not in ds_group.columns:
                continue
            for left, right in itertools.combinations(models, 2):
                a = pd.to_numeric(
                    ds_group.loc[ds_group["model"] == left, metric],
                    errors="coerce",
                ).dropna()
                b = pd.to_numeric(
                    ds_group.loc[ds_group["model"] == right, metric],
                    errors="coerce",
                ).dropna()
                if len(a) < 2 or len(b) < 2:
                    p = np.nan
                    stat = np.nan
                else:
                    stat, p = mannwhitneyu(a, b, alternative="two-sided")
                rows.append(
                    {
                        **base,
                        "metric": metric,
                        "model_a": left,
                        "model_b": right,
                        "n_a": len(a),
                        "n_b": len(b),
                        "mean_a": float(a.mean()) if len(a) else np.nan,
                        "mean_b": float(b.mean()) if len(b) else np.nan,
                        "mannwhitney_u": stat,
                        "p_value": p,
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "mannwhitney_tests.csv"), index=False)
    return out


def metric_correlations(df, out_dir):
    available = [m for m in METRICS if m in df.columns]
    group = df.groupby(_group_cols(df, ["dataset", "model"]), as_index=False)[available].mean(numeric_only=True)
    rows = []
    for left, right in itertools.combinations(available, 2):
        x = pd.to_numeric(group[left], errors="coerce")
        y = pd.to_numeric(group[right], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(x[mask], y[mask])
        rows.append({"metric_a": left, "metric_b": right, "spearman_rho": rho, "p_value": p})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "metric_correlations.csv"), index=False)
    return out


def ads_validity(main_df, faith_df, out_dir, min_gap=0.05):
    main_group_cols = _group_cols(main_df, ["dataset", "model"])
    faith_group_cols = _group_cols(faith_df, ["dataset", "model"])
    merge_cols = [c for c in main_group_cols if c in faith_group_cols]
    main = main_df.groupby(main_group_cols, as_index=False).agg(
        {
            "ads_mean": "mean",
            "entropy_mean": "mean",
            **({"wads_mean": "mean"} if "wads_mean" in main_df.columns else {}),
            **(
                {"pp_stability_l2_mean": "mean"}
                if "pp_stability_l2_mean" in main_df.columns
                else {}
            ),
        }
    )
    faith = faith_df.groupby(faith_group_cols, as_index=False).agg(
        {"ig_del_auc_mean": "mean", "ig_ins_auc_mean": "mean"}
    )
    out = main.merge(faith, on=merge_cols, how="left")
    out["ig_auc_gap"] = out["ig_ins_auc_mean"] - out["ig_del_auc_mean"]
    out["ads_interpretation"] = np.where(
        out["ig_auc_gap"] >= min_gap,
        "attribution_signal_present",
        "low_attribution_signal",
    )
    out.to_csv(os.path.join(out_dir, "ads_validity.csv"), index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/icml2026")
    parser.add_argument("--min_faithfulness_gap", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    main_csv = os.path.join(args.out, "table1_main.csv")
    faith_csv = os.path.join(args.out, "tables5_6_faithfulness.csv")

    if not os.path.exists(main_csv):
        raise FileNotFoundError(main_csv)

    main_df = pd.read_csv(main_csv)
    summarize_with_ci(main_df, args.out)
    mann_whitney_grid(main_df, args.out)
    metric_correlations(main_df, args.out)

    if os.path.exists(faith_csv):
        faith_df = pd.read_csv(faith_csv)
        ads_validity(main_df, faith_df, args.out, min_gap=args.min_faithfulness_gap)

    print(f"Wrote statistical analysis files to: {args.out}")


if __name__ == "__main__":
    main()
