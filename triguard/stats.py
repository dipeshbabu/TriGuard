import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import mannwhitneyu, nct, spearmanr, t, wilcoxon

from .protocol import validate_protocol_values


METRICS = [
    "clean_acc",
    "adv_error",
    "empirical_probe_rate",
    "empirical_probe_violation_rate",
    "empirical_probe_min_margin_mean",
    "bound_check_rate",  # Legacy CSV compatibility.
    "crown_rate",
    "crown_proven_rate",
    "crown_conditional_rate",
    "entropy_mean",
    "entropy_raw_mean",
    "ads_mean",
    "ads_raw_l2_mean",
    "ads_raw_rms_mean",
    "ads_orthogonal_rms_mean",
    "ads_output_gap_mean",
    "wads_mean",
    "raw_wads_l2_mean",
    "raw_wads_rms_mean",
    "orthogonal_wads_rms_mean",
    "baseline_output_gap_mean",
    "baseline_attr_mass_min_mean",
    "baseline_attr_mass_q05",
    "baseline_attr_mass_near_zero_rate",
    "baseline_attr_mass_ratio_min_mean",
    "baseline_attr_mass_ratio_q05",
    "baseline_attr_mass_ratio_below_floor_rate",
    "heldout_wads_mean",
    "heldout_orthogonal_wads_rms_mean",
    "heldout_attr_mass_min_mean",
    "heldout_attr_mass_ratio_min_mean",
    "heldout_attr_mass_ratio_q05",
    "heldout_attr_mass_ratio_below_floor_rate",
    "ig_completeness_error_mean",
    "ig_completeness_relative_error_mean",
    "ig_completeness_abs_max_mean",
    "ig_completeness_relative_max_mean",
    "pp_stability_l2_mean",
    "pp_stability_rms_mean",
    "pp_stability_cosine_mean",
    "pp_stability_topk_jaccard_mean",
    "pp_stability_keep_rate",
    "ig_del_auc_mean",
    "ig_ins_auc_mean",
    "sg2_del_auc_mean",
    "sg2_ins_auc_mean",
    "random_del_auc_mean",
    "random_ins_auc_mean",
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
    "lambda_attr_mass",
]
METHOD_COLUMNS = [
    "condition_hash",
    "code_hash",
    "attack_suite",
    "reference_risk",
    "reference_cvar_alpha",
    "reference_distance",
    "reference_bank_samples",
    "eval_reference_bank_samples",
    "training_reservation_hash",
    "reference_bank_hash",
    "heldout_reference_bank_hash",
]
CONFIG_COLUMNS = ["lambda_entropy", *REGULARIZER_COLUMNS, *METHOD_COLUMNS]
PRIMARY_METRICS = [
    "heldout_wads_mean",
]
PRIMARY_GUARDRAIL_METRICS = [
    "clean_acc",
    "adv_error",
    "heldout_attr_mass_ratio_below_floor_rate",
    "ig_del_auc_mean",
    "ig_ins_auc_mean",
]


def _group_cols(df, base_cols):
    columns = list(base_cols)
    for column in CONFIG_COLUMNS:
        if column in df.columns and column not in columns:
            columns.append(column)
    return columns


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
    # A condition hash intentionally includes the model name, so grouping on it
    # would leave one architecture per group and silently produce no tests.
    group_cols = ["dataset"] + [
        column
        for column in CONFIG_COLUMNS
        if column != "condition_hash" and column in df.columns
    ]
    for keys, ds_group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
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
    if not out.empty:
        correction_groups = [
            column
            for column in ["dataset", *CONFIG_COLUMNS, "metric"]
            if column in out.columns
        ]
        out["p_value_holm"] = out.groupby(
            correction_groups, dropna=False
        )["p_value"].transform(lambda values: _holm_adjust(values.to_numpy()))
    out.to_csv(os.path.join(out_dir, "mannwhitney_tests.csv"), index=False)
    return out


def _setting_label(values: dict) -> str:
    names = {
        "condition_hash": "condition",
        "lambda_entropy": "ent",
        "lambda_wads": "wads",
        "lambda_rar": "rar",
        "lambda_far": "far",
        "lambda_curvature": "curv",
        "lambda_robust": "rob",
        "lambda_attr_mass": "mass",
        "reference_risk": "risk",
        "reference_cvar_alpha": "cvar",
        "reference_distance": "dist",
        "reference_bank_samples": "train_refs",
        "eval_reference_bank_samples": "eval_refs",
        "training_reservation_hash": "reservation",
        "reference_bank_hash": "bank",
        "heldout_reference_bank_hash": "heldout",
        "attack_suite": "attack",
    }
    def format_value(value):
        try:
            return f"{float(value):.6g}"
        except (TypeError, ValueError):
            return str(value)

    return "|".join(
        f"{names[column]}={format_value(values.get(column, ''))}"
        for column in CONFIG_COLUMNS
        if column in values
    )


def _holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p_values))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(p_values[valid])]
    running = 0.0
    m = order.size
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def paired_regularizer_tests(df, out_dir):
    """Compare regularizers within a model using seed-matched Wilcoxon tests."""
    config_cols = [column for column in CONFIG_COLUMNS if column in df.columns]
    base_cols = ["dataset", "model"]
    if "input_profile" in df.columns:
        base_cols.append("input_profile")
    if "comparison_hash" in df.columns:
        base_cols.append("comparison_hash")

    rows = []
    if not config_cols or "seed" not in df.columns:
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(out_dir, "regularizer_paired_tests.csv"), index=False)
        return out

    for base_keys, base_group in df.groupby(base_cols, dropna=False):
        if not isinstance(base_keys, tuple):
            base_keys = (base_keys,)
        base = dict(zip(base_cols, base_keys))
        configurations = base_group[config_cols].drop_duplicates().to_dict("records")
        for metric in METRICS:
            if metric not in base_group.columns:
                continue
            for left, right in itertools.combinations(configurations, 2):
                left_mask = np.ones(len(base_group), dtype=bool)
                right_mask = np.ones(len(base_group), dtype=bool)
                for column in config_cols:
                    left_mask &= base_group[column].eq(left[column]).to_numpy()
                    right_mask &= base_group[column].eq(right[column]).to_numpy()
                left_values = base_group.loc[left_mask, ["seed", metric]].rename(
                    columns={metric: "left"}
                )
                right_values = base_group.loc[right_mask, ["seed", metric]].rename(
                    columns={metric: "right"}
                )
                paired = left_values.merge(right_values, on="seed", how="inner")
                paired["left"] = pd.to_numeric(paired["left"], errors="coerce")
                paired["right"] = pd.to_numeric(paired["right"], errors="coerce")
                paired = paired.dropna()
                differences = paired["right"].to_numpy() - paired["left"].to_numpy()
                ci_low, ci_high = _bootstrap_ci(
                    differences, np.random.default_rng(0)
                )
                if differences.size < 3:
                    statistic = np.nan
                    p_value = np.nan
                elif np.allclose(differences, 0.0):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    statistic, p_value = wilcoxon(differences, alternative="two-sided")
                rows.append(
                    {
                        **base,
                        "metric": metric,
                        "setting_a": _setting_label(left),
                        "setting_b": _setting_label(right),
                        "paired_n": int(differences.size),
                        "mean_a": float(paired["left"].mean()) if len(paired) else np.nan,
                        "mean_b": float(paired["right"].mean()) if len(paired) else np.nan,
                        "mean_delta_b_minus_a": (
                            float(np.mean(differences)) if differences.size else np.nan
                        ),
                        "median_delta_b_minus_a": (
                            float(np.median(differences)) if differences.size else np.nan
                        ),
                        "mean_delta_ci95_low": ci_low,
                        "mean_delta_ci95_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        correction_groups = [*base_cols, "metric"]
        out["p_value_holm"] = out.groupby(
            correction_groups, dropna=False
        )["p_value"].transform(lambda values: _holm_adjust(values.to_numpy()))
    out.to_csv(os.path.join(out_dir, "regularizer_paired_tests.csv"), index=False)
    return out


def _primary_treatment(row):
    def value(name):
        return float(row.get(name, 0.0))

    entropy = value("lambda_entropy")
    wads = value("lambda_wads")
    rar = value("lambda_rar")
    far = value("lambda_far")
    curvature = value("lambda_curvature")
    robust = value("lambda_robust")
    mass = value("lambda_attr_mass")
    risk = str(row.get("reference_risk", "max"))
    if all(number == 0.0 for number in [entropy, wads, rar, far, curvature, robust, mass]):
        return "ce"
    shared = entropy == 0.0 and wads > 0.0 and rar == 0.0 and far == 0.0
    if shared and curvature == 0.0 and robust == 0.0 and mass == 0.0 and risk == "mean":
        return "mean_reference_only"
    if shared and curvature == 0.0 and robust == 0.0 and mass == 0.0 and risk == "max":
        return "max_reference_only"
    if shared and curvature > 0.0 and robust > 0.0 and mass == 0.0 and risk == "max":
        return "max_reference_controls"
    if shared and curvature > 0.0 and robust > 0.0 and mass > 0.0 and risk == "cvar":
        return "cvar_mass"
    return None


def primary_paired_tests(df, out_dir):
    """Run the two preregistered training contrasts, paired by seed."""
    active = df.copy()
    if "faithfulness_enabled" in active.columns:
        active = active[
            pd.to_numeric(active["faithfulness_enabled"], errors="coerce").eq(1)
        ]
    active["treatment"] = active.apply(_primary_treatment, axis=1)
    active = active[active["treatment"].notna()]
    contrasts = [
        ("mean_reference_only", "max_reference_only"),
        ("max_reference_controls", "cvar_mass"),
    ]
    base_cols = ["dataset", "model"]
    for column in [
        "code_hash",
        "comparison_hash",
        "input_profile",
        "attack_suite",
        "reference_bank_hash",
        "heldout_reference_bank_hash",
    ]:
        if column in active.columns:
            base_cols.append(column)

    rows = []
    for base_keys, group in active.groupby(base_cols, dropna=False):
        if not isinstance(base_keys, tuple):
            base_keys = (base_keys,)
        base = dict(zip(base_cols, base_keys))
        for metric in [*PRIMARY_METRICS, *PRIMARY_GUARDRAIL_METRICS]:
            if metric not in group.columns:
                continue
            role = "primary" if metric in PRIMARY_METRICS else "guardrail"
            for left, right in contrasts:
                left_values = group.loc[
                    group["treatment"] == left, ["seed", metric]
                ].rename(columns={metric: "left"})
                right_values = group.loc[
                    group["treatment"] == right, ["seed", metric]
                ].rename(columns={metric: "right"})
                if left_values["seed"].duplicated().any() or right_values[
                    "seed"
                ].duplicated().any():
                    raise ValueError(
                        f"Duplicate seed rows in primary contrast {left} vs {right}."
                    )
                paired = left_values.merge(right_values, on="seed", how="inner")
                paired[["left", "right"]] = paired[["left", "right"]].apply(
                    pd.to_numeric, errors="coerce"
                )
                paired = paired.dropna()
                differences = paired["right"].to_numpy() - paired["left"].to_numpy()
                ci_low, ci_high = _bootstrap_ci(
                    differences, np.random.default_rng(0)
                )
                if role == "guardrail" or differences.size < 6:
                    statistic, p_value = np.nan, np.nan
                elif np.allclose(differences, 0.0):
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = wilcoxon(
                        differences, alternative="two-sided"
                    )
                rows.append(
                    {
                        **base,
                        "metric": metric,
                        "role": role,
                        "treatment_a": left,
                        "treatment_b": right,
                        "paired_n": int(differences.size),
                        "mean_a": (
                            float(paired["left"].mean()) if differences.size else np.nan
                        ),
                        "mean_b": (
                            float(paired["right"].mean()) if differences.size else np.nan
                        ),
                        "mean_delta_b_minus_a": (
                            float(differences.mean()) if differences.size else np.nan
                        ),
                        "median_delta_b_minus_a": (
                            float(np.median(differences))
                            if differences.size
                            else np.nan
                        ),
                        "mean_delta_ci95_low": ci_low,
                        "mean_delta_ci95_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                    }
                )
    result_columns = [
        *base_cols,
        "metric",
        "role",
        "treatment_a",
        "treatment_b",
        "paired_n",
        "mean_a",
        "mean_b",
        "mean_delta_b_minus_a",
        "median_delta_b_minus_a",
        "mean_delta_ci95_low",
        "mean_delta_ci95_high",
        "wilcoxon_statistic",
        "p_value",
        "p_value_holm",
    ]
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_holm"] = np.nan
        primary_mask = out["role"].eq("primary")
        out.loc[primary_mask, "p_value_holm"] = _holm_adjust(
            out.loc[primary_mask, "p_value"].to_numpy()
        )
        out = out[result_columns]
    else:
        out = pd.DataFrame(columns=result_columns)
    out.to_csv(os.path.join(out_dir, "primary_paired_tests.csv"), index=False)
    decisions = []
    if not out.empty:
        decision_cols = [*base_cols, "treatment_a", "treatment_b"]
        for keys, group in out.groupby(decision_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            identity = dict(zip(decision_cols, keys))
            indexed = group.set_index("metric")

            def metric_value(metric, column):
                if metric not in indexed.index:
                    return np.nan
                return float(indexed.loc[metric, column])

            wads_p = metric_value("heldout_wads_mean", "p_value_holm")
            wads_ci_high = metric_value(
                "heldout_wads_mean", "mean_delta_ci95_high"
            )
            clean_ci_low = metric_value("clean_acc", "mean_delta_ci95_low")
            adversarial_ci_high = metric_value(
                "adv_error", "mean_delta_ci95_high"
            )
            mass_floor_rate = metric_value(
                "heldout_attr_mass_ratio_below_floor_rate", "mean_b"
            )
            deletion_ci_high = metric_value(
                "ig_del_auc_mean", "mean_delta_ci95_high"
            )
            insertion_ci_low = metric_value(
                "ig_ins_auc_mean", "mean_delta_ci95_low"
            )
            criteria = {
                "wads_holm_significant": np.isfinite(wads_p) and wads_p < 0.05,
                "wads_ci_below_zero": (
                    np.isfinite(wads_ci_high) and wads_ci_high < 0.0
                ),
                "clean_noninferior_1pp": (
                    np.isfinite(clean_ci_low) and clean_ci_low >= -0.01
                ),
                "adversarial_error_noninferior_2pp": (
                    np.isfinite(adversarial_ci_high)
                    and adversarial_ci_high <= 0.02
                ),
                "mass_floor_violation_at_most_5pct": (
                    np.isfinite(mass_floor_rate) and mass_floor_rate <= 0.05
                ),
                "deletion_auc_noninferior_2pt": (
                    np.isfinite(deletion_ci_high) and deletion_ci_high <= 0.02
                ),
                "insertion_auc_noninferior_2pt": (
                    np.isfinite(insertion_ci_low) and insertion_ci_low >= -0.02
                ),
            }
            decisions.append(
                {
                    **identity,
                    **criteria,
                    "success": all(criteria.values()),
                }
            )
    decision_columns = [
        *base_cols,
        "treatment_a",
        "treatment_b",
        "wads_holm_significant",
        "wads_ci_below_zero",
        "clean_noninferior_1pp",
        "adversarial_error_noninferior_2pp",
        "mass_floor_violation_at_most_5pct",
        "deletion_auc_noninferior_2pt",
        "insertion_auc_noninferior_2pt",
        "success",
    ]
    pd.DataFrame(decisions, columns=decision_columns).to_csv(
        os.path.join(out_dir, "primary_decisions.csv"), index=False
    )
    return out


def metric_correlations(df, out_dir):
    available = [m for m in METRICS if m in df.columns]
    unit_cols = _group_cols(df, ["dataset", "model"])
    if "input_profile" in df.columns:
        unit_cols.append("input_profile")
    units = df.groupby(unit_cols, dropna=False, as_index=False)[available].mean(
        numeric_only=True
    )
    rows = []
    for dataset, group in units.groupby("dataset", dropna=False):
        for left, right in itertools.combinations(available, 2):
            x = pd.to_numeric(group[left], errors="coerce")
            y = pd.to_numeric(group[right], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(x[mask], y[mask])
            rows.append(
                {
                    "dataset": dataset,
                    "unit": "seed_averaged_configuration",
                    "metric_a": left,
                    "metric_b": right,
                    "n": int(mask.sum()),
                    "spearman_rho": rho,
                    "p_value": p,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_holm"] = out.groupby("dataset", dropna=False)[
            "p_value"
        ].transform(lambda values: _holm_adjust(values.to_numpy()))
    out.to_csv(os.path.join(out_dir, "metric_correlations.csv"), index=False)
    return out


def paired_design_sensitivity(
    out_dir,
    paired_n=10,
    family_alpha=0.05,
    primary_contrasts=2,
    target_power=0.8,
):
    """Write the paired-t standardized effect detectable by the fixed design.

    This is a planning approximation for scale, not observed power and not a
    replacement for the preregistered Wilcoxon analysis.
    """
    paired_n = int(paired_n)
    if paired_n < 3 or primary_contrasts < 1:
        raise ValueError("Design sensitivity requires n >= 3 and at least one contrast.")
    per_contrast_alpha = float(family_alpha) / int(primary_contrasts)
    degrees_freedom = paired_n - 1
    critical = t.ppf(1.0 - per_contrast_alpha / 2.0, degrees_freedom)

    def achieved_power(effect):
        noncentrality = float(effect) * np.sqrt(paired_n)
        return nct.sf(critical, degrees_freedom, noncentrality) + nct.cdf(
            -critical, degrees_freedom, noncentrality
        )

    upper = next(
        (
            effect
            for effect in np.linspace(0.05, 2.0, 40)
            if np.isfinite(achieved_power(effect))
            and achieved_power(effect) >= float(target_power)
        ),
        None,
    )
    if upper is None:
        raise ValueError("Could not bracket the requested design-sensitivity power.")
    detectable_effect = brentq(
        lambda effect: achieved_power(effect) - float(target_power), 0.0, upper
    )
    out = pd.DataFrame(
        [
            {
                "paired_n": paired_n,
                "family_alpha": family_alpha,
                "primary_contrasts": primary_contrasts,
                "per_contrast_alpha_bonferroni": per_contrast_alpha,
                "target_power": target_power,
                "standardized_paired_effect": detectable_effect,
                "approximation": "two_sided_paired_t",
            }
        ]
    )
    out.to_csv(os.path.join(out_dir, "design_sensitivity.csv"), index=False)
    return out


def ads_validity(main_df, faith_df, out_dir):
    main_group_cols = _group_cols(main_df, ["dataset", "model"])
    faith_group_cols = _group_cols(faith_df, ["dataset", "model"])
    merge_cols = [
        c
        for c in main_group_cols
        if c in faith_group_cols and c not in {"condition_hash", "attack_suite"}
    ]
    main = main_df.groupby(main_group_cols, as_index=False).agg(
        {
            "ads_mean": "mean",
            "entropy_mean": "mean",
            **({"wads_mean": "mean"} if "wads_mean" in main_df.columns else {}),
            **(
                {"pp_stability_rms_mean": "mean"}
                if "pp_stability_rms_mean" in main_df.columns
                else {}
            ),
        }
    )
    faith = faith_df.groupby(merge_cols, dropna=False, as_index=False).agg(
        {"ig_del_auc_mean": "mean", "ig_ins_auc_mean": "mean"}
    )
    out = main.merge(faith, on=merge_cols, how="left")
    out["ig_auc_gap"] = out["ig_ins_auc_mean"] - out["ig_del_auc_mean"]
    out.to_csv(os.path.join(out_dir, "ads_validity.csv"), index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/icml2026")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    main_csv = os.path.join(args.out, "table1_main.csv")
    faith_csv = os.path.join(args.out, "tables5_6_faithfulness.csv")

    if not os.path.exists(main_csv):
        raise FileNotFoundError(main_csv)

    main_df = pd.read_csv(main_csv)
    validate_protocol_values(main_df["protocol_version"], main_csv)
    summarize_with_ci(main_df, args.out)
    mann_whitney_grid(main_df, args.out)
    paired_regularizer_tests(main_df, args.out)
    primary_results = primary_paired_tests(main_df, args.out)
    primary_rows = (
        primary_results[primary_results["role"].eq("primary")]
        if "role" in primary_results.columns
        else pd.DataFrame()
    )
    paired_n = (
        int(primary_rows["paired_n"].min()) if not primary_rows.empty else 10
    )
    paired_design_sensitivity(args.out, paired_n=paired_n)
    metric_correlations(main_df, args.out)

    if os.path.exists(faith_csv):
        faith_df = pd.read_csv(faith_csv)
        validate_protocol_values(faith_df["protocol_version"], faith_csv)
        ads_validity(main_df, faith_df, args.out)

    print(f"Wrote statistical analysis files to: {args.out}")


if __name__ == "__main__":
    main()
