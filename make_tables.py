import argparse
import os
import pandas as pd


def to_latex_table(df, cols, caption, label):
    t = df[cols].to_latex(index=False, float_format=lambda x: f"{x:.4f}")
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        t,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
        ""
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/icml2026")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tex_path = os.path.join(args.out, "paper_tables.tex")

    pieces = []

    # Table 1
    t1 = os.path.join(args.out, "table1_main.csv")
    if os.path.exists(t1):
        df = pd.read_csv(t1)
        # If multiple seeds, average them
        g = df.groupby(["dataset", "model", "lambda_entropy"], as_index=False).agg({
            "clean_acc": "mean",
            "adv_error": "mean",
            "entropy_mean": "mean",
            "ads_mean": "mean",
            "ads_adv_mean": "mean",
            "crown_rate": "mean",
        })
        cols = ["dataset", "model", "lambda_entropy", "clean_acc", "adv_error",
                "entropy_mean", "ads_mean", "ads_adv_mean", "crown_rate"]
        pieces.append(to_latex_table(
            g, cols,
            "TriGuard main results: clean accuracy, adversarial error, attribution entropy, ADS (baseline drift), ADS-Adv (adversarial drift), and CROWN-IBP certification rate.",
            "tab:main_results"
        ))

    # Table 2
    t2 = os.path.join(args.out, "table2_baseline_sensitivity.csv")
    if os.path.exists(t2):
        df = pd.read_csv(t2)
        cols = ["dataset", "model", "K", "ads_zero_blur", "ads_zero_noise",
                "ads_zero_uniform", "ads_blur_noise", "ads_blur_uniform", "ads_noise_uniform"]
        pieces.append(to_latex_table(
            df, cols,
            "Baseline sensitivity of ADS across baseline pairs.",
            "tab:baseline_sensitivity"
        ))

    # Table 4
    t4 = os.path.join(args.out, "table4_lambda_ablation.csv")
    if os.path.exists(t4):
        df = pd.read_csv(t4)
        g = df.groupby(["dataset", "model", "lambda_entropy"], as_index=False).agg({
            "clean_acc": "mean",
            "adv_error": "mean",
            "entropy_mean": "mean",
            "ads_mean": "mean",
            "ads_adv_mean": "mean",
            "crown_rate": "mean",
        })
        cols = ["dataset", "model", "lambda_entropy", "clean_acc", "adv_error",
                "entropy_mean", "ads_mean", "ads_adv_mean", "crown_rate"]
        pieces.append(to_latex_table(
            g, cols,
            "Effect of entropy regularization strength $\\lambda$ on robustness and explanation stability.",
            "tab:lambda_ablation"
        ))

    # Tables 5-6 (faithfulness)
    tf = os.path.join(args.out, "tables5_6_faithfulness.csv")
    if os.path.exists(tf):
        df = pd.read_csv(tf)
        g = df.groupby(["dataset", "model"], as_index=False).agg({
            "ig_del_auc_mean": "mean",
            "ig_ins_auc_mean": "mean",
            "sg2_del_auc_mean": "mean",
            "sg2_ins_auc_mean": "mean",
        })
        cols = ["dataset", "model", "ig_del_auc_mean",
                "ig_ins_auc_mean", "sg2_del_auc_mean", "sg2_ins_auc_mean"]
        pieces.append(to_latex_table(
            g, cols,
            "Faithfulness via Deletion/Insertion AUC for IG vs SmoothGrad$^2$.",
            "tab:faithfulness_auc"
        ))

    with open(tex_path, "w") as f:
        f.write("\n".join(pieces))

    print("Wrote:", tex_path)


if __name__ == "__main__":
    main()
