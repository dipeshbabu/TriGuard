import os
import numpy as np
import matplotlib.pyplot as plt


def save_curve_plot(del_curve, ins_curve, title, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure()
    plt.plot(del_curve, label="Deletion")
    plt.plot(ins_curve, label="Insertion")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("p(target)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_hist(values, title, outpath, bins=30):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure()
    plt.hist(np.asarray(values), bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_correlation_heatmap(corr_df, title, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    metrics = sorted(set(corr_df["metric_a"]).union(set(corr_df["metric_b"])))
    mat = np.eye(len(metrics))
    idx = {metric: i for i, metric in enumerate(metrics)}
    for _, row in corr_df.iterrows():
        i = idx[row["metric_a"]]
        j = idx[row["metric_b"]]
        mat[i, j] = row["spearman_rho"]
        mat[j, i] = row["spearman_rho"]

    plt.figure(figsize=(7, 6))
    im = plt.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(metrics)), metrics, rotation=35, ha="right")
    plt.yticks(range(len(metrics)), metrics)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_radar_plot(group_df, dataset, metrics, title, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df = group_df[group_df["dataset"] == dataset].copy()
    if df.empty:
        return

    values = df[metrics].astype(float)
    norm = values.copy()
    for metric in metrics:
        lo = values[metric].min()
        hi = values[metric].max()
        if hi == lo:
            norm[metric] = 0.5
        else:
            norm[metric] = (values[metric] - lo) / (hi - lo)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, polar=True)
    for row_idx, row in df.reset_index(drop=True).iterrows():
        vals = norm.iloc[row_idx].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, label=row["model"])
        ax.fill(angles, vals, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _to_image(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr.squeeze()


def save_saliency_panel(image, ig_zero, ig_blur, diff, adv_image, title, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    panels = [
        ("input", image, "gray" if np.asarray(image).shape[0] == 1 else None),
        ("IG zero", ig_zero, "magma"),
        ("IG blur", ig_blur, "magma"),
        ("ADS diff", diff, "viridis"),
        ("PGD example", adv_image, None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3))
    for ax, (label, arr, cmap) in zip(axes, panels):
        ax.imshow(_to_image(arr), cmap=cmap)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()
