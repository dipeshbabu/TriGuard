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
