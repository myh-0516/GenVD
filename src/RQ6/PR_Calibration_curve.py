import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

def ece(y_true, probs, n_bins=10):
    confs = np.zeros(n_bins)
    accs = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if np.sum(mask) > 0:
            counts[i] = np.sum(mask)
            accs[i] = np.mean(y_true[mask])
            confs[i] = np.mean(probs[mask])
    weights = counts / counts.sum()
    return np.sum(weights * np.abs(accs - confs))

datasets = ["devign", "reveal", "bigvul"]


fig, axes = plt.subplots(3, 2, figsize=(8, 9))

for i, d in enumerate(datasets):
    path_gen = rf"results\RQ1\codebert\generate\{d}\predictions.csv"
    path_cls = rf"results\RQ1\codebert\classify\{d}\predictions.csv"

    gen = pd.read_csv(path_gen).sort_values("idx")
    cls = pd.read_csv(path_cls).sort_values("idx")

    y_true = gen["true_label"].values
    p_gen = gen["probability"].values
    p_cls = cls["probability"].values

    precision_gen, recall_gen, _ = precision_recall_curve(y_true, p_gen)
    precision_cls, recall_cls, _ = precision_recall_curve(y_true, p_cls)

    ap_gen = average_precision_score(y_true, p_gen)
    ap_cls = average_precision_score(y_true, p_cls)

  
    ax1 = axes[i, 0]
    ax1.plot(recall_cls, precision_cls, label=f"Disc (AUPR={ap_cls:.4f})",
             color="#0ca8df")
    ax1.plot(recall_gen, precision_gen, label=f"Gen (AUPR={ap_gen:.4f})",
             color="#785db0")
    ax1.legend(loc="lower left", fontsize=10, framealpha=0.5)
    ax1.set_title(f"PR Curve – {d.capitalize()}", fontsize=12)
    ax1.set_xlabel("Recall",fontsize=11)
    ax1.set_ylabel("Precision",fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)


    frac_gen, mean_gen = calibration_curve(y_true, p_gen, n_bins=10)
    frac_cls, mean_cls = calibration_curve(y_true, p_cls, n_bins=10)

    ece_gen = ece(y_true, p_gen)
    ece_cls = ece(y_true, p_cls)

    ax2 = axes[i, 1]
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
    ax2.plot(mean_cls, frac_cls, marker="s", color="#0ca8df", linewidth=2,
             label=f"Disc (ECE={ece_cls:.4f})")
    ax2.plot(mean_gen, frac_gen, marker="o", color="#785db0", linewidth=2,
             label=f"Gen (ECE={ece_gen:.4f})")
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.5)
    ax2.set_title(f"Calibration – {d.capitalize()}", fontsize=12)
    ax2.set_xlabel("Mean Predicted Probability",fontsize=11)
    ax2.set_ylabel("Fraction of Positives",fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(r"results\RQ6\pr-calib.pdf", dpi=300, bbox_inches="tight")
plt.savefig(r"results\RQ6\pr-calib.png", dpi=300, bbox_inches="tight")
plt.close()