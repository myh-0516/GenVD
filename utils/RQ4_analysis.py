import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

paths = [
    r"results\RQ4\codebert\generate\top10cwe\predictions.csv",
    r"results\RQ4\codebert\classify\top10cwe\predictions.csv",
    # r"results\RQ4\graphcodebert\classify\top10cwe\predictions.csv",
    # r"results\RQ4\graphcodebert\generate\top10cwe\predictions.csv"
]

all_class = []
all_summary = []
all_pairs = []

for path in paths:
    df = pd.read_csv(path).sort_values("idx")

    y_true = df["true_label"].values
    y_pred = df["pred_label"].values

    labels = sorted(list(set(y_true) | set(y_pred)))

    report = classification_report(
        y_true, y_pred, labels=labels,
        output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    base = path.replace("\\", "/").split("/")
    model = base[-3]
    method = base[-2]

    for cls in labels:
        if cls in report:
            all_class.append({
                "model": model,
                "method": method,
                "class": cls,
                "precision": report[cls]["precision"] * 100,
                "recall": report[cls]["recall"] * 100,
                "f1": report[cls]["f1-score"] * 100,
                "support": report[cls]["support"]
            })

    all_summary.append({
        "model": model,
        "method": method,
        "macro_f1": report["macro avg"]["f1-score"] * 100,
        "weighted_f1": report["weighted avg"]["f1-score"] * 100
    })

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i][j] > 0:
                all_pairs.append({
                    "model": model,
                    "method": method,
                    "true_label": labels[i],
                    "pred_label": labels[j],
                    "count": cm[i][j]
                })

os.makedirs(r"results\RQ4", exist_ok=True)

pd.DataFrame(all_class).to_csv(r"results\RQ4\per_class_metrics_all.csv", index=False, float_format='%.2f')
pd.DataFrame(all_summary).to_csv(r"results\RQ4\summary_metrics_all.csv", index=False, float_format='%.2f')
pd.DataFrame(all_pairs).to_csv(r"results\RQ4\top_confusions_all.csv", index=False)