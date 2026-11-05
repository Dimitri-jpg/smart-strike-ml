import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

B3_FILE = ROOT / "data/processed/transfer_learning_b3/smartstrike_test_metrics.json"
B4_FILE = ROOT / "data/processed/baseline_b4/baseline_test_metrics.json"
B5_FILE = ROOT / "data/processed/robustness_b5/b5_robustness_results.json"

OUTPUT = ROOT / "data/processed/final_comparison_b6"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_values(data):
    return {
        "Accuracy": data["accuracy"],
        "Macro Precision": data["macro_precision"],
        "Macro Recall": data["macro_recall"],
        "Macro F1": data["macro_f1"]
    }


def main():
    b3 = metric_values(read_json(B3_FILE))
    b4 = metric_values(read_json(B4_FILE))
    b5_data = read_json(B5_FILE)

    b5 = {
        "Accuracy": b5_data["summary"]["accuracy"]["mean"],
        "Macro Precision": b5_data["summary"]["macro_precision"]["mean"],
        "Macro Recall": b5_data["summary"]["macro_recall"]["mean"],
        "Macro F1": b5_data["summary"]["macro_f1"]["mean"]
    }

    comparison = {
        "B3 Transfer Learning": b3,
        "B4 Baseline": b4,
        "B5 Baseline Mean (5 seeds)": b5
    }

    with open(OUTPUT / "b6_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)

    metric_names = list(b3.keys())
    x = np.arange(len(metric_names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, list(b3.values()), width, label="B3 Transfer Learning")
    plt.bar(x, list(b4.values()), width, label="B4 Baseline")
    plt.bar(x + width, list(b5.values()), width, label="B5 Mean")

    plt.xticks(x, metric_names, rotation=15)
    plt.ylim(0, 1.10)
    plt.ylabel("Score")
    plt.title("SmartStrike – B3, B4 and B5 Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "b6_metrics_comparison.png", dpi=200)
    plt.close()

    b5_std = {
        name: b5_data["summary"][key]["std"]
        for name, key in [
            ("Accuracy", "accuracy"),
            ("Macro Precision", "macro_precision"),
            ("Macro Recall", "macro_recall"),
            ("Macro F1", "macro_f1")
        ]
    }

    plt.figure(figsize=(9, 5))
    plt.bar(b5_std.keys(), b5_std.values())
    plt.ylabel("Standard deviation")
    plt.title("B5 – Metric Variation Across Five Seeds")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT / "b6_b5_stability.png", dpi=200)
    plt.close()

    lines = [
        "# SmartStrike – Phase B Final Report",
        "",
        "## 1. Experiments",
        "- B1: Dataset audit and source-data analysis.",
        "- B2: Silent Impact classification preprocessing.",
        "- B3: Transfer Learning from Silent Impact to SmartStrike.",
        "- B4: SmartStrike-only baseline CNN.",
        "- B5: Repeated baseline evaluation using five random seeds.",
        "",
        "## 2. Results",
        "| Metric | B3 Transfer Learning | B4 Baseline | B5 Mean |",
        "|---|---:|---:|---:|"
    ]

    for name in metric_names:
        lines.append(
            f"| {name} | {b3[name]:.4f} | {b4[name]:.4f} | {b5[name]:.4f} |"
        )

    lines += [
        "",
        "## 3. B5 Stability",
        "| Metric | Mean | Standard deviation | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|"
    ]

    for name, key in [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1", "macro_f1")
    ]:
        s = b5_data["summary"][key]
        lines.append(
            f"| {name} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} |"
        )

    lines += [
        "",
        "## 4. Interpretation",
        "B3 uses Silent Impact pretraining followed by SmartStrike fine-tuning.",
        "B4 is trained from random initialization using SmartStrike data.",
        "B5 evaluates the stability of a SmartStrike-only baseline across five "
        "different stratified recording-level splits.",
        "",
        "The B3 and B4 results should be interpreted cautiously because their "
        "normalization procedures differ. B3 uses Silent Impact training "
        "statistics, while B4 uses SmartStrike training statistics.",
        "",
        "B5 uses different train, validation and test splits from B4. Its "
        "average metrics therefore describe repeated-split performance and "
        "should not be treated as a direct evaluation on the same test samples.",
        "",
        "## 5. Limitations",
        "- SmartStrike recordings do not contain participant IDs.",
        "- Recording-level splits do not guarantee participant or session independence.",
        "- The test sets are relatively small.",
        "- High scores do not by themselves prove generalization to unseen users.",
        "- B5 evaluates repeated random splits, not independent participants.",
        "",
        "## 6. Generated artifacts",
        "- `b6_comparison.json`",
        "- `b6_metrics_comparison.png`",
        "- `b6_b5_stability.png`",
        ""
    ]

    report_path = OUTPUT / "experiment_B_final_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("B6 FINAL COMPARISON COMPLETED")
    print(f"Comparison: {OUTPUT / 'b6_comparison.json'}")
    print(f"Report: {report_path}")
    print(f"Charts saved in: {OUTPUT}")


if __name__ == "__main__":
    main()