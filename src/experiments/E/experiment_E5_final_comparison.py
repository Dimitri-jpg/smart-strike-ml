import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path.cwd()
BASE_DIR = ROOT / "data/processed/experiment_e"
E3_PATH = BASE_DIR / "e3/e3_classical_results.json"
E4_PATH = BASE_DIR / "e4/e4_cnn_results.json"

OUTPUT_DIR = BASE_DIR / "e5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


print("E5 FINAL MODEL COMPARISON")

e3 = load_json(E3_PATH)
e4 = load_json(E4_PATH)

comparison = []

# Classical models: include the final test result for the selected model.
best_classical = e3["metadata"]["best_model"]
classical_test = e3[best_classical].get("test")

if classical_test is None:
    raise ValueError("E3 test metrics are missing.")

comparison.append({
    "model": best_classical,
    "type": "Classical ML",
    **classical_test,
})

# Deep learning model.
cnn_test = e4.get("test")

if cnn_test is None:
    raise ValueError("E4 test metrics are missing.")

comparison.append({
    "model": "1D CNN",
    "type": "Deep Learning",
    **cnn_test,
})

# Sort by test Macro F1 for presentation.
comparison.sort(
    key=lambda item: item["macro_f1"],
    reverse=True,
)

print("\nFINAL TEST COMPARISON")
for item in comparison:
    print(
        f"{item['model']:20s} | "
        f"Accuracy: {item['accuracy']:.4f} | "
        f"Precision: {item['macro_precision']:.4f} | "
        f"Recall: {item['macro_recall']:.4f} | "
        f"Macro-F1: {item['macro_f1']:.4f}"
    )

best_model = comparison[0]["model"]

report = {
    "experiment": "Experiment E - Final Model Comparison",
    "models": comparison,
    "best_test_macro_f1_model": best_model,
    "selection_note": (
        "The classical model was selected using validation Macro F1. "
        "The CNN checkpoint was selected using validation Macro F1. "
        "The test set was used for final evaluation."
    ),
    "limitations": [
        "The dataset contains 500 recordings.",
        "The split is recording-level, not participant-level.",
        "The recordings have date and class correlations.",
        "Test results should not be interpreted as proof of generalization "
        "to unseen recording sessions or new players.",
        "The classical models use engineered statistical features, while "
        "the CNN uses time-series sensor data."
    ],
}

json_path = OUTPUT_DIR / "e5_final_comparison.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

# Create comparison chart.
metric_names = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]

model_names = [item["model"] for item in comparison]
x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(10, 6))

for index, item in enumerate(comparison):
    values = [item[name] for name in metric_names]
    offset = (index - (len(comparison) - 1) / 2) * width

    plt.bar(
        x + offset,
        values,
        width,
        label=item["model"],
    )

plt.xticks(
    x,
    ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
)

plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Experiment E - Final Model Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

chart_path = OUTPUT_DIR / "e5_model_comparison.png"
plt.savefig(chart_path, dpi=200)
plt.close()

# Create Markdown report.
lines = [
    "# Experiment E – Final Model Comparison",
    "",
    "## Test results",
    "",
    "| Model | Type | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
    "|---|---|---:|---:|---:|---:|",
]

for item in comparison:
    lines.append(
        f"| {item['model']} | {item['type']} | "
        f"{item['accuracy']:.4f} | "
        f"{item['macro_precision']:.4f} | "
        f"{item['macro_recall']:.4f} | "
        f"{item['macro_f1']:.4f} |"
    )

lines += [
    "",
    f"## Highest test Macro F1: {best_model}",
    "",
    "The classical model was selected using validation Macro F1. "
    "The CNN checkpoint was selected using validation Macro F1. "
    "The test set was used for final evaluation.",
    "",
    "## Limitations",
]

for limitation in report["limitations"]:
    lines.append(f"- {limitation}")

lines += [
    "",
    "## Output files",
    "- `e5_final_comparison.json`",
    "- `e5_model_comparison.png`",
]

markdown_path = OUTPUT_DIR / "experiment_E_final_report.md"

markdown_path.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("\nE5 completed successfully.")
print("Best model by test Macro F1:", best_model)
print("Saved:", json_path)
print("Saved:", chart_path)
print("Saved:", markdown_path)