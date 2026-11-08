
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUTPUT = DATA / "experiment_f" / "f4"

OUTPUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Load previous experiment results
e3 = load_json(
    DATA / "experiment_e" / "e3" / "e3_classical_results.json"
)

e4 = load_json(
    DATA / "experiment_e" / "e4" / "e4_cnn_results.json"
)

f1 = load_json(
    DATA / "experiment_f" / "f1" / "f1_randomforest_results.json"
)

f2 = load_json(
    DATA / "experiment_f" / "f2" / "f2_svm_results.json"
)

f3 = load_json(
    DATA / "experiment_f" / "f3" / "f3_gradientboosting_results.json"
)


# Build a unified comparison table
models = [
    {
        "model": "Random Forest (Baseline)",
        "source": "E3",
        "validation": e3["RandomForest"]["validation"],
        "test": e3["RandomForest"]["test"],
    },
    {
        "model": "SVM (Baseline)",
        "source": "E3",
        "validation": e3["SVM"]["validation"],
        "test": None,
    },
    {
        "model": "Gradient Boosting (Baseline)",
        "source": "E3",
        "validation": e3["GradientBoosting"]["validation"],
        "test": None,
    },
    {
        "model": "1D CNN",
        "source": "E4",
        "validation": {
            "macro_f1": e4["best_validation_macro_f1"]
        },
        "test": e4["test"],
    },
    {
        "model": "Random Forest (Optimized)",
        "source": "F1",
        "validation": f1["validation_metrics"],
        "test": f1["test_metrics"],
    },
    {
        "model": "SVM (Optimized)",
        "source": "F2",
        "validation": f2["validation_metrics"],
        "test": f2["test_metrics"],
    },
    {
        "model": "Gradient Boosting (Optimized)",
        "source": "F3",
        "validation": f3["validation_metrics"],
        "test": f3["test_metrics"],
    },
]


# Select the model using validation Macro F1 only
best_model = max(
    models,
    key=lambda item: item["validation"]["macro_f1"]
)


# Prepare final report
report = {
    "experiment": "F4 Final Model Comparison",
    "selection_metric": "validation_macro_f1",
    "best_model_by_validation": best_model["model"],
    "models": models,
    "notes": [
        "Models are compared using the same E1/E2 data split.",
        "Test metrics are reported where available.",
        "Baseline SVM and Gradient Boosting test metrics "
        "were not available in the E3 results file.",
        "Test metrics were not used for model selection.",
        "High scores should be interpreted with the dataset "
        "session/date limitations described in Experiment D."
    ]
}


# Save JSON results
json_path = OUTPUT / "f4_final_comparison.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4, ensure_ascii=False)


# Save a readable Markdown table
md_path = OUTPUT / "f4_final_comparison.md"

with open(md_path, "w", encoding="utf-8") as file:
    file.write("# Experiment F4 – Final Model Comparison\n\n")
    file.write(
        "Model selection is based on Validation Macro F1. "
        "Test metrics are reported separately.\n\n"
    )

    file.write(
        "| Model | Source | Validation Macro F1 | "
        "Test Accuracy | Test Macro F1 |\n"
    )
    file.write("|---|---|---:|---:|---:|\n")

    for item in models:
        val_f1 = item["validation"]["macro_f1"]
        test = item["test"]

        test_acc = (
            f"{test['accuracy']:.4f}" if test else "N/A"
        )
        test_f1 = (
            f"{test['macro_f1']:.4f}" if test else "N/A"
        )

        file.write(
            f"| {item['model']} | {item['source']} | "
            f"{val_f1:.4f} | {test_acc} | {test_f1} |\n"
        )

    file.write(
        f"\n## Best model by validation Macro F1\n\n"
        f"{best_model['model']} "
        f"({best_model['validation']['macro_f1']:.4f})\n"
    )

    file.write(
        "\n## Methodological notes\n\n"
        "- Test results were not used for model selection.\n"
        "- Baseline SVM and Gradient Boosting test results "
        "were unavailable in E3.\n"
        "- Dataset date/session limitations from Experiment D "
        "remain relevant.\n"
    )


# Plot test Macro F1 for models with available test results
test_models = [
    item for item in models if item["test"] is not None
]

names = [item["model"] for item in test_models]
scores = [
    item["test"]["macro_f1"] * 100
    for item in test_models
]

plt.figure(figsize=(11, 6))
bars = plt.bar(names, scores)

plt.ylabel("Test Macro F1 (%)")
plt.title("Experiment F4 – Test Macro F1 Comparison")
plt.ylim(0, 105)
plt.xticks(rotation=25, ha="right")

for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{score:.2f}%",
        ha="center"
    )

plt.tight_layout()

plot_path = OUTPUT / "f4_test_macro_f1_comparison.png"
plt.savefig(plot_path, dpi=300)
plt.close()


print("\nExperiment F4 completed successfully.")
print(f"Best model by validation Macro F1: {best_model['model']}")
print(
    "Validation Macro F1: "
    f"{best_model['validation']['macro_f1']:.4f}"
)
print(f"\nJSON report: {json_path}")
print(f"Markdown report: {md_path}")
print(f"Comparison plot: {plot_path}")
