
import json
from pathlib import Path

import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models/experiment_f/f1_randomforest_best.joblib"
DATA_PATH = ROOT / "data/processed/experiment_e/experiment_E2_data.npz"
OUTPUT_DIR = ROOT / "data/processed/experiment_g/g3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FEATURE AND CLASS NAMES
# --------------------------------------------------

channels = [
    "linear_x", "linear_y", "linear_z",
    "gyro_x", "gyro_y", "gyro_z"
]

statistics = [
    "mean", "std", "min", "max", "rms",
    "q25", "median", "q75", "range", "energy"
]

feature_names = [
    f"{channel}_{stat}"
    for channel in channels
    for stat in statistics
]

class_names = {
    0: "BACKHAND",
    1: "BACKHAND_SLICE",
    2: "FOREHAND",
    3: "NONE"
}


def class_label(value):
    value = value.item() if isinstance(value, np.generic) else value

    if isinstance(value, (int, np.integer)):
        return class_names.get(int(value), str(value))

    return str(value)


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

print("Loading Random Forest model...")
model = joblib.load(MODEL_PATH)

print("Loading processed dataset...")
data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

print(f"Test samples: {len(X_test)}")
print(f"Features: {X_test.shape[1]}")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

misclassified = np.where(y_pred != y_test)[0]
correct = np.where(y_pred == y_test)[0]

print(f"Correct predictions: {len(correct)}")
print(f"Incorrect predictions: {len(misclassified)}")


# --------------------------------------------------
# SELECT SAMPLES
# --------------------------------------------------

selected_samples = []

# Select the first misclassified test sample
if len(misclassified) > 0:
    selected_samples.append({
        "test_position": int(misclassified[0]),
        "sample_type": "misclassified"
    })

# Select one correctly classified sample per class
for class_value in model.classes_:
    candidates = [
        index for index in correct
        if y_test[index] == class_value
    ]

    if candidates:
        selected_samples.append({
            "test_position": int(candidates[0]),
            "sample_type": "correct"
        })


# --------------------------------------------------
# SHAP EXPLANATIONS
# --------------------------------------------------

print("\nCalculating SHAP values...")

rng = np.random.default_rng(42)

X_train = data["X_train"]
background_indices = rng.choice(
    len(X_train),
    size=min(100, len(X_train)),
    replace=False
)

explainer = shap.TreeExplainer(
    model,
    data=X_train[background_indices],
    feature_perturbation="interventional"
)

shap_explanation = explainer(X_test)
shap_values = np.asarray(shap_explanation.values)
base_values = np.asarray(shap_explanation.base_values)


# --------------------------------------------------
# CREATE INDIVIDUAL WATERFALL PLOTS
# --------------------------------------------------

results = []

for sample_number, sample in enumerate(selected_samples, start=1):

    index = sample["test_position"]

    true_value = y_test[index]
    predicted_value = y_pred[index]

    true_label = class_label(true_value)
    predicted_label = class_label(predicted_value)

    # Explain the class predicted by the model
    class_position = list(model.classes_).index(predicted_value)

    if shap_values.ndim == 3:
        sample_shap = shap_values[index, :, class_position]
        sample_base = base_values[index, class_position]
    else:
        sample_shap = shap_values[index]
        sample_base = base_values[index]

    explanation = shap.Explanation(
        values=sample_shap,
        base_values=sample_base,
        data=X_test[index],
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(
        f"G3 Sample {sample_number}: "
        f"True={true_label}, Predicted={predicted_label}"
    )
    plt.tight_layout()

    plot_name = f"g3_sample_{sample_number}_{sample['sample_type']}.png"
    plot_path = OUTPUT_DIR / plot_name

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()



    # --------------------------------------------------
    # ADDITIONAL EXPLANATION FOR TRUE CLASS
    # Only for misclassified samples
    # --------------------------------------------------

    true_class_plot = None
    true_class_features = None

    if sample["sample_type"] == "misclassified":

        print("\nGenerating explanation for the TRUE class...")

        true_class_position = list(model.classes_).index(true_value)

        if shap_values.ndim == 3:
            true_shap = shap_values[index, :, true_class_position]
            true_base = base_values[index, true_class_position]
        else:
            raise ValueError(
                "Multiclass SHAP values were expected for this comparison."
            )

        true_explanation = shap.Explanation(
            values=true_shap,
            base_values=true_base,
            data=X_test[index],
            feature_names=feature_names
        )

        plt.figure(figsize=(10, 7))

        shap.plots.waterfall(
            true_explanation,
            max_display=12,
            show=False
        )

        plt.title(
            f"G3 True Class Explanation: {true_label}"
        )

        plt.tight_layout()

        true_class_plot = (
            OUTPUT_DIR / "g3_sample_1_true_class_BACKHAND_SLICE.png"
        )

        plt.savefig(
            true_class_plot,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        true_class_features = {
            "explained_class": true_label,
            "top_positive_features": [
                {
                    "feature": feature_names[i],
                    "shap_value": float(true_shap[i])
                }
                for i in np.argsort(true_shap)[-5:][::-1]
            ],
            "top_negative_features": [
                {
                    "feature": feature_names[i],
                    "shap_value": float(true_shap[i])
                }
                for i in np.argsort(true_shap)[:5]
            ],
            "plot": str(true_class_plot.relative_to(ROOT))
        }

        print(f"True class plot saved: {true_class_plot.name}")

    results.append({
        "sample_number": sample_number,
        "test_position": index,
        "sample_type": sample["sample_type"],
        "true_class": true_label,
        "predicted_class": predicted_label,
        "correct_prediction": bool(true_value == predicted_value),
        "explained_class": predicted_label,
        "plot": str(plot_path.relative_to(ROOT)),
        "top_positive_features": [

            {
                "feature": feature_names[i],
                "shap_value": float(sample_shap[i])
            }
            for i in np.argsort(sample_shap)[-5:][::-1]
        ],
        "top_negative_features": [
            {
                "feature": feature_names[i],
                "shap_value": float(sample_shap[i])
            }
            for i in np.argsort(sample_shap)[:5]
        ],
        "true_class_explanation": true_class_features

    })

    print(f"\nSample {sample_number}")
    print(f"Test position: {index}")
    print(f"Type: {sample['sample_type']}")
    print(f"True class: {true_label}")
    print(f"Predicted class: {predicted_label}")
    print(f"Correct: {true_value == predicted_value}")
    print(f"Saved plot: {plot_name}")


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output = {
    "experiment": "G3",
    "description": "Individual SHAP prediction explanations",
    "model": "Optimized Random Forest",
    "test_samples": int(len(X_test)),
    "correct_predictions": int(len(correct)),
    "incorrect_predictions": int(len(misclassified)),
    "explained_samples": results
}

json_path = OUTPUT_DIR / "g3_individual_explanations.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(output, file, indent=4, ensure_ascii=False)

print("\nG3 completed successfully.")
print(f"JSON results: {json_path}")
print(f"Output directory: {OUTPUT_DIR}")
