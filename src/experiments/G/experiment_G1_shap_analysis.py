
import json
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

from pathlib import Path


# Project paths
ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "experiment_f" / "f1_randomforest_best.joblib"
DATA_PATH = ROOT / "data" / "processed" / "experiment_e" / "experiment_E2_data.npz"

OUTPUT_DIR = ROOT / "data" / "processed" / "experiment_g" / "g1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading Random Forest model...")
model = joblib.load(MODEL_PATH)

print("Loading processed dataset...")
data = np.load(DATA_PATH, allow_pickle=True)

X_train = data["X_train"]
X_test = data["X_test"]

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")


# Use a small training sample as SHAP background
rng = np.random.default_rng(42)
background_indices = rng.choice(
    len(X_train),
    size=min(100, len(X_train)),
    replace=False
)

background = X_train[background_indices]

# Explain test predictions
explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional"
)

print("Calculating SHAP values...")
shap_values = explainer(X_test)


# Handle different SHAP output formats
values = shap_values.values

if values.ndim == 3:
    # Mean absolute SHAP value across classes
    mean_abs_shap = np.mean(np.abs(values), axis=(0, 2))
elif values.ndim == 2:
    mean_abs_shap = np.mean(np.abs(values), axis=0)
else:
    raise ValueError(f"Unexpected SHAP shape: {values.shape}")


# Feature names
metadata_path = (
    ROOT / "data" / "processed" / "experiment_e"
    / "e2_preprocessing_metadata.json"
)

feature_names = None

if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if "feature_names" in metadata:
        feature_names = metadata["feature_names"]

if not feature_names or len(feature_names) != len(mean_abs_shap):
    feature_names = [
        f"feature_{i}" for i in range(len(mean_abs_shap))
    ]


# Sort features by importance
sorted_indices = np.argsort(mean_abs_shap)[::-1]

importance = [
    {
        "feature": str(feature_names[i]),
        "mean_absolute_shap": float(mean_abs_shap[i])
    }
    for i in sorted_indices
]


# Save JSON results
results = {
    "experiment": "G1 SHAP Analysis",
    "model": "Optimized Random Forest (F1)",
    "background_samples": int(len(background)),
    "explained_samples": int(len(X_test)),
    "number_of_features": int(len(feature_names)),
    "shap_values_shape": list(values.shape),
    "feature_importance": importance,
    "method": "Mean absolute SHAP value across test samples and classes"
}

json_path = OUTPUT_DIR / "g1_shap_results.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)


# SHAP summary plot
if values.ndim == 3:
    shap.summary_plot(
        values,
        X_test,
        feature_names=feature_names,
        show=False,
        max_display=15
    )
else:
    shap.summary_plot(
        values,
        X_test,
        feature_names=feature_names,
        show=False,
        max_display=15
    )

plt.tight_layout()
summary_path = OUTPUT_DIR / "g1_shap_summary.png"
plt.savefig(summary_path, dpi=300, bbox_inches="tight")
plt.close()


# SHAP global importance bar plot
top_n = min(15, len(importance))

top_features = importance[:top_n]
names = [item["feature"] for item in top_features]
scores = [item["mean_absolute_shap"] for item in top_features]

plt.figure(figsize=(10, 7))
plt.barh(names[::-1], scores[::-1])
plt.xlabel("Mean Absolute SHAP Value")
plt.title("G1 – Global SHAP Feature Importance")
plt.tight_layout()

bar_path = OUTPUT_DIR / "g1_shap_bar.png"
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()


print("\nG1 SHAP analysis completed successfully.")
print("\nTop 10 most influential features:")

for index, item in enumerate(importance[:10], start=1):
    print(
        f"{index}. {item['feature']}: "
        f"{item['mean_absolute_shap']:.6f}"
    )

print(f"\nJSON results: {json_path}")
print(f"Summary plot: {summary_path}")
print(f"Bar plot: {bar_path}")
