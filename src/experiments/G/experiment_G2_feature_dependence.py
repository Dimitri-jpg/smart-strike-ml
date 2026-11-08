
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
OUTPUT_DIR = ROOT / "data/processed/experiment_g/g2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FEATURE NAMES
# --------------------------------------------------

channels = [
    "linear_x",
    "linear_y",
    "linear_z",
    "gyro_x",
    "gyro_y",
    "gyro_z"
]

statistics = [
    "mean",
    "std",
    "min",
    "max",
    "rms",
    "q25",
    "median",
    "q75",
    "range",
    "energy"
]

feature_names = [
    f"{channel}_{stat}"
    for channel in channels
    for stat in statistics
]


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

print("Loading Random Forest model...")
model = joblib.load(MODEL_PATH)

print("Loading processed dataset...")
data = np.load(DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_test.shape[1]}")


# --------------------------------------------------
# CALCULATE SHAP VALUES
# --------------------------------------------------

print("Calculating SHAP values...")

rng = np.random.default_rng(42)
background_indices = rng.choice(
    len(X_train),
    size=min(100, len(X_train)),
    replace=False
)

background = X_train[background_indices]

explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional"
)

shap_values = explainer(X_test)

values = np.asarray(shap_values.values)

# Average absolute SHAP importance across classes
if values.ndim == 3:
    mean_abs_shap = np.mean(np.abs(values), axis=(0, 2))
elif values.ndim == 2:
    mean_abs_shap = np.mean(np.abs(values), axis=0)
else:
    raise ValueError(f"Unexpected SHAP shape: {values.shape}")


# --------------------------------------------------
# SELECT TOP 5 FEATURES
# --------------------------------------------------

top_indices = np.argsort(mean_abs_shap)[::-1][:5]

print("\nTop 5 features:")

top_features = []

for rank, index in enumerate(top_indices, start=1):
    feature_info = {
        "rank": rank,
        "feature_index": int(index),
        "feature_name": feature_names[index],
        "mean_absolute_shap": float(mean_abs_shap[index])
    }

    top_features.append(feature_info)

    print(
        f"{rank}. {feature_names[index]}: "
        f"{mean_abs_shap[index]:.6f}"
    )


# --------------------------------------------------
# DEPENDENCE PLOTS
# --------------------------------------------------

print("\nGenerating SHAP dependence plots...")

dependence_results = []

for feature in top_features:
    index = feature["feature_index"]
    name = feature["feature_name"]

    plt.figure(figsize=(9, 6))

    shap.dependence_plot(
        index,
        values[:, :, 0] if values.ndim == 3 else values,
        X_test,
        feature_names=feature_names,
        interaction_index=None,
        show=False
    )

    plt.title(f"SHAP Dependence Plot: {name}")
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"g2_dependence_{name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    dependence_results.append({
        **feature,
        "plot": str(plot_path.relative_to(ROOT))
    })

    print(f"Saved: {plot_path.name}")


# --------------------------------------------------
# SAVE JSON RESULTS
# --------------------------------------------------

results = {
    "experiment": "G2",
    "description": "SHAP feature dependence analysis",
    "model": "Optimized Random Forest",
    "number_of_test_samples": int(len(X_test)),
    "number_of_features": int(X_test.shape[1]),
    "top_features": dependence_results,
    "method": (
        "SHAP dependence plots showing feature values "
        "against their SHAP contributions."
    )
}

json_path = OUTPUT_DIR / "g2_feature_dependence_results.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print("\nG2 completed successfully.")
print(f"JSON results: {json_path}")
print(f"Output directory: {OUTPUT_DIR}")
