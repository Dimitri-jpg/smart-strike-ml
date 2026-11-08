import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path.cwd()

DATA_PATH = ROOT / "data/processed/experiment_e/experiment_E2_data.npz"
RF_PATH = ROOT / "models/experiment_e/e3_randomforest_model.joblib"
CNN_PATH = ROOT / "models/experiment_e/e4_cnn_1d_best.pth"

OUTPUT_DIR = ROOT / "data/processed/experiment_e/e6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_array(data, possible_names):
    for name in possible_names:
        if name in data.files:
            return data[name]

    raise KeyError(
        f"Could not find {possible_names}. Available keys: {data.files}"
    )


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
    }


class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_feature_names():
    channels = [
        "linear_x", "linear_y", "linear_z",
        "gyro_x", "gyro_y", "gyro_z",
    ]

    statistics = [
        "mean", "std", "min", "max", "rms",
        "q25", "median", "q75", "range", "mean_square_energy",
    ]

    return [
        f"{channel}_{statistic}"
        for channel in channels
        for statistic in statistics
    ]


print("=" * 60)
print("E6 MODEL INTERPRETABILITY AND ERROR ANALYSIS")
print("=" * 60)

data = np.load(DATA_PATH, allow_pickle=True)

# Load labels and split indices.
y_all = get_array(data, ["y", "labels", "y_all", "labels_all"]).astype(str)

train_idx = get_array(
    data, ["train_indices", "idx_train", "train_idx"]
).astype(int)

val_idx = get_array(
    data, ["validation_indices", "val_indices", "idx_validation", "val_idx"]
).astype(int)

test_idx = get_array(
    data, ["test_indices", "idx_test", "test_idx"]
).astype(int)

classes = sorted(np.unique(y_all).tolist())
class_to_id = {name: index for index, name in enumerate(classes)}
y_encoded = np.array([class_to_id[label] for label in y_all])

y_val = y_encoded[val_idx]
y_test = y_encoded[test_idx]

print("Classes:", classes)
print("Validation samples:", len(val_idx))
print("Test samples:", len(test_idx))


# ---------------------------------------------------------
# PART 1: RANDOM FOREST INTERPRETABILITY
# ---------------------------------------------------------

print("\nPART 1: RANDOM FOREST FEATURE IMPORTANCE")

X_train = get_array(data, ["X_train", "X_features_train"])
X_val = get_array(data, ["X_validation", "X_val", "X_features_validation"])
X_test = get_array(data, ["X_test", "X_features_test"])

rf_model = joblib.load(RF_PATH)

feature_names = make_feature_names()

if X_train.shape[1] != len(feature_names):
    raise ValueError(
        f"Expected {len(feature_names)} features, "
        f"but found {X_train.shape[1]}."
    )

# Built-in Random Forest importance.
rf_importances = rf_model.feature_importances_

# Permutation importance is calculated on validation data,
# not on the final test set.
print("Calculating permutation importance...")

permutation = permutation_importance(
    rf_model,
    X_val,
    y_val,
    scoring="f1_macro",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

feature_results = []

for index, feature in enumerate(feature_names):
    feature_results.append({
        "feature": feature,
        "random_forest_importance": float(rf_importances[index]),
        "permutation_mean": float(permutation.importances_mean[index]),
        "permutation_std": float(permutation.importances_std[index]),
    })

feature_results.sort(
    key=lambda item: item["permutation_mean"],
    reverse=True,
)

print("\nTOP 10 FEATURES BY PERMUTATION IMPORTANCE")

for item in feature_results[:10]:
    print(
        f"{item['feature']:30s} "
        f"Importance: {item['permutation_mean']:.6f}"
    )

# Plot top 15 features.
top_features = feature_results[:15]
plot_names = [item["feature"] for item in reversed(top_features)]
plot_values = [
    item["permutation_mean"] for item in reversed(top_features)
]

plt.figure(figsize=(11, 7))
plt.barh(plot_names, plot_values)
plt.xlabel("Mean decrease in validation Macro F1")
plt.title("E6 - Random Forest Permutation Feature Importance")
plt.tight_layout()

feature_plot_path = OUTPUT_DIR / "e6_feature_importance.png"
plt.savefig(feature_plot_path, dpi=200)
plt.close()


# ---------------------------------------------------------
# PART 2: RANDOM FOREST TEST ERROR ANALYSIS
# ---------------------------------------------------------

print("\nPART 2: RANDOM FOREST TEST ANALYSIS")

rf_predictions = rf_model.predict(X_test)

rf_metrics = calculate_metrics(y_test, rf_predictions)

rf_report = classification_report(
    y_test,
    rf_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

rf_matrix = confusion_matrix(
    y_test,
    rf_predictions,
    labels=list(range(len(classes))),
)

rf_errors = []

for local_index, (true_id, predicted_id) in enumerate(
    zip(y_test, rf_predictions)
):
    if true_id != predicted_id:
        original_index = int(test_idx[local_index])

        rf_errors.append({
            "dataset_index": original_index,
            "true_class": classes[int(true_id)],
            "predicted_class": classes[int(predicted_id)],
        })

print("Random Forest test metrics:", rf_metrics)
print("Random Forest errors:", len(rf_errors))


# ---------------------------------------------------------
# PART 3: CNN TEST ERROR ANALYSIS
# ---------------------------------------------------------

print("\nPART 3: CNN TEST ANALYSIS")

X_series = get_array(
    data, ["X_series", "X_time_series", "time_series"]
).astype(np.float32)

X_train_series = X_series[train_idx]
X_test_series = X_series[test_idx]

X_train_series = np.nan_to_num(
    X_train_series, nan=0.0, posinf=0.0, neginf=0.0
)

X_test_series = np.nan_to_num(
    X_test_series, nan=0.0, posinf=0.0, neginf=0.0
)

# Same training-only normalization approach as E4.
channel_mean = X_train_series.mean(axis=(0, 1), keepdims=True)
channel_std = X_train_series.std(axis=(0, 1), keepdims=True)
channel_std[channel_std == 0] = 1.0

X_test_series = (X_test_series - channel_mean) / channel_std

# Convert (samples, time, channels) to (samples, channels, time).
X_test_series = np.transpose(X_test_series, (0, 2, 1))

checkpoint = torch.load(
    CNN_PATH,
    map_location=DEVICE,
    weights_only=False,
)

cnn_model = CNN1D(num_classes=len(classes)).to(DEVICE)
cnn_model.load_state_dict(checkpoint["model_state_dict"])
cnn_model.eval()

X_test_tensor = torch.tensor(
    X_test_series,
    dtype=torch.float32,
    device=DEVICE,
)

with torch.no_grad():
    outputs = cnn_model(X_test_tensor)
    cnn_predictions = torch.argmax(outputs, dim=1).cpu().numpy()

cnn_metrics = calculate_metrics(y_test, cnn_predictions)

cnn_report = classification_report(
    y_test,
    cnn_predictions,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

cnn_matrix = confusion_matrix(
    y_test,
    cnn_predictions,
    labels=list(range(len(classes))),
)

cnn_errors = []

for local_index, (true_id, predicted_id) in enumerate(
    zip(y_test, cnn_predictions)
):
    if true_id != predicted_id:
        original_index = int(test_idx[local_index])

        cnn_errors.append({
            "dataset_index": original_index,
            "true_class": classes[int(true_id)],
            "predicted_class": classes[int(predicted_id)],
        })

print("CNN test metrics:", cnn_metrics)
print("CNN errors:", len(cnn_errors))


# ---------------------------------------------------------
# PART 4: COMPARE MODEL PREDICTIONS
# ---------------------------------------------------------

print("\nPART 4: MODEL ERROR COMPARISON")

rf_error_indices = {
    item["dataset_index"] for item in rf_errors
}

cnn_error_indices = {
    item["dataset_index"] for item in cnn_errors
}

common_errors = sorted(rf_error_indices & cnn_error_indices)
rf_only_errors = sorted(rf_error_indices - cnn_error_indices)
cnn_only_errors = sorted(cnn_error_indices - rf_error_indices)

print("Errors made by both models:", len(common_errors))
print("Errors made only by Random Forest:", len(rf_only_errors))
print("Errors made only by CNN:", len(cnn_only_errors))


# ---------------------------------------------------------
# PART 5: CONFUSION MATRIX PLOTS
# ---------------------------------------------------------

def save_confusion_matrix(matrix, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    positions = np.arange(len(classes))
    plt.xticks(positions, classes, rotation=30, ha="right")
    plt.yticks(positions, classes)

    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


save_confusion_matrix(
    rf_matrix,
    "E6 - Random Forest Confusion Matrix",
    "e6_randomforest_confusion_matrix.png",
)

save_confusion_matrix(
    cnn_matrix,
    "E6 - 1D CNN Confusion Matrix",
    "e6_cnn_confusion_matrix.png",
)


# ---------------------------------------------------------
# PART 6: SAVE RESULTS
# ---------------------------------------------------------

results = {
    "experiment": "E6 - Model Interpretability and Error Analysis",
    "classes": classes,
    "validation_samples": int(len(y_val)),
    "test_samples": int(len(y_test)),
    "random_forest": {
        "test_metrics": rf_metrics,
        "classification_report": rf_report,
        "confusion_matrix": rf_matrix.tolist(),
        "errors": rf_errors,
        "feature_importance": feature_results,
    },
    "cnn": {
        "test_metrics": cnn_metrics,
        "classification_report": cnn_report,
        "confusion_matrix": cnn_matrix.tolist(),
        "errors": cnn_errors,
    },
    "error_comparison": {
        "both_models": common_errors,
        "random_forest_only": rf_only_errors,
        "cnn_only": cnn_only_errors,
    },
    "interpretation_notes": [
        "Permutation importance was calculated on validation data.",
        "Test data was used for final error analysis.",
        "Feature importance indicates model reliance, not causation.",
        "The evaluation split is recording-level, not session-level.",
        "Date and class correlations may affect the results.",
    ],
}

results_path = OUTPUT_DIR / "e6_interpretability_results.json"

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nE6 COMPLETED SUCCESSFULLY")
print("Random Forest metrics:", rf_metrics)
print("CNN metrics:", cnn_metrics)
print("Feature plot:", feature_plot_path)
print("Results:", results_path)
print("Output folder:", OUTPUT_DIR)