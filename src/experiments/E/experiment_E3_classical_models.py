import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


ROOT = Path.cwd()
DATA_PATH = ROOT / "data/processed/experiment_e/experiment_E2_data.npz"
OUTPUT_DIR = ROOT / "data/processed/experiment_e/e3"
MODEL_DIR = ROOT / "models/experiment_e"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_array(data, possible_names):
    for name in possible_names:
        if name in data.files:
            return data[name]
    raise KeyError(
        f"Could not find any of {possible_names}. "
        f"Available keys: {data.files}"
    )


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


print("E3 CLASSICAL MACHINE LEARNING")
print("Loading E2 dataset...")

data = np.load(DATA_PATH, allow_pickle=True)

X_train = get_array(data, ["X_train", "X_features_train"])
X_val = get_array(data, ["X_validation", "X_val", "X_features_validation"])
X_test = get_array(data, ["X_test", "X_features_test"])

y_train = get_array(data, ["y_train", "labels_train"])
y_val = get_array(data, ["y_validation", "y_val", "labels_validation"])
y_test = get_array(data, ["y_test", "labels_test"])

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train.astype(str))
y_val = encoder.transform(y_val.astype(str))
y_test = encoder.transform(y_test.astype(str))

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("Classes:", list(encoder.classes_))

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_metrics = metrics(y_val, val_predictions)

    results[name] = {
        "validation": val_metrics,
        "validation_classification_report": classification_report(
            y_val,
            val_predictions,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }

    trained_models[name] = model

    model_path = MODEL_DIR / f"e3_{name.lower()}_model.joblib"
    joblib.dump(model, model_path)

    print("Validation:", val_metrics)
    print("Saved model:", model_path)

# Select the model using validation Macro F1.
best_name = max(
    results,
    key=lambda name: results[name]["validation"]["macro_f1"]
)

best_model = trained_models[best_name]

print(f"\nBest model selected using validation: {best_name}")

# Final test evaluation after model selection.
test_predictions = best_model.predict(X_test)
test_metrics = metrics(y_test, test_predictions)

results[best_name]["test"] = test_metrics
results[best_name]["test_classification_report"] = classification_report(
    y_test,
    test_predictions,
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0,
)

results["metadata"] = {
    "best_model": best_name,
    "selection_metric": "validation_macro_f1",
    "classes": encoder.classes_.tolist(),
    "train_samples": int(len(y_train)),
    "validation_samples": int(len(y_val)),
    "test_samples": int(len(y_test)),
    "feature_count": int(X_train.shape[1]),
}

output_path = OUTPUT_DIR / "e3_classical_results.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nE3 completed successfully.")
print("Best model:", best_name)
print("Final test metrics:", test_metrics)
print("Results saved:", output_path)