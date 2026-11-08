
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from joblib import dump


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path.cwd()

DATA_PATH = ROOT / "data/processed/experiment_e/experiment_E2_data.npz"
OUTPUT_DIR = ROOT / "data/processed/experiment_f/f1"
MODEL_DIR = ROOT / "models/experiment_f"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("F1 RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
print("=" * 60)

data = np.load(DATA_PATH, allow_pickle=True)

print("Available arrays:", data.files)

X_train = data["X_train"]
y_train = data["y_train"]

X_validation = data["X_validation"]
y_validation = data["y_validation"]

X_test = data["X_test"]
y_test = data["y_test"]

classes = np.unique(y_train)

print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Test:", X_test.shape)
print("Classes:", classes)


# --------------------------------------------------
# HYPERPARAMETER GRID
# --------------------------------------------------

parameter_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"]
}


# --------------------------------------------------
# CROSS VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

base_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

search = GridSearchCV(
    estimator=base_model,
    param_grid=parameter_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

print("\nStarting Grid Search...")
print("This may take some time.")

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation Macro F1:")
print(round(search.best_score_, 4))


# --------------------------------------------------
# VALIDATION EVALUATION
# --------------------------------------------------

validation_predictions = best_model.predict(X_validation)

validation_metrics = {
    "accuracy": float(
        accuracy_score(y_validation, validation_predictions)
    ),
    "macro_precision": float(
        precision_score(
            y_validation,
            validation_predictions,
            average="macro",
            zero_division=0
        )
    ),
    "macro_recall": float(
        recall_score(
            y_validation,
            validation_predictions,
            average="macro",
            zero_division=0
        )
    ),
    "macro_f1": float(
        f1_score(
            y_validation,
            validation_predictions,
            average="macro",
            zero_division=0
        )
    )
}

print("\nValidation metrics:")
print(validation_metrics)


# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

# Test is evaluated only after hyperparameters
# have been selected using training and validation.

test_predictions = best_model.predict(X_test)

test_metrics = {
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "macro_precision": float(
        precision_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0
        )
    ),
    "macro_recall": float(
        recall_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0
        )
    ),
    "macro_f1": float(
        f1_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0
        )
    )
}

print("\nFinal Test metrics:")
print(test_metrics)


# --------------------------------------------------
# SAVE MODEL AND RESULTS
# --------------------------------------------------

model_path = MODEL_DIR / "f1_randomforest_best.joblib"
dump(best_model, model_path)

results = {
    "experiment": "F1 Random Forest Hyperparameter Optimization",
    "best_parameters": search.best_params_,
    "best_cv_macro_f1": float(search.best_score_),
    "validation_metrics": validation_metrics,
    "test_metrics": test_metrics,
    "cv_folds": 5,
    "scoring": "macro_f1",
    "number_of_candidates": len(search.cv_results_["params"])
}

results_path = OUTPUT_DIR / "f1_randomforest_results.json"

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nF1 COMPLETED SUCCESSFULLY")
print("Model saved:", model_path)
print("Results saved:", results_path)
