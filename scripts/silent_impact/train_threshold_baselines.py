
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path("data/processed/silent_impact/threshold_experiments")
OUTPUT_DIR = Path("models/silent_impact_baseline")
RESULTS_DIR = Path("reports/silent_impact_baseline")

THRESHOLDS = [10, 25, 50]


def extract_window_features(X):
    """Extract time-domain statistical features from (windows, frames, channels)."""
    features = []

    for window in X:
        row = []

        for channel in range(window.shape[1]):
            values = window[:, channel]
            differences = np.diff(values)

            row.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.max(values) - np.min(values),
                np.sqrt(np.mean(values ** 2)),
                np.mean(np.abs(differences)),
                np.std(differences),
            ])

        features.append(row)

    return np.asarray(features, dtype=np.float32)


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def evaluate_groups(y_true, y_pred, groups):
    results = {}

    for group in np.unique(groups):
        mask = groups == group

        if len(np.unique(y_true[mask])) < 2:
            results[str(group)] = {
                "samples": int(mask.sum()),
                "note": "Only one true class in this group; metrics may be incomplete.",
                **calculate_metrics(y_true[mask], y_pred[mask]),
            }
        else:
            results[str(group)] = {
                "samples": int(mask.sum()),
                **calculate_metrics(y_true[mask], y_pred[mask]),
            }

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for threshold in THRESHOLDS:
        print("\n" + "=" * 65)
        print(f"RANDOM FOREST BASELINE — THRESHOLD {threshold}%")
        print("=" * 65)

        train = np.load(
            DATA_DIR / f"train_threshold_{threshold}.npz"
        )
        test = np.load(
            DATA_DIR / f"test_threshold_{threshold}.npz"
        )

        X_train = extract_window_features(train["X"])
        X_test = extract_window_features(test["X"])

        y_train = train["y"]
        y_test = test["y"]

        print("Train feature shape:", X_train.shape)
        print("Test feature shape:", X_test.shape)

        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        print("\nTraining model...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        print("\nOverall metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["NO_SHOT", "SHOT"],
                zero_division=0,
            )
        )

        participant_results = evaluate_groups(
            y_test, predictions, test["participant_id"]
        )

        arm_results = evaluate_groups(
            y_test, predictions, test["arm"]
        )

        print("\nResults by participant:")
        for participant, result in participant_results.items():
            print(participant, result)

        print("\nResults by arm:")
        for arm, result in arm_results.items():
            print(arm, result)

        model_path = OUTPUT_DIR / f"random_forest_threshold_{threshold}.joblib"
        joblib.dump(model, model_path)

        report = {
            "threshold": threshold / 100,
            "model": "RandomForestClassifier",
            "parameters": {
                "n_estimators": 300,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
            },
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "metrics": metrics,
            "participant_results": participant_results,
            "arm_results": arm_results,
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["NO_SHOT", "SHOT"],
                output_dict=True,
                zero_division=0,
            ),
        }

        report_path = RESULTS_DIR / f"threshold_{threshold}_report.json"
        report_path.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        summary.append({
            "threshold": threshold,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

        print(f"\nModel saved: {model_path}")
        print(f"Report saved: {report_path}")

    summary_df = pd.DataFrame(summary)
    summary_path = RESULTS_DIR / "threshold_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("THRESHOLD COMPARISON")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
