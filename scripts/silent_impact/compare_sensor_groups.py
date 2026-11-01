
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR = Path("data/processed/silent_impact/threshold_experiments")
OUTPUT_DIR = Path("reports/silent_impact_baseline")

SENSOR_GROUPS = {
    "accelerometer_only": [0, 1, 2],
    "gyroscope_only": [3, 4, 5],
    "all_sensors": [0, 1, 2, 3, 4, 5],
}


def extract_features(X):
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = np.load(DATA_DIR / "train_threshold_25.npz")
    test = np.load(DATA_DIR / "test_threshold_25.npz")

    results = []

    for group_name, channels in SENSOR_GROUPS.items():
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: {group_name}")
        print(f"{'=' * 60}")

        X_train = extract_features(train["X"][:, :, channels])
        X_test = extract_features(test["X"][:, :, channels])

        y_train = train["y"]
        y_test = test["y"]

        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "sensor_group": group_name,
            "feature_count": X_train.shape[1],
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
        }

        results.append(metrics)

        print(pd.Series(metrics).to_string())

        joblib.dump(
            model,
            OUTPUT_DIR / f"random_forest_{group_name}.joblib",
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "sensor_group_comparison.csv",
        index=False,
    )

    print("\nFINAL SENSOR GROUP COMPARISON")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
