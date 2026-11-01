
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "silent_impact"
    / "threshold_experiments"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "silent_impact_baseline"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def extract_features(X):
    """Extract 8 statistical features from each of 6 sensor channels."""

    features = []

    for channel in range(X.shape[2]):
        signal = X[:, :, channel]

        mean = np.mean(signal, axis=1)
        std = np.std(signal, axis=1)
        minimum = np.min(signal, axis=1)
        maximum = np.max(signal, axis=1)
        value_range = maximum - minimum
        rms = np.sqrt(np.mean(signal ** 2, axis=1))

        diff = np.diff(signal, axis=1)
        mean_abs_diff = np.mean(np.abs(diff), axis=1)
        std_diff = np.std(diff, axis=1)

        channel_features = np.column_stack([
            mean,
            std,
            minimum,
            maximum,
            value_range,
            rms,
            mean_abs_diff,
            std_diff,
        ])

        features.append(channel_features)

    return np.concatenate(features, axis=1)


def main():

    train_path = DATA_DIR / "train_threshold_25.npz"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    data = np.load(train_path)

    X = data["X"]
    y = data["y"]
    groups = data["participant_id"]

    print("=" * 60)
    print("GROUP CROSS-VALIDATION - SILENT IMPACT")
    print("=" * 60)

    print(f"Windows: {len(X)}")
    print(f"Participants: {np.unique(groups).tolist()}")
    print(f"Features per window: 48")
    print(f"NO_SHOT: {np.sum(y == 0)}")
    print(f"SHOT: {np.sum(y == 1)}")

    X_features = extract_features(X)

    cv = GroupKFold(n_splits=5)

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_features, y, groups),
        start=1
    ):

        X_train = X_features[train_idx]
        y_train = y[train_idx]

        X_val = X_features[val_idx]
        y_val = y[val_idx]

        train_groups = np.unique(groups[train_idx])
        val_groups = np.unique(groups[val_idx])

        overlap = set(train_groups) & set(val_groups)

        if overlap:
            raise ValueError(
                f"Participant leakage detected: {overlap}"
            )

        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        metrics = {
            "fold": fold,
            "participants": ",".join(map(str, val_groups)),
            "train_windows": len(train_idx),
            "validation_windows": len(val_idx),
            "accuracy": accuracy_score(y_val, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_val, predictions
            ),
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
        }

        results.append(metrics)

        print("\n" + "=" * 60)
        print(f"FOLD {fold}")
        print("=" * 60)

        print(f"Validation participants: {val_groups.tolist()}")
        print(f"Training participants: {train_groups.tolist()}")
        print(f"Validation windows: {len(val_idx)}")

        print("\nMetrics:")

        for metric, value in metrics.items():
            if isinstance(value, (float, np.floating)):
                print(f"{metric}: {value:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_val,
                predictions,
                target_names=["NO_SHOT", "SHOT"],
                zero_division=0,
            )
        )

    results_df = pd.DataFrame(results)

    output_path = REPORT_DIR / "group_cross_validation.csv"
    results_df.to_csv(output_path, index=False)

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print("\nMean metrics:")
    print(results_df[metric_columns].mean().to_string())

    print("\nStandard deviation:")
    print(results_df[metric_columns].std().to_string())

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
