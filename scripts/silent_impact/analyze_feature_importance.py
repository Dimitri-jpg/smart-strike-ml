
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

DATA_DIR = Path("data/processed/silent_impact/threshold_experiments")
MODEL_PATH = Path(
    "models/silent_impact_baseline/random_forest_threshold_25.joblib"
)
OUTPUT_DIR = Path("reports/silent_impact_baseline")

CHANNELS = [
    "channel_0",
    "channel_1",
    "channel_2",
    "channel_3",
    "channel_4",
    "channel_5",
]

STATISTICS = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "rms",
    "mean_abs_diff",
    "std_diff",
]

FEATURE_NAMES = [
    f"{channel}_{stat}"
    for channel in CHANNELS
    for stat in STATISTICS
]


def extract_window_features(X):
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

    model = joblib.load(MODEL_PATH)

    X_train = extract_window_features(train["X"])
    X_test = extract_window_features(test["X"])

    y_train = train["y"]
    y_test = test["y"]

    # Random Forest built-in feature importance
    impurity_importance = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTOP 15 FEATURES — BUILT-IN IMPORTANCE")
    print(impurity_importance.head(15).to_string(index=False))

    # Permutation importance on held-out test data
    print("\nCalculating permutation importance...")

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="f1",
        n_jobs=-1,
    )

    permutation_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance_mean": permutation.importances_mean,
        "importance_std": permutation.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print("\nTOP 15 FEATURES — PERMUTATION IMPORTANCE")
    print(permutation_df.head(15).to_string(index=False))

    impurity_importance.to_csv(
        OUTPUT_DIR / "feature_importance_builtin.csv",
        index=False,
    )

    permutation_df.to_csv(
        OUTPUT_DIR / "feature_importance_permutation.csv",
        index=False,
    )

    print("\nFeature importance analysis completed.")


if __name__ == "__main__":
    main()
