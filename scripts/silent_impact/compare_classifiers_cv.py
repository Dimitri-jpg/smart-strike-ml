
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    """Extract 8 statistical features from each sensor channel."""

    features = []

    for channel in range(X.shape[2]):
        signal = X[:, :, channel]

        channel_features = np.column_stack([
            np.mean(signal, axis=1),
            np.std(signal, axis=1),
            np.min(signal, axis=1),
            np.max(signal, axis=1),
            np.max(signal, axis=1) - np.min(signal, axis=1),
            np.sqrt(np.mean(signal ** 2, axis=1)),
            np.mean(np.abs(np.diff(signal, axis=1)), axis=1),
            np.std(np.diff(signal, axis=1), axis=1),
        ])

        features.append(channel_features)

    return np.concatenate(features, axis=1)


def create_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def main():

    train_path = DATA_DIR / "train_threshold_25.npz"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {train_path}"
        )

    data = np.load(train_path)

    X = data["X"]
    y = data["y"]
    groups = data["participant_id"]

    print("=" * 65)
    print("CLASSIFIER COMPARISON - GROUP CROSS VALIDATION")
    print("=" * 65)

    print(f"Windows: {len(X)}")
    print(f"Participants: {np.unique(groups).tolist()}")
    print(f"Features: 48")
    print(f"NO_SHOT: {np.sum(y == 0)}")
    print(f"SHOT: {np.sum(y == 1)}")

    X_features = extract_features(X)

    cv = GroupKFold(n_splits=5)

    models = create_models()

    all_results = []

    for model_name, model_template in models.items():

        print("\n" + "#" * 65)
        print(f"MODEL: {model_name}")
        print("#" * 65)

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

            if set(train_groups) & set(val_groups):
                raise ValueError("Participant leakage detected!")

            # Create a fresh model for every fold.
            model = type(model_template)(
                **model_template.get_params()
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            result = {
                "model": model_name,
                "fold": fold,
                "participants": ",".join(
                    map(str, val_groups)
                ),
                "accuracy": accuracy_score(
                    y_val, predictions
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_val, predictions
                ),
                "precision": precision_score(
                    y_val, predictions,
                    zero_division=0
                ),
                "recall": recall_score(
                    y_val, predictions,
                    zero_division=0
                ),
                "f1": f1_score(
                    y_val, predictions,
                    zero_division=0
                ),
            }

            all_results.append(result)

            print(
                f"Fold {fold} | "
                f"Participants: {val_groups.tolist()} | "
                f"F1: {result['f1']:.4f} | "
                f"Balanced Accuracy: "
                f"{result['balanced_accuracy']:.4f}"
            )

    results_df = pd.DataFrame(all_results)

    results_path = (
        REPORT_DIR / "classifier_comparison_cv_folds.csv"
    )

    results_df.to_csv(results_path, index=False)

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]

    summary = (
        results_df
        .groupby("model")[metrics]
        .agg(["mean", "std"])
    )

    summary_path = (
        REPORT_DIR / "classifier_comparison_cv_summary.csv"
    )

    summary.to_csv(summary_path)

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df
        .groupby("model")[metrics]
        .mean()
        .sort_values("f1", ascending=False)
        .to_string()
    )

    print("\nMEAN F1 ± STANDARD DEVIATION")

    for model_name, group in results_df.groupby("model"):
        mean_f1 = group["f1"].mean()
        std_f1 = group["f1"].std()

        print(
            f"{model_name}: "
            f"{mean_f1:.4f} ± {std_f1:.4f}"
        )

    print(f"\nFold results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
