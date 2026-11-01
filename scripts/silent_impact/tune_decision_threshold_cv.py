
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

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def extract_features(X):
    features = []

    for channel in range(X.shape[2]):
        signal = X[:, :, channel]
        diff = np.diff(signal, axis=1)

        channel_features = np.column_stack([
            np.mean(signal, axis=1),
            np.std(signal, axis=1),
            np.min(signal, axis=1),
            np.max(signal, axis=1),
            np.max(signal, axis=1) - np.min(signal, axis=1),
            np.sqrt(np.mean(signal ** 2, axis=1)),
            np.mean(np.abs(diff), axis=1),
            np.std(diff, axis=1),
        ])

        features.append(channel_features)

    return np.concatenate(features, axis=1)


def create_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


def main():

    train_path = DATA_DIR / "train_threshold_25.npz"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    data = np.load(train_path)

    X = data["X"]
    y = data["y"]
    groups = data["participant_id"]

    X_features = extract_features(X)

    print("=" * 65)
    print("DECISION THRESHOLD TUNING - NESTED GROUP CV")
    print("=" * 65)

    print(f"Windows: {len(X)}")
    print(f"Participants: {np.unique(groups).tolist()}")
    print(f"Thresholds: {THRESHOLDS}")

    outer_cv = GroupKFold(n_splits=5)

    fold_results = []
    threshold_details = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(
        outer_cv.split(X_features, y, groups),
        start=1
    ):

        X_outer_train = X_features[outer_train_idx]
        y_outer_train = y[outer_train_idx]
        groups_outer_train = groups[outer_train_idx]

        X_outer_val = X_features[outer_val_idx]
        y_outer_val = y[outer_val_idx]

        outer_val_groups = np.unique(groups[outer_val_idx])

        # Inner CV: select threshold using outer training participants only.
        inner_cv = GroupKFold(n_splits=4)

        inner_scores = {t: [] for t in THRESHOLDS}

        for inner_train_idx, inner_val_idx in inner_cv.split(
            X_outer_train,
            y_outer_train,
            groups_outer_train
        ):

            X_inner_train = X_outer_train[inner_train_idx]
            y_inner_train = y_outer_train[inner_train_idx]

            X_inner_val = X_outer_train[inner_val_idx]
            y_inner_val = y_outer_train[inner_val_idx]

            model = create_model()
            model.fit(X_inner_train, y_inner_train)

            probabilities = model.predict_proba(X_inner_val)[:, 1]

            for threshold in THRESHOLDS:
                metrics = calculate_metrics(
                    y_inner_val,
                    probabilities,
                    threshold
                )

                inner_scores[threshold].append(metrics["f1"])

        mean_inner_f1 = {
            threshold: np.mean(scores)
            for threshold, scores in inner_scores.items()
        }

        # Select the threshold with the highest mean inner F1.
        selected_threshold = max(
            THRESHOLDS,
            key=lambda t: mean_inner_f1[t]
        )

        # Refit using all outer training participants.
        final_model = create_model()
        final_model.fit(X_outer_train, y_outer_train)

        outer_probabilities = final_model.predict_proba(
            X_outer_val
        )[:, 1]

        final_metrics = calculate_metrics(
            y_outer_val,
            outer_probabilities,
            selected_threshold
        )

        result = {
            "outer_fold": outer_fold,
            "validation_participants": ",".join(
                map(str, outer_val_groups)
            ),
            "selected_threshold": selected_threshold,
            **final_metrics,
        }

        fold_results.append(result)

        for threshold in THRESHOLDS:
            threshold_details.append({
                "outer_fold": outer_fold,
                "threshold": threshold,
                "inner_mean_f1": mean_inner_f1[threshold],
                "selected": threshold == selected_threshold,
            })

        print("\n" + "=" * 65)
        print(f"OUTER FOLD {outer_fold}")
        print("=" * 65)

        print(f"Validation participants: {outer_val_groups.tolist()}")
        print(f"Selected threshold: {selected_threshold:.2f}")

        print("Inner CV mean F1:")
        for threshold in THRESHOLDS:
            print(
                f"  {threshold:.2f}: "
                f"{mean_inner_f1[threshold]:.4f}"
            )

        print("\nOuter validation metrics:")
        for metric, value in final_metrics.items():
            print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame(fold_results)
    details_df = pd.DataFrame(threshold_details)

    results_path = (
        REPORT_DIR / "nested_threshold_cv_results.csv"
    )

    details_path = (
        REPORT_DIR / "nested_threshold_cv_details.csv"
    )

    results_df.to_csv(results_path, index=False)
    details_df.to_csv(details_path, index=False)

    print("\n" + "=" * 65)
    print("NESTED GROUP CV SUMMARY")
    print("=" * 65)

    print(results_df.to_string(index=False))

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]

    print("\nMean metrics:")
    print(results_df[metric_columns].mean().to_string())

    print("\nStandard deviation:")
    print(results_df[metric_columns].std().to_string())

    print("\nSelected thresholds:")
    print(
        results_df["selected_threshold"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(f"\nResults saved: {results_path}")
    print(f"Threshold details saved: {details_path}")


if __name__ == "__main__":
    main()
