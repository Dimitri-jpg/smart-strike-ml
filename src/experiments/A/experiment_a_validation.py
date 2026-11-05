
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = ROOT / "reports/experiment_a/recording_features.csv"
OUTPUT_DIR = ROOT / "reports/experiment_a/validation"

RANDOM_STATE = 42
N_SPLITS = 5

CLASSES = [
    "BACKHAND",
    "BACKHAND_SLICE",
    "FOREHAND",
    "NONE",
]

METADATA_COLUMNS = ["file", "path", "label"]


# --------------------------------------------------
# 2. LOAD FEATURE DATA
# --------------------------------------------------

def load_data():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_FILE}"
        )

    df = pd.read_csv(FEATURES_FILE)

    df = df.replace([np.inf, -np.inf], np.nan)

    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df["label"].copy()

    # Impute missing values using dataset medians.
    # For cross-validation, use a Pipeline for model scaling;
    # the feature imputation here is only a safeguard.
    medians = X.median()
    X = X.fillna(medians).fillna(0)

    print("\nDataset loaded")
    print(f"Recordings: {len(df)}")
    print(f"Features: {len(feature_columns)}")

    print("\nClass distribution:")
    print(y.value_counts())

    return df, X, y, feature_columns


# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

def create_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
            )),
        ]),
    }


# --------------------------------------------------
# 4. STRATIFIED 5-FOLD CROSS-VALIDATION
# --------------------------------------------------

def run_cross_validation(X, y):
    print("\n" + "=" * 60)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = create_models()

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

            print(
                f"{metric}: "
                f"{values.mean():.4f} "
                f"+/- {values.std():.4f}"
            )

        results.append(row)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "cross_validation_results.csv",
        index=False,
    )

    return results_df


# --------------------------------------------------
# 5. EXTRACT TIMESTAMP FROM FILENAME
# --------------------------------------------------

def extract_timestamp(filename):
    """
    Extract YYYYMMDD_HHMMSS from a recording filename.
    """

    match = re.search(
        r"_(\d{8})_(\d{6})_",
        str(filename)
    )

    if not match:
        return pd.NaT

    timestamp_string = (
        match.group(1) + match.group(2)
    )

    return pd.to_datetime(
        timestamp_string,
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )


# --------------------------------------------------
# 6. CHRONOLOGICAL HOLDOUT
# --------------------------------------------------

def run_chronological_holdout(df, X, y, feature_columns):
    print("\n" + "=" * 60)
    print("CHRONOLOGICAL HOLDOUT")
    print("=" * 60)

    timestamps = df["file"].apply(extract_timestamp)

    if timestamps.isna().any():
        invalid_count = timestamps.isna().sum()

        print(
            f"WARNING: {invalid_count} recordings "
            "have no parseable timestamp."
        )

    valid_mask = timestamps.notna()

    if valid_mask.sum() < len(df):
        print(
            "Rows without timestamps will be excluded "
            "from chronological validation."
        )

    valid_indices = np.where(valid_mask)[0]

    sorted_indices = sorted(
        valid_indices,
        key=lambda i: timestamps.iloc[i],
    )

    split_position = int(len(sorted_indices) * 0.80)

    train_indices = sorted_indices[:split_position]
    test_indices = sorted_indices[split_position:]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print(f"\nTraining recordings: {len(X_train)}")
    print(f"Testing recordings: {len(X_test)}")

    print(
        "\nTraining date range:",
        timestamps.iloc[train_indices].min(),
        "to",
        timestamps.iloc[train_indices].max(),
    )

    print(
        "\nTesting date range:",
        timestamps.iloc[test_indices].min(),
        "to",
        timestamps.iloc[test_indices].max(),
    )

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    missing_classes = set(CLASSES) - set(y_train.unique())
    missing_test_classes = set(CLASSES) - set(y_test.unique())

    if missing_classes or missing_test_classes:
        print("\nChronological split cannot evaluate all classes.")

        print("Missing from training:", missing_classes)
        print("Missing from testing:", missing_test_classes)

        print(
            "Results will not be calculated because "
            "the split does not contain all four classes."
        )

        return None

    models = create_models()
    results = []

    for name, model in models.items():
        print("\n" + "-" * 50)
        print(f"Training {name}")
        print("-" * 50)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=CLASSES,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=CLASSES,
        )

        print("Confusion matrix:")
        print(matrix)

        results.append({
            "model": name,
            "accuracy": accuracy,
            "f1_macro": f1,
        })

        prediction_df = pd.DataFrame({
            "file": df.iloc[test_indices]["file"].to_numpy(),
            "timestamp": timestamps.iloc[test_indices].to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": predictions,
        })

        safe_name = name.lower().replace(" ", "_")

        prediction_df.to_csv(
            OUTPUT_DIR / f"chronological_{safe_name}_predictions.csv",
            index=False,
        )

        if name == "Random Forest":
            joblib.dump(
                model,
                OUTPUT_DIR / "chronological_random_forest.joblib",
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "chronological_results.csv",
        index=False,
    )

    return results_df


# --------------------------------------------------
# 7. RANDOM FOREST FEATURE IMPORTANCE
# --------------------------------------------------

def analyze_feature_importance(X, y, feature_columns):
    print("\n" + "=" * 60)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    print("\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))

    joblib.dump(
        model,
        OUTPUT_DIR / "feature_importance_model.joblib",
    )

    return importance_df


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSMARTSTRIKE - EXPERIMENT A VALIDATION")

    df, X, y, feature_columns = load_data()

    # Check that all classes have enough samples for 5 folds
    class_counts = y.value_counts()

    if class_counts.min() < N_SPLITS:
        raise ValueError(
            "Not enough samples in one or more classes "
            "for 5-fold cross-validation."
        )

    cv_results = run_cross_validation(X, y)

    chronological_results = run_chronological_holdout(
        df,
        X,
        y,
        feature_columns,
    )

    importance_df = analyze_feature_importance(
        X,
        y,
        feature_columns,
    )

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print("\nCross-validation results:")
    print(cv_results.to_string(index=False))

    if chronological_results is not None:
        print("\nChronological results:")
        print(chronological_results.to_string(index=False))

    print("\nTop 10 important features:")
    print(importance_df.head(10).to_string(index=False))

    print("\nAll validation files saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
