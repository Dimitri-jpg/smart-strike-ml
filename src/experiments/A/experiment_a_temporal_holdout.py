
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "reports/experiment_a/dataset_profiling/"
    / "complete_dataset_metadata.csv"
)

FEATURE_FILE = (
    ROOT
    / "reports/experiment_a/"
    / "recording_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "reports/experiment_a/temporal_holdout"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    metadata = pd.read_csv(INPUT_FILE)
    features = pd.read_csv(FEATURE_FILE)

    metadata["timestamp"] = pd.to_datetime(
        metadata["timestamp"]
    )

    # Match metadata and features by filename.
    if "file" in features.columns:
        features["filename"] = features["file"].apply(
            lambda x: Path(str(x)).name
        )
    elif "filename" not in features.columns:
        raise ValueError(
            "Feature file must contain 'file' or 'filename'."
        )

    if "label" not in features.columns:
        raise ValueError(
            "Feature file does not contain a label column."
        )

    merged = metadata.merge(
        features,
        on="filename",
        suffixes=("_metadata", "_features"),
        how="inner",
    )

    if len(merged) != len(metadata):
        raise ValueError(
            f"Metadata/features mismatch: "
            f"{len(metadata)} metadata rows, "
            f"{len(merged)} matched rows."
        )

    if not (
        merged["label_metadata"]
        == merged["label_features"]
    ).all():
        raise ValueError("Label mismatch detected.")

    merged["label"] = merged["label_metadata"]

    return merged


# --------------------------------------------------
# CHRONOLOGICAL SPLIT WITHIN EACH CLASS
# --------------------------------------------------

def create_temporal_split(df):
    train_parts = []
    test_parts = []

    print("\nTEMPORAL SPLIT BY CLASS")
    print("=" * 65)

    for label, group in df.groupby("label"):
        group = group.sort_values("timestamp").copy()

        split_index = int(len(group) * (1 - TEST_SIZE))

        train_group = group.iloc[:split_index]
        test_group = group.iloc[split_index:]

        train_parts.append(train_group)
        test_parts.append(test_group)

        print(f"\nClass: {label}")
        print(f"Total: {len(group)}")
        print(f"Train: {len(train_group)}")
        print(f"Test: {len(test_group)}")

        print(
            "Train period:",
            train_group["timestamp"].min(),
            "to",
            train_group["timestamp"].max(),
        )

        print(
            "Test period:",
            test_group["timestamp"].min(),
            "to",
            test_group["timestamp"].max(),
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    return train_df, test_df


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(train_df, test_df):
    excluded_columns = {
        "filename",
        "file",
        "path",
        "label",
        "label_metadata",
        "label_features",
        "date",
        "time",
        "timestamp",
        "recording_id",
        "score",
        "dataset_folder",
    }

    feature_columns = [
        col
        for col in train_df.columns
        if col not in excluded_columns
        and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    X_train = train_df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_test = test_df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    return X_train, X_test, y_train, y_test, feature_columns


# --------------------------------------------------
# MODELS
# --------------------------------------------------

def build_models():
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]),

        "SVM RBF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                ),
            ),
        ]),
    }


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"Macro F1:       {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    labels = sorted(y_train.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("\nConfusion matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "predictions": predictions,
        "fitted_model": model,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("SMARTSTRIKE - WITHIN-CLASS TEMPORAL HOLDOUT")
    print("=" * 65)

    df = load_data()

    print(f"\nTotal recordings: {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    train_df, test_df = create_temporal_split(df)

    X_train, X_test, y_train, y_test, feature_columns = (
        prepare_features(train_df, test_df)
    )

    print("\nFINAL SPLIT")
    print("=" * 65)
    print(f"Training recordings: {len(X_train)}")
    print(f"Testing recordings:  {len(X_test)}")
    print(f"Features: {len(feature_columns)}")

    print("\nTrain class distribution:")
    print(y_train.value_counts())

    print("\nTest class distribution:")
    print(y_test.value_counts())

    results = []

    for name, model in build_models().items():
        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(result)

    # Save metrics
    metrics = pd.DataFrame([
        {
            "model": result["model"],
            "accuracy": result["accuracy"],
            "precision_macro": result["precision_macro"],
            "recall_macro": result["recall_macro"],
            "f1_macro": result["f1_macro"],
        }
        for result in results
    ])

    metrics.to_csv(
        OUTPUT_DIR / "temporal_holdout_metrics.csv",
        index=False,
    )

    # Save predictions
    prediction_rows = []

    for result in results:
        for index, (_, row) in enumerate(test_df.iterrows()):
            prediction_rows.append({
                "model": result["model"],
                "filename": row["filename"],
                "timestamp": row["timestamp"],
                "actual_label": y_test.iloc[index],
                "predicted_label": result["predictions"][index],
            })

    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT_DIR / "temporal_holdout_predictions.csv",
        index=False,
    )

    # Save split metadata
    train_df.to_csv(
        OUTPUT_DIR / "train_recordings.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "test_recordings.csv",
        index=False,
    )

    summary = {
        "total_recordings": len(df),
        "train_recordings": len(train_df),
        "test_recordings": len(test_df),
        "test_size_per_class": TEST_SIZE,
        "feature_count": len(feature_columns),
        "metrics": metrics.to_dict(orient="records"),
    }

    with open(
        OUTPUT_DIR / "temporal_holdout_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("TEMPORAL HOLDOUT COMPLETED")
    print("=" * 65)

    print(metrics.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
