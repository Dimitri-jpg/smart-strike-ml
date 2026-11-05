
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRS = [
    PROJECT_ROOT / "data/raw/smartstrike/SensorReadings/DatasetV4",
    PROJECT_ROOT / "data/raw/smartstrike/SensorReadings/NoneDatasetV4",
]

REPORT_DIR = PROJECT_ROOT / "reports/experiment_a"
MODEL_DIR = PROJECT_ROOT / "models/experiment_a"

RANDOM_STATE = 42
TEST_SIZE = 0.20

SENSOR_COLUMNS = [
    "linear_x",
    "linear_y",
    "linear_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

EXPECTED_CLASSES = [
    "BACKHAND",
    "BACKHAND_SLICE",
    "FOREHAND",
    "NONE",
]


# --------------------------------------------------
# 2. LOAD RECORDINGS
# --------------------------------------------------

def load_recordings():
    recordings = []

    for dataset_dir in DATASET_DIRS:
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir}"
            )

        csv_files = sorted(dataset_dir.rglob("*.csv"))

        print(f"\nFolder: {dataset_dir.name}")
        print(f"CSV files found: {len(csv_files)}")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                missing_columns = [
                    col for col in SENSOR_COLUMNS + ["shot_type"]
                    if col not in df.columns
                ]

                if missing_columns:
                    print(
                        f"Skipping {csv_file.name}: "
                        f"missing columns {missing_columns}"
                    )
                    continue

                labels = (
                    df["shot_type"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .unique()
                )

                if len(labels) != 1:
                    print(
                        f"Skipping {csv_file.name}: "
                        f"expected one label, found {labels}"
                    )
                    continue

                label = labels[0]

                if label not in EXPECTED_CLASSES:
                    print(
                        f"Skipping {csv_file.name}: "
                        f"unknown label {label}"
                    )
                    continue

                sensor_data = df[SENSOR_COLUMNS].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                sensor_data = sensor_data.replace(
                    [np.inf, -np.inf],
                    np.nan
                ).dropna()

                if len(sensor_data) < 2:
                    print(
                        f"Skipping {csv_file.name}: "
                        "not enough valid sensor rows"
                    )
                    continue

                recordings.append({
                    "file": csv_file.name,
                    "path": str(csv_file),
                    "label": label,
                    "data": sensor_data.reset_index(drop=True),
                })

            except Exception as error:
                print(f"Error reading {csv_file.name}: {error}")

    if not recordings:
        raise ValueError("No valid recordings were loaded.")

    return recordings


# --------------------------------------------------
# 3. FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(recording):
    """
    Extract statistical features from one recording.

    Each CSV recording becomes one feature vector.
    """

    data = recording["data"]
    features = {}

    for column in SENSOR_COLUMNS:
        values = data[column].to_numpy(dtype=float)

        features[f"{column}_mean"] = np.mean(values)
        features[f"{column}_std"] = np.std(values)
        features[f"{column}_min"] = np.min(values)
        features[f"{column}_max"] = np.max(values)
        features[f"{column}_median"] = np.median(values)

        features[f"{column}_q25"] = np.percentile(values, 25)
        features[f"{column}_q75"] = np.percentile(values, 75)

        features[f"{column}_rms"] = np.sqrt(
            np.mean(values ** 2)
        )

        features[f"{column}_abs_mean"] = np.mean(
            np.abs(values)
        )

        features[f"{column}_range"] = (
            np.max(values) - np.min(values)
        )

    # Calculate 3D sensor magnitudes
    linear_magnitude = np.sqrt(
        data["linear_x"] ** 2
        + data["linear_y"] ** 2
        + data["linear_z"] ** 2
    )

    gyro_magnitude = np.sqrt(
        data["gyro_x"] ** 2
        + data["gyro_y"] ** 2
        + data["gyro_z"] ** 2
    )

    for name, values in [
        ("linear_magnitude", linear_magnitude),
        ("gyro_magnitude", gyro_magnitude),
    ]:
        values = values.to_numpy(dtype=float)

        features[f"{name}_mean"] = np.mean(values)
        features[f"{name}_std"] = np.std(values)
        features[f"{name}_max"] = np.max(values)
        features[f"{name}_rms"] = np.sqrt(
            np.mean(values ** 2)
        )

    # Recording length in rows
    features["recording_length"] = len(data)

    return features


def create_feature_dataset(recordings):
    rows = []

    for index, recording in enumerate(recordings, start=1):
        features = extract_features(recording)

        features["file"] = recording["file"]
        features["path"] = recording["path"]
        features["label"] = recording["label"]

        rows.append(features)

        if index % 50 == 0:
            print(
                f"Extracted features from "
                f"{index}/{len(recordings)} recordings"
            )

    return pd.DataFrame(rows)


# --------------------------------------------------
# 4. PREPARE TRAIN AND TEST DATA
# --------------------------------------------------

def prepare_data(feature_df):
    metadata_columns = ["file", "path", "label"]

    feature_columns = [
        column
        for column in feature_df.columns
        if column not in metadata_columns
    ]

    X = feature_df[feature_columns].copy()
    y = feature_df["label"].copy()

    # Replace any remaining invalid feature values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Use training-independent median imputation values
    # by splitting first, then fitting medians on training data.
    indices = np.arange(len(feature_df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Fit imputation values using only training data
    medians = X_train.median()

    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
        train_indices,
        test_indices,
    )


# --------------------------------------------------
# 5. DEFINE BASELINE MODELS
# --------------------------------------------------

def create_models():
    models = {
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
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
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    return models


# --------------------------------------------------
# 6. EVALUATE MODELS
# --------------------------------------------------

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    predictions = {}

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training: {model_name}")
        print("=" * 60)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        })

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=EXPECTED_CLASSES,
                zero_division=0,
            )
        )

        print("Confusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                y_pred,
                labels=EXPECTED_CLASSES,
            )
        )

    results_df = pd.DataFrame(results)

    return results_df, predictions, models


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

def save_results(
    feature_df,
    results_df,
    predictions,
    models,
    y_test,
    feature_columns,
    test_indices,
):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save extracted features
    feature_df.drop(columns=["data"], errors="ignore").to_csv(
        REPORT_DIR / "recording_features.csv",
        index=False,
    )

    # Save model comparison
    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    # Save predictions for each model
    prediction_df = pd.DataFrame({
        "file": feature_df.iloc[test_indices]["file"].to_numpy(),
        "actual": y_test.to_numpy(),
    })

    for model_name, y_pred in predictions.items():
        safe_name = model_name.lower().replace(" ", "_")
        prediction_df[safe_name] = y_pred

    prediction_df.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    # Save classification reports and confusion matrices
    detailed_results = {}

    for model_name, model in models.items():
        y_pred = predictions[model_name]

        detailed_results[model_name] = {
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=EXPECTED_CLASSES,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                y_pred,
                labels=EXPECTED_CLASSES,
            ).tolist(),
        }

    with open(
        REPORT_DIR / "detailed_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            detailed_results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Save trained models
    for model_name, model in models.items():
        safe_name = model_name.lower().replace(" ", "_")

        joblib.dump(
            model,
            MODEL_DIR / f"{safe_name}.joblib",
        )

    # Save feature names
    with open(
        REPORT_DIR / "feature_names.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(feature_columns, file, indent=4)

    print("\nAll results saved successfully.")


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("SMARTSTRIKE - EXPERIMENT A")
    print("Baseline Classification using SmartStrike V4")
    print("=" * 60)

    # Load recordings
    recordings = load_recordings()

    print(f"\nTotal valid recordings: {len(recordings)}")

    print("\nDataset class distribution:")
    for label in EXPECTED_CLASSES:
        count = sum(
            recording["label"] == label
            for recording in recordings
        )
        print(f"{label}: {count}")

    # Extract features
    print("\nExtracting features...")
    feature_df = create_feature_dataset(recordings)

    print(f"\nFeature dataset shape: {feature_df.shape}")

    # Save a copy of features before training
    # (metadata and features only; raw sensor rows are not included)
    # Split data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_columns,
        train_indices,
        test_indices,
    ) = prepare_data(feature_df)

    print("\nTrain/Test split:")
    print(f"Training recordings: {len(X_train)}")
    print(f"Testing recordings:  {len(X_test)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    # Create and train models
    models = create_models()

    results_df, predictions, trained_models = evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Display comparison
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.sort_values(
            by="f1_macro",
            ascending=False,
        ).to_string(index=False)
    )

    # Save outputs
    save_results(
        feature_df,
        results_df,
        predictions,
        trained_models,
        y_test,
        feature_columns,
        test_indices,
    )

    print("\nExperiment A completed.")
    print(f"Reports: {REPORT_DIR}")
    print(f"Models:  {MODEL_DIR}")


if __name__ == "__main__":
    main()
