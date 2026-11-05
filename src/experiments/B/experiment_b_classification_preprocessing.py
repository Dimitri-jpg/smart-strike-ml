
import pickle
import json
from pathlib import Path

import numpy as np


# ============================================================
# EXPERIMENT B2 - CLASSIFICATION PREPROCESSING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "public"
    / "silent_impact"
    / "raw"
    / "Classification_Dominant.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "silent_impact"
    / "classification_b2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# Silent Impact classification labels
CLASS_NAMES = {
    0: "Forehand Stroke",
    1: "Backhand Stroke",
    2: "Forehand Volley",
    3: "Backhand Volley",
    4: "Smash",
    5: "Serve",
}

CHANNEL_NAMES = [
    "linear_acc_x",
    "linear_acc_y",
    "linear_acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def load_dataset(path):
    print("\n[1] Loading Classification dataset...")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "rb") as file:
        data = pickle.load(file)

    data = np.asarray(data)

    print("Original shape:", data.shape)

    if data.ndim != 3:
        raise ValueError("Expected a 3-dimensional dataset.")

    if data.shape[1] != 181 or data.shape[2] != 6:
        raise ValueError(
            f"Unexpected dataset shape: {data.shape}. "
            "Expected (samples, 181, 6)."
        )

    return data


def extract_data(data):
    print("\n[2] Extracting metadata and sensor readings...")

    # First row contains metadata.
    labels = data[:, 0, 0].astype(int)
    participant_ids = data[:, 0, 1].astype(int)

    # Remaining 180 rows contain sensor readings.
    X = data[:, 1:, :].astype(np.float32)

    y = labels.copy()
    groups = participant_ids.copy()

    print("Sensor data shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Participants:", len(np.unique(groups)))

    return X, y, groups


def validate_data(X, y, groups):
    print("\n[3] Validating dataset...")

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or Inf values.")

    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y and groups have different lengths.")

    unique_labels = np.unique(y)

    if not set(unique_labels).issubset(set(CLASS_NAMES.keys())):
        raise ValueError(f"Unexpected class IDs: {unique_labels}")

    print("NaN values:", np.isnan(X).sum())
    print("Inf values:", np.isinf(X).sum())
    print("All values finite: YES")

    print("\nClass distribution:")

    for label in sorted(CLASS_NAMES):
        count = int(np.sum(y == label))
        print(f"{label} - {CLASS_NAMES[label]}: {count}")

    print("\nParticipant distribution:")

    for participant in sorted(np.unique(groups)):
        count = int(np.sum(groups == participant))
        print(f"Participant {participant}: {count} samples")


def create_participant_split(groups):
    print("\n[4] Creating participant-wise split...")

    participants = np.unique(groups)

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(participants)

    n_participants = len(shuffled)

    n_train = int(round(n_participants * TRAIN_RATIO))
    n_validation = int(round(n_participants * VALIDATION_RATIO))

    # Remaining participants go to test.
    train_participants = shuffled[:n_train]
    validation_participants = shuffled[
        n_train:n_train + n_validation
    ]
    test_participants = shuffled[n_train + n_validation:]

    train_mask = np.isin(groups, train_participants)
    validation_mask = np.isin(groups, validation_participants)
    test_mask = np.isin(groups, test_participants)

    print("Train participants:", sorted(train_participants.tolist()))
    print("Validation participants:", sorted(validation_participants.tolist()))
    print("Test participants:", sorted(test_participants.tolist()))

    # Verify that no participant appears in multiple splits.
    assert len(set(train_participants) & set(validation_participants)) == 0
    assert len(set(train_participants) & set(test_participants)) == 0
    assert len(set(validation_participants) & set(test_participants)) == 0

    print("\nParticipant overlap: NONE")

    return train_mask, validation_mask, test_mask


def normalize_data(X_train, X_validation, X_test):
    print("\n[5] Applying train-only standardization...")

    # Calculate channel statistics using TRAIN data only.
    # Combine samples and frames, keeping channels separate.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)

    # Avoid division by zero.
    std[std == 0] = 1.0

    X_train_normalized = (X_train - mean) / std
    X_validation_normalized = (X_validation - mean) / std
    X_test_normalized = (X_test - mean) / std

    print("Normalization method: Standardization")
    print("Mean calculated from: TRAIN only")
    print("Standard deviation calculated from: TRAIN only")

    print("\nNormalized train mean:", X_train_normalized.mean())
    print("Normalized train std:", X_train_normalized.std())

    return (
        X_train_normalized.astype(np.float32),
        X_validation_normalized.astype(np.float32),
        X_test_normalized.astype(np.float32),
        mean.reshape(-1).astype(float),
        std.reshape(-1).astype(float),
    )


def save_results(
    X_train, y_train, groups_train,
    X_validation, y_validation, groups_validation,
    X_test, y_test, groups_test,
    mean, std
):
    print("\n[6] Saving processed datasets...")

    output_path = OUTPUT_DIR / "classification_b2.npz"

    np.savez_compressed(
        output_path,

        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,

        X_validation=X_validation,
        y_validation=y_validation,
        groups_validation=groups_validation,

        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,

        normalization_mean=mean,
        normalization_std=std,
    )

    print("Saved:", output_path)

    metadata = {
        "experiment": "B2 - Classification Preprocessing",
        "dataset": "Silent Impact Classification Dominant",
        "original_samples": 6000,
        "frames_per_sample": 180,
        "sensor_channels": CHANNEL_NAMES,
        "classes": CLASS_NAMES,
        "random_seed": RANDOM_SEED,
        "split_method": "Participant-wise",
        "train_participants": sorted(
            np.unique(groups_train).astype(int).tolist()
        ),
        "validation_participants": sorted(
            np.unique(groups_validation).astype(int).tolist()
        ),
        "test_participants": sorted(
            np.unique(groups_test).astype(int).tolist()
        ),
        "normalization": "Standardization",
        "normalization_fit": "Training data only",
        "resampling": "Not applied",
        "trimming": "Not applied",
        "gyro_conversion": "Not applied - units require confirmation",
        "original_frame_count_preserved": True,
    }

    metadata_path = OUTPUT_DIR / "classification_b2_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("Saved:", metadata_path)


def main():
    print("=" * 60)
    print("SMARTSTRIKE - EXPERIMENT B2")
    print("CLASSIFICATION PREPROCESSING")
    print("=" * 60)

    data = load_dataset(DATA_PATH)

    X, y, groups = extract_data(data)

    validate_data(X, y, groups)

    train_mask, validation_mask, test_mask = (
        create_participant_split(groups)
    )

    X_train = X[train_mask]
    y_train = y[train_mask]
    groups_train = groups[train_mask]

    X_validation = X[validation_mask]
    y_validation = y[validation_mask]
    groups_validation = groups[validation_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]
    groups_test = groups[test_mask]

    print("\n[5] Split sizes:")
    print("Train:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)

    (
        X_train,
        X_validation,
        X_test,
        mean,
        std,
    ) = normalize_data(
        X_train,
        X_validation,
        X_test,
    )

    save_results(
        X_train, y_train, groups_train,
        X_validation, y_validation, groups_validation,
        X_test, y_test, groups_test,
        mean, std,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT B2 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
