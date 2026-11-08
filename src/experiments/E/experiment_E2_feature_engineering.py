import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path.cwd()

MANIFEST = (
    ROOT / "data/processed/experiment_e/e1_split_manifest.json"
)

OUTPUT_DIR = ROOT / "data/processed/experiment_e"

CHANNELS = [
    "linear_x",
    "linear_y",
    "linear_z",
    "gyro_x",
    "gyro_y",
    "gyro_z"
]

TARGET_LENGTH = 180


def load_record(relative_path):
    path = ROOT / relative_path

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty recording: {relative_path}")

    signals = []

    for channel in CHANNELS:
        values = []

        for row in rows:
            value = row.get(channel)

            if value is None or value == "":
                values.append(np.nan)
            else:
                try:
                    values.append(float(value))
                except ValueError:
                    values.append(np.nan)

        values = np.asarray(values, dtype=np.float64)

        # Replace invalid values.
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        signals.append(values)

    return np.stack(signals, axis=1)


def resample_signal(signal, target_length=180):
    old_length = len(signal)

    if old_length == 0:
        raise ValueError("Signal has zero length.")

    if old_length == 1:
        return np.repeat(signal, target_length, axis=0)

    old_positions = np.linspace(0, 1, old_length)
    new_positions = np.linspace(0, 1, target_length)

    resampled = np.zeros(
        (target_length, signal.shape[1]),
        dtype=np.float32
    )

    for channel in range(signal.shape[1]):
        resampled[:, channel] = np.interp(
            new_positions,
            old_positions,
            signal[:, channel]
        )

    return resampled


def extract_features(signal):
    features = []

    for channel in range(signal.shape[1]):
        x = signal[:, channel]

        mean = np.mean(x)
        std = np.std(x)
        minimum = np.min(x)
        maximum = np.max(x)
        rms = np.sqrt(np.mean(x ** 2))

        q25 = np.percentile(x, 25)
        median = np.percentile(x, 50)
        q75 = np.percentile(x, 75)

        value_range = maximum - minimum
        energy = np.mean(x ** 2)

        features.extend([
            mean,
            std,
            minimum,
            maximum,
            rms,
            q25,
            median,
            q75,
            value_range,
            energy
        ])

    return np.asarray(features, dtype=np.float32)


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            "E1 manifest not found. Run E1 first."
        )

    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    records = manifest["records"]

    X_series = []
    X_features = []
    y = []

    print("E2 FEATURE ENGINEERING")
    print("Loading recordings:", len(records))

    for i, record in enumerate(records):
        signal = load_record(record["file"])

        signal = resample_signal(
            signal,
            TARGET_LENGTH
        )

        features = extract_features(signal)

        X_series.append(signal)
        X_features.append(features)
        y.append(record["label"])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(records)}")

    X_series = np.asarray(
        X_series,
        dtype=np.float32
    )

    X_features = np.asarray(
        X_features,
        dtype=np.float32
    )

    y = np.asarray(y)

    # Use the shared split indices from E1.
    splits = manifest["splits"]

    train_idx = np.asarray(splits["train"])
    val_idx = np.asarray(splits["validation"])
    test_idx = np.asarray(splits["test"])

    # Fit standardization using TRAINING data only.
    mean = X_features[train_idx].mean(axis=0)
    std = X_features[train_idx].std(axis=0)

    std[std < 1e-8] = 1.0

    X_scaled = (X_features - mean) / std

    # Save both representations:
    # 1. Statistical features for classical ML.
    # 2. Resampled time series for the later 1D CNN.
    output_file = OUTPUT_DIR / "experiment_E2_data.npz"

    np.savez_compressed(
        output_file,

        X_features=X_features,
        X_scaled=X_scaled,

        X_train=X_scaled[train_idx],
        X_validation=X_scaled[val_idx],
        X_test=X_scaled[test_idx],

        X_series=X_series,
        X_series_train=X_series[train_idx],
        X_series_validation=X_series[val_idx],
        X_series_test=X_series[test_idx],

        y=y,
        y_train=y[train_idx],
        y_validation=y[val_idx],
        y_test=y[test_idx],

        feature_mean=mean,
        feature_std=std,

        train_indices=train_idx,
        validation_indices=val_idx,
        test_indices=test_idx
    )

    metadata = {
        "experiment": "E2",
        "total_recordings": len(records),
        "channels": CHANNELS,
        "target_sequence_length": TARGET_LENGTH,
        "features_per_channel": 10,
        "total_features": int(X_features.shape[1]),
        "time_series_shape": list(X_series.shape),
        "feature_matrix_shape": list(X_features.shape),
        "train_shape": list(X_scaled[train_idx].shape),
        "validation_shape": list(X_scaled[val_idx].shape),
        "test_shape": list(X_scaled[test_idx].shape),
        "normalization": "StandardScaler-style normalization",
        "normalization_fit": "Training split only",
        "random_state": manifest["random_state"],
        "limitations": [
            "Recording-level splits are not session-level splits.",
            "Date/class confounding remains.",
            "Resampling to 180 frames does not guarantee physical time alignment."
        ]
    }

    metadata_file = OUTPUT_DIR / "e2_preprocessing_metadata.json"

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\nE2 completed successfully.")
    print("Feature matrix:", X_features.shape)
    print("Time series:", X_series.shape)
    print("Training:", X_scaled[train_idx].shape)
    print("Validation:", X_scaled[val_idx].shape)
    print("Test:", X_scaled[test_idx].shape)
    print("Saved:", output_file)
    print("Saved:", metadata_file)


if __name__ == "__main__":
    main()