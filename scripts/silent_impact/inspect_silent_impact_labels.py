
from pathlib import Path
import pickle
from collections import Counter

import numpy as np

DATA_DIR = Path("../../data/public/silent_impact/raw")

CLASSIFICATION_FILES = [
    "Classification_Dominant.pkl",
    "Classification_Passive.pkl",
]

DETECTION_FILES = [
    "Detection_Dominant.pkl",
    "Detection_Passive.pkl",
]


def inspect_classification(path):
    print("\n" + "=" * 65)
    print(f"CLASSIFICATION: {path.name}")

    with path.open("rb") as f:
        data = pickle.load(f)

    print("Shape:", data.shape)
    print("Expected interpretation to verify: samples x frames x channels")

    # The first frame appears to contain six one-hot label values.
    # Check this hypothesis; do not assume it is correct.
    candidate = data[:, 0, :]

    print("\nFirst frame: first 6 values")
    print(candidate[:10, :6])

    rounded = np.rint(candidate[:, :6]).astype(int)
    is_binary = np.isin(rounded, [0, 1]).all(axis=1)
    one_hot = (
        is_binary
        & (rounded.sum(axis=1) == 1)
    )

    print("\nFirst-frame label hypothesis:")
    print("Rows that look one-hot:", int(one_hot.sum()), "/", len(data))

    if one_hot.any():
        labels = np.argmax(rounded[one_hot], axis=1)
        counts = Counter(labels.tolist())

        for label, count in sorted(counts.items()):
            print(f"  Candidate class {label}: {count} samples")

    # Check whether the remaining 180 frames contain plausible sensor data.
    sensor_data = data[:, 1:, :]
    print("\nRemaining data shape:", sensor_data.shape)
    print("All finite:", bool(np.isfinite(sensor_data).all()))
    print("Min:", float(np.min(sensor_data)))
    print("Max:", float(np.max(sensor_data)))
    print("Mean:", float(np.mean(sensor_data)))
    print("Std:", float(np.std(sensor_data)))


def inspect_detection(path):
    print("\n" + "=" * 65)
    print(f"DETECTION: {path.name}")

    with path.open("rb") as f:
        data = pickle.load(f)

    print("Shape:", data.shape)
    print("All finite:", bool(np.isfinite(data).all()))

    # Inspect the first participant/sequence without assuming column meanings.
    sample = data[0]
    print("\nFirst sequence shape:", sample.shape)
    print("First 10 rows:")
    print(sample[:10])

    print("\nColumn statistics for first sequence:")
    for col in range(sample.shape[1]):
        values = sample[:, col]
        unique = np.unique(values)

        print(
            f"Column {col}: "
            f"min={values.min():.5f}, "
            f"max={values.max():.5f}, "
            f"mean={values.mean():.5f}, "
            f"unique_count={len(unique)}, "
            f"first_unique_values={unique[:12].tolist()}"
        )


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR.resolve()}"
        )

    print("SILENT IMPACT LABEL AND COLUMN INSPECTION")
    print("Read-only: no models or dataset files will be changed.")

    for filename in CLASSIFICATION_FILES:
        path = DATA_DIR / filename
        if not path.is_file():
            print(f"\nMISSING: {path}")
            continue
        inspect_classification(path)

    for filename in DETECTION_FILES:
        path = DATA_DIR / filename
        if not path.is_file():
            print(f"\nMISSING: {path}")
            continue
        inspect_detection(path)

    print("\n" + "=" * 65)
    print("Inspection finished.")
    print("Column meanings still need to be verified against the dataset documentation.")


if __name__ == "__main__":
    main()
