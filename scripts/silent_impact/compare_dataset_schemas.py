
from pathlib import Path
import numpy as np
import pandas as pd

# Project root: smartstike-ml
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "silent_impact"

# Android SmartStrike expected CSV schema
ANDROID_COLUMNS = [
    "timestamp",
    "shot_id",
    "shot_type",
    "accel_x", "accel_y", "accel_z",
    "linear_x", "linear_y", "linear_z",
    "gyro_x", "gyro_y", "gyro_z",
    "rot_x", "rot_y", "rot_z", "rot_w",
]

ANDROID_SENSOR_GROUPS = {
    "Accelerometer": ["accel_x", "accel_y", "accel_z"],
    "Linear acceleration": ["linear_x", "linear_y", "linear_z"],
    "Gyroscope": ["gyro_x", "gyro_y", "gyro_z"],
    "Rotation vector": ["rot_x", "rot_y", "rot_z", "rot_w"],
}

SILENT_IMPACT_CHANNELS = {
    "channel_0": "Linear acceleration X",
    "channel_1": "Linear acceleration Y",
    "channel_2": "Linear acceleration Z",
    "channel_3": "Angular velocity X",
    "channel_4": "Angular velocity Y",
    "channel_5": "Angular velocity Z",
}


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def inspect_silent_impact():
    print_section("1. SILENT IMPACT DATASET")

    print("Expected sensor channels:")
    for channel, description in SILENT_IMPACT_CHANNELS.items():
        print(f"  {channel}: {description}")

    files = [
        PROCESSED_DIR / "train_windows.npz",
        PROCESSED_DIR / "test_windows.npz",
    ]

    for file_path in files:
        if not file_path.exists():
            print(f"\nNot found: {file_path}")
            continue

        print(f"\nFile: {file_path.name}")

        with np.load(file_path, allow_pickle=True) as data:
            X = data["X"]
            y = data["y"]

            print(f"  X shape: {X.shape}")
            print(f"  Number of windows: {len(X)}")
            print(f"  Frames per window: {X.shape[1]}")
            print(f"  Number of channels: {X.shape[2]}")
            print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
            print(f"  All values finite: {np.isfinite(X).all()}")

            if "participant_id" in data:
                print(
                    "  Participants:",
                    sorted(np.unique(data["participant_id"]).tolist())
                )

            if "arm" in data:
                print(
                    "  Arms:",
                    dict(zip(*np.unique(
                        data["arm"], return_counts=True
                    )))
                )

    metadata = PROCESSED_DIR / "preprocessing_metadata.txt"

    if metadata.exists():
        print("\nPreprocessing metadata:")
        print(metadata.read_text(encoding="utf-8"))


def inspect_android_csv(file_path):
    print_section(f"ANDROID CSV: {file_path.name}")

    try:
        df = pd.read_csv(file_path)
    except Exception as error:
        print(f"Could not read CSV: {error}")
        return

    print(f"Path: {file_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nCSV columns:")
    for column in df.columns:
        print(f"  - {column}")

    available = set(df.columns)
    expected = set(ANDROID_COLUMNS)

    print("\nExpected Android columns:")
    print(f"  Present: {sorted(expected & available)}")
    print(f"  Missing: {sorted(expected - available)}")
    print(f"  Additional: {sorted(available - expected)}")

    print("\nSensor group availability:")
    for group, columns in ANDROID_SENSOR_GROUPS.items():
        present = [column for column in columns if column in available]
        print(f"  {group}: {len(present)}/{len(columns)} channels")

    # Inspect timestamp intervals if possible
    if "timestamp" in df.columns:
        timestamps = pd.to_numeric(
            df["timestamp"], errors="coerce"
        ).dropna()

        if len(timestamps) > 1:
            differences = timestamps.diff().dropna()
            differences = differences[differences > 0]

            if len(differences) > 0:
                median_delta = differences.median()

                print("\nTimestamp analysis:")
                print(f"  Median interval: {median_delta:.3f} ms")
                print(f"  Mean interval: {differences.mean():.3f} ms")
                print(f"  Minimum interval: {differences.min():.3f} ms")
                print(f"  Maximum interval: {differences.max():.3f} ms")

                if median_delta > 0:
                    print(
                        "  Approximate frequency:",
                        f"{1000 / median_delta:.2f} Hz"
                    )
            else:
                print("\nNo positive timestamp intervals found.")
        else:
            print("\nNot enough valid timestamps.")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


def find_android_csv_files():
    print_section("2. SEARCHING FOR ANDROID CSV FILES")

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return []

    csv_files = []

    for file_path in DATA_DIR.rglob("*.csv"):
        relative_path = file_path.relative_to(DATA_DIR).as_posix()

        # Exclude Silent Impact and generated experiment data
        if "silent_impact" in relative_path.lower():
            continue

        csv_files.append(file_path)

    return csv_files


def print_schema_comparison():
    print_section("3. SENSOR SCHEMA COMPARISON")

    print("SILENT IMPACT:")
    print("  - 6 channels")
    print("  - 3 linear acceleration channels")
    print("  - 3 angular velocity channels")

    print("\nSMARTSTRIKE ANDROID:")
    print("  - 13 sensor values")
    print("  - 3 accelerometer channels")
    print("  - 3 linear acceleration channels")
    print("  - 3 gyroscope channels")
    print("  - 4 rotation vector values")

    print("\nPotentially comparable groups:")
    print("  Silent Impact linear acceleration <-> Android linear_x/y/z")
    print("  Silent Impact angular velocity <-> Android gyro_x/y/z")

    print("\nImportant differences:")
    print("  - Silent Impact has 6 channels; Android has 13.")
    print("  - Silent Impact does not contain rotation vector channels.")
    print("  - Silent Impact does not contain raw accelerometer channels.")
    print("  - Sensor axes and physical placement must be verified.")
    print("  - Sampling frequency and units must be verified.")
    print("  - Timestamp alignment must be checked.")


def main():
    print("=" * 70)
    print("SMARTSTRIKE DATASET SCHEMA COMPARISON")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")

    inspect_silent_impact()

    print_section("4. SEARCHING FOR SMARTSTRIKE ANDROID CAPTURES")

    csv_files = find_android_csv_files()

    if not csv_files:
        print("No Android CSV files found under data/.")
        print(
            "\nIf you have an actual Android recording, place a copy in:"
        )
        print("  data/samples/android/")
        print("Then run this script again.")
    else:
        print(f"Found {len(csv_files)} CSV file(s):")

        for file_path in csv_files:
            print(f"\nFound: {file_path.relative_to(PROJECT_ROOT)}")
            inspect_android_csv(file_path)

    print_schema_comparison()

    print_section("5. ANALYSIS CHECKLIST")

    print("Before integrating the datasets, we need to verify:")
    print("  [ ] Sensor units")
    print("  [ ] Coordinate system and axis orientation")
    print("  [ ] Sampling frequency")
    print("  [ ] Sensor placement on the arm/wrist")
    print("  [ ] Timestamp alignment")
    print("  [ ] Gravity removal / linear acceleration handling")
    print("  [ ] Window size and stride compatibility")

    print("\nAnalysis completed. No files were modified.")


if __name__ == "__main__":
    main()
