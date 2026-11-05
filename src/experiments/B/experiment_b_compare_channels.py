
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SI_FILE = (
    PROJECT_ROOT
    / "data/public/silent_impact/raw/Classification_Dominant.pkl"
)

SMARTSTRIKE_DIR = (
    PROJECT_ROOT
    / "data/raw/smartstrike/SensorReadings/DatasetV4"
)

CHANNELS = [
    "linear_x",
    "linear_y",
    "linear_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def load_silent_impact():
    with open(SI_FILE, "rb") as file:
        data = pickle.load(file)

    # First row contains metadata.
    # Remaining 180 rows contain six sensor channels.
    return np.asarray(data[:, 1:, :], dtype=np.float64)


def load_smartstrike():
    files = list(SMARTSTRIKE_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {SMARTSTRIKE_DIR}"
        )

    recordings = []

    for file in files:
        df = pd.read_csv(file)

        missing = [
            channel for channel in CHANNELS
            if channel not in df.columns
        ]

        if missing:
            print(f"Skipping {file.name}: missing {missing}")
            continue

        values = df[CHANNELS].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        if not values.empty:
            recordings.append(values.to_numpy())

    if not recordings:
        raise ValueError("No valid SmartStrike recordings found.")

    return np.concatenate(recordings, axis=0)


def print_statistics(name, data):
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    print(f"Shape: {data.shape}")

    for index, channel in enumerate(CHANNELS):
        values = data[:, index]
        values = values[np.isfinite(values)]

        print(f"\n{channel}")
        print(f"  Min:    {np.min(values):.4f}")
        print(f"  Max:    {np.max(values):.4f}")
        print(f"  Mean:   {np.mean(values):.4f}")
        print(f"  Std:    {np.std(values):.4f}")
        print(f"  Median: {np.median(values):.4f}")
        print(
            "  P05/P95: "
            f"{np.percentile(values, 5):.4f} / "
            f"{np.percentile(values, 95):.4f}"
        )


def main():
    print("Loading Silent Impact...")
    si = load_silent_impact()

    # Combine all 180-frame sequences.
    si = si.reshape(-1, 6)

    print("Loading SmartStrike...")
    smartstrike = load_smartstrike()

    print_statistics("SILENT IMPACT - DOMINANT", si)
    print_statistics("SMARTSTRIKE - V4", smartstrike)

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
