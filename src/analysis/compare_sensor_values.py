
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SILENT_IMPACT_DIR = (
    ROOT / "data" / "public" / "silent_impact" / "raw"
)

SMARTSTRIKE_DIRS = [
    ROOT / "data" / "raw" / "smartstrike"
    / "SensorReadings" / "DatasetV4",

    ROOT / "data" / "raw" / "smartstrike"
    / "SensorReadings" / "NoneDatasetV4",
]

CHANNELS = {
    "linear_x": 0,
    "linear_y": 1,
    "linear_z": 2,
    "gyro_x": 3,
    "gyro_y": 4,
    "gyro_z": 5,
}


def get_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    return {
        "count": len(values),
        "min": np.min(values),
        "p5": np.percentile(values, 5),
        "p25": np.percentile(values, 25),
        "median": np.percentile(values, 50),
        "mean": np.mean(values),
        "p75": np.percentile(values, 75),
        "p95": np.percentile(values, 95),
        "max": np.max(values),
        "std": np.std(values),
    }


def load_silent_impact(path):
    print(f"Loading {path.name}...")

    with path.open("rb") as f:
        data = pickle.load(f)

    lengths = data[:, 0, 0].astype(int)

    collected = {channel: [] for channel in CHANNELS}

    for i, length in enumerate(lengths):
        valid_data = data[i, 1:length + 1, :6]

        for channel, index in CHANNELS.items():
            collected[channel].append(valid_data[:, index])

    return {
        channel: np.concatenate(parts)
        for channel, parts in collected.items()
    }


def load_smartstrike():
    print("Loading SmartStrike CSV files...")

    collected = {channel: [] for channel in CHANNELS}

    total_files = 0

    for folder in SMARTSTRIKE_DIRS:
        if not folder.exists():
            print(f"WARNING: Folder not found: {folder}")
            continue

        for csv_path in folder.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)

                for channel in CHANNELS:
                    if channel not in df.columns:
                        raise ValueError(
                            f"Missing column {channel}"
                        )

                    values = pd.to_numeric(
                        df[channel], errors="coerce"
                    ).dropna().to_numpy()

                    collected[channel].append(values)

                total_files += 1

            except Exception as exc:
                print(f"WARNING: {csv_path.name}: {exc}")

    print(f"SmartStrike files loaded: {total_files}")

    return {
        channel: np.concatenate(parts)
        if parts else np.array([])
        for channel, parts in collected.items()
    }


def main():
    datasets = {}

    for filename in [
        "Detection_Passive.pkl",
        "Detection_Dominant.pkl",
    ]:
        path = SILENT_IMPACT_DIR / filename

        if path.exists():
            datasets[filename.replace(".pkl", "")] = (
                load_silent_impact(path)
            )
        else:
            print(f"WARNING: Missing file: {path}")

    datasets["SmartStrike"] = load_smartstrike()

    rows = []

    for dataset_name, channels in datasets.items():
        for channel, values in channels.items():
            if len(values) == 0:
                print(
                    f"WARNING: No values for "
                    f"{dataset_name} / {channel}"
                )
                continue

            stats = get_stats(values)

            rows.append({
                "dataset": dataset_name,
                "channel": channel,
                **stats,
            })

    result = pd.DataFrame(rows)

    output_dir = ROOT / "reports" / "dataset_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / "sensor_value_comparison.csv"
    )

    result.to_csv(output_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\nSENSOR VALUE COMPARISON")
    print(result.to_string(index=False))

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
