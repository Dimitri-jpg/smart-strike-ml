
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRS = [
    ROOT / "data/raw/smartstrike/SensorReadings/DatasetV4",
    ROOT / "data/raw/smartstrike/SensorReadings/NoneDatasetV4",
]

OUTPUT_DIR = ROOT / "reports/experiment_a/dataset_profiling"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. PARSE FILENAME
# --------------------------------------------------

def parse_filename(file_path):
    filename = file_path.name

    # Expected format:
    # session_BACKHAND_20260803_100457_101_score5.csv

    pattern = (
        r"^session_"
        r"(?P<label>[A-Za-z_]+)_"
        r"(?P<date>\d{8})_"
        r"(?P<time>\d{6})_"
        r"(?P<recording_id>\d+)_"
        r"score(?P<score>\d+)"
    )

    match = re.search(pattern, filename, re.IGNORECASE)

    if not match:
        return None

    values = match.groupdict()

    timestamp = pd.to_datetime(
        values["date"] + values["time"],
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )

    if pd.isna(timestamp):
        return None

    return {
        "filename": filename,
        "label": values["label"].upper(),
        "date": values["date"],
        "time": values["time"],
        "timestamp": timestamp,
        "recording_id": int(values["recording_id"]),
        "score": int(values["score"]),
        "dataset_folder": file_path.parent.name,
    }


# --------------------------------------------------
# 3. LOAD FILE METADATA
# --------------------------------------------------

def load_metadata():
    records = []
    invalid_files = []

    for folder in DATASET_DIRS:
        if not folder.exists():
            raise FileNotFoundError(
                f"Folder not found: {folder}"
            )

        csv_files = sorted(folder.rglob("*.csv"))

        print(f"\n{folder.name}: {len(csv_files)} CSV files")

        for file_path in csv_files:
            metadata = parse_filename(file_path)

            if metadata is None:
                invalid_files.append(file_path.name)
                continue

            records.append(metadata)

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No valid filenames found.")

    print(f"\nValid filenames: {len(df)}")
    print(f"Invalid filenames: {len(invalid_files)}")

    if invalid_files:
        print("\nInvalid filenames:")
        for name in invalid_files:
            print(name)

    return df, invalid_files


# --------------------------------------------------
# 4. GENERAL DATASET SUMMARY
# --------------------------------------------------

def general_summary(df):
    print("\n" + "=" * 65)
    print("GENERAL DATASET SUMMARY")
    print("=" * 65)

    print(f"Total recordings: {len(df)}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Unique recording IDs: {df['recording_id'].nunique()}")

    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nDate range:")
    print("First:", df["timestamp"].min())
    print("Last: ", df["timestamp"].max())

    print("\nRecording ID range:")
    print("Minimum:", df["recording_id"].min())
    print("Maximum:", df["recording_id"].max())

    print("\nDuplicate recording IDs:")
    duplicates = df[
        df.duplicated("recording_id", keep=False)
    ].sort_values("recording_id")

    if duplicates.empty:
        print("No duplicate IDs.")
    else:
        print(duplicates[
            ["filename", "label", "recording_id"]
        ].to_string(index=False))


# --------------------------------------------------
# 5. CLASS DISTRIBUTION BY DATE
# --------------------------------------------------

def analyze_by_date(df):
    print("\n" + "=" * 65)
    print("CLASS DISTRIBUTION BY DATE")
    print("=" * 65)

    date_table = pd.crosstab(
        df["date"],
        df["label"],
    )

    print(date_table.to_string())

    date_table.to_csv(
        OUTPUT_DIR / "class_distribution_by_date.csv"
    )

    return date_table


# --------------------------------------------------
# 6. CLASS DISTRIBUTION BY HOUR
# --------------------------------------------------

def analyze_by_hour(df):
    print("\n" + "=" * 65)
    print("CLASS DISTRIBUTION BY HOUR")
    print("=" * 65)

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour

    hour_table = pd.crosstab(
        df["hour"],
        df["label"],
    )

    print(hour_table.to_string())

    hour_table.to_csv(
        OUTPUT_DIR / "class_distribution_by_hour.csv"
    )

    return hour_table


# --------------------------------------------------
# 7. RECORDING ID ANALYSIS
# --------------------------------------------------

def analyze_recording_ids(df):
    print("\n" + "=" * 65)
    print("RECORDING ID ANALYSIS")
    print("=" * 65)

    sorted_df = df.sort_values("recording_id").copy()

    sorted_df["id_difference"] = (
        sorted_df["recording_id"].diff()
    )

    print("\nID distribution by class:")

    id_summary = sorted_df.groupby("label")[
        "recording_id"
    ].agg([
        "count",
        "min",
        "max",
        "mean",
    ])

    print(id_summary.to_string())

    id_summary.to_csv(
        OUTPUT_DIR / "recording_id_summary.csv"
    )

    print("\nLargest ID gaps:")
    print(
        sorted_df.nlargest(
            15,
            "id_difference"
        )[
            ["filename", "label", "recording_id", "id_difference"]
        ].to_string(index=False)
    )

    sorted_df.to_csv(
        OUTPUT_DIR / "recording_id_analysis.csv",
        index=False,
    )


# --------------------------------------------------
# 8. TIME GAPS BETWEEN RECORDINGS
# --------------------------------------------------

def analyze_time_gaps(df):
    print("\n" + "=" * 65)
    print("TIME GAP ANALYSIS")
    print("=" * 65)

    sorted_df = df.sort_values("timestamp").copy()

    sorted_df["time_gap_seconds"] = (
        sorted_df["timestamp"].diff().dt.total_seconds()
    )

    print("\nTime gap statistics (seconds):")
    print(
        sorted_df["time_gap_seconds"].describe()
    )

    print("\nLargest time gaps:")
    print(
        sorted_df.nlargest(
            15,
            "time_gap_seconds"
        )[
            [
                "filename",
                "label",
                "timestamp",
                "time_gap_seconds",
            ]
        ].to_string(index=False)
    )

    sorted_df.to_csv(
        OUTPUT_DIR / "time_gap_analysis.csv",
        index=False,
    )


# --------------------------------------------------
# 9. TEMPORAL WINDOWS
# --------------------------------------------------

def analyze_temporal_windows(df):
    print("\n" + "=" * 65)
    print("TEMPORAL WINDOWS")
    print("=" * 65)

    sorted_df = df.sort_values("timestamp").copy()

    # Divide recordings into consecutive groups of 50.
    sorted_df["temporal_group"] = (
        sorted_df.reset_index(drop=True).index // 50
    )

    temporal_table = pd.crosstab(
        sorted_df["temporal_group"],
        sorted_df["label"],
    )

    print("\nClass distribution in chronological groups:")
    print(temporal_table.to_string())

    temporal_table.to_csv(
        OUTPUT_DIR / "temporal_group_distribution.csv"
    )

    group_details = sorted_df.groupby(
        "temporal_group"
    ).agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        recordings=("filename", "count"),
    )

    group_details.to_csv(
        OUTPUT_DIR / "temporal_group_details.csv"
    )

    print("\nTemporal group date ranges:")
    print(group_details.to_string())


# --------------------------------------------------
# 10. VISUALIZATIONS
# --------------------------------------------------

def create_visualizations(df, date_table, hour_table):
    # Class distribution by date
    ax = date_table.plot(
        kind="bar",
        figsize=(12, 6),
        title="SmartStrike Class Distribution by Date",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Number of recordings")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_distribution_by_date.png",
        dpi=150,
    )
    plt.close()

    # Class distribution by hour
    ax = hour_table.plot(
        kind="bar",
        figsize=(12, 6),
        title="SmartStrike Class Distribution by Hour",
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Number of recordings")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_distribution_by_hour.png",
        dpi=150,
    )
    plt.close()

    # Recordings over time
    fig, ax = plt.subplots(figsize=(14, 6))

    for label, group in df.groupby("label"):
        ax.scatter(
            group["timestamp"],
            [label] * len(group),
            label=label,
            alpha=0.7,
        )

    ax.set_title("SmartStrike Recordings Timeline")
    ax.set_xlabel("Recording timestamp")
    ax.set_ylabel("Class")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "recordings_timeline.png",
        dpi=150,
    )
    plt.close()


# --------------------------------------------------
# 11. MAIN
# --------------------------------------------------

def main():
    print("\nSMARTSTRIKE - DATASET PROFILING")
    print("=" * 65)

    df, invalid_files = load_metadata()

    df = df.sort_values("timestamp").reset_index(drop=True)

    general_summary(df)

    date_table = analyze_by_date(df)

    hour_table = analyze_by_hour(df)

    analyze_recording_ids(df)

    analyze_time_gaps(df)

    analyze_temporal_windows(df)

    create_visualizations(
        df,
        date_table,
        hour_table,
    )

    # Save complete metadata
    df.to_csv(
        OUTPUT_DIR / "complete_dataset_metadata.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("DATASET PROFILING COMPLETED")
    print("=" * 65)

    print(f"Valid recordings: {len(df)}")
    print(f"Invalid filenames: {len(invalid_files)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
