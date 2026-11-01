
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SI_RAW = ROOT / "data" / "public" / "silent_impact" / "raw"

SMARTSTRIKE_DIRS = [
    ROOT / "data" / "raw" / "smartstrike"
    / "SensorReadings" / "DatasetV4",
    ROOT / "data" / "raw" / "smartstrike"
    / "SensorReadings" / "NoneDatasetV4",
]

OUTPUT_DIR = ROOT / "reports" / "dataset_analysis"

CHANNELS = [
    "linear_x", "linear_y", "linear_z",
    "gyro_x", "gyro_y", "gyro_z",
]


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {}

    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


# --------------------------------------------------
# 1. SILENT IMPACT METADATA AND STRUCTURE
# --------------------------------------------------

def inspect_silent_impact():
    print("\n=== SILENT IMPACT INSPECTION ===")

    results = {
        "files": {},
        "metadata": {},
        "sampling_rate_hz": None,
        "sensor_units": "Not confirmed from raw PKL",
        "axis_orientation": "Not confirmed from raw PKL",
    }

    for filename in [
        "Detection_Passive.pkl",
        "Detection_Dominant.pkl",
    ]:
        path = SI_RAW / filename

        if not path.exists():
            print(f"Missing: {path}")
            continue

        with path.open("rb") as f:
            data = pickle.load(f)

        lengths = data[:, 0, 0].astype(int)
        participant_ids = data[:, 0, 1].astype(int)

        results["files"][filename] = {
            "shape": list(data.shape),
            "sequence_count": int(len(data)),
            "min_sequence_length": int(lengths.min()),
            "median_sequence_length": float(np.median(lengths)),
            "max_sequence_length": int(lengths.max()),
            "participant_ids": sorted(
                np.unique(participant_ids).tolist()
            ),
            "sampling_rate_hz": None,
            "timestamps_available": False,
        }

        print(f"\n{filename}")
        print("Shape:", data.shape)
        print("Sequences:", len(data))
        print(
            "Sequence length min/median/max:",
            lengths.min(),
            np.median(lengths),
            lengths.max(),
        )
        print(
            "Participants:",
            sorted(np.unique(participant_ids).tolist()),
        )

    # Read participant metadata, without assuming it contains
    # sensor sampling or orientation information.
    info_path = SI_RAW / "Participant_Information.json"

    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)

        results["metadata"]["participant_information"] = info

        print("\nParticipant metadata loaded.")
        print("Top-level type:", type(info).__name__)

    return results


# --------------------------------------------------
# 2. SEARCH LOCAL DOCUMENTATION FOR SENSOR DETAILS
# --------------------------------------------------

def search_local_documentation():
    print("\n=== LOCAL DOCUMENTATION SEARCH ===")

    terms = [
        "sampling rate",
        "sampling frequency",
        "sample rate",
        "hz",
        "degree/s",
        "deg/s",
        "rad/s",
        "radian",
        "accelerometer",
        "gyroscope",
        "coordinate system",
        "axis",
        "orientation",
    ]

    extensions = {".md", ".txt", ".json", ".yaml", ".yml"}

    search_roots = [
        ROOT / "data" / "public" / "silent_impact",
        ROOT / "README.md",
    ]

    matches = []

    for search_root in search_roots:
        if search_root.is_file():
            files = [search_root]
        elif search_root.exists():
            files = [
                p for p in search_root.rglob("*")
                if p.is_file()
                and p.suffix.lower() in extensions
            ]
        else:
            continue

        for path in files:
            try:
                content = path.read_text(
                    encoding="utf-8", errors="ignore"
                )
            except OSError:
                continue

            for line_number, line in enumerate(
                content.splitlines(), start=1
            ):
                lower = line.lower()

                if any(term in lower for term in terms):
                    matches.append({
                        "file": str(path.relative_to(ROOT)),
                        "line": line_number,
                        "text": line.strip()[:300],
                    })

    for item in matches[:100]:
        print(
            f"{item['file']}:{item['line']} "
            f"{item['text']}"
        )

    if not matches:
        print("No relevant documentation lines found.")

    return matches


# --------------------------------------------------
# 3. SMARTSTRIKE TIMESTAMP ANALYSIS
# --------------------------------------------------

def analyze_smartstrike_timing():
    print("\n=== SMARTSTRIKE TIMING ANALYSIS ===")

    file_results = []
    all_intervals = []

    for folder in SMARTSTRIKE_DIRS:
        if not folder.exists():
            print("Missing folder:", folder)
            continue

        for csv_path in sorted(folder.glob("*.csv")):
            try:
                df = pd.read_csv(
                    csv_path,
                    usecols=["timestamp"],
                )

                timestamps = pd.to_numeric(
                    df["timestamp"], errors="coerce"
                ).dropna().to_numpy(dtype=float)

                timestamps = timestamps[
                    np.isfinite(timestamps)
                ]

                if len(timestamps) < 2:
                    continue

                # Preserve original row order. Do not deduplicate.
                intervals = np.diff(timestamps)

                positive_intervals = intervals[
                    intervals > 0
                ]

                if len(positive_intervals) == 0:
                    continue

                median_interval = float(
                    np.median(positive_intervals)
                )

                frequency = (
                    1000.0 / median_interval
                    if median_interval > 0 else None
                )

                duration = float(
                    (timestamps[-1] - timestamps[0]) / 1000
                )

                result = {
                    "file": csv_path.name,
                    "rows": int(len(timestamps)),
                    "duration_seconds": duration,
                    "median_interval_ms": median_interval,
                    "mean_interval_ms": float(
                        np.mean(positive_intervals)
                    ),
                    "p95_interval_ms": float(
                        np.percentile(positive_intervals, 95)
                    ),
                    "estimated_frequency_hz": frequency,
                    "repeated_timestamp_count": int(
                        np.sum(intervals == 0)
                    ),
                    "negative_interval_count": int(
                        np.sum(intervals < 0)
                    ),
                    "gaps_over_5ms": int(
                        np.sum(intervals > 5)
                    ),
                    "gaps_over_10ms": int(
                        np.sum(intervals > 10)
                    ),
                    "gaps_over_20ms": int(
                        np.sum(intervals > 20)
                    ),
                }

                file_results.append(result)
                all_intervals.extend(
                    positive_intervals.tolist()
                )

            except Exception as exc:
                print(f"WARNING {csv_path.name}: {exc}")

    if not file_results:
        return {
            "files": [],
            "summary": {},
        }

    interval_array = np.asarray(all_intervals)

    frequencies = [
        item["estimated_frequency_hz"]
        for item in file_results
        if item["estimated_frequency_hz"] is not None
    ]

    durations = [
        item["duration_seconds"]
        for item in file_results
    ]

    summary = {
        "file_count": len(file_results),
        "duration_seconds": percentile_summary(durations),
        "positive_interval_ms": percentile_summary(
            interval_array
        ),
        "per_file_median_frequency_hz": percentile_summary(
            frequencies
        ),
        "total_repeated_timestamps": sum(
            x["repeated_timestamp_count"]
            for x in file_results
        ),
        "total_negative_intervals": sum(
            x["negative_interval_count"]
            for x in file_results
        ),
    }

    print("Files analyzed:", len(file_results))
    print("\nDuration (seconds):", summary["duration_seconds"])
    print(
        "\nPositive timestamp intervals (ms):",
        summary["positive_interval_ms"],
    )
    print(
        "\nPer-file estimated frequency (Hz):",
        summary["per_file_median_frequency_hz"],
    )
    print(
        "\nRepeated timestamps:",
        summary["total_repeated_timestamps"],
    )
    print(
        "Negative intervals:",
        summary["total_negative_intervals"],
    )

    return {
        "files": file_results,
        "summary": summary,
    }


# --------------------------------------------------
# 4. FINAL COMPATIBILITY ASSESSMENT
# --------------------------------------------------

def create_assessment(si, smartstrike, documentation):
    timing = smartstrike["summary"]

    return {
        "common_channels": CHANNELS,
        "sensor_units": {
            "smartstrike": {
                "linear_acceleration": "m/s^2",
                "gyroscope": "rad/s",
                "source": "Android Sensor API types used in SensorManagerHelper",
            },
            "silent_impact": "Not independently confirmed from available local metadata",
            "status": "PARTIAL - SI units need documentary confirmation",
        },
        "axis_orientation": {
            "smartstrike": "Android device-relative sensor axes; physical watch orientation not confirmed",
            "silent_impact": "Not confirmed from available PKL structure",
            "status": "UNRESOLVED",
        },
        "sampling_rate": {
            "smartstrike": timing.get(
                "per_file_median_frequency_hz"
            ),
            "silent_impact": None,
            "status": (
                "PARTIAL - SmartStrike estimated from timestamps; "
                "Silent Impact rate unavailable in PKL"
            ),
        },
        "temporal_alignment": {
            "smartstrike_timestamps": "Unix epoch milliseconds",
            "silent_impact_timestamps": "Not present in inspected PKL",
            "status": "NOT DIRECTLY ALIGNABLE BY ABSOLUTE TIMESTAMP",
        },
        "label_compatibility": {
            "silent_impact": "Frame-level binary SHOT / NO_SHOT",
            "smartstrike": "CSV-level BACKHAND / BACKHAND_SLICE / FOREHAND / NONE",
            "status": "DIFFERENT TASK GRANULARITY",
        },
        "experiment_recommendations": {
            "A_smartstrike_only": "FEASIBLE",
            "B_silent_impact_pretraining_then_smartstrike": (
                "CONDITIONALLY FEASIBLE - confirm SI units, "
                "sensor placement, sampling rate and adapt labels/task"
            ),
            "C_combined_training": (
                "NOT RECOMMENDED YET - resolve domain and label differences first"
            ),
        },
        "documentation_matches": documentation[:100],
        "overall_status": (
            "PARTIAL ALIGNMENT - common channels exist, "
            "but units/orientation and SI sampling rate remain unverified"
        ),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    silent_impact = inspect_silent_impact()
    documentation = search_local_documentation()
    smartstrike_timing = analyze_smartstrike_timing()

    assessment = create_assessment(
        silent_impact,
        smartstrike_timing,
        documentation,
    )

    report = {
        "silent_impact": silent_impact,
        "smartstrike_timing": smartstrike_timing,
        "final_assessment": assessment,
    }

    output_path = OUTPUT_DIR / "final_alignment_report.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    timing_csv = OUTPUT_DIR / "smartstrike_timing_per_file.csv"

    pd.DataFrame(
        smartstrike_timing["files"]
    ).to_csv(timing_csv, index=False)

    print("\n=== FINAL COMPATIBILITY ASSESSMENT ===")
    print(json.dumps(
        assessment,
        indent=2,
        ensure_ascii=False,
    ))

    print("\nReports saved:")
    print(output_path)
    print(timing_csv)


if __name__ == "__main__":
    main()
