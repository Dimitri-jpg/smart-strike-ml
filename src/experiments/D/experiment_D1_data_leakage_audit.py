"""
Experiment D1 - SmartStrike data and leakage audit.
Run from the smartstike-ml project root:
    python src/experiments/experiment_D1_data_leakage_audit.py

Read-only: does not modify raw data or train models.
"""
from __future__ import annotations
import csv, hashlib, json, re, statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
DATA_ROOT = ROOT / "data/raw/smartstrike/SensorReadings"
FOLDERS = [DATA_ROOT / "DatasetV4", DATA_ROOT / "NoneDatasetV4"]
OUT = ROOT / "data/processed/experiment_d/d1_data_leakage_audit.json"

EXPECTED = [
    "timestamp", "shot_id", "shot_type", "accel_x", "accel_y", "accel_z",
    "linear_x", "linear_y", "linear_z", "gyro_x", "gyro_y", "gyro_z",
    "rot_x", "rot_y", "rot_z", "rot_w"
]
SENSOR_COLUMNS = EXPECTED[3:]
NAME_RE = re.compile(
    r"^session_(?P<label>.+?)_(?P<date>\d{8})_(?P<time>\d{6})_"
    r"(?P<shot_id>\d+)_score(?P<score>\d+)(?:_score\d+)*\.csv$", re.I
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def to_float(value):
    try:
        x = float(value)
        return x if x == x and abs(x) != float("inf") else None
    except (TypeError, ValueError):
        return None

def inspect_file(path: Path, folder_name: str):
    parsed = NAME_RE.match(path.name)
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)

        types = Counter((r.get("shot_type") or "").strip().upper()
                        for r in rows if (r.get("shot_type") or "").strip())
        shot_ids = sorted({(r.get("shot_id") or "").strip()
                           for r in rows if (r.get("shot_id") or "").strip()})
        timestamps = [to_float(r.get("timestamp")) for r in rows]
        timestamps = [x for x in timestamps if x is not None]
        diffs = [b-a for a, b in zip(timestamps, timestamps[1:])]
        numeric_stats = {}
        for col in SENSOR_COLUMNS:
            vals = [to_float(r.get(col)) for r in rows]
            vals = [x for x in vals if x is not None]
            numeric_stats[col] = {
                "valid_count": len(vals),
                "missing_or_invalid_count": len(rows) - len(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "mean": statistics.fmean(vals) if vals else None,
            }

        filename_info = parsed.groupdict() if parsed else None
        label_from_filename = filename_info["label"].upper() if filename_info else None
        label_from_column = list(types) if types else []
        duration = ((timestamps[-1] - timestamps[0]) / 1000.0
                    if len(timestamps) > 1 else None)
        return {
            "file": str(path.relative_to(ROOT)),
            "folder": folder_name,
            "filename_parse_ok": bool(parsed),
            "filename_metadata": filename_info,
            "label_from_filename": label_from_filename,
            "shot_types_in_csv": label_from_column,
            "label_agreement": (
                label_from_filename == label_from_column[0]
                if label_from_filename and len(label_from_column) == 1 else None
            ),
            "rows": len(rows),
            "missing_columns": [c for c in EXPECTED if c not in headers],
            "extra_columns": [c for c in headers if c not in EXPECTED],
            "shot_ids_in_csv": shot_ids,
            "one_shot_id_per_file": len(shot_ids) == 1,
            "duration_seconds": duration,
            "timestamp_non_increasing_count": sum(d <= 0 for d in diffs),
            "timestamp_gap_over_20ms_count": sum(d > 20 for d in diffs),
            "sensor_stats": numeric_stats,
            "sha256": sha256(path),
            "read_error": None,
        }
    except Exception as e:
        return {
            "file": str(path.relative_to(ROOT)), "folder": folder_name,
            "filename_parse_ok": bool(parsed),
            "filename_metadata": parsed.groupdict() if parsed else None,
            "read_error": repr(e),
        }

def main():
    records = []
    folder_summary = {}
    for folder in FOLDERS:
        paths = sorted(folder.rglob("*.csv")) if folder.exists() else []
        current = [inspect_file(p, folder.name) for p in paths]
        records.extend(current)
        folder_summary[folder.name] = {
            "path": str(folder), "exists": folder.exists(),
            "csv_count": len(paths),
        }

    valid = [r for r in records if not r.get("read_error")]
    class_counts, date_class, date_counts, id_counts = Counter(), Counter(), Counter(), Counter()
    hashes = defaultdict(list)
    for r in valid:
        label = r.get("label_from_csv")
        # label_from_csv is derived below from the actual CSV shot_type values.
        csv_labels = r.get("shot_types_in_csv", [])
        label = csv_labels[0] if len(csv_labels) == 1 else (
            "EMPTY" if not csv_labels else "MIXED"
        )
        r["label_from_csv"] = label
        class_counts[label] += 1
        meta = r.get("filename_metadata") or {}
        date = meta.get("date", "UNPARSED")
        date_class[(date, label)] += 1
        date_counts[date] += 1
        for sid in r.get("shot_ids_in_csv", []):
            id_counts[sid] += 1
        hashes[r["sha256"]].append(r["file"])

    date_table = [
        {"date": d, "class": c, "recording_count": n}
        for (d, c), n in sorted(date_class.items())
    ]
    class_date_counts = defaultdict(int)
    for (date, label), count in date_class.items():
        class_date_counts[label] += 1

    # Detect class/date confounding: each date has one class, or each class has one date.
    dates_per_class = defaultdict(set)
    classes_per_date = defaultdict(set)
    for (date, label), count in date_class.items():
        if date != "UNPARSED" and label not in ("EMPTY", "MIXED"):
            dates_per_class[label].add(date)
            classes_per_date[date].add(label)

    exact_duplicate_files = [v for v in hashes.values() if len(v) > 1]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "folders": folder_summary,
        "summary": {
            "files_found": len(records),
            "files_read": len(valid),
            "read_errors": len(records) - len(valid),
            "class_distribution_recordings": dict(class_counts),
            "date_by_class": date_table,
            "dates_per_class": {k: sorted(v) for k, v in dates_per_class.items()},
            "classes_per_date": {k: sorted(v) for k, v in classes_per_date.items()},
            "shot_ids_repeated_across_files": {k: v for k, v in id_counts.items() if v > 1},
            "exact_duplicate_file_hash_groups": exact_duplicate_files,
            "filename_parse_failures": [r["file"] for r in valid if not r["filename_parse_ok"]],
            "label_disagreements": [
                r["file"] for r in valid if r.get("label_agreement") is False
            ],
            "files_with_missing_columns": {
                r["file"]: r["missing_columns"] for r in valid if r["missing_columns"]
            },
            "class_date_confounding_warning": (
                "Potentially severe: at least one class occurs on only one date, "
                "or a date contains only one class."
                if any(len(v) <= 1 for v in dates_per_class.values())
                   or any(len(v) <= 1 for v in classes_per_date.values())
                else "No one-date-per-class pattern detected; inspect the full table."
            ),
        },
        "files": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("D1 DATA LEAKAGE AUDIT")
    print("Files found/read:", len(records), "/", len(valid))
    print("Class distribution:", dict(class_counts))
    print("Date by class:")
    for row in date_table:
        print(f"  {row['date']} | {row['class']}: {row['recording_count']}")
    print("Duplicate shot IDs:", report["summary"]["shot_ids_repeated_across_files"] or "None")
    print("Exact duplicate hashes:", len(exact_duplicate_files))
    print("WARNING:", report["summary"]["class_date_confounding_warning"])
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
