"""
Experiment D2 - Check whether a leakage-resistant group split is feasible.
Run from project root:
    python src/experiments/experiment_D2_group_split_feasibility.py

This script does NOT train a model and does NOT silently create a random split.
It tests whether date/session groups contain enough examples of every class.
"""
from __future__ import annotations
import csv, json, re
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path.cwd()
DATA_ROOT = ROOT / "data/raw/smartstrike/SensorReadings"
FOLDERS = [DATA_ROOT / "DatasetV4", DATA_ROOT / "NoneDatasetV4"]
OUT = ROOT / "data/processed/experiment_d/d2_group_split_feasibility.json"
NAME_RE = re.compile(
    r"^session_(?P<label>.+?)_(?P<date>\d{8})_(?P<time>\d{6})_"
    r"(?P<shot_id>\d+)_score\d+(?:_score\d+)*\.csv$", re.I
)

def read_records():
    records = []
    for folder in FOLDERS:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.csv")):
            m = NAME_RE.match(path.name)
            if not m:
                continue
            try:
                with path.open(newline="", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                labels = Counter((r.get("shot_type") or "").strip().upper()
                                 for r in rows if (r.get("shot_type") or "").strip())
                if len(labels) != 1:
                    label = "MIXED_OR_UNKNOWN"
                else:
                    label = next(iter(labels))
                records.append({
                    "path": str(path.relative_to(ROOT)),
                    "label": label,
                    "date": m.group("date"),
                    "time": m.group("time"),
                    "shot_id": m.group("shot_id"),
                })
            except Exception as e:
                records.append({"path": str(path.relative_to(ROOT)),
                                "label": "READ_ERROR", "error": repr(e)})
    return records

def class_table(records, group_key):
    table = defaultdict(Counter)
    for r in records:
        table[r[group_key]][r["label"]] += 1
    return {g: dict(c) for g, c in sorted(table.items())}

def main():
    records = read_records()
    if not records:
        raise RuntimeError("No parseable SmartStrike CSV files found. Run D1 first.")
    date_table = class_table(records, "date")
    classes = sorted({r["label"] for r in records})
    dates_per_class = {
        c: sorted({r["date"] for r in records if r["label"] == c})
        for c in classes
    }
    classes_per_date = {
        d: sorted({r["label"] for r in records if r["date"] == d})
        for d in sorted({r["date"] for r in records})
    }

    # A strict date holdout is only class-complete if each class appears on >=2 dates.
    date_holdout_possible = all(len(v) >= 2 for v in dates_per_class.values())
    result = {
        "records": len(records),
        "class_counts": dict(Counter(r["label"] for r in records)),
        "date_class_table": date_table,
        "dates_per_class": dates_per_class,
        "classes_per_date": classes_per_date,
        "strict_date_group_holdout_feasible_for_all_classes": date_holdout_possible,
        "decision": (
            "Date-group holdout is feasible to attempt, but use a class-aware group "
            "selection and inspect exact counts."
            if date_holdout_possible else
            "A class-complete date-group holdout is NOT feasible with the current "
            "recordings. Do not claim unseen-date generalization. Collect each class "
            "on multiple dates/sessions before this evaluation."
        ),
        "important": (
            "The filename date is a conservative group identifier. If multiple "
            "recording sessions occurred on one date, date grouping may be stricter "
            "than session grouping. The current file naming does not itself prove "
            "session boundaries."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("D2 GROUP SPLIT FEASIBILITY")
    print("Records:", len(records))
    print("Class counts:", result["class_counts"])
    print("Dates per class:", dates_per_class)
    print("Classes per date:", classes_per_date)
    print("Date holdout feasible:", date_holdout_possible)
    print(result["decision"])
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
