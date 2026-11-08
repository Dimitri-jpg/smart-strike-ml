#!/usr/bin/env python3
"""
D4 - Session/Date Confounding Analysis
Read-only analysis of the SmartStrike CSV recordings.
No model is trained and no raw files are modified.
"""
from pathlib import Path
import csv, re, json, math
from collections import Counter, defaultdict

ROOT = Path.cwd()
DATA_DIRS = [
    ROOT / "data/raw/smartstrike/SensorReadings/DatasetV4",
    ROOT / "data/raw/smartstrike/SensorReadings/NoneDatasetV4",
]
OUT = ROOT / "data/processed/experiment_d/d4_session_date_analysis.json"
FILENAME_RE = re.compile(
    r"^session_(?P<label>.+?)_(?P<date>\d{8})_(?P<time>\d{6})_"
    r"(?P<shot_id>\d+)_score(?P<score>\d+)(?:_score\d+)*\.csv$", re.I
)

def read_records():
    rows = []
    for folder in DATA_DIRS:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.csv")):
            m = FILENAME_RE.match(path.name)
            date = m.group("date") if m else "UNKNOWN"
            filename_label = m.group("label").upper() if m else "UNKNOWN"
            label = filename_label
            try:
                with path.open("r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    first = next(reader, None)
                    if first and first.get("shot_type"):
                        label = first["shot_type"].strip().upper()
            except Exception as e:
                rows.append({"file": str(path), "date": date, "label": label, "error": str(e)})
                continue
            rows.append({"file": str(path.relative_to(ROOT)), "date": date, "label": label})
    return rows

def entropy(counts):
    total = sum(counts.values())
    if not total: return 0.0
    return -sum((n/total) * math.log2(n/total) for n in counts.values() if n)

def main():
    records = read_records()
    valid = [r for r in records if r["date"] != "UNKNOWN" and r["label"] != "UNKNOWN" and "error" not in r]
    date_label = defaultdict(Counter)
    label_date = defaultdict(Counter)
    for r in valid:
        date_label[r["date"]][r["label"]] += 1
        label_date[r["label"]][r["date"]] += 1

    total = len(valid)
    # Mutual information I(Date;Label), computed from the empirical contingency table.
    date_totals = Counter(r["date"] for r in valid)
    label_totals = Counter(r["label"] for r in valid)
    mi = 0.0
    for d, labels in date_label.items():
        for label, n in labels.items():
            pxy = n / total
            mi += pxy * math.log2((n * total) / (date_totals[d] * label_totals[label]))

    # Date-only majority classifier, evaluated on the same data: descriptive leakage diagnostic,
    # not a valid independent test estimate.
    correct = sum(max(c.values()) for c in date_label.values()) if date_label else 0
    date_only_accuracy = correct / total if total else None
    report = {
        "records_found": len(records),
        "valid_records": total,
        "class_counts": dict(label_totals),
        "date_counts": dict(date_totals),
        "date_by_class": {d: dict(c) for d, c in date_label.items()},
        "dates_per_class": {label: sorted(c.keys()) for label, c in label_date.items()},
        "date_entropy_bits": entropy(date_totals),
        "label_entropy_bits": entropy(label_totals),
        "mutual_information_date_label_bits": mi,
        "date_only_majority_classifier_training_accuracy": date_only_accuracy,
        "interpretation": (
            "Date and label are strongly confounded if each date contains only one/few labels. "
            "The date-only accuracy is descriptive on these same recordings, not a generalization score."
        ),
        "limitations": [
            "Date may proxy session, player, environment, device placement, or collection procedure.",
            "This analysis cannot identify which of those factors caused the association.",
            "A date-only classifier evaluated on the same data is not an independent test."
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("D4 SESSION/DATE CONFOUNDING ANALYSIS")
    print("Valid records:", total)
    print("Date by class:", report["date_by_class"])
    print("Mutual information (bits):", round(mi, 4))
    print("Date-only in-sample accuracy:", round(date_only_accuracy, 4) if date_only_accuracy is not None else "N/A")
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
