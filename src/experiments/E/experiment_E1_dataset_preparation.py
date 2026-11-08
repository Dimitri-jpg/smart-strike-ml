import csv
import json
import re
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split

ROOT = Path.cwd()

DATA_DIRS = [
    ROOT / "data/raw/smartstrike/SensorReadings/DatasetV4",
    ROOT / "data/raw/smartstrike/SensorReadings/NoneDatasetV4"
]

OUTPUT_DIR = ROOT / "data/processed/experiment_e"
OUTPUT_FILE = OUTPUT_DIR / "e1_split_manifest.json"

FILENAME_RE = re.compile(
    r"^session_(?P<label>.+?)_(?P<date>\d{8})_"
    r"(?P<time>\d{6})_(?P<shot_id>\d+)_score",
    re.I
)


def load_records():
    records = []

    for folder in DATA_DIRS:
        if not folder.exists():
            continue

        for path in sorted(folder.rglob("*.csv")):
            match = FILENAME_RE.match(path.name)

            if not match:
                print("WARNING: Filename not recognized:", path.name)
                continue

            try:
                with path.open(
                    "r",
                    newline="",
                    encoding="utf-8-sig"
                ) as f:
                    reader = csv.DictReader(f)
                    first = next(reader, None)

                if not first:
                    continue

                label = first.get("shot_type", "").strip().upper()

                if not label:
                    continue

                records.append({
                    "file": str(path.relative_to(ROOT)),
                    "label": label,
                    "date": match.group("date"),
                    "shot_id": match.group("shot_id")
                })

            except Exception as e:
                print("ERROR:", path.name, e)

    return records


def main():
    records = load_records()

    if not records:
        raise RuntimeError("No valid CSV recordings found.")

    labels = [r["label"] for r in records]

    print("E1 DATASET PREPARATION")
    print("Total recordings:", len(records))
    print("Class distribution:", dict(Counter(labels)))

    indices = list(range(len(records)))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    temp_labels = [labels[i] for i in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels
    )

    splits = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx
    }

    for name, idx in splits.items():
        distribution = Counter(records[i]["label"] for i in idx)

        print(
            f"{name.upper()}: {len(idx)} recordings",
            dict(distribution)
        )

    # Confirm that every recording appears in exactly one split.
    all_indices = train_idx + val_idx + test_idx

    if len(all_indices) != len(set(all_indices)):
        raise RuntimeError("Duplicate recording detected across splits.")

    if len(all_indices) != len(records):
        raise RuntimeError("Some recordings are missing from the splits.")

    manifest = {
        "experiment": "E1",
        "description": "Shared stratified recording-level splits",
        "random_state": 42,
        "split_ratio": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },
        "total_recordings": len(records),
        "class_distribution": dict(Counter(labels)),
        "records": records,
        "splits": splits,
        "limitations": [
            "Recording-level split, not session-level split.",
            "Date and class are confounded in the current dataset.",
            "Does not establish generalization to unseen sessions."
        ]
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\nE1 completed successfully.")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()