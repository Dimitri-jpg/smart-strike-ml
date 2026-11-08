#!/usr/bin/env python3
"""
D5 - Recording-level diagnostic re-evaluation.
Compares a date-only diagnostic baseline with a sensor-feature Random Forest on
repeated stratified recording-level splits. This is NOT unseen-session validation.
Requires: numpy, scikit-learn.
"""
from pathlib import Path
import csv, re, json
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder

ROOT = Path.cwd()
DATA_DIRS = [
    ROOT / "data/raw/smartstrike/SensorReadings/DatasetV4",
    ROOT / "data/raw/smartstrike/SensorReadings/NoneDatasetV4",
]
OUT = ROOT / "data/processed/experiment_d/d5_recording_level_reevaluation.json"
FILENAME_RE = re.compile(r"^session_(?P<label>.+?)_(?P<date>\d{8})_(?P<time>\d{6})_", re.I)
CHANNELS = ["accel_x","accel_y","accel_z","linear_x","linear_y","linear_z","gyro_x","gyro_y","gyro_z"]

def features_for(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("empty CSV")
    feats = []
    for col in CHANNELS:
        vals = np.array([float(r[col]) for r in rows if r.get(col) not in (None, "")], dtype=np.float64)
        if vals.size == 0:
            vals = np.array([0.0])
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        feats.extend([
            float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)),
            float(np.max(vals)), float(np.sqrt(np.mean(vals**2))),
            float(np.percentile(vals, 25)), float(np.percentile(vals, 50)),
            float(np.percentile(vals, 75))
        ])
    return feats

def load_data():
    X, y, dates, files = [], [], [], []
    errors = []
    for folder in DATA_DIRS:
        if not folder.exists(): continue
        for path in sorted(folder.rglob("*.csv")):
            m = FILENAME_RE.match(path.name)
            try:
                with path.open("r", newline="", encoding="utf-8-sig") as f:
                    first = next(csv.DictReader(f), None)
                if not first: raise ValueError("empty CSV")
                label = first.get("shot_type", "").strip().upper()
                date = m.group("date") if m else "UNKNOWN"
                if not label: raise ValueError("missing shot_type")
                X.append(features_for(path)); y.append(label); dates.append(date)
                files.append(str(path.relative_to(ROOT)))
            except Exception as e:
                errors.append({"file": str(path), "error": str(e)})
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(dates), files, errors

def metrics(y_true, y_pred, labels):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}

def main():
    X, y_text, dates, files, errors = load_data()
    if len(y_text) < 20:
        raise RuntimeError(f"Not enough usable recordings: {len(y_text)}")
    labels = sorted(set(y_text.tolist()))
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    seeds = [21, 42, 100, 123, 2026]
    results = []
    for seed in seeds:
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.15, random_state=seed, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced", random_state=seed, n_jobs=-1
        )
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        sensor_metrics = metrics(y[test_idx], pred, np.arange(len(labels)))

        # Date-only baseline: learn the majority label for each date from training records.
        date_map = {}
        for d in sorted(set(dates[train_idx])):
            ids = train_idx[dates[train_idx] == d]
            date_map[d] = Counter(y[ids]).most_common(1)[0][0]
        majority = Counter(y[train_idx]).most_common(1)[0][0]
        date_pred = np.array([date_map.get(d, majority) for d in dates[test_idx]])
        date_metrics = metrics(y[test_idx], date_pred, np.arange(len(labels)))
        results.append({
            "seed": seed, "train_count": len(train_idx), "test_count": len(test_idx),
            "sensor_random_forest": sensor_metrics,
            "date_only_baseline": date_metrics,
            "test_date_distribution": dict(Counter(dates[test_idx].tolist()))
        })

    report = {
        "dataset_records": len(y), "feature_count": int(X.shape[1]),
        "classes": labels, "class_counts": dict(Counter(y_text.tolist())),
        "date_counts": dict(Counter(dates.tolist())), "seeds": seeds,
        "results": results, "read_errors": errors,
        "important_interpretation": [
            "These are random recording-level splits, not date/session-grouped splits.",
            "Because class and date are confounded, these scores do not establish unseen-session generalization.",
            "Date-only baseline performance is a diagnostic comparison, not a deployable model.",
            "Random Forest uses summary statistics from each complete recording; it is not directly comparable to the earlier 1D CNN."
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("D5 RECORDING-LEVEL RE-EVALUATION")
    print("Records:", len(y), "| features:", X.shape[1], "| classes:", labels)
    for r in results:
        print(f"Seed {r['seed']}: RF accuracy={r['sensor_random_forest']['accuracy']:.4f}, "
              f"macro-F1={r['sensor_random_forest']['macro_f1']:.4f}; "
              f"date-only accuracy={r['date_only_baseline']['accuracy']:.4f}")
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
