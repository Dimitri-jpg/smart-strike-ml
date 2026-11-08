"""
Experiment D3 - Audit saved split metadata and preprocessing artifacts.
Run from project root:
    python src/experiments/experiment_D3_existing_split_audit.py

This is an artifact audit. It reports what can be verified from saved JSON/NPZ
files and explicitly marks unavailable evidence instead of guessing.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
CANDIDATES = {
    "B2_metadata": [
        ROOT / "data/processed/silent_impact/classification_b2/classification_b2_metadata.json",
        ROOT / "data/processed/silent_impact/classification_b2/metadata.json",
    ],
    "B3_results": [
        ROOT / "data/processed/transfer_learning_b3",
    ],
    "B4_results": [
        ROOT / "data/processed/baseline_b4",
    ],
    "B5_results": [
        ROOT / "data/processed/robustness_b5/b5_robustness_results.json",
    ],
    "C_results": [
        ROOT / "data/processed/combined_c/C2_preprocessing.json",
        ROOT / "data/processed/combined_c/C3_source_split.json",
        ROOT / "data/processed/combined_c/C4_combined_split.json",
        ROOT / "data/processed/combined_c/C5_normalization.json",
        ROOT / "data/processed/combined_c/experiment_C_full_log.json",
    ],
}
OUT = ROOT / "data/processed/experiment_d/d3_existing_split_audit.json"

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_read_error": repr(e)}

def find_metadata():
    found = {}
    for key, paths in CANDIDATES.items():
        found[key] = []
        for p in paths:
            if p.is_file():
                found[key].append({"path": str(p.relative_to(ROOT)), "content": read_json(p)})
            elif p.is_dir():
                found[key].append({
                    "path": str(p.relative_to(ROOT)),
                    "files": [str(x.relative_to(ROOT)) for x in sorted(p.glob("*")) if x.is_file()]
                })
    return found

def recursive_find(obj, terms, path="$"):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if any(t in k.lower() for t in terms):
                hits.append({"path": p, "value": v})
            hits.extend(recursive_find(v, terms, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(recursive_find(v, terms, f"{path}[{i}]"))
    return hits

def main():
    artifacts = find_metadata()
    findings = []
    for category, entries in artifacts.items():
        for entry in entries:
            content = entry.get("content")
            if content is not None:
                findings.append({
                    "category": category,
                    "path": entry["path"],
                    "split_fields": recursive_find(content, ["train", "val", "test", "participant", "seed", "index"]),
                    "normalization_fields": recursive_find(content, ["mean", "std", "normaliz", "scaler"]),
                })

    # Inspect saved B2 NPZ if present, without assuming array names.
    npz_path = ROOT / "data/processed/silent_impact/classification_b2/classification_b2.npz"
    npz_info = None
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            npz_info = {
                "path": str(npz_path.relative_to(ROOT)),
                "arrays": {
                    k: {"shape": list(data[k].shape), "dtype": str(data[k].dtype)}
                    for k in data.files
                }
            }

    report = {
        "artifacts_found": artifacts,
        "split_and_normalization_field_findings": findings,
        "b2_npz": npz_info,
        "limitations": [
            "A JSON field named train/val/test is not proof of leakage-free splitting; "
            "the actual sample-to-group mapping must also be checked.",
            "A saved mean/std does not prove it was fitted only on training samples "
            "unless the source code or provenance confirms the fitting indices.",
            "If a model folder contains weights but no saved split indices, exact "
            "reconstruction of its test set may be impossible.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("D3 EXISTING SPLIT / PREPROCESSING AUDIT")
    for category, entries in artifacts.items():
        print(f"{category}: {len(entries)} artifact(s) found")
        for e in entries:
            print("  -", e["path"])
    print("B2 NPZ:", npz_info or "Not found")
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
