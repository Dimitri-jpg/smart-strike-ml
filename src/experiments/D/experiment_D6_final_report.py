#!/usr/bin/env python3
"""D6 - Build a Markdown report from D1-D5 JSON outputs."""
from pathlib import Path
import json, statistics

ROOT = Path.cwd()
D_DIR = ROOT / "data/processed/experiment_d"
FILES = {
    "D1": D_DIR / "d1_data_leakage_audit.json",
    "D2": D_DIR / "d2_group_split_feasibility.json",
    "D3": D_DIR / "d3_existing_split_audit.json",
    "D4": D_DIR / "d4_session_date_analysis.json",
    "D5": D_DIR / "d5_recording_level_reevaluation.json",
}
OUT = D_DIR / "experiment_D_final_report.md"

def load(key):
    p = FILES[key]
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run the corresponding D script first.")
    return json.loads(p.read_text(encoding="utf-8"))

def pct(x):
    return "N/A" if x is None else f"{100*x:.2f}%"

def main():
    d1, d2, d3, d4, d5 = [load(k) for k in ("D1","D2","D3","D4","D5")]
    lines = [
        "# Experiment D – Robustness, Leakage and Validation Audit",
        "",
        "## 1. Objective",
        "Evaluate whether the earlier high SmartStrike classification scores may be affected by date/session confounding, and document what the current dataset can and cannot validate.",
        "",
        "## 2. D1 – Raw data audit",
        f"- Files read: {d1.get('files_read', d1.get('valid_records', 'N/A'))}",
        f"- Class distribution: `{d1.get('class_distribution', d1.get('class_counts', {}))}`",
        f"- Date by class: `{d1.get('date_by_class', {})}`",
        f"- Duplicate shot IDs: `{d1.get('duplicate_shot_ids', 'see D1 JSON')}`",
        f"- Exact duplicate hashes: `{d1.get('exact_duplicate_hashes', 'see D1 JSON')}`",
        "",
        "## 3. D2 – Group split feasibility",
        f"- Date holdout feasible: **{d2.get('date_holdout_feasible', 'N/A')}**",
        f"- Dates per class: `{d2.get('dates_per_class', {})}`",
        "",
        "## 4. D3 – Existing artifact audit",
        f"- B2 NPZ inspection: `{d3.get('b2_npz', 'See D3 JSON')}`",
        "- D3 output inventories are retained in `d3_existing_split_audit.json`.",
        "",
        "## 5. D4 – Date/label association",
        f"- Valid records: {d4.get('valid_records', 'N/A')}",
        f"- Date by class: `{d4.get('date_by_class', {})}`",
        f"- Mutual information: {d4.get('mutual_information_date_label_bits', 'N/A')} bits",
        f"- Date-only in-sample accuracy: {pct(d4.get('date_only_majority_classifier_training_accuracy'))}",
        "",
        "## 6. D5 – Recording-level diagnostic re-evaluation",
        "| Seed | Random Forest accuracy | Random Forest macro-F1 | Date-only accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for r in d5.get("results", []):
        rf, date = r["sensor_random_forest"], r["date_only_baseline"]
        lines.append(f"| {r['seed']} | {pct(rf['accuracy'])} | {pct(rf['macro_f1'])} | {pct(date['accuracy'])} |")
    lines += [
        "",
        "## 7. Interpretation and limitations",
        "- D5 uses recording-level random splits; it is not unseen-date/session validation.",
        "- The Random Forest uses summary statistics and is not directly comparable to the previous 1D CNN.",
        "- A high date-only score indicates that date carries label information in this dataset; it does not prove the sensor model uses date.",
        "- If each class occurs on only one date, a class-complete date-group holdout cannot be constructed.",
        "- New data should include every class across multiple dates/sessions, with participant/session identifiers and consistent recording procedures.",
        "",
        "## 8. Conclusion",
        "The current dataset supports exploratory recording-level classification, but does not support a reliable claim of generalization to unseen dates or sessions. Earlier very high scores must therefore be reported as preliminary and interpreted with the dataset-collection limitation.",
        "",
        "## 9. Reproducibility",
        "Run D1, D2, D3, D4, D5, then D6 from the project root. The JSON outputs are preserved under `data/processed/experiment_d/`.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("D6 FINAL REPORT")
    print("Saved:", OUT)
    print("\n".join(lines[:20]))

if __name__ == "__main__":
    main()
