"""
Smartwatch tennis dataset audit for SmartStrike.
Read-only: does not modify source data or existing SmartStrike models.

Run from the smartstike-ml project root:
    python audit_smartwatch_tennis.py

Creates data/public/smartwatch_tennis/audit_report.txt
"""
from pathlib import Path
from collections import Counter, defaultdict
from scipy.io import loadmat
import numpy as np
import sys

ROOT = Path("../../data/public/smartwatch_tennis/raw")
REPORT_PATH = Path("../../data/public/smartwatch_tennis/audit_report.txt")
KEY = "d_iner"


def inspect(path):
    r = {"path": str(path), "ok": False}
    try:
        mat = loadmat(path)
        if KEY not in mat:
            r["error"] = f"Missing {KEY}; variables: " + ", ".join(
                k for k in mat if not k.startswith("__"))
            return r
        x = np.asarray(mat[KEY])
        r["shape"] = x.shape
        r["dtype"] = str(x.dtype)
        if x.ndim != 2 or not np.issubdtype(x.dtype, np.number) or not x.size:
            r["error"] = "Expected a non-empty numeric 2D matrix"
            return r
        x = x.astype(float, copy=False)
        r["rows"], r["cols"] = x.shape
        r["nonfinite"] = int((~np.isfinite(x)).sum())
        r["unique_rows"] = int(np.unique(x, axis=0).shape[0])
        r["repeat_pairs"] = int(np.all(x[1:] == x[:-1], axis=1).sum())
        r["constant_cols"] = [i for i in range(x.shape[1])
                              if np.isfinite(x[:, i]).all() and np.ptp(x[:, i]) == 0]
        r["stats"] = []
        for i in range(x.shape[1]):
            v = x[:, i]
            v = v[np.isfinite(v)]
            r["stats"].append({
                "mean": float(v.mean()) if len(v) else None,
                "std": float(v.std()) if len(v) else None,
                "min": float(v.min()) if len(v) else None,
                "max": float(v.max()) if len(v) else None,
                "unique": int(np.unique(v).size),
            })
        r["ok"] = True
        return r
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
        return r


def main():
    lines = []

    def out(s=""):
        s = str(s)
        lines.append(s)
        print(s)

    out("=" * 72)
    out("SMARTSTRIKE - SMARTWATCH TENNIS DATASET AUDIT")
    out("Read-only audit; existing model files will not be changed.")
    out(f"Data directory: {ROOT}")
    out("=" * 72)

    if not ROOT.exists():
        out("ERROR: Dataset directory not found. Run from ML repository root.")
        return 2
    files = sorted(ROOT.rglob("*_inertial.mat"))
    out(f"MAT files found: {len(files)}")
    if not files:
        out("ERROR: No *_inertial.mat files found.")
        return 2

    results = [inspect(p) for p in files]
    good = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]

    out("\n[1] CLASS BALANCE (labels inferred from folder names)")
    folder_counts = Counter(p.parent.name for p in files)
    for folder, count in sorted(folder_counts.items()):
        out(f"  {folder}: {count}")
    out(f"  Total: {len(files)}")

    out("\n[2] FILE INTEGRITY")
    out(f"  Readable matrices: {len(good)}/{len(files)}")
    out(f"  Invalid/unreadable files: {len(bad)}")
    for r in bad:
        out(f"  ERROR {r['path']}: {r.get('error')}")
    out(f"  Matrix shapes: {dict(Counter(str(r.get('shape')) for r in good))}")
    out(f"  Data types: {dict(Counter(r.get('dtype') for r in good))}")
    out(f"  Files containing NaN/Inf: {sum(r.get('nonfinite', 0) > 0 for r in good)}")
    out(f"  Total NaN/Inf values: {sum(r.get('nonfinite', 0) for r in good)}")

    out("\n[3] DUPLICATES AND SIGNAL VARIATION")
    if good:
        out(f"  Total rows: {sum(r['rows'] for r in good)}")
        out(f"  Sum of unique rows per file: {sum(r['unique_rows'] for r in good)}")
        out(f"  Consecutive identical row pairs: {sum(r['repeat_pairs'] for r in good)}")
        out(f"  Files with consecutive identical rows: "
            f"{sum(r['repeat_pairs'] > 0 for r in good)}")
        constant = defaultdict(int)
        for r in good:
            for c in r["constant_cols"]:
                constant[c] += 1
        out(f"  Constant column indices (number of files): {dict(constant)}")
        out("  Repeated rows are flagged for investigation, not automatically errors.")

    out("\n[4] AGGREGATED COLUMN STATISTICS")
    col_counts = Counter(r["cols"] for r in good)
    out(f"  Column counts: {dict(col_counts)}")
    if len(col_counts) == 1 and good:
        ncols = next(iter(col_counts))
        for c in range(ncols):
            stats = [r["stats"][c] for r in good if r["stats"][c]["mean"] is not None]
            means = np.array([s["mean"] for s in stats])
            stds = np.array([s["std"] for s in stats])
            mins = np.array([s["min"] for s in stats])
            maxs = np.array([s["max"] for s in stats])
            uniques = np.array([s["unique"] for s in stats])
            out(f"  col {c}: mean of file means={means.mean():.4f}; "
                f"mean within-file std={stds.mean():.4f}; "
                f"min range=[{mins.min():.4f}, {mins.max():.4f}]; "
                f"max range=[{maxs.min():.4f}, {maxs.max():.4f}]; "
                f"unique values/file mean={uniques.mean():.1f}")
        out("  Column identities and units are unknown from the README provided.")

    out("\n[5] SMARTSTRIKE COMPATIBILITY")
    out("  Labels inferred from folders: FOREHAND, SERVE, BACKHAND.")
    out("  No explicit NO_SHOT, BACKHAND_SLICE, or shot-quality labels were provided.")
    out("  Existing SmartStrike pipeline expects 13 named channels (accel/linear/gyro/"
        "rotation) and its own feature extraction.")
    out("  The .mat files provide a 6-column d_iner matrix, so this dataset is NOT "
        "plug-and-play compatible.")
    out("  Do not train the current scripts directly on these files.")
    out("  A separate adapter requires verified channel order, units, sample rate, "
        "and segmentation details.")

    out("\n[6] DATASET LIMITATIONS")
    out(f"  Total labeled files: {len(files)}; only three stroke classes.")
    out("  39 rows/file is a short sequence; duration is unknown without sample rate.")
    out("  Filenames include s# and t#, but their meanings are not confirmed by the README.")
    out("  Repeated rows may reduce temporal information; their cause is unknown.")
    out("  For model evaluation, group related trials/sessions/players when metadata allows "
        "to avoid data leakage.")

    out("\n[7] VERDICT")
    out("  EXPLORATION: suitable for a small, preliminary 3-class experiment.")
    out("  SMARTSTRIKE TRAINING/DEPLOYMENT: not ready yet.")
    out("  Blockers: undocumented channels/units/rate, short clips, repeated rows, "
        "only three classes, and incompatible input schema.")
    out("  Next: find original paper/documentation or ask dataset author for channel order, "
        "units, sample rate, and repeated-row explanation.")
    out("  Keep current .pkl and ONNX models unchanged.")
    out("\nAudit complete. Source .mat files and models were not modified.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
