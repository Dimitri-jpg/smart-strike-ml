
from pathlib import Path
from collections import Counter
import json
import pickle
import sys

import numpy as np

# Run from the smartstike-ml project root.
DATA_DIR = Path("../../data/public/silent_impact/raw")
REPORT_PATH = Path("../../data/public/silent_impact/audit_report.txt")

EXPECTED_FILES = [
    "Classification_Dominant.pkl",
    "Classification_Passive.pkl",
    "Detection_Dominant.pkl",
    "Detection_Passive.pkl",
    "Participant_Information.json",
]

SHOT_LABELS = {
    0: "Forehand Stroke",
    1: "Backhand Stroke",
    2: "Forehand Volley",
    3: "Backhand Volley",
    4: "Smash",
    5: "Serve",
}


def describe(obj, path="root", depth=0, max_depth=3):
    """Describe nested objects without printing the full dataset."""
    indent = "  " * depth
    lines = []

    if isinstance(obj, dict):
        lines.append(f"{indent}{path}: dict, {len(obj)} keys")

        if depth < max_depth:
            for key, value in list(obj.items())[:15]:
                lines.extend(
                    describe(
                        value,
                        f"{path}[{key!r}]",
                        depth + 1,
                        max_depth,
                    )
                )

    elif isinstance(obj, (list, tuple)):
        lines.append(
            f"{indent}{path}: {type(obj).__name__}, "
            f"length={len(obj)}"
        )

        if obj and depth < max_depth:
            lines.extend(
                describe(
                    obj[0],
                    f"{path}[0]",
                    depth + 1,
                    max_depth,
                )
            )

    else:
        try:
            arr = np.asarray(obj)

            if arr.dtype != object and arr.ndim > 0:
                lines.append(
                    f"{indent}{path}: shape={arr.shape}, "
                    f"dtype={arr.dtype}"
                )
            else:
                lines.append(
                    f"{indent}{path}: {type(obj).__name__}"
                )
        except Exception:
            lines.append(f"{indent}{path}: {type(obj).__name__}")

    return lines


def find_arrays(obj, path="root", depth=0, max_depth=8):
    """Find numeric arrays inside dictionaries and sequences."""
    if depth > max_depth:
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from find_arrays(
                value,
                f"{path}[{key!r}]",
                depth + 1,
                max_depth,
            )

    elif isinstance(obj, (list, tuple)):
        # Limit inspection to the first 20 elements.
        for i, value in enumerate(obj[:20]):
            yield from find_arrays(
                value,
                f"{path}[{i}]",
                depth + 1,
                max_depth,
            )

    else:
        try:
            arr = np.asarray(obj)

            if (
                arr.ndim >= 1
                and arr.dtype != object
                and np.issubdtype(arr.dtype, np.number)
            ):
                yield path, arr
        except Exception:
            pass


def audit_pickle(path, log):
    log.append("")
    log.append("=" * 60)
    log.append(f"FILE: {path.name}")
    log.append(f"Size: {path.stat().st_size:,} bytes")

    # Only load pickle files obtained from the official source.
    try:
        with path.open("rb") as file:
            data = pickle.load(file)
    except Exception as exc:
        log.append(
            f"LOAD ERROR: {type(exc).__name__}: {exc}"
        )
        return

    log.append("\nOBJECT STRUCTURE:")
    log.extend(describe(data))

    arrays = list(find_arrays(data))
    log.append(f"\nNumeric arrays inspected: {len(arrays)}")

    shapes = Counter(
        (tuple(arr.shape), str(arr.dtype))
        for _, arr in arrays
    )

    log.append("\nArray shapes and data types:")
    for (shape, dtype), count in shapes.most_common(20):
        log.append(
            f"  shape={shape}, dtype={dtype}, count={count}"
        )

    total_values = 0
    nonfinite_values = 0

    for _, arr in arrays:
        total_values += arr.size
        nonfinite_values += int(
            (~np.isfinite(arr)).sum()
        )

    log.append(f"\nNumeric values inspected: {total_values:,}")
    log.append(f"NaN/Inf values: {nonfinite_values:,}")

    # Print a small sample of actual structure.
    for array_path, arr in arrays[:5]:
        log.append(f"\nSample: {array_path}")
        log.append(f"  shape={arr.shape}")

        if arr.size:
            log.append(f"  first values: {arr.reshape(-1)[:8].tolist()}")

    # Do not assume label positions until the structure is confirmed.
    log.append(
        "\nLabel interpretation: not inferred automatically. "
        "The structure and official metadata must be checked."
    )


def main():
    log = []

    def add(message=""):
        log.append(str(message))
        print(message)

    add("SMARTSTRIKE - SILENT IMPACT DATASET AUDIT")
    add("=" * 60)
    add(f"Project root: {Path.cwd()}")
    add(f"Dataset folder: {DATA_DIR.resolve()}")
    add("Read-only audit: no training or model modification.")
    add("")

    if not DATA_DIR.is_dir():
        add("ERROR: Dataset folder not found.")
        add(f"Create this folder: {DATA_DIR}")
        add("Put the downloaded files inside it.")
        return 1

    add("EXPECTED FILES:")
    for name in EXPECTED_FILES:
        path = DATA_DIR / name

        if path.is_file():
            add(f"  FOUND   {name} ({path.stat().st_size:,} bytes)")
        else:
            add(f"  MISSING {name}")

    pickle_files = sorted(DATA_DIR.glob("*.pkl"))
    json_files = sorted(DATA_DIR.glob("*.json"))

    if not pickle_files:
        add("\nNo .pkl files found. Check the extraction folder.")

    for path in pickle_files:
        audit_pickle(path, log)

    for path in json_files:
        add("")
        add("=" * 60)
        add(f"JSON FILE: {path.name}")

        try:
            with path.open("r", encoding="utf-8") as file:
                info = json.load(file)

            if isinstance(info, dict):
                add(f"Top-level object: dict, {len(info)} keys")
                add(f"Keys: {list(info.keys())[:40]}")

            elif isinstance(info, list):
                add(f"Top-level object: list, {len(info)} items")
                if info:
                    add(f"First item: {str(info[0])[:300]}")

            else:
                add(f"Top-level type: {type(info).__name__}")

        except Exception as exc:
            add(f"JSON ERROR: {type(exc).__name__}: {exc}")

    add("")
    add("=" * 60)
    add("SMARTSTRIKE COMPATIBILITY NOTES")
    add("- Silent Impact has six IMU channels.")
    add("- SmartStrike currently expects a different channel schema.")
    add("- Backhand slice is not a separate documented class.")
    add("- Detection labels and quality labels need separate inspection.")
    add("- This audit does not prove deployment compatibility.")
    add("- Do not run training scripts on these files yet.")

    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            "\n".join(log) + "\n",
            encoding="utf-8",
        )
        print(f"\nReport saved to: {REPORT_PATH.resolve()}")

    except Exception as exc:
        print(f"\nCould not save report: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
