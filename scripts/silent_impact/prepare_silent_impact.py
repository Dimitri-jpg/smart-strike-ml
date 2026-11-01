"""
Prepare the Silent Impact detection data for a first window-based baseline.

Expected raw files:
  data/public/silent_impact/raw/Detection_Dominant.pkl
  data/public/silent_impact/raw/Detection_Passive.pkl

Silent Impact layout (as audited):
  array shape: (sequence_count, padded_frames, 7)
  row 0, column 0: valid sequence length
  row 0, column 1: participant ID
  rows 1..length, columns 0..5: six IMU channels
  rows 1..length, column 6: frame label

This script deliberately:
- never modifies raw data or existing SmartStrike models;
- uses frame-count windows, not seconds, because the sample rate/timebase must
  be confirmed before interpreting window length as a duration;
- maps label 0 to NO_SHOT and any label > 0 to SHOT, matching the audited
  official binary detection mapping;
- labels a window SHOT when at least the configured fraction of its frames
  have a positive label (default 0.5);
- splits by participant so windows from one participant cannot appear in both
  train and test. Both arms use the same participant split.

Run from the repository root:
  python scripts/silent_impact/prepare_silent_impact.py

Optional:
  python scripts/silent_impact/prepare_silent_impact.py --window-size 128 --stride 64
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "public" / "silent_impact" / "raw"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "silent_impact"

RAW_FILES = {
    "dominant": RAW_DIR / "Detection_Dominant.pkl",
    "passive": RAW_DIR / "Detection_Passive.pkl",
}


def load_pickle(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Place the official Silent Impact detection pickle files in "
            "data/public/silent_impact/raw/."
        )
    with path.open("rb") as f:
        data: Any = pickle.load(f)
    array = np.asarray(data)
    if array.ndim != 3 or array.shape[-1] < 7:
        raise ValueError(
            f"Unexpected array shape for {path.name}: {array.shape}; "
            "expected (sequences, padded_frames, 7)."
        )
    return array


def extract_sequences(array: np.ndarray, arm: str) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for sequence_index in range(array.shape[0]):
        row0 = array[sequence_index, 0]
        length_value = row0[0]
        participant_value = row0[1]

        if not np.isfinite(length_value) or not np.isfinite(participant_value):
            raise ValueError(
                f"Invalid sequence metadata in {arm} sequence {sequence_index}."
            )

        length = int(length_value)
        participant_id = int(participant_value)
        if length <= 0 or length > array.shape[1] - 1:
            raise ValueError(
                f"Invalid length {length} in {arm} sequence {sequence_index}; "
                f"available frame capacity is {array.shape[1] - 1}."
            )

        frames = np.asarray(array[sequence_index, 1 : length + 1, :7], dtype=np.float32)
        if not np.isfinite(frames).all():
            raise ValueError(
                f"NaN/Inf found in {arm} sequence {sequence_index}, "
                f"participant {participant_id}."
            )

        sensor_values = frames[:, :6]
        frame_labels = frames[:, 6].astype(np.int64)
        sequences.append(
            {
                "x": sensor_values,
                "labels": frame_labels,
                "participant_id": participant_id,
                "arm": arm,
                "sequence_index": sequence_index,
            }
        )
    return sequences


def make_windows(
    sequences: list[dict[str, Any]],
    window_size: int,
    stride: int,
    shot_fraction_threshold: float,
) -> dict[str, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    shot_ratios: list[float] = []
    participants: list[int] = []
    arms: list[str] = []
    sequence_indices: list[int] = []
    start_frames: list[int] = []

    for sequence in sequences:
        x = sequence["x"]
        labels = sequence["labels"]
        if len(x) < window_size:
            continue

        for start in range(0, len(x) - window_size + 1, stride):
            end = start + window_size
            window_x = x[start:end]
            window_labels = labels[start:end]
            positive_ratio = float(np.mean(window_labels > 0))
            window_y = int(positive_ratio >= shot_fraction_threshold)

            xs.append(window_x)
            ys.append(window_y)
            shot_ratios.append(positive_ratio)
            participants.append(int(sequence["participant_id"]))
            arms.append(str(sequence["arm"]))
            sequence_indices.append(int(sequence["sequence_index"]))
            start_frames.append(start)

    if not xs:
        raise ValueError(
            "No windows were created. Check the input sequences and "
            "window-size/stride arguments."
        )

    return {
        "X": np.stack(xs).astype(np.float32),
        "y": np.asarray(ys, dtype=np.int8),
        "shot_ratio": np.asarray(shot_ratios, dtype=np.float32),
        "participant_id": np.asarray(participants, dtype=np.int16),
        "arm": np.asarray(arms, dtype="<U8"),
        "sequence_index": np.asarray(sequence_indices, dtype=np.int16),
        "start_frame": np.asarray(start_frames, dtype=np.int32),
    }


def split_by_participant(
    data: dict[str, np.ndarray], test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    participants = sorted(int(v) for v in np.unique(data["participant_id"]))
    if len(participants) < 2:
        raise ValueError("At least two distinct participants are required for a split.")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(participants, dtype=np.int16)
    rng.shuffle(shuffled)

    test_count = max(1, int(round(len(participants) * test_fraction)))
    test_count = min(test_count, len(participants) - 1)
    test_participants = sorted(int(v) for v in shuffled[:test_count])
    train_participants = sorted(int(v) for v in shuffled[test_count:])

    train_mask = np.isin(data["participant_id"], train_participants)
    test_mask = np.isin(data["participant_id"], test_participants)

    if np.any(train_mask & test_mask):
        raise AssertionError("Participant leakage detected.")
    if set(data["participant_id"][train_mask]) & set(data["participant_id"][test_mask]):
        raise AssertionError("Participant IDs overlap between train and test.")

    return train_mask, test_mask, train_participants, test_participants


def save_split(path: Path, data: dict[str, np.ndarray], mask: np.ndarray) -> None:
    np.savez_compressed(path, **{key: value[mask] for key, value in data.items()})


def distribution(y: np.ndarray) -> dict[str, int]:
    return {
        "NO_SHOT (0)": int(np.sum(y == 0)),
        "SHOT (1)": int(np.sum(y == 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=int, default=128, help="Frames per window (default: 128).")
    parser.add_argument("--stride", type=int, default=64, help="Frame step between windows (default: 64).")
    parser.add_argument(
        "--shot-fraction-threshold",
        type=float,
        default=0.5,
        help="A window is SHOT when this fraction of its frames have label > 0 (default: 0.5).",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of participants held out for test (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for participant split (default: 42).")
    args = parser.parse_args()

    if args.window_size <= 0 or args.stride <= 0:
        parser.error("--window-size and --stride must be positive integers.")
    if not 0.0 < args.shot_fraction_threshold <= 1.0:
        parser.error("--shot-fraction-threshold must be in (0, 1].")
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be in (0, 1).")

    sequences: list[dict[str, Any]] = []
    for arm, path in RAW_FILES.items():
        array = load_pickle(path)
        sequences.extend(extract_sequences(array, arm))

    data = make_windows(
        sequences=sequences,
        window_size=args.window_size,
        stride=args.stride,
        shot_fraction_threshold=args.shot_fraction_threshold,
    )
    train_mask, test_mask, train_participants, test_participants = split_by_participant(
        data, test_fraction=args.test_fraction, seed=args.seed
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_split(OUTPUT_DIR / "train_windows.npz", data, train_mask)
    save_split(OUTPUT_DIR / "test_windows.npz", data, test_mask)

    metadata = (
        "Silent Impact detection preprocessing\n"
        f"window_size_frames={args.window_size}\n"
        f"stride_frames={args.stride}\n"
        f"shot_fraction_threshold={args.shot_fraction_threshold}\n"
        "label_mapping: frame label 0 -> NO_SHOT; frame label > 0 -> SHOT\n"
        "window_label: SHOT if positive-frame fraction >= threshold; otherwise NO_SHOT\n"
        "split_method: participant-level; dominant/passive arms share the same split\n"
        f"train_participants={train_participants}\n"
        f"test_participants={test_participants}\n"
        f"total_windows={len(data['y'])}\n"
        f"train_windows={int(train_mask.sum())}\n"
        f"test_windows={int(test_mask.sum())}\n"
        f"all_distribution={distribution(data['y'])}\n"
        f"train_distribution={distribution(data['y'][train_mask])}\n"
        f"test_distribution={distribution(data['y'][test_mask])}\n"
        "NOTE: window size is in frames, not seconds. Confirm the dataset timebase "
        "before interpreting it as a duration.\n"
        "NOTE: dominant/passive sequences may be paired views of the same activity; "
        "participant-level splitting prevents participant leakage but does not make "
        "paired views statistically independent.\n"
    )
    (OUTPUT_DIR / "preprocessing_metadata.txt").write_text(metadata, encoding="utf-8")

    print("Silent Impact preprocessing completed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Window tensor shape: {data['X'].shape}  # (windows, frames, 6 channels)")
    print(f"All windows: {len(data['y'])} | distribution: {distribution(data['y'])}")
    print(f"Train windows: {int(train_mask.sum())} | distribution: {distribution(data['y'][train_mask])}")
    print(f"Test windows: {int(test_mask.sum())} | distribution: {distribution(data['y'][test_mask])}")
    print(f"Train participants: {train_participants}")
    print(f"Test participants: {test_participants}")
    print(f"Saved: {OUTPUT_DIR / 'train_windows.npz'}")
    print(f"Saved: {OUTPUT_DIR / 'test_windows.npz'}")
    print(f"Saved: {OUTPUT_DIR / 'preprocessing_metadata.txt'}")


if __name__ == "__main__":
    main()
