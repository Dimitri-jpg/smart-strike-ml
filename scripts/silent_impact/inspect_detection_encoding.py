
import pickle
from pathlib import Path

import numpy as np


ROOT = Path("../../data/public/silent_impact/raw")

FILES = [
    "Detection_Dominant.pkl",
    "Detection_Passive.pkl",
]


def inspect_file(filename: str) -> None:
    path = ROOT / filename

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    # Read the dataset without modifying it.
    with path.open("rb") as file:
        data = pickle.load(file)

    data = np.asarray(data)

    print("Shape:", data.shape)
    print("Data type:", data.dtype)

    if data.ndim != 3 or data.shape[2] != 7:
        print("ERROR: Unexpected data format.")
        return

    total_frames = 0
    total_no_shot = 0
    total_shot = 0
    all_original_labels = set()
    invalid_sequences = []

    for sequence_index in range(data.shape[0]):
        sequence = data[sequence_index]

        # Metadata in the first row.
        length_value = sequence[0, 0]
        participant_value = sequence[0, 1]

        if not np.isfinite(length_value):
            invalid_sequences.append(sequence_index)
            print(f"Sequence {sequence_index}: invalid length.")
            continue

        length = int(length_value)
        participant = int(participant_value)

        if length <= 0 or length >= sequence.shape[0]:
            invalid_sequences.append(sequence_index)
            print(
                f"Sequence {sequence_index}: "
                f"invalid declared length {length}."
            )
            continue

        # Use only actual frames, excluding padded rows.
        frames = sequence[1:length + 1, :]

        sensor_values = frames[:, :6]
        labels_raw = frames[:, 6]

        if not np.isfinite(sensor_values).all():
            print(
                f"WARNING: Non-finite sensor values "
                f"in sequence {sequence_index}."
            )

        if not np.isfinite(labels_raw).all():
            print(
                f"WARNING: Non-finite labels "
                f"in sequence {sequence_index}."
            )
            continue

        labels = labels_raw.astype(np.int64)

        # Verify that labels are integer-valued.
        if not np.allclose(labels_raw, labels):
            print(
                f"WARNING: Non-integer labels "
                f"in sequence {sequence_index}."
            )
            continue

        values, counts = np.unique(labels, return_counts=True)
        label_counts = {
            int(value): int(count)
            for value, count in zip(values, counts)
        }

        all_original_labels.update(values.tolist())

        # According to the original Silent Impact detection
        # pipeline: 0 = no shot, values > 0 = shot.
        binary_labels = (labels > 0).astype(np.int64)

        no_shot_count = int(np.sum(binary_labels == 0))
        shot_count = int(np.sum(binary_labels == 1))

        total_frames += len(frames)
        total_no_shot += no_shot_count
        total_shot += shot_count

        print(f"\nSequence {sequence_index:02d}")
        print("  Participant:", participant)
        print("  Frames:", len(frames))
        print("  Original labels:", label_counts)
        print("  No-shot frames:", no_shot_count)
        print("  Shot frames:", shot_count)
        print(
            "  Sensor values finite:",
            bool(np.isfinite(sensor_values).all())
        )

        if sequence_index == 0:
            print("\n  First 5 actual frames:")
            print(frames[:5])

    print("\n" + "-" * 70)
    print("SUMMARY:", filename)
    print("-" * 70)
    print("Valid sequences:", data.shape[0] - len(invalid_sequences))
    print("Invalid sequences:", invalid_sequences)
    print("Total actual frames:", total_frames)
    print("Original labels:", sorted(all_original_labels))
    print("No-shot frames (0):", total_no_shot)
    print("Shot frames (label > 0):", total_shot)

    if total_frames > 0:
        print(f"No-shot percentage: {total_no_shot / total_frames * 100:.2f}%")
        print(f"Shot percentage: {total_shot / total_frames * 100:.2f}%")

    print("\nNOTE:")
    print("Binary mapping follows the original Silent Impact pipeline.")
    print("Original pickle files were not modified.")


def main() -> None:
    print("SILENT IMPACT - DETECTION ENCODING INSPECTION")
    print("Read-only inspection; no models or datasets will be changed.")

    for filename in FILES:
        inspect_file(filename)

    print("\nInspection finished.")


if __name__ == "__main__":
    main()
