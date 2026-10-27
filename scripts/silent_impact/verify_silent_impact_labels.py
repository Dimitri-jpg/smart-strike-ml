
from pathlib import Path
from collections import Counter
import pickle

import numpy as np

DATA_DIR = Path("../../data/public/silent_impact/raw")

CLASS_FILES = [
    "Classification_Dominant.pkl",
    "Classification_Passive.pkl",
]

DETECTION_FILES = [
    "Detection_Dominant.pkl",
    "Detection_Passive.pkl",
]

SHOT_NAMES = {
    0: "Forehand Stroke",
    1: "Backhand Stroke",
    2: "Forehand Volley",
    3: "Backhand Volley",
    4: "Smash",
    5: "Serve",
}


def load_array(filename):
    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    # Only load pickle files obtained from the official dataset source.
    with path.open("rb") as f:
        return pickle.load(f)


def verify_classification(filename):
    print("\n" + "=" * 65)
    print("CLASSIFICATION:", filename)

    data = load_array(filename)

    print("Shape:", data.shape)

    # Official format:
    # row 0, column 0 = shot class
    # row 0, column 1 = participant ID
    # rows 1..180 = sensor readings (180 frames, 6 channels)
    labels = data[:, 0, 0].astype(int)
    participants = data[:, 0, 1].astype(int)
    sensors = data[:, 1:, :]

    print("\nShot class counts:")
    for label, count in sorted(Counter(labels).items()):
        print(
            f"  {label} ({SHOT_NAMES.get(label, 'UNKNOWN')}): "
            f"{count}"
        )

    print("\nParticipant IDs:", sorted(set(participants)))
    print("Number of participants:", len(set(participants)))
    print("Sensor shape:", sensors.shape)
    print("Sensor values finite:", bool(np.isfinite(sensors).all()))
    print("Sensor min:", float(sensors.min()))
    print("Sensor max:", float(sensors.max()))

    unknown = sorted(set(labels) - set(SHOT_NAMES))
    print("Unknown class IDs:", unknown)

    if len(labels) == 6000 and not unknown:
        print("CLASSIFICATION FORMAT: plausible")
    else:
        print("CLASSIFICATION FORMAT: needs investigation")


def verify_detection(filename):
    print("\n" + "=" * 65)
    print("DETECTION:", filename)

    data = load_array(filename)

    print("Shape:", data.shape)

    all_labels = Counter()
    total_valid_frames = 0
    total_shot = 0
    total_no_shot = 0
    bad_lengths = []
    bad_labels = []
    participants = []

    # Official format:
    # row 0, column 0 = sequence length
    # row 0, column 1 = participant ID
    # subsequent rows, columns 0..5 = IMU channels
    # subsequent rows, column 6 = frame label
    for i in range(data.shape[0]):
        length_value = data[i, 0, 0]
        participant_value = data[i, 0, 1]

        length = int(length_value)
        participant = int(participant_value)
        participants.append(participant)

        if length < 0 or length > data.shape[1] - 1:
            bad_lengths.append((i, length))
            continue

        # Only inspect actual frames, not padding after the sequence.
        frames = data[i, 1:length + 1, :]
        sensors = frames[:, :6]
        labels = frames[:, 6]

        if not np.isfinite(frames).all():
            print(f"WARNING: non-finite values in sequence {i}")

        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            bad_labels.append((i, sorted(unique_labels)))

        label_counts = Counter(labels.astype(int).tolist())
        all_labels.update(label_counts)

        total_valid_frames += len(labels)
        total_no_shot += label_counts.get(0, 0)
        total_shot += label_counts.get(1, 0)

        print(
            f"Sequence {i:02d}: participant={participant}, "
            f"frames={length}, labels={dict(label_counts)}"
        )

    print("\nSUMMARY")
    print("Participants:", sorted(set(participants)))
    print("Number of sequences:", data.shape[0])
    print("Valid frames counted:", total_valid_frames)
    print("Label distribution:", dict(sorted(all_labels.items())))
    print("No-shot frames (0):", total_no_shot)
    print("Shot frames (1):", total_shot)
    print("Invalid sequence lengths:", bad_lengths)
    print("Sequences with unexpected labels:", bad_labels)

    if not bad_lengths and not bad_labels and total_valid_frames:
        print("DETECTION LABEL FORMAT: consistent with documented 0/1")
    else:
        print("DETECTION LABEL FORMAT: needs investigation")


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR.resolve()}"
        )

    print("SILENT IMPACT - LABEL VERIFICATION")
    print("Read-only: models and dataset files will not be changed.")

    for filename in CLASS_FILES:
        verify_classification(filename)

    for filename in DETECTION_FILES:
        verify_detection(filename)

    print("\nVerification finished.")


if __name__ == "__main__":
    main()
