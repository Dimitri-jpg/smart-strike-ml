import pickle
import json
from pathlib import Path
from collections import Counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "public" / "silent_impact" / "raw"


def load_pkl(filename):
    with open(DATA_DIR / filename, "rb") as file:
        return pickle.load(file)


def inspect_metadata(data, name):
    metadata = data[:, 0, :]

    class_ids = metadata[:, 0].astype(int)
    participant_ids = metadata[:, 1].astype(int)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Class IDs:", sorted(np.unique(class_ids)))
    print("Participant IDs:", sorted(np.unique(participant_ids)))

    print("\nSamples per class:")
    print(Counter(class_ids))

    print("\nSamples per participant:")
    print(Counter(participant_ids))

    print("\nSamples per class and participant:")

    for class_id in sorted(np.unique(class_ids)):
        for participant_id in sorted(np.unique(participant_ids)):
            count = np.sum(
                (class_ids == class_id) &
                (participant_ids == participant_id)
            )

            print(
                f"Class {class_id}, "
                f"Participant {participant_id}: {count}"
            )

    print("\nSensor data shape:", data[:, 1:, :].shape)


def inspect_participants():
    path = DATA_DIR / "Participant_Information.json"

    print("\n" + "=" * 60)
    print("PARTICIPANT INFORMATION")
    print("=" * 60)

    if not path.exists():
        print("Participant_Information.json not found.")
        return

    with open(path, "r", encoding="utf-8") as file:
        info = json.load(file)

    print(json.dumps(info, indent=2, ensure_ascii=False)[:5000])


def main():
    dominant = load_pkl("Classification_Dominant.pkl")
    passive = load_pkl("Classification_Passive.pkl")

    inspect_metadata(dominant, "DOMINANT")
    inspect_metadata(passive, "PASSIVE")

    inspect_participants()

    print("\nDominant and Passive metadata identical:",
          np.array_equal(dominant[:, 0, :], passive[:, 0, :]))


if __name__ == "__main__":
    main()