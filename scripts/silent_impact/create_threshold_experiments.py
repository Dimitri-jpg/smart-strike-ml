from pathlib import Path
import numpy as np

DATA_DIR = Path("data/processed/silent_impact")
OUTPUT_DIR = DATA_DIR / "threshold_experiments"

THRESHOLDS = [0.10, 0.25, 0.50]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for threshold in THRESHOLDS:
        print(f"\nPreparing threshold: {threshold}")

        for split in ("train", "test"):
            source = np.load(DATA_DIR / f"{split}_windows.npz")

            X = source["X"]
            shot_ratio = source["shot_ratio"]

            y = (shot_ratio >= threshold).astype(np.int8)

            output_path = (
                OUTPUT_DIR /
                f"{split}_threshold_{int(threshold * 100)}.npz"
            )

            np.savez_compressed(
                output_path,
                X=X,
                y=y,
                shot_ratio=shot_ratio,
                participant_id=source["participant_id"],
                arm=source["arm"],
                sequence_index=source["sequence_index"],
                start_frame=source["start_frame"],
            )

            unique, counts = np.unique(y, return_counts=True)
            distribution = dict(zip(unique.tolist(), counts.tolist()))

            print(
                f"{split.upper()}: "
                f"{len(y)} windows | "
                f"distribution={distribution}"
            )

    print("\nThreshold experiments created successfully.")


if __name__ == "__main__":
    main()