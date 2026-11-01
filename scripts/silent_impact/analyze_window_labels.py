
from pathlib import Path
import numpy as np

data_dir = Path("data/processed/silent_impact")

for split in ("train", "test"):
    data = np.load(data_dir / f"{split}_windows.npz")
    ratios = data["shot_ratio"]

    print(f"\n--- {split.upper()} ---")
    print(f"Total windows: {len(ratios)}")

    for threshold in (0.10, 0.25, 0.50, 0.75):
        y = ratios >= threshold
        shots = int(y.sum())
        no_shots = int((~y).sum())

        print(
            f"Threshold {threshold:.2f}: "
            f"NO_SHOT={no_shots}, SHOT={shots}, "
            f"SHOT share={shots / len(y):.1%}"
        )

    print("\nPositive-frame ratio percentiles:")
    for p in (0, 25, 50, 75, 90, 95, 99, 100):
        print(f"  p{p}: {np.percentile(ratios, p):.4f}")
