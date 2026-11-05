
import numpy as np

path = "data/processed/silent_impact/classification_b2/classification_b2.npz"

data = np.load(path)

names = [
    "Forehand Stroke",
    "Backhand Stroke",
    "Forehand Volley",
    "Backhand Volley",
    "Smash",
    "Serve"
]

for split in ["train", "validation", "test"]:
    y = data["y_" + split]

    print(f"\n{split.upper()} - {len(y)} samples")

    for i, name in enumerate(names):
        count = np.sum(y == i)
        print(f"{i} - {name}: {count}")
