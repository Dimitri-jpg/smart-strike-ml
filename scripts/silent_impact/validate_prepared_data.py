
from pathlib import Path
import numpy as np

data_dir = Path("data/processed/silent_impact")

train = np.load(data_dir / "train_windows.npz")
test = np.load(data_dir / "test_windows.npz")

for name, data in [("TRAIN", train), ("TEST", test)]:
    X = data["X"]
    y = data["y"]
    participants = data["participant_id"]
    arms = data["arm"]

    print(f"\n--- {name} ---")
    print("X shape:", X.shape)
    print("y distribution:", dict(zip(*np.unique(y, return_counts=True))))
    print("Participants:", np.unique(participants).tolist())
    print("Arms:", dict(zip(*np.unique(arms, return_counts=True))))
    print("All finite:", bool(np.isfinite(X).all()))

    print("Class distribution per participant:")
    for participant in np.unique(participants):
        mask = participants == participant
        print(
            f"  Participant {participant}: "
            f"{dict(zip(*np.unique(y[mask], return_counts=True)))}"
        )

train_ids = set(train["participant_id"].tolist())
test_ids = set(test["participant_id"].tolist())

print("\nParticipant overlap:", train_ids & test_ids)
assert not (train_ids & test_ids), "Data leakage: participant overlap!"
assert np.isfinite(train["X"]).all(), "Non-finite values in train!"
assert np.isfinite(test["X"]).all(), "Non-finite values in test!"
print("Basic validation passed.")
