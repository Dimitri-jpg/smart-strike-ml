import pandas as pd
import glob
import os


FEATURE_COLUMNS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "linear_x",
    "linear_y",
    "linear_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w"
]


def load_dataset(folder):

    samples = []

    files = sorted(
        [
            f for f in glob.glob(os.path.join(folder, "*.csv"))
            if "session_75_20260720_210457" not in os.path.basename(f)
        ]
    )

    print(f"Found {len(files)} files")

    for file in files:

        df = pd.read_csv(file)

        label = df["shot_type"].iloc[0]

        samples.append({
            "file": file,
            "data": df[FEATURE_COLUMNS],
            "label": label
        })

        print(
            os.path.basename(file),
            "->",
            label
        )

    return samples