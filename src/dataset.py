import glob
import os

import pandas as pd


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


def parse_filename(filename):

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    parts = name.split("_")

    score = None
    session_id = None

    for part in reversed(parts):

        if part.startswith("score"):
            score = int(
                part.replace("score", "")
            )
            break

    score_index = parts.index(
        next(
            p for p in parts
            if p.startswith("score")
        )
    )

    session_id = int(
        parts[score_index - 1]
    )

    shot_type = "_".join(
        parts[1:score_index - 3]
    )

    date = parts[score_index - 3]
    time = parts[score_index - 2]

    return {
        "shot_type": shot_type,
        "score": score,
        "session_id": session_id,
        "date": date,
        "time": time
    }


def load_dataset(folder):

    samples = []

    files = sorted(
        glob.glob(
            os.path.join(folder, "*.csv")
        )
    )

    print(f"Found {len(files)} files")

    for file in files:

        info = parse_filename(file)

        df = pd.read_csv(file)

        sample = {
            "file": file,
            "data": df[FEATURE_COLUMNS],
            "label": info["shot_type"],
            "score": info["score"],
            "session_id": info["session_id"]
        }

        samples.append(sample)

        print(
            os.path.basename(file),
            "->",
            info["shot_type"],
            f"(score={info['score']})"
        )

    return samples