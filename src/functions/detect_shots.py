import joblib
import numpy as np
import pandas as pd

from scipy.signal import find_peaks

from src.features import extract_features


WINDOW_SIZE = 700
HALF_WINDOW = WINDOW_SIZE // 2

MIN_PEAK_DISTANCE = 500
MIN_PROMINENCE = 2.0
MIN_MOTION = 3.0


shot_detector = joblib.load(
    "models/shot_detector.pkl"
)


def detect_shots(
    df: pd.DataFrame,
    classifier,
    quality_regressor
):

    accel = np.sqrt(
        df.accel_x.values ** 2 +
        df.accel_y.values ** 2 +
        df.accel_z.values ** 2
    )

    peaks, _ = find_peaks(
        accel,
        prominence=MIN_PROMINENCE,
        distance=MIN_PEAK_DISTANCE
    )

    if len(peaks) == 0:
        return []

    candidates = sorted(
        peaks,
        key=lambda p: accel[p],
        reverse=True
    )

    accepted_peaks = []

    for peak in candidates:

        too_close = False

        for accepted in accepted_peaks:

            if abs(peak - accepted) < MIN_PEAK_DISTANCE:
                too_close = True
                break

        if too_close:
            continue

        start = peak - HALF_WINDOW
        end = peak + HALF_WINDOW

        if start < 0 or end > len(df):
            continue

        window = df.iloc[
            start:end
        ].reset_index(drop=True)

        if len(window) != WINDOW_SIZE:
            continue

        window_motion = accel[start:end]

        if np.max(window_motion) < MIN_MOTION:
            continue

        features = extract_features({
            "data": window
        })

        detector_X = pd.DataFrame(
            [features],
            columns=shot_detector.feature_names_in_
        )

        is_shot = shot_detector.predict(
            detector_X
        )[0]

        if is_shot != "SHOT":
            print("NO SHOT")
            continue

        accepted_peaks.append(peak)

    accepted_peaks.sort()

    predictions = []

    for peak in accepted_peaks:

        start = peak - HALF_WINDOW
        end = peak + HALF_WINDOW

        window = df.iloc[
            start:end
        ].reset_index(drop=True)

        features = extract_features({
            "data": window
        })

        classifier_X = pd.DataFrame(
            [features],
            columns=classifier.feature_names_in_
        )

        prediction = classifier.predict(
            classifier_X
        )[0]

        probabilities = classifier.predict_proba(
            classifier_X
        )[0]

        score = quality_regressor.predict(
            classifier_X
        )[0]

        predictions.append({

            "start_sample": int(start),

            "peak_sample": int(peak),

            "end_sample": int(end),

            "prediction": prediction,

            "confidence": float(
                np.max(probabilities)
            ),

            "score": float(
                np.clip(score, 1, 10)
            )
        })


    return predictions