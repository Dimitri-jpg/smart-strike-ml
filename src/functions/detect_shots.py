import numpy as np
import pandas as pd

from scipy.signal import find_peaks

from src.features import extract_features


WINDOW_SIZE = 700
HALF_WINDOW = WINDOW_SIZE // 2


def detect_shots(df: pd.DataFrame, model):

    accel = np.sqrt(
        df.accel_x**2 +
        df.accel_y**2 +
        df.accel_z**2
    )

    peaks, _ = find_peaks(
        accel,
        prominence=2.0,
        distance=WINDOW_SIZE // 2
    )

    predictions = []

    for peak in peaks:

        start = peak - HALF_WINDOW
        end = peak + HALF_WINDOW

        if start < 0 or end > len(df):
            continue

        window = df.iloc[start:end].reset_index(drop=True)

        features = extract_features({
            "data": window
        })

        X = pd.DataFrame(
            [features],
            columns=model.feature_names_in_
        )

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        predictions.append({

            "sample": int(peak),

            "prediction": prediction,

            "confidence": float(np.max(probabilities))
        })

    return predictions