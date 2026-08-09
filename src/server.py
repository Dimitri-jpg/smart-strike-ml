from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import joblib
import pandas as pd
import time

from src.features import extract_features
from src.functions.visualization import create_visualization
from src.functions.detect_shots import detect_shots

app = FastAPI()

classifier = joblib.load(
    "models/classifier.pkl"
)

quality_regressor = joblib.load(
    "models/quality_regressor.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

shot_detector = joblib.load(
    "models/shot_detector.pkl"
)

print("=" * 50)
print("CLASSIFIER")
print("=" * 50)
print("features:", len(classifier.feature_names_in_))
print(classifier.feature_names_in_)
print("classes:", classifier.classes_)

print()
print("=" * 50)
print("QUALITY REGRESSOR")
print("=" * 50)
print("features:", len(quality_regressor.feature_names_in_))
print(quality_regressor.feature_names_in_)

print()
print("=" * 50)
print("SHOT DETECTOR")
print("=" * 50)
print("features:", len(shot_detector.feature_names_in_))
print(shot_detector.feature_names_in_)
print("classes:", shot_detector.classes_)

print()
print("=" * 50)
print("LABEL ENCODER")
print("=" * 50)
print(label_encoder.classes_)


class SensorData(BaseModel):
    samples: list


class SensorSample(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float

    linear_x: float
    linear_y: float
    linear_z: float

    gyro_x: float
    gyro_y: float
    gyro_z: float

    rot_x: float
    rot_y: float
    rot_z: float
    rot_w: float


class VisualizationRequest(BaseModel):
    samples: list[SensorSample]


@app.post("/predict")
def predict(data: SensorData):

    df = pd.DataFrame(data.samples)

    sample = {
        "data": df
    }

    features = extract_features(sample)

    classifier_input = pd.DataFrame(
        [features],
        columns=classifier.feature_names_in_
    )

    prediction = classifier.predict(
        classifier_input
    )[0]

    probabilities = classifier.predict_proba(
        classifier_input
    )[0]

    confidence = float(
        probabilities.max()
    )

    regressor_features = features.copy()

    regressor_features["shot_type"] = label_encoder.transform(
        [prediction]
    )[0]

    regressor_input = pd.DataFrame(
        [regressor_features],
        columns=quality_regressor.feature_names_in_
    )

    score = float(
        quality_regressor.predict(
            regressor_input
        )[0]
    )

    score = max(
        1.0,
        min(
            10.0,
            score
        )
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "score": round(score, 2)
    }


@app.post("/visualize")
def visualize(request: VisualizationRequest):

    df = pd.DataFrame(
        [
            sample.model_dump()
            for sample in request.samples
        ]
    )

    image = create_visualization(df)

    return StreamingResponse(
        image,
        media_type="image/png"
    )


@app.post("/detect-multiple")
def detect_multiple(data: SensorData):
    df = pd.DataFrame(data.samples)

    predictions = detect_shots(
        df,
        classifier,
        quality_regressor
    )

    if not predictions:
        return [
            {
                "start_sample": 0,
                "peak_sample": 0,
                "end_sample": 0,
                "prediction": "NO_SHOT",
                "confidence": 0.0,
                "score": 0.0
            }
        ]

    return predictions