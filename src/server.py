from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from src.features import extract_features
from src.functions.visualization import create_visualization
from src.functions.detect_shots import detect_shots
from fastapi.responses import StreamingResponse

app = FastAPI()

model = joblib.load("models/forehand_backhand_model.pkl")


class SensorData(BaseModel):
    samples: list

class PredictSample(BaseModel):
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
    samples: list[PredictSample]



@app.post("/predict")
def predict(data: SensorData):

    df = pd.DataFrame(data.samples)

    sample = {
        "data": df
    }

    features = extract_features(sample)

    X = pd.DataFrame([features], columns=model.feature_names_in_)

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    return {
        "prediction": prediction,
        "confidence": float(max(probabilities))
    }


@app.post("/visualize")
def visualize(request: VisualizationRequest):

    df = pd.DataFrame(
        [sample.model_dump() for sample in request.samples]
    )

    image = create_visualization(df)

    return StreamingResponse(
        image,
        media_type="image/png"
    )


@app.post("/predict-multiple")
def predict_multiple(data: SensorData):

    df = pd.DataFrame(data.samples)

    return detect_shots(df, model)