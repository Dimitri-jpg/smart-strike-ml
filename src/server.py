from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from src.features import extract_features

app = FastAPI()

model = joblib.load("models/forehand_backhand_model.pkl")


class SensorData(BaseModel):
    samples: list


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