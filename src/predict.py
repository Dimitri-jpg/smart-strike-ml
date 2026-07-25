import pandas as pd
import joblib

from features import extract_features


model = joblib.load(
    "models/forehand_backhand_model.pkl"
)



def predict(file):

    df = pd.read_csv(file)


    sample = {
        "data": df[
            [
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
        ]
    }


    features = extract_features(sample)


    X = pd.DataFrame(
        [features]
    )



    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    print("Prediction:", prediction)
    print()

    for cls, prob in zip(model.classes_, probabilities):
        print(f"{cls}: {prob:.2%}")


    return prediction, probabilities



result = predict(
    "data/session_75_20260720_210457.csv"
)


