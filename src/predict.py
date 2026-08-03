import joblib
import pandas as pd

from features import extract_features


classifier = joblib.load(
    "models/classifier.pkl"
)

quality_regressor = joblib.load(
    "models/quality_regressor.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)


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


def predict(file):

    df = pd.read_csv(file)

    sample = {
        "data": df[FEATURE_COLUMNS]
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

    print()
    print("Prediction:", prediction)
    print(f"Confidence: {confidence:.2%}")
    print()

    print("Class probabilities")
    print("-------------------")

    for cls, prob in zip(
        classifier.classes_,
        probabilities
    ):
        print(
            f"{cls:<18} {prob:.2%}"
        )

    print()
    print(
        f"Predicted quality: {score:.2f}/10"
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "score": score
    }


predict(
    "data/session_BACKHAND_20260803_100457_101_score5.csv"
)