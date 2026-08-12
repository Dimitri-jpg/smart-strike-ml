from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort


MODELS_DIR = Path("models")


def validate_classifier():

    sklearn_model = joblib.load(
        MODELS_DIR / "classifier.pkl"
    )

    session = ort.InferenceSession(
        str(MODELS_DIR / "classifier.onnx"),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(20, 83)
    ).astype(np.float32)

    sklearn_predictions = sklearn_model.predict(X)

    sklearn_probabilities = (
        sklearn_model.predict_proba(X)
    )

    outputs = session.run(
        None,
        {
            input_name: X
        }
    )

    print()
    print("CLASSIFIER")
    print("==========")

    for i, output in enumerate(outputs):
        print(
            f"ONNX output {i}:",
            np.asarray(output).shape
        )

    # ONNX classifier output conventions can vary,
    # so inspect them first.
    print("Sklearn predictions:")
    print(sklearn_predictions)

    print("ONNX outputs:")
    for output in outputs:
        print(np.asarray(output))


def validate_regressor():

    sklearn_model = joblib.load(
        MODELS_DIR / "quality_regressor.pkl"
    )

    session = ort.InferenceSession(
        str(MODELS_DIR / "quality_regressor.onnx"),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(20, 83)
    ).astype(np.float32)

    sklearn_predictions = sklearn_model.predict(X)

    outputs = session.run(
        None,
        {
            input_name: X
        }
    )

    onnx_predictions = np.asarray(
        outputs[0]
    ).reshape(-1)

    print()
    print("REGRESSOR")
    print("=========")

    print(
        "Max absolute difference:",
        np.max(
            np.abs(
                sklearn_predictions -
                onnx_predictions
            )
        )
    )

    print(
        "Sklearn:",
        sklearn_predictions[:5]
    )

    print(
        "ONNX:",
        onnx_predictions[:5]
    )

    np.testing.assert_allclose(
        sklearn_predictions,
        onnx_predictions,
        rtol=1e-4,
        atol=1e-5
    )

    print("REGRESSOR VALIDATION PASSED")


def validate_shot_detector():

    sklearn_model = joblib.load(
        MODELS_DIR / "shot_detector.pkl"
    )

    session = ort.InferenceSession(
        str(MODELS_DIR / "shot_detector.onnx"),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(20, 83)
    ).astype(np.float32)

    sklearn_predictions = sklearn_model.predict(X)

    outputs = session.run(
        None,
        {
            input_name: X
        }
    )

    print()
    print("SHOT DETECTOR")
    print("=============")

    print("Sklearn:")
    print(sklearn_predictions)

    print("ONNX outputs:")

    for output in outputs:
        print(np.asarray(output))


if __name__ == "__main__":

    validate_classifier()
    validate_regressor()
    validate_shot_detector()