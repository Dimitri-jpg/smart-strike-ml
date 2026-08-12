from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


MODELS_DIR = Path("models")


def convert_model(model_path, output_path):
    model = joblib.load(model_path)

    feature_names = model.feature_names_in_
    feature_count = len(feature_names)

    print()
    print("=" * 60)
    print(model_path)
    print("=" * 60)
    print("Model:", type(model))
    print("Features:", feature_count)
    print("Feature names:")
    for i, name in enumerate(feature_names):
        print(f"{i:3}: {name}")

    initial_type = [
        (
            "float_input",
            FloatTensorType([None, feature_count])
        )
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=17
    )

    output_path.write_bytes(
        onnx_model.SerializeToString()
    )

    print()
    print("Saved:", output_path)


def main():

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    convert_model(
        MODELS_DIR / "classifier.pkl",
        MODELS_DIR / "classifier.onnx"
    )

    convert_model(
        MODELS_DIR / "quality_regressor.pkl",
        MODELS_DIR / "quality_regressor.onnx"
    )

    convert_model(
        MODELS_DIR / "shot_detector.pkl",
        MODELS_DIR / "shot_detector.onnx"
    )


if __name__ == "__main__":
    main()