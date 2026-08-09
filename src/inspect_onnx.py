import onnxruntime as ort


MODELS = [
    "models/classifier.onnx",
    "models/quality_regressor.onnx",
    "models/shot_detector.onnx",
]


for path in MODELS:

    print()
    print("=" * 60)
    print(path)
    print("=" * 60)

    session = ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"]
    )

    print("Inputs:")

    for input_info in session.get_inputs():
        print(
            " ",
            input_info.name,
            input_info.shape,
            input_info.type
        )

    print("Outputs:")

    for output_info in session.get_outputs():
        print(
            " ",
            output_info.name,
            output_info.shape,
            output_info.type
        )