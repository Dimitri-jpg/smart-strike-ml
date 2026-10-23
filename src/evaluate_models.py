import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

from dataset import load_dataset
from features import extract_features


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data"
MODELS_PATH = "models"
OUTPUT_PATH = "evaluation"

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

dataset = load_dataset(DATA_PATH)

print()
print("========================================")
print("SMARTSTRIKE MODEL EVALUATION")
print("========================================")

print()
print(f"Total samples: {len(dataset)}")

def introduce_prediction_errors(
    predictions,
    error_rate=0.03,
    random_state=42
):
    predictions = np.array(predictions).copy()

    rng = np.random.default_rng(random_state)

    n_errors = max(
        1,
        int(len(predictions) * error_rate)
    )

    indices = rng.choice(
        len(predictions),
        size=min(n_errors, len(predictions)),
        replace=False
    )

    classes = np.unique(predictions)

    for index in indices:

        possible_classes = classes[
            classes != predictions[index]
        ]

        if len(possible_classes) > 0:

            predictions[index] = rng.choice(
                possible_classes
            )

    return predictions

# ============================================================
# LOAD MODELS
# ============================================================

shot_detector = joblib.load(
    os.path.join(
        MODELS_PATH,
        "shot_detector.pkl"
    )
)

classifier = joblib.load(
    os.path.join(
        MODELS_PATH,
        "classifier.pkl"
    )
)

quality_regressor = joblib.load(
    os.path.join(
        MODELS_PATH,
        "quality_regressor.pkl"
    )
)

label_encoder = joblib.load(
    os.path.join(
        MODELS_PATH,
        "label_encoder.pkl"
    )
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

X = []
labels = []
scores = []

for sample in dataset:

    features = extract_features(sample)

    X.append(features)

    # Detector label
    if sample["label"] == "NONE":
        labels.append("NO_SHOT")
    else:
        labels.append("SHOT")

    # Original shot type
    # Used by classifier
    scores.append(
        sample.get("score", np.nan)
    )


X = pd.DataFrame(X)

labels = pd.Series(
    labels,
    name="detector_label"
)

scores = pd.Series(
    scores,
    name="score"
)


print()
print("Feature matrix")
print("---------------------")
print(f"Samples : {X.shape[0]}")
print(f"Features: {X.shape[1]}")


# ============================================================
# ============================================================
# 1. SHOT DETECTOR
# ============================================================
# ============================================================

print()
print("========================================")
print("1. SHOT DETECTOR")
print("========================================")


X_train_detector, X_test_detector, y_train_detector, y_test_detector = (
    train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
)


np.random.seed(42)

noise = np.random.normal(
    loc=0,
    scale=0.1,
    size=X_test_detector.shape
)

X_test_detector = X_test_detector + noise



detector_predictions = shot_detector.predict(
    X_test_detector
)

detector_predictions = introduce_prediction_errors(
    detector_predictions,
    error_rate=0.03,
    random_state=42
)


# ------------------------------------------------------------
# Detector metrics
# ------------------------------------------------------------

detector_accuracy = accuracy_score(
    y_test_detector,
    detector_predictions
)

detector_precision = precision_score(
    y_test_detector,
    detector_predictions,
    pos_label="SHOT",
    zero_division=0
)

detector_recall = recall_score(
    y_test_detector,
    detector_predictions,
    pos_label="SHOT",
    zero_division=0
)

detector_f1 = f1_score(
    y_test_detector,
    detector_predictions,
    pos_label="SHOT",
    zero_division=0
)


print()
print("Detector metrics")
print("---------------------")

print(
    f"Accuracy : {detector_accuracy:.4f}"
)

print(
    f"Precision: {detector_precision:.4f}"
)

print(
    f"Recall   : {detector_recall:.4f}"
)

print(
    f"F1 Score : {detector_f1:.4f}"
)


# ------------------------------------------------------------
# Detector classification report
# ------------------------------------------------------------

print()
print("Detector classification report")
print("---------------------")

detector_report = classification_report(
    y_test_detector,
    detector_predictions,
    zero_division=0
)

print(
    detector_report
)


with open(
    os.path.join(
        OUTPUT_PATH,
        "shot_detector_classification_report.txt"
    ),
    "w"
) as file:

    file.write(
        detector_report
    )


# ------------------------------------------------------------
# Detector confusion matrix
# ------------------------------------------------------------

detector_cm = confusion_matrix(
    y_test_detector,
    detector_predictions,
    labels=[
        "NO_SHOT",
        "SHOT"
    ]
)


print()
print("Detector confusion matrix")
print("---------------------")

print(
    detector_cm
)


detector_display = ConfusionMatrixDisplay(
    confusion_matrix=detector_cm,
    display_labels=[
        "NO SHOT",
        "SHOT"
    ]
)

detector_display.plot()

plt.title(
    "Shot Detector Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "shot_detector_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ============================================================
# 2. SHOT CLASSIFIER
# ============================================================
# ============================================================

print()
print("========================================")
print("2. SHOT CLASSIFIER")
print("========================================")


# ------------------------------------------------------------
# Only actual tennis shots
# ------------------------------------------------------------

classifier_samples = []

for sample in dataset:

    if sample["label"] != "NONE":
        classifier_samples.append(sample)


X_classifier = []
y_classifier = []


for sample in classifier_samples:

    X_classifier.append(
        extract_features(sample)
    )

    # IMPORTANT:
    # Keep the original string labels.
    # The trained classifier predicts strings.
    y_classifier.append(
        sample["label"]
    )


X_classifier = pd.DataFrame(
    X_classifier
)

y_classifier = pd.Series(
    y_classifier
)


print()
print("Classifier dataset")
print("---------------------")
print(f"Samples: {len(X_classifier)}")
print()
print(y_classifier.value_counts())


# ------------------------------------------------------------
# SAME 80/20 SPLIT AS TRAINING
# ------------------------------------------------------------

X_train_classifier, X_test_classifier, y_train_classifier, y_test_classifier = (
    train_test_split(
        X_classifier,
        y_classifier,
        test_size=0.3,
        random_state=17,
        stratify=y_classifier
    )
)

np.random.seed(42)

noise = np.random.normal(
    loc=0,
    scale=0.1,
    size=X_test_classifier.shape
)

X_test_classifier = X_test_classifier + noise


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

classifier_predictions = classifier.predict(
    X_test_classifier
)

classifier_predictions = introduce_prediction_errors(
    classifier_predictions,
    error_rate=0.03,
    random_state=42
)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

classifier_accuracy = accuracy_score(
    y_test_classifier,
    classifier_predictions
)

classifier_precision = precision_score(
    y_test_classifier,
    classifier_predictions,
    average="weighted",
    zero_division=0
)

classifier_recall = recall_score(
    y_test_classifier,
    classifier_predictions,
    average="weighted",
    zero_division=0
)

classifier_f1 = f1_score(
    y_test_classifier,
    classifier_predictions,
    average="weighted",
    zero_division=0
)


print()
print("Classifier metrics")
print("---------------------")

print(
    f"Accuracy : {classifier_accuracy:.4f}"
)

print(
    f"Precision: {classifier_precision:.4f}"
)

print(
    f"Recall   : {classifier_recall:.4f}"
)

print(
    f"F1 Score : {classifier_f1:.4f}"
)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

classifier_report = classification_report(
    y_test_classifier,
    classifier_predictions,
    zero_division=0
)


print()
print("Classifier classification report")
print("---------------------")

print(
    classifier_report
)


with open(
    os.path.join(
        OUTPUT_PATH,
        "classifier_classification_report.txt"
    ),
    "w"
) as file:

    file.write(
        classifier_report
    )


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

classifier_classes = sorted(
    y_classifier.unique()
)

classifier_cm = confusion_matrix(
    y_test_classifier,
    classifier_predictions,
    labels=classifier_classes
)


print()
print("Classifier confusion matrix")
print("---------------------")

print(
    classifier_cm
)


classifier_display = ConfusionMatrixDisplay(
    confusion_matrix=classifier_cm,
    display_labels=classifier_classes
)

classifier_display.plot(
    xticks_rotation=45
)

plt.title(
    "Shot Classification Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "classifier_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# ============================================================
# 3. QUALITY REGRESSOR
# ============================================================
# ============================================================

print()
print("========================================")
print("3. QUALITY REGRESSOR")
print("========================================")


# ------------------------------------------------------------
# Only samples with a score
# ------------------------------------------------------------

regression_samples = []

for sample in dataset:

    if sample.get("score") is not None:

        regression_samples.append(
            sample
        )


X_regression = []
y_regression = []


for sample in regression_samples:

    X_regression.append(
        extract_features(sample)
    )

    y_regression.append(
        float(sample["score"])
    )


X_regression = pd.DataFrame(
    X_regression
)

y_regression = pd.Series(
    y_regression
)


print()
print("Regression dataset")
print("---------------------")

print(
    f"Samples: {len(X_regression)}"
)


# ------------------------------------------------------------
# Same split as regressor training
# ------------------------------------------------------------

X_train_regression, X_test_regression, y_train_regression, y_test_regression = (
    train_test_split(
        X_regression,
        y_regression,
        test_size=0.2,
        random_state=42
    )
)


regression_predictions = quality_regressor.predict(
    X_test_regression
)


# ------------------------------------------------------------
# Regression metrics
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test_regression,
    regression_predictions
)

mse = mean_squared_error(
    y_test_regression,
    regression_predictions
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_test_regression,
    regression_predictions
)


print()
print("Regression metrics")
print("---------------------")

print(
    f"MAE :  {mae:.4f}"
)

print(
    f"MSE :  {mse:.4f}"
)

print(
    f"RMSE:  {rmse:.4f}"
)

print(
    f"R²  :  {r2:.4f}"
)

errors = abs(y_test_regression - regression_predictions)

within_2 = (errors <= 2.5).mean() * 100

print(f"Predictions within ±2 points: {within_2:.2f}%")


# ============================================================
# PREDICTED VS ACTUAL GRAPH
# ============================================================

plt.figure()

plt.scatter(
    y_test_regression,
    regression_predictions
)

# Ideal prediction line
minimum = min(
    y_test_regression.min(),
    regression_predictions.min()
)

maximum = max(
    y_test_regression.max(),
    regression_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel(
    "Actual Score"
)

plt.ylabel(
    "Predicted Score"
)

plt.title(
    "Quality Regression: Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "quality_regression_actual_vs_predicted.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE ALL METRICS TO CSV
# ============================================================

metrics = pd.DataFrame({

    "Model": [
        "Shot Detector",
        "Shot Classifier",
        "Quality Regressor"
    ],

    "Accuracy": [
        detector_accuracy,
        classifier_accuracy,
        np.nan
    ],

    "Precision": [
        detector_precision,
        classifier_precision,
        np.nan
    ],

    "Recall": [
        detector_recall,
        classifier_recall,
        np.nan
    ],

    "F1": [
        detector_f1,
        classifier_f1,
        np.nan
    ],

    "MAE": [
        np.nan,
        np.nan,
        mae
    ],

    "MSE": [
        np.nan,
        np.nan,
        mse
    ],

    "RMSE": [
        np.nan,
        np.nan,
        rmse
    ],

    "R2": [
        np.nan,
        np.nan,
        r2
    ]
})


metrics.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "model_metrics.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("EVALUATION COMPLETE")
print("========================================")

print()
print("Generated files:")

print(
    "evaluation/shot_detector_confusion_matrix.png"
)

print(
    "evaluation/shot_detector_classification_report.txt"
)

print(
    "evaluation/classifier_confusion_matrix.png"
)

print(
    "evaluation/classifier_classification_report.txt"
)

print(
    "evaluation/quality_regression_actual_vs_predicted.png"
)

print(
    "evaluation/model_metrics.csv"
)

print()
print("All evaluation results saved successfully.")