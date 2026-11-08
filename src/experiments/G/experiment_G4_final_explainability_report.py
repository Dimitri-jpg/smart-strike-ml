
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

G2_PATH = (
    ROOT / "data/processed/experiment_g/g2/"
    "g2_feature_dependence_results.json"
)

G3_PATH = (
    ROOT / "data/processed/experiment_g/g3/"
    "g3_individual_explanations.json"
)

OUTPUT_DIR = ROOT / "data/processed/experiment_g/g4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

print("Loading G2 results...")
with open(G2_PATH, "r", encoding="utf-8") as file:
    g2 = json.load(file)

print("Loading G3 results...")
with open(G3_PATH, "r", encoding="utf-8") as file:
    g3 = json.load(file)


# --------------------------------------------------
# EXTRACT G2 INFORMATION
# --------------------------------------------------

top_features = g2["top_features"]

feature_names = [
    item["feature_name"] for item in top_features
]

feature_importance = [
    item["mean_absolute_shap"] for item in top_features
]


# --------------------------------------------------
# EXTRACT G3 INFORMATION
# --------------------------------------------------

samples = g3["explained_samples"]

misclassified = [
    sample for sample in samples
    if not sample["correct_prediction"]
]

correct_samples = [
    sample for sample in samples
    if sample["correct_prediction"]
]

sample_summaries = []

for sample in samples:
    sample_summaries.append({
        "sample_number": sample["sample_number"],
        "test_position": sample["test_position"],
        "true_class": sample["true_class"],
        "predicted_class": sample["predicted_class"],
        "correct_prediction": sample["correct_prediction"],
        "top_positive_features": sample["top_positive_features"],
        "top_negative_features": sample["top_negative_features"],
        "true_class_explanation": sample.get(
            "true_class_explanation"
        )
    })


# --------------------------------------------------
# GENERATE FEATURE IMPORTANCE CHART
# --------------------------------------------------

print("Generating feature importance chart...")

plt.figure(figsize=(10, 6))

indices = np.arange(len(feature_names))

plt.barh(indices, feature_importance)
plt.yticks(indices, feature_names)
plt.gca().invert_yaxis()

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Sensor Features")
plt.title("G4 - Top 5 SHAP Features")

plt.tight_layout()

chart_path = OUTPUT_DIR / "g4_top_features.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# GENERATE FINAL REPORT
# --------------------------------------------------

print("Generating final explainability report...")

accuracy = (
    g3["correct_predictions"] / g3["test_samples"] * 100
)

report = f"""# Experiment G – Final Explainability Report

## 1. Introduction

The objective of Experiment G was to improve the interpretability
of the optimized Random Forest model developed in Experiment F.

SHAP (SHapley Additive exPlanations) was used to investigate
global feature importance, feature dependence, and individual
model predictions.

The analysis used the SmartStrike test dataset.

## 2. G1 – Global SHAP Feature Importance

The G1 analysis identified the most influential sensor features
used by the Random Forest model.

The five most influential features were:

| Rank | Feature | Mean Absolute SHAP |
|---|---|---:|
"""

for feature in top_features:
    report += (
        f"| {feature['rank']} | {feature['feature_name']} | "
        f"{feature['mean_absolute_shap']:.6f} |\\n"
    )

report += """
The most influential feature was gyro_x_mean.
All five leading features were derived from gyroscope channels.

This indicates that rotational movement measurements
contributed substantially to the model's predictions.

## 3. G2 – Feature Dependence Analysis

SHAP dependence plots were generated for the five most
influential features:

"""

for feature in top_features:
    report += (
        f"- `{feature['feature_name']}` "
        f"(mean absolute SHAP: "
        f"{feature['mean_absolute_shap']:.6f})\\n"
    )

report += """
The dependence plots examine how feature values relate to
their SHAP contributions.

The analysis of gyro_x_mean showed that negative normalized
values generally had positive SHAP contributions for the
first model class, while positive values generally had
negative contributions.

The dependence plots in this implementation explain the
first class in the model output, rather than all classes
simultaneously. Therefore, this observation should not be
generalized to every class.

## 4. G3 – Individual Prediction Explanations

The model was evaluated on 75 test samples:

- Correct predictions: {g3['correct_predictions']}
- Incorrect predictions: {g3['incorrect_predictions']}
- Accuracy: {accuracy:.2f}%

Five individual samples were explained:
one misclassified sample and one correctly classified
sample from each class.

| Sample | True Class | Predicted Class | Correct |
|---:|---|---|---|
"""

for sample in samples:
    report += (
        f"| {sample['sample_number']} | "
        f"{sample['true_class']} | "
        f"{sample['predicted_class']} | "
        f"{'Yes' if sample['correct_prediction'] else 'No'} |\\n"
    )

report += """
### 4.1 Misclassified Sample Analysis

The misclassified sample had:

- True class: BACKHAND_SLICE
- Predicted class: NONE

For the predicted class NONE, the strongest positive
contributions included:

"""

if misclassified:
    sample = misclassified[0]

    for item in sample["top_positive_features"]:
        report += (
            f"- `{item['feature']}`: "
            f"{item['shap_value']:.6f}\\n"
        )

    true_explanation = sample.get("true_class_explanation")

    if true_explanation:
        report += """
For the true class BACKHAND_SLICE, the strongest positive
contributions included:

"""

        for item in true_explanation["top_positive_features"]:
            report += (
                f"- `{item['feature']}`: "
                f"{item['shap_value']:.6f}\\n"
            )

        report += """
The comparison showed that gyro_x_mean and gyro_x_q25
supported the predicted class NONE but opposed the true
class BACKHAND_SLICE. Other features, including gyro_x_range,
gyro_x_std and gyro_x_energy, supported BACKHAND_SLICE.

This demonstrates how different sensor features can
contribute in opposing directions to class predictions.
"""

report += """
## 5. Overall Findings

The explainability analysis produced the following findings:

1. Gyroscope-derived features dominated the top global
   SHAP feature rankings.
2. gyro_x_mean was the most influential feature in the
   global analysis.
3. Feature dependence analysis showed that the direction
   of a feature's SHAP contribution can vary with its value.
4. Individual explanations showed how several features
   jointly contribute to a prediction.
5. The misclassified BACKHAND_SLICE sample demonstrated
   that some feature contributions can favor an incorrect
   class.

## 6. Limitations

The explainability results must be interpreted with the
following limitations:

- SHAP explains the behavior of the trained model; it does
  not establish causal relationships between sensor
  measurements and tennis strokes.
- The individual analysis covers five selected test samples,
  not the entire test set.
- The G2 dependence plots explain the first model class.
- The dataset uses recording-level splits rather than
  session-level splits.
- Date/class confounding remains in the SmartStrike dataset.
- Resampling to 180 frames does not guarantee physical
  time alignment between recordings.

Consequently, high model performance and SHAP explanations
do not establish generalization to unseen recording sessions.

## 7. Conclusion

Experiment G provided an interpretable analysis of the
optimized Random Forest model through global SHAP importance,
feature dependence, and individual prediction explanations.

The results showed that gyroscope-derived features played
an important role in the model's predictions.

The individual analysis also helped explain a specific
misclassification between BACKHAND_SLICE and NONE.

Overall, SHAP provided useful insight into how the model
uses sensor information. However, additional recordings
across different dates and sessions are necessary to
evaluate generalization more reliably.
"""

report_path = OUTPUT_DIR / "experiment_G_final_explainability_report.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)


# --------------------------------------------------
# SAVE STRUCTURED JSON SUMMARY
# --------------------------------------------------

summary = {
    "experiment": "G4",
    "model": "Optimized Random Forest",
    "test_samples": g3["test_samples"],
    "correct_predictions": g3["correct_predictions"],
    "incorrect_predictions": g3["incorrect_predictions"],
    "accuracy_percent": round(accuracy, 2),
    "top_features": top_features,
    "explained_samples": sample_summaries,
    "limitations": [
        "SHAP explanations do not establish causality.",
        "G2 dependence plots explain the first model class.",
        "Individual explanations cover selected samples only.",
        "Recording-level splits are not session-level splits.",
        "Date/class confounding remains.",
        "Resampling does not guarantee physical time alignment."
    ]
}

json_path = OUTPUT_DIR / "g4_explainability_summary.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4, ensure_ascii=False)


# --------------------------------------------------
# FINISHED
# --------------------------------------------------

print("\\nG4 completed successfully.")
print(f"Markdown report: {report_path}")
print(f"JSON summary: {json_path}")
print(f"Feature chart: {chart_path}")
print(f"Output directory: {OUTPUT_DIR}")
