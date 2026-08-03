import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataset import load_dataset
from features import extract_features


dataset = load_dataset("data")

classifier_features = []
classifier_labels = []

for sample in dataset:

    classifier_features.append(
        extract_features(sample)
    )

    classifier_labels.append(
        sample["label"]
    )


X_classifier = pd.DataFrame(classifier_features)
y_classifier = pd.Series(classifier_labels)


print()
print("Classifier dataset")
print("------------------")
print(X_classifier.shape)
print(y_classifier.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X_classifier,
    y_classifier,
    test_size=0.2,
    random_state=42,
    stratify=y_classifier
)


classifier = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    random_state=42
)


classifier.fit(
    X_train,
    y_train
)


predictions = classifier.predict(
    X_test
)


print()
print("Classification report")
print("---------------------")
print(
    classification_report(
        y_test,
        predictions
    )
)


encoder = LabelEncoder()

encoder.fit(
    y_classifier
)


regressor_rows = []

for sample in dataset:

    features = extract_features(sample)

    features["score"] = sample["score"]

    regressor_rows.append(features)


regressor_df = pd.DataFrame(regressor_rows)


X_regressor = regressor_df.drop(
    columns=["score"]
)

y_regressor = regressor_df["score"]


X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regressor,
    y_regressor,
    test_size=0.2,
    random_state=42
)


regressor = RandomForestRegressor(
    n_estimators=400,
    max_depth=10,
    random_state=42
)


regressor.fit(
    X_train_reg,
    y_train_reg
)


score_predictions = regressor.predict(
    X_test_reg
)


print()
print("Regression report")
print("-----------------")
print(
    "MAE:",
    round(
        mean_absolute_error(
            y_test_reg,
            score_predictions
        ),
        3
    )
)

print(
    "R²:",
    round(
        r2_score(
            y_test_reg,
            score_predictions
        ),
        3
    )
)


os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    classifier,
    "models/classifier.pkl"
)

joblib.dump(
    regressor,
    "models/quality_regressor.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print()
print("Saved:")
print("models/classifier.pkl")
print("models/quality_regressor.pkl")
print("models/label_encoder.pkl")