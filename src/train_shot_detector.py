import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from dataset import load_dataset
from features import extract_features


dataset = load_dataset("data")


X = []
y = []


for sample in dataset:

    X.append(
        extract_features(sample)
    )

    if sample["label"] == "NONE":
        y.append("NO_SHOT")
    else:
        y.append("SHOT")


X = pd.DataFrame(X)
y = pd.Series(y)


print()
print("Shot detector dataset")
print("---------------------")
print(X.shape)
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


predictions = model.predict(
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


os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    model,
    "models/shot_detector.pkl"
)


print()
print("Saved:")
print("models/shot_detector.pkl")