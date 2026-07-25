import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import joblib

from dataset import load_dataset
from features import extract_features



dataset = load_dataset("data")


X=[]
y=[]


for sample in dataset:

    X.append(
        extract_features(sample)
    )

    y.append(
        sample["label"]
    )


X = pd.DataFrame(X)



print(X.shape)

print(
    pd.Series(y).value_counts()
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


predictions = model.predict(
    X_test
)


print(
    classification_report(
        y_test,
        predictions
    )
)


joblib.dump(
    model,
    "models/forehand_backhand_model.pkl"
)