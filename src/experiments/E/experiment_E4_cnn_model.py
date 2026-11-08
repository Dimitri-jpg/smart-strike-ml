import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


ROOT = Path.cwd()
DATA_PATH = ROOT / "data/processed/experiment_e/experiment_E2_data.npz"
OUTPUT_DIR = ROOT / "data/processed/experiment_e/e4"
MODEL_DIR = ROOT / "models/experiment_e"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("E4 1D CNN")
print("Device:", device)


def get_array(data, possible_names):
    for name in possible_names:
        if name in data.files:
            return data[name]
    raise KeyError(
        f"Could not find any of {possible_names}. "
        f"Available keys: {data.files}"
    )


data = np.load(DATA_PATH, allow_pickle=True)

# Load the full time-series array and split indices.
X_series = get_array(data, ["X_series", "X_time_series", "time_series"])

train_idx = get_array(
    data, ["train_indices", "idx_train", "train_idx"]
).astype(int)

val_idx = get_array(
    data, ["validation_indices", "val_indices", "idx_validation", "val_idx"]
).astype(int)

test_idx = get_array(
    data, ["test_indices", "idx_test", "test_idx"]
).astype(int)

y_all = get_array(data, ["y", "labels", "y_all", "labels_all"])

# If labels are stored as strings, encode them.
encoder = LabelEncoder()
y_all = encoder.fit_transform(y_all.astype(str))

X_train = X_series[train_idx].astype(np.float32)
X_val = X_series[val_idx].astype(np.float32)
X_test = X_series[test_idx].astype(np.float32)

y_train = y_all[train_idx]
y_val = y_all[val_idx]
y_test = y_all[test_idx]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("Classes:", list(encoder.classes_))

# Replace invalid values.
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize each sensor channel using training data only.
channel_mean = X_train.mean(axis=(0, 1), keepdims=True)
channel_std = X_train.std(axis=(0, 1), keepdims=True)
channel_std[channel_std == 0] = 1.0

X_train = (X_train - channel_mean) / channel_std
X_val = (X_val - channel_mean) / channel_std
X_test = (X_test - channel_mean) / channel_std

# Convert from (samples, time, channels) to (samples, channels, time).
X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))


class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val, False)
test_loader = make_loader(X_test, y_test, False)

model = CNN1D(num_classes=len(encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def evaluate(loader):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(all_labels)

    result = {
        "loss": float(average_loss),
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_precision": float(
            precision_score(
                all_labels, all_predictions,
                average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                all_labels, all_predictions,
                average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                all_labels, all_predictions,
                average="macro", zero_division=0
            )
        ),
    }

    return result, np.array(all_labels), np.array(all_predictions)


best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0
history = []

best_model_path = MODEL_DIR / "e4_cnn_1d_best.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_metrics, _, _ = evaluate(val_loader)

    history.append({
        "epoch": epoch,
        "train_loss": float(train_loss),
        "validation": val_metrics,
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_metrics['loss']:.4f} | "
        f"Val accuracy: {val_metrics['accuracy']:.4f} | "
        f"Val macro-F1: {val_metrics['macro_f1']:.4f}"
    )

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": len(encoder.classes_),
            "classes": encoder.classes_.tolist(),
            "channel_mean": channel_mean,
            "channel_std": channel_std,
        }, best_model_path)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break


# Load the best checkpoint and evaluate the test set once.
checkpoint = torch.load(
    best_model_path,
    map_location=device,
    weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])

test_metrics, y_test_true, y_test_pred = evaluate(test_loader)

test_report = classification_report(
    y_test_true,
    y_test_pred,
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0,
)

results = {
    "model": "1D CNN",
    "best_epoch": best_epoch,
    "best_validation_macro_f1": float(best_val_f1),
    "test": test_metrics,
    "test_classification_report": test_report,
    "classes": encoder.classes_.tolist(),
    "train_samples": int(len(y_train)),
    "validation_samples": int(len(y_val)),
    "test_samples": int(len(y_test)),
    "input_shape": list(X_series.shape[1:]),
    "normalization": "Training-set channel mean and standard deviation",
    "history": history,
}

results_path = OUTPUT_DIR / "e4_cnn_results.json"

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nE4 completed successfully.")
print("Best epoch:", best_epoch)
print("Test metrics:", test_metrics)
print("Model saved:", best_model_path)
print("Results saved:", results_path)