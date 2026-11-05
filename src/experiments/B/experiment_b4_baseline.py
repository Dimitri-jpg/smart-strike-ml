import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FRAMES = 180

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data/raw/smartstrike/SensorReadings/DatasetV4"
NONE_PATH = ROOT / "data/raw/smartstrike/SensorReadings/NoneDatasetV4"

SPLIT_PATH = ROOT / "data/processed/transfer_learning_b3/smartstrike_split.json"

OUTPUT_PATH = ROOT / "data/processed/baseline_b4"
MODEL_PATH = ROOT / "models/experiment_b4"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = {
    "BACKHAND": 0,
    "BACKHAND_SLICE": 1,
    "FOREHAND": 2,
    "NONE": 3
}

FEATURES = [
    "linear_x", "linear_y", "linear_z",
    "gyro_x", "gyro_y", "gyro_z"
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATA LOADING
# ============================================================

def load_recording(path, label):
    df = pd.read_csv(path)

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    values = df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).interpolate(limit_direction="both").fillna(0)

    data = values.to_numpy(dtype=np.float32)

    if len(data) < 2:
        raise ValueError(f"Recording too short: {path.name}")

    old_positions = np.linspace(0, 1, len(data))
    new_positions = np.linspace(0, 1, FRAMES)

    resampled = np.column_stack([
        np.interp(new_positions, old_positions, data[:, channel])
        for channel in range(data.shape[1])
    ])

    return resampled, label, path.name


def load_dataset():
    X, y, filenames = [], [], []

    for folder in [DATASET_PATH, NONE_PATH]:
        for path in sorted(folder.glob("*.csv")):
            name = path.stem.upper()

            if "BACKHAND_SLICE" in name:
                label = CLASSES["BACKHAND_SLICE"]
            elif "BACKHAND" in name:
                label = CLASSES["BACKHAND"]
            elif "FOREHAND" in name:
                label = CLASSES["FOREHAND"]
            else:
                label = CLASSES["NONE"]

            recording, label, filename = load_recording(path, label)

            X.append(recording)
            y.append(label)
            filenames.append(filename)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(filenames)
    )


# ============================================================
# SPLIT LOADING
# ============================================================

def load_existing_split(filenames, labels):
    """
    Reuses the exact B3 split.
    Supports split JSON files containing filename lists
    under train/validation/test keys.
    """

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    def find_names(obj, possible_keys):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in possible_keys:
                    if isinstance(value, list):
                        return value

                result = find_names(value, possible_keys)
                if result is not None:
                    return result

        return None

    train_names = find_names(split_data, {"train", "train_files"})
    val_names = find_names(
        split_data, {"validation", "val", "validation_files", "val_files"}
    )
    test_names = find_names(split_data, {"test", "test_files"})

    if not all([train_names, val_names, test_names]):
        raise ValueError(
            "Could not identify train/validation/test filenames "
            "in smartstrike_split.json. Check its structure."
        )

    train_names = {Path(str(x)).name for x in train_names}
    val_names = {Path(str(x)).name for x in val_names}
    test_names = {Path(str(x)).name for x in test_names}

    indices = {
        "train": [],
        "val": [],
        "test": []
    }

    for i, filename in enumerate(filenames):
        if filename in train_names:
            indices["train"].append(i)
        elif filename in val_names:
            indices["val"].append(i)
        elif filename in test_names:
            indices["test"].append(i)

    if sum(map(len, indices.values())) != len(filenames):
        raise ValueError(
            "Some recordings were not matched with the B3 split. "
            "Check filename format in smartstrike_split.json."
        )

    return indices


# ============================================================
# MODEL
# ============================================================

class SmartStrikeCNN(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
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

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == targets).sum().item()
            train_total += targets.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_model is not None:
        model.load_state_dict(best_model)

    return model, history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, loader):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            y_true.extend(targets.numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(CLASSES.keys()),
            output_dict=True,
            zero_division=0
        )
    }

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3]
    )

    print("\nTEST RESULTS")
    print(json.dumps(metrics, indent=4))

    return metrics, matrix


# ============================================================
# PLOTS
# ============================================================

def save_plots(history, matrix):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_accuracy"], label="Train")
    plt.plot(epochs, history["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("B4 Baseline Accuracy")
    plt.savefig(OUTPUT_PATH / "baseline_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("B4 Baseline Loss")
    plt.savefig(OUTPUT_PATH / "baseline_loss.png")
    plt.close()

    plt.figure()
    plt.imshow(matrix)
    plt.colorbar()
    plt.xticks(range(4), CLASSES.keys(), rotation=45)
    plt.yticks(range(4), CLASSES.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("B4 Confusion Matrix")

    for i in range(4):
        for j in range(4):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "baseline_confusion_matrix.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading SmartStrike dataset...")

    X, y, filenames = load_dataset()

    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    split = load_existing_split(filenames, y)

    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # Fit normalization only on training data.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    def make_loader(features, labels, shuffle=False):
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )

        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle
        )

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    print("\nInitializing fresh baseline CNN...")
    model = SmartStrikeCNN(num_classes=4).to(DEVICE)

    print(f"Training on: {DEVICE}")
    model, history = train_model(
        model,
        train_loader,
        val_loader
    )

    print("\nEvaluating on test set...")
    metrics, matrix = evaluate_model(model, test_loader)

    torch.save(
        model.state_dict(),
        MODEL_PATH / "smartstrike_baseline_model.pth"
    )

    with open(OUTPUT_PATH / "baseline_test_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    with open(OUTPUT_PATH / "baseline_training_history.json", "w") as file:
        json.dump(history, file, indent=4)

    np.save(
        OUTPUT_PATH / "baseline_confusion_matrix.npy",
        matrix
    )

    with open(OUTPUT_PATH / "b4_experiment_metadata.json", "w") as file:
        json.dump({
            "experiment": "B4 Baseline Model",
            "dataset": "SmartStrike",
            "architecture": "1D CNN",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "frames": FRAMES,
            "classes": CLASSES,
            "train_samples": len(train_idx),
            "validation_samples": len(val_idx),
            "test_samples": len(test_idx),
            "normalization": "SmartStrike training set statistics",
            "pretraining": False,
            "device": str(DEVICE)
        }, file, indent=4)

    save_plots(history, matrix)

    print("\nB4 BASELINE COMPLETED!")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Results saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()