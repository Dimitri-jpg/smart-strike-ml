
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# EXPERIMENT B3 - TRANSFER LEARNING
# Silent Impact (2 classes) -> SmartStrike (4 classes)
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SI_PATH = (
    ROOT / "data/public/silent_impact/raw/"
    "Classification_Dominant.pkl"
)

B2_METADATA_PATH = (
    ROOT / "data/processed/silent_impact/"
    "classification_b2/classification_b2_metadata.json"
)

SMARTSTRIKE_DIRS = [
    ROOT / "data/raw/smartstrike/SensorReadings/DatasetV4",
    ROOT / "data/raw/smartstrike/SensorReadings/NoneDatasetV4",
]

OUTPUT_DIR = ROOT / "data/processed/transfer_learning_b3"
MODEL_DIR = ROOT / "models/experiment_b3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FRAMES = 180
CHANNELS = 6

BATCH_SIZE = 64
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 30

PRETRAIN_LR = 0.001
FINETUNE_LR = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Assumptions requiring confirmation from source documentation.
SI_GYRO_DEG_TO_RAD = True
SMARTSTRIKE_GYRO_ALREADY_RAD = True

# Channel order in both datasets:
CHANNEL_NAMES = [
    "linear_acc_x",
    "linear_acc_y",
    "linear_acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

# Target class IDs:
TARGET_CLASSES = {
    "BACKHAND": 0,
    "BACKHAND_SLICE": 1,
    "FOREHAND": 2,
    "NONE": 3,
}

TARGET_CLASS_NAMES = [
    "BACKHAND",
    "BACKHAND_SLICE",
    "FOREHAND",
    "NONE",
]

# Silent Impact class IDs:
SOURCE_CLASSES = {
    0: "Forehand Stroke",
    1: "Backhand Stroke",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# GENERAL HELPERS
# ============================================================

def resample_sequence(sequence, target_frames=FRAMES):
    """Interpolate one recording to a fixed number of frames."""

    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2 or sequence.shape[1] != CHANNELS:
        raise ValueError(
            f"Expected (frames, 6), got {sequence.shape}"
        )

    if len(sequence) < 2:
        raise ValueError("Recording has fewer than 2 frames.")

    if not np.isfinite(sequence).all():
        raise ValueError("Recording contains NaN or Inf.")

    old_positions = np.linspace(0, 1, len(sequence))
    new_positions = np.linspace(0, 1, target_frames)

    result = np.zeros(
        (target_frames, CHANNELS),
        dtype=np.float32,
    )

    for channel in range(CHANNELS):
        result[:, channel] = np.interp(
            new_positions,
            old_positions,
            sequence[:, channel],
        )

    return result


def convert_silent_impact_units(X):
    """Convert SI gyro channels from deg/s to rad/s."""

    X = X.copy()

    if SI_GYRO_DEG_TO_RAD:
        X[:, :, 3:6] *= np.pi / 180.0

    return X


# ============================================================
# 1. LOAD SILENT IMPACT SOURCE DATA
# ============================================================

def load_source_data():
    print("\n" + "=" * 60)
    print("B3.1 - LOADING SILENT IMPACT SOURCE DATA")
    print("=" * 60)

    if not SI_PATH.exists():
        raise FileNotFoundError(SI_PATH)

    if not B2_METADATA_PATH.exists():
        raise FileNotFoundError(B2_METADATA_PATH)

    with open(SI_PATH, "rb") as file:
        raw = pickle.load(file)

    raw = np.asarray(raw)

    if raw.shape != (6000, 181, 6):
        raise ValueError(
            f"Unexpected SI shape: {raw.shape}"
        )

    labels = raw[:, 0, 0].astype(int)
    participants = raw[:, 0, 1].astype(int)

    X = raw[:, 1:, :].astype(np.float32)

    if not np.isfinite(X).all():
        raise ValueError("SI contains NaN or Inf.")

    # Only the two shared stroke classes are used for pretraining.
    mask = np.isin(labels, [0, 1])

    X = X[mask]
    y = labels[mask]
    groups = participants[mask]

    # Use participant split already defined in B2.
    with open(B2_METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    train_ids = metadata["train_participants"]
    validation_ids = metadata["validation_participants"]
    test_ids = metadata["test_participants"]

    # Convert gyro to rad/s before calculating normalization.
    X = convert_silent_impact_units(X)

    train_mask = np.isin(groups, train_ids)
    val_mask = np.isin(groups, validation_ids)
    test_mask = np.isin(groups, test_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("Source samples:", len(X))
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # Fit source normalization on source training data only.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print("Source normalization fit on source train only.")

    return (
        X_train.astype(np.float32),
        y_train,
        X_val.astype(np.float32),
        y_val,
        X_test.astype(np.float32),
        y_test,
        mean,
        std,
    )


# ============================================================
# 2. LOAD SMARTSTRIKE TARGET DATA
# ============================================================

def find_column(df, candidates):
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    raise ValueError(
        f"Missing expected column. Available: {list(df.columns)}"
    )


def load_smartstrike_data():
    print("\n" + "=" * 60)
    print("B3.2 - LOADING SMARTSTRIKE TARGET DATA")
    print("=" * 60)

    recordings = []
    labels = []
    filenames = []

    for directory in SMARTSTRIKE_DIRS:
        if not directory.exists():
            raise FileNotFoundError(directory)

        for path in sorted(directory.rglob("*.csv")):
            df = pd.read_csv(path)

            shot_column = find_column(
                df, ["shot_type"]
            )

            label_text = str(
                df[shot_column].dropna().iloc[0]
            ).strip().upper()

            if label_text not in TARGET_CLASSES:
                raise ValueError(
                    f"Unknown shot_type '{label_text}' in {path}"
                )

            # Six channels shared with Silent Impact.
            columns = [
                find_column(df, ["linear_x"]),
                find_column(df, ["linear_y"]),
                find_column(df, ["linear_z"]),
                find_column(df, ["gyro_x"]),
                find_column(df, ["gyro_y"]),
                find_column(df, ["gyro_z"]),
            ]

            sensor_data = df[columns].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()

            sequence = sensor_data.to_numpy(
                dtype=np.float32
            )

            if len(sequence) < 2:
                print("Skipping short recording:", path.name)
                continue

            if not np.isfinite(sequence).all():
                raise ValueError(
                    f"Invalid sensor values in {path}"
                )

            sequence = resample_sequence(sequence)

            if not SMARTSTRIKE_GYRO_ALREADY_RAD:
                raise ValueError(
                    "SmartStrike gyro units need conversion "
                    "before this experiment."
                )

            recordings.append(sequence)
            labels.append(TARGET_CLASSES[label_text])
            filenames.append(str(path.relative_to(ROOT)))

    if not recordings:
        raise ValueError("No SmartStrike recordings found.")

    X = np.stack(recordings).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print("SmartStrike samples:", len(X))
    print("SmartStrike shape:", X.shape)

    for class_id, name in enumerate(TARGET_CLASS_NAMES):
        print(name, ":", int(np.sum(y == class_id)))

    return X, y, filenames


# ============================================================
# 3. CREATE SMARTSTRIKE SPLIT
# ============================================================

def create_target_split(X, y, filenames):
    print("\n" + "=" * 60)
    print("B3.3 - SMARTSTRIKE TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)

    indices = np.arange(len(y))

    train_idx, temporary_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    val_idx, test_idx = train_test_split(
        temporary_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=y[temporary_idx],
    )

    # Save exact file assignments for reproducibility.
    split_info = {
        "train": [filenames[i] for i in train_idx],
        "validation": [filenames[i] for i in val_idx],
        "test": [filenames[i] for i in test_idx],
        "method": "Stratified recording-level split",
        "seed": SEED,
        "limitation": (
            "SmartStrike participant IDs are unavailable; "
            "recording-level split does not guarantee session "
            "or participant independence."
        ),
    }

    with open(
        OUTPUT_DIR / "smartstrike_split.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(split_info, file, indent=4)

    print("Train:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


# ============================================================
# 4. APPLY SOURCE NORMALIZATION TO TARGET
# ============================================================

def normalize_target(
    X_train, X_val, X_test, source_mean, source_std
):
    print("\nApplying source normalization to SmartStrike...")

    X_train = (X_train - source_mean) / source_std
    X_val = (X_val - source_mean) / source_std
    X_test = (X_test - source_mean) / source_std

    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
    )


# ============================================================
# 5. PYTORCH MODEL
# ============================================================

class TennisCNN(nn.Module):

    def __init__(self, number_of_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(
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
            nn.Flatten(),
        )

        self.classifier = nn.Linear(
            128, number_of_classes
        )

    def forward(self, x):
        # Input: batch x frames x channels.
        x = x.transpose(1, 2)
        features = self.feature_extractor(x)
        return self.classifier(features)


# ============================================================
# 6. DATA LOADERS
# ============================================================

def make_loader(X, y, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


# ============================================================
# 7. TRAINING AND VALIDATION
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(y_batch)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )
        all_labels.extend(y_batch.detach().cpu().numpy())

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    return average_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    name,
):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["validation_accuracy"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ============================================================
# 8. EVALUATION
# ============================================================

def evaluate_model(model, X, y, class_names, output_prefix):
    print("\n" + "=" * 60)
    print("FINAL EVALUATION:", output_prefix)
    print("=" * 60)

    loader = make_loader(X, y)

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch.to(DEVICE))
            predicted = outputs.argmax(dim=1)

            predictions.extend(
                predicted.cpu().numpy()
            )
            actual.extend(y_batch.numpy())

    accuracy = accuracy_score(actual, predictions)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual,
            predictions,
            labels=list(range(len(class_names))),
            zero_division=0,
        )
    )

    report = classification_report(
        actual,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print("Accuracy:", round(accuracy, 4))
    print(
        classification_report(
            actual,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        actual,
        predictions,
        labels=list(range(len(class_names))),
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {output_prefix}")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{output_prefix}_confusion_matrix.png"
    )
    plt.close()

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    with open(
        OUTPUT_DIR / f"{output_prefix}_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    return metrics


# ============================================================
# 9. PLOT TRAINING HISTORY
# ============================================================

def save_history(history, prefix):
    with open(
        OUTPUT_DIR / f"{prefix}_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=4)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(
        epochs,
        history["validation_loss"],
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_loss.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train",
    )
    plt.plot(
        epochs,
        history["validation_accuracy"],
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix} - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_accuracy.png")
    plt.close()


# ============================================================
# 10. MAIN B3 PIPELINE
# ============================================================

def main():
    set_seed()

    print("Device:", DEVICE)

    if not SI_GYRO_DEG_TO_RAD:
        raise ValueError(
            "Confirm Silent Impact gyro units before training."
        )

    if not SMARTSTRIKE_GYRO_ALREADY_RAD:
        raise ValueError(
            "Confirm SmartStrike gyro units before training."
        )

    # A. Source dataset
    (
        X_source_train,
        y_source_train,
        X_source_val,
        y_source_val,
        X_source_test,
        y_source_test,
        source_mean,
        source_std,
    ) = load_source_data()

    # B. Target dataset
    X_target, y_target, filenames = load_smartstrike_data()

    (
        X_target_train,
        y_target_train,
        X_target_val,
        y_target_val,
        X_target_test,
        y_target_test,
    ) = create_target_split(
        X_target, y_target, filenames
    )

    # C. Use source train normalization for target data.
    (
        X_target_train,
        X_target_val,
        X_target_test,
    ) = normalize_target(
        X_target_train,
        X_target_val,
        X_target_test,
        source_mean,
        source_std,
    )

    # D. Source pretraining: two classes.
    source_train_loader = make_loader(
        X_source_train, y_source_train, shuffle=True
    )
    source_val_loader = make_loader(
        X_source_val, y_source_val
    )

    model = TennisCNN(number_of_classes=2).to(DEVICE)

    model, pretrain_history = train_model(
        model,
        source_train_loader,
        source_val_loader,
        PRETRAIN_EPOCHS,
        PRETRAIN_LR,
        "B3.4 - SILENT IMPACT PRETRAINING",
    )

    save_history(pretrain_history, "pretraining")

    # Save source-pretrained model.
    torch.save(
        model.state_dict(),
        MODEL_DIR / "silent_impact_pretrained.pth",
    )

    # Evaluate source model on held-out SI participants.
    evaluate_model(
        model,
        X_source_test,
        y_source_test,
        ["Forehand Stroke", "Backhand Stroke"],
        "silent_impact_source_test",
    )

    # E. Replace two-class head with four-class head.
    # Keep the learned feature extractor.
    transferred_model = TennisCNN(
        number_of_classes=4
    ).to(DEVICE)

    transferred_model.feature_extractor.load_state_dict(
        model.feature_extractor.state_dict()
    )

    print("\nTransferred feature extractor from Silent Impact.")

    # F. Fine-tuning on SmartStrike.
    target_train_loader = make_loader(
        X_target_train,
        y_target_train,
        shuffle=True,
    )

    target_val_loader = make_loader(
        X_target_val,
        y_target_val,
    )

    transferred_model, finetune_history = train_model(
        transferred_model,
        target_train_loader,
        target_val_loader,
        FINETUNE_EPOCHS,
        FINETUNE_LR,
        "B3.5 - SMARTSTRIKE FINE-TUNING",
    )

    save_history(finetune_history, "finetuning")

    # G. Final evaluation on SmartStrike test set.
    final_metrics = evaluate_model(
        transferred_model,
        X_target_test,
        y_target_test,
        TARGET_CLASS_NAMES,
        "smartstrike_test",
    )

    # H. Save final model.
    torch.save(
        transferred_model.state_dict(),
        MODEL_DIR / "smartstrike_transfer_model.pth",
    )

    # I. Save experiment metadata.
    experiment_metadata = {
        "experiment": "B3 - Transfer Learning",
        "source": "Silent Impact Classification Dominant",
        "source_classes": SOURCE_CLASSES,
        "target": "SmartStrike",
        "target_classes": TARGET_CLASSES,
        "frames": FRAMES,
        "channels": CHANNEL_NAMES,
        "architecture": "1D CNN",
        "pretraining_epochs": PRETRAIN_EPOCHS,
        "finetuning_epochs": FINETUNE_EPOCHS,
        "random_seed": SEED,
        "device": str(DEVICE),
        "si_gyro_deg_to_rad": SI_GYRO_DEG_TO_RAD,
        "smartstrike_gyro_already_rad": SMARTSTRIKE_GYRO_ALREADY_RAD,
        "temporal_resampling": (
            "Linear interpolation to 180 points per recording"
        ),
        "target_split": (
            "Stratified recording-level split 70/15/15"
        ),
        "target_split_limitation": (
            "No participant IDs; session independence is not guaranteed."
        ),
        "target_normalization": (
            "Source training mean and standard deviation"
        ),
        "test_metrics": final_metrics,
    }

    with open(
        OUTPUT_DIR / "b3_experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(experiment_metadata, file, indent=4)

    print("\n" + "=" * 60)
    print("EXPERIMENT B3 COMPLETED")
    print("=" * 60)
    print("Final SmartStrike test accuracy:",
          round(final_metrics["accuracy"], 4))
    print("Final SmartStrike macro F1:",
          round(final_metrics["macro_f1"], 4))
    print("Model:", MODEL_DIR / "smartstrike_transfer_model.pth")
    print("Results:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
