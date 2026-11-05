
from __future__ import annotations

import json
import pickle
import random
import re
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# -------------------- CONFIGURATION --------------------
ROOT = Path.cwd()
SMART_DIRS = [
    ROOT / "data/raw/smartstrike/SensorReadings/DatasetV4",
    ROOT / "data/raw/smartstrike/SensorReadings/NoneDatasetV4",
]
SI_PKL = ROOT / "data/public/silent_impact/raw/Classification_Dominant.pkl"
B2_METADATA = ROOT / "data/processed/silent_impact/classification_b2/classification_b2_metadata.json"

OUT = ROOT / "data/processed/combined_c"
MODEL_DIR = ROOT / "models/experiment_c"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FRAMES = 180
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Experiment C deliberately uses only the two shared classes.
LABELS = {"BACKHAND": 0, "FOREHAND": 1}
CLASS_NAMES = ["BACKHAND", "FOREHAND"]
CHANNELS = ["linear_x", "linear_y", "linear_z", "gyro_x", "gyro_y", "gyro_z"]

# Silent Impact class IDs from the dataset audit:
# 0 = Forehand Stroke, 1 = Backhand Stroke.
SI_CLASS_TO_TARGET = {0: 1, 1: 0}

# -------------------- UTILITIES --------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

def resample_sequence(arr: np.ndarray, frames=FRAMES) -> np.ndarray:
    """Linear interpolation along time; input shape (T, C)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"Invalid sequence shape: {arr.shape}")
    old_x = np.linspace(0.0, 1.0, arr.shape[0])
    new_x = np.linspace(0.0, 1.0, frames)
    out = np.stack(
        [np.interp(new_x, old_x, arr[:, c]) for c in range(arr.shape[1])],
        axis=1
    )
    return out.astype(np.float32)

def smartstrike_label(path: Path):
    """Strict label detection: unknown files are skipped, never silently mapped to NONE."""
    text = (path.stem + " " + str(path.parent)).upper()
    if "BACKHAND_SLICE" in text or "BACKHAND-SLICE" in text:
        return None  # Not part of the two shared classes.
    if "BACKHAND" in text:
        return 0
    if "FOREHAND" in text:
        return 1
    return None

def load_smartstrike():
    X, y, names, skipped = [], [], [], []
    for folder in SMART_DIRS:
        if not folder.exists():
            continue
        for file in sorted(folder.rglob("*.csv")):
            label = smartstrike_label(file)
            if label is None:
                skipped.append(str(file.relative_to(ROOT)))
                continue
            try:
                df = pd.read_csv(file)
                missing = [c for c in CHANNELS if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {missing}")
                seq = df[CHANNELS].apply(pd.to_numeric, errors="coerce").to_numpy()
                seq = seq[np.isfinite(seq).all(axis=1)]
                if len(seq) < 2:
                    raise ValueError("Fewer than 2 valid sensor rows")
                X.append(resample_sequence(seq))
                y.append(label)
                names.append(str(file.relative_to(ROOT)))
            except Exception as e:
                skipped.append({"file": str(file.relative_to(ROOT)), "reason": str(e)})
    if not X:
        raise RuntimeError(
            "No SmartStrike BACKHAND/FOREHAND files found. "
            "Check filename/folder naming. Do not change unknown labels to NONE."
        )
    return np.stack(X), np.asarray(y, dtype=np.int64), names, skipped

def load_silent_impact():
    if not SI_PKL.exists():
        raise FileNotFoundError(f"Silent Impact file not found: {SI_PKL}")
    with SI_PKL.open("rb") as f:
        data = pickle.load(f)
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[1] < 181 or data.shape[2] < 6:
        raise ValueError(f"Unexpected Silent Impact shape: {data.shape}")
    # First row stores metadata. The next 180 rows are sensor frames.
    X = data[:, 1:181, :6].astype(np.float32)
    source_class = data[:, 0, 0].astype(int)
    participant = data[:, 0, 1].astype(int)
    keep = np.isin(source_class, [0, 1])
    X, source_class, participant = X[keep], source_class[keep], participant[keep]
    y = np.asarray([SI_CLASS_TO_TARGET[int(c)] for c in source_class], dtype=np.int64)

    # Silent Impact gyro channels are treated as deg/s, consistent with B3.
    # If source documentation does not confirm this unit, verify before final reporting.
    X[:, :, 3:6] = np.deg2rad(X[:, :, 3:6])
    return X, y, participant, source_class

def load_b2_participant_split():
    """Reuse B2 participant split if metadata contains it; otherwise deterministic fallback."""
    if B2_METADATA.exists():
        try:
            meta = json.loads(B2_METADATA.read_text(encoding="utf-8"))
            for key in ("train_participants", "train_ids", "train"):
                if key in meta and isinstance(meta[key], list):
                    vals = [int(v) for v in meta[key]]
                    if vals:
                        return set(vals), "B2 metadata"
        except Exception:
            pass
    # Fallback: deterministic participant split, explicitly logged.
    participants = np.arange(1, 21)
    train_ids, other = train_test_split(
        participants, test_size=0.20, random_state=SEED
    )
    return set(map(int, train_ids)), "deterministic fallback (B2 train IDs not parsed)"

# -------------------- MODEL --------------------
class SensorDataset(Dataset):
    def __init__(self, X, y):
        # (N, T, C) -> (N, C, T)
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN1D(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, n_classes)
    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))

def train_model(Xtr, ytr, Xval, yval):
    model = CNN1D(2).to(DEVICE)
    train_loader = DataLoader(SensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SensorDataset(Xval, yval), batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_state = float("inf"), None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):
        model.train()
        losses, correct, total = [], 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        train_loss = float(np.mean(losses))
        train_acc = correct / total

        model.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(yb)
        val_loss = float(np.mean(val_losses))
        val_acc = val_correct / val_total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

def evaluate(model, X, y):
    loader = DataLoader(SensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy().tolist())
    p = np.asarray(preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, p, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "class_report": classification_report(
            y, p, labels=[0, 1], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
        "predictions": p.tolist()
    }

# -------------------- SEVEN PHASES --------------------
def main():
    set_seed()
    log = {"experiment": "C - Combined Datasets", "started_at": datetime.now().isoformat(),
           "device": DEVICE, "phases": {}}

    print("\n========== C1: DATASET AUDIT ==========")
    Xsi, ysi, si_participants, si_original_labels = load_silent_impact()
    Xss, yss, ss_names, skipped = load_smartstrike()
    audit = {
        "silent_impact_shape": list(Xsi.shape),
        "silent_impact_class_counts": {CLASS_NAMES[i]: int((ysi == i).sum()) for i in [0, 1]},
        "silent_impact_participants": sorted(map(int, np.unique(si_participants))),
        "smartstrike_shape": list(Xss.shape),
        "smartstrike_class_counts": {CLASS_NAMES[i]: int((yss == i).sum()) for i in [0, 1]},
        "smartstrike_skipped_count": len(skipped),
        "smartstrike_skipped_examples": skipped[:30],
        "channels": CHANNELS,
        "frames": FRAMES
    }
    write_json(OUT / "C1_dataset_audit.json", audit)
    log["phases"]["C1"] = audit
    print(json.dumps(audit, indent=2))

    print("\n========== C2: PREPROCESSING ==========")
    # SI already has 180 frames; SmartStrike was interpolated to 180 in loader.
    if not (np.isfinite(Xsi).all() and np.isfinite(Xss).all()):
        raise ValueError("NaN/Inf detected after preprocessing.")
    c2 = {
        "silent_impact_frames": int(Xsi.shape[1]),
        "smartstrike_frames": int(Xss.shape[1]),
        "silent_impact_finite": True,
        "smartstrike_finite": True,
        "smartstrike_resampling": "linear interpolation to 180 frames",
        "silent_impact_gyro_conversion": "deg/s to rad/s (verify source documentation)"
    }
    write_json(OUT / "C2_preprocessing.json", c2)
    log["phases"]["C2"] = c2
    print(c2)

    print("\n========== C3: SOURCE PARTICIPANT SPLIT ==========")
    si_train_participants, split_origin = load_b2_participant_split()
    si_train_mask = np.isin(si_participants, list(si_train_participants))
    Xsi_train, ysi_train = Xsi[si_train_mask], ysi[si_train_mask]
    si_excluded = sorted(set(map(int, np.unique(si_participants))) - si_train_participants)
    c3 = {
        "participant_split_origin": split_origin,
        "source_train_participants": sorted(si_train_participants),
        "source_excluded_participants": si_excluded,
        "source_training_samples": len(ysi_train)
    }
    write_json(OUT / "C3_source_split.json", c3)
    log["phases"]["C3"] = c3
    print(c3)

    print("\n========== C4: SMARTSTRIKE SPLIT + COMBINATION ==========")
    idx = np.arange(len(yss))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.30, random_state=SEED, stratify=yss
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=SEED, stratify=yss[idx_temp]
    )
    Xss_train, yss_train = Xss[idx_train], yss[idx_train]
    Xss_val, yss_val = Xss[idx_val], yss[idx_val]
    Xss_test, yss_test = Xss[idx_test], yss[idx_test]

    # Combined training set: source-domain training data + target-domain training data.
    X_combined = np.concatenate([Xsi_train, Xss_train], axis=0)
    y_combined = np.concatenate([ysi_train, yss_train], axis=0)

    # Shuffle combined training examples deterministically.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y_combined))
    X_combined, y_combined = X_combined[order], y_combined[order]

    c4 = {
        "smartstrike_train": len(yss_train),
        "smartstrike_validation": len(yss_val),
        "smartstrike_test": len(yss_test),
        "silent_impact_train": len(ysi_train),
        "combined_training_total": len(y_combined),
        "combined_class_counts": {CLASS_NAMES[i]: int((y_combined == i).sum()) for i in [0, 1]},
        "evaluation_domain": "SmartStrike held-out test recordings"
    }
    write_json(OUT / "C4_combined_split.json", c4)
    log["phases"]["C4"] = c4
    print(c4)

    print("\n========== C5: TRAIN-ONLY NORMALIZATION ==========")
    # Fit normalization only on the combined training set.
    mean = X_combined.mean(axis=(0, 1), keepdims=True)
    std = X_combined.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    X_combined_n = (X_combined - mean) / std
    Xval_n = (Xss_val - mean) / std
    Xtest_n = (Xss_test - mean) / std
    np.savez_compressed(
        OUT / "experiment_C_data.npz",
        X_train=X_combined_n, y_train=y_combined,
        X_val=Xval_n, y_val=yss_val,
        X_test=Xtest_n, y_test=yss_test,
        mean=mean, std=std
    )
    c5 = {"normalization": "mean/std fitted on combined training set only",
          "channel_mean": mean.reshape(-1).tolist(),
          "channel_std": std.reshape(-1).tolist()}
    write_json(OUT / "C5_normalization.json", c5)
    log["phases"]["C5"] = c5
    print("Saved normalized combined dataset.")

    print("\n========== C6: MODEL TRAINING ==========")
    model, history = train_model(X_combined_n, y_combined, Xval_n, yss_val)
    model_path = MODEL_DIR / "smartstrike_combined_model.pth"
    torch.save(model.state_dict(), model_path)
    write_json(OUT / "C6_training_history.json", history)

    plt.figure()
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "C6_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="Train accuracy")
    plt.plot(history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "C6_accuracy.png", dpi=150)
    plt.close()

    c6 = {"model": str(model_path.relative_to(ROOT)), "epochs": EPOCHS,
          "batch_size": BATCH_SIZE, "learning_rate": LR, "device": DEVICE}
    log["phases"]["C6"] = c6
    print(c6)

    print("\n========== C7: SMARTSTRIKE TEST EVALUATION ==========")
    metrics = evaluate(model, Xtest_n, yss_test)
    cm = np.asarray(metrics["confusion_matrix"])
    plt.figure()
    plt.imshow(cm)
    plt.title("Experiment C – SmartStrike Test Confusion Matrix")
    plt.xticks([0, 1], CLASS_NAMES, rotation=20)
    plt.yticks([0, 1], CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUT / "C7_confusion_matrix.png", dpi=150)
    plt.close()

    metrics_without_preds = {k: v for k, v in metrics.items() if k != "predictions"}
    write_json(OUT / "C7_test_metrics.json", metrics_without_preds)
    log["phases"]["C7"] = metrics_without_preds

    report = f"""# Experiment C – Combined Datasets

## Цел
Тренирање на 1D CNN со комбинирани податоци од Silent Impact и SmartStrike за две заеднички класи: BACKHAND и FOREHAND.

## Податоци
- Silent Impact training примероци: {len(ysi_train)}
- SmartStrike training примероци: {len(yss_train)}
- Combined training примероци: {len(y_combined)}
- SmartStrike validation примероци: {len(yss_val)}
- SmartStrike test примероци: {len(yss_test)}

## Резултати на SmartStrike Test
- Accuracy: {metrics['accuracy']:.4f}
- Macro Precision: {metrics['macro_precision']:.4f}
- Macro Recall: {metrics['macro_recall']:.4f}
- Macro F1: {metrics['macro_f1']:.4f}

## Confusion Matrix
{metrics['confusion_matrix']}

## Ограничувања
- SmartStrike поделбата е на ниво на снимки бидејќи нема participant IDs.
- Линеарната интерполација на 180 временски точки не гарантира физичко временско усогласување.
- Silent Impact жироскопските вредности се конвертирани од deg/s во rad/s; мерната единица треба да се потврди со оригиналната документација.
- Класификацијата е ограничена на BACKHAND и FOREHAND.
- Test резултатите се од SmartStrike снимки и не претставуваат независна евалуација кај нови учесници.
"""
    (OUT / "experiment_C_report.md").write_text(report, encoding="utf-8")

    log["completed_at"] = datetime.now().isoformat()
    write_json(OUT / "experiment_C_full_log.json", log)

    print("\n========== EXPERIMENT C COMPLETED ==========")
    print(json.dumps(metrics_without_preds, indent=2))
    print(f"\nAll outputs saved in: {OUT}")
    print(f"Report: {OUT / 'experiment_C_report.md'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nEXPERIMENT C FAILED:")
        print(str(exc))
        traceback.print_exc()
        sys.exit(1)
