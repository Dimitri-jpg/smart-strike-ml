import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from experiment_b4_baseline import (
    ROOT, load_dataset, SmartStrikeCNN, CLASSES,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE
)

OUTPUT = ROOT / "data/processed/robustness_b5"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEEDS = [21, 42, 100, 123, 2026]


def run_experiment(seed, X, y):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    indices = np.arange(len(y))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=y
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=y[temp_idx]
    )

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    def loader(features, labels, shuffle=False):
        ds = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            generator=generator if shuffle else None
        )

    train_loader = loader(X_train, y_train, True)
    val_loader = loader(X_val, y_val)
    test_loader = loader(X_test, y_test)

    model = SmartStrikeCNN(num_classes=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        count = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                loss = criterion(model(inputs), targets)
                val_loss += loss.item() * len(targets)
                count += len(targets)

        val_loss /= count

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(f"Seed {seed} | Epoch {epoch + 1}/{EPOCHS} | Val loss: {val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    actual, predicted = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(DEVICE))
            actual.extend(targets.numpy())
            predicted.extend(outputs.argmax(dim=1).cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="macro", zero_division=0
    )

    result = {
        "seed": seed,
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx)
    }

    print(f"Seed {seed} result: {result}")
    return result


def main():
    print("Loading SmartStrike dataset...")
    X, y, filenames = load_dataset()

    results = [
        run_experiment(seed, X, y)
        for seed in SEEDS
    ]

    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]

    summary = {}
    for metric in metrics:
        values = [r[metric] for r in results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

    output = {
        "experiment": "B5 Robustness Evaluation",
        "description": "Five repeated stratified random recording-level splits",
        "seeds": SEEDS,
        "results": results,
        "summary": summary,
        "limitation": (
            "SmartStrike has no participant IDs. These random recording-level "
            "splits do not establish participant-independent generalization."
        )
    }

    with open(OUTPUT / "b5_robustness_results.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\nB5 SUMMARY")
    print(json.dumps(summary, indent=4))
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()