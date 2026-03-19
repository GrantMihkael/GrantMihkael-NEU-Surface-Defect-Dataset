import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    preds_all = []
    labels_all = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds_all.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return total_loss / max(1, len(loader)), float(acc)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds_all.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            labels_all.extend(labels.detach().cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return total_loss / max(1, len(loader)), float(acc), np.array(labels_all), np.array(preds_all)


def append_baseline_comparison(metrics, metrics_dir: Path):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "baseline_comparison.csv"
    header = "model,accuracy,precision_weighted,recall_weighted,f1_weighted\n"

    if not csv_path.exists():
        csv_path.write_text(header, encoding="utf-8")

    line = (
        f"{metrics['model']},{metrics['accuracy']:.4f},{metrics['precision_weighted']:.4f},"
        f"{metrics['recall_weighted']:.4f},{metrics['f1_weighted']:.4f}\n"
    )
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(line)


def save_curves(history, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train simple CNN baseline")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits_dir = Path(args.splits_dir)
    metrics_dir = Path(args.metrics_dir)
    experiments_dir = Path(args.experiments_dir)
    models_dir = Path(args.models_dir)

    train_dir = splits_dir / "train"
    val_dir = splits_dir / "val"
    test_dir = splits_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected split folders in data/splits/{train,val,test}")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    test_ds = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), models_dir / "cnn_baseline_best.pth")

    model.load_state_dict(torch.load(models_dir / "cnn_baseline_best.pth", map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "model": "SimpleCNN",
        "seed": args.seed,
        "epochs": args.epochs,
        "test_loss": float(test_loss),
        "accuracy": float(test_acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "class_names": train_ds.classes,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=train_ds.classes,
            output_dict=True,
            zero_division=0,
        ),
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "cnn_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    history_csv = experiments_dir / "cnn_history.csv"
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        for i in range(len(history["train_loss"])):
            writer.writerow(
                {
                    "epoch": i + 1,
                    "train_loss": history["train_loss"][i],
                    "val_loss": history["val_loss"][i],
                    "train_acc": history["train_acc"][i],
                    "val_acc": history["val_acc"][i],
                }
            )

    np.savetxt(
        metrics_dir / "cnn_baseline_confusion_matrix.csv",
        np.array(metrics["confusion_matrix"]),
        delimiter=",",
        fmt="%d",
    )

    save_curves(history, experiments_dir / "cnn_learning_curves.png")
    append_baseline_comparison(metrics, metrics_dir)

    print("CNN baseline training complete")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"classification_report", "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
