from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from .data import build_dataloaders
from .models import build_model
from .utils import ensure_dir, set_seed


def _epoch_train(model, loader, criterion, optimizer, device):
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


def _epoch_eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds_all.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            labels_all.extend(labels.detach().cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return total_loss / max(1, len(loader)), float(acc)


def train_model(config: dict):
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, class_names = build_dataloaders(
        splits_dir=config["splits_dir"],
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        use_augmentation=bool(config["use_augmentation"]),
    )

    model = build_model(config["model_name"], len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    best_val_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    checkpoint_dir = ensure_dir(config["checkpoint_dir"])
    checkpoint_path = checkpoint_dir / f"{config['run_name']}_best.pth"

    for epoch in range(int(config["epochs"])):
        train_loss, train_acc = _epoch_train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = _epoch_eval(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    history_path = Path(config["results_dir"]) / "tables" / f"{config['run_name']}_history.csv"
    ensure_dir(history_path.parent)
    with history_path.open("w", newline="", encoding="utf-8") as f:
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

    return {
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "class_names": class_names,
        "best_val_acc": float(best_val_acc),
    }
