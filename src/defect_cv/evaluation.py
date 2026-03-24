from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .data import build_dataloaders
from .models import build_model
from .utils import ensure_dir, save_json


def _false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true != y_pred))


def _predict(model, loader, device):
    model.eval()
    y_true, y_pred, y_paths = [], [], []
    with torch.no_grad():
        for batch in loader:
            try:
                images, labels, paths = batch
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.numpy())
                y_paths.extend(paths)
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
                print(f"Skipping unreadable batch due to image error: {err}")
                continue
    return np.array(y_true), np.array(y_pred), y_paths


def evaluate_model(config: dict, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_names = build_dataloaders(
        splits_dir=config["splits_dir"],
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        use_augmentation=False,
    )

    model = build_model(config["model_name"], len(class_names)).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
    test_loss = test_loss / max(1, len(test_loader))

    y_true, y_pred, y_paths = _predict(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "run_name": config["run_name"],
        "model_name": config["model_name"],
        "seed": int(config["seed"]),
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "false_alarm_rate": _false_alarm_rate(y_true, y_pred),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    tables_dir = ensure_dir(Path(config["results_dir"]) / "tables")
    analysis_dir = ensure_dir(Path(config["results_dir"]) / "analysis")

    save_json(metrics, tables_dir / f"{config['run_name']}_metrics.json")
    np.savetxt(tables_dir / f"{config['run_name']}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    mis_idx = np.where(y_true != y_pred)[0]
    mis_rows = [
        {
            "image_path": y_paths[i],
            "true_class": class_names[int(y_true[i])],
            "pred_class": class_names[int(y_pred[i])],
        }
        for i in mis_idx
    ]
    pd.DataFrame(mis_rows).to_csv(analysis_dir / f"{config['run_name']}_misclassified.csv", index=False)

    per_class_rows = []
    for class_name in class_names:
        class_stats = report.get(class_name, {})
        per_class_rows.append(
            {
                "class_name": class_name,
                "precision": class_stats.get("precision", 0.0),
                "recall": class_stats.get("recall", 0.0),
                "f1-score": class_stats.get("f1-score", 0.0),
                "support": class_stats.get("support", 0),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(analysis_dir / f"{config['run_name']}_slice_per_class.csv", index=False)

    return metrics
