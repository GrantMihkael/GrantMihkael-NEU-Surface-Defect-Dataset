import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


IMAGE_SIZE = (64, 64)


def load_split(split_name, splits_dir: Path):
    split_dir = splits_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    class_names = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    features = []
    labels = []

    for class_name in class_names:
        class_dir = split_dir / class_name
        for img_path in class_dir.rglob("*"):
            if not img_path.is_file():
                continue
            with Image.open(img_path) as img:
                gray = img.convert("L").resize(IMAGE_SIZE)
                arr = np.array(gray, dtype=np.float32) / 255.0
            features.append(arr.flatten())
            labels.append(label_to_idx[class_name])

    if not features:
        raise ValueError(f"No images found in split: {split_name}")

    return np.stack(features), np.array(labels), class_names


def append_baseline_comparison(row, metrics_dir: Path):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "baseline_comparison.csv"
    header = "model,accuracy,precision_weighted,recall_weighted,f1_weighted\n"

    if not csv_path.exists():
        csv_path.write_text(header, encoding="utf-8")

    line = (
        f"{row['model']},{row['accuracy']:.4f},{row['precision_weighted']:.4f},"
        f"{row['recall_weighted']:.4f},{row['f1_weighted']:.4f}\n"
    )
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Train simple ML baseline")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-name", default="ml_baseline_metrics.json")
    parser.add_argument("--cm-name", default="ml_baseline_confusion_matrix.csv")
    parser.add_argument("--model-name", default="LogisticRegression(flattened_grayscale)")
    return parser.parse_args()


def main():
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    metrics_dir = Path(args.metrics_dir)

    x_train, y_train, class_names = load_split("train", splits_dir)
    x_test, y_test, _ = load_split("test", splits_dir)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    random_state=args.seed,
                    n_jobs=1,
                ),
            ),
        ]
    )

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    metrics = {
        "model": args.model_name,
        "seed": args.seed,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "class_names": class_names,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / args.metrics_name).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    cm_path = metrics_dir / args.cm_name
    cm = np.array(metrics["confusion_matrix"])
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d")

    append_baseline_comparison(metrics, metrics_dir)

    print("ML baseline training complete")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"classification_report", "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
