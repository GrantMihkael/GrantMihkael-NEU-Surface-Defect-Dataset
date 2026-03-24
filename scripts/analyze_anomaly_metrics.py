import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute binary anomaly-vs-normal metrics from multiclass confusion matrix"
    )
    parser.add_argument(
        "--metrics-json",
        default="metrics/cnn_baseline_metrics.json",
        help="Path to multiclass metrics JSON containing class_names and confusion_matrix",
    )
    parser.add_argument(
        "--normal-class",
        default="clean_sample",
        help="Class name considered normal (all other classes are anomalies)",
    )
    parser.add_argument(
        "--out-json",
        default="metrics/anomaly_detection_metrics.json",
        help="Output JSON path for binary anomaly metrics",
    )
    parser.add_argument(
        "--out-csv",
        default="metrics/anomaly_detection_confusion_matrix.csv",
        help="Output CSV path for 2x2 confusion matrix",
    )
    return parser.parse_args()


def load_multiclass_metrics(path: Path) -> tuple[list[str], np.ndarray, str | None]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    class_names = payload.get("class_names")
    cm = payload.get("confusion_matrix")
    model_name = payload.get("model")

    if not isinstance(class_names, list) or not class_names:
        raise ValueError("metrics JSON must include a non-empty class_names list")
    if not isinstance(cm, list) or not cm:
        raise ValueError("metrics JSON must include a non-empty confusion_matrix")

    matrix = np.array(cm, dtype=np.int64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be square")
    if matrix.shape[0] != len(class_names):
        raise ValueError("confusion_matrix size must match class_names length")

    return class_names, matrix, model_name


def to_binary_confusion(class_names: list[str], cm: np.ndarray, normal_class: str) -> dict[str, int]:
    if normal_class not in class_names:
        raise ValueError(
            f"normal class '{normal_class}' not found in class_names: {class_names}"
        )

    normal_idx = class_names.index(normal_class)

    # Rows are true labels, columns are predicted labels.
    tn = int(cm[normal_idx, normal_idx])
    fp = int(cm[normal_idx, :].sum() - tn)
    fn = int(cm[:, normal_idx].sum() - tn)
    tp = int(cm.sum() - tn - fp - fn)

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def compute_binary_metrics(binary_cm: dict[str, int]) -> dict[str, float | int]:
    tn = binary_cm["tn"]
    fp = binary_cm["fp"]
    fn = binary_cm["fn"]
    tp = binary_cm["tp"]

    total = tn + fp + fn + tp

    accuracy = _safe_div(tp + tn, total)
    anomaly_precision = _safe_div(tp, tp + fp)
    anomaly_recall = _safe_div(tp, tp + fn)
    normal_precision = _safe_div(tn, tn + fn)
    normal_recall = _safe_div(tn, tn + fp)
    specificity = _safe_div(tn, tn + fp)
    false_alarm_rate = _safe_div(fp, fp + tn)
    miss_rate = _safe_div(fn, fn + tp)
    f1_anomaly = _safe_div(2 * anomaly_precision * anomaly_recall, anomaly_precision + anomaly_recall)
    f1_normal = _safe_div(2 * normal_precision * normal_recall, normal_precision + normal_recall)
    macro_f1_binary = (f1_normal + f1_anomaly) / 2.0
    balanced_accuracy = (normal_recall + anomaly_recall) / 2.0

    return {
        "total_samples": int(total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_binary": accuracy,
        "anomaly_precision": anomaly_precision,
        "anomaly_recall": anomaly_recall,
        "anomaly_f1": f1_anomaly,
        "normal_precision": normal_precision,
        "normal_recall": normal_recall,
        "normal_f1": f1_normal,
        "macro_f1_binary": macro_f1_binary,
        "balanced_accuracy": balanced_accuracy,
        "specificity_normal": specificity,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
    }


def save_binary_confusion_csv(path: Path, binary_cm: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "pred_normal", "pred_anomaly"])
        writer.writerow(["true_normal", binary_cm["tn"], binary_cm["fp"]])
        writer.writerow(["true_anomaly", binary_cm["fn"], binary_cm["tp"]])


def main() -> None:
    args = parse_args()

    metrics_path = Path(args.metrics_json)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    class_names, multiclass_cm, model_name = load_multiclass_metrics(metrics_path)
    binary_cm = to_binary_confusion(class_names, multiclass_cm, args.normal_class)
    scores = compute_binary_metrics(binary_cm)

    output = {
        "source_metrics_json": str(metrics_path.as_posix()),
        "model": model_name,
        "normal_class": args.normal_class,
        "anomaly_classes": [c for c in class_names if c != args.normal_class],
        "binary_confusion_matrix": {
            "layout": "rows=true labels [normal, anomaly], cols=pred labels [normal, anomaly]",
            "matrix": [
                [binary_cm["tn"], binary_cm["fp"]],
                [binary_cm["fn"], binary_cm["tp"]],
            ],
        },
        "metrics": scores,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    save_binary_confusion_csv(out_csv, binary_cm)

    print("Anomaly metrics computed")
    print(json.dumps(output["metrics"], indent=2))
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
