from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build consolidated results table")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-file", default="results/tables/final_results_table.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    tables_dir = Path(args.results_dir) / "tables"
    rows = []

    for p in sorted(tables_dir.glob("*_metrics.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        accuracy = data.get("accuracy")
        report = data.get("classification_report") or {}
        macro_f1 = (report.get("macro avg") or {}).get("f1-score")
        false_alarm_rate = data.get("false_alarm_rate")
        if false_alarm_rate is None and accuracy is not None:
            false_alarm_rate = 1.0 - float(accuracy)

        rows.append(
            {
                "run_name": data.get("run_name"),
                "model_name": data.get("model_name"),
                "seed": data.get("seed"),
                "test_loss": data.get("test_loss"),
                "accuracy": accuracy,
                "f1_macro": macro_f1,
                "false_alarm_rate": false_alarm_rate,
                "precision_weighted": data.get("precision_weighted"),
                "recall_weighted": data.get("recall_weighted"),
                "f1_weighted": data.get("f1_weighted"),
            }
        )

    if not rows:
        raise FileNotFoundError("No *_metrics.json files found. Run training/evaluation first.")

    df = pd.DataFrame(rows).sort_values(by=["accuracy", "f1_weighted"], ascending=False)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
