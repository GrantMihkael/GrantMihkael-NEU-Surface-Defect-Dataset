from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import json

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

plot_mod = importlib.import_module("defect_cv.plotting")
plot_confusion_matrix = plot_mod.plot_confusion_matrix
plot_history = plot_mod.plot_history
utils_mod = importlib.import_module("defect_cv.utils")
ensure_dir = utils_mod.ensure_dir
load_config = utils_mod.load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training and evaluation plots")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    results_dir = Path(config["results_dir"])
    tables_dir = results_dir / "tables"
    plots_dir = ensure_dir(results_dir / "plots")

    history_csv = tables_dir / f"{config['run_name']}_history.csv"
    cm_csv = tables_dir / f"{config['run_name']}_confusion_matrix.csv"
    metrics_json = tables_dir / f"{config['run_name']}_metrics.json"

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    class_names = metrics["class_names"]

    plot_history(str(history_csv), str(plots_dir / config["run_name"]))
    plot_confusion_matrix(str(cm_csv), class_names=class_names, out_file=str(plots_dir / f"{config['run_name']}_confusion_matrix.png"))

    print("Plots generated")


if __name__ == "__main__":
    main()
