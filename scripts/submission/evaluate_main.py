from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

evaluate_model = importlib.import_module("defect_cv.evaluation").evaluate_model
load_config = importlib.import_module("defect_cv.utils").load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate defect classifier")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--checkpoint", required=False, help="Optional checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = str(Path(config["checkpoint_dir"]) / f"{config['run_name']}_best.pth")

    metrics = evaluate_model(config, checkpoint_path=checkpoint)
    print("Evaluation complete")
    print({k: v for k, v in metrics.items() if k not in {"classification_report", "confusion_matrix", "class_names"}})


if __name__ == "__main__":
    main()
