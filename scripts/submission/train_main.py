from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

train_model = importlib.import_module("defect_cv.trainer").train_model
utils_mod = importlib.import_module("defect_cv.utils")
ensure_dir = utils_mod.ensure_dir
load_config = utils_mod.load_config
save_json = utils_mod.save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train defect classifier")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    run_out = ensure_dir(Path(config["results_dir"]) / "tables")
    train_out = train_model(config)
    save_json(train_out, run_out / f"{config['run_name']}_train_summary.json")

    print("Training complete")
    print(train_out)


if __name__ == "__main__":
    main()
