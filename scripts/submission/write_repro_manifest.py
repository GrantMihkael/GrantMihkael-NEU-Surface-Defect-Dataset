from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write reproducibility manifest for code freeze runs")
    parser.add_argument("--config", required=True, help="Config file used for the run")
    parser.add_argument(
        "--out-file",
        default="results/tables/repro_manifest.json",
        help="Output manifest path",
    )
    return parser.parse_args()


def _try_git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = _load_config(config_path)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _try_git_head(),
        "config_path": str(config_path.as_posix()),
        "run_name": cfg.get("run_name"),
        "seed": cfg.get("seed"),
        "model_name": cfg.get("model_name"),
        "img_size": cfg.get("img_size"),
        "batch_size": cfg.get("batch_size"),
        "epochs": cfg.get("epochs"),
        "learning_rate": cfg.get("learning_rate"),
        "use_augmentation": cfg.get("use_augmentation"),
        "requirements_file": "requirements.txt",
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
