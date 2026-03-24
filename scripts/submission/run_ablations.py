from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--configs", nargs="+", required=True, help="Ablation config list")
    return parser.parse_args()


def _run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    for cfg in args.configs:
        cfg_path = str((repo_root / cfg).resolve())
        _run([sys.executable, str(repo_root / "scripts/submission/train_main.py"), "--config", cfg_path])
        _run([sys.executable, str(repo_root / "scripts/submission/evaluate_main.py"), "--config", cfg_path])
        _run([sys.executable, str(repo_root / "scripts/submission/plot_main.py"), "--config", cfg_path])

    print("Ablation runs completed")


if __name__ == "__main__":
    main()
