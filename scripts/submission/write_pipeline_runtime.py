from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write runtime JSON for full pipeline")
    parser.add_argument("--total-seconds", type=float, required=True, help="Total wall-clock runtime in seconds")
    parser.add_argument(
        "--out-file",
        default="results/tables/pipeline_runtime.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_seconds = max(0.0, float(args.total_seconds))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60.0,
        "meets_90_min_requirement": (total_seconds / 60.0) <= 90.0,
        "requirement_limit_minutes": 90.0,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
