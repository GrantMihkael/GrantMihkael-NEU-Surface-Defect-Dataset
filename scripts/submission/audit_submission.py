from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit submission artifacts against minimum expectations")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--runtime-json", default="results/tables/pipeline_runtime.json", help="Pipeline runtime JSON path")
    parser.add_argument("--out-file", default="results/tables/submission_audit.json", help="Output audit JSON")
    parser.add_argument("--min-ablations", type=int, default=2, help="Minimum number of ablation runs")
    parser.add_argument("--max-runtime-minutes", type=float, default=90.0, help="Maximum allowed runtime in minutes")
    return parser.parse_args()


def _exists(path: Path) -> bool:
    return path.exists()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables = results_dir / "tables"
    analysis = results_dir / "analysis"
    plots = results_dir / "plots"

    baseline_metrics = tables / "baseline_resnet18_metrics.json"
    baseline_history = tables / "baseline_resnet18_history.csv"
    baseline_cm = tables / "baseline_resnet18_confusion_matrix.csv"
    final_table = tables / "final_results_table.csv"
    anomaly_metrics = tables / "baseline_resnet18_anomaly_metrics.json"
    anomaly_cm = tables / "baseline_resnet18_anomaly_confusion_matrix.csv"

    ablation_metrics = sorted(tables.glob("ablation_*_metrics.json"))

    required_plots = [
        plots / "baseline_resnet18_loss.png",
        plots / "baseline_resnet18_accuracy.png",
        plots / "baseline_resnet18_confusion_matrix.png",
    ]

    required_analysis = [
        analysis / "baseline_resnet18_misclassified.csv",
        analysis / "baseline_resnet18_slice_per_class.csv",
    ]

    runtime_data = _load_json(Path(args.runtime_json))
    runtime_minutes = None
    runtime_pass = False
    if runtime_data is not None:
        runtime_minutes = float(runtime_data.get("total_minutes", 0.0))
        runtime_pass = runtime_minutes <= float(args.max_runtime_minutes)

    checks = {
        "baseline_metrics": _exists(baseline_metrics),
        "baseline_history": _exists(baseline_history),
        "baseline_confusion_matrix": _exists(baseline_cm),
        "final_results_table": _exists(final_table),
        "anomaly_metrics": _exists(anomaly_metrics),
        "anomaly_confusion_matrix": _exists(anomaly_cm),
        "plots_present": all(_exists(p) for p in required_plots),
        "error_and_slice_analysis_present": all(_exists(p) for p in required_analysis),
        "ablations_count_ok": len(ablation_metrics) >= int(args.min_ablations),
        "runtime_90min_ok": runtime_pass,
    }

    payload = {
        "results_dir": str(results_dir.as_posix()),
        "requirements": {
            "min_ablations": int(args.min_ablations),
            "max_runtime_minutes": float(args.max_runtime_minutes),
        },
        "runtime": {
            "runtime_json": str(Path(args.runtime_json).as_posix()),
            "runtime_minutes": runtime_minutes,
            "runtime_limit_minutes": float(args.max_runtime_minutes),
        },
        "artifacts": {
            "baseline_metrics": str(baseline_metrics.as_posix()),
            "baseline_history": str(baseline_history.as_posix()),
            "baseline_confusion_matrix": str(baseline_cm.as_posix()),
            "final_results_table": str(final_table.as_posix()),
            "anomaly_metrics": str(anomaly_metrics.as_posix()),
            "anomaly_confusion_matrix": str(anomaly_cm.as_posix()),
            "ablation_metrics": [str(p.as_posix()) for p in ablation_metrics],
            "plots": [str(p.as_posix()) for p in required_plots],
            "analysis": [str(p.as_posix()) for p in required_analysis],
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
    }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(json.dumps({"overall_pass": payload["overall_pass"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
