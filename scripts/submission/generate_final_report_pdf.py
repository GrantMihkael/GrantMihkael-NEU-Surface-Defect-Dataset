from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conference-style final report PDF")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--runtime-json", default="results/tables/pipeline_runtime.json", help="Runtime JSON")
    parser.add_argument("--audit-json", default="results/tables/submission_audit.json", help="Audit JSON")
    parser.add_argument("--output-file", default="reports/final_report.pdf", help="Output PDF path")
    return parser.parse_args()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _wrap(s: str, width: int = 108) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


def _new_page(title: str, body_lines: list[str], pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")

    y = 0.91
    for line in body_lines:
        y -= 0.026
        fig.text(0.08, y, _wrap(line), fontsize=10, va="top")
        if y < 0.08:
            break

    pdf.savefig(fig)
    plt.close(fig)


def _image_page(title: str, image_paths: list[Path], captions: list[str], pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")

    slots = [(0.08, 0.52, 0.84, 0.35), (0.08, 0.1, 0.84, 0.35)]
    for idx, slot in enumerate(slots):
        x, y, w, h = slot
        ax = fig.add_axes([x, y, w, h])
        ax.axis("off")
        if idx < len(image_paths) and image_paths[idx].exists():
            img = mpimg.imread(image_paths[idx])
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Plot not available", ha="center", va="center", fontsize=11)
        if idx < len(captions):
            fig.text(x, y - 0.03, captions[idx], fontsize=9, va="top")

    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables = results_dir / "tables"
    analysis = results_dir / "analysis"
    plots = results_dir / "plots"

    metrics = _load_json(tables / "baseline_resnet18_metrics.json") or {}
    anomaly = _load_json(tables / "baseline_resnet18_anomaly_metrics.json") or {}
    runtime = _load_json(Path(args.runtime_json)) or {}
    audit = _load_json(Path(args.audit_json)) or {}

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        page1 = [
            "Project: Metal Surface Defect Anomaly Detection",
            "Format: Conference-style concise final report.",
            "",
            "Abstract",
            "This report summarizes a reproducible anomaly-detection pipeline for metal surface defects. "
            "A multiclass classifier is trained and then converted into a binary normal-vs-anomaly evaluator.",
            "",
            "1. Problem, Motivation, and Constraints",
            "Motivation: automated inspection support for faster and more consistent surface QA.",
            "Success metrics: multiclass accuracy/macro-F1 plus binary anomaly precision/recall/F1 and false-alarm/miss rates.",
            "Constraints: one-command reproducibility, deterministic seed, <=90 min runtime target, traceable artifacts.",
            "",
            "2. Dataset and Splits",
            "Task: classify defect categories and report anomaly detection quality by treating clean_sample as normal.",
            "Data split: deterministic train/val/test folders under data/splits.",
            "Source groups observed in split metadata: ali2018 and neu_det.",
            "License note: verify upstream dataset terms before redistribution.",
            "",
            "3. Methods and Modeling",
            "Model from scratch: SimpleCNN (small custom convolutional network).",
            "Fine-tuned baseline: ResNet18 transfer learning.",
            "Ablations: no augmentation and low-resolution (128x128).",
            "NLP and RL modules are present as project scaffolds and are not in the deployed vision inference path.",
            "",
            "4. Reproducibility",
            "Code freeze and run configuration are tracked via pinned requirements, config files, and manifest JSON.",
            "Full pipeline can be executed with one command via run.sh or run.ps1.",
        ]
        _new_page("Final Report: Anomaly Detection Pipeline", page1, pdf)

        run_name = metrics.get("run_name", "baseline_resnet18")
        model_name = metrics.get("model_name", "resnet18")
        acc = metrics.get("accuracy")
        f1w = metrics.get("f1_weighted")
        f1m = ((metrics.get("classification_report") or {}).get("macro avg") or {}).get("f1-score")

        anomaly_metrics = anomaly.get("metrics") or {}
        page2 = [
            "5. Training and Compute Details",
            "Config snapshot: seed=42, epochs=10, batch_size=16, lr=0.001, img_size=224, augmentation=true.",
            "Optimization: cross-entropy with Adam optimizer; fixed learning rate schedule.",
            "Environment and commit hash are logged in results/tables/repro_manifest.json.",
            "",
            "6. Core Results",
            f"Run name: {run_name}",
            f"Model: {model_name}",
            f"Multiclass accuracy: {acc}",
            f"Multiclass macro-F1: {f1m}",
            f"Multiclass weighted F1: {f1w}",
            "",
            "7. Anomaly (Binary) Results",
            f"Binary accuracy: {anomaly_metrics.get('accuracy_binary')}",
            f"Anomaly precision: {anomaly_metrics.get('anomaly_precision')}",
            f"Anomaly recall: {anomaly_metrics.get('anomaly_recall')}",
            f"Anomaly F1: {anomaly_metrics.get('anomaly_f1')}",
            f"Binary macro-F1: {anomaly_metrics.get('macro_f1_binary')}",
            f"False alarm rate: {anomaly_metrics.get('false_alarm_rate')}",
            f"Miss rate: {anomaly_metrics.get('miss_rate')}",
            "",
            "8. Runtime and Compliance",
            f"Pipeline runtime (minutes): {runtime.get('total_minutes')}",
            f"Meets <=90 minute requirement: {runtime.get('meets_90_min_requirement')}",
            f"Submission audit pass: {audit.get('overall_pass')}",
        ]
        _new_page("Results and Compliance", page2, pdf)

        _image_page(
            "Figures: Learning Curves",
            [
                plots / "baseline_resnet18_loss.png",
                plots / "baseline_resnet18_accuracy.png",
            ],
            [
                "Figure 1. Baseline training/validation loss curves.",
                "Figure 2. Baseline training/validation accuracy curves.",
            ],
            pdf,
        )

        page4 = [
            "9. Evaluation Extensions",
            "Baselines and ablations are reported through results/tables artifacts.",
            "Current run should include two configured ablations: no augmentation and low resolution.",
            "Calibration analysis is not yet included in this pipeline and should be added if required.",
            "",
            "10. Error and Slice Analysis",
            "Misclassifications file: results/analysis/baseline_resnet18_misclassified.csv",
            "Slice metrics file: results/analysis/baseline_resnet18_slice_per_class.csv",
            "",
            "11. Ablation Summary",
            "Ablation 1: no augmentation (configs/ablation_no_augmentation.yaml)",
            "Ablation 2: low resolution 128x128 (configs/ablation_low_resolution.yaml)",
            "Combined summary table: results/tables/final_results_table.csv",
            "",
            "12. Ethics, Policy, and Model Card Summary",
            "Intended use: industrial QA decision support, not autonomous safety-critical control.",
            "Risks: false alarms increase workload; misses may pass defective material; domain shift may degrade performance.",
            "Mitigations: human-in-the-loop review, periodic revalidation, per-class slice monitoring, drift checks.",
            "Privacy/Fairness: no personal data in scope; monitor minority defect classes for imbalance-driven errors.",
            "Deployment guidance: keep thresholding and alert policy configurable; rerun full evaluation after retraining.",
            "",
            "13. One-Command Reproduction",
            "Windows PowerShell: .\\run.ps1",
            "Linux/macOS: bash run.sh",
            "",
            "14. Conclusion",
            "The pipeline demonstrates a reproducible anomaly-detection workflow with baseline and ablation evidence. "
            "The report artifacts include metrics tables, plots, confusion matrices, and analysis CSV files.",
            "",
            "Appendix: Generated files",
            "- results/tables/repro_manifest.json",
            "- results/tables/pipeline_runtime.json",
            "- results/tables/submission_audit.json",
            "- results/tables/baseline_resnet18_anomaly_metrics.json",
            "- results/tables/baseline_resnet18_anomaly_confusion_matrix.csv",
        ]
        _new_page("Analysis and Conclusion", page4, pdf)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
