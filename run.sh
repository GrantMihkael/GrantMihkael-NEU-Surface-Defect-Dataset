#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

START_EPOCH=$(date +%s)
mkdir -p results/tables

python scripts/submission/write_repro_manifest.py --config configs/base.yaml --out-file results/tables/repro_manifest.json

python scripts/submission/train_main.py --config configs/base.yaml
python scripts/submission/evaluate_main.py --config configs/base.yaml
python scripts/submission/plot_main.py --config configs/base.yaml

python scripts/submission/run_ablations.py \
  --configs configs/ablation_no_augmentation.yaml configs/ablation_low_resolution.yaml

python scripts/submission/build_results_table.py --results-dir results --out-file results/tables/final_results_table.csv

python scripts/analyze_anomaly_metrics.py \
  --metrics-json results/tables/baseline_resnet18_metrics.json \
  --normal-class clean_sample \
  --out-json results/tables/baseline_resnet18_anomaly_metrics.json \
  --out-csv results/tables/baseline_resnet18_anomaly_confusion_matrix.csv

END_EPOCH=$(date +%s)
TOTAL_SECONDS=$((END_EPOCH - START_EPOCH))
python scripts/submission/write_pipeline_runtime.py --total-seconds "$TOTAL_SECONDS" --out-file results/tables/pipeline_runtime.json

python scripts/submission/audit_submission.py \
  --results-dir results \
  --runtime-json results/tables/pipeline_runtime.json \
  --out-file results/tables/submission_audit.json

python scripts/submission/generate_final_report_pdf.py \
  --results-dir results \
  --runtime-json results/tables/pipeline_runtime.json \
  --audit-json results/tables/submission_audit.json \
  --output-file reports/final_report.pdf

echo "Pipeline complete. Check results/tables, results/plots, and results/analysis."
