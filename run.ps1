$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD/src"
$PY = Join-Path $PWD ".venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
	throw "Python interpreter not found at $PY. Create/activate your .venv first."
}

$start = Get-Date

& $PY scripts/submission/write_repro_manifest.py --config configs/base.yaml --out-file results/tables/repro_manifest.json

& $PY scripts/submission/train_main.py --config configs/base.yaml
& $PY scripts/submission/evaluate_main.py --config configs/base.yaml
& $PY scripts/submission/plot_main.py --config configs/base.yaml

& $PY scripts/submission/run_ablations.py --configs configs/ablation_no_augmentation.yaml configs/ablation_low_resolution.yaml

& $PY scripts/submission/build_results_table.py --results-dir results --out-file results/tables/final_results_table.csv

& $PY scripts/analyze_anomaly_metrics.py --metrics-json results/tables/baseline_resnet18_metrics.json --normal-class clean_sample --out-json results/tables/baseline_resnet18_anomaly_metrics.json --out-csv results/tables/baseline_resnet18_anomaly_confusion_matrix.csv

$end = Get-Date
$seconds = [int]($end - $start).TotalSeconds
& $PY scripts/submission/write_pipeline_runtime.py --total-seconds $seconds --out-file results/tables/pipeline_runtime.json

& $PY scripts/submission/audit_submission.py --results-dir results --runtime-json results/tables/pipeline_runtime.json --out-file results/tables/submission_audit.json

& $PY scripts/submission/generate_final_report_pdf.py --results-dir results --runtime-json results/tables/pipeline_runtime.json --audit-json results/tables/submission_audit.json --output-file reports/final_report.pdf

Write-Host "Pipeline complete. Check results/tables, results/plots, and results/analysis."
