# Metal Surface Defect Classification - Final Submission Pipeline

A reproducible deep learning pipeline for metal surface defect classification using ResNet18 transfer learning on the NEU surface defect dataset.

## Key Results

- **Baseline Accuracy:** 92.4% (multiclass, 18 classes)
- **Anomaly Detection:** 96.98% binary accuracy (defective vs. normal)
- **Ablations:** 2 studies comparing design choices (low resolution, no augmentation)
- **Reproducibility:** Seed=42, pinned dependencies, config-driven training

## Pipeline Features

1. **Code freeze** via fixed package versions and deterministic random seeds
2. **One-command execution** via `run.sh` (Linux/macOS) or `run.ps1` (Windows)
3. **Automatic artifact generation**:
   - Metrics JSON (accuracy, F1, loss, confusion matrix)
   - Loss/accuracy/confusion matrix plots
   - Error analysis (misclassified images)
   - Per-class slice analysis
   - Anomaly detection metrics (binary classification)
4. **Two ablation studies** validating key design decisions
5. **Compliance audit** (`submission_audit.json`) verifying all deliverables
6. **Conference-style report PDF** with results, ethics, and model card

## Final Submission Structure

Key files and folders for defense:

- `configs/`: experiment configs (`base.yaml`, two ablation configs)
- `src/defect_cv/`: modular training/evaluation/plotting package
- `scripts/submission/`: executable pipeline scripts
- `run.sh`, `run.ps1`: full pipeline entrypoints
- `models/submission/`: trained checkpoints
- `results/tables/`: metrics JSON/CSV and final table
- `results/plots/`: loss curve, accuracy curve, confusion matrix plots
- `results/analysis/`: misclassification and per-class slice reports
- `reports/`: final PDF report

## Installation

1. Create and activate a virtual environment.

PowerShell:

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

## Dataset Assumption

Training scripts expect class-folder splits:

```text
data/splits/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

If your split path differs, update `splits_dir` in config files.

## Reproducible Run

### Full Pipeline (recommended)

Linux/macOS:

```bash
bash run.sh
```

Windows PowerShell:

```powershell
.\run.ps1
```

### Manual Run (step-by-step)

```bash
python scripts/submission/train_main.py --config configs/base.yaml
python scripts/submission/evaluate_main.py --config configs/base.yaml
python scripts/submission/plot_main.py --config configs/base.yaml
python scripts/submission/run_ablations.py --configs configs/ablation_no_augmentation.yaml configs/ablation_low_resolution.yaml
python scripts/submission/build_results_table.py --results-dir results --out-file results/tables/final_results_table.csv
python scripts/analyze_anomaly_metrics.py --metrics-json results/tables/baseline_resnet18_metrics.json --normal-class clean_sample --out-json results/tables/baseline_resnet18_anomaly_metrics.json --out-csv results/tables/baseline_resnet18_anomaly_confusion_matrix.csv
python scripts/submission/write_pipeline_runtime.py --total-seconds 0 --out-file results/tables/pipeline_runtime.json
python scripts/submission/audit_submission.py --results-dir results --runtime-json results/tables/pipeline_runtime.json --out-file results/tables/submission_audit.json
python scripts/submission/generate_final_report_pdf.py --results-dir results --runtime-json results/tables/pipeline_runtime.json --audit-json results/tables/submission_audit.json --output-file reports/final_report.pdf
```

## Checking Training Results

After training completes, view the results using these commands:

### View Baseline Metrics (PowerShell)

```powershell
$PY="C:\NEU Surface Defect Dataset\.venv\Scripts\python.exe"
# View baseline accuracy and F1
& $PY -c "import json; m=json.load(open('results/tables/baseline_resnet18_metrics.json')); print(f'Accuracy: {m.get(\"accuracy\", 0):.1%}\nF1 (weighted): {m.get(\"f1_weighted\", 0):.1%}\nTest Loss: {m.get(\"test_loss\", 0):.4f}')"
```

### View Audit Results (All Systems)

```bash
# Check that all artifacts were generated correctly
cat results/tables/submission_audit.json

# Expected output: {"checks": {"ablations_count_ok": true, "overall_pass": true}}
```

### View Final Results Table

```bash
# Compare baseline vs ablations
cat results/tables/final_results_table.csv
```

### View Anomaly Detection Metrics

```bash
# Binary classification: normal vs defects
cat results/tables/baseline_resnet18_anomaly_metrics.json
```

### Open Reports

- **Full PDF Report:** `reports/final_report.pdf` (includes results, plots, model card, ethics)
- **Markdown Report:** `reports/final_project_report.md` (narrative documentation)

## Image Input Inference

The project supports direct image input prediction (single image or folder), which is required for a vision-based defect detection task.

Single image:

```bash
python scripts/submission/predict_image.py --config configs/base.yaml --image path/to/image.jpg --top-k 3
```

Folder of images:

```bash
python scripts/submission/predict_image.py --config configs/base.yaml --input-dir path/to/images --top-k 3 --output-json results/analysis/batch_predictions.json
```

Notes:

- Uses the trained checkpoint at `models/submission/<run_name>_best.pth` by default.
- Class labels are loaded from `results/tables/<run_name>_train_summary.json` (or metrics JSON if available).
- Pass `--checkpoint` to override the default checkpoint.

## Anomaly Detection Check (Binary: normal vs defect)

You can convert multiclass evaluation into binary anomaly metrics by treating `clean_sample` as normal and all other classes as anomalies.

```bash
python scripts/analyze_anomaly_metrics.py --metrics-json metrics/cnn_baseline_metrics.json --normal-class clean_sample --out-json metrics/anomaly_detection_metrics.json --out-csv metrics/anomaly_detection_confusion_matrix.csv
```

This generates:

- `metrics/anomaly_detection_metrics.json` (precision/recall/F1, macro-F1 for binary, false alarm rate, miss rate)
- `metrics/anomaly_detection_confusion_matrix.csv` (2x2 confusion matrix)

## Ablation Studies Included

1. `configs/ablation_no_augmentation.yaml`
   - Removes augmentation to quantify augmentation benefit.
2. `configs/ablation_low_resolution.yaml`
   - Reduces image size from 224 to 128 to measure resolution sensitivity.

## Output Artifacts for Report/Defense

After running the pipeline:

- Main and ablation metrics: `results/tables/*_metrics.json`
- Confusion matrices (CSV): `results/tables/*_confusion_matrix.csv`
- Combined final table: `results/tables/final_results_table.csv`
- Binary anomaly summary: `results/tables/*_anomaly_metrics.json`
- Training history per run: `results/tables/*_history.csv`
- Loss and accuracy curves: `results/plots/*_loss.png`, `results/plots/*_accuracy.png`
- Confusion matrix figures: `results/plots/*_confusion_matrix.png`
- Error analysis (misclassified samples): `results/analysis/*_misclassified.csv`
- Slice analysis (per-class metrics): `results/analysis/*_slice_per_class.csv`
- Runtime and reproducibility logs: `results/tables/pipeline_runtime.json`, `results/tables/repro_manifest.json`
- Submission checklist audit: `results/tables/submission_audit.json`
- Final report PDF: `reports/final_report.pdf`

## Results Table Format

Template file:

- `results/tables/results_table_template.csv`

Columns:

- `run_name`
- `model_name`
- `seed`
- `test_loss`
- `accuracy`
- `f1_macro`
- `precision_weighted`
- `recall_weighted`
- `f1_weighted`

## Notes on Reproducibility

- Fixed random seed in config (`seed: 42` by default).
- Fixed package versions in `requirements.txt`.
- Configuration-driven runs for baseline and ablations.
- Deterministic split reuse through pre-created `data/splits` folders.
