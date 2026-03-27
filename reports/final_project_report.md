# Final Project Report

## Title
Steel Surface Defect Anomaly Detection with Reproducible CNN Pipeline

## 1. Problem and Motivation
Industrial steel surface inspection is quality-critical, but manual inspection is slow and inconsistent at scale. This project targets automated defect recognition and anomaly-oriented decision support: detect whether a sample is normal or defective, while also preserving multi-class defect predictions for analysis and process diagnostics.

### Objective
Build a reproducible vision pipeline that:
1. Trains a defect classifier on class-folder image splits.
2. Converts multi-class predictions into binary anomaly decisions (normal vs anomaly).
3. Produces submission-ready artifacts: metrics tables, plots, confusion matrices, error/slice analysis, runtime audit, and report.

### Success Metrics and Constraints
Primary success metrics:
1. Multi-class: accuracy and macro-F1.
2. Binary anomaly: anomaly precision, recall, F1, macro-F1 (binary), false alarm rate, miss rate.

Project constraints:
1. One-command reproducibility.
2. Deterministic seed-based training/evaluation configuration.
3. End-to-end runtime target <= 90 minutes in configured environment.
4. Traceable artifacts for defense and auditing.

## 2. Dataset, Preprocessing, and Splits
### Dataset Source
This workspace combines classes from two sources tracked in split metadata:
1. ali2018
2. neu_det

See split provenance in [metrics/split_summary_combined.json](metrics/split_summary_combined.json).

### License and Collection Notes
License and usage rights must be validated against the original upstream dataset pages before public redistribution. This repository stores processed data folders and expects local split folders for training. No personal data is used in this project context.

### Preprocessing
Image preprocessing pipeline:
1. Convert to 3-channel grayscale representation.
2. Resize to configured image size.
3. Optional augmentation for training (horizontal flip, small rotation).
4. Normalize tensors with fixed channel statistics.

Implemented in [src/defect_cv/data.py](src/defect_cv/data.py).

### Data Splits
The working split configuration is deterministic and class-folder based.

Expected structure:
1. data/splits/train/<class_name>/*.jpg
2. data/splits/val/<class_name>/*.jpg
3. data/splits/test/<class_name>/*.jpg

Current split summary (combined reference):
1. Train: 3085
2. Val: 379
3. Test: 397

See [metrics/split_summary_combined.json](metrics/split_summary_combined.json).

## 3. Training Details, Compute, and Reproducibility
### Baseline Configuration
From [configs/base.yaml](configs/base.yaml):
1. Model: resnet18
2. Seed: 42
3. Image size: 224
4. Batch size: 16
5. Epochs: 10
6. Learning rate: 0.001
7. Workers: 0
8. Augmentation: true

### Optimization and Schedule
1. Loss: cross-entropy.
2. Optimizer: Adam.
3. Learning-rate schedule: fixed LR (no scheduler in current baseline).

Training/evaluation logic is in [src/defect_cv/trainer.py](src/defect_cv/trainer.py) and [src/defect_cv/evaluation.py](src/defect_cv/evaluation.py).

### Compute Environment
From [results/tables/repro_manifest.json](results/tables/repro_manifest.json):
1. Python: 3.14.3
2. Platform: Windows 10
3. Git commit recorded for run traceability.

Pinned dependencies are in [requirements.txt](requirements.txt).

### Reproducibility Steps
1. Fixed random seed in config and code.
2. Pinned package versions.
3. Config-driven scripts for train/eval/plot/ablation.
4. One-command run scripts on Windows/Linux.
5. Repro manifest and runtime JSON outputs.

Run entrypoints:
1. [run.ps1](run.ps1)
2. [run.sh](run.sh)

## 4. Modeling: Architectures and System Connections
### CNN Components
Implemented architectures in [src/defect_cv/models.py](src/defect_cv/models.py):
1. SimpleCNN (from-scratch compact CNN).
2. ResNet18 transfer learning baseline.

Rationale:
1. SimpleCNN demonstrates core fundamentals and an interpretable lightweight baseline.
2. ResNet18 provides stronger representation quality under limited project time.

### Anomaly Conversion Layer
After multi-class evaluation, predictions are collapsed into binary labels:
1. Normal class: clean_sample.
2. Anomaly classes: all remaining defect classes.

Implemented in [scripts/analyze_anomaly_metrics.py](scripts/analyze_anomaly_metrics.py).

### NLP and RL Components
NLP and RL are included as scaffolds/prototypes, not as production-connected blocks in the final vision inference path:
1. NLP prototype: [nlp/prototype.py](nlp/prototype.py), output in [metrics/nlp_prototype_output.json](metrics/nlp_prototype_output.json).
2. RL stub: [rl/run_stub_experiment.py](rl/run_stub_experiment.py), output in [metrics/rl_stub_metrics.json](metrics/rl_stub_metrics.json).

Connection note:
1. Current deployment path is vision-only (CNN + anomaly conversion).
2. NLP/RL modules are exploratory extensions and do not affect core anomaly results.

## 5. Evaluation: Metrics, Baselines, Ablations, and Analysis
### Baseline Multi-class Results
From [results/tables/final_results_table.csv](results/tables/final_results_table.csv):
1. Test loss: 0.2137
2. Accuracy: 0.9244
3. Macro-F1: 0.8890
4. Weighted F1: 0.9260

Detailed report and confusion matrix in:
1. [results/tables/baseline_resnet18_metrics.json](results/tables/baseline_resnet18_metrics.json)
2. [results/tables/baseline_resnet18_confusion_matrix.csv](results/tables/baseline_resnet18_confusion_matrix.csv)

### Binary Anomaly Results
From [results/tables/baseline_resnet18_anomaly_metrics.json](results/tables/baseline_resnet18_anomaly_metrics.json):
1. Binary accuracy: 0.9698
2. Anomaly precision: 0.9638
3. Anomaly recall: 0.9966
4. Anomaly F1: 0.9799
5. Binary macro-F1: 0.9594
6. False alarm rate: 0.1068
7. Miss rate: 0.0034

Binary confusion matrix in [results/tables/baseline_resnet18_anomaly_confusion_matrix.csv](results/tables/baseline_resnet18_anomaly_confusion_matrix.csv).

### Baselines
Available baseline artifacts include:
1. CNN baseline metrics in [metrics/cnn_baseline_metrics.json](metrics/cnn_baseline_metrics.json).
2. Classical ML baseline metrics in [metrics/ml_baseline_metrics.json](metrics/ml_baseline_metrics.json).

### Ablations (>=2)
Configured ablations:
1. No augmentation: [configs/ablation_no_augmentation.yaml](configs/ablation_no_augmentation.yaml).
2. Low resolution 128x128: [configs/ablation_low_resolution.yaml](configs/ablation_low_resolution.yaml).

Current audit status indicates ablation artifacts are not yet present in results/tables for this run. See [results/tables/submission_audit.json](results/tables/submission_audit.json) where ablations_count_ok is false.

### Error and Slice Analysis
Artifacts generated:
1. Misclassified samples: [results/analysis/baseline_resnet18_misclassified.csv](results/analysis/baseline_resnet18_misclassified.csv).
2. Per-class slice metrics: [results/analysis/baseline_resnet18_slice_per_class.csv](results/analysis/baseline_resnet18_slice_per_class.csv).

### Calibration
Calibration (e.g., ECE/reliability plots) is not yet included in the current pipeline outputs.

## 6. Ethics and Policy
### Intended Use
1. Assistance for industrial QA triage and defect screening.
2. Support tool for operators, not an autonomous safety-critical controller.

### Risks
1. False positives may increase inspection workload.
2. False negatives may pass defective material.
3. Domain shift risk across camera setups, lighting, and surface finish.
4. Class imbalance can reduce minority defect reliability.

### Mitigations
1. Report both false alarm and miss rates in anomaly mode.
2. Maintain per-class slice analysis to monitor weak classes.
3. Keep human-in-the-loop review for uncertain or high-cost decisions.
4. Revalidate model after distribution shift or hardware changes.

### Privacy and Fairness
1. No personal data is processed.
2. Fairness here is class-coverage fairness: monitor minority-class performance and rebalance data if needed.

### Limitations
1. Performance is tied to available split composition and labeling quality.
2. Current pipeline does not include uncertainty calibration thresholds for abstention.

## 7. Model Card Summary
### Model Details
1. Name: baseline_resnet18.
2. Type: image classifier (multi-class) with binary anomaly post-processing.
3. Framework: PyTorch.

### Inputs and Outputs
1. Input: grayscale-like metal surface images transformed to 3 channels.
2. Output A: multi-class probabilities over defect labels.
3. Output B: binary anomaly decision using clean_sample as normal class.

### Performance Snapshot
1. Multi-class accuracy: 0.9244.
2. Multi-class macro-F1: 0.8890.
3. Binary anomaly F1: 0.9799.
4. Binary macro-F1: 0.9594.

### Deployment Guidance and Disclaimers
1. Use as decision support, not sole acceptance gate without human QA.
2. Track drift with periodic re-evaluation on latest production samples.
3. Keep thresholds and alert rules configurable per production tolerance.
4. Re-run full evaluation after any retraining or data pipeline change.

## 8. How to Run (One-Command Reproduction)
### Windows (recommended in this workspace)
1. Open PowerShell in repository root.
2. Activate venv and run:

```powershell
& ".\.venv\Scripts\Activate.ps1"
.\run.ps1
```

### Linux/macOS

```bash
bash run.sh
```

### Expected Outputs
Primary outputs after run:
1. Metrics tables: [results/tables](results/tables)
2. Plots: [results/plots](results/plots)
3. Error/slice analysis: [results/analysis](results/analysis)
4. Runtime/audit logs: [results/tables/pipeline_runtime.json](results/tables/pipeline_runtime.json), [results/tables/submission_audit.json](results/tables/submission_audit.json)
5. PDF report: [reports/final_report.pdf](reports/final_report.pdf)

## 9. Compliance Checklist Status (Current)
From [results/tables/submission_audit.json](results/tables/submission_audit.json):
1. Baseline metrics/history/plots/analysis: pass.
2. Anomaly metrics + confusion matrix: pass.
3. Runtime <= 90 minutes check field: pass flag present.
4. Ablation count >= 2: not yet passed for current artifact set.

Final action to fully satisfy the checklist:
1. Complete ablation runs through [run.ps1](run.ps1) or [run.sh](run.sh) and confirm updated audit shows overall_pass true.
