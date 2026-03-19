# Model Card - Manufacturing Defect Detection (ML + CNN + Grad-CAM)

Status: Updated for Week 2 checkpoint. Target final version by Week 3.

## Model Details

| Field | Details |
|---|---|
| Model Name | Surface Defect Detection Baselines + Grad-CAM |
| Version | v0.1 (Week 2 checkpoint) |
| Type | Multi-class image classification (steel surface defects) |
| Developed By | 6INTELSY Team (Holy Angel University) |
| Date | 2025-2026 |
| Frameworks | PyTorch (CNN), scikit-learn (ML baseline) |

## Intended Use

- Primary use: offline academic experimentation for steel defect classification.
- Intended users: students and researchers exploring computer vision in quality inspection.
- Decision role: assistive signal for defect triage and analysis.

Out-of-scope:

- Autonomous production pass/fail decisions without human oversight.
- Safety-critical decisions based only on model output.
- Deployment to unseen factories/imaging setups without re-validation.

## Dataset

| Field | Details |
|---|---|
| Source | Combined industrial defect datasets (ali2018 + NEU-DET variants prepared in repo) |
| Raw Data Path | `data/raw/` |
| Cleaned Data Path | `data/cleaned/` |
| Split Data Path | `data/splits/` |
| Data Quality Summary | `outputs/cleaning_summary.json` |
| Split Summary | `outputs/split_summary_table.csv` |
| PII | None expected (industrial image dataset) |

Current split configuration:

- Seed: `42`
- Ratio: train/val/test = `80/10/10`
- Total samples: `3861` (Train `3085`, Val `379`, Test `397`)
- Classes: `17`

## Model Variants

1. ML baseline:
- Script: `scripts/train_ml_baseline.py`
- Model: Logistic Regression on flattened grayscale features (64x64)

2. CNN baseline:
- Script: `scripts/train_cnn_baseline.py`
- Model: SimpleCNN (3 conv blocks + linear head)

3. Explainability companion:
- Script: `gradcam_predict.py`
- Output path: `experiments/gradcam/`

## Metrics (Week 2 Initial Results)

From latest artifacts:

- ML baseline (`metrics/ml_baseline_metrics.json`)
	- Accuracy: `0.5718`
	- Precision (weighted): `0.5810`
	- Recall (weighted): `0.5718`
	- F1 (weighted): `0.5492`

- CNN baseline (`metrics/cnn_baseline_metrics.json`)
	- Accuracy: `0.8212`
	- Precision (weighted): `0.7794`
	- Recall (weighted): `0.8212`
	- F1 (weighted): `0.7863`

Additional evidence:

- Comparison table: `metrics/baseline_comparison.csv`
- CNN curves: `experiments/cnn_learning_curves.png`
- EDA visuals: `outputs/eda_figures/`

## Ethical Considerations

- False negatives in defect inspection can have high operational impact.
- Class imbalance can cause uneven class-wise performance.
- Grad-CAM is supportive evidence only and not causal proof.
- Human-in-the-loop validation is required for high-impact decisions.

See full policy statement: `docs/ethics_statement.md`.

## Caveats and Recommendations

- Performance can degrade under domain shift (lighting, camera, or material variation).
- Minority classes may require augmentation or class-weighted training.
- Threshold and calibration review is needed before practical deployment.
- Failure cases should be logged and included in report updates.

## Stakeholders

- Manufacturing QA engineers
- Production supervisors
- ML engineers and reviewers
- Academic project evaluators

## Planned Week 3 Updates

- Add class-wise precision/recall/F1 summary table directly in this card.
- Add clearer dataset licensing/provenance section.
- Add failure-case examples and corrective actions.
- Add versioned reproducibility block (commands + artifact hashes).
