# Model Card (Draft)

## Model Name
- Surface Defect Detection Baselines (ML + CNN)

## Intended Use
- Assist visual quality inspection of steel surface defects.
- Prioritize suspect samples for human review.

## Out-of-Scope Use
- Fully autonomous reject/accept decisions without human oversight.
- Use on domains outside trained steel defect imagery without validation.

## Data
- Source: NEU Surface Defect Dataset (to be documented with exact acquisition link).
- Current pipeline: `data/raw` -> `scripts/clean_dataset.py` -> `data/cleaned` -> `scripts/create_splits.py` -> `data/splits`.

## Metrics (Initial)
- ML baseline metrics: `metrics/ml_baseline_metrics.json`
- CNN baseline metrics: `metrics/cnn_baseline_metrics.json`
- Comparison table: `metrics/baseline_comparison.csv`

## Limitations
- Performance may degrade on unseen imaging conditions.
- Potential class imbalance effects.
- Early experiments may be sensitive to split seed and preprocessing choices.

## Ethical Considerations
- Risk of automation bias in QA workflows.
- Misclassification may cause production or safety impacts.
- Human review required for high-stakes decisions.

## Stakeholders
- Manufacturing QA engineers
- Production supervisors
- ML engineers and auditors
