# Model Card — Manufacturing Defect Detection (CNN + Grad-CAM)

Status: Draft (Week 1) — to be completed by Week 3

## Model Details

| Field | Details |
|---|---|
| Model Name | Manufacturing Defect Classifier + Grad-CAM |
| Version | v0.1 (Proposal) |
| Type | Image Classification (Steel Surface Defects) |
| Developed By | 6INTELSY Team (Holy Angel University) |
| Date | 2025–2026 |
| Framework | PyTorch |

## Intended Use
- Primary use: Offline academic experimentation for steel defect classification.
- Intended users: Students and researchers studying computer vision for quality inspection.
- Out-of-scope: Not intended for direct production deployment or autonomous safety-critical decisions.

## Dataset

| Field | Details |
|---|---|
| Source | NEU Surface Defect Database (Kaggle mirror used for access) |
| License | To be verified and documented in final report |
| PII | None expected (industrial image dataset) |
| Splits | Planned: 70% train / 15% val / 15% test |
| Classes | Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches |

## Metrics
- Primary: Macro-F1, Accuracy
- Supporting: Confusion Matrix, False Alarm Rate
- Planned extension (if segmentation/anomaly path is used): mIoU

## Ethical Considerations
- Risk of false negatives in inspection flow; predictions require human validation.
- Potential class imbalance and uneven performance across defect categories.
- Explainability outputs (Grad-CAM) are supportive signals, not causal proof.

## Caveats and Recommendations
- Results are expected to vary with domain shift (lighting, camera, material changes).
- The model should be calibrated and validated on target-factory data before any practical use.
- Misclassifications and failure cases must be tracked in final evaluation.
