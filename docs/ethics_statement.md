# Ethics & Policy Statement — Manufacturing Defect Detection

Status: Draft (Week 1) — to be expanded to full 1–2 page statement by Week 3

## 1. Intended Use & Limitations
This system is designed for academic research on steel defect classification using computer vision. It is not intended for unsupervised production deployment or as the sole basis for safety-critical decisions.

## 2. Ethics Risk Register (Top 3 Risks)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | False negatives (missed defects) | Medium | High | Prioritize recall, tune thresholds, and require human verification on uncertain predictions |
| 2 | Class imbalance and unfair performance across defect classes | Medium | High | Report per-class metrics, use class balancing/augmentation, and track subgroup performance |
| 3 | Misinterpretation of Grad-CAM heatmaps | Medium | Medium | Document explainability limits; treat Grad-CAM as supportive evidence only |

## 3. Fairness Checks (Planned)
- Compare class-wise precision/recall/F1, not just aggregate accuracy.
- Inspect confusion matrix to detect consistently underperforming classes.
- Track error slices by imaging conditions (if metadata is available).

## 4. Privacy & Consent
- Uses public research dataset; no personal user data collection in this repo.
- No hidden tracking or behavioral data collection is performed.
- Dataset source and license details are cited in project documentation.

## 5. Misuse Considerations
- Must not be used to bypass existing industrial quality assurance and safety procedures.
- Must not be marketed as guaranteed defect detection beyond validated test conditions.

## 6. Transparency
- Architecture, training setup, metrics, and known limitations are documented in this repository.
- Failure cases and corrective actions will be summarized in final report artifacts.
