# Ethics and Policy Statement - Manufacturing Defect Detection

Status: Expanded from Week 1 draft for Week 2 checkpoint. Planned to be finalized as a full 1-2 page statement by Week 3.

## 1. Intended Use and Limitations

This system is designed for academic research on steel surface defect classification using computer vision. It is intended as decision support for experimentation and analysis, not as a fully autonomous industrial acceptance/rejection authority.

Out-of-scope use includes:

- Unsupervised production deployment
- Sole reliance for safety-critical quality decisions
- Use outside validated imaging and material conditions without re-evaluation

## 2. Ethics Risk Register (Top 3 Risks)

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | False negatives (missed defects) | Medium | High | Prioritize recall, tune thresholds, and require human verification on uncertain predictions |
| 2 | Class imbalance and unfair performance across defect classes | Medium | High | Report per-class metrics, use class balancing/augmentation, and track subgroup performance |
| 3 | Misinterpretation of Grad-CAM heatmaps | Medium | Medium | Document explainability limits and treat Grad-CAM as supportive evidence only |

## 3. Fairness and Reliability Checks

Current and planned checks:

- Compare class-wise precision/recall/F1, not just aggregate accuracy
- Inspect confusion matrix to detect consistently underperforming classes
- Monitor recall for minority classes and track imbalance effects over runs
- Track error slices by imaging conditions when metadata is available
- Re-check behavior after preprocessing or split changes

## 4. Dataset Governance

- Dataset provenance is documented and licensing is verified before release use
- Only approved, non-personal industrial image data is included
- Data cleaning removes invalid or duplicate records to reduce quality artifacts
- Split generation is reproducible (fixed seed) to support auditability

## 5. Privacy and Consent

- No personal data collection is expected in this repository
- No hidden tracking or behavioral user telemetry is implemented
- Repository contributors must avoid committing sensitive operational metadata
- Dataset source and license notes are documented in project files

## 6. Misuse and Safety Considerations

- The model must not bypass existing industrial QA and safety procedures
- The model must not be marketed as guaranteed defect detection beyond validated conditions
- Outputs must not be interpreted as final pass/fail decisions without human review
- Deployment in new domains requires fresh validation and risk review

## 7. Human Oversight and Accountability

- Human-in-the-loop review is required for final accept/reject decisions
- Uncertain, borderline, or high-impact cases should be escalated to human inspectors
- Model performance monitoring should be part of routine QA governance
- Project decisions should be jointly reviewed by both teammates before milestone submission

## 8. Transparency and Documentation Commitments

- Architecture, training setup, metrics, and limitations are documented in this repository
- Failure cases and corrective actions are summarized in checkpoint/final reports
- Explainability outputs (for example Grad-CAM) are presented with caveats
- Changes in data pipeline or model settings should be logged with reproducible commands

## 9. Ongoing Actions

- Continue auditing class imbalance and failure modes during each training cycle
- Expand fairness checks with stronger subgroup analysis when metadata is available
- Update this statement as experiments evolve and deployment assumptions change
