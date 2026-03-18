# Ethics Statement (Updated)

## Dataset Ethics
- Dataset provenance must be documented and licensing verified before release.
- Only approved, non-personal industrial images are used.

## Bias and Fairness Risks
- Distribution bias may exist across defect categories.
- Underrepresented classes can reduce recall and increase failure risk.
- Class-wise metrics and confusion matrices are tracked to monitor this.

## Model Limitations
- Baselines are early-stage and not production-safe without calibration.
- Domain shift (lighting, camera setup, material variation) can reduce reliability.

## Privacy and Security
- No personal data is expected in this dataset.
- Repository should avoid storing sensitive operational metadata.

## Misuse Risks
- Over-trusting model outputs for autonomous rejection decisions.
- Deploying outside validated context.

## Human Oversight
- Keep human-in-the-loop for final accept/reject actions.
- Use model as decision support, not sole authority.

## Ongoing Actions
- Continue auditing class imbalance and failure modes.
- Update this statement as new experiments and data are added.
