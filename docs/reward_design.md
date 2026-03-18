# RL Reward Design Draft

## State
- `defect_score`: continuous estimate [0, 1] of defect likelihood.
- `confidence`: model confidence [0, 1].

## Actions
- `pass`: allow item to pass quality gate.
- `inspect`: send item for human/manual inspection.
- `reject`: reject item as defective.

## Reward
- `+2.0` for `reject` when item is truly defective.
- `+1.0` for `pass` when item is truly non-defective.
- `+0.2` for `inspect` (safe but costly fallback).
- `-1.5` for incorrect high-impact decisions.

## Rationale
- Strong positive reward for catching true defects.
- Moderate reward for correctly passing non-defects.
- Small positive reward for inspection to reflect operational cost.
- Penalty encourages minimizing false positives/false negatives.

## Current Status
- Environment and agent are stubs for Week 2.
- Learning curves are exploratory and may be unstable/noisy.
