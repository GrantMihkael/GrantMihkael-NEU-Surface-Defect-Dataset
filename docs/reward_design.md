# Reward Design - RL Quality Control Policy (Week 2)

## 1. Objective

Design a reward function for an RL agent that chooses one of three quality-control decisions per item:

- `pass`
- `inspect`
- `reject`

The policy should maximize safety and quality while controlling inspection/rejection cost.

Primary goal priority:

1. Avoid missed defects (false negatives)
2. Avoid unnecessary rejection of good parts (false positives)
3. Use manual inspection as a controlled fallback, not as default behavior

## 2. State Space (Current Stub)
    
For each item, the current state vector is:

- `defect_score` in `[0, 1]`: estimated defect likelihood from an upstream signal
- `confidence` in `[0, 1]`: confidence in that estimate

Planned extension (Week 3+):

- running defect-rate estimate
- per-class uncertainty
- shift indicators (lighting/material domain flags)

## 3. Action Space

- `pass`: allow item to continue
- `inspect`: route to human/manual inspection
- `reject`: reject as defective

## 4. Reward Function

Let `y` be ground truth (`1 = defective`, `0 = non-defective`) and `a` be action.

Base reward table:

| Action | Condition | Reward |
|---|---|---:|
| `reject` | `y = 1` | +2.0 |
| `pass` | `y = 0` | +1.0 |
| `inspect` | any `y` | +0.2 |
| `reject` | `y = 0` | -1.5 |
| `pass` | `y = 1` | -1.5 |

Operational interpretation:

- True defect capture (`reject` on defective) gets highest reward
- Correct throughput (`pass` on clean) gets moderate reward
- Inspection gets small positive reward because it is safe but operationally costly
- Wrong terminal decisions (`pass` defective or `reject` clean) get strong penalty

## 5. Optional Reward Shaping (Recommended)

To reduce policy instability and discourage degenerate strategies, apply small shaping terms:

- Confidence alignment bonus:
	- `+0.1 * confidence` when decision matches truth
- Over-inspection penalty:
	- `-lambda_i` when inspection rate exceeds threshold `tau_i`
- Over-rejection penalty:
	- `-lambda_r` when rejection rate exceeds expected defect prevalence

A practical shaped reward can be written as:

`R_total = R_base + R_conf - P_inspect - P_reject`

Default starting values:

- `tau_i = 0.35`
- `lambda_i = 0.2`
- `lambda_r = 0.1`

## 6. Why This Design

- Industrial defect screening is asymmetric: missed defects are often more costly than extra inspections.
- Rewarding `inspect` slightly above zero keeps it available for uncertain cases.
- High penalties on wrong direct decisions push the policy toward safer behavior.
- Shaping terms prevent the trivial policy: always inspect everything.

## 7. Safety and Policy Constraints

During evaluation, enforce these soft constraints:

- False negative rate must remain below a target bound
- Inspection rate must not exceed operational capacity
- Rejection rate must be plausible relative to observed defect prevalence

If constraints are violated, treat the policy as non-deployable even if total reward is high.

## 8. Evaluation Plan

Track these metrics per run:

- mean episode reward
- min/max episode reward
- action distribution (`pass/inspect/reject`)
- false negative and false positive counts
- inspection rate and rejection rate

Current output artifacts:

- `metrics/rl_stub_metrics.json`
- `experiments/rl_learning_curves.png`

## 9. Known Week 2 Limitations

- Environment is a simplified stub and does not yet model sequential production dynamics.
- Ground-truth generation in stub episodes is synthetic.
- Reward coefficients are heuristic and not yet calibrated with real factory cost models.

## 10. Next Iteration (Week 3)

Planned upgrades:

1. Replace synthetic labels with replayed model outputs and real split labels.
2. Add uncertainty-aware state features.
3. Tune reward coefficients using cost-sensitive validation.
4. Add ablation study comparing no-shaping vs shaping rewards.
