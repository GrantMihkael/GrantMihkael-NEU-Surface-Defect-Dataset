# Initial Metrics Log

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | Notes |
|---|---:|---:|---:|---:|---|
| LogisticRegression(flattened_grayscale,combined) | 0.5718 | 0.5790 | 0.5718 | 0.5501 | `metrics/ml_baseline_metrics_combined.json` |
| SimpleCNN | 0.7154 | 0.6849 | 0.7154 | 0.6653 | `metrics/cnn_baseline_metrics.json` |
| NLP Prototype | n/a | n/a | n/a | n/a | Early scaffold in `nlp/prototype.py` |
| RL Stub | n/a | n/a | n/a | n/a | Reward trend in `experiments/rl_learning_curves.png` |

Metrics above were generated from the combined split (`data/splits_combined`) with seed 42.
