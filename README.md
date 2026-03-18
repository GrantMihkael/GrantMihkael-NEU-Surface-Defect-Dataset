# NEU Surface Defect Dataset Project

This repository is organized to satisfy Week 2 project requirements for data readiness, reproducible splits, EDA, ML/DL baselines, NLP prototype, RL stub, and documentation artifacts.

## Project Structure

- `data/raw/`: raw acquired dataset
- `data/cleaned/`: cleaned dataset after deduplication/validation
- `data/splits/`: reproducible train/val/test splits
- `scripts/`: data prep and baseline training scripts
- `notebooks/eda.ipynb`: EDA notebook
- `metrics/`: JSON/CSV metrics outputs
- `experiments/`: training curves and experimental artifacts
- `models/`: saved model checkpoints
- `nlp/`: NLP scaffold/prototype
- `rl/`: RL environment and agent stub
- `docs/`: model card, ethics statement, data and GitHub setup docs

## Quick Start

1. Place raw data under `data/raw/` using either layout:
   - `data/raw/<class_name>/<images>`
   - `data/raw/<dataset_name>/<class_name>/<images>`
2. Run cleaning:
   - `python scripts/clean_dataset.py`
3. Create reproducible splits:
   - `python scripts/create_splits.py --seed 42 --train-ratio 0.8 --val-ratio 0.1`
4. Run baselines:
   - `python scripts/train_ml_baseline.py`
   - `python scripts/train_cnn_baseline.py --epochs 5`
5. Run NLP prototype:
   - `python nlp/prototype.py`
6. Run RL stub experiment:
   - `python rl/run_stub_experiment.py`

## Key Week 2 Evidence Files

- Data cleaning summary: `metrics/data_cleaning_summary.json`
- Split summary: `metrics/split_summary.json`
- EDA notebook: `notebooks/eda.ipynb`
- Baseline metrics: `metrics/ml_baseline_metrics.json`, `metrics/cnn_baseline_metrics.json`
- Baseline comparison: `metrics/baseline_comparison.csv`
- CNN learning curves: `experiments/cnn_learning_curves.png`
- RL learning curves: `experiments/rl_learning_curves.png`
- Model card draft: `docs/model_card.md`
- Ethics statement: `docs/ethics_statement.md`
- GitHub setup notes: `docs/github_setup.md`
