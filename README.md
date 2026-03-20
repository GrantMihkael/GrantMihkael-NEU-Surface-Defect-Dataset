# NEU Surface Defect Dataset Project

This repository is organized to satisfy Week 2 project requirements for data readiness, reproducible splits, EDA, ML/DL baselines, NLP prototype, RL stub, and documentation artifacts.

## Project Structure

- `data/raw/`: raw acquired dataset
- `data/cleaned/`: cleaned dataset after deduplication/validation
- `data/splits/`: reproducible train/val/test splits
- `scripts/`: data prep and baseline training scripts
- `notebooks/eda.ipynb`: EDA notebook
- `metrics/`: JSON/CSV metrics outputs
- `outputs/`: checkpoint summaries and exported EDA figures
- `experiments/`: training curves and experimental artifacts
- `models/`: saved model checkpoints
- `nlp/`: NLP scaffold/prototype
- `rl/`: RL environment and agent stub
- `docs/`: model card, ethics statement, data and GitHub setup docs

## Week 2 Data Prep Structure

Use this simple structure for the checkpoint:

- `data/raw/`: class-based raw dataset (`data/raw/<class_name>/*.jpg`)
- `data/cleaned/`: cleaned images with preserved class folders
- `data/splits/`: finalized train/val/test folders by class
- `notebooks/`: EDA notebook(s)
- `scripts/`: cleaning and splitting scripts
- `outputs/`: generated tables, figures, and logs

## Quick Start

Recommended (Windows PowerShell):

- Activate venv: `& ".\\.venv\\Scripts\\Activate.ps1"`

1. Place raw data under `data/raw/` using either layout:
   - `data/raw/<class_name>/<images>`
   - `data/raw/<dataset_name>/<class_name>/<images>`
2. Run cleaning:
   - `python scripts/clean_dataset.py --raw-dir data/raw --cleaned-dir data/cleaned --summary outputs/cleaning_summary.json --normalize-filenames`
   - optional resize: add `--resize-width 224 --resize-height 224`
3. Create reproducible splits:
   - `python scripts/split_dataset.py --cleaned-dir data/cleaned --out-dir data/splits --train-ratio 0.8 --val-ratio 0.1 --seed 42 --summary-csv outputs/split_summary_table.csv`
4. Open EDA notebook:
   - `jupyter notebook notebooks/eda.ipynb`
5. Run baselines (optional for later stages):
   - `python scripts/train_ml_baseline.py --splits-dir data/splits`
   - `python scripts/train_cnn_baseline.py --splits-dir data/splits --epochs 10`
6. Run NLP prototype:
   - `python nlp/prototype.py`
7. Run RL stub experiment:
   - `python rl/run_stub_experiment.py`

Note:

- Older `*_combined` artifacts may still exist from previous runs.
- Week 2 checkpoint flow in this README uses `data/cleaned`, `data/splits`, and `outputs/*` files.

## Week 2 README Snippet

Expected input/output folders for checkpoint scripts:

- Input raw data: `data/raw/<class_name>/<images>`
- Cleaning output: `data/cleaned/<class_name>/<images>` and `outputs/cleaning_summary.json`
- Split output: `data/splits/train|val|test/<class_name>/<images>` and `outputs/split_summary_table.csv`
- EDA notebook: `notebooks/eda.ipynb` (reads from `data/splits`)

## Key Week 2 Evidence Files

- Data cleaning summary: `outputs/cleaning_summary.json`
- Split summary: `outputs/split_summary_table.csv`
- EDA notebook: `notebooks/eda.ipynb`
- EDA exported figures: `outputs/eda_figures/`
- Baseline metrics: `metrics/ml_baseline_metrics.json`, `metrics/cnn_baseline_metrics.json`
- Baseline comparison: `metrics/baseline_comparison.csv`
- CNN learning curves: `experiments/cnn_learning_curves.png`
- RL learning curves: `experiments/rl_learning_curves.png`
- Model card draft: `docs/model_card.md`
- Ethics statement: `docs/ethics_statement.md`
- GitHub setup notes: `docs/github_setup.md`
