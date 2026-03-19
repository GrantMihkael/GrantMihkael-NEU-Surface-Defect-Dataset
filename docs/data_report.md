# Data Acquisition and Cleaning Report

## Data Source
- Datasets: NEU Surface Defect Dataset and Aluminum Profile Surface Defects Dataset
- Acquisition path for this project:
  - Raw data location: `data/raw/`
  - Supported layouts:
    - `data/raw/<class_name>/<images>`
    - `data/raw/<dataset_name>/<class_name>/<images>`

## Cleaning Process
- Script: `scripts/clean_dataset.py`
- Operations performed:
  - image validity checks (`PIL.Image.verify`)
  - duplicate removal (SHA-256 hash based)
  - invalid sample removal
  - standardized class naming (lowercase + underscore)
  - dataset-aware cleaning summary when nested datasets are used
- Cleaning summary output: `metrics/data_cleaning_summary_combined.json`

## Split Process
- Script: `scripts/create_splits.py`
- Reproducible seed: `42`
- Default ratio: `80 / 10 / 10`
- Output folders: `data/splits_combined/train`, `data/splits_combined/val`, `data/splits_combined/test`
- Split summary output: `metrics/split_summary_combined.json`

## Final Number of Samples
- After running scripts, fill with actual values from:
  - `metrics/data_cleaning_summary_combined.json`
  - `metrics/split_summary_combined.json`
