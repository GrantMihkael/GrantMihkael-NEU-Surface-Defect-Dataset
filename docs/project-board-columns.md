# Project Board Columns

The project board uses the following columns:

1. Backlog
2. To Do
3. In Progress
4. Review
5. Done

## Board Setup Checklist

- [x] Create a GitHub Project named `Manufacturing Defect Detection`
- [x] Add the five columns exactly as listed above
- [x] Create labels: `setup`, `proposal`, `data`, `model`, `evaluation`, `explainability`, `release`, `priority:high`, `priority:medium`, `priority:low`
- [x] Create milestone `v0.1`
- [x] Add all seven issues to this project

## Suggested Flow

- New issues start in **Backlog**
- Move selected work to **To Do**
- Active tasks go to **In Progress**
- PR-ready tasks move to **Review**
- Completed and merged tasks move to **Done**

## Initial Issue-to-Column Mapping

- `Set up repository structure` -> **To Do**
- `Finalize proposal draft` -> **In Progress**
- `Prepare dataset download and folder structure` -> **To Do**
- `Implement baseline CNN classifier` -> **Backlog**
- `Add evaluation metrics` -> **Backlog**
- `Add Grad-CAM visualization` -> **Backlog**
- `Prepare v0.1 release` -> **Backlog**

## Current Team Split

- This is a shared project and proposal effort for both teammates.
- Current lead split for speed: Grant leads proposal drafting, while Jabez leads repository and GitHub setup.
- The proposal issue remains in **In Progress** under Grant while setup and engineering issues are advanced in parallel; reassignment or co-assignment is applied as work overlaps.

## Operating Rules

- Limit **In Progress** to 2 active issues max to avoid context switching.
- Every issue moved to **Review** must have a linked PR or attached evidence.
- Move to **Done** only after merge/verification, not just local completion.

## Week 2 Progress Snapshot

Use this as the current status update for the board.

- `Set up repository structure` -> **Done**
- `Finalize proposal draft` -> **Done**
- `Prepare dataset download and folder structure` -> **Done**
- `Implement baseline CNN classifier` -> **Done**
- `Add evaluation metrics` -> **Done**
- `Add Grad-CAM visualization` -> **Done**
- `Prepare v0.1 release` -> **To Do**

## Week 2 Evidence Links

- Data cleaned: `outputs/cleaning_summary.json`
- Splits finalized: `outputs/split_summary_table.csv`, `data/splits/`
- EDA completed: `notebooks/eda.ipynb`, `outputs/eda_figures/`
- Baseline metrics: `metrics/ml_baseline_metrics.json`, `metrics/cnn_baseline_metrics.json`
- CNN curves: `experiments/cnn_learning_curves.png`
- Grad-CAM outputs: `experiments/gradcam/`

## Release Tag Note

When proposal and verification are complete, create the Week 2 checkpoint tag:

- `git tag v0.1`
- `git push origin v0.1`
