# Week 2 GitHub Issues to Create

The following Week 2 issues can be created in this order.

Recommended common settings:

- Milestone: `v0.1`
- Project board: `Manufacturing Defect Detection`
- Labels: `data`, `model`, `evaluation`, `explainability`, `proposal`, `release`, `priority:high`, `priority:medium`, `priority:low`

## Ownership Note

- This is still a shared project effort.
- Current lead split for speed: Jabez leads implementation/data/model work, Grant leads proposal/report polish.
- Assignees should reflect the current lead, with the second teammate added for review or paired updates when needed.

## 1) Clean dataset and standardize labels
**Title:** Clean dataset and standardize labels
**Current lead:** Grant
**Labels:** `data`, `priority:high`
**Milestone:** `v0.1`
**Initial board column:** `To Do`
**Body:**
- Validate raw dataset location and class folders under `data/raw/`
- Run `scripts/clean_dataset.py` with reproducible options
- Remove invalid/corrupt/duplicate images and normalize labels
- Save cleaning summary to `outputs/cleaning_summary.json`

**Acceptance criteria:**
- Cleaning script runs end-to-end with a documented command
- Cleaned dataset is written to `data/cleaned/`
- Summary JSON includes total/valid/skipped counts and per-class counts
- Output is reproducible on a fresh machine

## 2) Finalize reproducible train/val/test splits
**Title:** Finalize reproducible train/val/test splits
**Current lead:** Jabez
**Labels:** `data`, `priority:high`
**Milestone:** `v0.1`
**Initial board column:** `To Do`
**Body:**
- Run `scripts/split_dataset.py` on cleaned data
- Use fixed seed and 80/10/10 ratio
- Preserve class folder structure in each split
- Save split summary to `outputs/split_summary_table.csv`

**Acceptance criteria:**
- `data/splits/train`, `data/splits/val`, and `data/splits/test` exist
- Split is class-aware and reproducible with the same seed
- Summary table includes per-class and total counts
- Commands are documented in `README.md`

## 3) Complete EDA notebook for checkpoint
**Title:** Complete EDA notebook for checkpoint
**Current lead:** Grant (shared with Jabez)
**Labels:** `evaluation`, `priority:high`
**Milestone:** `v0.1`
**Initial board column:** `In Progress`
**Body:**
- Finalize `notebooks/eda.ipynb` sections
- Include class distribution, sample images, image dimension analysis, and pixel intensity plot
- Add observations and conclusion markdown
- Export key figures to `outputs/eda_figures/`

**Acceptance criteria:**
- Notebook runs from top to bottom without manual edits
- EDA visuals are saved for report screenshots
- Observations include dataset size, class count, split ratios, and imbalance note
- Conclusion summarizes data quality and risks

## 4) Train ML and CNN baselines
**Title:** Train ML and CNN baselines
**Current lead:** Jabez (shared with Grant)
**Labels:** `model`, `priority:high`
**Milestone:** `v0.1`
**Initial board column:** `Backlog`
**Body:**
- Train ML baseline with `scripts/train_ml_baseline.py`
- Train CNN baseline with `scripts/train_cnn_baseline.py`
- Store outputs in `metrics/`, `experiments/`, and `models/`
- Append both runs to `metrics/baseline_comparison.csv`

**Acceptance criteria:**
- ML and CNN training commands complete successfully
- Metrics JSON and confusion matrices are generated
- CNN history/learning curve files are generated
- Baseline comparison CSV contains both model rows

## 5) Log initial metrics and checkpoint evidence
**Title:** Log initial metrics and checkpoint evidence
**Current lead:** Grant (shared with Jabez)
**Labels:** `evaluation`, `priority:medium`
**Milestone:** `v0.1`
**Initial board column:** `Backlog`
**Body:**
- Update `metrics/initial_metrics.md` with latest baseline values
- Ensure evidence artifacts are linked in docs
- Add or refresh confusion matrices and summary tables
- Cross-check that metric file names are consistent across docs

**Acceptance criteria:**
- Initial metrics markdown matches latest JSON/CSV outputs
- Evidence links in docs resolve to real files
- Baseline comparison and confusion matrices are present
- Metrics are traceable to reproducible commands

## 6) Add explainability and prototype components
**Title:** Add explainability and prototype components
**Current lead:** Jabez (shared with Grant)
**Labels:** `explainability`, `priority:medium`
**Milestone:** `v0.1`
**Initial board column:** `Backlog`
**Body:**
- Generate Grad-CAM samples with `gradcam_predict.py`
- Validate NLP prototype run and output file
- Validate RL stub run and early learning curve output
- Document result locations in report/docs

**Acceptance criteria:**
- Grad-CAM outputs exist under `experiments/gradcam/`
- NLP output exists in `metrics/nlp_prototype_output.json`
- RL output exists in `metrics/rl_stub_metrics.json` and `experiments/rl_learning_curves.png`
- Commands and artifacts are documented for reviewers

## 7) Prepare Week 2 checkpoint package and release
**Title:** Prepare Week 2 checkpoint package and release
**Current lead:** Shared (Jabez + Grant)
**Labels:** `release`, `priority:medium`
**Milestone:** `v0.1`
**Initial board column:** `Backlog`
**Body:**
- Finalize `docs/checkpoint_report.md` (1-2 pages)
- Verify report includes screenshots and code evidence
- Ensure `docs/model_card.md` and `docs/ethics_statement.md` are updated
- Create and push `v0.1` tag when all blocker tasks are complete

**Acceptance criteria:**
- Checkpoint report is complete and submission-ready
- Model card and ethics docs reflect current experiments
- No open `priority:high` blocker issues remain for Week 2 scope
- `v0.1` tag is created and pushed


