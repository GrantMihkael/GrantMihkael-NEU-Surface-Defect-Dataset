# Manufacturing Defect Detection using CNN and Grad-CAM

## Project Overview
This project is an academic prototype for classifying steel surface defects using the NEU Surface Defect Database. The model predicts one of six defect classes and uses Grad-CAM to visualize which image regions influence predictions.

This repository follows the 6INTELSY AY2526 final project guideline: include CNN, NLP, and RL components, plus reproducible training/evaluation artifacts and ethics documentation.

## Intended Use & Limitations
This system is designed for academic research and classroom experimentation on defect classification. It is not intended for direct deployment in production quality-control systems without additional validation, calibration, and domain testing.

## Quick Start
1. Create and activate a Python virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Prepare dataset instructions:
   `python data/download_dataset.py`
4. Place extracted dataset under `data/raw/` using class folders listed by the script.
5. Run one-command reproducibility scaffold:
  `bash run.sh`

## Defect Classes
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

## Evaluation Metrics
- Accuracy
- F1 Score
- Confusion Matrix
- False Alarm Rate

## Required Components (Course Guideline)
- CNN: Defect classification backbone with Grad-CAM explainability.
- NLP: Planned supporting NLP component for documentation/metadata analysis.
- RL: Planned lightweight RL component (e.g., threshold tuning or policy optimization).

## 6) GitHub Repository (Required Structure)
```text
manufacturing-defect-detection/
  README.md                          # project overview, quick start, results highlights, team
  LICENSE
  requirements.txt                   # python dependencies
  run.sh | Makefile                  # one-command reproduce
  data/
    README.md                        # how to obtain data (no raw PII in repo)
    download_dataset.py              # dataset setup helper and folder expectations
  src/
    data_pipeline.py                 # preprocessing and data loading scaffold
    model.py                         # CNN architecture definition
    models/                          # CNN/NLP architecture modules
    train.py                         # training script (current scaffold)
    eval.py                          # evaluation script scaffold
    rl_agent.py                      # RL component scaffold
    predict.py                       # inference script (current scaffold)
    gradcam.py                       # Grad-CAM generation utilities
    utils/                           # utility helpers
  notebooks/
    baseline.ipynb                   # EDA/baseline notebook
  experiments/
    configs/                         # experiment configs
    logs/                            # training logs, curves
    results/                         # tables, plots, figures
  results/                           # current metrics, plots, and model outputs
  docs/
    proposal.pdf                     # project proposal
    proposal.docx                    # editable proposal source
    model_card.md                    # model card (week 1 draft)
    ethics_statement.md              # ethics statement (week 1 draft)
    checkpoint.pdf                   # week 2 checkpoint report (planned)
    final_report.pdf                 # week 3 final report (planned)
    slides.pdf                       # final slides (planned)
    release-notes-v0.1.md            # release notes
    github-issues.md                 # issue list reference
    project-board-columns.md         # board workflow reference
```

## Ethics & Policy Statement (Draft)

### 1. Intended Use & Limitations
- This work supports learning and research on computer vision for industrial defect detection.
- Outputs must be interpreted as decision support, not as a standalone final quality verdict.

### 2. Ethics Risk Register (Top 3 Risks)
| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | Class imbalance bias (some defect types underrepresented) | Medium | High | Track per-class metrics, apply balanced sampling/augmentation, and report class-wise performance |
| 2 | False negatives in safety-critical inspection flow | Medium | High | Optimize recall for defect classes, use threshold tuning, and require human verification for uncertain predictions |
| 3 | Over-trust in model explanations (Grad-CAM misuse) | Medium | Medium | Document Grad-CAM limits and treat heatmaps as supporting evidence only |

### 3. Fairness Checks (Planned)
- Compare performance across all defect categories, not only overall accuracy.
- Review confusion matrix for systematic under-detection of specific classes.
- Report macro and weighted F1 to avoid majority-class masking.

### 4. Privacy & Data Governance
- Dataset source is a public research dataset; no personal user data is collected in this project.
- No hidden data collection is performed during experiments.
- Dataset and references must be cited with license/source attribution.

### 5. Misuse Considerations
- This model must not be used to replace required industrial safety protocols without full validation.
- It should not be presented as universally reliable beyond the evaluated dataset domain.

### 6. Transparency
- Model architecture, training setup, and evaluation metrics are documented in this repository.
- Known limitations and failure cases should be reported in release notes and final report.

## Results Highlights
- Baseline and model result artifacts are stored in `results/`.
- Visualization and analysis experiments are tracked in `notebooks/baseline.ipynb`.

## Week 1 Submission Status
- Proposal documents: completed (`docs/proposal.pdf`, `docs/proposal.docx`)
- Repository setup and issue planning docs: completed
- Data setup instructions and governance notes: completed
- Model card and ethics statement drafts: completed
- CNN/NLP/RL implementation depth: scaffolded for Week 2 and Week 3 completion

## Milestones
- `v0.1`: Proposal and repository setup
- `v0.9`: Release candidate with baseline training/evaluation outputs
- `v1.0`: Final project submission

## Deliverables Checklist (v1.0)
- `README.md` with quick start and results summary
- `requirements.txt` (or `environment.yml`) and one-command repro script
- `docs/proposal.pdf`, `docs/checkpoint.pdf`, `docs/final_report.pdf`, and `docs/slides.pdf`
- `docs/model_card.md` and `docs/ethics_statement.md`
- Ablation and error/slice analysis outputs in experiment/result folders

## Team
- Add member names and roles here.
