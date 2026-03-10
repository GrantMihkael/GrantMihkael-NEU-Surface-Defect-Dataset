# GitHub Issues to Create

The following issues should be created in this order.

Recommended common settings:
- Milestone: `v0.1`
- Project board: `Manufacturing Defect Detection`
- Labels to create first: `setup`, `proposal`, `data`, `model`, `evaluation`, `explainability`, `release`, `priority:high`, `priority:medium`, `priority:low`

## Ownership Note
- This is a shared project; both teammates contribute to implementation and proposal completion.
- Current split for speed: setup/GitHub workflow is led by Jabez, while proposal drafting is led by Grant.
- Assignees should reflect the current lead, with the other teammate added as collaborator/reviewer when needed.

## 1) Set up repository structure
**Title:** Set up repository structure
**Current lead:** Jabez
**Labels:** `setup`, `priority:high`
**Milestone:** `v0.1`
**Initial board column:** `To Do`
**Body:**
- Confirm Week 1 folder layout is complete
- Validate required starter files exist (`README.md`, `requirements.txt`, `.gitignore`, `docs/`, `data/`, `src/`, `notebooks/`, `results/`)
- Add any missing placeholders or README notes

**Acceptance criteria:**
- Folder structure matches project plan
- Required starter files are present and committed
- Root `README.md` has setup and run instructions
- `.gitignore` prevents large dataset/checkpoint artifacts from being committed

## 2) Finalize proposal draft
**Title:** Finalize proposal draft
**Current lead:** Grant
**Labels:** `proposal`, `priority:high`
**Milestone:** `v0.1`
**Initial board column:** `In Progress`
**Body:**
- Complete `docs/proposal.docx`
- Export and update `docs/proposal.pdf`
- Ensure problem statement, dataset, model plan, and timeline are finalized

**Acceptance criteria:**
- Proposal has complete sections: problem, dataset, method, timeline, risks
- `proposal.docx` and `proposal.pdf` match latest version
- Proposal is reviewed by both teammates once before submission

## 3) Prepare dataset download and folder structure
**Title:** Prepare dataset download and folder structure
**Current lead:** Jabez
**Labels:** `data`, `priority:high`
**Milestone:** `v0.1`
**Initial board column:** `To Do`
**Body:**
- Verify `data/download_dataset.py` instructions
- Download NEU dataset from selected Kaggle source
- Extract into `data/raw/` with expected class folders
- Document dataset source and structure in `data/README.md`

**Acceptance criteria:**
- Dataset is downloadable with clear, reproducible steps
- `data/raw/` structure matches what training code expects
- `data/README.md` includes source link, license note, and class distribution summary
- Team can run download/setup flow on a fresh machine

## 4) Implement baseline CNN classifier
**Title:** Implement baseline CNN classifier
**Current lead:** Jabez (shared with Grant)
**Labels:** `model`, `priority:high`
**Milestone:** `v0.1`
**Initial board column:** `Backlog`
**Body:**
- Build baseline model in `src/model.py`
- Add training pipeline in `src/train.py`
- Add inference script in `src/predict.py`
- Save first baseline results under `results/`

**Acceptance criteria:**
- Training runs end-to-end without manual file edits
- Inference script works on at least one sample image
- Baseline metrics are saved under `results/`
- Core training/inference commands are documented in `README.md`

## 5) Add evaluation metrics
**Title:** Add evaluation metrics
**Current lead:** Jabez (shared with Grant)
**Labels:** `evaluation`, `priority:medium`
**Milestone:** `v0.1`
**Initial board column:** `Backlog`
**Body:**
- Compute Accuracy and F1 Score
- Generate confusion matrix
- Add false alarm rate calculation
- Report metrics in `results/` and summarize in project README

**Acceptance criteria:**
- Metrics include accuracy, F1, confusion matrix, false alarm rate
- Metrics are reproducibly generated from latest model outputs
- A concise results summary is added to `README.md`

## 6) Add Grad-CAM visualization
**Title:** Add Grad-CAM visualization
**Current lead:** Jabez (shared with Grant)
**Labels:** `explainability`, `priority:medium`
**Milestone:** `v0.1`
**Initial board column:** `Backlog`
**Body:**
- Implement Grad-CAM in `src/gradcam.py`
- Generate visual explanations for sample predictions
- Save visualization outputs to `results/`
- Add usage notes to README

**Acceptance criteria:**
- Grad-CAM runs for at least one defective and one non-defective sample
- Heatmap outputs are saved under `results/`
- Usage instructions are documented clearly

## 7) Prepare v0.1 release
**Title:** Prepare v0.1 release
**Current lead:** Shared (Jabez + Grant)
**Labels:** `release`, `priority:medium`
**Milestone:** `v0.1`
**Initial board column:** `Backlog`
**Body:**
- Confirm proposal artifacts are complete
- Confirm baseline code is organized and runnable
- Tag release as `v0.1`
- Publish release notes summarizing scope and known gaps

**Acceptance criteria:**
- Tag `v0.1` exists
- Release notes include implemented features, known limitations, and next steps
- Repository has no blocker issues labeled `priority:high` left open for v0.1

## GitHub issue template block
Use the following reusable block while creating each issue:

```markdown
Title: <issue title>
Labels: <label list>
Milestone: v0.1
Project: Manufacturing Defect Detection

Description
- <task item>
- <task item>

Acceptance Criteria
- <success condition>
- <success condition>
```

## Suggested issue order for this week
1. Set up repository structure
2. Prepare dataset download and folder structure
3. Finalize proposal draft
4. Implement baseline CNN classifier
5. Add evaluation metrics
6. Add Grad-CAM visualization
7. Prepare v0.1 release
