# Project Board Columns

The project board should use the following columns:

1. Backlog
2. To Do
3. In Progress
4. Review
5. Done

## Board setup checklist
- Create a GitHub Project named `Manufacturing Defect Detection`
- Add the five columns exactly as listed above
- Create labels: `setup`, `proposal`, `data`, `model`, `evaluation`, `explainability`, `release`, `priority:high`, `priority:medium`, `priority:low`
- Create milestone `v0.1`
- Add all seven issues to this project

## Suggested flow
- New issues start in **Backlog**
- Move selected work to **To Do**
- Active tasks go to **In Progress**
- PR-ready tasks move to **Review**
- Completed and merged tasks move to **Done**

## Initial issue-to-column mapping
- `Set up repository structure` -> **To Do**
- `Finalize proposal draft` -> **In Progress**
- `Prepare dataset download and folder structure` -> **To Do**
- `Implement baseline CNN classifier` -> **Backlog**
- `Add evaluation metrics` -> **Backlog**
- `Add Grad-CAM visualization` -> **Backlog**
- `Prepare v0.1 release` -> **Backlog**

## Current team split
- This is a shared project and proposal effort for both teammates.
- Current lead split for speed: Grant leads proposal drafting, while Jabez leads repository and GitHub setup.
- The proposal issue remains in **In Progress** under Grant while setup and engineering issues are advanced in parallel; reassignment or co-assignment is applied as work overlaps.

## Operating rules
- Limit **In Progress** to 2 active issues max to avoid context switching.
- Every issue moved to **Review** must have a linked PR or attached evidence.
- Move to **Done** only after merge/verification, not just local completion.
