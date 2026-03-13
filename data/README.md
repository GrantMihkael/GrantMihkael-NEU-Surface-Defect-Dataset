# Data Setup

## Dataset Source
- Primary dataset: NEU Surface Defect Database
- Access path for this project: selected Kaggle mirror

## Governance Notes
- No raw PII is included in this repository.
- Do not commit raw downloaded dataset files to Git.
- Cite dataset source and license in reports.

## Expected Local Structure
After downloading and extracting, place classes under `data/raw/`:

```text
data/raw/
	crazing/
	inclusion/
	patches/
	pitted_surface/
	rolled_in_scale/
	scratches/
```

## Setup Command
Run:

```bash
python data/download_dataset.py
```

This script prints the required class folder names and target extraction directory.
