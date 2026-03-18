# Week 2 Requirement Coverage

1. GitHub organization
- Evidence: `.github/ISSUE_TEMPLATE/task.md`, `.github/workflows/release-tags.yml`, `docs/github_setup.md`

2. Data acquired and cleaned
- Evidence: `data/raw/`, `data/cleaned/`, `scripts/clean_dataset.py`, `metrics/data_cleaning_summary.json`

3. Data splits finalized
- Evidence: `scripts/create_splits.py`, `data/splits/`, `metrics/split_summary.json`

4. EDA notebook
- Evidence: `notebooks/eda.ipynb`

5. Baselines trained (simple ML + DL)
- Evidence: `scripts/train_ml_baseline.py`, `scripts/train_cnn_baseline.py`, outputs in `metrics/` and `experiments/`

6. Initial metrics logged
- Evidence: `metrics/initial_metrics.md`, `metrics/baseline_comparison.csv`, model metric JSON files

7. CNN experiment running with first results
- Evidence: `scripts/train_cnn_baseline.py`, `experiments/cnn_history.csv`, `experiments/cnn_learning_curves.png`, `models/cnn_baseline_best.pth`

8. NLP component scaffolded/prototyped
- Evidence: `nlp/prototype.py`, `metrics/nlp_prototype_output.json`

9. RL agent stubbed with reward design
- Evidence: `rl/environment.py`, `rl/agent_stub.py`, `docs/reward_design.md`

10. Early learning curves for RL
- Evidence: `rl/run_stub_experiment.py`, `experiments/rl_learning_curves.png`, `metrics/rl_stub_metrics.json`

11. Draft model card started
- Evidence: `docs/model_card.md`

12. Ethics statement updated
- Evidence: `docs/ethics_statement.md`
