# Notebook Policy

Notebooks are for exploration, not long-term production logic.

## Allowed Uses

- EDA and visualization
- Early hypothesis testing
- Feature exploration
- Side-by-side experiment comparison

## Required Follow-through

- Move reusable logic into `src/mmlm2026/`.
- Add tests for reusable logic.
- Log meaningful experiment outcomes in `docs/experiments/` and MLflow.

## Naming Guidance

- `01_eda_*.ipynb`
- `10_feature_*.ipynb`
- `20_model_*.ipynb`
- `30_simulation_*.ipynb`

## Prohibited Patterns

- Durable business/model logic existing only in notebooks
- Final conclusions only in notebook cells without experiment log updates
