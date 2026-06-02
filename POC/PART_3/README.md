# Part 3 — Churn Prediction Model & Model Card

This folder contains the churn-modelling workflow for the capstone, including a baseline model, a stronger challenger, threshold selection, error analysis, and the model card.

## Deliverables

- `churn_model.ipynb`
- `model.pkl`
- `metrics.json`
- `error_analysis.md`
- `model_card.md`
- `charts/`
- `train_churn_model.py`
- `requirements.txt`

## Data Loading

The script looks in these relative locations:

1. `PART_3/data/`
2. `../d2c churn data package/d2c churn data package/`

If you move this folder into a standalone repository, place the churn CSV files inside `data/`.

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_churn_model.py
```

## How To Load The Saved Model

Run these commands from inside `PART_3/` after generating the artifacts:

```python
import joblib

artifact = joblib.load("model.pkl")
model = artifact["model"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]
```

## Modeling Notes

- The workflow uses `rfm_modeling_snapshot.csv` and excludes `customer_id`, `snapshot_date`, `split`, and `churn_next_60d` from features.
- The provided train/validation/test split is preserved.
- A logistic-regression baseline is compared against a tree-based challenger.
- The final threshold is chosen on the validation set under a CRM-capacity cap rather than defaulting to `0.50`.

## Integrity Notes

- Metrics, customer IDs, and error examples are produced from the actual dataset run.
- No post-snapshot raw order data is used as a model feature.
- The folder contains no secrets, API keys, or local-only absolute paths.
