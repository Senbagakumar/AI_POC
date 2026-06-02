# Part 4 — FastAPI Churn Scoring Service

This folder packages the churn model behind a small FastAPI service with single and batch scoring endpoints, validation, tests, and monitoring guidance.

## Deliverables

- `app/main.py`
- `model/model.pkl`
- `model/model_metadata.json`
- `tests/test_api.py`
- `monitoring_plan.md`
- `train_model.py`
- `requirements.txt`

## Data Loading

The training script checks these relative locations:

1. `PART_4/data/`
2. `../d2c churn data package/d2c churn data package/`

If this folder is copied into its own repository, place the churn CSV files inside `data/` before running `train_model.py`.

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
uvicorn app.main:app --reload --port 8000
```

## How To Test

```bash
pytest
```

## Endpoints

- `GET /health`
- `POST /predict`
- `POST /batch_predict`

## Example Request

```json
{
  "city_tier": "Tier 1",
  "age_group": "18-24",
  "acquisition_channel": "Instagram",
  "loyalty_tier": "Silver",
  "preferred_category": "Makeup",
  "marketing_consent": "Yes",
  "recency_days": 107,
  "frequency_180d": 1,
  "monetary_180d": 362.73,
  "return_rate_180d": 0.0,
  "avg_discount_pct_180d": 0.23,
  "avg_rating_180d": 3.0,
  "category_diversity_180d": 1,
  "ticket_count_90d": 0,
  "negative_ticket_rate_90d": 0.0,
  "avg_resolution_hours_90d": 0.0,
  "days_since_signup": 524,
  "sessions_30d": 1,
  "product_views_30d": 4,
  "cart_adds_30d": 0,
  "wishlist_adds_30d": 0,
  "abandoned_carts_30d": 0,
  "email_opens_30d": 2,
  "campaign_clicks_30d": 0,
  "last_visit_days_ago": 20
}
```

## Example Response Shape

```json
{
  "churn_probability": 0.8421,
  "predicted_class": 1,
  "threshold": 0.71,
  "risk_explanation": "High churn risk driven by last purchase is very old, recent app/web activity is weak, repeat purchase depth is low."
}
```

## Integrity Notes

- The service uses the same real feature schema as the modeling snapshot.
- The training artifact is built locally from the provided dataset; no external secrets or hosted dependencies are required.
- All paths are relative and safe to move into a standalone repository.
