from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "model" / "model.pkl"
if not MODEL_PATH.exists():
    from train_model import main as train_main

    train_main()

from app.main import app

client = TestClient(app)

PAYLOAD_1 = {
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
    "last_visit_days_ago": 20,
}

PAYLOAD_2 = {
    "city_tier": "Tier 2",
    "age_group": "25-34",
    "acquisition_channel": "Marketplace",
    "loyalty_tier": "Silver",
    "preferred_category": "Hair Care",
    "marketing_consent": "Yes",
    "recency_days": 40,
    "frequency_180d": 1,
    "monetary_180d": 581.0,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.23,
    "avg_rating_180d": 4.0,
    "category_diversity_180d": 1,
    "ticket_count_90d": 1,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 1.0,
    "days_since_signup": 121,
    "sessions_30d": 8,
    "product_views_30d": 31,
    "cart_adds_30d": 4,
    "wishlist_adds_30d": 2,
    "abandoned_carts_30d": 3,
    "email_opens_30d": 0,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 0,
}

PAYLOAD_3 = {
    "city_tier": "Tier 1",
    "age_group": "25-34",
    "acquisition_channel": "Influencer",
    "loyalty_tier": None,
    "preferred_category": "Skin Care",
    "marketing_consent": "Yes",
    "recency_days": 171,
    "frequency_180d": 1,
    "monetary_180d": 649.98,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.47,
    "avg_rating_180d": 2.0,
    "category_diversity_180d": 1,
    "ticket_count_90d": 0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 206,
    "sessions_30d": 1,
    "product_views_30d": 3,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 0,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 26,
}


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "threshold" in payload


def test_predict_endpoint_returns_probability_and_explanation() -> None:
    response = client.post("/predict", json=PAYLOAD_1)
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["churn_probability"] <= 1.0
    assert payload["predicted_class"] in [0, 1]
    assert payload["risk_explanation"]


def test_batch_predict_endpoint_handles_multiple_customers() -> None:
    response = client.post("/batch_predict", json={"customers": [PAYLOAD_1, PAYLOAD_2, PAYLOAD_3]})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 3
    assert all("churn_probability" in item for item in payload["predictions"])


def test_validation_error_for_negative_recency() -> None:
    invalid = dict(PAYLOAD_1)
    invalid["recency_days"] = -1
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
