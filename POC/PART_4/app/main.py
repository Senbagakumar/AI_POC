from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city_tier: Literal["Tier 1", "Tier 2", "Tier 3"]
    age_group: Literal["18-24", "25-34", "35-44", "45+"]
    acquisition_channel: Literal[
        "Google Search",
        "Instagram",
        "Influencer",
        "Referral",
        "Marketplace",
        "Organic",
    ]
    loyalty_tier: Optional[Literal["Silver", "Gold", "Platinum"]] = None
    preferred_category: Literal[
        "Skin Care",
        "Hair Care",
        "Makeup",
        "Fragrance",
        "Wellness",
        "Baby Care",
    ]
    marketing_consent: Literal["Yes", "No"]
    recency_days: int = Field(ge=0)
    frequency_180d: int = Field(ge=0)
    monetary_180d: float = Field(ge=0)
    return_rate_180d: float = Field(ge=0, le=1)
    avg_discount_pct_180d: float = Field(ge=0, le=1)
    avg_rating_180d: float = Field(ge=0, le=5)
    category_diversity_180d: int = Field(ge=0)
    ticket_count_90d: int = Field(ge=0)
    negative_ticket_rate_90d: float = Field(ge=0, le=1)
    avg_resolution_hours_90d: float = Field(ge=0)
    days_since_signup: int = Field(ge=0)
    sessions_30d: int = Field(ge=0)
    product_views_30d: int = Field(ge=0)
    cart_adds_30d: int = Field(ge=0)
    wishlist_adds_30d: int = Field(ge=0)
    abandoned_carts_30d: int = Field(ge=0)
    email_opens_30d: int = Field(ge=0)
    campaign_clicks_30d: int = Field(ge=0)
    last_visit_days_ago: int = Field(ge=0)


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_class: int
    threshold: float
    risk_explanation: str


def model_candidates() -> list[Path]:
    return [
        ROOT / "model" / "model.pkl",
        ROOT / "model.pkl",
    ]


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    for candidate in model_candidates():
        if candidate.exists():
            return joblib.load(candidate)
    raise FileNotFoundError(
        "Model artifact not found. Run `python train_model.py` inside PART_4 first."
    )


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    metadata_path = ROOT / "model" / "model_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact = load_artifact()
    return {"model_name": artifact["model_name"], "threshold": artifact["threshold"]}


def explain_risk(payload: CustomerFeatures, probability: float, predicted_class: int) -> str:
    reasons = []
    if payload.recency_days > 120:
        reasons.append("last purchase is very old")
    elif payload.recency_days > 60:
        reasons.append("order cadence is slowing")
    if payload.sessions_30d <= 2:
        reasons.append("recent app/web activity is weak")
    if payload.frequency_180d <= 1:
        reasons.append("repeat purchase depth is low")
    if payload.return_rate_180d > 0.25:
        reasons.append("returns are elevated")
    if payload.ticket_count_90d >= 1 and payload.negative_ticket_rate_90d >= 0.5:
        reasons.append("support friction is visible")
    if payload.campaign_clicks_30d == 0 and payload.email_opens_30d == 0:
        reasons.append("campaign engagement is absent")

    if predicted_class == 0:
        if payload.recency_days <= 45 and payload.sessions_30d >= 5:
            return "Recent purchasing and healthy engagement keep this customer below the current churn threshold."
        return "The score is below threshold because the customer still shows enough recent activity or value to avoid an immediate churn flag."

    if not reasons:
        reasons.append("multiple weaker signals are stacking up")
    return f"High churn risk driven by {', '.join(reasons[:3])}. Probability: {probability:.2f}."


def score_payload(payload: CustomerFeatures) -> PredictionResponse:
    artifact = load_artifact()
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]

    frame = pd.DataFrame([payload.model_dump()])
    frame = frame.reindex(columns=feature_columns)
    probability = float(model.predict_proba(frame)[:, 1][0])
    predicted_class = int(probability >= threshold)
    explanation = explain_risk(payload, probability, predicted_class)
    return PredictionResponse(
        churn_probability=round(probability, 4),
        predicted_class=predicted_class,
        threshold=round(threshold, 4),
        risk_explanation=explanation,
    )


app = FastAPI(title="D2C Churn Scoring Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    metadata = load_metadata()
    return {
        "status": "ok",
        "model_name": metadata["model_name"],
        "threshold": metadata["threshold"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    return score_payload(payload)


@app.post("/batch_predict")
def batch_predict(payload: BatchRequest) -> dict:
    return {"predictions": [score_payload(customer) for customer in payload.customers]}
