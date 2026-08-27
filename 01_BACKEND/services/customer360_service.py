"""
customer360_service.py
=======================
Composes a single reusable Customer 360 profile out of data and services
that already exist elsewhere in PACE — this module does not add any new
raw data or run any new model:

  - base customer record          -> final_customer_data.csv
  - churn prediction               -> the same row's ChurnProbability/ChurnRiskBand
  - SHAP explanation                -> explainability_service
  - revenue at risk                 -> revenue_service
  - category recommendations        -> customer_recommendations.csv
  - retention scenario preview      -> simulator_service

Two fields the spec asks for (Customer Tier, Customer Health Score) are not
raw pipeline columns. Both are computed here with a plain, documented
formula from existing columns only — never invented data:

  - Tier: quartile of EstimatedAnnualSpendINR across the real customer base
    (Platinum/Gold/Silver/Bronze).
  - Health Score: a 0-100 weighted composite of normalised Recency,
    Frequency, Monetary, EngagementScore, SatisfactionScore and
    (1 - ChurnProbability) — all real existing columns.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FINAL_DATA_PATH, RECOMMENDATIONS_PATH

import explainability_service
import revenue_service
import simulator_service

COMPARISON_METRICS = [
    "Recency", "Frequency", "Monetary", "EngagementScore",
    "SatisfactionScore", "EstimatedAnnualSpendINR", "ChurnProbability",
]

# label/unit/format metadata + whether a LOWER value is the better outcome
# (Recency and ChurnProbability are the only two) — drives bar colour and
# phrasing in the Comparison section of the Customer 360 template.
METRIC_META = {
    "Recency": {"label": "Recency", "unit": "days", "format": "days", "lower_is_better": True},
    "Frequency": {"label": "Frequency", "unit": "orders", "format": "number", "lower_is_better": False},
    "Monetary": {"label": "Monetary Value", "unit": "₹", "format": "currency", "lower_is_better": False},
    "EngagementScore": {"label": "Engagement Score", "unit": "/100", "format": "number", "lower_is_better": False},
    "SatisfactionScore": {"label": "Satisfaction", "unit": "/5", "format": "number", "lower_is_better": False},
    "EstimatedAnnualSpendINR": {"label": "Est. Annual Spend", "unit": "₹", "format": "currency", "lower_is_better": False},
    "ChurnProbability": {"label": "Churn Probability", "unit": "%", "format": "percent", "lower_is_better": True},
}

HEALTH_WEIGHTS = {
    "recency": 0.15,      # lower recency (more recent) -> healthier
    "frequency": 0.15,
    "monetary": 0.15,
    "engagement": 0.20,
    "satisfaction": 0.15,
    "retention": 0.20,    # 1 - churn probability
}

_dataset_cache = None


def _dataset():
    """Whole customer base, loaded once per process. Used for the quartile
    tier cut-offs and the min/max ranges the health score normalises with —
    both computed straight from the real data, no fabricated bounds."""
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = pd.read_csv(FINAL_DATA_PATH) if os.path.exists(FINAL_DATA_PATH) else pd.DataFrame()
    return _dataset_cache


def _tier(spend, df):
    if df.empty or "EstimatedAnnualSpendINR" not in df.columns:
        return "—"
    q25, q50, q75 = df["EstimatedAnnualSpendINR"].quantile([0.25, 0.5, 0.75])
    if spend >= q75:
        return "Platinum"
    if spend >= q50:
        return "Gold"
    if spend >= q25:
        return "Silver"
    return "Bronze"


def _norm(value, series):
    lo, hi = series.min(), series.max()
    if pd.isna(value) or hi == lo:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def _health_score(row, df):
    if df.empty:
        return None
    recency_component = 1 - _norm(row.get("Recency"), df["Recency"])
    frequency_component = _norm(row.get("Frequency"), df["Frequency"])
    monetary_component = _norm(row.get("Monetary"), df["Monetary"])
    engagement_component = _norm(row.get("EngagementScore"), df["EngagementScore"])
    satisfaction_component = _norm(row.get("SatisfactionScore"), df["SatisfactionScore"])
    retention_component = 1 - float(row.get("ChurnProbability") or 0)

    score = (
        HEALTH_WEIGHTS["recency"] * recency_component
        + HEALTH_WEIGHTS["frequency"] * frequency_component
        + HEALTH_WEIGHTS["monetary"] * monetary_component
        + HEALTH_WEIGHTS["engagement"] * engagement_component
        + HEALTH_WEIGHTS["satisfaction"] * satisfaction_component
        + HEALTH_WEIGHTS["retention"] * retention_component
    )
    return round(score * 100, 1)


def _get_recommendation_row(customer_id):
    if not os.path.exists(RECOMMENDATIONS_PATH):
        return None
    rec_df = pd.read_csv(RECOMMENDATIONS_PATH)
    match = rec_df[rec_df["CustomerID"] == int(customer_id)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _comparison_block(row, df):
    if df.empty:
        return {}
    segment_df = df[df["SegmentName"] == row.get("SegmentName")] if "SegmentName" in df.columns else df.iloc[0:0]
    state_df = df[df["State"] == row.get("State")] if "State" in df.columns else df.iloc[0:0]

    block = {}
    for metric in COMPARISON_METRICS:
        if metric not in df.columns:
            continue
        customer_val = round(float(row.get(metric)), 4) if pd.notna(row.get(metric)) else None
        segment_avg = round(float(segment_df[metric].mean()), 4) if len(segment_df) else None
        regional_avg = round(float(state_df[metric].mean()), 4) if len(state_df) else None
        overall_avg = round(float(df[metric].mean()), 4) if len(df) else None
        max_value = max([v for v in (customer_val, segment_avg, regional_avg, overall_avg) if v is not None] or [1])

        entry = dict(METRIC_META.get(metric, {"label": metric, "unit": "", "format": "number", "lower_is_better": False}))
        entry.update({
            "customer": customer_val,
            "segment_avg": segment_avg,
            "regional_avg": regional_avg,
            "overall_avg": overall_avg,
            "max_value": max_value or 1,
        })
        block[metric] = entry
    return block


def search_customers(query, limit=25):
    """Simple ID/name/city search used by the Customer 360 landing page."""
    df = _dataset()
    if df.empty or not query:
        return []
    query = str(query).strip().lower()
    mask = pd.Series(False, index=df.index)
    if "CustomerID" in df.columns:
        mask |= df["CustomerID"].astype(str).str.lower().str.contains(query, na=False)
    for col in ("CustomerName", "City", "State", "SegmentName"):
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(query, na=False)
    cols = [c for c in ["CustomerID", "CustomerName", "City", "State", "SegmentName", "ChurnRiskBand"] if c in df.columns]
    return df[mask][cols].head(limit).to_dict(orient="records")


def get_customer_360(customer_id):
    df = _dataset()
    if df.empty:
        return None
    match = df[df["CustomerID"] == int(customer_id)]
    if match.empty:
        return None
    row = match.iloc[0]

    overview = {
        "customer_id": int(row["CustomerID"]),
        "name": row.get("CustomerName"),
        "city": row.get("City"),
        "state": row.get("State"),
        "segment": row.get("SegmentName"),
        "tier": _tier(row.get("EstimatedAnnualSpendINR"), df),
        "health_score": _health_score(row, df),
        "churn_probability": round(float(row.get("ChurnProbability") or 0) * 100, 1),
        "risk_band": row.get("ChurnRiskBand"),
    }

    behaviour = {
        "recency_days": row.get("Recency"),
        "frequency": row.get("Frequency"),
        "monetary": row.get("Monetary"),
        "avg_order_value": round(float(row.get("Monetary")) / float(row.get("Frequency")), 0)
            if row.get("Frequency") not in (None, 0) and pd.notna(row.get("Frequency")) else None,
        "satisfaction_score": row.get("SatisfactionScore"),
        "engagement_score": row.get("EngagementScore"),
        "tenure_months": row.get("Tenure"),
        "order_count": row.get("OrderCount"),
        "day_since_last_order": row.get("DaySinceLastOrder"),
        "complain": bool(row.get("Complain")) if pd.notna(row.get("Complain")) else None,
        "preferred_category": row.get("PreferedOrderCat"),
        "preferred_payment": row.get("PreferredPaymentMode"),
    }

    explanation = None
    if explainability_service.is_available():
        try:
            explanation = explainability_service.explain_customer(customer_id, customer_row=row)
        except Exception:
            explanation = None

    revenue = revenue_service.get_customer_revenue(customer_id, customer_row=row)
    recommendation_row = _get_recommendation_row(customer_id)
    simulator_preview = simulator_service.simulate_customer(
        customer_id, action_keys=["discount", "coupon", "loyalty_points", "personalized_offer"], customer_row=row
    )

    intelligence = {
        "explanation": explanation,
        "recommended_action": row.get("RecommendedAction"),
        "recommended_categories": recommendation_row,
        "revenue_at_risk": revenue,
        "simulator_preview": simulator_preview,
    }

    comparison = _comparison_block(row, df)

    return {
        "overview": overview,
        "behaviour": behaviour,
        "intelligence": intelligence,
        "comparison": comparison,
    }
