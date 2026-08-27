"""
revenue_service.py
===================
Revenue at Risk = EstimatedAnnualSpendINR x ChurnProbability, per customer.

Both inputs are real columns already produced by the existing pipeline
(EstimatedAnnualSpendINR from 01_data_loading.py's INR scaling, ChurnProbability
from the trained churn model — see 05_churn_prediction.py). This module only
multiplies and aggregates them; it does not invent or predict any new value.

This is a transparent expected-value estimate, NOT a guaranteed future loss —
every place this number is surfaced in the UI must label it as such.

"Critical" risk is not a separate model output — the pipeline only produces
Low/Medium/High risk bands. CRITICAL_PROBABILITY_THRESHOLD below is a
documented, configurable business threshold layered on top of the existing
High band purely for revenue-exposure reporting.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FINAL_DATA_PATH

CRITICAL_PROBABILITY_THRESHOLD = 0.8
TOP_N_GEOGRAPHY = 10


def _load():
    if not os.path.exists(FINAL_DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(FINAL_DATA_PATH)
    if "EstimatedAnnualSpendINR" in df.columns and "ChurnProbability" in df.columns:
        df["RevenueAtRisk"] = df["EstimatedAnnualSpendINR"] * df["ChurnProbability"]
    else:
        df["RevenueAtRisk"] = 0.0
    return df


def _group_summary(df, col, top_n=None):
    if col not in df.columns or df.empty:
        return []
    g = df.groupby(col).agg(
        customers=("CustomerID", "count"),
        revenue_at_risk=("RevenueAtRisk", "sum"),
        avg_churn_probability=("ChurnProbability", "mean"),
    ).reset_index().rename(columns={col: "name"})
    g["revenue_at_risk"] = g["revenue_at_risk"].round(0)
    g["avg_churn_probability"] = (g["avg_churn_probability"] * 100).round(1)
    g = g.sort_values("revenue_at_risk", ascending=False)
    if top_n:
        g = g.head(top_n)
    return g.to_dict(orient="records")


def get_summary():
    """Business-level Revenue at Risk aggregates for the Revenue at Risk page
    and the dashboard. Every number here is a real aggregate over the actual
    customer base — no synthetic or placeholder figures."""
    df = _load()
    if df.empty:
        return {"has_data": False}

    total_revenue_at_risk = float(df["RevenueAtRisk"].sum())
    high_band = df[df["ChurnRiskBand"] == "High"] if "ChurnRiskBand" in df.columns else df.iloc[0:0]
    critical = df[df["ChurnProbability"] >= CRITICAL_PROBABILITY_THRESHOLD]
    at_risk = df[df["ChurnRiskBand"].isin(["Medium", "High"])] if "ChurnRiskBand" in df.columns else df.iloc[0:0]

    return {
        "has_data": True,
        "total_revenue_at_risk": round(total_revenue_at_risk, 0),
        "high_risk_revenue": round(float(high_band["RevenueAtRisk"].sum()), 0),
        "high_risk_customers": int(len(high_band)),
        "critical_risk_revenue": round(float(critical["RevenueAtRisk"].sum()), 0),
        "critical_risk_customers": int(len(critical)),
        "critical_threshold_pct": round(CRITICAL_PROBABILITY_THRESHOLD * 100, 0),
        "at_risk_customers": int(len(at_risk)),
        "total_customers": int(len(df)),
        "avg_revenue_at_risk_per_customer": round(total_revenue_at_risk / len(df), 0) if len(df) else 0,
        "by_segment": _group_summary(df, "SegmentName"),
        "by_state": _group_summary(df, "State", top_n=TOP_N_GEOGRAPHY),
        "by_city": _group_summary(df, "City", top_n=TOP_N_GEOGRAPHY),
        "by_risk_band": _group_summary(df, "ChurnRiskBand"),
    }


def get_priority_customers(n=5):
    """Top-N customers by real revenue at risk within the High risk band —
    feeds the dashboard's "AI priority actions" panel. Uses the existing
    RecommendedAction column already produced by the pipeline
    (06_business_recommendations.py); no new recommendation is invented."""
    df = _load()
    if df.empty or "ChurnRiskBand" not in df.columns:
        return []
    high = df[df["ChurnRiskBand"] == "High"].sort_values("RevenueAtRisk", ascending=False).head(n)
    cols = [c for c in ["CustomerID", "CustomerName", "City", "State", "SegmentName",
                         "ChurnProbability", "RevenueAtRisk", "RecommendedAction"] if c in high.columns]
    records = high[cols].to_dict(orient="records")
    for r in records:
        r["RevenueAtRisk"] = round(r["RevenueAtRisk"], 0)
        r["ChurnProbabilityPct"] = round(r["ChurnProbability"] * 100, 1)
    return records


def get_customer_revenue(customer_id, customer_row=None):
    """Single-customer Revenue at Risk breakdown, used by Customer 360."""
    if customer_row is None:
        df = _load()
        match = df[df["CustomerID"] == int(customer_id)]
        if match.empty:
            return None
        customer_row = match.iloc[0]

    spend = float(customer_row.get("EstimatedAnnualSpendINR") or 0)
    proba = float(customer_row.get("ChurnProbability") or 0)
    revenue_at_risk = round(spend * proba, 0)
    return {
        "estimated_annual_spend": round(spend, 0),
        "churn_probability": round(proba * 100, 1),
        "revenue_at_risk": revenue_at_risk,
        "is_critical": proba >= CRITICAL_PROBABILITY_THRESHOLD,
    }
