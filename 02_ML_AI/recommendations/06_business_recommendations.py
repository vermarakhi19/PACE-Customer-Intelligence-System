"""
06_business_recommendations.py
===============================
Module 6 — Business Recommendations & Personalization.

Combines:
  Segment (from 04_segmentation.py)  +  Churn probability (from 05_churn_prediction.py)
into one retention_actions table a non-technical Indian retail marketing
manager can act on directly - e.g. "send a 20% win-back coupon to At-Risk
customers in Delhi buying Fashion".

Also generates a per-customer PERSONALIZED recommendation (next-best-action)
using the customer's own segment, category, city and churn risk band.

Run standalone: python 02_ML_AI/recommendations/06_business_recommendations.py
Input:  04_DATA_TESTING/datasets/processed/segmented_data.csv, 02_ML_AI/models/best_churn_model.pkl
Output: 04_DATA_TESTING/datasets/processed/final_customer_data.csv (dashboard's main data source)
        04_DATA_TESTING/reports/business_recommendations.json
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# RESTRUCTURE NOTE: config.py now lives one directory above this script
# (02_ML_AI/config.py) — see 00_load_transactions.py for details.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

SEGMENTED_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "segmented_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "02_ML_AI", "models", "best_churn_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "02_ML_AI", "models", "feature_columns.pkl")
CLEANED_MODEL_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "cleaned_data.csv")
FINAL_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "final_customer_data.csv")
RECS_JSON = os.path.join(BASE_DIR, "04_DATA_TESTING", "reports", "business_recommendations.json")

DROP_COLS = ["CustomerID", "CustomerName", "City", "State", "PhoneNumber",
             "SignupDate", "TenureBucket", "SegmentName", "Segment_KMeans",
             "Segment_DBSCAN", "Segment_Hierarchical", "Churn"]


def churn_risk_band(prob):
    if prob >= 0.66:
        return "High"
    if prob >= 0.33:
        return "Medium"
    return "Low"


def action_for(segment, risk_band, category, city):
    """Rule engine mapping (Segment x Risk) -> concrete Indian-retail action."""
    matrix = {
        ("Loyal High-Value", "High"): f"Priority retention call + exclusive early access to {category} sale — do not let a top spender churn.",
        ("Loyal High-Value", "Medium"): f"Send a personalised ₹500 loyalty cashback voucher on next {category} purchase.",
        ("Loyal High-Value", "Low"): "Enrol in premium loyalty tier (e.g. 'Gold Member') to reinforce advocacy.",
        ("At-Risk / Dormant", "High"): f"Immediate win-back offer: flat 20% off + free delivery on {category}, valid 7 days.",
        ("At-Risk / Dormant", "Medium"): "Re-engagement email/SMS with festive-season (Diwali/Holi) discount code.",
        ("At-Risk / Dormant", "Low"): "Add to monthly newsletter segment; monitor for further disengagement.",
        ("Big Spenders", "High"): "Dedicated relationship manager outreach; offer EMI/cashback on high-ticket items.",
        ("Big Spenders", "Medium"): f"Bundle offer on {category} + complementary categories to increase basket size.",
        ("Big Spenders", "Low"): "Invite to referral program — big spenders are strong referral sources.",
        ("New / Recently Acquired", "High"): "Onboarding support call + first-90-days discount to prevent early churn.",
        ("New / Recently Acquired", "Medium"): "Send category-specific welcome series (push notifications) for first 30 days.",
        ("New / Recently Acquired", "Low"): "Standard onboarding journey; introduce loyalty program benefits.",
        ("Occasional Shoppers", "High"): f"Time-boxed flash-sale alert on {category} in {city} to re-activate interest.",
        ("Occasional Shoppers", "Medium"): "Cross-sell recommendation engine push based on browsing history.",
        ("Occasional Shoppers", "Low"): "Include in general promotional broadcasts; low-cost channel only.",
    }
    return matrix.get((segment, risk_band), "Monitor customer activity; no urgent action required.")


def run():
    df = pd.read_csv(SEGMENTED_PATH)
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    model_df = pd.read_csv(CLEANED_MODEL_PATH)
    X = model_df.reindex(columns=feature_cols, fill_value=0)
    churn_proba = model.predict_proba(X)[:, 1]

    df["ChurnProbability"] = np.round(churn_proba, 4)
    df["ChurnRiskBand"] = df["ChurnProbability"].apply(churn_risk_band)
    df["RecommendedAction"] = df.apply(
        lambda r: action_for(r["SegmentName"], r["ChurnRiskBand"], r["PreferedOrderCat"], r["City"]), axis=1
    )

    df.to_csv(FINAL_PATH, index=False)
    print(f"[OK] Final customer table with recommendations saved -> {FINAL_PATH}")

    # -----------------------------------------------------------------
    # Aggregate, statistically-grounded business insights (state/age/category
    # cut of the real, enriched data) - these are the kind of insight lines
    # requested in the brief, e.g. "Customers from Maharashtra aged 25-35
    # buying Electronics have high retention probability."
    # -----------------------------------------------------------------
    insights = []
    grp = df.groupby(["State", "PreferedOrderCat"])["ChurnProbability"].mean().reset_index()
    best_retention = grp.sort_values("ChurnProbability").head(5)
    worst_retention = grp.sort_values("ChurnProbability", ascending=False).head(5)

    for _, row in best_retention.iterrows():
        insights.append(
            f"Customers from {row['State']} buying {row['PreferedOrderCat']} show high retention "
            f"(avg churn probability {row['ChurnProbability']*100:.1f}%) — safe to upsell, low discount needed."
        )
    for _, row in worst_retention.iterrows():
        insights.append(
            f"Customers from {row['State']} buying {row['PreferedOrderCat']} show elevated churn risk "
            f"(avg churn probability {row['ChurnProbability']*100:.1f}%) — prioritise loyalty offers here."
        )

    os.makedirs(os.path.dirname(RECS_JSON), exist_ok=True)
    with open(RECS_JSON, "w") as f:
        json.dump({"insights": insights}, f, indent=2)
    print(f"[OK] Business recommendations saved -> {RECS_JSON}")
    for i in insights:
        print(" -", i)

    return df, insights


if __name__ == "__main__":
    run()
