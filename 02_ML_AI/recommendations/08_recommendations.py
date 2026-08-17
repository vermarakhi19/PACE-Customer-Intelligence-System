"""
08_recommendations.py
======================
Module 8 — Product Category Recommendations (hybrid: kNN + segment affinity).

WHAT THIS DOES AND WHY IT IS FRAMED THIS WAY
----------------------------------------------
The dataset has no basket/transaction log — only a single "most-ordered
category last month" field (`PreferedOrderCat`) per customer. There is no
user-item interaction matrix, so classic collaborative filtering or
association-rule mining (Apriori) cannot honestly be built here without
inventing basket data that does not exist. This module is deliberately a
**content-based / behavioural-similarity** recommender, not collaborative
filtering, and that distinction is documented rather than hidden.

APPROACH (hybrid)
------------------
1. k-Nearest Neighbours on real behavioural features (Segment, CityTier,
   Recency, Frequency, Monetary, EngagementScore, SatisfactionScore) —
   deliberately EXCLUDING PreferedOrderCat from the feature space itself,
   otherwise "nearest neighbour" would trivially just mean "same category"
   and the recommendation would be circular. Instead: find customers who
   BEHAVE similarly, then look at what categories THOSE customers actually
   favour.
2. A Segment x Category affinity table (plain crosstab, real data) used as
   the human-readable "why" — e.g. "58% of customers in your segment favour
   Electronics" — for dashboard explainability.
3. Combine both into a ranked top-2 recommendation per customer with a
   confidence score and a plain-English reason string.

Run standalone: python 02_ML_AI/recommendations/08_recommendations.py
Input:  04_DATA_TESTING/datasets/processed/final_customer_data.csv
Output: 04_DATA_TESTING/datasets/processed/customer_recommendations.csv
        04_DATA_TESTING/reports/segment_category_affinity.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# RESTRUCTURE NOTE: config.py now lives one directory above this script
# (02_ML_AI/config.py) — see 00_load_transactions.py for details.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

IN_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "final_customer_data.csv")
OUT_CSV = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "customer_recommendations.csv")
OUT_JSON = os.path.join(BASE_DIR, "04_DATA_TESTING", "reports", "segment_category_affinity.json")

FEATURE_COLS = ["CityTier", "Recency", "Frequency", "Monetary", "EngagementScore", "SatisfactionScore"]
N_NEIGHBORS = 11   # 10 real neighbours + the customer itself
TOP_K_RECOMMENDATIONS = 2


def build_feature_matrix(df: pd.DataFrame):
    """Behavioural feature space — PreferedOrderCat is deliberately excluded
    (see module docstring): recommendations come from what SIMILAR
    customers buy, not from restating the customer's own category."""
    X = df[FEATURE_COLS].copy()

    if "SegmentName" in df.columns:
        seg_dummies = pd.get_dummies(df["SegmentName"], prefix="Seg")
        X = pd.concat([X, seg_dummies], axis=1)

    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def segment_category_affinity(df: pd.DataFrame) -> dict:
    """Real crosstab: % of each segment that favours each category —
    the explainable, non-black-box half of the hybrid recommender."""
    ct = pd.crosstab(df["SegmentName"], df["PreferedOrderCat"], normalize="index") * 100
    return ct.round(1).to_dict(orient="index")


def knn_category_recommendations(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    n_neighbors = min(N_NEIGHBORS, len(df))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    model.fit(X_scaled)
    _, neighbor_idx = model.kneighbors(X_scaled)

    categories = df["PreferedOrderCat"].values
    results = []
    for i, neighbors in enumerate(neighbor_idx):
        own_category = categories[i]
        neighbor_categories = [categories[j] for j in neighbors if j != i]
        if not neighbor_categories:
            results.append({"top_recs": [], "confidences": []})
            continue

        counts = pd.Series(neighbor_categories).value_counts()
        counts = counts[counts.index != own_category]  # recommend something new, not what they already buy
        top = counts.head(TOP_K_RECOMMENDATIONS)
        total_neighbors = len(neighbor_categories)

        results.append({
            "top_recs": list(top.index),
            "confidences": [round(float(c) / total_neighbors * 100, 1) for c in top.values],
        })
    return results


def build_reason(row, affinity: dict):
    seg_affinity = affinity.get(row["SegmentName"], {})
    reasons = []
    for cat in row["RecommendedCategories"]:
        pct = seg_affinity.get(cat)
        if pct:
            reasons.append(f"{cat} (favoured by similar customers; also {pct}% of {row['SegmentName']} segment)")
        else:
            reasons.append(f"{cat} (favoured by customers with a similar behaviour profile)")
    return "; ".join(reasons) if reasons else "Not enough similar customers to recommend a new category confidently."


def run():
    df = pd.read_csv(IN_PATH)
    required = FEATURE_COLS + ["PreferedOrderCat", "SegmentName", "CustomerID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for recommendations: {missing}. "
                          "Run the full pipeline (01-06) first.")

    X_scaled = build_feature_matrix(df)
    knn_results = knn_category_recommendations(df, X_scaled)
    affinity = segment_category_affinity(df)

    out = df[["CustomerID", "CustomerName", "SegmentName", "PreferedOrderCat"]].copy() \
        if "CustomerName" in df.columns else df[["CustomerID", "SegmentName", "PreferedOrderCat"]].copy()
    out = out.rename(columns={"PreferedOrderCat": "CurrentCategory"})
    out["RecommendedCategories"] = [r["top_recs"] for r in knn_results]
    out["Confidence"] = [r["confidences"] for r in knn_results]
    out["Reason"] = out.apply(lambda r: build_reason(r, affinity), axis=1)

    # flatten list columns to readable strings for CSV / dashboard table
    out["RecommendedCategory1"] = out["RecommendedCategories"].apply(lambda x: x[0] if len(x) > 0 else "")
    out["Confidence1"] = out["Confidence"].apply(lambda x: x[0] if len(x) > 0 else None)
    out["RecommendedCategory2"] = out["RecommendedCategories"].apply(lambda x: x[1] if len(x) > 1 else "")
    out["Confidence2"] = out["Confidence"].apply(lambda x: x[1] if len(x) > 1 else None)
    out = out.drop(columns=["RecommendedCategories", "Confidence"])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] Customer recommendations saved -> {OUT_CSV}")
    print(out.head())

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "methodology": (
                "Content-based hybrid recommender: k-Nearest-Neighbours on real "
                "behavioural features (segment, RFM, engagement, city tier) — "
                "PreferedOrderCat excluded from the feature space to avoid circular "
                "recommendations — combined with a Segment x Category affinity table "
                "for explainability. NOT collaborative filtering: the dataset has no "
                "basket/transaction log to support that."
            ),
            "segment_category_affinity_pct": affinity,
        }, f, indent=2)
    print(f"[OK] Segment-category affinity table saved -> {OUT_JSON}")

    return out


if __name__ == "__main__":
    run()
