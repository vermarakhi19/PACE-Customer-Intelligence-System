"""
02_preprocessing.py
====================
Module 2 — Data Preprocessing.

Steps (each is its own function so it can be unit-tested / explained line by
line in the viva):
  1. remove_duplicates       - drop exact duplicate CustomerID rows
  2. handle_missing_values   - median-impute numeric, mode-impute categorical
  3. detect_outliers         - IQR method, flags + caps (winsorises) instead
                                of deleting rows (deleting real customers
                                would bias churn analysis)
  4. encode_categoricals     - label-encode binary cols, one-hot the rest
  5. scale_numeric            - StandardScaler on continuous numeric columns
  6. engineer_features        - business-meaningful derived features
  7. correlation_analysis     - Pearson correlation matrix + heatmap PNG

Run standalone:  python 02_ML_AI/preprocessing/02_preprocessing.py
Input:  04_DATA_TESTING/datasets/processed/raw_enriched.csv   (from 01_data_loading.py)
Output: 04_DATA_TESTING/datasets/processed/cleaned_data.csv   (encoded+scaled, for modelling)
        04_DATA_TESTING/datasets/processed/cleaned_readable.csv (human-readable, for EDA/dashboard)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# RESTRUCTURE NOTE: config.py now lives one directory above this script
# (02_ML_AI/config.py) — see 00_load_transactions.py for details.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, FIGURES_DIR, INR_SCALE_FACTOR

IN_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "raw_enriched.csv")
CLEAN_MODEL_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "cleaned_data.csv")
CLEAN_READABLE_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "cleaned_readable.csv")

NUMERIC_COLS = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "SatisfactionScore", "NumberOfAddress", "OrderAmountHikeFromlastYear",
    "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount",
]
CATEGORICAL_COLS = [
    "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
    "PreferedOrderCat", "MaritalStatus", "City", "State",
]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["CustomerID"]).reset_index(drop=True)
    print(f"[Dedup] Removed {before - len(df)} duplicate CustomerID rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_report = df.isnull().sum()
    missing_report = missing_report[missing_report > 0]
    print("[Missing] Columns with missing values before imputation:")
    print(missing_report)

    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    print(f"[Missing] Remaining missing values after imputation: {int(df.isnull().sum().sum())}")
    return df


def detect_and_cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """IQR method: cap (winsorise) values beyond 1.5*IQR instead of dropping
    rows, so we don't lose real churned/loyal customers from the dataset."""
    df = df.copy()
    capped_counts = {}
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        capped_counts[col] = int(n_out)
        df[col] = df[col].clip(lower, upper)
    print("[Outliers] Values capped per column:", capped_counts)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Business-meaningful derived features used by both segmentation and
    churn models."""
    df = df.copy()

    # Monetary proxy in real INR for an Indian retail dashboard
    df["EstimatedAnnualSpendINR"] = (
        (df["OrderCount"].clip(lower=1) * (df["CashbackAmount"].clip(lower=1)))
        * INR_SCALE_FACTOR / 10
    ).round(2)

    # RFM-style engineered features (Recency/Frequency/Monetary proxies)
    df["Recency"] = df["DaySinceLastOrder"]
    df["Frequency"] = df["OrderCount"]
    df["Monetary"] = df["EstimatedAnnualSpendINR"]

    # Engagement score: blends app usage, devices, addresses -> 0-100 scale
    df["EngagementScore"] = (
        (df["HourSpendOnApp"] / df["HourSpendOnApp"].max()) * 40
        + (df["NumberOfDeviceRegistered"] / df["NumberOfDeviceRegistered"].max()) * 20
        + (df["SatisfactionScore"] / 5) * 40
    ).round(2)

    # Loyalty tenure bucket (business-readable segments)
    df["TenureBucket"] = pd.cut(
        df["Tenure"], bins=[-1, 3, 12, 24, 999],
        labels=["New (0-3mo)", "Growing (4-12mo)", "Established (1-2yr)", "Loyal (2yr+)"]
    )

    # Complaint risk flag
    df["HighComplaintRisk"] = ((df["Complain"] == 1) & (df["SatisfactionScore"] <= 2)).astype(int)

    print("[FeatureEng] Added: EstimatedAnnualSpendINR, Recency, Frequency, "
          "Monetary, EngagementScore, TenureBucket, HighComplaintRisk")
    return df


def encode_and_scale(df: pd.DataFrame):
    """Returns a MODEL-READY numeric dataframe (encoded + scaled) while
    leaving the human-readable dataframe (with original category strings)
    untouched for EDA / dashboard use."""
    model_df = df.copy()

    # Binary label-encode
    for col in ["Gender"]:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])

    # One-hot encode multi-class categoricals
    onehot_cols = ["PreferredLoginDevice", "PreferredPaymentMode", "PreferedOrderCat", "MaritalStatus"]
    model_df = pd.get_dummies(model_df, columns=onehot_cols, drop_first=True)

    # Scale continuous numeric features
    scale_cols = NUMERIC_COLS + ["EstimatedAnnualSpendINR", "EngagementScore"]
    scaler = StandardScaler()
    model_df[scale_cols] = scaler.fit_transform(model_df[scale_cols])

    print(f"[Encode/Scale] Model-ready shape: {model_df.shape}")
    return model_df, scaler


def correlation_analysis(df: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="RdYlGn", center=0, annot=False, linewidths=0.3)
    plt.title("Correlation Matrix — Numeric Features (Indian E-Commerce Retail Data)")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Correlation] Heatmap saved -> {out}")

    churn_corr = corr["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
    print("[Correlation] Top features correlated with Churn:")
    print(churn_corr.head(10))
    return corr


def run():
    df = pd.read_csv(IN_PATH)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = detect_and_cap_outliers(df)
    df = engineer_features(df)

    correlation_analysis(df)

    df.to_csv(CLEAN_READABLE_PATH, index=False)
    print(f"[OK] Readable cleaned data saved -> {CLEAN_READABLE_PATH}")

    model_df, scaler = encode_and_scale(df)
    model_df.to_csv(CLEAN_MODEL_PATH, index=False)
    print(f"[OK] Model-ready encoded/scaled data saved -> {CLEAN_MODEL_PATH}")

    return df, model_df


if __name__ == "__main__":
    run()
