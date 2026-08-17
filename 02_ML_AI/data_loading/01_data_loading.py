"""
01_data_loading.py
===================
Module 1 — Data Acquisition & Indian-Context Enrichment.

WHAT THIS DOES
---------------
1. Loads the real Kaggle dataset:
   "Ecommerce Customer Churn Analysis and Prediction"
   https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
   -> Save the Kaggle CSV/XLSX as: 04_DATA_TESTING/datasets/raw/ecommerce_churn.csv

2. Validates it has the 20 expected columns (fails loudly if the schema
   doesn't match, instead of silently producing wrong results downstream).

3. Enriches every row with an Indian customer identity layer:
   - Customer Name (Indian first + last name)
   - City (derived deterministically from the dataset's real CityTier column)
   - State (mapped from city)
   - Phone Number (Indian 10-digit, starting 6-9, formatted +91-XXXXXXXXXX)
   - SignupDate (Indian DD/MM/YYYY format)
   This does NOT touch or fabricate any of the analytical columns
   (Tenure, Churn, SatisfactionScore, etc.) — those stay exactly as
   Kaggle published them. Only cosmetic "who is this row" fields are
   synthesised, which is standard practice since the raw dataset has no
   PII (it was anonymised by the publisher) and a retail CRM demo needs
   a name/city/phone to display per customer.

WHY A SEPARATE STEP
--------------------
Keeping identity-enrichment isolated from statistical preprocessing means
the EDA/model scripts always work off the *original* numeric/categorical
distributions — nothing about Tenure, Complain, SatisfactionScore etc. is
altered here, so every chart and metric downstream is still 100% derived
from the real Kaggle data.
"""
import os
import sys
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# RESTRUCTURE NOTE: see 00_load_transactions.py for why this is needed —
# config.py now lives one directory above this script (02_ML_AI/config.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_PATH, BASE_DIR, CITY_TIER_MAP, INDIAN_STATE_MAP,
    INDIAN_FIRST_NAMES, INDIAN_LAST_NAMES, RANDOM_STATE,
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

EXPECTED_COLUMNS = [
    "CustomerID", "Churn", "Tenure", "PreferredLoginDevice", "CityTier",
    "WarehouseToHome", "PreferredPaymentMode", "Gender", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "PreferedOrderCat", "SatisfactionScore",
    "MaritalStatus", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount",
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Kaggle CSV/XLSX and validate its schema."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRaw dataset not found at: {path}\n"
            "Download it from:\n"
            "https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction\n"
            "and save it as 04_DATA_TESTING/datasets/raw/ecommerce_churn.csv (convert the .xlsx to .csv if needed)."
        )

    df = pd.read_csv(path)
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset schema mismatch. Missing expected columns: {missing_cols}\n"
            "Make sure you downloaded 'Ecommerce Customer Churn Analysis and Prediction' "
            "and not a different Kaggle dataset."
        )
    print(f"[OK] Loaded raw data: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def _random_indian_phone() -> str:
    """Generate a syntactically valid Indian mobile number (starts 6-9)."""
    first_digit = random.choice("6789")
    rest = "".join(random.choices("0123456789", k=9))
    return f"+91-{first_digit}{rest}"


def _random_signup_date() -> str:
    """Random signup date in the last 3 years, Indian DD/MM/YYYY format."""
    days_back = random.randint(30, 3 * 365)
    d = datetime(2026, 7, 25) - timedelta(days=days_back)
    return d.strftime("%d/%m/%Y")


def enrich_with_indian_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Name / City / State / Phone / SignupDate columns."""
    df = df.copy()
    names, cities, states, phones, signup_dates = [], [], [], [], []

    for _, row in df.iterrows():
        tier = int(row["CityTier"]) if row["CityTier"] in (1, 2, 3) else 1
        city = random.choice(CITY_TIER_MAP.get(tier, CITY_TIER_MAP[1]))
        state = INDIAN_STATE_MAP.get(city, "Maharashtra")
        name = f"{random.choice(INDIAN_FIRST_NAMES)} {random.choice(INDIAN_LAST_NAMES)}"

        names.append(name)
        cities.append(city)
        states.append(state)
        phones.append(_random_indian_phone())
        signup_dates.append(_random_signup_date())

    df["CustomerName"] = names
    df["City"] = cities
    df["State"] = states
    df["PhoneNumber"] = phones
    df["SignupDate"] = signup_dates
    return df


def run():
    df = load_raw_data()
    df = enrich_with_indian_identity(df)
    out_path = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "raw_enriched.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Enriched dataset saved -> {out_path}")
    print(df[["CustomerID", "CustomerName", "City", "State", "PhoneNumber", "SignupDate"]].head())
    return df


if __name__ == "__main__":
    run()
