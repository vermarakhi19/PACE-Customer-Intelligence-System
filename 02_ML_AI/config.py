"""
config.py
=========
Single source of truth for file paths and Indian-context lookup tables used
across every script in the pipeline. Centralising these avoids "magic
strings" scattered across 6 files and matches the industry pattern of a
project-wide settings module.
"""
import os

# ---------------------------------------------------------------------------
# Paths (all relative to project root, computed from this file's location so
# the scripts work no matter which directory they're run from)
#
# RESTRUCTURE NOTE (team-folder reorg): this file lives at
# 02_ML_AI/config.py — exactly one directory below the project root, the same
# depth src/config.py used to be at — so the BASE_DIR formula below is
# unchanged and still resolves correctly. Only the data/reports/models
# sub-paths changed, to match the new 4-folder layout:
#   data/raw, data/processed        -> 04_DATA_TESTING/datasets/raw|processed
#   reports/, reports/figures       -> 04_DATA_TESTING/reports/...
#   models/                         -> 02_ML_AI/models/
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TESTING_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING")
RAW_DATA_PATH = os.path.join(DATA_TESTING_DIR, "datasets", "raw", "ecommerce_churn.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_TESTING_DIR, "datasets", "processed", "cleaned_data.csv")
SEGMENTED_DATA_PATH = os.path.join(DATA_TESTING_DIR, "datasets", "processed", "segmented_data.csv")
FINAL_DATA_PATH = os.path.join(DATA_TESTING_DIR, "datasets", "processed", "final_customer_data.csv")
FIGURES_DIR = os.path.join(DATA_TESTING_DIR, "reports", "figures")
MODELS_DIR = os.path.join(BASE_DIR, "02_ML_AI", "models")
METRICS_PATH = os.path.join(DATA_TESTING_DIR, "reports", "model_metrics.json")

# ---------------------------------------------------------------------------
# Indian business context — the raw Kaggle dataset only has a numeric
# "CityTier" (1/2/3), which is the real classification the Govt of India /
# RBI uses for Indian cities. We map each tier to real Indian city names so
# every downstream chart, dashboard filter and report reads in Indian terms
# instead of "Tier 1 / Tier 2 / Tier 3".
# ---------------------------------------------------------------------------
CITY_TIER_MAP = {
    1: ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad"],
    2: ["Pune", "Ahmedabad", "Nagpur", "Lucknow", "Indore", "Jaipur"],
    3: ["Nashik", "Kolhapur", "Aurangabad", "Amravati", "Bhopal", "Patna"],
}

INDIAN_STATE_MAP = {
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra", "Kolhapur": "Maharashtra", "Aurangabad": "Maharashtra",
    "Amravati": "Maharashtra",
    "Delhi": "Delhi", "Bangalore": "Karnataka", "Chennai": "Tamil Nadu",
    "Kolkata": "West Bengal", "Hyderabad": "Telangana", "Ahmedabad": "Gujarat",
    "Lucknow": "Uttar Pradesh", "Indore": "Madhya Pradesh", "Jaipur": "Rajasthan",
    "Bhopal": "Madhya Pradesh", "Patna": "Bihar",
}

INDIAN_FIRST_NAMES = [
    "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohan", "Kavita",
    "Arjun", "Neha", "Sanjay", "Pooja", "Karan", "Divya", "Aditya", "Meera",
    "Suresh", "Ritu", "Manish", "Swati",
]
INDIAN_LAST_NAMES = [
    "Sharma", "Patil", "Verma", "Kulkarni", "Reddy", "Iyer", "Gupta", "Nair",
    "Deshmukh", "Joshi", "Mehta", "Chavan", "Rao", "Singh", "Agarwal",
]

CURRENCY_SYMBOL = "₹"
DATE_FORMAT = "%d/%m/%Y"

# Business: average order value multiplier to convert the dataset's spend
# proxy into INR for reporting (dataset has no absolute currency column, so
# we scale CashbackAmount/OrderCount-based spend into a realistic INR band
# for an Indian e-commerce/retail shopper — documented assumption, not
# fabricated raw data).
INR_SCALE_FACTOR = 850  # ₹ per "unit" of dataset spend proxy

RANDOM_STATE = 42
