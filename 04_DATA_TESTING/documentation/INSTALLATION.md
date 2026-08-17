# Installation Guide

## Prerequisites
- Python 3.10+
- pip
- ~500 MB free disk space (libraries + generated charts/models)
- A free Kaggle account (to download the dataset)

## Step 1 — Clone / unzip the project
```bash
cd PACE-Customer-Intelligence-System
```

## Step 2 — Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

## Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
If `catboost` or `lightgbm` fail to build on your machine, install the
prebuilt wheels individually:
```bash
pip install catboost --only-binary=:all:
pip install lightgbm --only-binary=:all:
```

## Step 4 — Download the dataset
1. Go to https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
2. Click **Download** (Kaggle login required)
3. If it downloads as `.xlsx`, convert to CSV:
   ```python
   import pandas as pd
   pd.read_excel("E Commerce Dataset.xlsx", sheet_name="E Comm").to_csv("ecommerce_churn.csv", index=False)
   ```
4. Save the file as `04_DATA_TESTING/datasets/raw/ecommerce_churn.csv`
   (a zipped copy of the original `.xlsx` already ships at
   `04_DATA_TESTING/datasets/raw/ecommerce_churn.csv.zip` if you'd rather
   convert that one than re-download)

## Step 5 — Run the ML pipeline
```bash
python 02_ML_AI/run_pipeline.py
```
This runs all 9 modules in order and populates
`04_DATA_TESTING/datasets/processed/`, `04_DATA_TESTING/reports/`, and
`02_ML_AI/models/`. Takes 1-3 minutes depending on your machine.

## Step 6 — (optional) Run the smoke test
```bash
python 04_DATA_TESTING/testing/smoke_test_flask_app.py
```
Confirms every page responds before you open a browser.

## Step 7 — Launch the dashboard
```bash
python 01_BACKEND/app.py
```
Open **http://127.0.0.1:5000** and log in with `admin` / `admin123`.

## Troubleshooting
| Problem | Fix |
|---|---|
| `FileNotFoundError: raw dataset not found` | Confirm the CSV is at exactly `04_DATA_TESTING/datasets/raw/ecommerce_churn.csv` |
| `ValueError: Dataset schema mismatch` | You downloaded a different Kaggle dataset — re-check the link in README |
| Dashboard shows "No data yet" | Run `python 02_ML_AI/run_pipeline.py` before starting the Flask app |
| `catboost`/`lightgbm` install errors | Use `--only-binary=:all:` as shown above, or run on Python 3.10-3.11 |
