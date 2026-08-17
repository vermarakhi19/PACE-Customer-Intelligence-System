# 02_ML_AI — Data Pipeline, Segmentation, Churn Prediction & Forecasting

Owner: **ML/AI** team member(s). This folder owns everything that turns raw
data into the CSV/JSON/`.pkl` files the backend serves. It contains no Flask
routes and no HTML/CSS/JS.

## Structure

```
02_ML_AI/
├── config.py               — shared paths + Indian-context lookup tables (single source of truth)
├── run_pipeline.py         — runs all 9 stages below, in order
├── data_loading/
│   ├── 00_load_transactions.py   — optional real transaction layer (Online Retail II)
│   └── 01_data_loading.py        — loads + validates the Kaggle churn dataset, adds identity fields
├── preprocessing/
│   └── 02_preprocessing.py       — cleaning, outliers, encoding, scaling, feature engineering
├── eda/
│   └── 03_eda.py                 — 12 charts + written insights
├── segmentation/
│   └── 04_segmentation.py        — K-Means / DBSCAN / Hierarchical clustering + comparison
├── churn_prediction/
│   └── 05_churn_prediction.py    — 7-model churn classifier comparison, saves the best one
├── recommendations/
│   ├── 06_business_recommendations.py  — per-customer/segment retention actions
│   └── 08_recommendations.py           — kNN + segment-affinity product recommender
├── forecasting/
│   └── 07_sales_forecasting.py   — customer lifecycle revenue curve + forecast
├── models/                 — trained model artifacts (`.pkl`), gitignored
└── README.md                — this file
```

**Why `config.py` and `run_pipeline.py` sit directly in `02_ML_AI/` instead
of inside one of the stage subfolders:** both are genuinely cross-stage —
`config.py` is imported by all nine scripts, and `run_pipeline.py`
orchestrates all nine. Forcing either one into, say, `data_loading/` would
be an arbitrary, confusing choice. Keeping them at the `02_ML_AI/` root is
the same pattern the project already used (`src/config.py`,
`src/run_pipeline.py` sat above the individual stage scripts before this
reorg too) — mapped one level down `02_ML_AI/` rather than into any single
stage folder. Every stage script's own file was **not rewritten** — only its
file-path constants and one `sys.path` bootstrap line were updated so
`from config import ...` still resolves now that `config.py` is a directory
level away. See the "RESTRUCTURE NOTE" comment at the top of each file.

## ML pipeline (run in this order — `run_pipeline.py` does this automatically)

1. **00 — Transactions (optional)**: real invoice-level data (Online Retail
   II), if you place it at `04_DATA_TESTING/datasets/raw/online_retail_II.xlsx`.
   Powers monthly revenue trends, real Country geography and market-basket
   pairs. Skips gracefully (no crash, dashboard shows "unavailable") if the
   file isn't present — this project's core numbers do not depend on it.
2. **01 — Data loading**: loads the Kaggle "Ecommerce Customer Churn" CSV
   from `04_DATA_TESTING/datasets/raw/ecommerce_churn.csv`, validates its 20
   expected columns, adds a cosmetic Indian identity layer (name/city/
   state/phone/signup date) — never touches the real analytical columns.
3. **02 — Preprocessing**: dedup, missing-value handling, IQR outlier
   capping, encoding, scaling, feature engineering (RFM-style features,
   `EstimatedAnnualSpendINR`, `EngagementScore`, etc.).
4. **03 — EDA**: 12 chart PNGs + a JSON of plain-English insight strings
   (single source of truth reused by the dashboard's Reports page).
5. **04 — Segmentation**: K-Means, DBSCAN, Hierarchical clustering compared
   on Silhouette / Davies-Bouldin / Calinski-Harabasz; best model's clusters
   get plain-English segment names.
6. **05 — Churn prediction**: Logistic Regression, Decision Tree, Random
   Forest, SVM (+ XGBoost/LightGBM/CatBoost when installed) compared on
   Accuracy/Precision/Recall/F1/ROC-AUC; best model (by F1) is saved to
   `02_ML_AI/models/best_churn_model.pkl` for the dashboard.
7. **06 — Business recommendations**: combines Segment + churn probability
   into a per-customer recommended action.
8. **07 — Forecasting**: real customer-lifecycle revenue curve (spend by
   tenure month), forecast forward with Holt's linear trend.
9. **08 — Product recommendations**: kNN behavioural-similarity + segment
   affinity, hybrid top-2 product-category recommendation per customer.

### Input data

Place the raw Kaggle CSV at `04_DATA_TESTING/datasets/raw/ecommerce_churn.csv`
(a zipped copy of the source `.xlsx` ships in that folder — convert it to CSV
first; see `04_DATA_TESTING/documentation/INSTALLATION.md`).

### Features / RFM

Recency, Frequency, Monetary (RFM) plus `EngagementScore` and
`SatisfactionScore` drive segmentation; the churn model uses the full
preprocessed feature set (see `04_DATA_TESTING/documentation/DATA_DICTIONARY.md`
for the complete column list).

### Model outputs

- `02_ML_AI/models/best_churn_model.pkl`, `feature_columns.pkl`
- `04_DATA_TESTING/datasets/processed/*.csv` (cleaned, segmented, final,
  recommendations)
- `04_DATA_TESTING/reports/*.json` (metrics, insights, forecasts)
- `04_DATA_TESTING/reports/figures/*.png` (18 charts)

### How the backend should consume ML outputs

Read-only, from disk — see `01_BACKEND/README.md`. The backend never imports
these scripts; it only reads the files they produce. If you change a column
name, JSON key, or output file name here, you must coordinate with the
backend owner (and usually the frontend owner, since templates render those
exact keys).

## Run it

```bash
cd PACE-Customer-Intelligence-System
python 02_ML_AI/run_pipeline.py
```

Takes 1-3 minutes on the bundled ~5,600-row dataset. Verified working in this
restructure using only pandas/numpy/scikit-learn/scipy/matplotlib/seaborn —
XGBoost/LightGBM/CatBoost/imbalanced-learn are optional; the churn-prediction
stage detects their absence, prints a warning, and falls back to
`class_weight="balanced"` instead of SMOTE and to the 4 always-available
classifiers, rather than failing.
