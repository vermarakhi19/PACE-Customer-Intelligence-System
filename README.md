# RetailPulse India
### AI-Powered Customer Segmentation & Customer Retention Prediction System for Indian Retail Businesses

A final-year, production-style ML system that segments Indian retail
customers, predicts who is about to churn, and turns both into concrete,
Indian-context business actions — surfaced through a Flask web dashboard.

---

## 1. Business Problem

Indian retail and e-commerce businesses (D-Mart, JioMart, Blinkit, Flipkart,
etc.) acquire customers cheaply but lose a large share of them silently —
there is usually no early-warning signal before a customer stops ordering.
Two questions drive this project:

1. **Segmentation** — Who are our customers, in plain business language
   (Loyal High-Value, At-Risk, Big Spenders, New, Occasional)?
2. **Retention** — Which specific customers are about to churn, *before*
   they actually leave, so the business can intervene with a targeted,
   costed action (coupon, call, loyalty tier) instead of a blanket discount?

## 2. Dataset

| Field | Value |
|---|---|
| Name | Ecommerce Customer Churn Analysis and Prediction |
| Source | Kaggle — `ankitverma2010/ecommerce-customer-churn-analysis-and-prediction` |
| Link | https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction |
| Rows | 5,630 customers |
| Columns | 20 (raw) → 27 after Indian-context enrichment → 34+ after feature engineering |
| Target | `Churn` (1 = churned, 0 = retained) |
| Missing values | Present in 7 numeric columns (~5-7% each) — handled in preprocessing |

Full column-by-column data dictionary: [`04_DATA_TESTING/documentation/DATA_DICTIONARY.md`](04_DATA_TESTING/documentation/DATA_DICTIONARY.md)

**This repository does not ship the dataset as a ready-to-use CSV** (a zipped
copy of the original `.xlsx` export ships at
`04_DATA_TESTING/datasets/raw/ecommerce_churn.csv.zip`). Convert it (or
re-download from the Kaggle link above) and save it as
`04_DATA_TESTING/datasets/raw/ecommerce_churn.csv` (see
[`04_DATA_TESTING/documentation/INSTALLATION.md`](04_DATA_TESTING/documentation/INSTALLATION.md)).

## 3. Indian Context

The dataset's raw `CityTier` (1/2/3 — India's real official city
classification) is mapped to actual Indian cities (Mumbai, Pune, Nagpur,
Delhi, Hyderabad, Chennai, Bangalore, Ahmedabad, Kolkata, ...) and states.
Every customer also gets an Indian name, a valid-format Indian mobile number
(`+91-XXXXXXXXXX`), and a `DD/MM/YYYY` signup date. All currency in the
dashboard and reports is shown in **₹ (INR)**. See
`02_ML_AI/data_loading/01_data_loading.py` for exactly how this enrichment
works and why it never touches the real analytical columns.

## 4. Architecture

```
Kaggle CSV  (04_DATA_TESTING/datasets/raw/)
   │
   ▼
data_loading/01_data_loading.py     → Indian identity enrichment (name/city/state/phone/date)
   │
   ▼
preprocessing/02_preprocessing.py   → dedupe, missing values, outliers, encoding, scaling,
   │                                   feature engineering (RFM, EngagementScore), correlation
   ▼
eda/03_eda.py                       → 12 charts + business insights (JSON)
   │
   ▼
segmentation/04_segmentation.py     → K-Means vs DBSCAN vs Hierarchical (compared, best picked)
   │
   ▼
churn_prediction/05_churn_prediction.py → 7 classifiers compared (best picked, saved as .pkl)
   │
   ▼
recommendations/06_business_recommendations.py → Segment × Risk → concrete action + personalised recs
   │
   ▼
forecasting/07_sales_forecasting.py → real-data customer lifecycle revenue curve (Tenure-grouped),
   │                                   Holt linear trend forecast, backtested vs naive baseline
   ▼
recommendations/08_recommendations.py → hybrid kNN + segment-affinity product category recommender
   │                                     (all stages above live under 02_ML_AI/)
   ▼
04_DATA_TESTING/datasets/processed/final_customer_data.csv  ──►  Flask dashboard (01_BACKEND/app.py)
   │                                                                using templates/static from 03_FRONTEND/
   ▼
                                                                    USER
```

## 4a. Forecasting & Recommendations — design notes

The raw dataset is one row per customer (a snapshot), not a transaction
log, so two features that would normally need extra data types are built
honestly around that constraint instead of inventing data that doesn't
exist:

- **Sales forecasting** (`07_sales_forecasting.py`) does not predict
  calendar-month sales (no dated transaction history exists to support
  that). Instead it builds a genuine **customer lifecycle revenue curve**
  by grouping real customers by `Tenure` (months since signup) and
  forecasts that real curve forward with a hand-implemented Holt's linear
  trend model (zero extra dependencies), backtested against a naive
  baseline. If `statsmodels` is installed it is used for an additional
  comparison model; the pipeline runs fine without it.
- **Product recommendations** (`08_recommendations.py`) is a
  **content-based hybrid recommender** (k-Nearest-Neighbours on
  behavioural features + a Segment × Category affinity table), not
  collaborative filtering — the dataset has no basket/transaction log to
  support that, so it isn't claimed.

## 5. Folder Structure — Four Team Work Areas

This repository is organized into **four clearly separated work areas**, one
per team member, inside a single repo — no branches, no separate
repositories, no duplicated project copies. Each folder has its own
`README.md` with full ownership, run instructions and a "which files are
safe to modify" note.

```
PACE-Customer-Intelligence-System/
├── 01_BACKEND/               # Backend owner — Flask app, routes, APIs
│   ├── app.py
│   ├── routes/                 # reserved for future route modules (empty)
│   ├── services/                # reserved for future service modules (empty)
│   └── README.md
│
├── 02_ML_AI/                 # ML/AI owner — full data + modeling pipeline
│   ├── config.py
│   ├── run_pipeline.py
│   ├── data_loading/            # 00_load_transactions.py, 01_data_loading.py
│   ├── preprocessing/           # 02_preprocessing.py
│   ├── eda/                     # 03_eda.py
│   ├── segmentation/            # 04_segmentation.py
│   ├── churn_prediction/        # 05_churn_prediction.py
│   ├── forecasting/             # 07_sales_forecasting.py
│   ├── recommendations/         # 06_business_recommendations.py, 08_recommendations.py
│   ├── models/                  # trained .pkl (generated, gitignored)
│   └── README.md
│
├── 03_FRONTEND/               # Frontend owner — templates & static assets
│   ├── templates/                # 13 Jinja2 templates (Bootstrap 5)
│   ├── static/{css,js}/
│   └── README.md
│
├── 04_DATA_TESTING/           # Data + Testing owner — datasets, tests, docs, reports
│   ├── datasets/{raw,processed}/
│   ├── validation/
│   ├── testing/                  # smoke_test_flask_app.py
│   ├── reports/{figures,*.json}
│   ├── documentation/            # every project doc (was docs/)
│   └── README.md
│
├── README.md                  # this file (shared, root-level)
├── requirements.txt            # shared, root-level
├── FINAL_REPORT.md             # shared, root-level
├── PPT_CONTENT.md              # shared, root-level
└── .gitignore
```

**Ownership rule:** every production file has exactly one owner folder. No
one edits another member's folder directly — if integration requires a
change in another folder (e.g. a new JSON key the frontend needs to render),
document the dependency in that folder's README and coordinate with its
owner, the same discipline as if these were separate branches.

**Data flow across the four folders** (also see the architecture diagram
above):

```
04_DATA_TESTING  (raw data)
      │
      ▼
02_ML_AI          (pipeline: cleans, segments, predicts, forecasts, recommends)
      │
      ▼
01_BACKEND         (reads pipeline output, serves it as routes/JSON)
      │
      ▼
03_FRONTEND         (renders it as pages)
      │
      ▼
     USER
```

## 6. Quickstart

The application still runs as one working app from the project root — only
the internal file locations changed.

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Get the Kaggle dataset as a CSV and save it as:
#    04_DATA_TESTING/datasets/raw/ecommerce_churn.csv
#    (a zipped copy of the original .xlsx ships at
#     04_DATA_TESTING/datasets/raw/ecommerce_churn.csv.zip — convert it)

# 2. Run the full ML pipeline (9 stages)
python 02_ML_AI/run_pipeline.py

# 3. (optional) Run the smoke test to confirm every page responds
python 04_DATA_TESTING/testing/smoke_test_flask_app.py

# 4. Launch the dashboard
python 01_BACKEND/app.py
# open http://127.0.0.1:5000   (login: admin / admin123)
```

Full step-by-step guide: [`04_DATA_TESTING/documentation/INSTALLATION.md`](04_DATA_TESTING/documentation/INSTALLATION.md)

## 7. Models

**Segmentation:** K-Means, DBSCAN, Agglomerative/Hierarchical — compared on
Silhouette Score, Davies-Bouldin Score, Calinski-Harabasz Index.

**Retention prediction:** Logistic Regression, Decision Tree, Random
Forest, SVM, XGBoost, LightGBM, CatBoost — compared on Accuracy, Precision,
Recall, F1, ROC-AUC, with confusion matrices and full classification
reports. SMOTE is applied to the training split only, since real churn
data is imbalanced (~83/17 split in this dataset).

Why the best model is selected the way it is, per algorithm family, is
explained in [`FINAL_REPORT.md`](FINAL_REPORT.md) §5.

## 8. Dashboard Pages

Login · Dashboard · Customer List · Segmentation · Prediction · Analytics ·
Model Comparison · Forecasting · Recommendations · Reports · Settings —
with KPI cards, Chart.js charts, DataTables search/pagination, dark mode,
and Excel/PDF export.

## 9. License / Academic Use

Built as a final-year academic project. The dataset is used under Kaggle's
dataset terms (see the dataset page for the exact license) — always check
and cite it in your submitted report.
