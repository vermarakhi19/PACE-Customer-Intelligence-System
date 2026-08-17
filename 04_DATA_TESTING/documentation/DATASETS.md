# PACE / RetailPulse — Dataset Setup

PACE now runs on a **hybrid of two real datasets**. Neither one alone can
support every module, and the split is deliberate.

---

## 1. Churn dataset (required) — already included

**E Commerce Dataset** — `04_DATA_TESTING/datasets/raw/ecommerce_churn.csv.zip`
(Kaggle: "E Commerce Dataset" by ankitverma2010)

| | |
|---|---|
| Records | 5,630 customers × 20 columns |
| Grain | One row per customer (a snapshot) |
| Real churn label | ✅ `Churn` — 16.84% churn rate |
| Dates | ❌ none |
| Revenue | ❌ none (`CashbackAmount` only) |
| Geography | `CityTier` (1/2/3) only — **no city names** |

**Setup:** unzip it and save the `E Comm` sheet as
`04_DATA_TESTING/datasets/raw/ecommerce_churn.csv`.

**Modules using it:** 02 preprocessing, 03 EDA, 04 segmentation,
05 churn prediction, 06 business recommendations.

---

## 2. Transactional dataset (recommended) — you must download

**Online Retail II** — real transactions from a UK-based non-store online retailer.

- Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- UCI: https://archive.ics.uci.edu/dataset/502/online+retail+ii

Save as `04_DATA_TESTING/datasets/raw/online_retail_II.xlsx` (the loader
also accepts `.csv` and several filename/column-name variants).

**Columns:** Invoice, StockCode, Description, Quantity, InvoiceDate,
Price, Customer ID, Country

**What it unlocks that the churn dataset cannot:**

| Capability | Why it needs this dataset |
|---|---|
| True RFM | Real invoice dates + real line-item revenue |
| Month-over-month KPI deltas | Requires two comparable periods |
| Revenue forecasting | Requires a dated revenue series |
| Market-basket recommendations | Requires multi-item baskets |
| Real geography | `Country` — the only real location field available |

**The pipeline runs without it.** Module 00 prints a skip notice and the
transaction-derived panels hide themselves rather than display invented
numbers.

---

## Known limitations (state these in your report — do not paper over them)

1. **No city-level data exists in either dataset.** Earlier versions of this
   project assigned Indian city names at random from a tier lookup. That was
   removed. The dashboard now ranks real **countries** by real customer counts.

2. **`EstimatedAnnualSpendINR` is a proxy**, not observed revenue:
   `OrderCount × CashbackAmount × scale / 10`. The churn dataset has no spend
   column. Real revenue is available only on the Online Retail II side.

3. **The two datasets describe different customer populations** and are not
   joined at customer level. They are used as parallel evidence sources:
   the labelled snapshot for churn/segmentation, the transaction log for
   time-based and basket-based analysis. Do not present a figure from one as
   if it described the other.

4. **The final month of Online Retail II is partial** (data ends 09/12/2011).
   Module 00 detects this and excludes it from month-over-month deltas, since
   comparing a 9-day month against a full one produces a large false decline.

5. **Churn label vs. transactional data:** Online Retail II has no churn
   label. If you later want churn on the transactional side, it must be
   *derived* (e.g. no purchase within N days of the snapshot) — a standard,
   defensible definition, but a derived one that must be stated as such.

---

## Optional Python packages

The pipeline degrades gracefully if these are missing, and says so in its output:

- `imbalanced-learn` — SMOTE. Without it, models fall back to
  `class_weight="balanced"`. The imbalance is still handled.
- `xgboost`, `lightgbm`, `catboost` — extra classifiers in the comparison.
  Without them the comparison runs on the scikit-learn models only.
- `statsmodels` — adds a Holt-Winters comparison model to forecasting.

```bash
pip install imbalanced-learn xgboost lightgbm catboost statsmodels
```
