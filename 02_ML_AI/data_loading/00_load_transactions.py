"""
00_load_transactions.py
========================
Module 0 — Real transactional layer (Online Retail II).

WHY THIS MODULE EXISTS
-----------------------
The original churn dataset (E Commerce Dataset.xlsx) is a per-customer
SNAPSHOT: it has a real churn label, but no transaction log, no dates, no
revenue column and no city. That forced several dashboard numbers to be
proxies or, worse, randomly generated placeholders.

This module adds a genuinely transactional dataset alongside it, so the
following become REAL instead of invented:

  * Recency / Frequency / Monetary  -> computed from actual invoice dates
                                       and actual line-item revenue
  * Monthly revenue series          -> real time series for forecasting (07)
  * Market-basket co-occurrence     -> real product pairs for the recommender (08)
  * Month-over-month KPI deltas     -> real period-over-period comparison
  * Geography                       -> real Country values (NOT random cities)

DATASET
-------
Online Retail II (UCI / Kaggle). Real transactions from a UK-based
non-store online retailer.
  Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
  UCI:    https://archive.ics.uci.edu/dataset/502/online+retail+ii

Expected columns (the loader is tolerant of the common naming variants):
  Invoice / InvoiceNo, StockCode, Description, Quantity,
  InvoiceDate, Price / UnitPrice, Customer ID / CustomerID, Country

Place the file at either of:
  04_DATA_TESTING/datasets/raw/online_retail_II.xlsx
  04_DATA_TESTING/datasets/raw/online_retail_II.csv

HONESTY NOTE
------------
This dataset has NO churn label and NO city column. Churn stays sourced from
the original labelled dataset; geography is reported at Country level, which
is the only real geography present. Nothing here invents values.

Run standalone: python 02_ML_AI/run_pipeline.py  (or via 02_ML_AI/data_loading/00_load_transactions.py directly)
Outputs:
  04_DATA_TESTING/datasets/processed/transactions_clean.csv
  04_DATA_TESTING/datasets/processed/customer_rfm.csv
  04_DATA_TESTING/reports/monthly_revenue.json
  04_DATA_TESTING/reports/country_stats.json
  04_DATA_TESTING/reports/basket_affinity.json
"""
import os
import sys
import json
import itertools
from collections import Counter

import numpy as np
import pandas as pd

# RESTRUCTURE NOTE: this script now lives in 02_ML_AI/data_loading/, one
# level deeper than the old flat src/ layout, so config.py (which lives
# directly in 02_ML_AI/) is no longer on sys.path by default. This makes
# both `python 02_ML_AI/run_pipeline.py` and running this file standalone
# work the same way as before.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

RAW_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "raw")
PROC_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING", "reports")

CANDIDATE_FILES = [
    "online_retail_II.xlsx", "online_retail_II.csv",
    "online_retail_ii.xlsx", "online_retail_ii.csv",
    "Online Retail II.xlsx", "online_retail.xlsx", "online_retail.csv",
]

# Column-name variants seen across the Kaggle/UCI distributions
COLUMN_ALIASES = {
    "Invoice": ["Invoice", "InvoiceNo", "invoice", "invoice_no"],
    "StockCode": ["StockCode", "stock_code", "stockcode"],
    "Description": ["Description", "description"],
    "Quantity": ["Quantity", "quantity"],
    "InvoiceDate": ["InvoiceDate", "invoice_date", "Date"],
    "Price": ["Price", "UnitPrice", "unit_price", "price"],
    "CustomerID": ["Customer ID", "CustomerID", "customer_id", "Customer Id"],
    "Country": ["Country", "country"],
}

TOP_N_BASKET_PRODUCTS = 60   # cap co-occurrence matrix size (O(n^2) pairs)
MIN_BASKET_SUPPORT = 5       # ignore product pairs seen fewer than this many times


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def find_raw_file():
    for name in CANDIDATE_FILES:
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _resolve_columns(df):
    """Map whatever naming variant the file uses onto our canonical names."""
    rename = {}
    lower_map = {c.lower().strip(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            actual = lower_map.get(alias.lower())
            if actual is not None:
                rename[actual] = canonical
                break
    df = df.rename(columns=rename)
    missing = [c for c in COLUMN_ALIASES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Transaction file is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def load_raw_transactions(path):
    if path.lower().endswith((".xlsx", ".xls")):
        # Online Retail II ships as two sheets (two year-periods); concat both.
        xl = pd.ExcelFile(path)
        frames = [xl.parse(sheet) for sheet in xl.sheet_names]
        df = pd.concat(frames, ignore_index=True)
        print(f"[Load] Read {len(xl.sheet_names)} sheet(s) from {os.path.basename(path)}")
    else:
        df = pd.read_csv(path, encoding="ISO-8859-1")
    return _resolve_columns(df)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_transactions(df):
    """Standard, documented Online Retail cleaning. Every rule below removes
    rows that are genuinely not customer purchases — none of it fabricates."""
    start = len(df)
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Cancellations are invoices prefixed with 'C' — real returns, not purchases.
    df["IsCancellation"] = df["Invoice"].astype(str).str.upper().str.startswith("C")
    n_cancel = int(df["IsCancellation"].sum())
    df = df[~df["IsCancellation"]]

    # Rows without a customer can't be attributed to a customer at all.
    n_nocust = int(df["CustomerID"].isna().sum())
    df = df[df["CustomerID"].notna()]

    # Non-positive quantity/price are returns, adjustments or data errors.
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[df["InvoiceDate"].notna()]

    df["CustomerID"] = df["CustomerID"].astype(float).astype(int)
    df["Revenue"] = (df["Quantity"] * df["Price"]).round(2)
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    print(f"[Clean] {start:,} raw rows -> {len(df):,} clean rows "
          f"(removed {n_cancel:,} cancellations, {n_nocust:,} rows with no CustomerID, "
          f"plus non-positive qty/price)")
    return df


# ---------------------------------------------------------------------------
# Real RFM
# ---------------------------------------------------------------------------
def build_customer_rfm(df):
    """True RFM from real dates and real revenue.

    Recency is measured against the dataset's own maximum invoice date (the
    standard approach for a historical extract) rather than today's date,
    because the data ends in the past.
    """
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        TotalItems=("Quantity", "sum"),
        Country=("Country", lambda s: s.mode().iat[0] if len(s.mode()) else "Unknown"),
    ).reset_index()

    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["TenureDays"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["AvgOrderValue"] = (rfm["Monetary"] / rfm["Frequency"]).round(2)

    print(f"[RFM] Built real RFM for {len(rfm):,} customers "
          f"(snapshot date {snapshot.date()})")
    return rfm, snapshot


# ---------------------------------------------------------------------------
# Real monthly revenue series (for forecasting + MoM deltas)
# ---------------------------------------------------------------------------
def build_monthly_revenue(df):
    monthly = df.groupby("InvoiceMonth").agg(
        Revenue=("Revenue", "sum"),
        Orders=("Invoice", "nunique"),
        Customers=("CustomerID", "nunique"),
    ).reset_index().sort_values("InvoiceMonth")

    monthly["Month"] = monthly["InvoiceMonth"].dt.strftime("%Y-%m")

    out = {
        "months": monthly["Month"].tolist(),
        "revenue": [round(float(v), 2) for v in monthly["Revenue"]],
        "orders": [int(v) for v in monthly["Orders"]],
        "customers": [int(v) for v in monthly["Customers"]],
    }

    # These extracts almost always end mid-month (Online Retail II stops on
    # 09/12/2011), so the final bucket covers only part of a month. Comparing
    # a partial month against a full one produces a large fake decline — the
    # kind of number that looks like a real business collapse on a dashboard.
    # Detect it and exclude it from the delta, reporting which months were
    # actually compared so the figure is auditable.
    last_date = df["InvoiceDate"].max()
    days_in_last_month = last_date.days_in_month
    last_month_complete = last_date.day >= days_in_last_month - 2
    out["final_month_partial"] = not last_month_complete
    out["final_month_coverage_days"] = int(last_date.day)

    # Compare only complete months.
    usable = list(range(len(out["months"])))
    if not last_month_complete and len(usable) > 2:
        usable = usable[:-1]

    def pct_change(series, idx):
        if len(idx) < 2:
            return None
        cur, prev = series[idx[-1]], series[idx[-2]]
        if prev == 0:
            return None
        return round((cur - prev) / prev * 100, 1)

    out["mom_delta"] = {
        "revenue_pct": pct_change(out["revenue"], usable),
        "orders_pct": pct_change(out["orders"], usable),
        "customers_pct": pct_change(out["customers"], usable),
        "latest_month": out["months"][usable[-1]] if len(usable) else None,
        "previous_month": out["months"][usable[-2]] if len(usable) > 1 else None,
        "basis": ("complete months only; final partial month excluded"
                  if not last_month_complete else "consecutive complete months"),
    }
    print(f"[Monthly] {len(monthly)} months of real revenue "
          f"({out['months'][0]} .. {out['months'][-1]})" if len(monthly) else "[Monthly] no data")
    return out


# ---------------------------------------------------------------------------
# Real geography (Country — the only real geography in this dataset)
# ---------------------------------------------------------------------------
def build_country_stats(rfm, top_n=6):
    grouped = rfm.groupby("Country").agg(
        customers=("CustomerID", "nunique"),
        avg_spend=("Monetary", "mean"),
        total_revenue=("Monetary", "sum"),
        avg_recency=("Recency", "mean"),
    ).reset_index().sort_values("customers", ascending=False)

    top = grouped.head(top_n)
    stats = [{
        "name": r["Country"],
        "customers": int(r["customers"]),
        "avg_spend": round(float(r["avg_spend"]), 2),
        "total_revenue": round(float(r["total_revenue"]), 2),
        "avg_recency_days": round(float(r["avg_recency"]), 1),
    } for _, r in top.iterrows()]

    print(f"[Geography] Top markets by real customer count: "
          f"{', '.join(s['name'] for s in stats)}")
    return stats


# ---------------------------------------------------------------------------
# Real market-basket affinity (for the recommender)
# ---------------------------------------------------------------------------
def build_basket_affinity(df):
    """Genuine co-occurrence counts from real invoices.

    Restricted to the most popular products so the pair matrix stays tractable;
    this is a performance bound, not a data change.
    """
    top_products = df["Description"].value_counts().head(TOP_N_BASKET_PRODUCTS).index
    sub = df[df["Description"].isin(top_products)]

    pair_counts = Counter()
    item_counts = Counter()
    for _, items in sub.groupby("Invoice")["Description"]:
        unique_items = sorted(set(items.dropna()))
        item_counts.update(unique_items)
        for a, b in itertools.combinations(unique_items, 2):
            pair_counts[(a, b)] += 1

    affinity = {}
    for (a, b), count in pair_counts.items():
        if count < MIN_BASKET_SUPPORT:
            continue
        # confidence(a -> b) = P(b | a)
        for src, dst in ((a, b), (b, a)):
            if item_counts[src] == 0:
                continue
            conf = round(count / item_counts[src] * 100, 1)
            affinity.setdefault(src, []).append({"product": dst, "confidence": conf, "support": count})

    for src in affinity:
        affinity[src] = sorted(affinity[src], key=lambda d: -d["confidence"])[:5]

    print(f"[Basket] {len(affinity)} products have real co-occurrence "
          f"partners (min support {MIN_BASKET_SUPPORT})")
    return affinity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    path = find_raw_file()
    if path is None:
        print(
            "\n[SKIP] No transactional dataset found.\n"
            f"       Looked in: {RAW_DIR}\n"
            f"       For any of: {', '.join(CANDIDATE_FILES[:4])}\n\n"
            "       Download Online Retail II and place it there:\n"
            "         https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci\n\n"
            "       The rest of the pipeline (modules 01-08) still runs without it;\n"
            "       the dashboard will simply hide the transaction-derived panels\n"
            "       rather than show invented numbers.\n"
        )
        return None

    print(f"[Load] Using transactional dataset: {path}")
    df = load_raw_transactions(path)
    df = clean_transactions(df)

    rfm, snapshot = build_customer_rfm(df)
    monthly = build_monthly_revenue(df)
    countries = build_country_stats(rfm)
    affinity = build_basket_affinity(df)

    # Persist
    keep_cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
                 "Price", "CustomerID", "Country", "Revenue", "InvoiceMonth"]
    df[keep_cols].to_csv(os.path.join(PROC_DIR, "transactions_clean.csv"), index=False)
    rfm.to_csv(os.path.join(PROC_DIR, "customer_rfm.csv"), index=False)

    summary = {
        "source_file": os.path.basename(path),
        "dataset": "Online Retail II (UCI/Kaggle) — real transactional data",
        "clean_transactions": int(len(df)),
        "unique_customers": int(df["CustomerID"].nunique()),
        "unique_products": int(df["Description"].nunique()),
        "date_range": [str(df["InvoiceDate"].min().date()), str(df["InvoiceDate"].max().date())],
        "total_revenue": round(float(df["Revenue"].sum()), 2),
        "snapshot_date": str(snapshot.date()),
    }
    monthly["dataset_summary"] = summary

    with open(os.path.join(REPORTS_DIR, "monthly_revenue.json"), "w") as f:
        json.dump(monthly, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "country_stats.json"), "w") as f:
        json.dump({"top_markets": countries, "note":
                   "Country is the only real geography in Online Retail II. "
                   "No city-level data exists in this dataset."}, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "basket_affinity.json"), "w") as f:
        json.dump({"affinity": affinity, "method":
                   "Real invoice co-occurrence. confidence(a->b) = P(b in basket | a in basket)."},
                  f, indent=2)

    print(f"[OK] Transactional layer built: {summary['clean_transactions']:,} rows, "
          f"{summary['unique_customers']:,} customers, "
          f"{summary['date_range'][0]} .. {summary['date_range'][1]}")
    return summary


if __name__ == "__main__":
    run()
