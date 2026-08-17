"""
app.py
======
Flask dashboard — "RetailPulse India" — Customer Segmentation & Retention
Prediction dashboard.

Pages: Login, Dashboard, Customer List, Segmentation, Prediction, Reports,
Analytics, Model Comparison, Settings.

Run:
    cd PACE-Customer-Intelligence-System
    python 01_BACKEND/app.py
Then open http://127.0.0.1:5000  (login: admin / admin123 — change in Settings/.env for real deployment)

RESTRUCTURE NOTE (team-folder reorg)
-------------------------------------
This file used to live at app/app.py with templates/static as siblings in
the same app/ folder (Flask's default template_folder="templates" and
static_folder="static" found them automatically next to app.py). It now
lives at 01_BACKEND/app.py while templates and static assets live under
03_FRONTEND/, so both folders are passed to Flask() explicitly below.
Data/reports/models paths are similarly updated to the new 04_DATA_TESTING/
and 02_ML_AI/ locations. No route, template, or business logic changed.
"""
import os
import io
import json
import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    jsonify, send_file, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "04_DATA_TESTING", "reports")
MODELS_DIR = os.path.join(BASE_DIR, "02_ML_AI", "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "03_FRONTEND", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "03_FRONTEND", "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("SECRET_KEY", "pace-india-dev-secret-change-in-prod")

# Demo credentials — in production these live in a database with hashed
# passwords; kept simple here so the dashboard runs standalone for a college
# project demo without needing a DB server.
USERS = {"admin": generate_password_hash("admin123")}


# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------
def _read_csv_safe(path, empty_cols=None):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=empty_cols or [])


def get_final_data():
    return _read_csv_safe(os.path.join(DATA_DIR, "final_customer_data.csv"))


def get_recommendations_data():
    return _read_csv_safe(os.path.join(DATA_DIR, "customer_recommendations.csv"))


def get_json(name):
    path = os.path.join(REPORTS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Public marketing page
# ---------------------------------------------------------------------------
@app.route("/welcome")
def welcome():
    return render_template("landing.html")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username in USERS and check_password_hash(USERS[username], password):
            session["user"] = username
            return redirect(url_for("dashboard"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
# NOTE: A CITY_COORDS lookup used to live here, mapping randomly-assigned
# Indian city names onto map pins. It was removed deliberately: neither
# dataset contains city-level geography. The churn dataset has only CityTier
# (1/2/3) and Online Retail II has Country. The dashboard now reports real
# Country-level markets from the transactional data instead of inventing
# cities. See reports/country_stats.json.


def _round_dict(d, digits=1):
    return {k: (round(float(v), digits) if pd.notna(v) else 0) for k, v in d.items()}


@app.route("/")
@login_required
def dashboard():
    df = get_final_data()
    churn_metrics_json = get_json("churn_model_metrics.json")
    seg_metrics_json = get_json("segmentation_metrics.json")
    lifecycle = get_json("lifecycle_forecast.json")
    monthly = get_json("monthly_revenue.json")          # real transactional layer
    country_stats = get_json("country_stats.json")

    mom = monthly.get("mom_delta", {}) if monthly else {}
    tx_summary = monthly.get("dataset_summary", {}) if monthly else {}
    has_transactions = bool(monthly.get("months"))

    kpis = {
        "total_customers": len(df),
        "churn_rate": round(df["Churn"].mean() * 100, 1) if "Churn" in df.columns and len(df) else 0,
        "avg_satisfaction": round(df["SatisfactionScore"].mean(), 2) if "SatisfactionScore" in df.columns and len(df) else 0,
        "total_segments": df["SegmentName"].nunique() if "SegmentName" in df.columns and len(df) else 0,
        "high_risk_count": int((df["ChurnRiskBand"] == "High").sum()) if "ChurnRiskBand" in df.columns and len(df) else 0,
        "avg_spend": round(df["EstimatedAnnualSpendINR"].mean(), 0) if "EstimatedAnnualSpendINR" in df.columns and len(df) else 0,
        "best_churn_model": churn_metrics_json.get("best_model", "\u2014"),
        "best_seg_model": seg_metrics_json.get("best_model", "\u2014"),
    }

    segment_counts = df["SegmentName"].value_counts().to_dict() if "SegmentName" in df.columns and len(df) else {}
    risk_counts = df["ChurnRiskBand"].value_counts().to_dict() if "ChurnRiskBand" in df.columns and len(df) else {}

    # Real churn-probability histogram (10 equal-width buckets). Every count is
    # a genuine aggregate of model output.
    churn_prob_hist = {"labels": [], "counts": []}
    if "ChurnProbability" in df.columns and len(df):
        bins = [i / 10 for i in range(11)]
        labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(10)]
        binned = pd.cut(df["ChurnProbability"].clip(0, 1), bins=bins, labels=labels, include_lowest=True)
        counts = binned.value_counts().reindex(labels, fill_value=0)
        churn_prob_hist = {"labels": labels, "counts": [int(c) for c in counts.values]}

    # Real geography: top markets ranked by actual customer count, from the
    # transactional dataset's Country column. Replaces the previous randomly
    # assigned Indian cities.
    top_markets = country_stats.get("top_markets", []) if country_stats else []
    market_max = max((m["customers"] for m in top_markets), default=0)
    for m in top_markets:
        m["share_pct"] = round(m["customers"] / market_max * 100, 1) if market_max else 0

    top_churn_model = next(
        (m for m in churn_metrics_json.get("all_models", []) if m.get("model") == kpis["best_churn_model"]), None
    )
    top_seg_model = next(
        (m for m in seg_metrics_json.get("comparison", []) if m.get("model") == kpis["best_seg_model"]), None
    )

    # ---- KPI sparklines -------------------------------------------------
    # Sourced ONLY from real series. Previously these plotted sorted segment
    # counts, which rendered as meaningless decaying curves; that is removed.
    # A card with no real series behind it simply shows no sparkline.
    sparklines = {}
    deltas = {}
    if has_transactions:
        sparklines["customers"] = monthly.get("customers", [])[-12:]
        sparklines["spend"] = monthly.get("revenue", [])[-12:]
        sparklines["orders"] = monthly.get("orders", [])[-12:]
        if mom.get("customers_pct") is not None:
            deltas["customers"] = {"pct": mom["customers_pct"], "basis": mom.get("basis", "")}
        if mom.get("revenue_pct") is not None:
            deltas["spend"] = {"pct": mom["revenue_pct"], "basis": mom.get("basis", "")}

    # Lifecycle curve is a real model output and is a legitimate sparkline source.
    if lifecycle and lifecycle.get("historical", {}).get("avg_monetary"):
        sparklines["lifecycle"] = lifecycle["historical"]["avg_monetary"][-14:]

    from datetime import datetime
    rendered_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

    return render_template(
        "dashboard.html", kpis=kpis, segment_counts=segment_counts,
        risk_counts=risk_counts, has_data=len(df) > 0,
        churn_prob_hist=churn_prob_hist, top_markets=top_markets,
        top_churn_model=top_churn_model, top_seg_model=top_seg_model,
        sparklines=sparklines, deltas=deltas, rendered_at=rendered_at,
        has_transactions=has_transactions, tx_summary=tx_summary,
        mom=mom,
    )


@app.route("/customers")
@login_required
def customers():
    df = get_final_data()
    cols = [c for c in ["CustomerID", "CustomerName", "City", "State", "PhoneNumber",
                         "PreferedOrderCat", "SegmentName", "ChurnRiskBand",
                         "EstimatedAnnualSpendINR", "Churn"] if c in df.columns]
    records = df[cols].to_dict(orient="records") if len(df) else []
    return render_template("customers.html", customers=records, columns=cols, has_data=len(df) > 0)


@app.route("/segmentation")
@login_required
def segmentation():
    df = get_final_data()
    seg_metrics = get_json("segmentation_metrics.json")
    segment_profile = {}
    if len(df) and "SegmentName" in df.columns:
        segment_profile = df.groupby("SegmentName")[
            ["Recency", "Frequency", "Monetary", "EngagementScore", "SatisfactionScore"]
        ].mean().round(2).to_dict(orient="index")
    return render_template("segmentation.html", seg_metrics=seg_metrics, segment_profile=segment_profile, has_data=len(df) > 0)


@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    result = None
    if request.method == "POST":
        try:
            model_path = os.path.join(MODELS_DIR, "best_churn_model.pkl")
            feat_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
            model = joblib.load(model_path)
            feature_cols = joblib.load(feat_path)

            form_vals = {k: float(v) for k, v in request.form.items() if v not in ("", None)}
            row = pd.DataFrame([form_vals]).reindex(columns=feature_cols, fill_value=0)
            proba = model.predict_proba(row)[0][1]
            result = {
                "churn_probability": round(float(proba) * 100, 1),
                "prediction": "Likely to Churn" if proba >= 0.5 else "Likely to Stay",
                "risk_band": "High" if proba >= 0.66 else ("Medium" if proba >= 0.33 else "Low"),
            }
        except FileNotFoundError:
            flash("Model not trained yet. Run the pipeline (02_ML_AI/run_pipeline.py) first.", "warning")
    return render_template("prediction.html", result=result)


@app.route("/reports")
@login_required
def reports():
    recs = get_json("business_recommendations.json").get("insights", [])
    eda_insights = get_json("eda_insights.json")
    return render_template("reports.html", recommendations=recs, eda_insights=eda_insights)


@app.route("/analytics")
@login_required
def analytics():
    df = get_final_data()
    charts = {}
    if len(df):
        charts["category_dist"] = df["PreferedOrderCat"].value_counts().to_dict()
        charts["payment_dist"] = df["PreferredPaymentMode"].value_counts().to_dict()
        charts["state_spend"] = df.groupby("State")["EstimatedAnnualSpendINR"].mean().round(0).to_dict()
        charts["satisfaction_churn"] = df.groupby("SatisfactionScore")["Churn"].mean().mul(100).round(1).to_dict()
    return render_template("analytics.html", charts=charts, has_data=len(df) > 0)


@app.route("/model-comparison")
@login_required
def model_comparison():
    churn_metrics = get_json("churn_model_metrics.json").get("all_models", [])
    seg_metrics = get_json("segmentation_metrics.json").get("comparison", [])
    return render_template("model_comparison.html", churn_metrics=churn_metrics, seg_metrics=seg_metrics)


@app.route("/forecasting")
@login_required
def forecasting():
    forecast = get_json("lifecycle_forecast.json")
    # Guard on the keys the template actually dereferences, not merely on the
    # dict being non-empty — a partial/older JSON would otherwise 500 the page.
    has_data = bool(
        forecast
        and forecast.get("historical", {}).get("tenure_months")
        and forecast.get("forecast", {}).get("tenure_months")
    )
    return render_template("forecasting.html", forecast=forecast, has_data=has_data)


@app.route("/recommendations")
@login_required
def recommendations():
    df = get_recommendations_data()
    affinity = get_json("segment_category_affinity.json").get("segment_category_affinity_pct", {})
    search = request.args.get("q", "").strip()
    if search and len(df):
        mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False))
        df = df[mask.any(axis=1)]
    records = df.to_dict(orient="records") if len(df) else []
    return render_template(
        "recommendations.html", customers=records, affinity=affinity,
        search=search, has_data=len(get_recommendations_data()) > 0
    )


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html", user=session.get("user"))


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------
@app.route("/export/excel")
@login_required
def export_excel():
    df = get_final_data()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Customers")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="customer_data.xlsx",
                      mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/export/pdf")
@login_required
def export_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    df = get_final_data()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "RetailPulse India — Customer Retention Report")
    y -= 30
    c.setFont("Helvetica", 10)
    if len(df):
        c.drawString(50, y, f"Total Customers: {len(df)}")
        y -= 15
        c.drawString(50, y, f"Churn Rate: {round(df['Churn'].mean()*100, 1)}%")
        y -= 15
        c.drawString(50, y, f"Avg Estimated Annual Spend: Rs. {round(df['EstimatedAnnualSpendINR'].mean(), 0)}")
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="retention_report.pdf", mimetype="application/pdf")


# ---------------------------------------------------------------------------
# JSON APIs (used by Chart.js on the front end for live filtering)
# ---------------------------------------------------------------------------
@app.route("/api/customers")
@login_required
def api_customers():
    df = get_final_data()
    return jsonify(df.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
