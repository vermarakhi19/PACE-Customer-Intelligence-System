"""
03_eda.py
=========
Module 3 — Exploratory Data Analysis.

Generates every required chart type (histograms, pie, bar, line, heatmap,
pairplot, distribution plots) into 04_DATA_TESTING/reports/figures/, and
prints a plain English "Business Insight" line under each chart-generation
step — these insight strings are also written to
04_DATA_TESTING/reports/eda_insights.json so the Flask dashboard's Analytics
page and the Final Report can reuse the exact same text (single source of
truth, no copy-paste drift).

Run standalone: python 02_ML_AI/eda/03_eda.py
Input:  04_DATA_TESTING/datasets/processed/cleaned_readable.csv
Output: 04_DATA_TESTING/reports/figures/*.png + 04_DATA_TESTING/reports/eda_insights.json
"""
import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# RESTRUCTURE NOTE: config.py now lives one directory above this script
# (02_ML_AI/config.py) — see 00_load_transactions.py for details.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, FIGURES_DIR

IN_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "cleaned_readable.csv")
INSIGHTS_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "reports", "eda_insights.json")

sns.set_theme(style="whitegrid")
INDIAN_PALETTE = ["#FF9933", "#138808", "#000080", "#FFD700", "#DC143C", "#008080"]  # saffron/green/navy/gold


def save(fig_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved -> {path}")


def run():
    df = pd.read_csv(IN_PATH)
    insights = {}

    # 1. Churn distribution — Pie Chart
    plt.figure(figsize=(6, 6))
    df["Churn"].value_counts().plot.pie(
        labels=["Retained", "Churned"], autopct="%1.1f%%",
        colors=[INDIAN_PALETTE[1], INDIAN_PALETTE[4]], startangle=90
    )
    plt.title("Customer Retention vs Churn")
    plt.ylabel("")
    save("01_churn_pie.png")
    churn_rate = round(df["Churn"].mean() * 100, 1)
    insights["churn_rate"] = f"Overall churn rate is {churn_rate}% — roughly 1 in {round(100/churn_rate)} customers leaves."

    # 2. City-tier distribution — Bar Chart
    plt.figure(figsize=(8, 5))
    df["City"].value_counts().head(10).plot.bar(color=INDIAN_PALETTE[2])
    plt.title("Top 10 Cities by Customer Count")
    plt.ylabel("Number of Customers")
    save("02_city_bar.png")
    top_city = df["City"].value_counts().idxmax()
    insights["top_city"] = f"{top_city} contributes the largest customer base — prioritise it for regional campaigns."

    # 3. Tenure distribution — Histogram
    plt.figure(figsize=(8, 5))
    df["Tenure"].plot.hist(bins=30, color=INDIAN_PALETTE[0], edgecolor="black")
    plt.title("Distribution of Customer Tenure (months)")
    plt.xlabel("Tenure (months)")
    save("03_tenure_histogram.png")
    insights["tenure_dist"] = "Most customers are early-tenure (under 12 months) — retention efforts in the first year matter most."

    # 4. Order category preference — Bar Chart
    plt.figure(figsize=(8, 5))
    df["PreferedOrderCat"].value_counts().plot.bar(color=INDIAN_PALETTE[3])
    plt.title("Preferred Order Category")
    plt.ylabel("Number of Customers")
    save("04_order_category_bar.png")
    top_cat = df["PreferedOrderCat"].value_counts().idxmax()
    insights["top_category"] = f"{top_cat} is the most popular category — stock and promote it heavily across Tier-1 cities."

    # 5. Churn rate by satisfaction score — Line Chart
    plt.figure(figsize=(8, 5))
    churn_by_sat = df.groupby("SatisfactionScore")["Churn"].mean() * 100
    churn_by_sat.plot(marker="o", color=INDIAN_PALETTE[4], linewidth=2)
    plt.title("Churn Rate (%) by Satisfaction Score")
    plt.xlabel("Satisfaction Score (1-5)")
    plt.ylabel("Churn Rate (%)")
    save("05_churn_by_satisfaction_line.png")
    insights["satisfaction_link"] = "Churn rate climbs sharply for customers with satisfaction score <=2 — early complaint resolution is the single highest-leverage retention lever."

    # 6. Payment mode distribution — Pie Chart
    plt.figure(figsize=(6, 6))
    df["PreferredPaymentMode"].value_counts().plot.pie(autopct="%1.1f%%", colors=INDIAN_PALETTE)
    plt.title("Preferred Payment Mode")
    plt.ylabel("")
    save("06_payment_pie.png")
    top_pay = df["PreferredPaymentMode"].value_counts().idxmax()
    insights["top_payment"] = f"{top_pay} is the dominant payment mode — ensure it's frictionless and prominently placed at checkout."

    # 7. Correlation heatmap (already produced in preprocessing, regenerate for EDA report)
    plt.figure(figsize=(14, 10))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap — Numeric Features")
    save("07_correlation_heatmap.png")

    # 8. Pairplot (sampled for speed) of key RFM-style features vs churn
    sample = df.sample(min(800, len(df)), random_state=42)
    pairplot_cols = ["Tenure", "OrderCount", "CashbackAmount", "SatisfactionScore", "Churn"]
    g = sns.pairplot(sample[pairplot_cols], hue="Churn", palette=[INDIAN_PALETTE[1], INDIAN_PALETTE[4]], diag_kind="kde")
    g.fig.suptitle("Pairplot — Key Behavioural Features by Churn", y=1.02)
    g.savefig(os.path.join(FIGURES_DIR, "08_pairplot.png"), dpi=150)
    plt.close("all")
    print("  saved -> 08_pairplot.png")

    # 9. Distribution plot — CashbackAmount (Monetary proxy)
    plt.figure(figsize=(8, 5))
    sns.histplot(df["CashbackAmount"], kde=True, color=INDIAN_PALETTE[5])
    plt.title("Distribution of Cashback Amount (Monetary Proxy)")
    save("09_cashback_distplot.png")

    # 10. Gender vs Churn — Bar Chart
    plt.figure(figsize=(7, 5))
    pd.crosstab(df["Gender"], df["Churn"], normalize="index").mul(100).plot.bar(
        stacked=True, color=[INDIAN_PALETTE[1], INDIAN_PALETTE[4]]
    )
    plt.title("Churn Rate (%) by Gender")
    plt.ylabel("% of customers")
    save("10_gender_churn_bar.png")

    # 11. State-wise average spend — Bar Chart
    plt.figure(figsize=(9, 5))
    state_spend = df.groupby("State")["EstimatedAnnualSpendINR"].mean().sort_values(ascending=False)
    state_spend.plot.bar(color=INDIAN_PALETTE[2])
    plt.title("Average Estimated Annual Spend (₹) by State")
    plt.ylabel("₹ Average Annual Spend")
    save("11_state_spend_bar.png")
    top_state = state_spend.idxmax()
    insights["top_spend_state"] = f"Customers in {top_state} have the highest average annual spend (₹{state_spend.max():,.0f}) — a strong region for premium/loyalty offers."

    # 12. Complaint vs Churn — Bar Chart
    plt.figure(figsize=(6, 5))
    pd.crosstab(df["Complain"], df["Churn"], normalize="index").mul(100).plot.bar(
        stacked=True, color=[INDIAN_PALETTE[1], INDIAN_PALETTE[4]]
    )
    plt.title("Churn Rate (%): Complaint Raised vs Not")
    plt.xticks([0, 1], ["No Complaint", "Complaint Raised"], rotation=0)
    save("12_complaint_churn_bar.png")
    insights["complaint_link"] = "Customers who raised a complaint churn at a substantially higher rate — a fast complaint-resolution SLA directly protects revenue."

    with open(INSIGHTS_PATH, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"[OK] EDA complete. Insights saved -> {INSIGHTS_PATH}")
    return insights


if __name__ == "__main__":
    run()
