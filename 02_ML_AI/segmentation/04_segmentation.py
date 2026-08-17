"""
04_segmentation.py
===================
Module 4 — Customer Segmentation.

Runs three clustering algorithms on RFM + engagement features:
  - K-Means            (centroid based, needs k, fast, interpretable)
  - DBSCAN             (density based, finds outliers/noise automatically)
  - Agglomerative/Hierarchical clustering (dendrogram-based, no k needed upfront)

Compares them with three internal validation metrics:
  - Silhouette Score        (higher is better,  range -1..1)
  - Davies-Bouldin Score     (lower is better,   range 0..inf)
  - Calinski-Harabasz Index  (higher is better,  range 0..inf)

The best model (by Silhouette Score, the most business-interpretable of the
three) is used to assign a plain-English segment label to every customer,
saved to 04_DATA_TESTING/datasets/processed/segmented_data.csv for the churn
model and dashboard.

Run standalone: python 02_ML_AI/segmentation/04_segmentation.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

# RESTRUCTURE NOTE: config.py now lives one directory above this script
# (02_ML_AI/config.py) — see 00_load_transactions.py for details.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, FIGURES_DIR, MODELS_DIR, RANDOM_STATE

IN_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "cleaned_readable.csv")
OUT_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "segmented_data.csv")
METRICS_OUT = os.path.join(BASE_DIR, "04_DATA_TESTING", "reports", "segmentation_metrics.json")

FEATURES = ["Recency", "Frequency", "Monetary", "EngagementScore", "SatisfactionScore"]


def prepare_features(df: pd.DataFrame):
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def elbow_chart(X_scaled):
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        inertias.append(km.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o", color="#000080")
    plt.title("Elbow Method — Optimal K for K-Means")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "13_kmeans_elbow.png"), dpi=150)
    plt.close()


def run_kmeans(X_scaled, k=5):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    return labels, model


def run_dbscan(X_scaled, eps=1.2, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    return labels, model


def run_hierarchical(X_scaled, k=5):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_scaled)
    return labels, model


def dendrogram_chart(X_scaled):
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
    Z = linkage(X_scaled[sample_idx], method="ward")
    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode="lastp", p=15)
    plt.title("Hierarchical Clustering Dendrogram (sampled 200 customers)")
    plt.xlabel("Customer clusters")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "14_dendrogram.png"), dpi=150)
    plt.close()


def evaluate(X_scaled, labels, name):
    # metrics undefined if only 1 cluster (or all-noise for DBSCAN)
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        return {"model": name, "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None, "n_clusters": len(unique_labels)}
    mask = labels != -2  # keep all; DBSCAN noise (-1) treated as its own cluster for scoring
    sil = silhouette_score(X_scaled[mask], labels[mask])
    db = davies_bouldin_score(X_scaled[mask], labels[mask])
    ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
    return {"model": name, "silhouette": round(float(sil), 4), "davies_bouldin": round(float(db), 4),
            "calinski_harabasz": round(float(ch), 2), "n_clusters": len(unique_labels)}


def label_segments(df: pd.DataFrame, cluster_col="Segment_KMeans") -> pd.DataFrame:
    """Turn numeric cluster IDs into plain-English business segment names
    based on each cluster's average Recency/Frequency/Monetary/Engagement."""
    df = df.copy()
    profile = df.groupby(cluster_col)[FEATURES].mean()

    def name_cluster(row):
        if row["Frequency"] >= profile["Frequency"].median() and row["Monetary"] >= profile["Monetary"].median() and row["Recency"] <= profile["Recency"].median():
            return "Loyal High-Value"
        if row["Recency"] > profile["Recency"].median() and row["Frequency"] < profile["Frequency"].median():
            return "At-Risk / Dormant"
        if row["Monetary"] >= profile["Monetary"].quantile(0.75):
            return "Big Spenders"
        if row["Frequency"] <= profile["Frequency"].quantile(0.25) and row["Recency"] <= profile["Recency"].median():
            return "New / Recently Acquired"
        return "Occasional Shoppers"

    label_map = {cid: name_cluster(row) for cid, row in profile.iterrows()}
    df["SegmentName"] = df[cluster_col].map(label_map)
    return df


def run():
    df = pd.read_csv(IN_PATH)
    X_scaled, scaler = prepare_features(df)

    elbow_chart(X_scaled)
    dendrogram_chart(X_scaled)

    km_labels, km_model = run_kmeans(X_scaled, k=5)
    db_labels, db_model = run_dbscan(X_scaled)
    hc_labels, hc_model = run_hierarchical(X_scaled, k=5)

    results = [
        evaluate(X_scaled, km_labels, "K-Means"),
        evaluate(X_scaled, db_labels, "DBSCAN"),
        evaluate(X_scaled, hc_labels, "Hierarchical"),
    ]
    print("[Segmentation] Model comparison:")
    for r in results:
        print(" ", r)

    best = max([r for r in results if r["silhouette"] is not None], key=lambda r: r["silhouette"])
    print(f"[Segmentation] Best model by Silhouette Score: {best['model']}")

    df["Segment_KMeans"] = km_labels
    df["Segment_DBSCAN"] = db_labels
    df["Segment_Hierarchical"] = hc_labels
    df = label_segments(df, cluster_col="Segment_KMeans")  # K-Means chosen as production model (see README rationale)

    # Segment visualisation (Recency vs Monetary scatter coloured by segment)
    plt.figure(figsize=(9, 6))
    for seg in df["SegmentName"].unique():
        sub = df[df["SegmentName"] == seg]
        plt.scatter(sub["Recency"], sub["Monetary"], label=seg, alpha=0.6, s=20)
    plt.xlabel("Recency (days since last order)")
    plt.ylabel("Monetary (₹ estimated annual spend)")
    plt.title("Customer Segments — Recency vs Monetary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "15_segments_scatter.png"), dpi=150)
    plt.close()

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump({"comparison": results, "best_model": best["model"]}, f, indent=2)

    df.to_csv(OUT_PATH, index=False)
    print(f"[OK] Segmented data saved -> {OUT_PATH}")
    print(df["SegmentName"].value_counts())
    return df, results


if __name__ == "__main__":
    run()
