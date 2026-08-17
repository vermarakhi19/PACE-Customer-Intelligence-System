"""
05_churn_prediction.py
=======================
Module 5 — Customer Retention Prediction (binary classification: will the
customer stay [0] or leave [1] — the 'Churn' column already published by
Kaggle, so labels are real, not synthetic).

Trains and compares 7 algorithms:
  Logistic Regression, Decision Tree, Random Forest, SVM,
  XGBoost, LightGBM, CatBoost

For each: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix,
full classification report. Saves the best model (by F1, since churn
datasets are imbalanced and F1 balances precision/recall better than
raw accuracy) as 02_ML_AI/models/best_churn_model.pkl for the Flask dashboard.

Run standalone: python 02_ML_AI/churn_prediction/05_churn_prediction.py
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
# ---------------------------------------------------------------------------
# Optional dependencies.
# SMOTE (imblearn) and the gradient-boosting libraries are optional so the
# pipeline still produces a complete, honest model comparison on a machine
# where they are not installed. When a library is absent we say so loudly and
# fall back to a documented alternative (class_weight="balanced" instead of
# SMOTE) rather than silently changing the methodology.
# ---------------------------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    SMOTE = None
    HAS_SMOTE = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# RESTRUCTURE NOTE: config.py now lives one directory above this script
# (02_ML_AI/config.py) — see 00_load_transactions.py for details.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, FIGURES_DIR, MODELS_DIR, RANDOM_STATE

IN_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "cleaned_data.csv")
READABLE_PATH = os.path.join(BASE_DIR, "04_DATA_TESTING", "datasets", "processed", "segmented_data.csv")
METRICS_OUT = os.path.join(BASE_DIR, "04_DATA_TESTING", "reports", "churn_model_metrics.json")

DROP_COLS = ["CustomerID", "CustomerName", "City", "State", "PhoneNumber",
             "SignupDate", "TenureBucket", "SegmentName", "Segment_KMeans",
             "Segment_DBSCAN", "Segment_Hierarchical"]


def load_model_data():
    df = pd.read_csv(IN_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=[np.number])  # safety: only numeric cols reach the models
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def get_models():
    """Every model that is actually available in this environment.

    class_weight="balanced" is applied to the models that support it whenever
    SMOTE is unavailable, so the class imbalance is still addressed.
    """
    balance = None if HAS_SMOTE else "balanced"

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                                   class_weight=balance),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE,
                                                 class_weight=balance),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1,
                                                 random_state=RANDOM_STATE, class_weight=balance),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE,
                   class_weight=balance),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                               eval_metric="logloss", random_state=RANDOM_STATE)
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                                 random_state=RANDOM_STATE, verbosity=-1)
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08,
                                                 random_state=RANDOM_STATE, verbose=0)

    unavailable = [n for n, ok in [("XGBoost", HAS_XGB), ("LightGBM", HAS_LGB),
                                   ("CatBoost", HAS_CATBOOST)] if not ok]
    if unavailable:
        print(f"[WARN] Not installed, skipped from comparison: {', '.join(unavailable)}. "
              f"Install with: pip install {' '.join(u.lower() for u in unavailable)}")
    if not HAS_SMOTE:
        print("[WARN] imblearn not installed - using class_weight='balanced' instead of SMOTE. "
              "Install with: pip install imbalanced-learn")
    return models


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(os.path.join(FIGURES_DIR, f"cm_{safe_name}.png"), dpi=150)
    plt.close()

    report = classification_report(y_test, y_pred, target_names=["Retained", "Churned"], output_dict=True)
    return metrics, y_proba, report


def run():
    X, y = load_model_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Churn datasets are imbalanced (~83% retained / 17% churned here).
    # SMOTE oversamples the minority (churned) class on the TRAIN split only,
    # never on test data, to avoid leaking synthetic rows into evaluation.
    # If imblearn isn't installed we keep the original distribution and rely
    # on class_weight="balanced" (set in get_models) instead — the imbalance
    # is still handled, just by a different documented mechanism.
    if HAS_SMOTE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        imbalance_strategy = "SMOTE oversampling (train split only)"
        print(f"[SMOTE] Train set balanced: {y_train_res.value_counts().to_dict()}")
    else:
        X_train_res, y_train_res = X_train, y_train
        imbalance_strategy = "class_weight='balanced' (imblearn not installed)"
        print(f"[Imbalance] {imbalance_strategy}")

    models = get_models()
    all_metrics = []
    roc_data = {}
    trained_models = {}

    for name, model in models.items():
        print(f"[Training] {name} ...")
        model.fit(X_train_res, y_train_res)
        metrics, y_proba, report = evaluate_model(name, model, X_test, y_test)
        all_metrics.append(metrics)
        trained_models[name] = model

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr.tolist(), tpr.tolist(), metrics["roc_auc"])
        print(" ", metrics)

    # Combined ROC curve chart for all 7 models
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison — All 7 Models")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "16_roc_comparison.png"), dpi=150)
    plt.close()

    # Metric comparison bar chart
    metrics_df = pd.DataFrame(all_metrics).set_index("model")
    metrics_df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot.bar(figsize=(11, 6))
    plt.title("Model Comparison — All Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "17_model_comparison_bar.png"), dpi=150)
    plt.close()

    best = max(all_metrics, key=lambda m: m["f1_score"])
    best_name = best["model"]
    best_model = trained_models[best_name]
    print(f"[Best Model] {best_name} selected (highest F1 = {best['f1_score']})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_churn_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))

    # Feature importance for the best tree-based model (explainability)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        fi.plot.barh(color="#138808")
        plt.gca().invert_yaxis()
        plt.title(f"Top 15 Feature Importances — {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "18_feature_importance.png"), dpi=150)
        plt.close()

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump({"all_models": all_metrics, "best_model": best_name,
                   "imbalance_strategy": imbalance_strategy,
                   "selection_criterion": "highest F1 on held-out test split"}, f, indent=2)

    print(f"[OK] Churn metrics saved -> {METRICS_OUT}")
    print(f"[OK] Best model saved -> models/best_churn_model.pkl")
    return all_metrics, best_name


if __name__ == "__main__":
    run()
