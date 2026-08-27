"""
explainability_service.py
==========================
SHAP-based explainability for the existing champion churn model
(02_ML_AI/models/best_churn_model.pkl — LightGBM, chosen by highest F1 on a
held-out split, see 04_DATA_TESTING/reports/churn_model_metrics.json).

Nothing here retrains or replaces the model. It loads the SAME model and
feature columns the /prediction route already uses, runs a SHAP
TreeExplainer against it, and turns the per-feature contribution values into
a human-readable, business-facing explanation for a real customer row from
04_DATA_TESTING/datasets/processed/final_customer_data.csv.

CACHING
-------
The trained model + TreeExplainer are expensive to build but constant for
the process lifetime, so they are built once (module-level singletons).
Per-customer explanations are cached in-memory by CustomerID since the
underlying data does not change between pipeline runs.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FINAL_DATA_PATH, CLEANED_MODEL_DATA_PATH, CHURN_MODEL_PATH, FEATURE_COLUMNS_PATH

# Risk band thresholds — same cutoffs already used by the /prediction route
# in app.py, kept in one place here so explanations and the live "what-if"
# predictor stay consistent.
HIGH_RISK_THRESHOLD = 0.66
MEDIUM_RISK_THRESHOLD = 0.33

# Human-readable (increase_phrase, decrease_phrase) per model feature — used
# to turn a signed SHAP contribution into a plain-English factor. Only
# covers columns that are actually in feature_columns.pkl.
FEATURE_LABELS = {
    "Tenure": ("short customer tenure", "long-standing tenure"),
    "CityTier": ("residence in a lower-tier city", "residence in a higher-tier city"),
    "WarehouseToHome": ("longer delivery distance from warehouse", "shorter delivery distance from warehouse"),
    "Gender": ("gender profile", "gender profile"),
    "HourSpendOnApp": ("low time spent on the app", "high time spent on the app"),
    "NumberOfDeviceRegistered": ("fewer registered devices", "more registered devices"),
    "SatisfactionScore": ("low satisfaction score", "high satisfaction score"),
    "NumberOfAddress": ("fewer saved addresses", "more saved addresses"),
    "Complain": ("a recent complaint", "no recent complaints"),
    "OrderAmountHikeFromlastYear": ("lower year-on-year order value growth", "higher year-on-year order value growth"),
    "CouponUsed": ("low coupon usage", "high coupon usage"),
    "OrderCount": ("low order count", "high order count"),
    "DaySinceLastOrder": ("long time since last order", "recent order activity"),
    "CashbackAmount": ("low cashback earned", "high cashback earned"),
    "EstimatedAnnualSpendINR": ("lower estimated annual spend", "higher estimated annual spend"),
    "Recency": ("long time since last purchase", "recent purchase activity"),
    "Frequency": ("low purchase frequency", "high purchase frequency"),
    "Monetary": ("lower monetary value", "higher monetary value"),
    "EngagementScore": ("low engagement score", "high engagement score"),
    "HighComplaintRisk": ("an elevated complaint-risk profile", "a low complaint-risk profile"),
}

_model = None
_feature_columns = None
_explainer = None
_cleaned_model_data = None
_explain_cache = {}


def _get_model():
    global _model, _feature_columns
    if _model is None:
        if not os.path.exists(CHURN_MODEL_PATH):
            raise FileNotFoundError(
                "Churn model not trained yet. Run 02_ML_AI/run_pipeline.py first."
            )
        _model = joblib.load(CHURN_MODEL_PATH)
        _feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return _model, _feature_columns


def _get_explainer():
    global _explainer
    if _explainer is None:
        import shap  # imported lazily so the rest of the app works even if shap fails to import
        model, _ = _get_model()
        _explainer = shap.TreeExplainer(model)
    return _explainer


def is_available():
    """Whether SHAP explanations can be produced at all (model + shap present)."""
    try:
        import shap  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(CHURN_MODEL_PATH) and os.path.exists(FEATURE_COLUMNS_PATH)


def _risk_band(proba):
    if proba >= HIGH_RISK_THRESHOLD:
        return "High"
    if proba >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def _get_cleaned_model_data():
    """The SAME encoded+scaled table (cleaned_data.csv) the model was
    trained on and the SAME one 06_business_recommendations.py used to
    produce the stored ChurnProbability column. StandardScaler's fitted
    mean/std are not persisted anywhere in the pipeline, so reconstructing a
    scaled row from the human-readable final_customer_data.csv would risk a
    silent mismatch (raw, unscaled values fed to a model trained on scaled
    ones). Looking the customer up here by CustomerID guarantees the SHAP
    explanation's probability is byte-for-byte consistent with the
    dashboard's stored churn probability for that same customer."""
    global _cleaned_model_data
    if _cleaned_model_data is None:
        _cleaned_model_data = pd.read_csv(CLEANED_MODEL_DATA_PATH)
    return _cleaned_model_data


def _get_model_ready_row(customer_id, feature_columns):
    cleaned = _get_cleaned_model_data()
    match = cleaned[cleaned["CustomerID"] == customer_id]
    if match.empty:
        return None
    return match.reindex(columns=feature_columns, fill_value=0)


def _display_value(feat, raw_value):
    if feat == "Gender":
        return raw_value
    try:
        return float(raw_value) if raw_value is not None else None
    except (TypeError, ValueError):
        return raw_value


def _extract_shap_row(shap_values, expected_value):
    """Normalise the several shapes shap can return for a binary classifier
    into (contributions_for_positive_class, base_value_for_positive_class)."""
    if isinstance(shap_values, list):
        # Older SHAP API: [class0_array, class1_array]
        contrib = np.asarray(shap_values[1])[0]
        base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            contrib = arr[0, :, 1]
            base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        else:
            contrib = arr[0]
            base = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    return contrib, float(base)


def explain_customer(customer_id, customer_row=None):
    """Return a full SHAP-based explanation dict for one real customer.

    customer_row: optional pre-fetched pandas Series/dict for this customer
    (CustomerID must match) to avoid re-reading the CSV when the caller
    already has it loaded.
    """
    customer_id = int(customer_id)
    if customer_id in _explain_cache:
        return _explain_cache[customer_id]

    if customer_row is None:
        df = pd.read_csv(FINAL_DATA_PATH)
        match = df[df["CustomerID"] == customer_id]
        if match.empty:
            return None
        customer_row = match.iloc[0].to_dict()
    elif hasattr(customer_row, "to_dict"):
        customer_row = customer_row.to_dict()

    model, feature_columns = _get_model()
    explainer = _get_explainer()

    X = _get_model_ready_row(customer_id, feature_columns)
    if X is None:
        return None
    proba = float(model.predict_proba(X)[0][1])

    shap_values = explainer.shap_values(X)
    contrib, base_value = _extract_shap_row(shap_values, explainer.expected_value)

    factors = []
    for feat, val in zip(feature_columns, contrib):
        inc_label, dec_label = FEATURE_LABELS.get(feat, (feat, feat))
        factors.append({
            "feature": feat,
            "value": _display_value(feat, customer_row.get(feat)),
            "shap_value": round(float(val), 5),
            "direction": "increases_risk" if val > 0 else "reduces_risk",
            "label": inc_label if val > 0 else dec_label,
        })

    factors_sorted = sorted(factors, key=lambda f: abs(f["shap_value"]), reverse=True)
    max_abs = max((abs(f["shap_value"]) for f in factors_sorted), default=0) or 1
    for f in factors_sorted:
        f["bar_pct"] = round(abs(f["shap_value"]) / max_abs * 100, 1)

    top_increasing = [f for f in factors_sorted if f["direction"] == "increases_risk"][:5]
    top_reducing = [f for f in factors_sorted if f["direction"] == "reduces_risk"][:5]

    risk_band = _risk_band(proba)
    explanation_text = _build_explanation_text(risk_band, proba, top_increasing, top_reducing)

    result = {
        "customer_id": customer_id,
        "churn_probability": round(proba * 100, 1),
        "risk_band": risk_band,
        "base_rate": round(base_value * 100, 1),
        "top_increasing_factors": top_increasing,
        "top_reducing_factors": top_reducing,
        "all_factors": factors_sorted,
        "explanation_text": explanation_text,
    }
    _explain_cache[customer_id] = result
    return result


def _build_explanation_text(risk_band, proba, top_increasing, top_reducing):
    if risk_band in ("High", "Medium") and top_increasing:
        reasons = [f["label"] for f in top_increasing[:3]]
        reason_str = _join_reasons(reasons)
        return (
            f"This customer has {risk_band.lower()} churn risk ({proba*100:.0f}%) "
            f"mainly because of {reason_str}."
        )
    if top_reducing:
        reasons = [f["label"] for f in top_reducing[:3]]
        reason_str = _join_reasons(reasons)
        return (
            f"This customer has {risk_band.lower()} churn risk ({proba*100:.0f}%), "
            f"supported mainly by {reason_str}."
        )
    return f"This customer has {risk_band.lower()} churn risk ({proba*100:.0f}%)."


def _join_reasons(reasons):
    if len(reasons) == 1:
        return reasons[0]
    if len(reasons) == 2:
        return f"{reasons[0]} and {reasons[1]}"
    return f"{', '.join(reasons[:-1])}, and {reasons[-1]}"
