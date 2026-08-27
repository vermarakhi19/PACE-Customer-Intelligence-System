"""
simulator_service.py
=====================
Retention Action Simulator.

IMPORTANT — READ BEFORE TRUSTING A NUMBER FROM THIS MODULE
------------------------------------------------------------
This dataset has no A/B test or experiment log of retention offers actually
being sent to customers, so there is no causal/uplift model anywhere in this
project that measures what an action really does to churn probability.
ACTIONS below is therefore a documented table of CONFIGURABLE BUSINESS
ASSUMPTIONS (a cost and an assumed churn-probability-point reduction per
action), not a learned effect. Every result this module returns must be
presented as a "scenario estimate based on configurable assumptions" — never
as a guarantee that an action will reduce churn.

The churn probability and customer value it starts from ARE real (the
existing churn model's ChurnProbability and EstimatedAnnualSpendINR).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FINAL_DATA_PATH

ASSUMPTIONS_NOTE = (
    "Cost and retention-improvement figures below are configurable business "
    "assumptions, not outputs of a causal/uplift model — this dataset has no "
    "experiment log of past retention offers to learn a real effect from. "
    "Treat every result as a scenario estimate, not a guaranteed outcome."
)

# cost_type: "flat" (fixed INR per customer) or "pct_of_spend" (fraction of
# EstimatedAnnualSpendINR). churn_reduction_pp: assumed ABSOLUTE reduction in
# churn probability (probability points, 0-1 scale) if the action is taken.
ACTIONS = {
    "no_action": {
        "label": "No Action",
        "description": "Baseline — no retention intervention offered.",
        "cost_type": "flat", "cost_value": 0.0,
        "churn_reduction_pp": 0.0,
    },
    "discount": {
        "label": "Discount",
        "description": "One-time percentage discount on the customer's next purchase.",
        "cost_type": "pct_of_spend", "cost_value": 0.08,
        "churn_reduction_pp": 0.12,
    },
    "coupon": {
        "label": "Coupon",
        "description": "Fixed-value coupon code sent to the customer.",
        "cost_type": "flat", "cost_value": 300.0,
        "churn_reduction_pp": 0.07,
    },
    "loyalty_points": {
        "label": "Loyalty Points",
        "description": "Bonus loyalty points credited to the customer's account.",
        "cost_type": "flat", "cost_value": 150.0,
        "churn_reduction_pp": 0.05,
    },
    "free_delivery": {
        "label": "Free Delivery",
        "description": "Free delivery offered on the customer's next few orders.",
        "cost_type": "flat", "cost_value": 120.0,
        "churn_reduction_pp": 0.04,
    },
    "personalized_offer": {
        "label": "Personalized Offer",
        "description": "Tailored bundle/offer based on the customer's preferred category and segment.",
        "cost_type": "pct_of_spend", "cost_value": 0.05,
        "churn_reduction_pp": 0.15,
    },
}


def _load():
    if not os.path.exists(FINAL_DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(FINAL_DATA_PATH)


def _action_cost(action_key, estimated_annual_spend):
    action = ACTIONS[action_key]
    if action["cost_type"] == "pct_of_spend":
        return round(action["cost_value"] * float(estimated_annual_spend or 0), 0)
    return round(action["cost_value"], 0)


def _simulate_row(action_key, churn_probability, estimated_annual_spend):
    action = ACTIONS[action_key]
    cost = _action_cost(action_key, estimated_annual_spend)
    current_p = float(churn_probability or 0)
    reduction = min(action["churn_reduction_pp"], current_p)
    new_p = round(current_p - reduction, 4)

    retained_value = round(estimated_annual_spend * reduction, 0)
    net_benefit = round(retained_value - cost, 0)
    roi = round(net_benefit / cost, 2) if cost > 0 else None

    return {
        "action_key": action_key,
        "label": action["label"],
        "description": action["description"],
        "current_churn_probability": round(current_p * 100, 1),
        "projected_churn_probability": round(new_p * 100, 1),
        "retention_improvement_pp": round(reduction * 100, 1),
        "customer_value": round(float(estimated_annual_spend or 0), 0),
        "action_cost": cost,
        "estimated_retained_value": retained_value,
        "net_benefit": net_benefit,
        "roi": roi,
    }


def estimate_retention_opportunity(risk_bands=("Medium", "High")):
    """Dashboard-level summary number: applying the single best-ROI action
    uniformly across every Medium/High-risk customer, what's the total
    estimated retained value? Vectorised over pandas (no per-row SHAP calls)
    so it stays cheap enough to run on every dashboard load.

    This intentionally picks ONE action for the whole at-risk base — a
    simplifying assumption for a single dashboard headline number. The
    Retention Simulator page lets a user compare actions per customer/group
    in full detail.
    """
    df = _load()
    if df.empty or "ChurnRiskBand" not in df.columns:
        return {"opportunity_value": 0, "net_benefit": 0, "customers": 0, "best_action_label": None}

    subset = df[df["ChurnRiskBand"].isin(risk_bands)]
    if subset.empty:
        return {"opportunity_value": 0, "net_benefit": 0, "customers": 0, "best_action_label": None}

    best = None
    for key, action in ACTIONS.items():
        if key == "no_action":
            continue
        if action["cost_type"] == "pct_of_spend":
            cost = action["cost_value"] * subset["EstimatedAnnualSpendINR"]
        else:
            cost = pd.Series(action["cost_value"], index=subset.index)

        reduction = subset["ChurnProbability"].clip(upper=action["churn_reduction_pp"])
        retained_value = subset["EstimatedAnnualSpendINR"] * reduction
        net_benefit = retained_value - cost
        total_net = float(net_benefit.sum())

        if best is None or total_net > best["net_benefit"]:
            best = {
                "action_key": key,
                "best_action_label": action["label"],
                "opportunity_value": round(float(retained_value.sum()), 0),
                "net_benefit": round(total_net, 0),
            }

    best["customers"] = int(len(subset))
    return best


def available_actions():
    return [{"key": k, "label": v["label"], "description": v["description"]} for k, v in ACTIONS.items()]


def simulate_customer(customer_id, action_keys=None, customer_row=None):
    """Compare one or more actions for a single real customer."""
    if customer_row is None:
        df = _load()
        match = df[df["CustomerID"] == int(customer_id)]
        if match.empty:
            return None
        customer_row = match.iloc[0]

    action_keys = action_keys or list(ACTIONS.keys())
    churn_probability = customer_row.get("ChurnProbability")
    spend = customer_row.get("EstimatedAnnualSpendINR")

    results = [_simulate_row(k, churn_probability, spend) for k in action_keys if k in ACTIONS]
    results.sort(key=lambda r: (r["net_benefit"] is None, -(r["net_benefit"] or -1e18)))
    for i, r in enumerate(results):
        r["is_best"] = (i == 0)

    return {
        "customer_id": int(customer_id) if customer_id is not None else None,
        "assumptions_note": ASSUMPTIONS_NOTE,
        "results": results,
    }


def simulate_group(filters=None, action_keys=None):
    """Compare one or more actions across a filtered customer group.

    filters: dict with optional keys "segment" (SegmentName), "risk_band"
    (ChurnRiskBand), "state" (State). Missing/None values are not filtered.
    """
    df = _load()
    if df.empty:
        return None

    filters = filters or {}
    if filters.get("segment"):
        df = df[df["SegmentName"] == filters["segment"]]
    if filters.get("risk_band"):
        df = df[df["ChurnRiskBand"] == filters["risk_band"]]
    if filters.get("state"):
        df = df[df["State"] == filters["state"]]

    if df.empty:
        return {"customer_count": 0, "assumptions_note": ASSUMPTIONS_NOTE, "results": []}

    action_keys = action_keys or list(ACTIONS.keys())
    results = []
    for key in action_keys:
        if key not in ACTIONS:
            continue
        per_customer = [
            _simulate_row(key, row.ChurnProbability, row.EstimatedAnnualSpendINR)
            for row in df.itertuples()
        ]
        total_cost = sum(r["action_cost"] for r in per_customer)
        total_retained = sum(r["estimated_retained_value"] for r in per_customer)
        total_net = round(total_retained - total_cost, 0)
        avg_improvement = round(sum(r["retention_improvement_pp"] for r in per_customer) / len(per_customer), 1)

        results.append({
            "action_key": key,
            "label": ACTIONS[key]["label"],
            "description": ACTIONS[key]["description"],
            "customers_targeted": len(per_customer),
            "avg_retention_improvement_pp": avg_improvement,
            "total_action_cost": round(total_cost, 0),
            "total_estimated_retained_value": round(total_retained, 0),
            "total_net_benefit": total_net,
            "roi": round(total_net / total_cost, 2) if total_cost > 0 else None,
        })

    results.sort(key=lambda r: (r["total_net_benefit"] is None, -(r["total_net_benefit"] or -1e18)))
    for i, r in enumerate(results):
        r["is_best"] = (i == 0)

    return {
        "customer_count": int(len(df)),
        "assumptions_note": ASSUMPTIONS_NOTE,
        "results": results,
    }
