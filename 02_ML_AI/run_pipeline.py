"""
run_pipeline.py
================
Runs all 9 modules in order. Use this once your real CSV is in
04_DATA_TESTING/datasets/raw/ecommerce_churn.csv.

    cd PACE-Customer-Intelligence-System
    python 02_ML_AI/run_pipeline.py

RESTRUCTURE NOTE (team-folder reorg)
-------------------------------------
Each pipeline stage now lives in its own subfolder under 02_ML_AI/
(data_loading/, preprocessing/, eda/, segmentation/, churn_prediction/,
forecasting/, recommendations/) instead of one flat src/ folder. Every
stage's own filename is unchanged (00_load_transactions.py ...
08_recommendations.py), so this just adds each stage folder — plus this
folder itself, for config.py — to sys.path before importing them by name.
"""
import importlib
import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("data_loading", "00_load_transactions", "Transactional Layer (Online Retail II)"),
    ("data_loading", "01_data_loading", "Data Loading & Enrichment"),
    ("preprocessing", "02_preprocessing", "Preprocessing"),
    ("eda", "03_eda", "Exploratory Data Analysis"),
    ("segmentation", "04_segmentation", "Customer Segmentation"),
    ("churn_prediction", "05_churn_prediction", "Retention / Churn Prediction"),
    ("recommendations", "06_business_recommendations", "Business Recommendations"),
    ("forecasting", "07_sales_forecasting", "Customer Lifecycle Revenue Forecasting"),
    ("recommendations", "08_recommendations", "Product Category Recommendations"),
]

# config.py lives directly in this folder; each stage subfolder needs to be
# importable too since importlib looks up modules by bare filename.
sys.path.insert(0, THIS_DIR)
for stage_folder, _, _ in STEPS:
    stage_path = os.path.join(THIS_DIR, stage_folder)
    if stage_path not in sys.path:
        sys.path.insert(0, stage_path)

if __name__ == "__main__":
    for stage_folder, module_name, label in STEPS:
        print("\n" + "=" * 70)
        print(f"MODULE: {label}  ({stage_folder}/{module_name}.py)")
        print("=" * 70)
        mod = importlib.import_module(module_name)
        mod.run()
    print("\n[PIPELINE COMPLETE] All outputs are in "
          "04_DATA_TESTING/datasets/processed/, 04_DATA_TESTING/reports/, 02_ML_AI/models/")
