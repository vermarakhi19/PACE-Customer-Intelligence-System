# PPT Content (Slide by Slide)

**Slide 1 — Title**
AI-Powered Customer Segmentation and Customer Retention Prediction System
for Indian Retail Businesses | Your Name, College, Guide Name, Date

**Slide 2 — Problem Statement**
Indian retail businesses lose customers silently — no early warning before
churn. Need: (1) understand customer groups, (2) predict who's leaving,
(3) act with specific, costed retention offers.

**Slide 3 — Objectives**
- Segment customers into actionable business groups
- Predict churn probability per customer
- Convert both into Indian-context business recommendations
- Deliver via an interactive dashboard

**Slide 4 — Dataset**
Kaggle: "Ecommerce Customer Churn Analysis and Prediction" — 5,630
customers, 20 columns, real Indian e-commerce company data. (Link + data
dictionary snapshot.)

**Slide 5 — System Architecture**
Diagram: Raw CSV → Indian Enrichment → Preprocessing → EDA → Segmentation
→ Churn Prediction → Business Recommendations → Dashboard.

**Slide 6 — Data Preprocessing**
Duplicate removal, missing-value imputation, IQR outlier capping,
encoding, scaling, feature engineering (RFM, EngagementScore).

**Slide 7 — EDA Highlights**
2-3 key charts: churn pie chart, satisfaction-vs-churn line chart,
correlation heatmap — each with a one-line business insight.

**Slide 8 — Segmentation Approach**
K-Means vs DBSCAN vs Hierarchical, compared on Silhouette / Davies-Bouldin
/ Calinski-Harabasz. Winner + why.

**Slide 9 — Segments Identified**
5 segments with short profiles: Loyal High-Value, At-Risk/Dormant, Big
Spenders, New, Occasional Shoppers.

**Slide 10 — Churn Prediction Approach**
7 classifiers compared, SMOTE for imbalance, metrics table (Accuracy,
Precision, Recall, F1, ROC-AUC).

**Slide 11 — Best Model & Why**
Winning model + confusion matrix + ROC curve + feature importance chart.

**Slide 12 — Business Recommendations**
Segment × Risk → Action matrix, plus 2-3 example state/category insights
("Customers from Maharashtra buying Electronics show high retention...").

**Slide 13 — Dashboard Walkthrough**
Screenshots: Dashboard KPIs, Customer List, Prediction form, Model
Comparison page.

**Slide 14 — Tech Stack**
Python, scikit-learn, XGBoost/LightGBM/CatBoost, Flask, Bootstrap 5,
Chart.js, DataTables.

**Slide 15 — Limitations & Future Scope**
Cold-start customers, real-time scoring, SHAP explainability, WhatsApp
integration, multi-tenant support.

**Slide 16 — Conclusion**
Recap: real data → compared models → explainable, Indian-context,
actionable retention system.

**Slide 17 — Q&A / Thank You**
