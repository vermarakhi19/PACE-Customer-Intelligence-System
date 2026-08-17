# Viva Questions & Answers

### Q1. Why did you choose K-Means over DBSCAN for the final segmentation model?
K-Means gave the highest Silhouette Score and produces roughly balanced
cluster sizes, which matters commercially — a marketing team needs every
customer assigned to an actionable group. DBSCAN is density-based and
tends to label sparse/borderline customers as "noise" (-1), which can't be
targeted with a campaign. DBSCAN is still valuable as a comparison because
it independently confirms (or challenges) how well-separated the segments
really are.

### Q2. Why is F1 Score used to pick the best churn model instead of Accuracy?
The dataset is imbalanced (~83% retained, ~17% churned). A model that
predicts "retained" for everyone would score ~83% accuracy while being
useless. F1 balances Precision (don't waste retention budget on customers
who wouldn't have churned) and Recall (don't miss customers who will
actually churn), which is what the business actually cares about.

### Q3. Why use SMOTE, and why only on the training set?
SMOTE synthetically oversamples the minority (churned) class so the model
doesn't just learn to always predict the majority class. Applying it only
to the training split (after the train/test split) prevents synthetic rows
from leaking into the test set, which would make the reported accuracy
artificially optimistic.

### Q4. How did you validate the clustering results without ground-truth labels?
Clustering is unsupervised, so there's no "correct answer" to check
against. We use three internal validation metrics that measure cluster
compactness/separation directly from the data itself: Silhouette Score,
Davies-Bouldin Index, and Calinski-Harabasz Index — then sanity-check
the resulting segments against known RFM business logic (e.g. the "Loyal
High-Value" cluster should have low Recency, high Frequency, high
Monetary).

### Q5. Why cap outliers instead of removing them?
Removing rows with unusual (but real) purchase behaviour would delete
exactly the big-spender/highly-engaged customers a retail business most
wants to understand — that's not noise, that's signal. Capping (Winsorising)
at 1.5×IQR limits the outlier's distorting effect on scaling/model training
while keeping the customer in the analysis.

### Q6. What is the business meaning of "Recency", "Frequency", "Monetary"?
RFM is a classic segmentation framework: Recency = how recently a customer
last purchased, Frequency = how often they purchase, Monetary = how much
they spend. Customers who are recent, frequent, and high-spend are the
most valuable and lowest churn-risk.

### Q7. How would this system handle a new customer with no order history?
The current model needs the behavioural columns (Tenure, OrderCount, etc.)
to score a customer, so a brand-new customer with zero history would be
placed in the "New / Recently Acquired" segment by rule (see
`04_segmentation.py`'s cluster-naming logic) rather than by the trained
model, until enough behavioural data accumulates — a cold-start limitation
documented in Future Scope.

### Q8. Why Flask instead of Django for the dashboard?
Flask is a microframework — lighter weight and faster to stand up for a
focused, single-purpose analytics dashboard with ~9 pages and no complex
ORM/admin requirements. Django's batteries-included ORM, migrations, and
admin panel add value for larger multi-app systems, which this project
doesn't need.

### Q9. How do you prevent data leakage between segmentation and churn prediction?
Segmentation features (Recency/Frequency/Monetary/EngagementScore) are
derived from the same underlying raw columns as some churn model features,
but the SegmentName label itself is never fed as an input to the churn
model — it's a downstream *output*, combined with the churn prediction only
at the business-recommendation stage (Segment × Risk), not before.

### Q10. What does CityTier represent and why does it matter for Indian retail?
CityTier is India's real classification of cities by population/economic
activity (Tier-1: Mumbai, Delhi, Bangalore, etc.; Tier-2: Pune, Nagpur,
etc.; Tier-3: smaller cities). It matters because logistics costs, income
levels, and category preferences vary substantially by tier — a Tier-1
customer's churn drivers and the right retention offer often differ from a
Tier-3 customer's.

### Q11. How is model explainability handled?
Tree-based models (Random Forest / XGBoost / LightGBM / CatBoost) expose
`feature_importances_`, plotted in `05_churn_prediction.py` (top 15
features). This shows which behavioural signals — e.g. SatisfactionScore,
Complain, DaySinceLastOrder — drive the churn prediction, which is what
gets translated into the plain-English business recommendations.

### Q12. What would you change for a real production deployment?
Move from CSV-based pipeline outputs to a proper database (Postgres),
add authentication with hashed passwords in a user table, schedule the
pipeline to re-run periodically, add model monitoring for drift, and put
the Flask app behind Gunicorn + Nginx with HTTPS (see DEPLOYMENT.md).
