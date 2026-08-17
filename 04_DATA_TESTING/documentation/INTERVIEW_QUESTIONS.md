# Interview Questions (Job Interview Prep, Based on This Project)

## Machine Learning / Data Science
1. Walk me through your end-to-end ML pipeline for this project.
2. How did you handle missing data, and why median/mode imputation
   specifically over other techniques (KNN imputation, MICE)?
3. Explain the difference between Silhouette Score, Davies-Bouldin, and
   Calinski-Harabasz — when would each disagree on the "best" model?
4. Why did you scale features before clustering but is scaling required
   for tree-based classifiers like Random Forest?
5. What's the difference between K-Means and Hierarchical Clustering
   computationally (time complexity, memory)?
6. How does SMOTE work internally, and what are its limitations
   (e.g. it can create unrealistic synthetic points in high-dimensional
   or categorical-heavy data)?
7. Compare XGBoost, LightGBM, and CatBoost — architectural differences
   (leaf-wise vs level-wise growth, categorical handling).
8. How would you detect if your churn model degrades over time in
   production (concept drift)?
9. If you only had budget to retain 100 customers, how would you use the
   ChurnProbability output to decide who?

## Software Engineering / Full Stack
10. Why did you separate the ML pipeline (src/) from the web app (app/)?
11. How is the Flask app structured to avoid re-training the model on
    every page load?
12. How would you scale this dashboard to 100,000+ customers — what
    changes to the DataTables/customer-list page would you make?
13. Walk me through your authentication implementation and its
    production gaps.
14. How would you convert the CSV-based data layer into a proper
    relational schema (draw the tables)?

## Business / Product
15. How would you quantify the ROI of this retention system for a
    business stakeholder?
16. How do you decide the discount size in a win-back offer without
    eroding margin?
17. What Indian-market-specific factors (COD prevalence, festival
    seasonality, Tier-2/3 logistics) would you add to this model?
18. If Marketing and Data Science disagree on which segment to target
    first, how would you resolve it?

## System Design
19. Design this as a real-time system: a customer places an order and
    their churn score should update within minutes — what changes?
20. How would you A/B test whether the recommended retention actions
    actually reduce churn?
