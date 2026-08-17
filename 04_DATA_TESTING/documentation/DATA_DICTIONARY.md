# Data Dictionary

## A. Raw Kaggle Columns (20)

| Column | Type | Description |
|---|---|---|
| CustomerID | int | Unique customer identifier |
| Churn | int (0/1) | Target — 1 = customer churned, 0 = retained |
| Tenure | float | Months since the customer joined |
| PreferredLoginDevice | categorical | Mobile Phone / Computer / Phone |
| CityTier | int (1/2/3) | India's real official city-tier classification |
| WarehouseToHome | float | Distance (km) from warehouse to customer's home |
| PreferredPaymentMode | categorical | Debit Card / Credit Card / UPI / COD / E-wallet |
| Gender | categorical | Male / Female |
| HourSpendOnApp | float | Avg. hours/day spent on the app or website |
| NumberOfDeviceRegistered | int | Devices registered to the account |
| PreferedOrderCat | categorical | Most-ordered category last month (Mobile Phone, Fashion, Electronics, Grocery, Laptop & Accessory, Others) |
| SatisfactionScore | int (1-5) | Customer satisfaction rating |
| MaritalStatus | categorical | Single / Married / Divorced |
| NumberOfAddress | int | Number of saved addresses |
| Complain | int (0/1) | Whether a complaint was raised in the last month |
| OrderAmountHikeFromlastYear | float | % increase in order value vs last year |
| CouponUsed | int | Coupons used in the last month |
| OrderCount | int | Orders placed in the last month |
| DaySinceLastOrder | int | Days since the customer's last order |
| CashbackAmount | float | Average cashback received |

**Missing values (raw):** Tenure, WarehouseToHome, HourSpendOnApp,
OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder each
carry missing entries (roughly 5-7% of rows) — imputed with the column
median in `02_preprocessing.py`.

## B. Indian-Context Enrichment (added in `01_data_loading.py`)

| Column | Description |
|---|---|
| CustomerName | Indian first + last name (e.g. "Rahul Sharma") |
| City | Real Indian city, derived deterministically from CityTier |
| State | Indian state mapped from City |
| PhoneNumber | Indian mobile format `+91-XXXXXXXXXX` |
| SignupDate | `DD/MM/YYYY` format |

## C. Engineered Features (added in `02_preprocessing.py`)

| Column | Description |
|---|---|
| EstimatedAnnualSpendINR | ₹ monetary proxy derived from OrderCount × CashbackAmount |
| Recency | = DaySinceLastOrder (renamed for RFM framing) |
| Frequency | = OrderCount (renamed for RFM framing) |
| Monetary | = EstimatedAnnualSpendINR (renamed for RFM framing) |
| EngagementScore | 0-100 blended score: app usage (40%) + devices (20%) + satisfaction (40%) |
| TenureBucket | New (0-3mo) / Growing (4-12mo) / Established (1-2yr) / Loyal (2yr+) |
| HighComplaintRisk | 1 if Complain=1 AND SatisfactionScore<=2, else 0 |

## D. Segmentation Output (added in `04_segmentation.py`)

| Column | Description |
|---|---|
| Segment_KMeans / Segment_DBSCAN / Segment_Hierarchical | Numeric cluster ID per algorithm |
| SegmentName | Plain-English label from the production model (K-Means): Loyal High-Value, At-Risk / Dormant, Big Spenders, New / Recently Acquired, Occasional Shoppers |

## E. Prediction Output (added in `06_business_recommendations.py`)

| Column | Description |
|---|---|
| ChurnProbability | Model's predicted probability (0-1) that the customer churns |
| ChurnRiskBand | High (>=0.66) / Medium (0.33-0.66) / Low (<0.33) |
| RecommendedAction | Rule-engine output combining Segment × Risk Band |

## F. Forecasting Output (`04_DATA_TESTING/reports/lifecycle_forecast.json`, from `02_ML_AI/forecasting/07_sales_forecasting.py`)

| Field | Description |
|---|---|
| historical | Real Tenure-month → avg. customer Monetary curve (no synthetic data) |
| backtest_comparison | Naive baseline vs Holt linear trend (vs statsmodels if installed), MAPE/RMSE |
| forecast | Tenure months beyond the last observed one, projected value + 95% CI |
| segment_lifecycle_summary | Per-segment current value vs next-month projection + trend direction |

## G. Recommendations Output (`04_DATA_TESTING/datasets/processed/customer_recommendations.csv`, from `02_ML_AI/recommendations/08_recommendations.py`)

| Column | Description |
|---|---|
| CurrentCategory | Customer's real `PreferedOrderCat` |
| RecommendedCategory1 / 2 | Top categories favoured by the customer's 10 nearest behavioural neighbours (own category excluded) |
| Confidence1 / 2 | % of those neighbours favouring that category |
| Reason | Plain-English explanation combining kNN result + Segment × Category affinity |
