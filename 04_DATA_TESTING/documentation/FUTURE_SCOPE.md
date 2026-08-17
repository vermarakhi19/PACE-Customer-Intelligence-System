# Future Scope

1. **Real-time scoring** — replace the batch pipeline with a streaming
   scorer (Kafka/Redis + a lightweight inference API) so churn probability
   updates as soon as new order/complaint events arrive.
2. **Database backend** — move from CSV files to Postgres/MySQL with a
   proper schema (customers, orders, segments, predictions tables) so the
   dashboard scales past a few hundred thousand rows.
3. **SHAP explainability** — add SHAP values per customer so the
   Prediction page can show *why* a specific customer was flagged
   high-risk, not just the probability.
4. **Multi-tenant support** — let multiple retail brands (e.g. a D-Mart
   franchise vs a JioMart regional partner) use the same dashboard with
   isolated data.
5. **A/B testing framework** — measure whether the recommended retention
   actions (coupon, call, loyalty tier) actually reduce churn vs a control
   group, closing the loop from prediction to measured business impact.
6. **Regional language support** — Hindi/Marathi/Tamil UI toggle for
   dashboard users across Indian regional offices.
7. **WhatsApp/SMS integration** — trigger the RecommendedAction directly
   as a WhatsApp Business API or SMS campaign for at-risk customers,
   given WhatsApp's dominance as a customer channel in India.
8. **Festival/seasonality modelling** — Indian retail has strong
   Diwali/Holi/Republic Day sales spikes; a time-series layer could
   separate seasonal dips from genuine churn risk.
9. **Cold-start handling** — a rules-based or content-based fallback
   model for brand-new customers with no order history yet.
10. **Model monitoring & retraining pipeline** — automated drift
    detection and scheduled retraining (Airflow/Prefect) as customer
    behaviour evolves.
