# Dynamic Pricing and Demand Forecasting Engine

End-to-end ML pipeline for price elasticity modeling and weekly demand forecasting across 5,000+ SKUs using the Kaggle Favorita dataset. Combines hierarchical regression, DiD causal inference, XGBoost, and ARIMA ensembles with a full MLOps stack on Databricks and Snowflake.

## Architecture

```
Snowflake (raw) -> PySpark (feature engineering) -> MLflow (experiment tracking)
    -> XGBoost + ARIMA ensemble (forecasting, 12% MAPE)
    -> Hierarchical regression + DiD (price elasticity)
    -> Power BI (pricing gap dashboards)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Data warehouse | Snowflake |
| Distributed compute | PySpark on Databricks |
| Causal inference | Statsmodels (DiD, OLS hierarchical) |
| Demand forecasting | XGBoost, Statsmodels ARIMA |
| Experiment tracking | MLflow |
| Visualization | Power BI (DAX measures in `/docs`) |
| Orchestration | Databricks Workflows |


## Project Structure

```
dynamic-pricing-engine/
    src/
        ingestion/          Snowflake connector and PySpark ingestion
        features/           Feature engineering and lag construction
        modeling/           Elasticity (DiD + hierarchical) and forecasting (XGBoost + ARIMA)
        inference/          Batch scoring across all SKUs
        viz/                Power BI data prep exports
    notebooks/              Exploratory analysis and prototyping
    mlflow_config/          MLflow tracking server config
    databricks/             Databricks Asset Bundles (DAB) config
    tests/                  Unit and integration tests
    docs/                   Power BI DAX measures and data dictionary
```

## Results

- Price elasticity modeled across 5,000+ SKUs with store-level hierarchical random effects
- 12% MAPE on weekly demand forecasts via XGBoost and ARIMA ensemble
- Pricing gap visualization surfacing underpriced and overpriced SKU clusters in Power BI

## Data

Kaggle Favorita Grocery Sales Forecasting dataset. Load into Snowflake using the schema in `docs/snowflake_schema.sql`.

