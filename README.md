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

## Quickstart

### Prerequisites

- Python 3.9+
- Java 11 (for PySpark local mode)
- Databricks CLI
- Snowflake account with Favorita data loaded

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure credentials

```bash
cp .env.example .env
# Fill in Snowflake and Databricks credentials
```

### Run locally (PySpark local mode)

```bash
python src/ingestion/snowflake_loader.py
python src/features/feature_engineering.py
python src/modeling/elasticity_model.py
python src/modeling/demand_forecast.py
python src/inference/score_skus.py
```

### Deploy to Databricks

```bash
databricks bundle deploy --target dev
databricks bundle run pricing_pipeline --target dev
```

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

## License

MIT
