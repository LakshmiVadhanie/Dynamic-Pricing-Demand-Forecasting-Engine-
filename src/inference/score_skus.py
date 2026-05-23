"""
inference/score_skus.py

Loads the registered XGBoost model from MLflow and scores all SKUs
in batch using PySpark pandas UDF for distributed inference.
Writes scored output to Delta for downstream Power BI consumption.
"""

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from dotenv import load_dotenv

load_dotenv()

FEATURE_PATH = "data/delta/features"
SCORED_OUTPUT_PATH = "data/delta/scored_forecasts"
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "pricing/demand_forecast")
MODEL_NAME = "xgb_demand_model"

LAG_FEATURE_COLS = [
    "units_lag_1w", "units_lag_2w", "units_lag_4w",
    "units_lag_8w", "units_lag_12w", "units_lag_26w", "units_lag_52w",
    "rolling_mean_4w", "rolling_mean_12w", "rolling_mean_26w",
    "rolling_std_4w", "rolling_std_12w",
    "any_promo", "promo_rate_4w",
    "week_of_year", "month", "quarter", "is_holiday_season", "yoy_growth",
    "price_gap_vs_family", "store_id", "item_id",
]


def build_spark():
    return (
        SparkSession.builder
        .appName("score_skus")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def resolve_model_uri() -> str:
    """
    Return the URI of the best (lowest MAPE) XGBoost run in the experiment.
    Falls back to 'runs:/latest' pattern if model registry is not configured.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{MLFLOW_EXPERIMENT}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.ensemble_holdout_mape < 0.20",
        order_by=["metrics.ensemble_holdout_mape ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No qualifying runs found in MLflow.")

    best_run_id = runs[0].info.run_id
    print(f"Loading model from run: {best_run_id}")
    return f"runs:/{best_run_id}/xgb_model"


def encode_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["store_id"] = pd.Categorical(df["store_nbr"]).codes
    df["item_id"] = pd.Categorical(df["item_nbr"]).codes
    return df


def score_batch(df_spark, model_uri: str):
    """
    Use Spark pandas UDF for distributed inference.
    Each partition receives a pandas DataFrame and returns predictions.
    """
    broadcast_uri = df_spark.sparkContext.broadcast(model_uri)
    feature_cols = [c for c in LAG_FEATURE_COLS if c in df_spark.columns]

    @F.pandas_udf(DoubleType())
    def predict_udf(*cols) -> pd.Series:
        model = mlflow.xgboost.load_model(broadcast_uri.value)
        df_part = pd.concat(cols, axis=1)
        df_part.columns = feature_cols
        preds = model.predict(df_part.fillna(0).values)
        return pd.Series(np.clip(preds, 0, None))

    col_exprs = [F.col(c) for c in feature_cols]
    return df_spark.withColumn("xgb_forecast", predict_udf(*col_exprs))


def add_pricing_gap_signal(df_spark):
    """
    Classify each SKU-week into a pricing bucket based on price_gap_vs_family
    and the ratio of forecast to rolling_mean_4w (demand signal).
    """
    return df_spark.withColumn(
        "pricing_recommendation",
        F.when(
            (F.col("price_gap_vs_family") > 0.15) & (F.col("xgb_forecast") < F.col("rolling_mean_4w") * 0.85),
            "OVERPRICED"
        ).when(
            (F.col("price_gap_vs_family") < -0.15) & (F.col("xgb_forecast") > F.col("rolling_mean_4w") * 1.10),
            "UNDERPRICED"
        ).otherwise("MARKET_RATE")
    )


def main():
    spark = build_spark()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Loading feature data for scoring...")
    df = (
        spark.read.format("delta").load(FEATURE_PATH)
        .withColumn("store_id", F.dense_rank().over(
            __import__("pyspark.sql", fromlist=["Window"]).Window.orderBy("store_nbr")
        ) - 1)
        .withColumn("item_id", F.dense_rank().over(
            __import__("pyspark.sql", fromlist=["Window"]).Window.orderBy("item_nbr")
        ) - 1)
    )

    try:
        model_uri = resolve_model_uri()
    except RuntimeError as e:
        print(f"WARNING: {e}")
        print("Scoring skipped. Run demand_forecast.py first.")
        spark.stop()
        return

    print("Running distributed batch scoring...")
    df_scored = score_batch(df, model_uri)
    df_scored = add_pricing_gap_signal(df_scored)

    summary = (
        df_scored
        .groupBy("pricing_recommendation")
        .agg(F.count("*").alias("sku_weeks"), F.avg("xgb_forecast").alias("avg_forecast"))
        .toPandas()
    )
    print("\nPricing Recommendation Summary:")
    print(summary.to_string(index=False))

    print(f"\nWriting scored output to {SCORED_OUTPUT_PATH}...")
    (
        df_scored
        .write
        .format("delta")
        .mode("overwrite")
        .partitionBy("year")
        .save(SCORED_OUTPUT_PATH)
    )
    print("Scoring complete.")
    spark.stop()


if __name__ == "__main__":
    main()
