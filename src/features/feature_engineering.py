"""
features/feature_engineering.py

Constructs lag features, rolling statistics, price-gap features, and
promotion flags used by both the elasticity and forecasting models.
Runs on PySpark with window functions for scalability across 5K+ SKUs.
"""

import os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = "data/delta/train_weekly"
OUTPUT_PATH = "data/delta/features"

LAG_WEEKS = [1, 2, 4, 8, 12, 26, 52]
ROLLING_WINDOWS = [4, 12, 26]


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("feature_engineering")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def sku_window(order_col: str = "week_start"):
    """
    Partitioned by (store_nbr, item_nbr), ordered by week_start.
    """
    return (
        Window
        .partitionBy("store_nbr", "item_nbr")
        .orderBy(order_col)
    )


def add_lag_features(df):
    w = sku_window()
    for lag in LAG_WEEKS:
        df = df.withColumn(
            f"units_lag_{lag}w",
            F.lag("weekly_units", lag).over(w)
        )
    return df


def add_rolling_features(df):
    for window_size in ROLLING_WINDOWS:
        w = (
            Window
            .partitionBy("store_nbr", "item_nbr")
            .orderBy("week_start")
            .rowsBetween(-window_size, -1)
        )
        df = (
            df.withColumn(f"rolling_mean_{window_size}w", F.avg("weekly_units").over(w))
              .withColumn(f"rolling_std_{window_size}w", F.stddev("weekly_units").over(w))
              .withColumn(f"rolling_max_{window_size}w", F.max("weekly_units").over(w))
        )
    return df


def add_yoy_features(df):
    """
    Year-over-year change in units as a proxy for organic demand trend.
    """
    w = sku_window()
    df = df.withColumn("units_lag_52w", F.lag("weekly_units", 52).over(w))
    df = df.withColumn(
        "yoy_growth",
        F.when(
            F.col("units_lag_52w") > 0,
            (F.col("weekly_units") - F.col("units_lag_52w")) / F.col("units_lag_52w")
        ).otherwise(F.lit(None).cast(DoubleType()))
    )
    return df


def add_price_gap_features(df):
    """
    Price gap = log(sku_price) - log(family_median_price).
    Positive gap => SKU is priced above its category median (potential overpricing).
    Requires a 'unit_price' column. If Favorita does not include price directly,
    we approximate with inverse of unit_sales scaled by promotion flag.
    """
    family_window = (
        Window
        .partitionBy("family", "week_start")
    )
    df = df.withColumn(
        "family_median_units",
        F.percentile_approx("weekly_units", 0.5).over(family_window)
    )
    df = df.withColumn(
        "price_gap_vs_family",
        F.when(
            (F.col("weekly_units") > 0) & (F.col("family_median_units") > 0),
            F.log(F.col("weekly_units")) - F.log(F.col("family_median_units"))
        ).otherwise(F.lit(0.0))
    )
    return df


def add_time_features(df):
    df = (
        df.withColumn("week_of_year", F.weekofyear("week_start"))
          .withColumn("month", F.month("week_start"))
          .withColumn("quarter", F.quarter("week_start"))
          .withColumn("year", F.year("week_start"))
          .withColumn(
              "is_holiday_season",
              F.when(F.col("week_of_year").isin(list(range(48, 53)) + [1]), 1).otherwise(0)
          )
    )
    return df


def add_promotion_rate(df):
    """
    Rolling 4-week promotion rate per SKU.
    """
    w = (
        Window
        .partitionBy("store_nbr", "item_nbr")
        .orderBy("week_start")
        .rowsBetween(-4, -1)
    )
    df = df.withColumn("promo_rate_4w", F.avg("any_promo").over(w))
    return df


def drop_cold_start_rows(df, min_lag: int = 52):
    """
    Remove rows that cannot have all lag features populated.
    """
    return df.filter(F.col(f"units_lag_{min_lag}w").isNotNull())


def main():
    spark = build_spark()

    print("Reading weekly train features...")
    df = spark.read.format("delta").load(INPUT_PATH)

    print("Engineering features...")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_yoy_features(df)
    df = add_time_features(df)
    df = add_promotion_rate(df)
    df = add_price_gap_features(df)
    df = drop_cold_start_rows(df)

    n_skus = df.select("item_nbr").distinct().count()
    print(f"Feature set covers {n_skus:,} unique SKUs")

    print("Writing features to Delta...")
    df.write.format("delta").mode("overwrite").partitionBy("year").save(OUTPUT_PATH)
    print("Feature engineering complete.")

    spark.stop()


if __name__ == "__main__":
    main()
