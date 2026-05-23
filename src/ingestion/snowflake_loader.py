"""
ingestion/snowflake_loader.py

Reads Favorita tables from Snowflake into PySpark DataFrames and
writes them as Delta tables to the Databricks lakehouse (or local
parquet in local mode).
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from dotenv import load_dotenv

load_dotenv()


SNOWFLAKE_OPTIONS = {
    "sfURL": os.environ["SNOWFLAKE_ACCOUNT"],
    "sfUser": os.environ["SNOWFLAKE_USER"],
    "sfPassword": os.environ["SNOWFLAKE_PASSWORD"],
    "sfDatabase": os.environ["SNOWFLAKE_DATABASE"],
    "sfSchema": os.environ["SNOWFLAKE_SCHEMA"],
    "sfWarehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    "sfRole": os.environ["SNOWFLAKE_ROLE"],
}

OUTPUT_BASE = "data/delta"


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("favorita_ingestion")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def read_snowflake_table(spark: SparkSession, table: str):
    return (
        spark.read
        .format("snowflake")
        .options(**SNOWFLAKE_OPTIONS)
        .option("dbtable", table)
        .load()
    )


def load_train(spark: SparkSession):
    """
    Favorita train table: date, store_nbr, item_nbr, unit_sales, onpromotion
    """
    df = read_snowflake_table(spark, "TRAIN")
    df = (
        df.withColumn("date", F.to_date("date"))
          .withColumn("unit_sales", F.greatest(F.col("unit_sales"), F.lit(0)))
          .withColumn("week_start", F.date_trunc("week", F.col("date")))
    )
    return df


def load_items(spark: SparkSession):
    """
    Item master: item_nbr, family, class, perishable
    """
    return read_snowflake_table(spark, "ITEMS")


def load_stores(spark: SparkSession):
    """
    Store master: store_nbr, city, state, type, cluster
    """
    return read_snowflake_table(spark, "STORES")


def load_oil(spark: SparkSession):
    """
    Daily oil price (Ecuador macro variable): date, dcoilwtico
    """
    df = read_snowflake_table(spark, "OIL")
    df = (
        df.withColumn("date", F.to_date("date"))
          .na.fill(method="forward", subset=["dcoilwtico"])
    )
    return df


def aggregate_weekly(df_train):
    """
    Collapse daily sales to weekly grain for forecasting.
    """
    return (
        df_train
        .groupBy("week_start", "store_nbr", "item_nbr")
        .agg(
            F.sum("unit_sales").alias("weekly_units"),
            F.avg("unit_sales").alias("avg_daily_units"),
            F.max("onpromotion").cast("int").alias("any_promo"),
            F.countDistinct("date").alias("days_in_week"),
        )
    )


def write_delta(df, path: str, mode: str = "overwrite"):
    df.write.format("delta").mode(mode).save(path)
    print(f"Written {df.count()} rows to {path}")


def main():
    spark = build_spark()

    print("Loading Snowflake tables...")
    train_raw = load_train(spark)
    items = load_items(spark)
    stores = load_stores(spark)
    oil = load_oil(spark)

    print("Aggregating to weekly grain...")
    train_weekly = aggregate_weekly(train_raw)

    print("Joining dimension tables...")
    train_enriched = (
        train_weekly
        .join(items, on="item_nbr", how="left")
        .join(stores, on="store_nbr", how="left")
    )

    write_delta(train_enriched, f"{OUTPUT_BASE}/train_weekly")
    write_delta(items, f"{OUTPUT_BASE}/items")
    write_delta(stores, f"{OUTPUT_BASE}/stores")
    write_delta(oil, f"{OUTPUT_BASE}/oil")

    print("Ingestion complete.")
    spark.stop()


if __name__ == "__main__":
    main()
