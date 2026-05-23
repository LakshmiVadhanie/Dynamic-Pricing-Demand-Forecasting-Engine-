"""
viz/powerbi_export.py

Prepares flattened, Power BI-compatible parquet exports from the
scored Delta tables. Also generates a local matplotlib preview of
key pricing gap distributions for validation before publishing.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from dotenv import load_dotenv

load_dotenv()

SCORED_PATH = "data/delta/scored_forecasts"
ELASTICITY_PATH = "data/elasticity_results.parquet"
EXPORT_DIR = "data/powerbi_exports"

os.makedirs(EXPORT_DIR, exist_ok=True)


def build_spark():
    return (
        SparkSession.builder
        .appName("powerbi_export")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def export_pricing_gaps(spark) -> pd.DataFrame:
    """
    Aggregate the latest week of scored data per SKU for the
    pricing gap scatter chart (x = price_gap, y = forecast vs baseline ratio).
    """
    df = spark.read.format("delta").load(SCORED_PATH)
    latest_week = df.agg(F.max("week_start")).collect()[0][0]

    df_latest = (
        df.filter(F.col("week_start") == latest_week)
          .select(
              "store_nbr", "item_nbr", "family",
              "price_gap_vs_family", "xgb_forecast",
              "rolling_mean_4w", "pricing_recommendation",
              "week_start",
          )
          .withColumn(
              "forecast_vs_baseline_ratio",
              F.when(F.col("rolling_mean_4w") > 0,
                     F.col("xgb_forecast") / F.col("rolling_mean_4w"))
               .otherwise(F.lit(1.0))
          )
    )
    return df_latest.toPandas()


def export_weekly_forecast_trend(spark) -> pd.DataFrame:
    """
    Weekly aggregate forecast and actuals for trend line charts in Power BI.
    """
    df = spark.read.format("delta").load(SCORED_PATH)
    weekly = (
        df.groupBy("week_start", "family")
          .agg(
              F.sum("xgb_forecast").alias("total_forecast_units"),
              F.sum("weekly_units").alias("total_actual_units"),
              F.countDistinct("item_nbr").alias("n_skus"),
          )
          .orderBy("week_start", "family")
    )
    return weekly.toPandas()


def export_elasticity_summary() -> pd.DataFrame:
    """
    Merge elasticity results with pricing recommendations for Power BI.
    """
    if not os.path.exists(ELASTICITY_PATH):
        print("Elasticity results not found. Run elasticity_model.py first.")
        return pd.DataFrame()
    df = pd.read_parquet(ELASTICITY_PATH)
    df["elasticity_bucket"] = pd.cut(
        df["elasticity_promo"],
        bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
        labels=["highly_elastic", "elastic", "inelastic", "slightly_inelastic", "highly_inelastic"]
    )
    return df


def plot_pricing_gap_scatter(df: pd.DataFrame, save_path: str):
    palette = {
        "OVERPRICED": "#E63946",
        "UNDERPRICED": "#2A9D8F",
        "MARKET_RATE": "#457B9D",
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    for rec, grp in df.groupby("pricing_recommendation"):
        ax.scatter(
            grp["price_gap_vs_family"],
            grp["forecast_vs_baseline_ratio"],
            c=palette.get(rec, "#888"),
            alpha=0.55,
            s=18,
            label=rec,
            edgecolors="none",
        )

    ax.axhline(1.0, color="#FFFFFF30", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="#FFFFFF30", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Price Gap vs Family Median (log ratio)", color="#CCCCCC", fontsize=11)
    ax.set_ylabel("Forecast / Baseline Demand Ratio", color="#CCCCCC", fontsize=11)
    ax.set_title("Pricing Gap vs Demand Signal by SKU", color="#FFFFFF", fontsize=14, pad=14)
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    legend = ax.legend(frameon=True, facecolor="#1A1D26", edgecolor="#333333")
    for text in legend.get_texts():
        text.set_color("#DDDDDD")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved pricing gap scatter to {save_path}")


def plot_forecast_vs_actual(df_weekly: pd.DataFrame, save_path: str):
    top_families = (
        df_weekly.groupby("family")["total_actual_units"].sum()
        .nlargest(5).index.tolist()
    )
    df_plot = df_weekly[df_weekly["family"].isin(top_families)]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    colors = ["#E63946", "#2A9D8F", "#F4A261", "#A8DADC", "#457B9D"]
    for i, fam in enumerate(top_families):
        grp = df_plot[df_plot["family"] == fam].sort_values("week_start")
        ax.plot(grp["week_start"], grp["total_actual_units"],
                color=colors[i], linewidth=1.5, label=f"{fam} actual", alpha=0.9)
        ax.plot(grp["week_start"], grp["total_forecast_units"],
                color=colors[i], linewidth=1.2, linestyle="--", alpha=0.55)

    ax.set_title("Weekly Demand: Actual vs Forecast (Top 5 Families)", color="#FFFFFF", fontsize=13)
    ax.set_xlabel("Week", color="#CCCCCC")
    ax.set_ylabel("Units Sold", color="#CCCCCC")
    ax.tick_params(colors="#AAAAAA")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    legend = ax.legend(frameon=True, facecolor="#1A1D26", edgecolor="#333333", ncol=2)
    for text in legend.get_texts():
        text.set_color("#DDDDDD")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved forecast vs actual chart to {save_path}")


def main():
    spark = build_spark()

    print("Preparing Power BI exports...")

    try:
        df_gap = export_pricing_gaps(spark)
        df_gap.to_parquet(f"{EXPORT_DIR}/pricing_gaps_latest.parquet", index=False)
        plot_pricing_gap_scatter(df_gap, f"{EXPORT_DIR}/pricing_gap_scatter.png")
    except Exception as e:
        print(f"Pricing gap export skipped: {e}")

    try:
        df_weekly = export_weekly_forecast_trend(spark)
        df_weekly.to_parquet(f"{EXPORT_DIR}/weekly_forecast_trend.parquet", index=False)
        plot_forecast_vs_actual(df_weekly, f"{EXPORT_DIR}/forecast_vs_actual.png")
    except Exception as e:
        print(f"Weekly trend export skipped: {e}")

    df_elast = export_elasticity_summary()
    if not df_elast.empty:
        df_elast.to_parquet(f"{EXPORT_DIR}/elasticity_summary.parquet", index=False)
        print(f"Elasticity summary exported: {len(df_elast):,} SKUs")

    print(f"\nAll exports written to {EXPORT_DIR}/")
    spark.stop()


if __name__ == "__main__":
    main()
