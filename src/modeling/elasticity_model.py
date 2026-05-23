"""
modeling/elasticity_model.py

Estimates price elasticity for each SKU using:
  1. OLS hierarchical regression (store-level random effects approximated
     via fixed-effect dummies to stay within statsmodels).
  2. Difference-in-Differences (DiD) causal inference to isolate the
     causal effect of promotions on unit sales, controlling for time trends.

Logs all experiments to MLflow with per-SKU elasticity coefficients.
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

INPUT_PATH = "data/delta/features"
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "pricing/elasticity")
MIN_OBS = int(os.getenv("ELASTICITY_MIN_OBS", 52))


def build_spark():
    return (
        SparkSession.builder
        .appName("elasticity_model")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def load_features(spark) -> pd.DataFrame:
    df = spark.read.format("delta").load(INPUT_PATH)
    cols = [
        "week_start", "store_nbr", "item_nbr", "family", "cluster",
        "weekly_units", "any_promo", "promo_rate_4w",
        "week_of_year", "year", "rolling_mean_4w", "price_gap_vs_family",
    ]
    return df.select(cols).toPandas()


def build_log_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-log specification: log(units) ~ log(price_proxy) + controls.
    We use rolling_mean_4w as a baseline demand proxy to derive an implicit
    price index when explicit prices are absent from the dataset.
    """
    df = df.copy()
    df["log_units"] = np.log1p(df["weekly_units"])
    df["log_demand_baseline"] = np.log1p(df["rolling_mean_4w"].fillna(0))
    df["t"] = df.groupby(["store_nbr", "item_nbr"])["week_start"].rank()
    df["post"] = (df["year"] >= df["year"].median()).astype(int)
    return df.dropna(subset=["log_units", "log_demand_baseline"])


def fit_hierarchical_ols(df_sku: pd.DataFrame, sku_id: str) -> dict:
    """
    Store-level fixed effects OLS regression per SKU.
    Elasticity is captured by the promotion coefficient in the log-log model.
    """
    if len(df_sku) < MIN_OBS:
        return None

    df_sku = df_sku.copy()
    df_sku["store_fe"] = pd.Categorical(df_sku["store_nbr"]).codes

    formula = (
        "log_units ~ any_promo + log_demand_baseline + "
        "week_of_year + t + C(store_fe)"
    )
    try:
        model = smf.ols(formula=formula, data=df_sku).fit()
        return {
            "sku": sku_id,
            "elasticity_promo": model.params.get("any_promo", np.nan),
            "elasticity_baseline": model.params.get("log_demand_baseline", np.nan),
            "r_squared": model.rsquared,
            "n_obs": int(model.nobs),
            "p_value_promo": model.pvalues.get("any_promo", np.nan),
            "aic": model.aic,
        }
    except Exception as exc:
        print(f"OLS failed for SKU {sku_id}: {exc}")
        return None


def fit_did(df_sku: pd.DataFrame, sku_id: str) -> dict:
    """
    Difference-in-Differences specification:

        log(units) = alpha + beta1*treated + beta2*post + beta3*(treated x post) + eps

    treated  = any_promo (1 = promotion week)
    post     = year >= median year (pre/post price intervention window)
    DiD ATT  = beta3 (average treatment effect on the treated)
    """
    if len(df_sku) < MIN_OBS:
        return None

    df_sku = df_sku.copy()
    df_sku["treated"] = df_sku["any_promo"]
    df_sku["treated_post"] = df_sku["treated"] * df_sku["post"]

    X = sm.add_constant(
        df_sku[["treated", "post", "treated_post", "t", "week_of_year"]]
    )
    try:
        model = sm.OLS(df_sku["log_units"], X).fit(cov_type="HC3")
        return {
            "sku": sku_id,
            "did_att": model.params.get("treated_post", np.nan),
            "did_se": model.bse.get("treated_post", np.nan),
            "did_pvalue": model.pvalues.get("treated_post", np.nan),
            "did_r2": model.rsquared,
        }
    except Exception as exc:
        print(f"DiD failed for SKU {sku_id}: {exc}")
        return None


def run_elasticity_pipeline(df: pd.DataFrame):
    df = build_log_log_features(df)
    skus = df["item_nbr"].unique()
    print(f"Fitting elasticity models for {len(skus):,} SKUs...")

    ols_results = []
    did_results = []

    for sku in skus:
        df_sku = df[df["item_nbr"] == sku]
        sku_id = str(sku)

        ols_result = fit_hierarchical_ols(df_sku, sku_id)
        if ols_result:
            ols_results.append(ols_result)

        did_result = fit_did(df_sku, sku_id)
        if did_result:
            did_results.append(did_result)

    ols_df = pd.DataFrame(ols_results)
    did_df = pd.DataFrame(did_results)

    combined = ols_df.merge(did_df, on="sku", how="outer")
    return combined


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    spark = build_spark()

    print("Loading features...")
    df = load_features(spark)

    with mlflow.start_run(run_name="elasticity_hierarchical_did"):
        mlflow.log_param("min_obs_per_sku", MIN_OBS)
        mlflow.log_param("specification", "log_log_ols_did")

        results = run_elasticity_pipeline(df)

        median_elasticity = results["elasticity_promo"].median()
        pct_significant = (results["did_pvalue"] < 0.05).mean()
        mean_r2 = results["r_squared"].mean()

        mlflow.log_metric("median_promo_elasticity", round(float(median_elasticity), 4))
        mlflow.log_metric("pct_did_significant_skus", round(float(pct_significant), 4))
        mlflow.log_metric("mean_ols_r2", round(float(mean_r2), 4))
        mlflow.log_metric("n_skus_modeled", len(results))

        out_path = "data/elasticity_results.parquet"
        results.to_parquet(out_path, index=False)
        mlflow.log_artifact(out_path)

        print(f"Modeled {len(results):,} SKUs")
        print(f"Median promo elasticity: {median_elasticity:.4f}")
        print(f"DiD significant SKUs: {pct_significant:.1%}")

    spark.stop()


if __name__ == "__main__":
    main()
