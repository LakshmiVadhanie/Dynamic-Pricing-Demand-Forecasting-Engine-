"""
mlflow_config/registry.py

Helpers for promoting models through the MLflow model registry:
  Staging -> Production, and archiving old versions.
Also provides a convenience function to compare run metrics across experiments.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

REGISTERED_MODEL_NAME = "favorita_demand_forecast"
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "pricing/demand_forecast")


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    return MlflowClient()


def register_best_run(mape_threshold: float = 0.15) -> str:
    """
    Find the best run (lowest MAPE below threshold) and register it.
    Returns the registered model version string.
    """
    client = get_client()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"Experiment '{MLFLOW_EXPERIMENT}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.ensemble_holdout_mape < {mape_threshold}",
        order_by=["metrics.ensemble_holdout_mape ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No run with MAPE < {mape_threshold} found.")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/xgb_model"
    print(f"Registering model from run {best_run.info.run_id} "
          f"(MAPE={best_run.data.metrics['ensemble_holdout_mape']:.4f})")

    result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    print(f"Registered as version {result.version}")
    return result.version


def promote_to_production(version: str):
    client = get_client()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Version {version} promoted to Production.")


def get_production_uri() -> str:
    return f"models:/{REGISTERED_MODEL_NAME}/Production"


def print_run_comparison(n_runs: int = 10):
    client = get_client()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.ensemble_holdout_mape ASC"],
        max_results=n_runs,
    )
    print(f"\nTop {n_runs} runs by ensemble MAPE:\n")
    print(f"{'Run ID':<36}  {'MAPE':>8}  {'XGB CV MAPE':>12}  {'n_estimators':>14}")
    print("-" * 75)
    for r in runs:
        mape = r.data.metrics.get("ensemble_holdout_mape", float("nan"))
        xgb_cv = r.data.metrics.get("xgb_cv_mape", float("nan"))
        n_est = r.data.params.get("xgb_n_estimators", "N/A")
        print(f"{r.info.run_id:<36}  {mape:>8.4f}  {xgb_cv:>12.4f}  {n_est:>14}")


if __name__ == "__main__":
    print_run_comparison()
