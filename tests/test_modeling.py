"""
tests/test_modeling.py

Unit tests for elasticity modeling, ARIMA fitting, ensemble logic,
and feature engineering functions. Uses synthetic data to avoid
Snowflake and Spark dependencies in CI.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima.model import ARIMA


def make_sku_series(n_weeks: int = 104, trend: float = 0.5, noise: float = 5.0) -> pd.Series:
    t = np.arange(n_weeks)
    signal = 100 + trend * t + noise * np.random.randn(n_weeks)
    return pd.Series(np.clip(signal, 0, None))


def make_feature_df(n_skus: int = 20, n_weeks: int = 104) -> pd.DataFrame:
    records = []
    for sku in range(n_skus):
        base = np.random.randint(50, 300)
        for w in range(n_weeks):
            records.append({
                "item_nbr": sku,
                "store_nbr": sku % 5,
                "week_start": pd.Timestamp("2018-01-01") + pd.Timedelta(weeks=w),
                "year": 2018 + w // 52,
                "week_of_year": (w % 52) + 1,
                "month": ((w % 52) // 4) + 1,
                "quarter": ((w % 52) // 13) + 1,
                "is_holiday_season": int((w % 52) >= 48),
                "weekly_units": max(0, base + 10 * np.random.randn()),
                "any_promo": np.random.randint(0, 2),
                "promo_rate_4w": np.random.uniform(0, 1),
                "rolling_mean_4w": base + np.random.randn() * 5,
                "rolling_mean_12w": base,
                "rolling_mean_26w": base,
                "rolling_std_4w": 10.0,
                "rolling_std_12w": 12.0,
                "units_lag_1w": max(0, base + np.random.randn() * 5),
                "units_lag_4w": max(0, base + np.random.randn() * 5),
                "units_lag_52w": max(0, base + np.random.randn() * 5),
                "yoy_growth": np.random.uniform(-0.2, 0.2),
                "price_gap_vs_family": np.random.uniform(-0.3, 0.3),
                "family": f"FAM_{sku % 5}",
            })
    return pd.DataFrame(records)


class TestARIMAFitting:
    def test_arima_returns_correct_horizon(self):
        from src.modeling.demand_forecast import fit_arima_per_sku
        series = make_sku_series(n_weeks=104)
        forecast = fit_arima_per_sku(series, horizon=12)
        assert len(forecast) == 12

    def test_arima_short_series_returns_zeros(self):
        from src.modeling.demand_forecast import fit_arima_per_sku
        series = make_sku_series(n_weeks=10)
        forecast = fit_arima_per_sku(series, horizon=12)
        assert len(forecast) == 12
        assert all(f == 0 for f in forecast)

    def test_arima_non_negative_forecasts(self):
        from src.modeling.demand_forecast import fit_arima_per_sku
        series = make_sku_series(n_weeks=104)
        forecast = fit_arima_per_sku(series, horizon=12)
        assert all(f >= 0 for f in forecast)


class TestEnsemble:
    def test_ensemble_weighted_average(self):
        from src.modeling.demand_forecast import ensemble_forecasts
        xgb_preds = np.array([100.0, 200.0, 150.0])
        arima_preds = np.array([90.0, 210.0, 140.0])
        result = ensemble_forecasts(xgb_preds, arima_preds, xgb_weight=0.65, arima_weight=0.35)
        expected = 0.65 * xgb_preds + 0.35 * arima_preds
        np.testing.assert_array_almost_equal(result, expected)

    def test_ensemble_weights_sum(self):
        from src.modeling.demand_forecast import XGB_ENSEMBLE_WEIGHT, ARIMA_ENSEMBLE_WEIGHT
        assert abs(XGB_ENSEMBLE_WEIGHT + ARIMA_ENSEMBLE_WEIGHT - 1.0) < 1e-6


class TestMAPE:
    def test_mape_zero_actuals_excluded(self):
        from src.modeling.demand_forecast import mape
        y_true = np.array([0, 100, 200])
        y_pred = np.array([50, 110, 190])
        result = mape(y_true, y_pred)
        assert 0 < result < 1

    def test_mape_perfect_forecast(self):
        from src.modeling.demand_forecast import mape
        y = np.array([100.0, 200.0, 300.0])
        result = mape(y, y)
        assert result == 0.0

    def test_mape_all_zeros_returns_nan(self):
        from src.modeling.demand_forecast import mape
        result = mape(np.array([0.0, 0.0]), np.array([10.0, 20.0]))
        assert np.isnan(result)


class TestElasticityFeatures:
    def test_build_log_log_features_no_nulls_in_log_units(self):
        from src.modeling.elasticity_model import build_log_log_features
        df = make_feature_df(n_skus=5, n_weeks=60)
        df_out = build_log_log_features(df)
        assert df_out["log_units"].notna().all()

    def test_did_columns_created(self):
        from src.modeling.elasticity_model import build_log_log_features
        df = make_feature_df(n_skus=5, n_weeks=60)
        df_out = build_log_log_features(df)
        assert "post" in df_out.columns
        assert "t" in df_out.columns

    def test_ols_returns_none_for_small_sku(self):
        from src.modeling.elasticity_model import fit_hierarchical_ols, build_log_log_features
        df = make_feature_df(n_skus=1, n_weeks=10)
        df = build_log_log_features(df)
        result = fit_hierarchical_ols(df, "sku_0")
        assert result is None

    def test_ols_returns_dict_for_sufficient_data(self):
        from src.modeling.elasticity_model import fit_hierarchical_ols, build_log_log_features
        df = make_feature_df(n_skus=1, n_weeks=104)
        df = build_log_log_features(df)
        result = fit_hierarchical_ols(df, "sku_0")
        assert result is not None
        assert "elasticity_promo" in result
        assert "r_squared" in result
        assert 0 <= result["r_squared"] <= 1
