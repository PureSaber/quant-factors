"""Descriptors that complete the public style set. They are not MSCI descriptors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_factors.core import DEFAULT_FACTORS, compute_factors
from quant_factors.expressions import expression_requirements

GAP_NAMES = (
    "liquidity_log_turnover_60d",
    "liquidity_log_turnover_240d",
    "beta_ew_252d",
    "short_term_reversal_21d",
    "long_term_reversal_504_252",
    "seasonality_21d_lag_252",
    "dividend_yield_ttm",
    "cash_earnings_yield_ttm",
    "market_leverage",
    "debt_to_assets",
    "sales_growth_yoy",
    "earnings_variability_1260d",
    "analyst_revision",
    "forward_earnings_yield",
    "expected_growth",
    "industry_momentum_20d",
)


def test_gap_descriptors_are_opt_in() -> None:
    for name in GAP_NAMES:
        assert name not in DEFAULT_FACTORS
        assert expression_requirements([name])[name]["role"] == "risk_exposure"


def test_exponential_beta_tracks_a_recent_shock_more_than_equal_weight() -> None:
    periods = 280
    market = np.linspace(-0.02, 0.02, periods)
    close = np.empty(periods)
    close[0] = 100.0
    for index in range(1, periods):
        close[index] = close[index - 1] * (1.0 + market[index])
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=periods),
            "symbol": ["A"] * periods,
            "close": close,
            "market_return": market,
        }
    )
    base = compute_factors(frame, ["beta_252d", "beta_ew_252d"])
    assert pd.isna(base["beta_ew_252d"].iloc[251])
    assert base["beta_ew_252d"].iloc[252] == pytest.approx(1.0)
    shocked = frame.copy()
    shocked.loc[periods - 1, "close"] = close[-2] * (1.0 + 4.0 * market[-1])
    again = compute_factors(shocked, ["beta_252d", "beta_ew_252d"])
    equal_move = abs(again["beta_252d"].iloc[-1] - base["beta_252d"].iloc[-1])
    exponential_move = abs(again["beta_ew_252d"].iloc[-1] - base["beta_ew_252d"].iloc[-1])
    assert exponential_move > equal_move


def test_reversal_seasonality_and_industry_momentum() -> None:
    close = np.arange(1, 520, dtype=float)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2018-01-01", periods=len(close)),
            "symbol": ["A"] * len(close),
            "close": close,
        }
    )
    out = compute_factors(
        frame, ["short_term_reversal_21d", "long_term_reversal_504_252", "seasonality_21d_lag_252"]
    )
    assert out["short_term_reversal_21d"].iloc[21] == pytest.approx(-(22.0 / 1.0 - 1.0))
    assert pd.isna(out["long_term_reversal_504_252"].iloc[503])
    assert out["long_term_reversal_504_252"].iloc[504] == pytest.approx(-(253.0 / 1.0 - 1.0))
    assert pd.isna(out["seasonality_21d_lag_252"].iloc[251])
    assert out["seasonality_21d_lag_252"].iloc[252] == pytest.approx(22.0 / 1.0 - 1.0)

    rally = np.r_[np.full(300, 100.0), np.linspace(100.0, 200.0, 21), np.full(300, 200.0)]
    seasonal = compute_factors(
        pd.DataFrame(
            {
                "date": pd.bdate_range("2018-01-01", periods=len(rally)),
                "symbol": ["A"] * len(rally),
                "close": rally,
            }
        ),
        ["seasonality_21d_lag_252"],
    )
    assert seasonal["seasonality_21d_lag_252"].iloc[300 + 252] == pytest.approx(1.0)
    assert seasonal["seasonality_21d_lag_252"].iloc[300 + 252 + 21] == pytest.approx(0.0)

    dates = pd.bdate_range("2020-01-01", periods=21)
    rows = []
    last = {"A1": 110.0, "A2": 130.0, "B1": 150.0}
    industry = {"A1": "Banks", "A2": "Banks", "B1": "Energy"}
    for symbol in ("A1", "A2", "B1"):
        for position, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 100.0 if position < 20 else last[symbol],
                    "industry": industry[symbol],
                }
            )
    momentum = compute_factors(pd.DataFrame(rows), ["industry_momentum_20d"])
    last_day = momentum.loc[momentum.date.eq(dates[-1])].set_index("symbol")
    assert last_day.loc["A1", "industry_momentum_20d"] == pytest.approx(0.30)
    assert last_day.loc["A2", "industry_momentum_20d"] == pytest.approx(0.10)
    assert pd.isna(last_day.loc["B1", "industry_momentum_20d"])


def test_statement_and_supplied_descriptors_keep_missing_data_missing() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "symbol": ["A", "B"],
            "close": [10.0, 10.0],
            "market_cap": [100.0, 50.0],
            "statement_currency": ["CNY", "CNY"],
            "cash_dividend_ttm": [2.0, -1.0],
            "operating_cashflow_ttm": [8.0, -4.0],
            "total_debt": [25.0, 0.0],
            "total_assets": [80.0, 40.0],
            "revenue_ttm": [12.0, 9.0],
            "revenue_ttm_prior_year": [10.0, 0.0],
            "forward_net_profit": [5.0, -2.0],
            "analyst_revision": [0.3, np.nan],
            "expected_growth": [0.1, np.nan],
            "net_profit_parent_ttm": [4.0, 1.0],
        }
    )
    out = compute_factors(
        frame,
        [
            "dividend_yield_ttm",
            "cash_earnings_yield_ttm",
            "market_leverage",
            "debt_to_assets",
            "sales_growth_yoy",
            "forward_earnings_yield",
            "analyst_revision",
            "expected_growth",
        ],
    )
    assert out["dividend_yield_ttm"].iloc[0] == pytest.approx(0.02)
    assert pd.isna(out["dividend_yield_ttm"].iloc[1])
    assert out["cash_earnings_yield_ttm"].iloc[1] == pytest.approx(-4.0 / 50.0)
    assert out["market_leverage"].iloc[0] == pytest.approx(1.25)
    assert out["debt_to_assets"].iloc[0] == pytest.approx(25.0 / 80.0)
    assert out["sales_growth_yoy"].iloc[0] == pytest.approx(0.2)
    assert pd.isna(out["sales_growth_yoy"].iloc[1])
    assert out["forward_earnings_yield"].iloc[1] == pytest.approx(-2.0 / 50.0)
    assert out["analyst_revision"].iloc[0] == pytest.approx(0.3)
    assert pd.isna(out["analyst_revision"].iloc[1])
    assert pd.isna(out["expected_growth"].iloc[1])
    missing = compute_factors(frame.drop(columns=["analyst_revision"]), ["analyst_revision"])
    assert missing["analyst_revision"].isna().all()


def test_earnings_variability_ignores_price_and_needs_a_full_window() -> None:
    periods = 1261
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2015-01-01", periods=periods),
            "symbol": ["A"] * periods,
            "close": np.linspace(10.0, 30.0, periods),
            "net_profit_parent_ttm": np.full(periods, 5.0),
            "statement_currency": ["CNY"] * periods,
        }
    )
    flat = compute_factors(frame, ["earnings_variability_1260d"])
    assert pd.isna(flat["earnings_variability_1260d"].iloc[1258])
    assert flat["earnings_variability_1260d"].iloc[1259] == pytest.approx(0.0)
    stepped = frame.assign(net_profit_parent_ttm=np.r_[np.full(630, 4.0), np.full(631, 6.0)])
    out = compute_factors(stepped, ["earnings_variability_1260d"])
    assert out["earnings_variability_1260d"].iloc[1259] == pytest.approx(
        np.std(np.r_[np.full(630, 4.0), np.full(630, 6.0)], ddof=1) / 5.0
    )
