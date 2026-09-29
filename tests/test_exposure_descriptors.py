"""Remaining neutralization descriptors: residual risk, liquidity, and statement ratios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_factors.core import DEFAULT_FACTORS, compute_factors
from quant_factors.expressions import expression_requirements

NAMES = (
    "residual_vol_252d",
    "liquidity_log_turnover_20d",
    "book_to_price",
    "earnings_yield_ttm",
    "earnings_growth_yoy",
    "book_leverage",
)


def test_descriptors_are_opt_in_risk_exposures() -> None:
    for name in NAMES:
        assert name not in DEFAULT_FACTORS
        requirement = expression_requirements([name])[name]
        assert requirement["role"] == "risk_exposure"
    assert expression_requirements(["book_to_price"])["book_to_price"]["pit_columns"] == [
        "book_equity"
    ]
    assert (
        "market_cap"
        not in expression_requirements(["book_to_price"])["book_to_price"]["pit_columns"]
    )
    growth = expression_requirements(["earnings_growth_yoy"])["earnings_growth_yoy"]
    assert growth["pit_columns"] == [
        "net_profit_parent_ttm",
        "net_profit_parent_ttm_prior_year",
    ]
    assert expression_requirements(["liquidity_log_turnover_20d"])["liquidity_log_turnover_20d"][
        "pit_columns"
    ] == ["free_float_shares"]


def test_earnings_yield_matches_ep_and_book_ratios_keep_signs() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "symbol": ["A"] * 4,
            "close": [10.0, 10.0, 10.0, 10.0],
            "market_cap": [100.0, 100.0, 0.0, 50.0],
            "net_profit_parent_ttm": [10.0, -5.0, 8.0, 4.0],
            "net_profit_parent_ttm_prior_year": [8.0, 10.0, 5.0, -2.0],
            "book_equity": [40.0, -20.0, 10.0, 0.0],
            "total_assets": [120.0, 80.0, 0.0, 30.0],
            "statement_currency": ["CNY"] * 4,
        }
    )
    out = compute_factors(
        frame,
        ["ep_ttm", "earnings_yield_ttm", "book_to_price", "earnings_growth_yoy", "book_leverage"],
    )
    pd.testing.assert_series_equal(out["earnings_yield_ttm"], out["ep_ttm"], check_names=False)
    assert out["book_to_price"].tolist() == pytest.approx([0.4, -0.2, np.nan, 0.0], nan_ok=True)
    assert out["earnings_growth_yoy"].tolist() == pytest.approx(
        [10.0 / 8.0 - 1.0, -5.0 / 10.0 - 1.0, 8.0 / 5.0 - 1.0, np.nan],
        nan_ok=True,
    )
    assert out["book_leverage"].tolist() == pytest.approx(
        [120.0 / 40.0, np.nan, np.nan, np.nan], nan_ok=True
    )


def test_statement_descriptors_do_not_use_a_panel_lag_or_missing_currency() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=3),
            "symbol": ["A"] * 3,
            "close": [10.0, 11.0, 12.0],
            "market_cap": [100.0, 100.0, 100.0],
            "net_profit_parent_ttm": [10.0, 12.0, 9.0],
            "net_profit_parent_ttm_prior_year": [5.0, 4.0, 3.0],
            "book_equity": [20.0, 20.0, 20.0],
            "total_assets": [40.0, 40.0, 40.0],
            "statement_currency": ["CNY"] * 3,
        }
    )
    out = compute_factors(frame, ["earnings_growth_yoy"])
    lagged = frame["net_profit_parent_ttm"].shift(1)
    assert out["earnings_growth_yoy"].iloc[0] == pytest.approx(10.0 / 5.0 - 1.0)
    assert pd.isna(lagged.iloc[0])
    bare = frame.drop(columns=["statement_currency"])
    with pytest.raises(ValueError, match="statement_currency"):
        compute_factors(bare, ["book_to_price"])
    missing = frame.drop(
        columns=["book_equity", "net_profit_parent_ttm", "net_profit_parent_ttm_prior_year"]
    )
    quiet = compute_factors(missing, ["book_to_price", "earnings_yield_ttm", "book_leverage"])
    assert quiet["book_to_price"].isna().all()
    assert quiet["earnings_yield_ttm"].isna().all()
    assert quiet["book_leverage"].isna().all()


def test_residual_vol_matches_an_independent_window_formula() -> None:
    periods = 253
    generator = np.random.default_rng(7)
    market = generator.normal(0.0, 0.01, periods)
    idiosyncratic = generator.normal(0.0, 0.02, periods)
    returns = np.zeros(periods)
    returns[1:] = 0.8 * market[1:] + idiosyncratic[1:]
    close = np.empty(periods)
    close[0] = 50.0
    for index in range(1, periods):
        close[index] = close[index - 1] * (1.0 + returns[index])
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=periods),
            "symbol": ["A"] * periods,
            "close": close,
            "market_return": market,
        }
    )
    out = compute_factors(frame, ["residual_vol_252d", "beta_252d"])
    window_return = close[1:] / close[:-1] - 1.0
    window_market = market[1:]
    beta = np.cov(window_return, window_market, ddof=1)[0, 1] / np.var(window_market, ddof=1)
    variance = np.var(window_return, ddof=1) - beta**2 * np.var(window_market, ddof=1)
    expected = np.sqrt(max(variance, 0.0)) * np.sqrt(252.0)
    assert out["beta_252d"].iloc[252] == pytest.approx(beta)
    assert out["residual_vol_252d"].iloc[252] == pytest.approx(expected)
    assert out["residual_vol_252d"].iloc[252] > 0
    assert pd.isna(out["residual_vol_252d"].iloc[251])
    exact = frame.copy()
    price = np.empty(periods)
    price[0] = 50.0
    for index in range(1, periods):
        price[index] = price[index - 1] * (1.0 + 1.5 * market[index])
    exact["close"] = price
    fitted = compute_factors(exact, ["residual_vol_252d"])
    assert fitted["residual_vol_252d"].iloc[252] == pytest.approx(0.0, abs=1e-8)


def test_log_turnover_uses_raw_float_and_drops_nonpositive_rates() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=21),
            "symbol": ["A"] * 21,
            "close": np.linspace(10.0, 12.0, 21),
            "volume": [10.0] * 20 + [0.0],
            "free_float_shares": [100.0] * 21,
            "volume_unit": ["shares"] * 21,
            "share_basis": ["raw"] * 21,
        }
    )
    out = compute_factors(frame, ["liquidity_log_turnover_20d"])
    assert pd.isna(out["liquidity_log_turnover_20d"].iloc[18])
    assert out["liquidity_log_turnover_20d"].iloc[19] == pytest.approx(np.log(0.1))
    assert out["liquidity_log_turnover_20d"].iloc[20] == pytest.approx(np.log(0.095))
    bad_unit = frame.copy()
    bad_unit["volume_unit"] = "lots"
    with pytest.raises(ValueError, match="raw share units"):
        compute_factors(bad_unit, ["liquidity_log_turnover_20d"])
    quiet = compute_factors(
        frame.drop(columns=["free_float_shares"]), ["liquidity_log_turnover_20d"]
    )
    assert quiet["liquidity_log_turnover_20d"].isna().all()
    changed = frame.copy()
    changed.loc[20, "volume"] = 1_000.0
    again = compute_factors(changed, ["liquidity_log_turnover_20d"])
    assert again["liquidity_log_turnover_20d"].iloc[19] == pytest.approx(
        out["liquidity_log_turnover_20d"].iloc[19]
    )
