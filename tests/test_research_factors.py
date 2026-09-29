"""Daily and intraday research factors stay outside the default registry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "src" / "quant_factors"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


research = _load("research_factors_under_test", "research_factors.py")
core = _load("core_under_test", "core.py")


def _prices(rows: int, **extra) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(10, 20, rows)
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAA",
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(rows, 100.0),
        }
    )
    for key, value in extra.items():
        frame[key] = value
    return frame


def test_names_do_not_replace_the_default_registry() -> None:
    assert set(research.RESEARCH_FACTORS).isdisjoint(core.FACTOR_REGISTRY)
    assert "close_to_year_high" in research.RESEARCH_FACTORS
    assert "smart_money" in research.MINUTE_FACTORS


def test_amplitude_split_separates_wide_and_narrow_days() -> None:
    close = [100.0]
    high = [100.0]
    low = [100.0]
    for step in range(1, 21):
        ret = 0.05 if step % 2 else -0.01
        close.append(close[-1] * (1 + ret))
        if step % 2:
            high.append(close[-1] * 1.2)
            low.append(close[-1] * 0.8)
        else:
            high.append(close[-1] * 1.001)
            low.append(close[-1] * 0.999)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=21, freq="D"),
            "symbol": "AAA",
            "close": close,
            "high": high,
            "low": low,
        }
    )
    result = research.compute_named_research_factors(
        frame,
        [
            "high_amplitude_return_20d",
            "low_amplitude_return_20d",
            "amplitude_return_spread_20d",
        ],
    )
    assert result["high_amplitude_return_20d"].iloc[-1] == pytest.approx(0.5)
    assert result["low_amplitude_return_20d"].iloc[-1] == pytest.approx(-0.1)
    assert result["amplitude_return_spread_20d"].iloc[-1] == pytest.approx(0.6)
    assert np.isnan(result["high_amplitude_return_20d"].iloc[-2])


def test_year_high_max_and_future_close() -> None:
    close = np.full(252, 10.0)
    close[10] = 50.0
    close[-1] = 40.0
    frame = _prices(252)
    frame["close"] = close
    ratio = research.compute_named_research_factors(frame, ["close_to_year_high"])
    assert ratio["close_to_year_high"].iloc[-1] == pytest.approx(0.8)

    returns = np.full(22, 0.01)
    returns[-1] = 0.2
    prices = [10.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    max_frame = _prices(23)
    max_frame["close"] = prices
    maximum = research.compute_named_research_factors(
        max_frame, ["max_return_21d", "max5_return_21d"]
    )
    assert maximum["max_return_21d"].iloc[-1] == pytest.approx(0.2)
    changed = max_frame.copy()
    changed.loc[changed.index[-1], "close"] = changed["close"].iloc[-1] * 2
    again = research.compute_named_research_factors(changed, ["max_return_21d"])
    assert again["max_return_21d"].iloc[-2] == pytest.approx(maximum["max_return_21d"].iloc[-2])


def test_market_residual_is_zero_when_return_matches_market() -> None:
    market = np.linspace(0.001, 0.02, 30)
    prices = [100.0]
    for ret in market:
        prices.append(prices[-1] * (1 + 2 * ret))
    frame = _prices(31)
    frame["close"] = prices
    frame["market_return"] = np.concatenate([[np.nan], market])
    result = research.compute_named_research_factors(
        frame, ["idio_vol_21d", "return_specificity_21d", "residual_return_21d"]
    )
    assert result["idio_vol_21d"].iloc[-1] == pytest.approx(0.0, abs=1e-10)
    assert result["return_specificity_21d"].iloc[-1] == pytest.approx(0.0, abs=1e-10)
    assert result["residual_return_21d"].iloc[-1] == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(research.ResearchFactorError, match="market_return"):
        research.compute_named_research_factors(
            frame.drop(columns=["market_return"]), ["idio_vol_21d"]
        )


def test_turnover_fundamentals_and_positioning() -> None:
    rows = 60
    frame = _prices(rows)
    frame["volume_unit"] = "shares"
    frame["share_basis"] = "raw"
    frame["free_float_shares"] = 50.0
    frame["afternoon_open"] = frame["close"]
    frame["open"] = frame["close"].shift(1).fillna(frame["close"]) * 1.01
    frame["net_income"] = 10.0
    frame["operating_cash_flow"] = 4.0
    frame["average_assets"] = 100.0
    frame["revenue"] = 30.0
    frame["cogs"] = 12.0
    frame["total_assets"] = 10.0
    frame.loc[frame.index[2:], "total_assets"] = 12.0
    frame["sga"] = 3.0
    frame["interest_expense"] = 1.0
    frame["book_equity"] = 20.0
    frame["consensus_eps"] = 1.0
    frame.loc[frame.index[-1], "consensus_eps"] = 1.5
    frame["reported_eps"] = 1.2
    frame["northbound_hold_ratio"] = 1.0
    frame.loc[frame.index[-1], "northbound_hold_ratio"] = 1.2
    result = research.compute_named_research_factors(
        frame,
        [
            "turnover_rate_vol_20d",
            "accruals_to_assets",
            "gross_profitability",
            "operating_profitability",
            "asset_growth",
            "earnings_surprise",
            "northbound_hold_change_5d",
        ],
    )
    assert result["turnover_rate_vol_20d"].iloc[-1] == pytest.approx(0.0)
    assert result["accruals_to_assets"].iloc[-1] == pytest.approx(0.06)
    assert result["gross_profitability"].iloc[-1] == pytest.approx(18 / 12)
    assert result["operating_profitability"].iloc[-1] == pytest.approx((30 - 12 - 3 - 1) / 20)
    assert result["asset_growth"].iloc[-1] == pytest.approx(0.2)
    assert result["asset_growth"].iloc[1] != result["asset_growth"].iloc[1]
    assert result["earnings_surprise"].iloc[-1] == pytest.approx(
        (1.2 - 1.5) / frame["close"].iloc[-1]
    )
    assert result["northbound_hold_change_5d"].iloc[-1] == pytest.approx(0.2)


def test_smart_money_uses_only_supplied_minute_bars() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "date": ["2024-01-02"] * 4,
            "time": [1, 2, 3, 4],
            "close": [10.0, 10.0, 11.0, 10.0],
            "volume": [10.0, 10.0, 10.0, 70.0],
        }
    )
    result = research.compute_minute_factors(frame, ["smart_money", "tail_bar_volume_share"])
    assert result["smart_money"].iloc[0] == pytest.approx(10.125 / 10.1 - 1)
    assert result["tail_bar_volume_share"].iloc[0] == pytest.approx(0.7)
    with pytest.raises(research.ResearchFactorError, match="close"):
        research.compute_minute_factors(frame.drop(columns=["close"]), ["smart_money"])


def test_medium_and_low_frequency_factors() -> None:
    rows = 274
    close = 100 * (1.01 ** np.arange(rows))
    frame = _prices(rows)
    frame["close"] = close
    frame["volume"] = 100.0
    frame["volume_unit"] = "shares"
    frame["share_basis"] = "raw"
    frame["free_float_shares"] = 50.0
    frame["revenue"] = 10.0
    frame.loc[frame.index[2:], "revenue"] = 12.0
    frame["net_income"] = 10.0
    frame.loc[frame.index[2:], "net_income"] = 15.0
    frame["cogs"] = 4.0
    frame["total_assets"] = 12.0
    frame["book_equity"] = 20.0
    frame["market_cap"] = 100.0
    result = research.compute_named_research_factors(
        frame,
        [
            "momentum_12_1",
            "momentum_6_1",
            "turnover_rate_120d",
            "sales_growth",
            "earnings_growth",
            "asset_turnover",
            "financial_leverage",
            "gross_margin",
            "roe",
            "book_to_price",
            "same_month_return",
        ],
    )
    assert result["momentum_12_1"].iloc[252] == pytest.approx(close[231] / close[0] - 1)
    assert result["momentum_6_1"].iloc[126] == pytest.approx(close[105] / close[0] - 1)
    assert result["turnover_rate_120d"].iloc[-1] == pytest.approx(2.0)
    assert result["sales_growth"].iloc[-1] == pytest.approx(0.2)
    assert result["earnings_growth"].iloc[-1] == pytest.approx(0.5)
    assert result["asset_turnover"].iloc[-1] == pytest.approx(1.0)
    assert result["financial_leverage"].iloc[-1] == pytest.approx(0.6)
    assert result["gross_margin"].iloc[-1] == pytest.approx(8 / 12)
    assert result["roe"].iloc[-1] == pytest.approx(0.75)
    assert result["book_to_price"].iloc[-1] == pytest.approx(0.2)
    market = np.linspace(0.001, 0.02, rows)
    market[0] = np.nan
    prices = [100.0]
    for ret in market[1:]:
        prices.append(prices[-1] * (1 + ret))
    matched = frame.copy()
    matched["close"] = prices
    matched["market_return"] = market
    residual = research.compute_named_research_factors(
        matched, ["beta_252d", "residual_momentum_12_1"]
    )
    assert residual["beta_252d"].iloc[-1] == pytest.approx(1.0, abs=1e-8)
    assert residual["residual_momentum_12_1"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert result["same_month_return"].iloc[273] == pytest.approx(close[21] / close[0] - 1)
    lows = frame.copy()
    lows["close"] = np.full(rows, 10.0)
    lows.loc[lows.index[-10], "close"] = 4.0
    lows.loc[lows.index[-1], "close"] = 8.0
    ratio = research.compute_named_research_factors(lows, ["close_to_year_low"])
    assert ratio["close_to_year_low"].iloc[-1] == pytest.approx(2.0)
