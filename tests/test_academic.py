"""Academic characteristics: identities, PIT labels, and opt-in registration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_factors.academic import academic_requirement
from quant_factors.core import DEFAULT_FACTORS, compute_factors, factor_requires_fundamental
from quant_factors.expressions import expression_requirements


def _statement_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "symbol": ["A"] * 4,
            "close": [10.0, 10.0, 11.0, 12.0],
            "market_cap": [1000.0, 500.0, 100.0, 0.0],
            "net_profit_parent_ttm": [100.0, -50.0, 0.0, 10.0],
            "book_equity": [200.0, 0.0, -20.0, 50.0],
            "gross_profit": [30.0, -5.0, -5.0, 8.0],
            "total_assets": [300.0, 0.0, 120.0, 80.0],
            "total_assets_prior_year": [250.0, 100.0, 0.0, 100.0],
            "statement_currency": ["CNY"] * 4,
            "pe_ratio": [-5.0, -5.0, -5.0, -5.0],
        }
    )


def test_academic_names_stay_out_of_the_default_set() -> None:
    assert academic_requirement("momentum_20d") is None
    for name in (
        "ep_ttm",
        "roe_ttm",
        "gross_profitability",
        "asset_growth_yoy",
        "reversal_20d",
    ):
        assert name not in DEFAULT_FACTORS
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=3),
            "symbol": ["A"] * 3,
            "close": [10.0, 11.0, 12.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    assert "ep_ttm" not in compute_factors(frame).columns


def test_earnings_yield_keeps_losses_and_rejects_nonpositive_cap() -> None:
    out = compute_factors(_statement_frame(), ["ep_ttm"])
    assert out["ep_ttm"].tolist() == pytest.approx([0.1, -0.1, 0.0, np.nan], nan_ok=True)
    assert factor_requires_fundamental("ep_ttm")
    requirements = expression_requirements(["ep_ttm"])["ep_ttm"]
    assert requirements["pit_columns"] == ["net_profit_parent_ttm"]
    assert "market_cap" in requirements["columns"]
    assert "market_cap" not in requirements["pit_columns"]
    assert requirements["role"] == "characteristic"


def test_roe_and_gross_profit_use_positive_stocks_only() -> None:
    out = compute_factors(_statement_frame(), ["roe_ttm", "gross_profitability"])
    assert out["roe_ttm"].tolist() == pytest.approx([0.5, np.nan, np.nan, 0.2], nan_ok=True)
    assert out["gross_profitability"].tolist() == pytest.approx(
        [0.1, np.nan, -5.0 / 120.0, 0.1], nan_ok=True
    )
    roe = expression_requirements(["roe_ttm"])["roe_ttm"]
    assert roe["pit_columns"] == ["net_profit_parent_ttm", "book_equity"]
    assert roe["warmup_bars"] == 1


def test_asset_growth_uses_supplied_prior_year_not_a_panel_shift() -> None:
    frame = _statement_frame()
    out = compute_factors(frame, ["asset_growth_yoy"])
    assert out["asset_growth_yoy"].iloc[0] == pytest.approx(300.0 / 250.0 - 1.0)
    assert pd.isna(out["asset_growth_yoy"].iloc[1])
    assert pd.isna(out["asset_growth_yoy"].iloc[2])
    assert out["asset_growth_yoy"].iloc[3] == pytest.approx(80.0 / 100.0 - 1.0)
    assert pd.isna(frame["total_assets"].shift(1).iloc[0])
    requirements = expression_requirements(["asset_growth_yoy"])["asset_growth_yoy"]
    assert requirements["pit_columns"] == ["total_assets", "total_assets_prior_year"]


def test_one_month_reversal_is_minus_momentum_and_not_fundamental() -> None:
    close = np.linspace(10.0, 20.0, 30)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=30),
            "symbol": ["A"] * 30,
            "close": close,
            "volume": 1.0,
        }
    )
    out = compute_factors(frame, ["reversal_20d", "momentum_20d"])
    pd.testing.assert_series_equal(out["reversal_20d"], -out["momentum_20d"], check_names=False)
    assert out["reversal_20d"].notna().any()
    assert factor_requires_fundamental("reversal_20d") is False
    assert expression_requirements(["reversal_20d"])["reversal_20d"]["warmup_bars"] == 21
    zero = frame.copy()
    zero.loc[0, "close"] = 0.0
    zero_out = compute_factors(zero, ["reversal_20d", "momentum_20d"])
    pd.testing.assert_series_equal(
        zero_out["reversal_20d"], -zero_out["momentum_20d"], check_names=False
    )
    assert np.isneginf(zero_out["reversal_20d"].iloc[20])


def test_missing_statement_inputs_stay_missing_without_raising() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=3),
            "symbol": ["A"] * 3,
            "close": [10.0, 11.0, 12.0],
        }
    )
    out = compute_factors(frame, ["ep_ttm", "roe_ttm", "gross_profitability", "asset_growth_yoy"])
    for name in ("ep_ttm", "roe_ttm", "gross_profitability", "asset_growth_yoy"):
        assert out[name].isna().all()


def test_statement_currency_contract_fails_closed() -> None:
    frame = _statement_frame().drop(columns=["statement_currency"])
    with pytest.raises(ValueError, match="statement_currency"):
        compute_factors(frame, ["ep_ttm"])
    mixed = _statement_frame()
    mixed.loc[1, "statement_currency"] = "HKD"
    with pytest.raises(ValueError, match="one non-null"):
        compute_factors(mixed, ["roe_ttm"])
    blank = _statement_frame()
    blank.loc[0, "statement_currency"] = None
    with pytest.raises(ValueError, match="one non-null"):
        compute_factors(blank, ["gross_profitability"])


def test_later_rows_do_not_change_earlier_characteristics() -> None:
    frame = _statement_frame()
    original = compute_factors(frame, ["ep_ttm", "asset_growth_yoy", "reversal_20d"])
    changed = frame.copy()
    changed.loc[3, ["net_profit_parent_ttm", "total_assets", "close"]] = [999.0, 999.0, 999.0]
    again = compute_factors(changed, ["ep_ttm", "asset_growth_yoy", "reversal_20d"])
    pd.testing.assert_frame_equal(original.iloc[:3], again.iloc[:3])


def test_symbols_do_not_share_statement_rows() -> None:
    left = _statement_frame()
    right = _statement_frame()
    right["symbol"] = "B"
    right["net_profit_parent_ttm"] = 1.0
    right["total_assets_prior_year"] = 300.0
    out = compute_factors(
        pd.concat([left, right], ignore_index=True),
        ["ep_ttm", "asset_growth_yoy"],
    )
    assert out.loc[out.symbol.eq("A"), "ep_ttm"].iloc[0] == pytest.approx(0.1)
    assert out.loc[out.symbol.eq("B"), "ep_ttm"].iloc[0] == pytest.approx(0.001)
    assert out.loc[out.symbol.eq("B"), "asset_growth_yoy"].iloc[0] == pytest.approx(0.0)
