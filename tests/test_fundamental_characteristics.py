"""Fundamental characteristics are formulas over caller-supplied columns."""

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


fundamental = _load("fundamental_characteristics_under_test", "fundamental_characteristics.py")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "symbol": "AAA",
            "total_assets": [100.0, 100.0, 100.0, 100.0],
            "revenue": [10.0, 10.0, 12.0, 15.0],
            "cogs": [4.0, 4.0, 4.0, 5.0],
            "net_income": [2.0, 2.0, 3.0, 3.0],
            "operating_cash_flow": [1.0, 1.0, 2.0, 2.0],
            "book_equity": [40.0, 40.0, 40.0, 50.0],
            "market_equity": [80.0, 80.0, 80.0, 90.0],
        }
    )


def test_catalog_covers_the_public_accounting_families() -> None:
    names = fundamental.FUNDAMENTAL_CHARACTERISTICS
    assert len(names) >= 200
    assert "revenue_growth_1y" in names
    assert "gross_profit_change_to_assets_1y" in names
    assert "book_to_market_equity" in names
    assert "cash_accruals_to_assets" in names
    inputs = fundamental.FUNDAMENTAL_INPUTS
    assert "total_assets" in inputs
    assert "market_equity" in inputs


def test_growth_change_and_ratio_use_only_supplied_columns() -> None:
    frame = _frame()
    result = fundamental.compute_fundamental_characteristics(
        frame,
        [
            "revenue_growth_1y",
            "gross_profit_change_to_assets_1y",
            "gross_margin_ratio",
            "cash_accruals_to_assets",
            "book_to_market_equity",
        ],
        year_lag=2,
        three_year_lag=3,
        five_year_lag=3,
    )
    assert result["revenue_growth_1y"].iloc[2] == pytest.approx(0.2)
    assert result["gross_profit_change_to_assets_1y"].iloc[2] == pytest.approx((8 - 6) / 100)
    assert result["gross_margin_ratio"].iloc[2] == pytest.approx(8 / 12)
    assert result["cash_accruals_to_assets"].iloc[2] == pytest.approx((3 - 2) / 100)
    assert result["book_to_market_equity"].iloc[3] == pytest.approx(50 / 90)
    assert np.isnan(result["revenue_growth_1y"].iloc[1])


def test_every_characteristic_dispatches_when_inputs_are_present() -> None:
    rows = 6
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "symbol": "AAA",
        }
    )
    for column in fundamental.FUNDAMENTAL_INPUTS:
        frame[column] = np.linspace(10, 20, rows)
    result = fundamental.compute_fundamental_characteristics(
        frame,
        list(fundamental.FUNDAMENTAL_CHARACTERISTICS),
        year_lag=2,
        three_year_lag=4,
        five_year_lag=5,
    )
    for name in fundamental.FUNDAMENTAL_CHARACTERISTICS:
        values = pd.to_numeric(result[name], errors="coerce")
        assert len(values) == rows
        assert not np.isinf(values.to_numpy()).any()


def test_nonpositive_denominator_stays_missing() -> None:
    frame = _frame()
    frame["total_assets"] = 0.0
    result = fundamental.compute_fundamental_characteristics(
        frame,
        ["cash_accruals_to_assets"],
        year_lag=2,
    )
    assert result["cash_accruals_to_assets"].isna().all()
    with pytest.raises(fundamental.FundamentalError, match="revenue"):
        fundamental.compute_fundamental_characteristics(
            frame.drop(columns=["revenue"]), ["revenue_growth_1y"]
        )
