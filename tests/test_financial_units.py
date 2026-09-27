import numpy as np
import pandas as pd
import pytest
from quant_data_kit.financial.units import normalize_trading_units

from quant_factors.core import (
    amihud_amount,
    average_volume,
    compute_factors,
    turnover,
    turnover_rate,
)


def test_legacy_is_average_volume_and_split_invariant_true_turnover():
    volume = pd.Series([10.0] * 20 + [20.0] * 20)
    shares = pd.Series([100.0] * 20 + [200.0] * 20)
    pd.testing.assert_series_equal(turnover(volume), average_volume(volume))
    assert np.allclose(turnover_rate(volume, shares).dropna(), 0.1)
    assert average_volume(volume).iloc[-1] == 20


def test_amount_not_close_times_volume_and_no_fill_missing():
    close = pd.Series([100.0, 101.0, np.nan, 102.0, 103.0])
    amount = pd.Series([1000.0] * 5)
    result = amihud_amount(close, amount, 1)
    assert result.iloc[1] == pytest.approx(0.01 / 1000)
    assert pd.isna(result.iloc[3])
    with pytest.raises(ValueError):
        turnover_rate(pd.Series([1.0]), pd.Series([0.0]), 1)


def test_normalized_panel_opt_in_factor_names():
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=22),
            "symbol": "A",
            "close": 10.0,
            "volume": 2.0,
            "amount": 3.0,
            "free_float_shares": 1000.0,
        }
    )
    raw = normalize_trading_units(
        panel,
        volume_unit="lots",
        lot_size=100,
        amount_unit="thousand_currency",
        currency="CNY",
        share_basis="raw",
    )
    raw["return_close"] = 10.0
    result = compute_factors(raw, ["turnover_rate_20d_v2", "amihud_illiq_20d_v2"])
    assert result.turnover_rate_20d_v2.iloc[-1] == pytest.approx(0.2)
    assert result.amihud_illiq_20d_v2.iloc[-1] == 0
    with pytest.raises(ValueError):
        compute_factors(panel, ["turnover_rate_20d_v2"])
