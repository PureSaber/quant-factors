import numpy as np
import pandas as pd
import pytest

from quant_factors.core import DEFAULT_FACTORS, FACTOR_REGISTRY, compute_factors, downside_deviation


def test_zero_target_downside_deviation_has_independent_hand_calculation():
    close = pd.Series(100 * 0.99 ** np.arange(21))
    assert downside_deviation(close).iloc[-1] == pytest.approx(0.01 * np.sqrt(252))
    assert downside_deviation(close).iloc[:20].isna().all()
    mixed = pd.Series([100, 90, 99, 89.1])  # -10%, +10%, -10%
    assert downside_deviation(mixed, 3).iloc[-1] == pytest.approx(np.sqrt(0.02 / 3 * 252))


def test_versioned_factor_preserves_gaps_and_identifies_legacy_definition():
    close = pd.Series([100, 101, np.nan, 103, 104, 105])
    assert downside_deviation(close, 2).iloc[1:5].isna().all()
    assert downside_deviation(close, 2).iloc[-1] == 0
    assert "Legacy" in FACTOR_REGISTRY["downside_vol_20d"]
    name = "downside_deviation_20d_v2"
    assert name not in DEFAULT_FACTORS
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=21),
            "symbol": "A",
            "close": 100 * 0.99 ** np.arange(21),
            "volume": 1,
        }
    )
    assert compute_factors(frame, factors=[name])[name].iloc[-1] == pytest.approx(
        0.01 * np.sqrt(252)
    )
