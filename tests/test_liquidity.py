import pandas as pd

from quant_factors.core import compute_factors


def test_liquidity_factors() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="B"),
            "symbol": ["L"] * 40,
            "close": 10.0,
            "volume": range(100, 140),
        }
    )
    out = compute_factors(df, factors=["turnover_20d", "volume_surge_5d", "amihud_illiq_20d"])
    assert out["turnover_20d"].notna().sum() > 0
    assert out["volume_surge_5d"].notna().sum() > 0
