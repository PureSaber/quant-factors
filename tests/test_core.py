import pandas as pd

from quant_factors.core import compute_factors, list_factors


def test_list_factors() -> None:
    factors = list_factors()
    assert "momentum_20d" in factors


def test_compute_factors_on_panel() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
            "symbol": ["AAA"] * 3 + ["BBB"] * 3,
            "close": [10.0, 10.5, 10.2, 20.0, 19.5, 20.5],
            "volume": [100, 110, 105, 200, 180, 220],
        }
    )
    out = compute_factors(df, factors=["momentum_20d", "volatility_20d"])
    assert "momentum_20d" in out.columns
    assert len(out) == 6
