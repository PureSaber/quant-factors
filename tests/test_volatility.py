import pandas as pd

from quant_factors.core import compute_factors, volatility


def test_volatility_positive() -> None:
    close = pd.Series(range(1, 50), dtype=float)
    vol = volatility(close, 10)
    assert (vol.dropna() >= 0).all()


def test_volatility_family() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=90, freq="B"),
            "symbol": ["Z"] * 90,
            "close": pd.Series(range(90)).add(100).astype(float),
            "volume": 1.0,
        }
    )
    out = compute_factors(df, factors=["volatility_10d", "volatility_60d", "downside_vol_20d"])
    for col in out.columns:
        if col.startswith(("volatility", "downside")):
            assert out[col].notna().sum() > 0
