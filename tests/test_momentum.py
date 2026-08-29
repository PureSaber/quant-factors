import pandas as pd

from quant_factors.core import compute_factors, momentum


def test_momentum_windows() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0])
    m5 = momentum(close, 2)
    assert m5.iloc[-1] == close.iloc[-1] / close.iloc[-3] - 1.0


def test_compute_momentum_family() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=80, freq="B").tolist() * 2,
            "symbol": ["A"] * 80 + ["B"] * 80,
            "close": list(range(80)) + list(range(100, 180)),
            "volume": 1000.0,
        }
    )
    out = compute_factors(
        df, factors=["momentum_5d", "momentum_10d", "momentum_60d", "log_momentum_20d"]
    )
    for col in ("momentum_5d", "momentum_10d", "momentum_60d", "log_momentum_20d"):
        assert col in out.columns
    assert out["momentum_5d"].notna().sum() > 0
