import pandas as pd

from quant_factors.core import compute_factors, reversal


def test_reversal_sign() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 10.0])
    rev = reversal(close, 1)
    # reversal = -pct_change; drop from 12->10 gives positive reversal factor
    assert rev.iloc[-1] > 0


def test_reversal_factors_in_panel() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="B"),
            "symbol": ["X"] * 30,
            "close": [10 + ((-1) ** i) * 0.5 for i in range(30)],
            "volume": 1.0,
        }
    )
    out = compute_factors(df, factors=["reversal_5d", "reversal_10d"])
    assert "reversal_5d" in out.columns
    assert out["reversal_5d"].notna().any()
