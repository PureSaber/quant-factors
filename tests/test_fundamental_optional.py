import pandas as pd

from quant_factors.core import compute_factors, factor_requires_fundamental


def test_fundamental_optional_skip() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "symbol": ["F"] * 5,
            "close": 10.0,
            "volume": 1.0,
        }
    )
    out = compute_factors(df, factors=["pe_inv", "pb_inv"])
    assert out["pe_inv"].isna().all()
    assert factor_requires_fundamental("pe_inv")


def test_fundamental_when_present() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "symbol": ["F"] * 5,
            "close": 10.0,
            "volume": 1.0,
            "pe_ratio": [10.0, 20.0, 0.0, 5.0, 8.0],
            "pb_ratio": [1.0, 2.0, 4.0, 2.0, 1.0],
        }
    )
    out = compute_factors(df, factors=["pe_inv", "pb_inv"])
    assert out["pe_inv"].iloc[0] == 0.1
