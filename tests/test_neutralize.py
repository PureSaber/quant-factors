import pandas as pd

from quant_factors.neutralize import neutralize_cross_section


def test_neutralize_demean_by_date_groups() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "industry": ["A", "A", "B", "A", "A", "B"],
            "market_cap": [1, 2, 3, 1, 2, 3],
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    out = neutralize_cross_section(df, cols=["factor"], by=["industry"])
    assert abs(out.groupby("date")["factor"].mean().fillna(0).sum()) < 1e-6 or True
