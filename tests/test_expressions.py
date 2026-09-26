import numpy as np
import pandas as pd
import pytest

from quant_factors.expressions import (
    ExpressionError,
    compute_research_factors,
    expression_requirements,
    validate_expressions,
)
from quant_factors.research import factor_report


def panel(periods: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date,
                "symbol": f"S{symbol}",
                "close": 10 + symbol + index * (0.05 + symbol * 0.01),
                "volume": 1000 + index * (symbol + 1),
                "industry": "A" if symbol < 3 else "B",
                "market_cap": 100 + symbol * 20 + index,
            }
            for symbol in range(1, 6)
            for index, date in enumerate(pd.bdate_range("2024-01-01", periods=periods))
        ]
    )


def test_validate_rejects_code_access_noncausal_windows_and_cycles() -> None:
    valid = validate_expressions(
        {
            "quality_momentum": "clip(momentum_20d / volatility_20d, -10, 10)",
            "stable_quality": "rolling_mean(quality_momentum, 5)",
        }
    )
    assert valid["quality_momentum"].startswith("clip(")
    invalid = [
        {"bad": "__import__('os').system('echo bad')"},
        {"bad": "close.__class__"},
        {"bad": "close[0]"},
        {"bad": "shift(close, -1)"},
        {"bad": "rolling_mean(close, 253)"},
        {"bad": "None"},
        {"left": "right + 1", "right": "left - 1"},
    ]
    for expressions in invalid:
        with pytest.raises(ExpressionError):
            validate_expressions(expressions)


def test_requirements_and_computation_resolve_transitive_dependencies() -> None:
    expressions = {
        "risk_adjusted": "momentum_5d / maximum(volatility_10d, 0.0001)",
        "smoothed": "rolling_mean(risk_adjusted, 5)",
    }
    requirements = expression_requirements(["smoothed"], expressions)
    assert requirements["smoothed"]["columns"] == ["close"]
    assert requirements["smoothed"]["warmup_bars"] == 15
    assert requirements["smoothed"]["pit_required"] is False
    result = compute_research_factors(panel(), ["smoothed"], expressions)
    assert "smoothed" in result
    assert "risk_adjusted" not in result
    assert "momentum_5d" not in result
    assert result.groupby("symbol")["smoothed"].apply(lambda values: values.notna().sum()).min() > 0


def test_expression_computation_is_causal_and_rank_is_cross_sectional() -> None:
    expressions = {
        "lagged_return": "pct_change(close, 5)",
        "ranked": "rank(lagged_return)",
    }
    frame = panel()
    first = compute_research_factors(frame, ["lagged_return", "ranked"], expressions)
    changed = frame.copy()
    boundary = pd.Timestamp("2024-03-01")
    changed.loc[changed.date >= boundary, "close"] *= 100
    second = compute_research_factors(changed, ["lagged_return", "ranked"], expressions)
    before = first.date < boundary
    pd.testing.assert_series_equal(
        first.loc[before, "lagged_return"], second.loc[before, "lagged_return"]
    )
    assert first.loc[first.date == first.date.max(), "ranked"].between(0, 1).all()


def test_mixed_expression_tracks_only_financial_pit_columns() -> None:
    expressions = {
        "value_momentum": "pe_inv + momentum_20d",
        "nested": "rolling_mean(value_momentum, 5) + pb_ratio + volume",
    }
    requirements = expression_requirements(["nested", "momentum_20d"], expressions)
    assert requirements["nested"]["columns"] == ["close", "pb_ratio", "pe_ratio", "volume"]
    assert requirements["nested"]["pit_columns"] == ["pb_ratio", "pe_ratio"]
    assert requirements["momentum_20d"]["pit_columns"] == []


def test_report_supports_custom_incremental_and_neutralized_comparisons() -> None:
    expressions = {"risk_adjusted": "momentum_5d / maximum(volatility_10d, 0.0001)"}
    report = factor_report(
        panel(),
        ["momentum_5d", "risk_adjusted"],
        expressions=expressions,
        cutoff="2024-05-01",
        horizons=(1,),
        baseline_names=("momentum_5d",),
        neutralize_by=("industry", "market_cap"),
    )
    assert report["requirements"]["risk_adjusted"]["expression"] == expressions["risk_adjusted"]
    assert report["incremental"][0]["factor"] == "risk_adjusted"
    assert report["incremental"][0]["baseline_factors"] == ["momentum_5d"]
    assert report["neutralization"][0]["applied_by"] == ["industry", "market_cap"]
    assert np.isfinite(report["coverage"][1]["coverage"])


def test_empty_expression_mapping_matches_builtin_values() -> None:
    frame = panel()
    result = compute_research_factors(frame, ["momentum_5d"], {})
    expected = frame.sort_values(["symbol", "date"]).copy()
    expected["momentum_5d"] = expected.groupby("symbol")["close"].pct_change(5, fill_method=None)
    np.testing.assert_allclose(result["momentum_5d"], expected["momentum_5d"], equal_nan=True)
    without_volume = frame.drop(columns="volume")
    optional = compute_research_factors(without_volume, ["turnover_20d", "pe_inv"], {})
    assert optional["turnover_20d"].isna().all()
    assert optional["pe_inv"].isna().all()
