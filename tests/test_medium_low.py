"""Medium and low-frequency factors are opt-in and use month-to-year windows."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "src" / "quant_factors"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


medium = _load("medium_low_under_test", "medium_low.py")
core = _load("core_under_test_medium", "core.py")
research = _load("research_factors_under_test_medium", "research_factors.py")


def _panel(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = 100 * (1.002 ** np.arange(rows))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAA",
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1_000.0),
        }
    )


def test_catalog_has_at_least_200_medium_low_factors() -> None:
    names = list(medium.MEDIUM_LOW_FACTORS)
    assert len(names) >= 200
    assert len(names) == len(set(names))
    assert set(names).isdisjoint(core.FACTOR_REGISTRY)
    assert set(names).isdisjoint(research.RESEARCH_FACTORS)
    assert "trail_return_252d" in names
    assert "close_to_high_252d" not in names
    assert all(spec["horizon"] >= 21 for spec in medium.MEDIUM_LOW_FACTORS.values())


def test_one_month_return_and_no_lookahead() -> None:
    frame = _panel(30)
    result = medium.compute_medium_low_factors(frame, ["trail_return_21d"])
    close = frame["close"].to_numpy()
    assert result["trail_return_21d"].iloc[21] == pytest.approx(close[21] / close[0] - 1)
    changed = frame.copy()
    changed.loc[changed.index[-1], "close"] = 1.0
    again = medium.compute_medium_low_factors(changed, ["trail_return_21d"])
    assert again["trail_return_21d"].iloc[21] == pytest.approx(result["trail_return_21d"].iloc[21])


def test_every_medium_low_factor_runs_on_a_daily_panel() -> None:
    frame = _panel(80)
    names = list(medium.MEDIUM_LOW_FACTORS)
    result = medium.compute_medium_low_factors(frame, names)
    for name in names:
        values = result[name]
        assert len(values) == len(frame)
        assert not np.isinf(values.to_numpy(dtype=float)).any()
    with pytest.raises(medium.MediumLowError, match="volume"):
        medium.compute_medium_low_factors(frame.drop(columns=["volume"]), ["amihud_63d"])
