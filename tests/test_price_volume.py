"""Opt-in price-volume formulas stay out of the default factor registry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1] / "src" / "quant_factors"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


price_volume = _load("price_volume_under_test", "price_volume.py")
catalog = _load("price_volume_catalog_under_test", "price_volume_catalog.py")


def _panel(rows: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    frames = []
    for symbol, shift in (("AAA", 0.0), ("BBB", 2.0)):
        close = np.linspace(10, 20, rows) + shift + np.sin(np.arange(rows))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": np.linspace(100, 200, rows) + shift,
                    "amount": close * np.linspace(100, 200, rows),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_catalog_uses_characteristic_names() -> None:
    names = list(catalog.PRICE_VOLUME_CATALOG)
    assert len(names) == len(set(names))
    assert len(names) > 100
    assert "log_volume_change_intraday_return_rank_corr" in names
    assert "overnight_return" in names
    assert all(not name.startswith(("gtja", "alpha")) for name in names)
    assert catalog.SKIPPED_SIMILAR_SOURCE_IDS


def test_close_change_does_not_read_the_future() -> None:
    frame = _panel(12)
    values = price_volume.evaluate_formula("CLOSE-DELAY(CLOSE,1)", frame)
    aaa = frame.loc[frame.symbol.eq("AAA"), "close"].to_numpy()
    got = values.iloc[:12].to_numpy()
    assert np.isnan(got[0])
    assert np.allclose(got[1:], np.diff(aaa))

    changed = frame.copy()
    changed.loc[changed.index[-1], "close"] = 999
    again = price_volume.evaluate_formula("CLOSE-DELAY(CLOSE,1)", changed)
    assert np.allclose(values.iloc[:-2], again.iloc[:-2], equal_nan=True)


def test_representative_factors_and_full_catalog() -> None:
    frame = _panel(40)
    for name in ("close_location_change", "overnight_return", "signed_volume_sum"):
        values = price_volume.evaluate_formula(catalog.PRICE_VOLUME_CATALOG[name]["formula"], frame)
        assert len(values) == len(frame)
        assert np.isfinite(values.dropna()).all()

    for name, entry in catalog.PRICE_VOLUME_CATALOG.items():
        values = price_volume.evaluate_formula(entry["formula"], frame)
        assert len(values) == len(frame), name
        assert not np.isinf(pd.to_numeric(values, errors="coerce").to_numpy()).any(), name
