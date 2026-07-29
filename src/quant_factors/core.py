from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_REGISTRY: dict[str, str] = {
    "momentum_20d": "20-day close return",
    "volatility_20d": "20-day annualized return volatility",
    "mean_reversion_z_20d": "Z-score of close vs 20-day mean",
    "volume_surge_5d": "5-day volume vs 20-day average ratio",
}


def _require_cols(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def momentum(close: pd.Series, window: int = 20) -> pd.Series:
    return close / close.shift(window) - 1.0


def volatility(close: pd.Series, window: int = 20) -> pd.Series:
    ret = close.pct_change()
    return ret.rolling(window).std() * np.sqrt(252)


def mean_reversion_z(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - ma) / sd.replace(0, np.nan)


def volume_surge(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    short_ma = volume.rolling(short).mean()
    long_ma = volume.rolling(long).mean()
    return short_ma / long_ma.replace(0, np.nan)


def compute_factors(df: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    """Compute selected factors on an OHLCV panel sorted by date per symbol."""
    _require_cols(df, ("date", "symbol", "close"))
    factors = factors or list(FACTOR_REGISTRY)
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = np.nan

    pieces: list[pd.DataFrame] = []
    for symbol, grp in out.groupby("symbol", sort=False):
        g = grp.sort_values("date").copy()
        if "momentum_20d" in factors:
            g["momentum_20d"] = momentum(g["close"])
        if "volatility_20d" in factors:
            g["volatility_20d"] = volatility(g["close"])
        if "mean_reversion_z_20d" in factors:
            g["mean_reversion_z_20d"] = mean_reversion_z(g["close"])
        if "volume_surge_5d" in factors:
            g["volume_surge_5d"] = volume_surge(g["volume"])
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def list_factors() -> dict[str, str]:
    return dict(FACTOR_REGISTRY)
