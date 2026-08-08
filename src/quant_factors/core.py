from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_REGISTRY: dict[str, str] = {
    "momentum_5d": "5-day close return",
    "momentum_10d": "10-day close return",
    "momentum_20d": "20-day close return",
    "momentum_60d": "60-day close return",
    "log_momentum_20d": "Log price momentum 20d",
    "reversal_5d": "5-day short-term reversal (inverted return)",
    "reversal_10d": "10-day short-term reversal",
    "volatility_10d": "10-day annualized volatility",
    "volatility_20d": "20-day annualized volatility",
    "volatility_60d": "60-day annualized volatility",
    "downside_vol_20d": "20-day downside deviation annualized",
    "mean_reversion_z_20d": "Z-score of close vs 20-day mean",
    "volume_surge_5d": "5d vs 20d volume ratio",
    "turnover_20d": "20-day average volume",
    "amihud_illiq_20d": "Simplified Amihud illiquidity",
    "pe_inv": "Inverse P/E (requires pe_ratio)",
    "pb_inv": "Inverse P/B (requires pb_ratio)",
}

REQUIRES_FUNDAMENTAL = frozenset({"pe_inv", "pb_inv"})


def _require_cols(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def momentum(close: pd.Series, window: int = 20) -> pd.Series:
    return close / close.shift(window) - 1.0


def log_momentum(close: pd.Series, window: int = 20) -> pd.Series:
    return np.log(close / close.shift(window))


def reversal(close: pd.Series, window: int = 5) -> pd.Series:
    return -close.pct_change(window)


def volatility(close: pd.Series, window: int = 20) -> pd.Series:
    ret = close.pct_change()
    return ret.rolling(window).std() * np.sqrt(252)


def downside_vol(close: pd.Series, window: int = 20) -> pd.Series:
    ret = close.pct_change()
    down = ret.where(ret < 0, 0.0)
    return down.rolling(window).std() * np.sqrt(252)


def mean_reversion_z(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - ma) / sd.replace(0, np.nan)


def volume_surge(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    short_ma = volume.rolling(short).mean()
    long_ma = volume.rolling(long).mean()
    return short_ma / long_ma.replace(0, np.nan)


def turnover(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window).mean()


def amihud_illiq(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    ret = close.pct_change().abs()
    dollar_vol = (close * volume).replace(0, np.nan)
    daily = ret / dollar_vol
    return daily.rolling(window).mean()


def pe_inv(series: pd.Series) -> pd.Series:
    return 1.0 / series.replace(0, np.nan)


def pb_inv(series: pd.Series) -> pd.Series:
    return 1.0 / series.replace(0, np.nan)


_FACTOR_COMPUTERS: dict[str, callable] = {
    "momentum_5d": lambda c, v, h, l, pe, pb: momentum(c, 5),
    "momentum_10d": lambda c, v, h, l, pe, pb: momentum(c, 10),
    "momentum_20d": lambda c, v, h, l, pe, pb: momentum(c, 20),
    "momentum_60d": lambda c, v, h, l, pe, pb: momentum(c, 60),
    "log_momentum_20d": lambda c, v, h, l, pe, pb: log_momentum(c, 20),
    "reversal_5d": lambda c, v, h, l, pe, pb: reversal(c, 5),
    "reversal_10d": lambda c, v, h, l, pe, pb: reversal(c, 10),
    "volatility_10d": lambda c, v, h, l, pe, pb: volatility(c, 10),
    "volatility_20d": lambda c, v, h, l, pe, pb: volatility(c, 20),
    "volatility_60d": lambda c, v, h, l, pe, pb: volatility(c, 60),
    "downside_vol_20d": lambda c, v, h, l, pe, pb: downside_vol(c, 20),
    "mean_reversion_z_20d": lambda c, v, h, l, pe, pb: mean_reversion_z(c, 20),
    "volume_surge_5d": lambda c, v, h, l, pe, pb: volume_surge(v, 5, 20),
    "turnover_20d": lambda c, v, h, l, pe, pb: turnover(v, 20),
    "amihud_illiq_20d": lambda c, v, h, l, pe, pb: amihud_illiq(c, v, 20),
    "pe_inv": lambda c, v, h, l, pe, pb: pe_inv(pe) if pe is not None else pd.Series(np.nan, index=c.index),
    "pb_inv": lambda c, v, h, l, pe, pb: pb_inv(pb) if pb is not None else pd.Series(np.nan, index=c.index),
}


def compute_factors(df: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    """Compute selected factors on an OHLCV panel sorted by date per symbol."""
    _require_cols(df, ("date", "symbol", "close"))
    factors = factors or list(FACTOR_REGISTRY)
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = np.nan

    pieces: list[pd.DataFrame] = []
    for _symbol, grp in out.groupby("symbol", sort=False):
        g = grp.sort_values("date").copy()
        close = g["close"]
        volume = g["volume"]
        high = g["high"] if "high" in g.columns else None
        low = g["low"] if "low" in g.columns else None
        pe = g["pe_ratio"] if "pe_ratio" in g.columns else None
        pb = g["pb_ratio"] if "pb_ratio" in g.columns else None

        for name in factors:
            if name not in _FACTOR_COMPUTERS:
                continue
            if name in REQUIRES_FUNDAMENTAL:
                col = "pe_ratio" if name == "pe_inv" else "pb_ratio"
                if col not in g.columns:
                    g[name] = np.nan
                    continue
            g[name] = _FACTOR_COMPUTERS[name](close, volume, high, low, pe, pb)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def list_factors() -> dict[str, str]:
    return dict(FACTOR_REGISTRY)


def factor_requires_fundamental(name: str) -> bool:
    return name in REQUIRES_FUNDAMENTAL
