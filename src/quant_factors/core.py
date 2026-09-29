from __future__ import annotations

import numpy as np
import pandas as pd

from quant_factors.academic import (
    ACADEMIC_REGISTRY,
    academic_is_fundamental,
    academic_scope,
    apply_cross_section,
    assert_academic_contracts,
    compute_symbol_academic,
)

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

# Preserve historical default experiments. New semantics require opt-in names.
DEFAULT_FACTORS = tuple(FACTOR_REGISTRY)
FACTOR_REGISTRY.update(
    {
        "average_volume_20d": "20-day average share volume (legacy turnover_20d alias)",
        "turnover_rate_20d_v2": "20-day mean daily raw shares / PIT raw free-float shares",
        "amihud_illiq_20d_v2": "20-day absolute economic return / actual traded amount",
    }
)
FACTOR_REGISTRY.update(ACADEMIC_REGISTRY)

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


def average_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window).mean()


def turnover(volume: pd.Series, window: int = 20) -> pd.Series:
    """Legacy average-volume alias; NOT a turnover rate. Kept for old experiments."""
    return average_volume(volume, window)


def turnover_rate(volume: pd.Series, free_float_shares: pd.Series, window=20) -> pd.Series:
    """Both series must use the same raw share unit and historical float vintage."""
    if not volume.index.equals(free_float_shares.index):
        raise ValueError("volume and free float indices must match")
    if (
        (volume.dropna() < 0).any()
        or (free_float_shares.dropna() <= 0).any()
        or not np.isfinite(volume.dropna()).all()
        or not np.isfinite(free_float_shares.dropna()).all()
    ):
        raise ValueError("finite nonnegative volume and positive float required")
    return (volume / free_float_shares).rolling(window).mean()


def amihud_amount(return_close: pd.Series, amount: pd.Series, window=20) -> pd.Series:
    """Amount is actual local-currency turnover, not adjusted close times raw volume."""
    if not return_close.index.equals(amount.index):
        raise ValueError("return prices and amount indices must match")
    if (
        (return_close.dropna() <= 0).any()
        or (amount.dropna() < 0).any()
        or not np.isfinite(return_close.dropna()).all()
        or not np.isfinite(amount.dropna()).all()
    ):
        raise ValueError("finite positive prices and nonnegative amount required")
    return (
        (return_close.pct_change(fill_method=None).abs() / amount.replace(0, np.nan))
        .rolling(window)
        .mean()
    )


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
    "momentum_5d": lambda c, v, h, low, pe, pb: momentum(c, 5),
    "momentum_10d": lambda c, v, h, low, pe, pb: momentum(c, 10),
    "momentum_20d": lambda c, v, h, low, pe, pb: momentum(c, 20),
    "momentum_60d": lambda c, v, h, low, pe, pb: momentum(c, 60),
    "log_momentum_20d": lambda c, v, h, low, pe, pb: log_momentum(c, 20),
    "reversal_5d": lambda c, v, h, low, pe, pb: reversal(c, 5),
    "reversal_10d": lambda c, v, h, low, pe, pb: reversal(c, 10),
    "volatility_10d": lambda c, v, h, low, pe, pb: volatility(c, 10),
    "volatility_20d": lambda c, v, h, low, pe, pb: volatility(c, 20),
    "volatility_60d": lambda c, v, h, low, pe, pb: volatility(c, 60),
    "downside_vol_20d": lambda c, v, h, low, pe, pb: downside_vol(c, 20),
    "mean_reversion_z_20d": lambda c, v, h, low, pe, pb: mean_reversion_z(c, 20),
    "volume_surge_5d": lambda c, v, h, low, pe, pb: volume_surge(v, 5, 20),
    "turnover_20d": lambda c, v, h, low, pe, pb: turnover(v, 20),
    "amihud_illiq_20d": lambda c, v, h, low, pe, pb: amihud_illiq(c, v, 20),
    "pe_inv": lambda c, v, h, low, pe, pb: (
        pe_inv(pe) if pe is not None else pd.Series(np.nan, index=c.index)
    ),
    "pb_inv": lambda c, v, h, low, pe, pb: (
        pb_inv(pb) if pb is not None else pd.Series(np.nan, index=c.index)
    ),
}


def compute_factors(df: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    """Compute selected factors on an OHLCV panel sorted by date per symbol."""
    _require_cols(df, ("date", "symbol", "close"))
    factors = factors or list(DEFAULT_FACTORS)
    unknown = set(factors) - set(FACTOR_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown factors: {sorted(unknown)}")
    if df.duplicated(["date", "symbol"]).any():
        raise ValueError("Factors require unique symbol/date rows")
    out = df.copy()
    assert_academic_contracts(out, [name for name in factors if name in ACADEMIC_REGISTRY])
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
            scope = academic_scope(name)
            if scope == "cross_section":
                continue
            if scope == "symbol":
                g[name] = compute_symbol_academic(name, g)
                continue
            if name == "average_volume_20d":
                g[name] = average_volume(volume)
                continue
            if name == "turnover_rate_20d_v2":
                _require_cols(g, ("free_float_shares", "volume_unit", "share_basis"))
                if not g.volume_unit.eq("shares").all() or not g.share_basis.eq("raw").all():
                    raise ValueError("turnover v2 requires explicit raw share units")
                g[name] = turnover_rate(volume, g.free_float_shares)
                continue
            if name == "amihud_illiq_20d_v2":
                _require_cols(g, ("return_close", "amount", "amount_unit", "currency"))
                if not g.amount_unit.eq("currency").all() or g.currency.nunique() != 1:
                    raise ValueError("Amihud v2 requires one currency and actual currency amounts")
                g[name] = amihud_amount(g.return_close, g.amount)
                continue
            if name in REQUIRES_FUNDAMENTAL:
                col = "pe_ratio" if name == "pe_inv" else "pb_ratio"
                if col not in g.columns:
                    g[name] = np.nan
                    continue
            g[name] = _FACTOR_COMPUTERS[name](close, volume, high, low, pe, pb)
        pieces.append(g)
    result = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else out.assign(**{name: np.nan for name in factors})
    )
    cross_section = [name for name in factors if academic_scope(name) == "cross_section"]
    if cross_section:
        result = apply_cross_section(result, cross_section)
    return result


def list_factors() -> dict[str, str]:
    return dict(FACTOR_REGISTRY)


def factor_requires_fundamental(name: str) -> bool:
    return name in REQUIRES_FUNDAMENTAL or academic_is_fundamental(name)
