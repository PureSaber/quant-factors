"""Opt-in medium and low-frequency factors.

Horizons are 21 to 252 sessions, about one month through one year. These names
are not part of the default registry. A stock price is never used as a market
return, and a missing high, low, open, or volume column is an error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (21, 42, 63, 84, 105, 126, 168, 189, 210, 252)

_CLOSE = ("close",)
_OPEN = ("open", "close")
_RANGE = ("high", "low", "close")
_VOLUME = ("close", "volume")


class MediumLowError(ValueError):
    """A medium or low-frequency factor cannot be computed from the supplied panel."""


def _grouped(frame: pd.DataFrame, columns: tuple[str, ...], compute):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MediumLowError(f"Missing columns: {missing}")
    if frame.duplicated(["symbol", "date"]).any():
        raise MediumLowError("Medium and low-frequency factors require unique symbol/date rows")
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    ordered = frame.sort_values(["symbol", "date"], kind="stable")
    for _, group in ordered.groupby("symbol", sort=False):
        out.loc[group.index] = compute(group)
    return out.replace([np.inf, -np.inf], np.nan)


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def _trail_return(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        close = group["close"].replace(0, np.nan)
        return close / close.shift(horizon) - 1

    return _grouped(frame, _CLOSE, compute)


def _log_trail_return(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        close = group["close"].where(group["close"] > 0)
        return np.log(close / close.shift(horizon))

    return _grouped(frame, _CLOSE, compute)


def _ann_volatility(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        return _returns(group["close"]).rolling(horizon, min_periods=horizon).std(ddof=1) * np.sqrt(
            252
        )

    return _grouped(frame, _CLOSE, compute)


def _downside_volatility(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        returns = _returns(group["close"]).where(lambda series: series < 0, 0.0)
        return returns.rolling(horizon, min_periods=horizon).std(ddof=1) * np.sqrt(252)

    return _grouped(frame, _CLOSE, compute)


def _sharpe(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        returns = _returns(group["close"])
        mean = returns.rolling(horizon, min_periods=horizon).mean()
        scale = returns.rolling(horizon, min_periods=horizon).std(ddof=1).replace(0, np.nan)
        return mean / scale * np.sqrt(252)

    return _grouped(frame, _CLOSE, compute)


def _max_daily_return(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        return _returns(group["close"]).rolling(horizon, min_periods=horizon).max()

    return _grouped(frame, _CLOSE, compute)


def _min_daily_return(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        return _returns(group["close"]).rolling(horizon, min_periods=horizon).min()

    return _grouped(frame, _CLOSE, compute)


def _return_skew(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        return _returns(group["close"]).rolling(horizon, min_periods=horizon).skew()

    return _grouped(frame, _CLOSE, compute)


def _close_to_high(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        highest = group["close"].rolling(horizon, min_periods=horizon).max()
        return group["close"] / highest.replace(0, np.nan)

    return _grouped(frame, _CLOSE, compute)


def _close_to_low(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        lowest = group["close"].rolling(horizon, min_periods=horizon).min()
        return group["close"] / lowest.replace(0, np.nan)

    return _grouped(frame, _CLOSE, compute)


def _close_to_mean(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        mean = group["close"].rolling(horizon, min_periods=horizon).mean()
        return group["close"] / mean.replace(0, np.nan) - 1

    return _grouped(frame, _CLOSE, compute)


def _positive_day_share(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        returns = _returns(group["close"])
        flags = returns.gt(0).astype(float).where(returns.notna())
        return flags.rolling(horizon, min_periods=horizon).mean()

    return _grouped(frame, _CLOSE, compute)


def _rolling_numpy(values: pd.Series, horizon: int, function) -> pd.Series:
    array = values.to_numpy(dtype=float)
    out = np.full(len(array), np.nan)
    for end in range(horizon - 1, len(array)):
        out[end] = function(array[end - horizon + 1 : end + 1])
    return pd.Series(out, index=values.index)


def _up_down_vol_ratio(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def ratio(window: np.ndarray) -> float:
        if not np.isfinite(window).all():
            return np.nan
        up = window[window > 0]
        down = window[window < 0]
        if len(up) < 2 or len(down) < 2:
            return np.nan
        down_scale = float(down.std(ddof=1))
        if down_scale == 0:
            return np.nan
        return float(up.std(ddof=1) / down_scale)

    def compute(group: pd.DataFrame) -> pd.Series:
        return _rolling_numpy(_returns(group["close"]), horizon, ratio)

    return _grouped(frame, _CLOSE, compute)


def _amihud(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        traded = (group["close"] * group["volume"]).where(lambda value: value > 0)
        return (
            _returns(group["close"]).abs().div(traded).rolling(horizon, min_periods=horizon).mean()
        )

    return _grouped(frame, _VOLUME, compute)


def _return_autocorr(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        returns = _returns(group["close"])
        return returns.rolling(horizon, min_periods=horizon).corr(returns.shift(1))

    return _grouped(frame, _CLOSE, compute)


def _overnight_return_sum(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        previous = group["close"].shift(1).replace(0, np.nan)
        overnight = group["open"] / previous - 1
        return overnight.rolling(horizon, min_periods=horizon).sum()

    return _grouped(frame, _OPEN, compute)


def _intraday_return_sum(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        intraday = group["close"] / group["open"].replace(0, np.nan) - 1
        return intraday.rolling(horizon, min_periods=horizon).sum()

    return _grouped(frame, _OPEN, compute)


def _range_mean(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        width = (group["high"] - group["low"]) / group["close"].replace(0, np.nan)
        return width.rolling(horizon, min_periods=horizon).mean()

    return _grouped(frame, _RANGE, compute)


def _gain_loss_ratio(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def ratio(window: np.ndarray) -> float:
        if not np.isfinite(window).all():
            return np.nan
        loss = float(window[window < 0].sum())
        if loss == 0:
            return np.nan
        return float(window[window > 0].sum() / abs(loss))

    def compute(group: pd.DataFrame) -> pd.Series:
        return _rolling_numpy(_returns(group["close"]), horizon, ratio)

    return _grouped(frame, _CLOSE, compute)


def _trend_slope(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def slope(window: np.ndarray) -> float:
        if not np.isfinite(window).all() or (window <= 0).any():
            return np.nan
        log_close = np.log(window)
        time = np.arange(horizon, dtype=float)
        centered = time - time.mean()
        variance = float(np.dot(centered, centered))
        if variance == 0:
            return np.nan
        return float(np.dot(centered, log_close - log_close.mean()) / variance)

    def compute(group: pd.DataFrame) -> pd.Series:
        return _rolling_numpy(group["close"], horizon, slope)

    return _grouped(frame, _CLOSE, compute)


def _max_drawdown(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def drawdown(window: np.ndarray) -> float:
        if not np.isfinite(window).all() or (window <= 0).any():
            return np.nan
        peak = np.maximum.accumulate(window)
        return float(np.min(window / peak - 1))

    def compute(group: pd.DataFrame) -> pd.Series:
        return _rolling_numpy(group["close"], horizon, drawdown)

    return _grouped(frame, _CLOSE, compute)


def _volume_expansion(frame: pd.DataFrame, horizon: int) -> pd.Series:
    def compute(group: pd.DataFrame) -> pd.Series:
        recent = group["volume"].rolling(21, min_periods=21).mean()
        baseline = group["volume"].rolling(horizon, min_periods=horizon).mean()
        return recent / baseline.replace(0, np.nan) - 1

    return _grouped(frame, ("volume",), compute)


_COMPUTERS = {
    "trail_return": _trail_return,
    "log_trail_return": _log_trail_return,
    "ann_volatility": _ann_volatility,
    "downside_volatility": _downside_volatility,
    "sharpe": _sharpe,
    "max_daily_return": _max_daily_return,
    "min_daily_return": _min_daily_return,
    "return_skew": _return_skew,
    "close_to_high": _close_to_high,
    "close_to_low": _close_to_low,
    "close_to_mean": _close_to_mean,
    "positive_day_share": _positive_day_share,
    "up_down_vol_ratio": _up_down_vol_ratio,
    "amihud": _amihud,
    "return_autocorr": _return_autocorr,
    "overnight_return_sum": _overnight_return_sum,
    "intraday_return_sum": _intraday_return_sum,
    "range_mean": _range_mean,
    "gain_loss_ratio": _gain_loss_ratio,
    "trend_slope": _trend_slope,
    "max_drawdown": _max_drawdown,
    "volume_expansion": _volume_expansion,
}

_COLUMNS = {
    "trail_return": _CLOSE,
    "log_trail_return": _CLOSE,
    "ann_volatility": _CLOSE,
    "downside_volatility": _CLOSE,
    "sharpe": _CLOSE,
    "max_daily_return": _CLOSE,
    "min_daily_return": _CLOSE,
    "return_skew": _CLOSE,
    "close_to_high": _CLOSE,
    "close_to_low": _CLOSE,
    "close_to_mean": _CLOSE,
    "positive_day_share": _CLOSE,
    "up_down_vol_ratio": _CLOSE,
    "amihud": _VOLUME,
    "return_autocorr": _CLOSE,
    "overnight_return_sum": _OPEN,
    "intraday_return_sum": _OPEN,
    "range_mean": _RANGE,
    "gain_loss_ratio": _CLOSE,
    "trend_slope": _CLOSE,
    "max_drawdown": _CLOSE,
    "volume_expansion": ("volume",),
}

_DESCRIPTIONS = {
    "trail_return": "close-to-close return over {horizon} sessions",
    "log_trail_return": "log close-to-close return over {horizon} sessions",
    "ann_volatility": "annualized volatility of daily returns over {horizon} sessions",
    "downside_volatility": "annualized downside deviation over {horizon} sessions",
    "sharpe": "annualized mean daily return divided by volatility over {horizon} sessions",
    "max_daily_return": "largest daily return over {horizon} sessions",
    "min_daily_return": "smallest daily return over {horizon} sessions",
    "return_skew": "skewness of daily returns over {horizon} sessions",
    "close_to_high": "close divided by the trailing {horizon}-session high",
    "close_to_low": "close divided by the trailing {horizon}-session low",
    "close_to_mean": "close divided by its trailing {horizon}-session mean, minus one",
    "positive_day_share": "share of positive daily returns over {horizon} sessions",
    "up_down_vol_ratio": "upside volatility divided by downside volatility over {horizon} sessions",
    "amihud": "mean absolute return divided by close times volume over {horizon} sessions",
    "return_autocorr": "correlation of daily return with its one-session lag over {horizon} sessions",
    "overnight_return_sum": "sum of overnight open-to-previous-close returns over {horizon} sessions",
    "intraday_return_sum": "sum of open-to-close returns over {horizon} sessions",
    "range_mean": "mean of high-low range divided by close over {horizon} sessions",
    "gain_loss_ratio": "sum of gains divided by absolute sum of losses over {horizon} sessions",
    "trend_slope": "slope of log close on time over {horizon} sessions",
    "max_drawdown": "worst peak-to-trough decline inside {horizon} sessions",
    "volume_expansion": "21-session average volume divided by the {horizon}-session average, minus one",
}


def _build_catalog() -> dict[str, dict]:
    catalog = {}
    for kind, compute in _COMPUTERS.items():
        for horizon in HORIZONS:
            if kind == "volume_expansion" and horizon <= 21:
                continue
            if kind == "close_to_high" and horizon == 252:
                continue
            if kind == "close_to_low" and horizon == 252:
                continue
            name = f"{kind}_{horizon}d"
            catalog[name] = {
                "description": _DESCRIPTIONS[kind].format(horizon=horizon),
                "columns": list(_COLUMNS[kind]),
                "warmup_bars": horizon + 1,
                "pit_required": False,
                "pit_columns": [],
                "kind": kind,
                "horizon": horizon,
                "compute": compute,
            }
    return catalog


MEDIUM_LOW_FACTORS = _build_catalog()


def list_medium_low_factors() -> dict[str, str]:
    return {name: spec["description"] for name, spec in MEDIUM_LOW_FACTORS.items()}


def medium_low_requirements(name: str) -> dict:
    spec = MEDIUM_LOW_FACTORS[name]
    return {
        "columns": list(spec["columns"]),
        "warmup_bars": int(spec["warmup_bars"]),
        "pit_required": False,
        "pit_columns": [],
        "dependencies": [name],
    }


def compute_medium_low_factors(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Compute requested medium and low-frequency factors."""

    unknown = sorted(set(names) - set(MEDIUM_LOW_FACTORS))
    if unknown:
        raise MediumLowError(f"Unknown medium or low-frequency factors: {unknown}")
    computed = {
        name: MEDIUM_LOW_FACTORS[name]["compute"](frame, MEDIUM_LOW_FACTORS[name]["horizon"])
        for name in names
    }
    base = frame.drop(columns=[name for name in names if name in frame.columns], errors="ignore")
    return pd.concat([base, pd.DataFrame(computed, index=frame.index)], axis=1)
