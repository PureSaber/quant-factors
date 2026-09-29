"""Opt-in factors from later research, outside the default registry.

Daily factors run on the shared symbol/date panel. Columns that the panel does
not have yet must be supplied by the caller. A stock's own price is never used
as a stand-in for the market, and minute statistics are not invented from daily
bars. Intraday factors are aggregated with ``compute_minute_factors``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ResearchFactorError(ValueError):
    """A research factor was requested without the columns it measures."""


def _unique_daily(frame: pd.DataFrame) -> None:
    if frame.duplicated(["symbol", "date"]).any():
        raise ResearchFactorError("Research factors require unique symbol/date rows")


def _per_symbol(frame: pd.DataFrame, compute) -> pd.Series:
    _unique_daily(frame)
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    ordered = frame.sort_values(["symbol", "date"], kind="stable")
    for _, group in ordered.groupby("symbol", sort=False):
        out.loc[group.index] = compute(group)
    return out


def _window_sums(amplitude: np.ndarray, returns: np.ndarray, window: int, half: int):
    high_sum = np.full(len(amplitude), np.nan)
    low_sum = np.full(len(amplitude), np.nan)
    for end in range(window - 1, len(amplitude)):
        start = end - window + 1
        amp = amplitude[start : end + 1]
        ret = returns[start : end + 1]
        if not np.isfinite(amp).all() or not np.isfinite(ret).all():
            continue
        order = np.argsort(-amp, kind="mergesort")
        high_sum[end] = float(ret[order[:half]].sum())
        low_sum[end] = float(ret[order[-half:]].sum())
    return high_sum, low_sum


def _amplitude_returns(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    previous = group["close"].shift(1).replace(0, np.nan)
    amplitude = ((group["high"] - group["low"]) / previous).to_numpy(dtype=float)
    returns = group["close"].pct_change().to_numpy(dtype=float)
    return _window_sums(amplitude, returns, 20, 10)


def _high_amplitude_return(frame: pd.DataFrame) -> pd.Series:
    return _per_symbol(frame, lambda group: _amplitude_returns(group)[0])


def _low_amplitude_return(frame: pd.DataFrame) -> pd.Series:
    return _per_symbol(frame, lambda group: _amplitude_returns(group)[1])


def _amplitude_spread(frame: pd.DataFrame) -> pd.Series:
    def spread(group: pd.DataFrame) -> np.ndarray:
        high, low = _amplitude_returns(group)
        return high - low

    return _per_symbol(frame, spread)


def _close_to_year_high(frame: pd.DataFrame) -> pd.Series:
    def ratio(group: pd.DataFrame) -> pd.Series:
        highest = group["close"].rolling(252, min_periods=252).max()
        return group["close"] / highest.replace(0, np.nan)

    return _per_symbol(frame, ratio)


def _max_return(frame: pd.DataFrame) -> pd.Series:
    def maximum(group: pd.DataFrame) -> pd.Series:
        return group["close"].pct_change().rolling(21, min_periods=21).max()

    return _per_symbol(frame, maximum)


def _max5_return(frame: pd.DataFrame) -> pd.Series:
    def average_top(window: np.ndarray) -> float:
        if not np.isfinite(window).all():
            return np.nan
        return float(np.partition(window, -5)[-5:].mean())

    def maximum(group: pd.DataFrame) -> pd.Series:
        returns = group["close"].pct_change()
        return returns.rolling(21, min_periods=21).apply(average_top, raw=True)

    return _per_symbol(frame, maximum)


def _daily_turnover(group: pd.DataFrame) -> pd.Series:
    required = ("volume", "free_float_shares", "volume_unit", "share_basis")
    missing = [column for column in required if column not in group.columns]
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")
    if not group["volume_unit"].eq("shares").all() or not group["share_basis"].eq("raw").all():
        raise ResearchFactorError("Turnover requires raw share volume and raw free float")
    shares = group["free_float_shares"]
    if shares.isna().any() or (shares <= 0).any() or (group["volume"] < 0).any():
        raise ResearchFactorError("Turnover requires nonnegative volume and positive free float")
    return group["volume"] / shares


def _turnover_change(frame: pd.DataFrame) -> pd.Series:
    def change(group: pd.DataFrame) -> pd.Series:
        turnover = _daily_turnover(group)
        recent = turnover.rolling(20, min_periods=20).mean()
        baseline = turnover.rolling(60, min_periods=60).mean()
        return recent / baseline.replace(0, np.nan) - 1

    return _per_symbol(frame, change)


def _turnover_vol(frame: pd.DataFrame) -> pd.Series:
    def volatility(group: pd.DataFrame) -> pd.Series:
        return _daily_turnover(group).rolling(20, min_periods=20).std(ddof=1)

    return _per_symbol(frame, volatility)


def _market_residuals(group: pd.DataFrame, window: int = 21):
    if "market_return" not in group.columns:
        raise ResearchFactorError("Missing columns: ['market_return']")
    returns = group["close"].pct_change().to_numpy(dtype=float)
    market = group["market_return"].to_numpy(dtype=float)
    ivol = np.full(len(group), np.nan)
    specificity = np.full(len(group), np.nan)
    residual_sum = np.full(len(group), np.nan)
    for end in range(window - 1, len(group)):
        start = end - window + 1
        yy = returns[start : end + 1]
        xx = market[start : end + 1]
        if not np.isfinite(yy).all() or not np.isfinite(xx).all():
            continue
        centered_x = xx - xx.mean()
        variance = float(np.dot(centered_x, centered_x))
        if variance == 0:
            continue
        centered_y = yy - yy.mean()
        beta = float(np.dot(centered_x, centered_y) / variance)
        alpha = float(yy.mean() - beta * xx.mean())
        residual = yy - (alpha + beta * xx)
        ss_res = float(np.dot(residual, residual))
        ss_tot = float(np.dot(centered_y, centered_y))
        ivol[end] = np.sqrt(ss_res / (window - 2))
        specificity[end] = np.nan if ss_tot == 0 else ss_res / ss_tot
        residual_sum[end] = float(residual.sum())
    return ivol, specificity, residual_sum


def _idio_vol(frame: pd.DataFrame) -> pd.Series:
    return _per_symbol(frame, lambda group: _market_residuals(group)[0])


def _specificity(frame: pd.DataFrame) -> pd.Series:
    return _per_symbol(frame, lambda group: _market_residuals(group)[1])


def _residual_return(frame: pd.DataFrame) -> pd.Series:
    return _per_symbol(frame, lambda group: _market_residuals(group)[2])


def _overnight_afternoon(frame: pd.DataFrame) -> pd.Series:
    if "afternoon_open" not in frame.columns:
        raise ResearchFactorError("Missing columns: ['afternoon_open']")

    def spread(group: pd.DataFrame) -> pd.Series:
        previous_close = group["close"].shift(1).replace(0, np.nan)
        afternoon_open = group["afternoon_open"].replace(0, np.nan)
        overnight = group["open"] / previous_close - 1
        afternoon = group["close"] / afternoon_open - 1
        return (
            overnight.rolling(20, min_periods=20).sum()
            - afternoon.rolling(20, min_periods=20).sum()
        )

    return _per_symbol(frame, spread)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def _accruals(frame: pd.DataFrame) -> pd.Series:
    required = ("net_income", "operating_cash_flow", "average_assets")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")
    return _ratio(
        frame["net_income"] - frame["operating_cash_flow"],
        frame["average_assets"],
    )


def _operating_profitability(frame: pd.DataFrame) -> pd.Series:
    required = ("revenue", "cogs", "sga", "interest_expense", "book_equity")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")
    profit = frame["revenue"] - frame["cogs"] - frame["sga"] - frame["interest_expense"]
    return _ratio(profit, frame["book_equity"])


def _carried_growth(series: pd.Series, *, signed: bool) -> pd.Series:
    changed = series.ne(series.shift(1)) & series.notna()
    previous = series.where(changed).ffill().shift(1)
    if signed:
        base = previous.abs().where(previous.ne(0))
        jump = ((series - previous) / base).where(changed)
    else:
        jump = (series / previous.where(previous > 0) - 1).where(changed)
    return jump.ffill()


def _asset_growth(frame: pd.DataFrame) -> pd.Series:
    if "total_assets" not in frame.columns:
        raise ResearchFactorError("Missing columns: ['total_assets']")

    def growth(group: pd.DataFrame) -> pd.Series:
        return _carried_growth(group["total_assets"], signed=False)

    return _per_symbol(frame, growth)


def _lagged_price_return(frame: pd.DataFrame, *, span: int, skip: int) -> pd.Series:
    def lagged(group: pd.DataFrame) -> pd.Series:
        close = group["close"].replace(0, np.nan)
        return close.shift(skip) / close.shift(span) - 1

    return _per_symbol(frame, lagged)


def _annualized_vol(frame: pd.DataFrame, window: int, *, downside: bool) -> pd.Series:
    def volatility(group: pd.DataFrame) -> pd.Series:
        returns = group["close"].pct_change()
        if downside:
            returns = returns.where(returns < 0, 0.0)
        return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)

    return _per_symbol(frame, volatility)


def _close_to_year_low(frame: pd.DataFrame) -> pd.Series:
    def ratio(group: pd.DataFrame) -> pd.Series:
        lowest = group["close"].rolling(252, min_periods=252).min()
        return group["close"] / lowest.replace(0, np.nan)

    return _per_symbol(frame, ratio)


def _max_return_60(frame: pd.DataFrame) -> pd.Series:
    def maximum(group: pd.DataFrame) -> pd.Series:
        return group["close"].pct_change().rolling(60, min_periods=60).max()

    return _per_symbol(frame, maximum)


def _amihud_120(frame: pd.DataFrame) -> pd.Series:
    def illiquidity(group: pd.DataFrame) -> pd.Series:
        traded = (group["close"] * group["volume"]).where(lambda value: value > 0)
        return group["close"].pct_change().abs().div(traded).rolling(120, min_periods=120).mean()

    return _per_symbol(frame, illiquidity)


def _same_month_return(frame: pd.DataFrame) -> pd.Series:
    def seasonal(group: pd.DataFrame) -> pd.Series:
        close = group["close"].replace(0, np.nan)
        lags = []
        for year in range(1, 6):
            end = close.shift(252 * year)
            start = close.shift(252 * year + 21)
            lags.append(end / start - 1)
        stacked = pd.concat(lags, axis=1)
        return stacked.mean(axis=1, skipna=True).where(stacked.notna().any(axis=1))

    return _per_symbol(frame, seasonal)


def _residual_momentum_12_1(frame: pd.DataFrame) -> pd.Series:
    def momentum(group: pd.DataFrame) -> np.ndarray:
        if "market_return" not in group.columns:
            raise ResearchFactorError("Missing columns: ['market_return']")
        returns = group["close"].pct_change().to_numpy(dtype=float)
        market = group["market_return"].to_numpy(dtype=float)
        values = np.full(len(group), np.nan)
        for end in range(252, len(group)):
            start = end - 252
            stop = end - 21
            yy = returns[start : stop + 1]
            xx = market[start : stop + 1]
            if not np.isfinite(yy).all() or not np.isfinite(xx).all():
                continue
            centered_x = xx - xx.mean()
            variance = float(np.dot(centered_x, centered_x))
            if variance == 0:
                continue
            beta = float(np.dot(centered_x, yy - yy.mean()) / variance)
            alpha = float(yy.mean() - beta * xx.mean())
            values[end] = float((yy - (alpha + beta * xx)).sum())
        return values

    return _per_symbol(frame, momentum)


def _turnover_mean(frame: pd.DataFrame, window: int) -> pd.Series:
    def average(group: pd.DataFrame) -> pd.Series:
        return _daily_turnover(group).rolling(window, min_periods=window).mean()

    return _per_symbol(frame, average)


def _sales_growth(frame: pd.DataFrame) -> pd.Series:
    if "revenue" not in frame.columns:
        raise ResearchFactorError("Missing columns: ['revenue']")

    def growth(group: pd.DataFrame) -> pd.Series:
        return _carried_growth(group["revenue"], signed=False)

    return _per_symbol(frame, growth)


def _earnings_growth(frame: pd.DataFrame) -> pd.Series:
    if "net_income" not in frame.columns:
        raise ResearchFactorError("Missing columns: ['net_income']")

    def growth(group: pd.DataFrame) -> pd.Series:
        return _carried_growth(group["net_income"], signed=True)

    return _per_symbol(frame, growth)


def _asset_turnover(frame: pd.DataFrame) -> pd.Series:
    required = ("revenue", "total_assets")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")
    return _ratio(frame["revenue"], frame["total_assets"])


def _financial_leverage(frame: pd.DataFrame) -> pd.Series:
    required = ("total_assets", "book_equity")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")
    return _ratio(frame["total_assets"], frame["book_equity"])


def _gross_margin(frame: pd.DataFrame) -> pd.Series:
    required = ("revenue", "cogs")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")
    return _ratio(frame["revenue"] - frame["cogs"], frame["revenue"])


def _roe(frame: pd.DataFrame) -> pd.Series:
    required = ("net_income", "book_equity")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")
    return _ratio(frame["net_income"], frame["book_equity"])


def _consensus_revision(frame: pd.DataFrame) -> pd.Series:
    required = ("consensus_eps", "close")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")

    def revision(group: pd.DataFrame) -> pd.Series:
        lagged = group["consensus_eps"].shift(60)
        return (group["consensus_eps"] - lagged) / group["close"].replace(0, np.nan)

    return _per_symbol(frame, revision)


def _earnings_surprise(frame: pd.DataFrame) -> pd.Series:
    required = ("reported_eps", "consensus_eps", "close")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")
    surprise = frame["reported_eps"] - frame["consensus_eps"]
    return surprise / frame["close"].replace(0, np.nan)


def _northbound_change(frame: pd.DataFrame) -> pd.Series:
    if "northbound_hold_ratio" not in frame.columns:
        raise ResearchFactorError("Missing columns: ['northbound_hold_ratio']")

    def change(group: pd.DataFrame) -> pd.Series:
        return group["northbound_hold_ratio"].pct_change(5, fill_method=None)

    return _per_symbol(frame, change)


_COMPUTERS = {
    "high_amplitude_return_20d": _high_amplitude_return,
    "low_amplitude_return_20d": _low_amplitude_return,
    "amplitude_return_spread_20d": _amplitude_spread,
    "close_to_year_high": _close_to_year_high,
    "max_return_21d": _max_return,
    "max5_return_21d": _max5_return,
    "turnover_rate_change_20_60d": _turnover_change,
    "turnover_rate_vol_20d": _turnover_vol,
    "idio_vol_21d": _idio_vol,
    "return_specificity_21d": _specificity,
    "residual_return_21d": _residual_return,
    "overnight_afternoon_spread_20d": _overnight_afternoon,
    "accruals_to_assets": _accruals,
    "operating_profitability": _operating_profitability,
    "asset_growth": _asset_growth,
    "momentum_6_1": lambda frame: _lagged_price_return(frame, span=126, skip=21),
    "momentum_12_1": lambda frame: _lagged_price_return(frame, span=252, skip=21),
    "volatility_120d": lambda frame: _annualized_vol(frame, 120, downside=False),
    "volatility_252d": lambda frame: _annualized_vol(frame, 252, downside=False),
    "downside_vol_120d": lambda frame: _annualized_vol(frame, 120, downside=True),
    "close_to_year_low": _close_to_year_low,
    "max_return_60d": _max_return_60,
    "amihud_illiq_120d": _amihud_120,
    "same_month_return": _same_month_return,
    "residual_momentum_12_1": _residual_momentum_12_1,
    "turnover_rate_120d": lambda frame: _turnover_mean(frame, 120),
    "turnover_rate_252d": lambda frame: _turnover_mean(frame, 252),
    "sales_growth": _sales_growth,
    "earnings_growth": _earnings_growth,
    "asset_turnover": _asset_turnover,
    "financial_leverage": _financial_leverage,
    "gross_margin": _gross_margin,
    "roe": _roe,
    "consensus_revision_60d": _consensus_revision,
    "earnings_surprise": _earnings_surprise,
    "northbound_hold_change_5d": _northbound_change,
}

RESEARCH_FACTORS: dict[str, dict] = {
    "high_amplitude_return_20d": {
        "description": "sum of returns on the 10 highest-amplitude days in 20 sessions",
        "columns": ["high", "low", "close"],
        "warmup_bars": 20,
        "pit_required": False,
        "pit_columns": [],
    },
    "low_amplitude_return_20d": {
        "description": "sum of returns on the 10 lowest-amplitude days in 20 sessions",
        "columns": ["high", "low", "close"],
        "warmup_bars": 20,
        "pit_required": False,
        "pit_columns": [],
    },
    "amplitude_return_spread_20d": {
        "description": "high-amplitude day return sum minus low-amplitude day return sum",
        "columns": ["high", "low", "close"],
        "warmup_bars": 20,
        "pit_required": False,
        "pit_columns": [],
    },
    "close_to_year_high": {
        "description": "close divided by the trailing 252-session high",
        "columns": ["close"],
        "warmup_bars": 252,
        "pit_required": False,
        "pit_columns": [],
    },
    "max_return_21d": {
        "description": "largest daily close-to-close return in 21 sessions",
        "columns": ["close"],
        "warmup_bars": 22,
        "pit_required": False,
        "pit_columns": [],
    },
    "max5_return_21d": {
        "description": "average of the five largest daily returns in 21 sessions",
        "columns": ["close"],
        "warmup_bars": 22,
        "pit_required": False,
        "pit_columns": [],
    },
    "turnover_rate_change_20_60d": {
        "description": "20-session mean turnover divided by 60-session mean turnover, minus one",
        "columns": ["volume", "free_float_shares", "volume_unit", "share_basis"],
        "warmup_bars": 60,
        "pit_required": True,
        "pit_columns": ["free_float_shares"],
    },
    "turnover_rate_vol_20d": {
        "description": "standard deviation of daily raw-share turnover over 20 sessions",
        "columns": ["volume", "free_float_shares", "volume_unit", "share_basis"],
        "warmup_bars": 20,
        "pit_required": True,
        "pit_columns": ["free_float_shares"],
    },
    "idio_vol_21d": {
        "description": "residual volatility from a 21-session regression on supplied market_return",
        "columns": ["close", "market_return"],
        "warmup_bars": 21,
        "pit_required": True,
        "pit_columns": ["market_return"],
    },
    "return_specificity_21d": {
        "description": "share of return variance left unexplained by supplied market_return",
        "columns": ["close", "market_return"],
        "warmup_bars": 21,
        "pit_required": True,
        "pit_columns": ["market_return"],
    },
    "residual_return_21d": {
        "description": "sum of residuals from the 21-session market regression",
        "columns": ["close", "market_return"],
        "warmup_bars": 21,
        "pit_required": True,
        "pit_columns": ["market_return"],
    },
    "overnight_afternoon_spread_20d": {
        "description": "20-session sum of overnight return minus afternoon return",
        "columns": ["open", "close", "afternoon_open"],
        "warmup_bars": 20,
        "pit_required": True,
        "pit_columns": ["afternoon_open"],
    },
    "accruals_to_assets": {
        "description": "(net income minus operating cash flow) divided by average assets",
        "columns": ["net_income", "operating_cash_flow", "average_assets"],
        "warmup_bars": 1,
        "pit_required": True,
        "pit_columns": ["net_income", "operating_cash_flow", "average_assets"],
    },
    "operating_profitability": {
        "description": "operating profit after interest divided by book equity",
        "columns": ["revenue", "cogs", "sga", "interest_expense", "book_equity"],
        "warmup_bars": 1,
        "pit_required": True,
        "pit_columns": ["revenue", "cogs", "sga", "interest_expense", "book_equity"],
    },
    "asset_growth": {
        "description": "growth between the two latest distinct total-asset observations",
        "columns": ["total_assets"],
        "warmup_bars": 2,
        "pit_required": True,
        "pit_columns": ["total_assets"],
    },
    "momentum_6_1": {
        "description": "close return from 126 sessions ago to 21 sessions ago",
        "columns": ["close"],
        "warmup_bars": 126,
        "pit_required": False,
        "pit_columns": [],
    },
    "momentum_12_1": {
        "description": "close return from 252 sessions ago to 21 sessions ago",
        "columns": ["close"],
        "warmup_bars": 252,
        "pit_required": False,
        "pit_columns": [],
    },
    "volatility_120d": {
        "description": "120-session annualized close-to-close volatility",
        "columns": ["close"],
        "warmup_bars": 121,
        "pit_required": False,
        "pit_columns": [],
    },
    "volatility_252d": {
        "description": "252-session annualized close-to-close volatility",
        "columns": ["close"],
        "warmup_bars": 253,
        "pit_required": False,
        "pit_columns": [],
    },
    "downside_vol_120d": {
        "description": "120-session annualized downside deviation",
        "columns": ["close"],
        "warmup_bars": 121,
        "pit_required": False,
        "pit_columns": [],
    },
    "close_to_year_low": {
        "description": "close divided by the trailing 252-session low",
        "columns": ["close"],
        "warmup_bars": 252,
        "pit_required": False,
        "pit_columns": [],
    },
    "max_return_60d": {
        "description": "largest daily close-to-close return in 60 sessions",
        "columns": ["close"],
        "warmup_bars": 61,
        "pit_required": False,
        "pit_columns": [],
    },
    "amihud_illiq_120d": {
        "description": "120-session mean of absolute return divided by close times volume",
        "columns": ["close", "volume"],
        "warmup_bars": 121,
        "pit_required": False,
        "pit_columns": [],
    },
    "same_month_return": {
        "description": "average of the one-month return ending one to five years ago",
        "columns": ["close"],
        "warmup_bars": 273,
        "pit_required": False,
        "pit_columns": [],
    },
    "residual_momentum_12_1": {
        "description": "sum of market-model residuals from 252 sessions ago through 21 sessions ago",
        "columns": ["close", "market_return"],
        "warmup_bars": 252,
        "pit_required": True,
        "pit_columns": ["market_return"],
    },
    "turnover_rate_120d": {
        "description": "120-session mean of raw-share turnover",
        "columns": ["volume", "free_float_shares", "volume_unit", "share_basis"],
        "warmup_bars": 120,
        "pit_required": True,
        "pit_columns": ["free_float_shares"],
    },
    "turnover_rate_252d": {
        "description": "252-session mean of raw-share turnover",
        "columns": ["volume", "free_float_shares", "volume_unit", "share_basis"],
        "warmup_bars": 252,
        "pit_required": True,
        "pit_columns": ["free_float_shares"],
    },
    "sales_growth": {
        "description": "growth between the two latest distinct revenue observations",
        "columns": ["revenue"],
        "warmup_bars": 2,
        "pit_required": True,
        "pit_columns": ["revenue"],
    },
    "earnings_growth": {
        "description": "growth between the two latest distinct net-income observations",
        "columns": ["net_income"],
        "warmup_bars": 2,
        "pit_required": True,
        "pit_columns": ["net_income"],
    },
    "asset_turnover": {
        "description": "revenue divided by total assets",
        "columns": ["revenue", "total_assets"],
        "warmup_bars": 1,
        "pit_required": True,
        "pit_columns": ["revenue", "total_assets"],
    },
    "financial_leverage": {
        "description": "total assets divided by book equity",
        "columns": ["total_assets", "book_equity"],
        "warmup_bars": 1,
        "pit_required": True,
        "pit_columns": ["total_assets", "book_equity"],
    },
    "gross_margin": {
        "description": "(revenue minus cost of goods) divided by revenue",
        "columns": ["revenue", "cogs"],
        "warmup_bars": 1,
        "pit_required": True,
        "pit_columns": ["revenue", "cogs"],
    },
    "roe": {
        "description": "net income divided by book equity",
        "columns": ["net_income", "book_equity"],
        "warmup_bars": 1,
        "pit_required": True,
        "pit_columns": ["book_equity", "net_income"],
    },
    "consensus_revision_60d": {
        "description": "60-session change in consensus EPS divided by close",
        "columns": ["consensus_eps", "close"],
        "warmup_bars": 60,
        "pit_required": True,
        "pit_columns": ["consensus_eps"],
    },
    "earnings_surprise": {
        "description": "(reported EPS minus consensus EPS) divided by close",
        "columns": ["reported_eps", "consensus_eps", "close"],
        "warmup_bars": 1,
        "pit_required": True,
        "pit_columns": ["reported_eps", "consensus_eps"],
    },
    "northbound_hold_change_5d": {
        "description": "five-session percent change in northbound holding ratio",
        "columns": ["northbound_hold_ratio"],
        "warmup_bars": 5,
        "pit_required": True,
        "pit_columns": ["northbound_hold_ratio"],
    },
}

MINUTE_FACTORS: dict[str, str] = {
    "smart_money": "smart-money VWAP divided by full-session VWAP, minus one",
    "smart_money_20d": "mean of daily smart money over 20 sessions",
    "minute_realized_vol": "square root of summed squared minute returns",
    "minute_return_skew": "skewness of minute returns",
    "downside_minute_vol_share": "share of squared minute returns that are negative",
    "minute_return_volume_corr": "correlation of minute returns and minute volume",
    "open_bar_volume_share": "volume share of the first eighth of ordered bars",
    "tail_bar_volume_share": "volume share of the last eighth of ordered bars",
    "afternoon_return": "last close divided by the midpoint bar close, minus one",
}


def list_named_research_factors() -> dict[str, str]:
    return {name: spec["description"] for name, spec in RESEARCH_FACTORS.items()}


def research_factor_requirements(name: str) -> dict:
    spec = RESEARCH_FACTORS[name]
    return {
        "columns": list(spec["columns"]),
        "warmup_bars": int(spec["warmup_bars"]),
        "pit_required": bool(spec["pit_required"]),
        "pit_columns": list(spec["pit_columns"]),
        "dependencies": [name],
    }


def compute_named_research_factors(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Compute requested daily research factors. Missing inputs raise."""

    unknown = sorted(set(names) - set(RESEARCH_FACTORS))
    if unknown:
        raise ResearchFactorError(f"Unknown research factors: {unknown}")
    result = frame.copy()
    for name in names:
        result[name] = _COMPUTERS[name](result)
    return result


def _ordered_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ResearchFactorError(f"Missing columns: {missing}")
    sort_by = ["symbol", "date"] + (["time"] if "time" in frame.columns else [])
    ordered = frame.sort_values(sort_by, kind="stable")
    if ordered.duplicated(["symbol", "date", "time"]).any() if "time" in ordered.columns else False:
        raise ResearchFactorError("Minute bars require unique symbol/date/time")
    return ordered


def _smart_money(close: np.ndarray, volume: np.ndarray) -> float:
    if len(close) < 3 or not np.isfinite(close).all() or not np.isfinite(volume).all():
        return np.nan
    if (volume < 0).any() or volume.sum() <= 0 or (close <= 0).any():
        return np.nan
    returns = close[1:] / close[:-1] - 1
    traded = volume[1:]
    score = np.abs(returns) / np.sqrt(np.maximum(traded, 1e-12))
    order = np.argsort(-score, kind="mergesort")
    target = 0.2 * float(volume.sum())
    chosen: list[int] = []
    accumulated = 0.0
    for position in order:
        chosen.append(int(position))
        accumulated += float(traded[position])
        if accumulated >= target:
            break
    picked = np.array(chosen)
    smart_amount = float(np.sum(close[1:][picked] * traded[picked]))
    smart_volume = float(np.sum(traded[picked]))
    full_amount = float(np.sum(close * volume))
    if smart_volume <= 0 or full_amount <= 0:
        return np.nan
    return smart_amount / smart_volume / (full_amount / float(volume.sum())) - 1


def _moment_stats(close: np.ndarray, volume: np.ndarray) -> dict[str, float]:
    empty = {
        "minute_realized_vol": np.nan,
        "minute_return_skew": np.nan,
        "downside_minute_vol_share": np.nan,
        "minute_return_volume_corr": np.nan,
        "open_bar_volume_share": np.nan,
        "tail_bar_volume_share": np.nan,
        "afternoon_return": np.nan,
    }
    if len(close) < 3 or volume.sum() <= 0 or not np.isfinite(close).all():
        return empty
    returns = close[1:] / close[:-1] - 1
    traded = volume[1:]
    if not np.isfinite(returns).all() or not np.isfinite(traded).all():
        return empty
    squared = returns**2
    total_square = float(squared.sum())
    deviation = returns - returns.mean()
    scale = float(returns.std(ddof=1))
    skew = np.nan
    if scale > 0 and len(returns) >= 3:
        skew = float(np.mean(deviation**3) / scale**3)
    correlation = np.nan
    if float(np.std(returns)) > 0 and float(np.std(traded)) > 0:
        correlation = float(np.corrcoef(returns, traded)[0, 1])
    eighth = max(1, len(volume) // 8)
    midpoint = max(1, len(close) // 2)
    empty.update(
        {
            "minute_realized_vol": float(np.sqrt(total_square)),
            "minute_return_skew": skew,
            "downside_minute_vol_share": (
                np.nan if total_square == 0 else float(squared[returns < 0].sum() / total_square)
            ),
            "minute_return_volume_corr": correlation,
            "open_bar_volume_share": float(volume[:eighth].sum() / volume.sum()),
            "tail_bar_volume_share": float(volume[-eighth:].sum() / volume.sum()),
            "afternoon_return": float(close[-1] / close[midpoint] - 1)
            if close[midpoint] > 0
            else np.nan,
        }
    )
    return empty


def compute_minute_factors(frame: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    """Aggregate ordered intraday bars to one row per symbol and session."""

    requested = list(MINUTE_FACTORS if names is None else names)
    unknown = sorted(set(requested) - set(MINUTE_FACTORS))
    if unknown:
        raise ResearchFactorError(f"Unknown minute factors: {unknown}")
    ordered = _ordered_minutes(frame)
    rows = []
    for (symbol, session), group in ordered.groupby(["symbol", "date"], sort=False):
        close = group["close"].to_numpy(dtype=float)
        volume = group["volume"].to_numpy(dtype=float)
        stats = _moment_stats(close, volume)
        stats["smart_money"] = _smart_money(close, volume)
        stats["symbol"] = symbol
        stats["date"] = session
        rows.append(stats)
    daily = pd.DataFrame(rows).sort_values(["symbol", "date"], kind="stable")
    pieces = []
    for _, group in daily.groupby("symbol", sort=False):
        group = group.copy()
        group["smart_money_20d"] = group["smart_money"].rolling(20, min_periods=20).mean()
        pieces.append(group)
    result = pd.concat(pieces, ignore_index=True)
    return result[["symbol", "date", *requested]]
