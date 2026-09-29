"""Opt-in stock characteristics with explicit accounting identities.

These values are descriptors for research screens. They are not return forecasts,
and they are not MSCI Barra descriptors. Statement fields must already be the
latest figures known on that row; this module does not look ahead or lag a
trading calendar to invent a fiscal year. ``statement_currency`` is the unit of
every amount column these ratios use, including market cap. Currencies are not
converted.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

Compute = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class AcademicSpec:
    """One opt-in characteristic."""

    name: str
    summary: str
    role: str
    data_columns: tuple[str, ...]
    pit_columns: tuple[str, ...]
    warmup_bars: int
    requires_statement_currency: bool
    compute: Compute
    scope: str = "symbol"
    currency_when_present: bool = False
    single_market_return: bool = False


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _empty(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def _ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    positive_denominator: bool,
) -> pd.Series:
    num = _numeric(numerator)
    den = _numeric(denominator)
    valid = np.isfinite(num.to_numpy()) & np.isfinite(den.to_numpy())
    den_values = den.to_numpy()
    valid = valid & (den_values > 0 if positive_denominator else den_values != 0)
    out = _empty(num.index)
    if valid.any():
        out.iloc[np.flatnonzero(valid)] = num.to_numpy()[valid] / den.to_numpy()[valid]
    return out.replace([np.inf, -np.inf], np.nan)


def _earnings_to_price(frame: pd.DataFrame) -> pd.Series:
    """Parent TTM earnings over same-day market cap. Losses stay negative."""
    return _ratio(
        frame["net_profit_parent_ttm"],
        frame["market_cap"],
        positive_denominator=True,
    )


def _roe(frame: pd.DataFrame) -> pd.Series:
    """TTM earnings over the latest known positive book equity.

    The denominator is the caller's already-public book equity, not an average
    with a later report that was not yet known.
    """
    return _ratio(
        frame["net_profit_parent_ttm"],
        frame["book_equity"],
        positive_denominator=True,
    )


def _gross_profitability(frame: pd.DataFrame) -> pd.Series:
    """Novy-Marx gross profit over the latest known positive total assets."""
    return _ratio(frame["gross_profit"], frame["total_assets"], positive_denominator=True)


def _asset_growth(frame: pd.DataFrame) -> pd.Series:
    """Total assets over the caller-supplied prior-year assets, minus one.

    Both asset stocks must be positive. The prior-year column is not filled by
    shifting this panel.
    """
    current = _numeric(frame["total_assets"])
    prior = _numeric(frame["total_assets_prior_year"])
    valid = (
        np.isfinite(current.to_numpy())
        & np.isfinite(prior.to_numpy())
        & (current.to_numpy() > 0)
        & (prior.to_numpy() > 0)
    )
    out = _empty(frame.index)
    if valid.any():
        grown = current.to_numpy()[valid] / prior.to_numpy()[valid] - 1.0
        out.iloc[np.flatnonzero(valid)] = grown
    return out.replace([np.inf, -np.inf], np.nan)


def _reversal_20d(frame: pd.DataFrame) -> pd.Series:
    """One-month simple reversal. Equal to minus the existing 20-observation momentum."""
    close = frame["close"]
    return -(close / close.shift(20) - 1.0)


SHELL_QUANTILE = 0.30
SHELL_MIN_NAMES = 10
BETA_WINDOW = 252
BETA_EW_HALF_LIFE = 63
MOMENTUM_SKIP = 21
MOMENTUM_LOOKBACK = 252
LONG_REVERSAL_LOOKBACK = 504
EARNINGS_VARIABILITY_WINDOW = 1260


def _positive_market_cap(frame: pd.DataFrame) -> pd.Series:
    cap = _numeric(frame["market_cap"])
    return cap.where(np.isfinite(cap.to_numpy()) & (cap.to_numpy() > 0))


def _size_log(frame: pd.DataFrame) -> pd.Series:
    cap = _positive_market_cap(frame)
    with np.errstate(invalid="ignore", divide="ignore"):
        logged = np.log(cap.to_numpy(dtype=float))
    return pd.Series(logged, index=frame.index)


def _size_ex_shell(frame: pd.DataFrame) -> pd.Series:
    """Log cap above the date's 30th percentile. Thin dates stay missing.

    The cutoff follows the Liu-Stambaugh-Yuan shell band as a stock screen.
    It is not their size-portfolio return.
    """
    out = _empty(frame.index)
    cap = _positive_market_cap(frame)
    for index in frame.groupby("date", sort=False).groups.values():
        values = cap.loc[index].dropna()
        if len(values) < SHELL_MIN_NAMES:
            continue
        kept = values[values > values.quantile(SHELL_QUANTILE)]
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[kept.index] = np.log(kept.to_numpy(dtype=float))
    return out


def _nonlinear_size(frame: pd.DataFrame) -> pd.Series:
    """Cube of the log-cap z-score, residualized on log cap within the date.

    Public approximation for a nonlinear-size exposure. Not an MSCI descriptor.
    """
    out = _empty(frame.index)
    cap = _positive_market_cap(frame)
    for index in frame.groupby("date", sort=False).groups.values():
        values = cap.loc[index].dropna()
        if len(values) < SHELL_MIN_NAMES:
            continue
        size = np.log(values.to_numpy(dtype=float))
        scale = float(size.std(ddof=0))
        if not np.isfinite(scale) or scale == 0.0:
            continue
        cube = ((size - size.mean()) / scale) ** 3
        design = np.column_stack([np.ones(len(size)), size])
        coef, _, _, _ = np.linalg.lstsq(design, cube, rcond=None)
        out.loc[values.index] = cube - design @ coef
    return out


def _market_model(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Equal-weight 252-session slope and annualized residual volatility.

    Residual variance is var(stock) - beta^2 * var(market) inside a complete
    window. This is not an exponentially weighted Barra regression.
    """
    ret = frame["close"] / frame["close"].shift(1) - 1.0
    market = _numeric(frame["market_return"])
    finite = np.isfinite(ret.to_numpy(dtype=float)) & np.isfinite(market.to_numpy(dtype=float))
    count = pd.Series(finite, index=frame.index).rolling(BETA_WINDOW, min_periods=BETA_WINDOW).sum()
    cov = ret.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).cov(market)
    var_market = market.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var()
    var_stock = ret.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var()
    complete = count == BETA_WINDOW
    beta = (cov / var_market.where(var_market > 0)).where(complete)
    var_residual = (var_stock - beta.pow(2) * var_market).where(beta.notna())
    usable = var_residual.notna() & (var_residual >= -1e-8)
    annualized = _empty(frame.index)
    if usable.any():
        scale = np.sqrt(np.maximum(var_residual.to_numpy()[usable.to_numpy()], 0.0))
        annualized.iloc[np.flatnonzero(usable.to_numpy())] = scale * np.sqrt(252.0)
    return beta, annualized


def _beta_252(frame: pd.DataFrame) -> pd.Series:
    """Plain 252-session slope on a supplied simple market return."""
    return _market_model(frame)[0]


def _residual_vol_252(frame: pd.DataFrame) -> pd.Series:
    """Annualized residual volatility from the same 252-session regression."""
    return _market_model(frame)[1]


def _momentum_252_21(frame: pd.DataFrame) -> pd.Series:
    """Relative strength from 252 sessions ago to 21 sessions ago.

    The latest month is left out so the exposure is not the short-term reversal.
    """
    close = frame["close"]
    return close.shift(MOMENTUM_SKIP) / close.shift(MOMENTUM_LOOKBACK) - 1.0


def _liquidity_log_turnover(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    """Log of the mean raw share turnover. Non-positive means stay missing."""
    if not frame["volume_unit"].eq("shares").all() or not frame["share_basis"].eq("raw").all():
        raise ValueError("liquidity_log_turnover_20d requires raw share units")
    volume = _numeric(frame["volume"])
    shares = _numeric(frame["free_float_shares"])
    if (
        (volume.dropna() < 0).any()
        or (shares.dropna() <= 0).any()
        or not np.isfinite(volume.dropna()).all()
        or not np.isfinite(shares.dropna()).all()
    ):
        raise ValueError(
            "liquidity_log_turnover_20d requires finite nonnegative volume and positive float"
        )
    rate = (volume / shares).rolling(window).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        logged = np.log(rate.where(rate > 0).to_numpy(dtype=float))
    return pd.Series(logged, index=frame.index)


def _liquidity_log_turnover_60(frame: pd.DataFrame) -> pd.Series:
    return _liquidity_log_turnover(frame, 60)


def _liquidity_log_turnover_240(frame: pd.DataFrame) -> pd.Series:
    return _liquidity_log_turnover(frame, 240)


def _ew_beta(frame: pd.DataFrame) -> pd.Series:
    """252-session beta with a 63-session half-life. The half-life is an independent default."""
    stock = frame["close"] / frame["close"].shift(1) - 1.0
    market = _numeric(frame["market_return"])
    out = np.full(len(frame), np.nan)
    if len(frame) < BETA_WINDOW:
        return pd.Series(out, index=frame.index)
    decay = math.exp(-math.log(2.0) / BETA_EW_HALF_LIFE)
    weights = decay ** np.arange(BETA_WINDOW - 1, -1, -1, dtype=float)
    weights = weights / float(weights.sum())
    stock_windows = sliding_window_view(stock.to_numpy(dtype=float), BETA_WINDOW)
    market_windows = sliding_window_view(market.to_numpy(dtype=float), BETA_WINDOW)
    complete = np.isfinite(stock_windows).all(axis=1) & np.isfinite(market_windows).all(axis=1)
    stock_centered = stock_windows - (stock_windows @ weights)[:, None]
    market_centered = market_windows - (market_windows @ weights)[:, None]
    variance = market_centered**2 @ weights
    covariance = (stock_centered * market_centered) @ weights
    usable = complete & (variance > 0)
    beta = np.divide(covariance, variance, out=np.full(len(variance), np.nan), where=usable)
    out[BETA_WINDOW - 1 :] = beta
    return pd.Series(out, index=frame.index)


def _short_reversal_21(frame: pd.DataFrame) -> pd.Series:
    """Minus the 21-session return, so the exposure is short-horizon reversal."""
    close = frame["close"]
    return -(close / close.shift(MOMENTUM_SKIP) - 1.0)


def _long_reversal(frame: pd.DataFrame) -> pd.Series:
    """Minus the second-year return, from 504 to 252 sessions ago.

    The window ends where the 252-session momentum window starts, so the two do not overlap.
    """
    close = frame["close"]
    return -(close.shift(BETA_WINDOW) / close.shift(LONG_REVERSAL_LOOKBACK) - 1.0)


def _seasonality_lag_year(frame: pd.DataFrame) -> pd.Series:
    """Return of the 21 sessions that began one year ago: the coming month, last year."""
    close = frame["close"]
    return close.shift(BETA_WINDOW - MOMENTUM_SKIP) / close.shift(BETA_WINDOW) - 1.0


def _dividend_yield(frame: pd.DataFrame) -> pd.Series:
    """Non-negative trailing cash dividend over positive market cap."""
    dividend = _numeric(frame["cash_dividend_ttm"])
    cap = _positive_market_cap(frame)
    valid = (
        np.isfinite(dividend.to_numpy()) & np.isfinite(cap.to_numpy()) & (dividend.to_numpy() >= 0)
    )
    out = _empty(frame.index)
    if valid.any():
        out.iloc[np.flatnonzero(valid)] = dividend.to_numpy()[valid] / cap.to_numpy()[valid]
    return out


def _cash_earnings_yield(frame: pd.DataFrame) -> pd.Series:
    """Trailing operating cash flow over market cap. Negative cash flow stays negative."""
    return _ratio(frame["operating_cashflow_ttm"], frame["market_cap"], positive_denominator=True)


def _market_leverage(frame: pd.DataFrame) -> pd.Series:
    """(market cap + non-negative debt) / market cap."""
    cap = _positive_market_cap(frame)
    debt = _numeric(frame["total_debt"])
    valid = np.isfinite(cap.to_numpy()) & np.isfinite(debt.to_numpy()) & (debt.to_numpy() >= 0)
    out = _empty(frame.index)
    if valid.any():
        out.iloc[np.flatnonzero(valid)] = (
            cap.to_numpy()[valid] + debt.to_numpy()[valid]
        ) / cap.to_numpy()[valid]
    return out


def _debt_to_assets(frame: pd.DataFrame) -> pd.Series:
    """Non-negative debt over positive total assets."""
    debt = _numeric(frame["total_debt"])
    assets = _numeric(frame["total_assets"])
    valid = (
        np.isfinite(debt.to_numpy())
        & np.isfinite(assets.to_numpy())
        & (debt.to_numpy() >= 0)
        & (assets.to_numpy() > 0)
    )
    out = _empty(frame.index)
    if valid.any():
        out.iloc[np.flatnonzero(valid)] = debt.to_numpy()[valid] / assets.to_numpy()[valid]
    return out


def _sales_growth(frame: pd.DataFrame) -> pd.Series:
    """TTM revenue over supplied prior positive revenue, minus one."""
    current = _numeric(frame["revenue_ttm"])
    prior = _numeric(frame["revenue_ttm_prior_year"])
    valid = np.isfinite(current.to_numpy()) & np.isfinite(prior.to_numpy()) & (prior.to_numpy() > 0)
    out = _empty(frame.index)
    if valid.any():
        out.iloc[np.flatnonzero(valid)] = current.to_numpy()[valid] / prior.to_numpy()[valid] - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def _earnings_variability(frame: pd.DataFrame) -> pd.Series:
    """Standard deviation of trailing earnings over its absolute mean, across 1260 sessions.

    Price does not enter, so a moving share price with flat earnings scores zero.
    """
    earnings = _numeric(frame["net_profit_parent_ttm"])
    rolling = earnings.rolling(EARNINGS_VARIABILITY_WINDOW, min_periods=EARNINGS_VARIABILITY_WINDOW)
    scale = rolling.mean().abs()
    return (rolling.std(ddof=1) / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)


def _passthrough(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric(frame[column])
    return values.where(np.isfinite(values.to_numpy()))


def _analyst_revision(frame: pd.DataFrame) -> pd.Series:
    """Caller-supplied analyst revision. This module does not estimate sentiment."""
    return _passthrough(frame, "analyst_revision")


def _forward_earnings_yield(frame: pd.DataFrame) -> pd.Series:
    """Caller-supplied forward earnings over market cap. Missing forecasts stay missing."""
    return _ratio(frame["forward_net_profit"], frame["market_cap"], positive_denominator=True)


def _expected_growth(frame: pd.DataFrame) -> pd.Series:
    """Caller-supplied expected growth. This module does not invent a forecast."""
    return _passthrough(frame, "expected_growth")


def _industry_momentum_20d(frame: pd.DataFrame) -> pd.Series:
    """Leave-one-out equal-weight 20-session industry return. A one-name industry stays missing."""
    ordered = frame.sort_values(["symbol", "date"])
    returns = ordered.groupby("symbol", sort=False)["close"].pct_change(20)
    work = ordered.assign(_industry_return=returns.to_numpy())
    out = _empty(frame.index)
    for _, day in work.groupby("date", sort=False):
        for _, peers in day.groupby("industry", sort=False):
            values = peers["_industry_return"]
            finite = values[np.isfinite(values.to_numpy(dtype=float))]
            if len(finite) < 2:
                continue
            total = float(finite.sum())
            count = len(finite)
            out.loc[finite.index] = (total - finite.to_numpy(dtype=float)) / (count - 1)
    return out


def _book_to_price(frame: pd.DataFrame) -> pd.Series:
    """Book equity over market cap. Negative book stays negative."""
    return _ratio(frame["book_equity"], frame["market_cap"], positive_denominator=True)


def _earnings_growth(frame: pd.DataFrame) -> pd.Series:
    """TTM earnings over supplied prior-year positive earnings, minus one.

    A non-positive prior-year earnings figure makes the growth rate undefined.
    """
    current = _numeric(frame["net_profit_parent_ttm"])
    prior = _numeric(frame["net_profit_parent_ttm_prior_year"])
    valid = np.isfinite(current.to_numpy()) & np.isfinite(prior.to_numpy()) & (prior.to_numpy() > 0)
    out = _empty(frame.index)
    if valid.any():
        grown = current.to_numpy()[valid] / prior.to_numpy()[valid] - 1.0
        out.iloc[np.flatnonzero(valid)] = grown
    return out.replace([np.inf, -np.inf], np.nan)


def _book_leverage(frame: pd.DataFrame) -> pd.Series:
    """Positive total assets over positive book equity. Not a blended leverage factor."""
    assets = _numeric(frame["total_assets"])
    book = _numeric(frame["book_equity"])
    valid = (
        np.isfinite(assets.to_numpy())
        & np.isfinite(book.to_numpy())
        & (assets.to_numpy() > 0)
        & (book.to_numpy() > 0)
    )
    out = _empty(frame.index)
    if valid.any():
        out.iloc[np.flatnonzero(valid)] = assets.to_numpy()[valid] / book.to_numpy()[valid]
    return out


def _specs() -> dict[str, AcademicSpec]:
    return {
        spec.name: spec
        for spec in (
            AcademicSpec(
                name="ep_ttm",
                summary=(
                    "Characteristic: parent TTM earnings / same-day market cap; "
                    "losses stay negative"
                ),
                role="characteristic",
                data_columns=("net_profit_parent_ttm", "market_cap"),
                pit_columns=("net_profit_parent_ttm",),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_earnings_to_price,
            ),
            AcademicSpec(
                name="roe_ttm",
                summary=(
                    "Characteristic: parent TTM earnings / latest positive book equity; "
                    "non-positive book is missing"
                ),
                role="characteristic",
                data_columns=("net_profit_parent_ttm", "book_equity"),
                pit_columns=("net_profit_parent_ttm", "book_equity"),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_roe,
            ),
            AcademicSpec(
                name="gross_profitability",
                summary="Characteristic: gross profit / latest positive total assets",
                role="characteristic",
                data_columns=("gross_profit", "total_assets"),
                pit_columns=("gross_profit", "total_assets"),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_gross_profitability,
            ),
            AcademicSpec(
                name="asset_growth_yoy",
                summary=(
                    "Characteristic: total assets / supplied prior-year assets - 1; "
                    "investment descriptor, not a trading-day lag"
                ),
                role="characteristic",
                data_columns=("total_assets", "total_assets_prior_year"),
                pit_columns=("total_assets", "total_assets_prior_year"),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_asset_growth,
            ),
            AcademicSpec(
                name="reversal_20d",
                summary="Characteristic: one-month reversal, equal to minus momentum_20d",
                role="characteristic",
                data_columns=("close",),
                pit_columns=(),
                warmup_bars=21,
                requires_statement_currency=False,
                compute=_reversal_20d,
            ),
            AcademicSpec(
                name="size_log_mcap",
                summary="Risk exposure: log of positive market cap; not an MSCI descriptor",
                role="risk_exposure",
                data_columns=("market_cap",),
                pit_columns=(),
                warmup_bars=1,
                requires_statement_currency=False,
                compute=_size_log,
                currency_when_present=True,
            ),
            AcademicSpec(
                name="size_log_mcap_ex_shell",
                summary=(
                    "Risk exposure: log cap above the date's smallest 30 percent; "
                    "missing when fewer than 10 names"
                ),
                role="risk_exposure",
                data_columns=("market_cap",),
                pit_columns=(),
                warmup_bars=1,
                requires_statement_currency=False,
                compute=_size_ex_shell,
                scope="cross_section",
                currency_when_present=True,
            ),
            AcademicSpec(
                name="nonlinear_size",
                summary=(
                    "Risk exposure: cubed log-cap z-score residualized on log cap; "
                    "not MSCI nonlinear size"
                ),
                role="risk_exposure",
                data_columns=("market_cap",),
                pit_columns=(),
                warmup_bars=1,
                requires_statement_currency=False,
                compute=_nonlinear_size,
                scope="cross_section",
                currency_when_present=True,
            ),
            AcademicSpec(
                name="beta_252d",
                summary=(
                    "Risk exposure: 252-session slope on a supplied market return; "
                    "not exponential Barra beta"
                ),
                role="risk_exposure",
                data_columns=("close", "market_return"),
                pit_columns=(),
                warmup_bars=BETA_WINDOW + 1,
                requires_statement_currency=False,
                compute=_beta_252,
                single_market_return=True,
            ),
            AcademicSpec(
                name="momentum_252_21",
                summary=(
                    "Risk exposure: close.shift(21) / close.shift(252) - 1; "
                    "neutralization momentum, not an A-share long alpha"
                ),
                role="risk_exposure",
                data_columns=("close",),
                pit_columns=(),
                warmup_bars=MOMENTUM_LOOKBACK + 1,
                requires_statement_currency=False,
                compute=_momentum_252_21,
            ),
            AcademicSpec(
                name="residual_vol_252d",
                summary=(
                    "Risk exposure: annualized residual vol from the 252-session market "
                    "regression; not Barra residual volatility"
                ),
                role="risk_exposure",
                data_columns=("close", "market_return"),
                pit_columns=(),
                warmup_bars=BETA_WINDOW + 1,
                requires_statement_currency=False,
                compute=_residual_vol_252,
                single_market_return=True,
            ),
            AcademicSpec(
                name="liquidity_log_turnover_20d",
                summary="Risk exposure: log of 20-session mean raw share turnover",
                role="risk_exposure",
                data_columns=("volume", "free_float_shares", "volume_unit", "share_basis"),
                pit_columns=("free_float_shares",),
                warmup_bars=20,
                requires_statement_currency=False,
                compute=_liquidity_log_turnover,
            ),
            AcademicSpec(
                name="book_to_price",
                summary=(
                    "Risk exposure: book equity / market cap; negative book stays negative; "
                    "not an A-share value alpha"
                ),
                role="risk_exposure",
                data_columns=("book_equity", "market_cap"),
                pit_columns=("book_equity",),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_book_to_price,
            ),
            AcademicSpec(
                name="earnings_yield_ttm",
                summary=(
                    "Risk exposure: same trailing earnings-to-price ratio as ep_ttm; "
                    "not a blended cash-earnings or forecast yield"
                ),
                role="risk_exposure",
                data_columns=("net_profit_parent_ttm", "market_cap"),
                pit_columns=("net_profit_parent_ttm",),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_earnings_to_price,
            ),
            AcademicSpec(
                name="earnings_growth_yoy",
                summary=(
                    "Risk exposure: TTM earnings / supplied prior positive earnings - 1; "
                    "not a blended growth factor"
                ),
                role="risk_exposure",
                data_columns=("net_profit_parent_ttm", "net_profit_parent_ttm_prior_year"),
                pit_columns=("net_profit_parent_ttm", "net_profit_parent_ttm_prior_year"),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_earnings_growth,
            ),
            AcademicSpec(
                name="book_leverage",
                summary=(
                    "Risk exposure: positive total assets / positive book equity; "
                    "not a blended leverage factor"
                ),
                role="risk_exposure",
                data_columns=("total_assets", "book_equity"),
                pit_columns=("total_assets", "book_equity"),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_book_leverage,
            ),
            AcademicSpec(
                name="liquidity_log_turnover_60d",
                summary="Risk exposure: log of 60-session mean raw share turnover",
                role="risk_exposure",
                data_columns=("volume", "free_float_shares", "volume_unit", "share_basis"),
                pit_columns=("free_float_shares",),
                warmup_bars=60,
                requires_statement_currency=False,
                compute=_liquidity_log_turnover_60,
            ),
            AcademicSpec(
                name="liquidity_log_turnover_240d",
                summary="Risk exposure: log of 240-session mean raw share turnover",
                role="risk_exposure",
                data_columns=("volume", "free_float_shares", "volume_unit", "share_basis"),
                pit_columns=("free_float_shares",),
                warmup_bars=240,
                requires_statement_currency=False,
                compute=_liquidity_log_turnover_240,
            ),
            AcademicSpec(
                name="beta_ew_252d",
                summary=(
                    "Risk exposure: 252-session beta with an independent 63-session half-life; "
                    "not an MSCI descriptor"
                ),
                role="risk_exposure",
                data_columns=("close", "market_return"),
                pit_columns=(),
                warmup_bars=BETA_WINDOW + 1,
                requires_statement_currency=False,
                compute=_ew_beta,
                single_market_return=True,
            ),
            AcademicSpec(
                name="short_term_reversal_21d",
                summary="Risk exposure: minus the 21-session return",
                role="risk_exposure",
                data_columns=("close",),
                pit_columns=(),
                warmup_bars=MOMENTUM_SKIP + 1,
                requires_statement_currency=False,
                compute=_short_reversal_21,
            ),
            AcademicSpec(
                name="long_term_reversal_504_252",
                summary=(
                    "Risk exposure: minus the return from 504 to 252 sessions ago; "
                    "does not overlap momentum_252_21"
                ),
                role="risk_exposure",
                data_columns=("close",),
                pit_columns=(),
                warmup_bars=LONG_REVERSAL_LOOKBACK + 1,
                requires_statement_currency=False,
                compute=_long_reversal,
            ),
            AcademicSpec(
                name="seasonality_21d_lag_252",
                summary=(
                    "Risk exposure: return of the 21 sessions that began 252 sessions ago, "
                    "the coming month last year"
                ),
                role="risk_exposure",
                data_columns=("close",),
                pit_columns=(),
                warmup_bars=BETA_WINDOW + 1,
                requires_statement_currency=False,
                compute=_seasonality_lag_year,
            ),
            AcademicSpec(
                name="dividend_yield_ttm",
                summary="Risk exposure: non-negative trailing cash dividend / market cap",
                role="risk_exposure",
                data_columns=("cash_dividend_ttm", "market_cap"),
                pit_columns=("cash_dividend_ttm",),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_dividend_yield,
                currency_when_present=True,
            ),
            AcademicSpec(
                name="cash_earnings_yield_ttm",
                summary=(
                    "Risk exposure: trailing operating cash flow / market cap; "
                    "negative cash flow stays negative"
                ),
                role="risk_exposure",
                data_columns=("operating_cashflow_ttm", "market_cap"),
                pit_columns=("operating_cashflow_ttm",),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_cash_earnings_yield,
            ),
            AcademicSpec(
                name="market_leverage",
                summary="Risk exposure: (market cap + non-negative debt) / market cap",
                role="risk_exposure",
                data_columns=("market_cap", "total_debt"),
                pit_columns=("total_debt",),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_market_leverage,
                currency_when_present=True,
            ),
            AcademicSpec(
                name="debt_to_assets",
                summary="Risk exposure: non-negative debt / positive total assets",
                role="risk_exposure",
                data_columns=("total_debt", "total_assets"),
                pit_columns=("total_debt", "total_assets"),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_debt_to_assets,
            ),
            AcademicSpec(
                name="sales_growth_yoy",
                summary="Risk exposure: TTM revenue / prior positive revenue - 1",
                role="risk_exposure",
                data_columns=("revenue_ttm", "revenue_ttm_prior_year"),
                pit_columns=("revenue_ttm", "revenue_ttm_prior_year"),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_sales_growth,
            ),
            AcademicSpec(
                name="earnings_variability_1260d",
                summary=(
                    "Risk exposure: std / |mean| of trailing earnings over 1260 sessions; "
                    "price does not enter"
                ),
                role="risk_exposure",
                data_columns=("net_profit_parent_ttm",),
                pit_columns=("net_profit_parent_ttm",),
                warmup_bars=EARNINGS_VARIABILITY_WINDOW,
                requires_statement_currency=True,
                compute=_earnings_variability,
            ),
            AcademicSpec(
                name="analyst_revision",
                summary="Risk exposure: caller-supplied analyst revision; not estimated here",
                role="risk_exposure",
                data_columns=("analyst_revision",),
                pit_columns=("analyst_revision",),
                warmup_bars=1,
                requires_statement_currency=False,
                compute=_analyst_revision,
            ),
            AcademicSpec(
                name="forward_earnings_yield",
                summary="Risk exposure: caller-supplied forward earnings / market cap",
                role="risk_exposure",
                data_columns=("forward_net_profit", "market_cap"),
                pit_columns=("forward_net_profit",),
                warmup_bars=1,
                requires_statement_currency=True,
                compute=_forward_earnings_yield,
            ),
            AcademicSpec(
                name="expected_growth",
                summary="Risk exposure: caller-supplied expected growth; not estimated here",
                role="risk_exposure",
                data_columns=("expected_growth",),
                pit_columns=("expected_growth",),
                warmup_bars=1,
                requires_statement_currency=False,
                compute=_expected_growth,
            ),
            AcademicSpec(
                name="industry_momentum_20d",
                summary=(
                    "Risk exposure: leave-one-out 20-session industry return; "
                    "not an MSCI industry momentum factor"
                ),
                role="risk_exposure",
                data_columns=("close", "industry"),
                pit_columns=(),
                warmup_bars=21,
                requires_statement_currency=False,
                compute=_industry_momentum_20d,
                scope="cross_section",
            ),
        )
    }


FACTOR_SPECS: dict[str, AcademicSpec] = _specs()
ACADEMIC_REGISTRY: dict[str, str] = {name: spec.summary for name, spec in FACTOR_SPECS.items()}


def academic_requirement(name: str) -> dict | None:
    """Column, warmup and PIT contract for an academic id, or None."""
    spec = FACTOR_SPECS.get(name)
    if spec is None:
        return None
    columns = list(spec.data_columns)
    if spec.requires_statement_currency:
        columns.append("statement_currency")
    return {
        "columns": columns,
        "warmup_bars": spec.warmup_bars,
        "pit_required": bool(spec.pit_columns),
        "pit_columns": list(spec.pit_columns),
        "role": spec.role,
    }


def academic_is_fundamental(name: str) -> bool:
    spec = FACTOR_SPECS.get(name)
    return spec is not None and bool(spec.pit_columns)


def academic_scope(name: str) -> str | None:
    spec = FACTOR_SPECS.get(name)
    return None if spec is None else spec.scope


def _require_one_currency(frame: pd.DataFrame, name: str) -> None:
    currency = frame["statement_currency"]
    if currency.isna().any() or currency.nunique() != 1:
        raise ValueError(f"{name} requires one non-null statement_currency")


def assert_academic_contracts(frame: pd.DataFrame, names: list[str]) -> None:
    """Fail closed on a broken currency or on two market returns in one date."""
    for name in names:
        spec = FACTOR_SPECS[name]
        if any(column not in frame.columns for column in spec.data_columns):
            continue
        if spec.requires_statement_currency:
            if "statement_currency" not in frame.columns:
                raise ValueError(f"{name} requires statement_currency")
            _require_one_currency(frame, name)
        elif spec.currency_when_present and "statement_currency" in frame.columns:
            _require_one_currency(frame, name)
        if not spec.single_market_return or "market_return" not in frame.columns:
            continue
        for _, day in frame.groupby("date", sort=False):
            numeric = pd.to_numeric(day["market_return"], errors="coerce")
            if (day["market_return"].notna() & numeric.isna()).any():
                raise ValueError(f"{name} requires a numeric market_return")
            finite = numeric.dropna()
            if finite.nunique() > 1:
                raise ValueError(f"{name} requires one market_return per date")


def apply_cross_section(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Add date-wise exposures. Each date uses only that date's rows."""
    result = frame.copy()
    for name in names:
        spec = FACTOR_SPECS[name]
        if any(column not in result.columns for column in spec.data_columns):
            result[name] = _empty(result.index)
        else:
            result[name] = spec.compute(result)
    return result


def compute_symbol_academic(name: str, frame: pd.DataFrame) -> pd.Series:
    """Compute one characteristic on one symbol. Missing inputs stay missing."""
    spec = FACTOR_SPECS[name]
    if any(column not in frame.columns for column in spec.data_columns):
        return _empty(frame.index)
    return spec.compute(frame)
