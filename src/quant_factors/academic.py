"""Opt-in stock characteristics with explicit accounting identities.

These values are descriptors for research screens. They are not return forecasts,
and they are not MSCI Barra descriptors. Statement fields must already be the
latest figures known on that row; this module does not look ahead or lag a
trading calendar to invent a fiscal year. ``statement_currency`` is the unit of
every amount column these ratios use, including market cap. Currencies are not
converted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
MOMENTUM_SKIP = 21
MOMENTUM_LOOKBACK = 252


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


def _liquidity_log_turnover(frame: pd.DataFrame) -> pd.Series:
    """Log of the 20-session mean raw share turnover. Non-positive means stay missing."""
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
    rate = (volume / shares).rolling(20).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        logged = np.log(rate.where(rate > 0).to_numpy(dtype=float))
    return pd.Series(logged, index=frame.index)


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
