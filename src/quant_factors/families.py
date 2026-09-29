"""Quant-family labels for every registered factor.

Labels follow the families used in Chinese quant research and the matching
academic definitions: Barra-style styles, plus the standard alpha groups for
value, growth, profitability, quality, accruals, investment, reversal,
seasonality, financing, sentiment, flow, and technical patterns. A factor may
belong to more than one family. The label does not change the formula or its sign.

A close-to-close return through 42 sessions is reversal, including default names
such as ``momentum_20d``. A return of 60 sessions or more is momentum.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

FAMILY_DEFINITIONS: dict[str, str] = {
    "value": "价值. A fundamental quantity divided by price, market equity, or enterprise value.",
    "growth": "成长. The change in a fundamental stock or flow.",
    "profitability": "盈利. Earnings, cash flow, or a margin scaled by assets, book equity, or sales.",
    "quality": "质量. Earnings cleanliness, balance-sheet solvency, or asset tangibility.",
    "accrual": "应计. Working-capital accruals and the balance-sheet changes that enter them.",
    "investment": "投资. Asset growth, capital expenditure, or another investment intensity.",
    "momentum": "动量. A price trend over about three months to a year.",
    "reversal": "反转. A price move of about one month or less, or short-horizon mean reversion.",
    "volatility": "波动. Dispersion, range, skew, drawdown, or idiosyncratic variance.",
    "liquidity": "流动性. Turnover, volume, traded amount, or price impact.",
    "size": "规模. Market capitalization, enterprise value, or an unscaled firm-scale stock.",
    "leverage": "杠杆. Debt or liabilities relative to assets, equity, or earnings.",
    "beta": "贝塔. Sensitivity of return to a supplied market return.",
    "seasonality": "季节. The return from the same calendar window in earlier years.",
    "financing": "融资. Equity or debt issuance, repurchase, or dividends.",
    "sentiment": "情绪. Analyst revisions, earnings surprise, or an overnight attention return.",
    "flow": "资金. A change in holdings, or a smart-money flow.",
    "technical": "技术. A price-volume pattern, rank, correlation, or intraday shape.",
}

_HORIZONS = (21, 42, 63, 84, 105, 126, 168, 189, 210, 252)
_WINDOW = re.compile(r"(\d+)d")


class FamilyError(ValueError):
    """A factor has no family label, or a label is outside the taxonomy."""


def _ordered(tags: Iterable[str]) -> tuple[str, ...]:
    chosen = set(tags)
    unknown = chosen - set(FAMILY_DEFINITIONS)
    if unknown:
        raise FamilyError(f"Unknown families: {sorted(unknown)}")
    if not chosen:
        raise FamilyError("A factor needs at least one family")
    return tuple(name for name in FAMILY_DEFINITIONS if name in chosen)


def _lookup(table: dict[str, tuple[str, ...]], key: str, label: str) -> tuple[str, ...]:
    try:
        tags = table[key]
    except KeyError as exc:
        raise FamilyError(f"Unknown {label}: {key}") from exc
    return _ordered(tags)


CORE_FAMILIES: dict[str, tuple[str, ...]] = {
    "momentum_5d": ("reversal",),
    "momentum_10d": ("reversal",),
    "momentum_20d": ("reversal",),
    "momentum_60d": ("momentum",),
    "log_momentum_20d": ("reversal",),
    "reversal_5d": ("reversal",),
    "reversal_10d": ("reversal",),
    "volatility_10d": ("volatility",),
    "volatility_20d": ("volatility",),
    "volatility_60d": ("volatility",),
    "downside_vol_20d": ("volatility",),
    "mean_reversion_z_20d": ("reversal",),
    "volume_surge_5d": ("liquidity",),
    "turnover_20d": ("liquidity",),
    "amihud_illiq_20d": ("liquidity",),
    "pe_inv": ("value",),
    "pb_inv": ("value",),
    "average_volume_20d": ("liquidity",),
    "turnover_rate_20d_v2": ("liquidity",),
    "amihud_illiq_20d_v2": ("liquidity",),
}

ACADEMIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "ep_ttm": ("value",),
    "roe_ttm": ("profitability",),
    "gross_profitability": ("profitability",),
    "asset_growth_yoy": ("growth", "investment"),
    "reversal_20d": ("reversal",),
    "size_log_mcap": ("size",),
    "size_log_mcap_ex_shell": ("size",),
    "nonlinear_size": ("size",),
    "beta_252d": ("beta",),
    "momentum_252_21": ("momentum",),
    "residual_vol_252d": ("volatility",),
    "liquidity_log_turnover_20d": ("liquidity",),
    "book_to_price": ("value",),
    "earnings_yield_ttm": ("value",),
    "earnings_growth_yoy": ("growth", "profitability"),
    "book_leverage": ("leverage",),
}

RESEARCH_FAMILIES: dict[str, tuple[str, ...]] = {
    "high_amplitude_return_20d": ("reversal", "volatility"),
    "low_amplitude_return_20d": ("reversal", "volatility"),
    "amplitude_return_spread_20d": ("reversal", "volatility"),
    "close_to_year_high": ("momentum",),
    "max_return_21d": ("volatility",),
    "max5_return_21d": ("volatility",),
    "turnover_rate_change_20_60d": ("liquidity",),
    "turnover_rate_vol_20d": ("volatility", "liquidity"),
    "idio_vol_21d": ("volatility",),
    "return_specificity_21d": ("volatility",),
    "residual_return_21d": ("reversal",),
    "overnight_afternoon_spread_20d": ("reversal", "sentiment"),
    "accruals_to_assets": ("quality", "accrual"),
    "operating_profitability": ("profitability",),
    "asset_growth": ("growth", "investment"),
    "momentum_6_1": ("momentum",),
    "momentum_12_1": ("momentum",),
    "volatility_120d": ("volatility",),
    "volatility_252d": ("volatility",),
    "downside_vol_120d": ("volatility",),
    "close_to_year_low": ("reversal",),
    "max_return_60d": ("volatility",),
    "amihud_illiq_120d": ("liquidity",),
    "same_month_return": ("seasonality",),
    "residual_momentum_12_1": ("momentum",),
    "turnover_rate_120d": ("liquidity",),
    "turnover_rate_252d": ("liquidity",),
    "sales_growth": ("growth",),
    "earnings_growth": ("growth", "profitability"),
    "asset_turnover": ("profitability",),
    "financial_leverage": ("leverage",),
    "gross_margin": ("profitability",),
    "roe": ("profitability",),
    "consensus_revision_60d": ("sentiment",),
    "earnings_surprise": ("sentiment",),
    "northbound_hold_change_5d": ("flow",),
}

MINUTE_FAMILIES: dict[str, tuple[str, ...]] = {
    "smart_money": ("flow", "technical"),
    "smart_money_20d": ("flow", "technical"),
    "minute_realized_vol": ("volatility",),
    "minute_return_skew": ("volatility",),
    "downside_minute_vol_share": ("volatility",),
    "minute_return_volume_corr": ("liquidity", "technical"),
    "open_bar_volume_share": ("liquidity", "technical"),
    "tail_bar_volume_share": ("liquidity", "technical"),
    "afternoon_return": ("reversal", "sentiment"),
}

GROWTH_ITEM_FAMILIES: dict[str, tuple[str, ...]] = {
    "total_assets": ("growth", "investment"),
    "revenue": ("growth",),
    "current_assets": ("growth", "investment"),
    "noncurrent_assets": ("growth", "investment"),
    "total_liabilities": ("growth", "leverage"),
    "current_liabilities": ("growth", "leverage"),
    "noncurrent_liabilities": ("growth", "leverage"),
    "book_equity": ("growth",),
    "preferred_stock": ("growth", "financing"),
    "total_debt": ("growth", "leverage"),
    "cogs": ("growth",),
    "sga": ("growth",),
    "operating_expenses": ("growth",),
    "capex": ("growth", "investment"),
    "rd_expense": ("growth", "investment"),
    "employees": ("growth",),
    "inventory": ("growth", "accrual", "investment"),
    "receivables": ("growth", "accrual", "investment"),
    "cash": ("growth",),
    "ppe_gross": ("growth", "investment"),
    "intangibles": ("growth", "investment"),
    "net_income": ("growth", "profitability"),
}

SCALED_CHANGE_FAMILIES: dict[str, tuple[str, ...]] = {
    "gross_profit": ("growth", "profitability"),
    "operating_cash_flow": ("growth", "profitability"),
    "cash": ("investment",),
    "inventory": ("quality", "accrual", "investment"),
    "receivables": ("quality", "accrual", "investment"),
    "ppe_gross": ("investment",),
    "intangibles": ("investment",),
    "short_term_debt": ("leverage",),
    "accounts_payable": ("quality", "accrual"),
    "income_tax_payable": ("quality", "accrual"),
    "long_term_debt": ("leverage",),
    "deferred_tax": ("quality", "accrual"),
    "current_operating_assets": ("quality", "accrual", "investment"),
    "current_operating_liabilities": ("quality", "accrual"),
    "current_operating_working_capital": ("quality", "accrual", "investment"),
    "noncurrent_operating_assets": ("investment",),
    "noncurrent_operating_liabilities": ("quality", "accrual"),
    "net_noncurrent_operating_assets": ("accrual", "investment"),
    "operating_assets": ("accrual", "investment"),
    "operating_liabilities": ("quality", "accrual"),
    "net_operating_assets": ("quality", "accrual", "investment"),
    "financial_assets": ("investment",),
    "financial_liabilities": ("leverage",),
    "net_financial_assets": ("leverage",),
    "ebitda": ("growth", "profitability"),
    "ebit": ("growth", "profitability"),
    "operating_earnings": ("growth", "profitability"),
    "net_income": ("growth", "profitability"),
    "depreciation": ("quality", "accrual"),
    "free_cash_flow": ("growth", "profitability"),
    "net_working_capital": ("quality", "accrual", "investment"),
    "net_income_extraordinary": ("growth", "profitability"),
    "equity_net_issuance": ("financing",),
    "net_debt_issuance": ("leverage", "financing"),
    "dividends": ("financing",),
    "capex": ("investment",),
    "rd_expense": ("investment",),
    "advertising": ("investment",),
}

RATIO_FAMILIES: dict[str, tuple[str, ...]] = {
    "gross_profit_to_assets": ("profitability",),
    "ebitda_to_assets": ("profitability",),
    "ebit_to_assets": ("profitability",),
    "net_income_to_assets": ("profitability",),
    "operating_cash_flow_to_assets": ("profitability",),
    "free_cash_flow_to_assets": ("profitability",),
    "capex_to_assets": ("investment",),
    "rd_to_assets": ("investment",),
    "cash_to_assets": ("quality",),
    "inventory_to_assets": ("quality", "accrual"),
    "receivables_to_assets": ("quality", "accrual"),
    "debt_to_assets": ("leverage",),
    "net_operating_assets_to_assets": ("investment",),
    "gross_margin_ratio": ("profitability",),
    "ebitda_margin": ("profitability",),
    "ebit_margin": ("profitability",),
    "pretax_margin": ("profitability",),
    "net_margin": ("profitability",),
    "extraordinary_margin": ("profitability",),
    "free_cash_flow_margin": ("profitability",),
    "operating_cash_flow_margin": ("profitability",),
    "rd_to_sales": ("investment",),
    "advertising_to_sales": ("investment",),
    "staff_to_sales": ("profitability",),
    "operating_earnings_to_book": ("profitability",),
    "net_income_to_book": ("profitability",),
    "extraordinary_income_to_book": ("profitability",),
    "operating_cash_flow_to_book": ("profitability",),
    "free_cash_flow_to_book": ("profitability",),
    "debt_to_book": ("leverage",),
    "assets_to_book": ("leverage",),
    "book_to_market_equity": ("value",),
    "assets_to_market_equity": ("value",),
    "cash_to_market_equity": ("value",),
    "gross_profit_to_market_equity": ("value",),
    "ebitda_to_market_equity": ("value",),
    "ebit_to_market_equity": ("value",),
    "operating_earnings_to_market_equity": ("value",),
    "net_income_to_market_equity": ("value",),
    "revenue_to_market_equity": ("value",),
    "operating_cash_flow_to_market_equity": ("value",),
    "free_cash_flow_to_market_equity": ("value",),
    "rd_to_market_equity": ("value", "investment"),
    "debt_to_market_equity": ("leverage",),
    "dividends_to_market_equity": ("value", "financing"),
    "equity_issuance_to_market_equity": ("financing",),
    "equity_repurchase_to_market_equity": ("financing",),
    "book_to_enterprise_value": ("value",),
    "assets_to_enterprise_value": ("value",),
    "cash_to_enterprise_value": ("value",),
    "gross_profit_to_enterprise_value": ("value",),
    "ebitda_to_enterprise_value": ("value",),
    "ebit_to_enterprise_value": ("value",),
    "revenue_to_enterprise_value": ("value",),
    "operating_cash_flow_to_enterprise_value": ("value",),
    "free_cash_flow_to_enterprise_value": ("value",),
    "debt_to_enterprise_value": ("leverage",),
    "equity_issuance_to_assets": ("financing",),
    "equity_repurchase_to_assets": ("financing",),
    "equity_net_issuance_to_assets": ("financing",),
    "net_debt_issuance_to_assets": ("leverage", "financing"),
    "dividends_to_assets": ("financing",),
    "current_ratio": ("quality",),
    "cash_ratio": ("quality",),
    "interest_coverage": ("leverage",),
    "ebitda_to_debt": ("leverage",),
    "operating_cash_flow_to_current_liabilities": ("quality",),
    "pretax_to_extraordinary_income": ("quality",),
}

_SPECIAL_FUNDAMENTAL: dict[str, tuple[str, ...]] = {
    "cash_accruals_to_assets": ("quality", "accrual"),
    "tangibility": ("quality",),
    "gross_profit_to_lagged_assets": ("profitability",),
    "operating_earnings_to_lagged_book": ("profitability",),
    "cash_flow_to_lagged_assets": ("profitability",),
    "roe_change_5y": ("growth", "profitability"),
    "roa_change_5y": ("growth", "profitability"),
    "gross_margin_change_5y": ("growth", "profitability"),
    "cash_flow_to_assets_change_5y": ("growth", "profitability"),
}

_LEVEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "total_assets": ("size",),
    "revenue": ("size",),
    "book_equity": ("size",),
    "net_income": ("profitability", "size"),
    "market_enterprise_value": ("size",),
    "employees": ("size",),
}

_MEDIUM_TREND = {
    "trail_return",
    "log_trail_return",
    "positive_day_share",
    "trend_slope",
    "close_to_high",
    "close_to_low",
}
_MEDIUM_VOLATILITY = {
    "ann_volatility",
    "downside_volatility",
    "return_skew",
    "up_down_vol_ratio",
    "range_mean",
    "gain_loss_ratio",
    "max_drawdown",
    "max_daily_return",
    "min_daily_return",
}

PRICE_VOLUME_FACTORS = frozenset(
    {
        "log_volume_change_intraday_return_rank_corr",
        "close_location_change",
        "directional_gap_sum",
        "mean_band_volume_state",
        "volume_high_ts_rank_corr_max",
        "weighted_open_high_change_rank",
        "rank_volume_vwap_3d",
        "rank_vwap_high_4d",
        "smooth_volume_high_7d",
        "rank_volatility_return_close_20d",
        "volume_high_6d",
        "rank_vwap_open_10d",
        "geometric_price_vwap_gap",
        "overnight_return",
        "corr_smooth_rank_volume_vwap_5d",
        "rank_vwap_close_15d",
        "close_5d",
        "close_mean_trend_slope",
        "smooth_close_12d",
        "smooth_volatility_close_100d",
        "smooth_close_5d",
        "decay_rank_volume_return_250d",
        "corr_vwap_close_230d",
        "weighted_mean_close_100d",
        "smooth_high_low_100d",
        "volume_close_6d",
        "close_100d",
        "corr_rank_volume_high_3d",
        "ts_rank_rank_volume_return_240d",
        "close_12d",
        "decay_corr_rank_volume_open_17d",
        "corr_rank_volume_vwap_6d",
        "rank_return_open_10d",
        "high_20d",
        "decay_corr_rank_volume_vwap_180d",
        "up_down_volume_ratio",
        "rank_vwap_5d",
        "high_volatility_volume_corr",
        "signed_volume_sum",
        "decay_corr_ts_rank_volume_vwap_15d",
        "corr_rank_volume_vwap_150d",
        "close_24d",
        "smooth_high_low_100d_9",
        "rank_volume_close_20d",
        "down_shadow_pressure",
        "high_low_12d",
        "up_shadow_pressure",
        "high_low_100d",
        "up_day_share",
        "corr_rank_volatility_open_close_10d",
        "open_high_20d",
        "corr_rank_volume_open_40d",
        "smooth_high_low_100d_3",
        "decay_corr_rank_volume_vwap_80d",
        "corr_rank_volume_high_5d",
        "smooth_up_close_ratio",
        "decay_corr_rank_volume_vwap_60d",
        "directional_move_balance",
        "amount_volatility",
        "decay_corr_ts_rank_volume_vwap_30d",
        "corr_rank_volume_vwap_40d",
        "return_per_volume_stability",
        "decay_corr_rank_volume_vwap_40d",
        "high_low_12d_12",
        "volume_change",
        "smooth_volume",
        "cov_rank_volume_high_5d",
        "ts_rank_rank_volume_close_20d",
        "close_20d",
        "decay_ts_rank_rank_vwap_open_11d",
        "smooth_close_27d",
        "corr_rank_volume_vwap_5d",
        "corr_rank_volume_low_40d",
        "decay_corr_ts_rank_volume_vwap_180d",
        "open_low_20d",
        "smooth_high_low_100d_96",
        "volume_volatility",
        "close_100d_3",
        "cov_rank_volume_close_5d",
        "corr_rank_volume_vwap_37d",
        "smooth_up_volume_ratio",
        "days_since_low_share",
        "corr_rank_volatility_volume_high_20d",
        "corr_rank_volume_open_10d",
        "rank_open_high_1d",
        "corr_rank_volume_vwap_120d",
        "smooth_high_low_10d",
        "true_range_balance",
        "smooth_volume_high_11d",
        "close_100d_100",
        "corr_rank_volume_close_20d",
        "rank_volume_vwap_5d",
        "corr_ts_rank_rank_volume_high_30d",
        "slope_close_20d",
        "ts_rank_rank_volume_return_32d",
        "upper_lower_wick_ratio",
        "decay_corr_ts_rank_volume_vwap_26d",
        "vwap_close_rank_gap",
        "corr_ts_rank_rank_volume_vwap_60d",
        "smooth_close_13d",
        "corr_rank_volume_high_60d",
        "decay_smooth_rank_vwap_close_30d",
        "decay_corr_rank_volume_vwap_80d_16",
        "high_low_3d",
        "high_gap_root_mean_square",
        "volume_high_100d",
        "close_12d_2",
        "decay_corr_rank_volume_vwap_40d_3",
        "corr_ts_rank_rank_volume_vwap_50d",
        "average_amount",
        "high_low_age_gap",
        "smooth_close_20d",
        "corr_rank_volume_return_10d",
        "open_high_16d",
        "decay_corr_ts_rank_volume_vwap_60d",
        "open_volume_corr",
        "decay_corr_ts_rank_volume_open_60d",
        "corr_rank_volume_high_15d",
        "ts_rank_rank_volume_close_20d_5",
        "close_1d",
        "down_day_impact",
        "volume_mean_gap",
        "corr_rank_volume_open_60d",
        "typical_price_volume",
        "smooth_close_26d",
        "close_24d_4",
        "corr_volume_vwap_180d",
        "smooth_volume_27d",
        "decay_rank_vwap_open_5d",
        "ts_rank_rank_return_close_6d",
        "smooth_high_low_15d",
        "high_low_100d_24",
        "smooth_volatility_close_20d",
        "mean_true_range",
        "smooth_close_100d",
        "rank_volume_vwap_20d",
        "smooth_high_low_100d_13",
        "close_12d_3",
        "relative_volume",
        "smooth_close_26d_10",
        "rank_volume_vwap_20d_5",
        "intraday_wick_leverage",
        "price_volume_100d",
        "smooth_close_13d_13",
        "smooth_volatility_close_20d_20",
        "corr_smooth_rank_volume_high_12d",
        "days_since_high_share",
        "corr_rank_volume_vwap_50d",
        "corr_rank_open_close_200d",
        "open_close_gap_square",
        "price_volume_100d_6",
        "open_high_20d_2",
        "smooth_high_low_100d_100",
        "close_mean_absolute_deviation",
        "conditional_sum_day_count_close_20d",
        "volume_low_corr_mid_gap",
    }
)


def _trend(horizon: int) -> set[str]:
    if horizon <= 42:
        return {"reversal"}
    return {"momentum"}


def _medium_families(name: str) -> tuple[str, ...] | None:
    match = re.fullmatch(r"(.+)_(\d+)d", name)
    if match is None:
        return None
    kind, horizon = match.group(1), int(match.group(2))
    if kind not in _MEDIUM_TREND | _MEDIUM_VOLATILITY | {
        "sharpe",
        "close_to_mean",
        "amihud",
        "return_autocorr",
        "overnight_return_sum",
        "intraday_return_sum",
        "volume_expansion",
    }:
        return None
    if horizon not in _HORIZONS:
        return None
    tags: set[str] = set()
    if kind in _MEDIUM_TREND or kind == "sharpe":
        tags |= _trend(horizon)
    if kind in _MEDIUM_VOLATILITY or kind == "sharpe":
        tags.add("volatility")
    elif kind == "close_to_mean" or kind == "return_autocorr":
        tags.add("reversal")
    elif kind == "amihud" or kind == "volume_expansion":
        tags.add("liquidity")
    elif kind == "overnight_return_sum":
        tags.update(("reversal", "sentiment"))
    elif kind == "intraday_return_sum":
        tags |= _trend(horizon)
        tags.add("technical")
    return _ordered(tags)


def _strip_suffix(name: str, suffix: str) -> str | None:
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return None


def _fundamental_families(name: str) -> tuple[str, ...] | None:
    for suffix, table in (
        ("_growth_1y", GROWTH_ITEM_FAMILIES),
        ("_growth_3y", GROWTH_ITEM_FAMILIES),
        ("_change_to_assets_1y", SCALED_CHANGE_FAMILIES),
        ("_change_to_assets_3y", SCALED_CHANGE_FAMILIES),
    ):
        item = _strip_suffix(name, suffix)
        if item is not None:
            return _lookup(table, item, "fundamental item")
    if name.endswith("_change_5y") or name in _SPECIAL_FUNDAMENTAL:
        if name not in _SPECIAL_FUNDAMENTAL:
            return None
        return _ordered(_SPECIAL_FUNDAMENTAL[name])
    item = _strip_suffix(name, "_level")
    if item is not None:
        return _lookup(_LEVEL_FAMILIES, item, "level item")
    if name in RATIO_FAMILIES:
        return _ordered(RATIO_FAMILIES[name])
    return None


def _window(name: str) -> int | None:
    found = [int(match) for match in _WINDOW.findall(name)]
    return max(found) if found else None


def _is_price_trend(name: str) -> bool:
    return name.startswith(
        (
            "close_",
            "smooth_close",
            "slope_close",
            "weighted_mean_close",
            "rank_return",
            "ts_rank_rank_return",
        )
    )


def _price_volume_families(name: str) -> tuple[str, ...]:
    tags = {"technical"}
    if any(
        token in name
        for token in (
            "volatility",
            "true_range",
            "wick",
            "shadow",
            "absolute_deviation",
            "gap_root",
            "gap_square",
        )
    ) or re.match(r"(smooth_)?high_low_\d", name):
        tags.add("volatility")
    if "volume" in name or "amount" in name:
        tags.add("liquidity")
    if name in {"up_day_share", "smooth_up_close_ratio", "days_since_high_share"}:
        tags.add("momentum")
    elif name == "days_since_low_share":
        tags.add("reversal")
    elif name == "overnight_return":
        tags.update(("reversal", "sentiment"))
    elif name in {"directional_gap_sum", "directional_move_balance"}:
        tags.add("reversal")
    elif _is_price_trend(name):
        window = _window(name)
        if window is not None:
            tags.add("momentum" if window >= 60 else "reversal")
    return _ordered(tags)


def factor_families(name: str) -> tuple[str, ...]:
    """Return the quant families for one registered factor, in taxonomy order."""

    if name in CORE_FAMILIES:
        return _ordered(CORE_FAMILIES[name])
    if name in ACADEMIC_FAMILIES:
        return _ordered(ACADEMIC_FAMILIES[name])
    if name in RESEARCH_FAMILIES:
        return _ordered(RESEARCH_FAMILIES[name])
    if name in MINUTE_FAMILIES:
        return _ordered(MINUTE_FAMILIES[name])
    medium = _medium_families(name)
    if medium is not None:
        return medium
    fundamental = _fundamental_families(name)
    if fundamental is not None:
        return fundamental
    if name in PRICE_VOLUME_FACTORS:
        return _price_volume_families(name)
    raise FamilyError(f"Unknown factor: {name}")
