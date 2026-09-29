"""Point-in-time fundamental characteristics as explicit formulas.

The formulas follow the public accounting identities used by Jensen, Kelly, and
Pedersen: one-year and three-year growth, changes scaled by assets, and ratios
to sales, assets, book equity, market equity, and enterprise value. The caller
supplies the columns. On a daily panel, one year is 252 sessions and three years
is 756. On a monthly panel, pass year_lag=12 and three_year_lag=36. A
non-positive denominator stays missing.
"""

from __future__ import annotations

import pandas as pd

FUNDAMENTAL_INPUTS: dict[str, str] = {
    "total_assets": "total assets",
    "revenue": "sales or revenue",
    "book_equity": "book equity",
    "net_income": "net income",
    "net_income_extraordinary": "net income including extraordinary items",
    "current_assets": "current assets",
    "current_liabilities": "current liabilities",
    "total_liabilities": "total liabilities",
    "cash": "cash and short-term investments",
    "inventory": "inventory",
    "receivables": "receivables",
    "ppe_gross": "gross property, plant, and equipment",
    "intangibles": "intangible assets",
    "short_term_debt": "short-term debt",
    "long_term_debt": "long-term debt",
    "accounts_payable": "accounts payable",
    "income_tax_payable": "income taxes payable",
    "deferred_tax": "deferred taxes and investment tax credit",
    "cogs": "cost of goods sold",
    "sga": "selling, general, and administrative expense",
    "depreciation": "depreciation and amortization",
    "interest_expense": "interest expense",
    "pretax_income": "pretax income",
    "rd_expense": "research and development expense",
    "capex": "capital expenditure",
    "dividends": "dividends",
    "special_items": "special items",
    "extraordinary_items": "extraordinary items and discontinued operations",
    "advertising": "advertising expense",
    "staff_expense": "staff or labor expense",
    "employees": "employees",
    "preferred_stock": "preferred stock",
    "equity_issuance": "equity issuance",
    "equity_repurchase": "equity repurchase",
    "long_term_debt_issuance": "net long-term debt issuance",
    "short_term_debt_issuance": "net short-term debt issuance",
    "market_equity": "market equity",
    "operating_cash_flow": "operating cash flow",
}

DERIVED_FORMULAS: dict[str, str] = {
    "noncurrent_assets": "total_assets - current_assets",
    "noncurrent_liabilities": "total_liabilities - current_liabilities",
    "total_debt": "short_term_debt + long_term_debt",
    "gross_profit": "revenue - cogs",
    "operating_expenses": "cogs + sga",
    "ebitda": "revenue - cogs - sga",
    "ebit": "revenue - cogs - sga - depreciation",
    "operating_earnings": "revenue - cogs - sga - depreciation - interest_expense",
    "free_cash_flow": "operating_cash_flow - capex",
    "net_working_capital": "current_assets - current_liabilities",
    "current_operating_assets": "current_assets - cash",
    "current_operating_liabilities": "current_liabilities - short_term_debt",
    "current_operating_working_capital": "current_operating_assets - current_operating_liabilities",
    "noncurrent_operating_assets": "ppe_gross + intangibles",
    "noncurrent_operating_liabilities": "noncurrent_liabilities - long_term_debt",
    "net_noncurrent_operating_assets": "noncurrent_operating_assets - noncurrent_operating_liabilities",
    "operating_assets": "current_operating_assets + noncurrent_operating_assets",
    "operating_liabilities": "current_operating_liabilities + noncurrent_operating_liabilities",
    "net_operating_assets": "operating_assets - operating_liabilities",
    "financial_assets": "cash",
    "financial_liabilities": "total_debt + preferred_stock",
    "net_financial_assets": "financial_assets - financial_liabilities",
    "equity_net_issuance": "equity_issuance - equity_repurchase",
    "net_debt_issuance": "long_term_debt_issuance + short_term_debt_issuance",
    "market_enterprise_value": "market_equity + total_debt - cash",
    "nonrecurring_items": "special_items + extraordinary_items",
}


class FundamentalError(ValueError):
    """A fundamental characteristic was requested without its input columns."""


def _dependencies(name: str) -> set[str]:
    if name in FUNDAMENTAL_INPUTS:
        return {name}
    if name not in DERIVED_FORMULAS:
        raise FundamentalError(f"Unknown accounting item: {name}")
    tokens = set(DERIVED_FORMULAS[name].replace("+", " ").replace("-", " ").split())
    found: set[str] = set()
    for token in tokens:
        if token in FUNDAMENTAL_INPUTS or token in DERIVED_FORMULAS:
            found |= _dependencies(token)
    return found


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.duplicated(["symbol", "date"]).any():
        raise FundamentalError("Fundamental characteristics require unique symbol/date rows")
    out = frame.sort_values(["symbol", "date"], kind="stable").copy()
    derived = {
        "noncurrent_assets": ("total_assets", "current_assets"),
        "noncurrent_liabilities": ("total_liabilities", "current_liabilities"),
        "total_debt": ("short_term_debt", "long_term_debt"),
        "gross_profit": ("revenue", "cogs"),
        "operating_expenses": ("cogs", "sga"),
        "ebitda": ("revenue", "cogs", "sga"),
        "ebit": ("revenue", "cogs", "sga", "depreciation"),
        "operating_earnings": ("revenue", "cogs", "sga", "depreciation", "interest_expense"),
        "free_cash_flow": ("operating_cash_flow", "capex"),
        "net_working_capital": ("current_assets", "current_liabilities"),
        "current_operating_assets": ("current_assets", "cash"),
        "current_operating_liabilities": ("current_liabilities", "short_term_debt"),
        "noncurrent_operating_assets": ("ppe_gross", "intangibles"),
        "equity_net_issuance": ("equity_issuance", "equity_repurchase"),
        "net_debt_issuance": ("long_term_debt_issuance", "short_term_debt_issuance"),
        "nonrecurring_items": ("special_items", "extraordinary_items"),
    }
    for name, columns in derived.items():
        if set(columns) <= set(out.columns):
            if name == "noncurrent_assets":
                out[name] = out["total_assets"] - out["current_assets"]
            elif name == "noncurrent_liabilities":
                out[name] = out["total_liabilities"] - out["current_liabilities"]
            elif name == "total_debt":
                out[name] = out["short_term_debt"] + out["long_term_debt"]
            elif name == "gross_profit":
                out[name] = out["revenue"] - out["cogs"]
            elif name == "operating_expenses":
                out[name] = out["cogs"] + out["sga"]
            elif name == "ebitda":
                out[name] = out["revenue"] - out["cogs"] - out["sga"]
            elif name == "ebit":
                out[name] = out["revenue"] - out["cogs"] - out["sga"] - out["depreciation"]
            elif name == "operating_earnings":
                out[name] = (
                    out["revenue"]
                    - out["cogs"]
                    - out["sga"]
                    - out["depreciation"]
                    - out["interest_expense"]
                )
            elif name == "free_cash_flow":
                out[name] = out["operating_cash_flow"] - out["capex"]
            elif name == "net_working_capital":
                out[name] = out["current_assets"] - out["current_liabilities"]
            elif name == "current_operating_assets":
                out[name] = out["current_assets"] - out["cash"]
            elif name == "current_operating_liabilities":
                out[name] = out["current_liabilities"] - out["short_term_debt"]
            elif name == "noncurrent_operating_assets":
                out[name] = out["ppe_gross"] + out["intangibles"]
            elif name == "equity_net_issuance":
                out[name] = out["equity_issuance"] - out["equity_repurchase"]
            elif name == "net_debt_issuance":
                out[name] = out["long_term_debt_issuance"] + out["short_term_debt_issuance"]
            elif name == "nonrecurring_items":
                out[name] = out["special_items"] + out["extraordinary_items"]
    if {"current_operating_assets", "current_operating_liabilities"} <= set(out.columns):
        out["current_operating_working_capital"] = (
            out["current_operating_assets"] - out["current_operating_liabilities"]
        )
    if {"noncurrent_liabilities", "long_term_debt"} <= set(out.columns):
        out["noncurrent_operating_liabilities"] = (
            out["noncurrent_liabilities"] - out["long_term_debt"]
        )
    if {"noncurrent_operating_assets", "noncurrent_operating_liabilities"} <= set(out.columns):
        out["net_noncurrent_operating_assets"] = (
            out["noncurrent_operating_assets"] - out["noncurrent_operating_liabilities"]
        )
    if {"current_operating_assets", "noncurrent_operating_assets"} <= set(out.columns):
        out["operating_assets"] = (
            out["current_operating_assets"] + out["noncurrent_operating_assets"]
        )
    if {"current_operating_liabilities", "noncurrent_operating_liabilities"} <= set(out.columns):
        out["operating_liabilities"] = (
            out["current_operating_liabilities"] + out["noncurrent_operating_liabilities"]
        )
    if {"operating_assets", "operating_liabilities"} <= set(out.columns):
        out["net_operating_assets"] = out["operating_assets"] - out["operating_liabilities"]
    if "cash" in out.columns:
        out["financial_assets"] = out["cash"]
    if {"total_debt", "preferred_stock"} <= set(out.columns):
        out["financial_liabilities"] = out["total_debt"] + out["preferred_stock"]
    if {"financial_assets", "financial_liabilities"} <= set(out.columns):
        out["net_financial_assets"] = out["financial_assets"] - out["financial_liabilities"]
    if {"market_equity", "total_debt", "cash"} <= set(out.columns):
        out["market_enterprise_value"] = out["market_equity"] + out["total_debt"] - out["cash"]
    return out


def _shift(frame: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return frame.groupby("symbol", sort=False)[column].shift(periods)


def _positive_growth(frame: pd.DataFrame, column: str, lag: int) -> pd.Series:
    previous = _shift(frame, column, lag)
    return frame[column] / previous.where(previous > 0) - 1


def _change_to_assets(frame: pd.DataFrame, column: str, lag: int) -> pd.Series:
    assets = frame["total_assets"]
    return (frame[column] - _shift(frame, column, lag)) / assets.where(assets > 0)


def _ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    base = frame[denominator]
    return frame[numerator] / base.where(base > 0)


GROWTH_ITEMS = (
    "total_assets",
    "revenue",
    "current_assets",
    "noncurrent_assets",
    "total_liabilities",
    "current_liabilities",
    "noncurrent_liabilities",
    "book_equity",
    "preferred_stock",
    "total_debt",
    "cogs",
    "sga",
    "operating_expenses",
    "capex",
    "rd_expense",
    "employees",
    "inventory",
    "receivables",
    "cash",
    "ppe_gross",
    "intangibles",
    "net_income",
)

SCALED_CHANGE_ITEMS = (
    "gross_profit",
    "operating_cash_flow",
    "cash",
    "inventory",
    "receivables",
    "ppe_gross",
    "intangibles",
    "short_term_debt",
    "accounts_payable",
    "income_tax_payable",
    "long_term_debt",
    "deferred_tax",
    "current_operating_assets",
    "current_operating_liabilities",
    "current_operating_working_capital",
    "noncurrent_operating_assets",
    "noncurrent_operating_liabilities",
    "net_noncurrent_operating_assets",
    "operating_assets",
    "operating_liabilities",
    "net_operating_assets",
    "financial_assets",
    "financial_liabilities",
    "net_financial_assets",
    "ebitda",
    "ebit",
    "operating_earnings",
    "net_income",
    "depreciation",
    "free_cash_flow",
    "net_working_capital",
    "net_income_extraordinary",
    "equity_net_issuance",
    "net_debt_issuance",
    "dividends",
    "capex",
    "rd_expense",
    "advertising",
)

RATIO_SPECS = (
    ("gross_profit_to_assets", "gross_profit", "total_assets"),
    ("ebitda_to_assets", "ebitda", "total_assets"),
    ("ebit_to_assets", "ebit", "total_assets"),
    ("net_income_to_assets", "net_income", "total_assets"),
    ("operating_cash_flow_to_assets", "operating_cash_flow", "total_assets"),
    ("free_cash_flow_to_assets", "free_cash_flow", "total_assets"),
    ("capex_to_assets", "capex", "total_assets"),
    ("rd_to_assets", "rd_expense", "total_assets"),
    ("cash_to_assets", "cash", "total_assets"),
    ("inventory_to_assets", "inventory", "total_assets"),
    ("receivables_to_assets", "receivables", "total_assets"),
    ("debt_to_assets", "total_debt", "total_assets"),
    ("net_operating_assets_to_assets", "net_operating_assets", "total_assets"),
    ("gross_margin_ratio", "gross_profit", "revenue"),
    ("ebitda_margin", "ebitda", "revenue"),
    ("ebit_margin", "ebit", "revenue"),
    ("pretax_margin", "pretax_income", "revenue"),
    ("net_margin", "net_income", "revenue"),
    ("extraordinary_margin", "net_income_extraordinary", "revenue"),
    ("free_cash_flow_margin", "free_cash_flow", "revenue"),
    ("operating_cash_flow_margin", "operating_cash_flow", "revenue"),
    ("rd_to_sales", "rd_expense", "revenue"),
    ("advertising_to_sales", "advertising", "revenue"),
    ("staff_to_sales", "staff_expense", "revenue"),
    ("operating_earnings_to_book", "operating_earnings", "book_equity"),
    ("net_income_to_book", "net_income", "book_equity"),
    ("extraordinary_income_to_book", "net_income_extraordinary", "book_equity"),
    ("operating_cash_flow_to_book", "operating_cash_flow", "book_equity"),
    ("free_cash_flow_to_book", "free_cash_flow", "book_equity"),
    ("debt_to_book", "total_debt", "book_equity"),
    ("assets_to_book", "total_assets", "book_equity"),
    ("book_to_market_equity", "book_equity", "market_equity"),
    ("assets_to_market_equity", "total_assets", "market_equity"),
    ("cash_to_market_equity", "cash", "market_equity"),
    ("gross_profit_to_market_equity", "gross_profit", "market_equity"),
    ("ebitda_to_market_equity", "ebitda", "market_equity"),
    ("ebit_to_market_equity", "ebit", "market_equity"),
    ("operating_earnings_to_market_equity", "operating_earnings", "market_equity"),
    ("net_income_to_market_equity", "net_income", "market_equity"),
    ("revenue_to_market_equity", "revenue", "market_equity"),
    ("operating_cash_flow_to_market_equity", "operating_cash_flow", "market_equity"),
    ("free_cash_flow_to_market_equity", "free_cash_flow", "market_equity"),
    ("rd_to_market_equity", "rd_expense", "market_equity"),
    ("debt_to_market_equity", "total_debt", "market_equity"),
    ("dividends_to_market_equity", "dividends", "market_equity"),
    ("equity_issuance_to_market_equity", "equity_issuance", "market_equity"),
    ("equity_repurchase_to_market_equity", "equity_repurchase", "market_equity"),
    ("book_to_enterprise_value", "book_equity", "market_enterprise_value"),
    ("assets_to_enterprise_value", "total_assets", "market_enterprise_value"),
    ("cash_to_enterprise_value", "cash", "market_enterprise_value"),
    ("gross_profit_to_enterprise_value", "gross_profit", "market_enterprise_value"),
    ("ebitda_to_enterprise_value", "ebitda", "market_enterprise_value"),
    ("ebit_to_enterprise_value", "ebit", "market_enterprise_value"),
    ("revenue_to_enterprise_value", "revenue", "market_enterprise_value"),
    ("operating_cash_flow_to_enterprise_value", "operating_cash_flow", "market_enterprise_value"),
    ("free_cash_flow_to_enterprise_value", "free_cash_flow", "market_enterprise_value"),
    ("debt_to_enterprise_value", "total_debt", "market_enterprise_value"),
    ("equity_issuance_to_assets", "equity_issuance", "total_assets"),
    ("equity_repurchase_to_assets", "equity_repurchase", "total_assets"),
    ("equity_net_issuance_to_assets", "equity_net_issuance", "total_assets"),
    ("net_debt_issuance_to_assets", "net_debt_issuance", "total_assets"),
    ("dividends_to_assets", "dividends", "total_assets"),
    ("current_ratio", "current_assets", "current_liabilities"),
    ("cash_ratio", "cash", "current_liabilities"),
    ("interest_coverage", "ebit", "interest_expense"),
    ("ebitda_to_debt", "ebitda", "total_debt"),
    ("operating_cash_flow_to_current_liabilities", "operating_cash_flow", "current_liabilities"),
    ("pretax_to_extraordinary_income", "pretax_income", "net_income_extraordinary"),
)


def _catalog_entry(name: str, formula: str, columns: set[str], warmup: int) -> dict:
    return {
        "description": formula,
        "formula": formula,
        "columns": sorted(columns),
        "warmup_bars": warmup,
        "pit_required": True,
        "pit_columns": sorted(columns),
    }


def _build_catalog() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for item in GROWTH_ITEMS:
        columns = _dependencies(item)
        catalog[f"{item}_growth_1y"] = _catalog_entry(
            f"{item}_growth_1y",
            f"{item} / {item}_lag_1y - 1, if lag > 0",
            columns,
            252,
        )
        catalog[f"{item}_growth_3y"] = _catalog_entry(
            f"{item}_growth_3y",
            f"{item} / {item}_lag_3y - 1, if lag > 0",
            columns,
            756,
        )
    for item in SCALED_CHANGE_ITEMS:
        columns = _dependencies(item) | {"total_assets"}
        catalog[f"{item}_change_to_assets_1y"] = _catalog_entry(
            f"{item}_change_to_assets_1y",
            f"({item} - {item}_lag_1y) / total_assets, if total_assets > 0",
            columns,
            252,
        )
        catalog[f"{item}_change_to_assets_3y"] = _catalog_entry(
            f"{item}_change_to_assets_3y",
            f"({item} - {item}_lag_3y) / total_assets, if total_assets > 0",
            columns,
            756,
        )
    for name, numerator, denominator in RATIO_SPECS:
        columns = _dependencies(numerator) | _dependencies(denominator)
        catalog[name] = _catalog_entry(
            name,
            f"{numerator} / {denominator}, if {denominator} > 0",
            columns,
            1,
        )
    catalog["cash_accruals_to_assets"] = _catalog_entry(
        "cash_accruals_to_assets",
        "(net_income - operating_cash_flow) / total_assets, if total_assets > 0",
        {"net_income", "operating_cash_flow", "total_assets"},
        1,
    )
    catalog["tangibility"] = _catalog_entry(
        "tangibility",
        "(cash + 0.715 * receivables + 0.547 * inventory + 0.535 * ppe_gross) / total_assets",
        {"cash", "receivables", "inventory", "ppe_gross", "total_assets"},
        1,
    )
    for item, numerator, denominator in (
        ("gross_profit_to_lagged_assets", "gross_profit", "total_assets"),
        ("operating_earnings_to_lagged_book", "operating_earnings", "book_equity"),
        ("cash_flow_to_lagged_assets", "operating_cash_flow", "total_assets"),
    ):
        columns = _dependencies(numerator) | _dependencies(denominator)
        catalog[item] = _catalog_entry(
            item,
            f"{numerator} / {denominator}_lag_1y, if lag > 0",
            columns,
            252,
        )
    for item, numerator, denominator in (
        ("roe_change_5y", "net_income", "book_equity"),
        ("roa_change_5y", "net_income", "total_assets"),
        ("gross_margin_change_5y", "gross_profit", "revenue"),
        ("cash_flow_to_assets_change_5y", "operating_cash_flow", "total_assets"),
    ):
        columns = _dependencies(numerator) | _dependencies(denominator)
        catalog[item] = _catalog_entry(
            item,
            f"{numerator} / {denominator} - ({numerator} / {denominator})_lag_5y",
            columns,
            1260,
        )
    for item in (
        "total_assets",
        "revenue",
        "book_equity",
        "net_income",
        "market_enterprise_value",
        "employees",
    ):
        catalog[f"{item}_level"] = _catalog_entry(
            f"{item}_level",
            item,
            _dependencies(item),
            1,
        )
    return catalog


FUNDAMENTAL_CHARACTERISTICS = _build_catalog()


def list_fundamental_characteristics() -> dict[str, str]:
    return {name: spec["formula"] for name, spec in FUNDAMENTAL_CHARACTERISTICS.items()}


def fundamental_requirements(name: str) -> dict:
    spec = FUNDAMENTAL_CHARACTERISTICS[name]
    return {
        "columns": list(spec["columns"]),
        "warmup_bars": int(spec["warmup_bars"]),
        "pit_required": True,
        "pit_columns": list(spec["pit_columns"]),
        "dependencies": [name],
    }


def compute_fundamental_characteristics(
    frame: pd.DataFrame,
    names: list[str],
    *,
    year_lag: int = 252,
    three_year_lag: int = 756,
    five_year_lag: int = 1260,
) -> pd.DataFrame:
    """Compute requested characteristics. Lags are sessions in the supplied panel."""

    unknown = sorted(set(names) - set(FUNDAMENTAL_CHARACTERISTICS))
    if unknown:
        raise FundamentalError(f"Unknown fundamental characteristics: {unknown}")
    if min(year_lag, three_year_lag, five_year_lag) < 1:
        raise FundamentalError("lags must be positive session counts")
    prepared = _prepare(frame)
    computed: dict[str, pd.Series] = {}
    for name in names:
        missing = [
            column
            for column in FUNDAMENTAL_CHARACTERISTICS[name]["columns"]
            if column not in prepared.columns
        ]
        if missing:
            raise FundamentalError(f"Missing columns for {name}: {missing}")
        if name.endswith("_growth_1y"):
            computed[name] = _positive_growth(prepared, name[: -len("_growth_1y")], year_lag)
        elif name.endswith("_growth_3y"):
            computed[name] = _positive_growth(prepared, name[: -len("_growth_3y")], three_year_lag)
        elif name.endswith("_change_to_assets_1y"):
            computed[name] = _change_to_assets(
                prepared, name[: -len("_change_to_assets_1y")], year_lag
            )
        elif name.endswith("_change_to_assets_3y"):
            computed[name] = _change_to_assets(
                prepared, name[: -len("_change_to_assets_3y")], three_year_lag
            )
        elif name.endswith(("_to_lagged_assets", "_to_lagged_book")):
            numerator, denominator = {
                "gross_profit_to_lagged_assets": ("gross_profit", "total_assets"),
                "cash_flow_to_lagged_assets": ("operating_cash_flow", "total_assets"),
                "operating_earnings_to_lagged_book": ("operating_earnings", "book_equity"),
            }[name]
            previous = _shift(prepared, denominator, year_lag)
            computed[name] = prepared[numerator] / previous.where(previous > 0)
        elif name.endswith("_change_5y"):
            numerator, denominator = {
                "roe_change_5y": ("net_income", "book_equity"),
                "roa_change_5y": ("net_income", "total_assets"),
                "gross_margin_change_5y": ("gross_profit", "revenue"),
                "cash_flow_to_assets_change_5y": ("operating_cash_flow", "total_assets"),
            }[name]
            current = prepared[numerator] / prepared[denominator].where(prepared[denominator] > 0)
            lagged_num = _shift(prepared, numerator, five_year_lag)
            lagged_den = _shift(prepared, denominator, five_year_lag)
            computed[name] = current - lagged_num / lagged_den.where(lagged_den > 0)
        elif name == "cash_accruals_to_assets":
            assets = prepared["total_assets"]
            computed[name] = (
                prepared["net_income"] - prepared["operating_cash_flow"]
            ) / assets.where(assets > 0)
        elif name == "tangibility":
            assets = prepared["total_assets"]
            numerator = (
                prepared["cash"]
                + 0.715 * prepared["receivables"]
                + 0.547 * prepared["inventory"]
                + 0.535 * prepared["ppe_gross"]
            )
            computed[name] = numerator / assets.where(assets > 0)
        elif name.endswith("_level"):
            computed[name] = prepared[name[: -len("_level")]]
        else:
            numerator, denominator = next(
                (num, den) for spec_name, num, den in RATIO_SPECS if spec_name == name
            )
            computed[name] = _ratio(prepared, numerator, denominator)
    aligned = pd.DataFrame(computed, index=prepared.index)
    base = frame.drop(columns=[name for name in names if name in frame.columns], errors="ignore")
    return pd.concat([base, aligned.reindex(frame.index)], axis=1)
