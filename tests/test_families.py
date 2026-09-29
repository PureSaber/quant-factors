"""Every factor carries one or more quant-family labels."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "src" / "quant_factors"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


families = _load("families_under_test", "families.py")
research = _load("research_factors_for_families", "research_factors.py")
medium = _load("medium_low_for_families", "medium_low.py")
fundamental = _load("fundamental_for_families", "fundamental_characteristics.py")
price_volume = _load("price_volume_for_families", "price_volume_catalog.py")
academic = _load("academic_for_families", "academic.py")


def test_family_definitions_use_the_quant_taxonomy() -> None:
    assert tuple(families.FAMILY_DEFINITIONS) == (
        "value",
        "growth",
        "profitability",
        "quality",
        "accrual",
        "investment",
        "momentum",
        "reversal",
        "volatility",
        "liquidity",
        "size",
        "leverage",
        "beta",
        "seasonality",
        "financing",
        "sentiment",
        "flow",
        "technical",
    )
    assert all(definition.strip() for definition in families.FAMILY_DEFINITIONS.values())


def test_named_examples_follow_quant_definitions() -> None:
    expected = {
        "momentum_20d": ("reversal",),
        "momentum_60d": ("momentum",),
        "pe_inv": ("value",),
        "beta_252d": ("beta",),
        "book_to_price": ("value",),
        "gross_profitability": ("profitability",),
        "asset_growth_yoy": ("growth", "investment"),
        "earnings_growth_yoy": ("growth", "profitability"),
        "accruals_to_assets": ("quality", "accrual"),
        "turnover_rate_vol_20d": ("volatility", "liquidity"),
        "residual_momentum_12_1": ("momentum",),
        "same_month_return": ("seasonality",),
        "northbound_hold_change_5d": ("flow",),
        "trail_return_21d": ("reversal",),
        "trail_return_63d": ("momentum",),
        "sharpe_252d": ("momentum", "volatility"),
        "book_to_market_equity": ("value",),
        "net_income_to_assets": ("profitability",),
        "cash_accruals_to_assets": ("quality", "accrual"),
        "net_debt_issuance_to_assets": ("leverage", "financing"),
        "close_5d": ("reversal", "technical"),
        "close_100d": ("momentum", "technical"),
        "close_location_change": ("technical",),
        "close_mean_absolute_deviation": ("volatility", "technical"),
        "overnight_return": ("reversal", "sentiment", "technical"),
        "intraday_wick_leverage": ("volatility", "technical"),
        "volume_volatility": ("volatility", "liquidity", "technical"),
        "smart_money": ("flow", "technical"),
    }
    for name, labels in expected.items():
        assert families.factor_families(name) == labels


def test_every_registered_factor_has_known_families() -> None:
    core_text = (_ROOT / "core.py").read_text(encoding="utf-8")
    core_names = set(re.findall(r'"([a-z0-9_]+)": "', core_text))
    assert core_names == set(families.CORE_FAMILIES)
    assert set(academic.FACTOR_SPECS) == set(families.ACADEMIC_FAMILIES)
    assert set(research.RESEARCH_FACTORS) == set(families.RESEARCH_FAMILIES)
    assert set(research.MINUTE_FACTORS) == set(families.MINUTE_FAMILIES)
    assert set(price_volume.PRICE_VOLUME_CATALOG) == set(families.PRICE_VOLUME_FACTORS)
    assert set(fundamental.GROWTH_ITEMS) == set(families.GROWTH_ITEM_FAMILIES)
    assert set(fundamental.SCALED_CHANGE_ITEMS) == set(families.SCALED_CHANGE_FAMILIES)
    assert {spec[0] for spec in fundamental.RATIO_SPECS} == set(families.RATIO_FAMILIES)

    names = (
        list(core_names)
        + list(academic.FACTOR_SPECS)
        + list(research.RESEARCH_FACTORS)
        + list(research.MINUTE_FACTORS)
        + list(medium.MEDIUM_LOW_FACTORS)
        + list(fundamental.FUNDAMENTAL_CHARACTERISTICS)
        + list(price_volume.PRICE_VOLUME_CATALOG)
    )
    allowed = set(families.FAMILY_DEFINITIONS)
    for name in names:
        labels = families.factor_families(name)
        assert labels
        assert set(labels) <= allowed
        assert labels == tuple(family for family in families.FAMILY_DEFINITIONS if family in labels)


def test_unknown_factor_has_no_family() -> None:
    with pytest.raises(families.FamilyError, match="not_a_factor"):
        families.factor_families("not_a_factor")
