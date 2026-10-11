"""Complete, deterministic public factor inventory without changing legacy defaults."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quant_factors.cache import SUPPORTED_ALGORITHMS
from quant_factors.core import DEFAULT_FACTORS, list_factors
from quant_factors.expressions import expression_requirements
from quant_factors.families import factor_families
from quant_factors.fundamental_characteristics import list_fundamental_characteristics
from quant_factors.medium_low import list_medium_low_factors
from quant_factors.price_volume import list_price_volume_factors
from quant_factors.research_factors import MINUTE_FACTORS, list_named_research_factors


def factor_catalog() -> dict:
    """Return all public named registries, their true dispatch paths and requirements.

    The cache consumes explicit FactorSpec algorithms, not these name registries.
    A similarly named algorithm does not establish identical numerical semantics.
    """
    groups = {
        "core": list_factors(),
        "price_volume": list_price_volume_factors(),
        "named_research": list_named_research_factors(),
        "medium_low": list_medium_low_factors(),
        "fundamental_characteristics": list_fundamental_characteristics(),
        "minute": dict(MINUTE_FACTORS),
    }
    daily = {name for group, items in groups.items() if group != "minute" for name in items}
    requirements = expression_requirements(sorted(daily))
    rows = []
    for name in sorted(daily | set(MINUTE_FACTORS)):
        memberships = [group for group, items in groups.items() if name in items]
        interfaces = []
        if name in daily:
            interfaces.append(
                {
                    "entrypoint": "quant_factors.expressions.compute_research_factors",
                    "input_kind": "daily_panel",
                    "requirements": requirements[name],
                }
            )
        if name in MINUTE_FACTORS:
            interfaces.append(
                {
                    "entrypoint": "quant_factors.research_factors.compute_minute_factors",
                    "input_kind": "ordered_intraday_bars",
                    "requirements": {
                        "columns": ["symbol", "date", "close", "volume"],
                        "optional_order_column": "time",
                        "ordering": "Within each symbol/date, chronological bars are caller-provided when time is absent.",
                        "warmup_sessions": 20 if name == "smart_money_20d" else 1,
                        "minimum_bars": 4 if name == "minute_return_skew" else 3,
                        "pit_required": False,
                        "pit_certified": False,
                    },
                }
            )
        notes = []
        if name == "turnover_20d":
            notes.append(
                "Legacy mean volume, not a turnover rate. Prefer average_volume_20d for this formula or turnover_rate_20d_v2 with its required units/PIT inputs."
            )
        elif name == "amihud_illiq_20d":
            notes.append(
                "Legacy close-times-volume proxy. amihud_illiq_20d_v2 requires actual traded amount and economic return inputs."
            )
        rows.append(
            {
                "name": name,
                "descriptions": {group: groups[group][name] for group in memberships},
                "catalogs": memberships,
                "families": list(factor_families(name)),
                "historical_default": name in DEFAULT_FACTORS,
                "interfaces": interfaces,
                "cache_named_dispatch": False,
                "notes": notes,
            }
        )
    source_names = (
        "catalog.py",
        "core.py",
        "academic.py",
        "expressions.py",
        "families.py",
        "price_volume.py",
        "price_volume_catalog.py",
        "research_factors.py",
        "medium_low.py",
        "fundamental_characteristics.py",
        "cache.py",
        "engine_v2.py",
        "contracts_v2.py",
        "pit_v2.py",
    )
    payload = {
        "schema_version": "quant.factor-catalog/v1",
        "factors": rows,
        "counts": {
            "unique_names": len(rows),
            "historical_defaults": len(DEFAULT_FACTORS),
            "by_catalog": {name: len(items) for name, items in groups.items()},
        },
        "persistent_cache": {
            "entrypoint": "quant_factors.cache.FactorCache.compute",
            "selection": "Explicit versioned FactorSpec with algorithm, dependency, frequency and as-of; no named-registry auto-dispatch.",
            "algorithms": sorted(SUPPORTED_ALGORITHMS),
            "constraints": [
                "one market dependency",
                "no auxiliary sources or fitted state",
                "full-table read/compare/serialize; factor evaluation alone is incremental",
            ],
        },
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in source_names
        },
        "claims": {
            "profitable_alpha_certified": False,
            "market_data_certified": False,
            "all_named_factors_cache_supported": False,
        },
    }
    payload["catalog_sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return payload


def catalog_text(catalog: dict) -> str:
    lines = [f"Factor catalog: {catalog['counts']['unique_names']} unique names"]
    for row in catalog["factors"]:
        paths = ", ".join(item["input_kind"] for item in row["interfaces"])
        lines.append(f"{row['name']} [{','.join(row['families'])}] ({paths})")
    lines.append(
        "Persistent cache algorithms: " + ", ".join(catalog["persistent_cache"]["algorithms"])
    )
    lines.append(
        "Cache requires explicit FactorSpec; named factors are not dispatched automatically."
    )
    return "\n".join(lines)
