import hashlib
import json

from quant_factors.catalog import catalog_text, factor_catalog
from quant_factors.cli import main
from quant_factors.core import DEFAULT_FACTORS, list_factors
from quant_factors.expressions import expression_requirements
from quant_factors.fundamental_characteristics import list_fundamental_characteristics
from quant_factors.medium_low import list_medium_low_factors
from quant_factors.price_volume import list_price_volume_factors
from quant_factors.research_factors import MINUTE_FACTORS, list_named_research_factors


def test_complete_catalog_exact_public_registry_union_and_requirements():
    catalog = factor_catalog()
    names = set().union(
        list_factors(),
        list_price_volume_factors(),
        list_named_research_factors(),
        list_medium_low_factors(),
        list_fundamental_characteristics(),
        MINUTE_FACTORS,
    )
    assert [row["name"] for row in catalog["factors"]] == sorted(names)
    assert catalog["counts"]["unique_names"] == len(names)
    assert catalog["counts"]["historical_defaults"] == 17
    for row in catalog["factors"]:
        assert row["historical_default"] == (row["name"] in DEFAULT_FACTORS)
        assert row["families"]
        assert row["cache_named_dispatch"] is False
        for interface in row["interfaces"]:
            if interface["input_kind"] == "daily_panel":
                assert (
                    interface["requirements"] == expression_requirements([row["name"]])[row["name"]]
                )
            else:
                assert row["name"] in MINUTE_FACTORS
                assert interface["entrypoint"].endswith("compute_minute_factors")
    assert not any(catalog["claims"].values())
    assert len(catalog["persistent_cache"]["algorithms"]) == 6


def test_catalog_identity_and_legacy_semantic_notes():
    catalog = factor_catalog()
    assert catalog == factor_catalog()
    digest = catalog.pop("catalog_sha256")
    raw = json.dumps(
        catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(raw).hexdigest() == digest
    assert all(len(value) == 64 for value in catalog["source_sha256"].values())
    rows = {row["name"]: row for row in catalog["factors"]}
    assert "not a turnover rate" in rows["turnover_20d"]["notes"][0]
    assert "actual traded amount" in rows["amihud_illiq_20d"]["notes"][0]
    assert rows["smart_money_20d"]["interfaces"][0]["requirements"]["warmup_sessions"] == 20


def test_cli_full_text_json_parity_and_legacy_compatibility(capsys):
    assert main(["list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == list_factors()
    assert main(["list", "--all", "--json"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert main(["list", "--all"]) == 0
    text = capsys.readouterr().out
    assert text.strip() == catalog_text(catalog)
    for row in catalog["factors"]:
        assert row["name"] + " [" in text
