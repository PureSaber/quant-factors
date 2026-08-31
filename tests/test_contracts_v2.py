from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timezone
from decimal import Decimal

import pyarrow as pa
import pytest

from quant_factors.contracts_v2 import (
    INT64_MAX,
    AsOfSpec,
    AuxiliarySource,
    ContractViolation,
    FactorDependency,
    FactorFrameManifest,
    FactorInputRef,
    FactorSpec,
    FixedPoint,
    FrequencySpec,
    OutputField,
    SourceLineage,
    factor_frame_canonical_envelope,
    factor_frame_logical_sha256,
    jcs_bytes,
    manifest_projection_sha256,
    typed_cell,
    validate_factor_frame_envelope,
    validate_typed_cell,
    verify_factor_frame_logical_sha256,
)

ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64
ZERO_SNAPSHOT = f"sha256-{ZERO_HASH}"
ONE_SNAPSHOT = f"sha256-{ONE_HASH}"
GOLDEN_TIMESTAMP_NS = 1_788_141_600_123_456_789
GOLDEN_LOGICAL_HASH = "2c2de633c284afc6f3f1e7044ebe06373dc9c79769cd384a56ab49ad1cf8e157"
GOLDEN_PROJECTION_HASH = "c47ff18ec7feb26564e0619b6804b917adf63076b2c09f40ee5911cce7489a94"


def golden_frequency(**changes: object) -> FrequencySpec:
    values = {
        "frequency_id": "bar-1m",
        "kind": "fixed_time_bar",
        "periods_per_year": "98280",
        "calendar_id": "crypto-24x7-v1",
        "session_policy_version": "v1",
        "interval_ns": "60000000000",
        "session_rollup": None,
        "event_bar_basis": None,
        "event_bar_threshold": None,
        "market_event_types": None,
    }
    values.update(changes)
    return FrequencySpec(**values)


def golden_factor(**changes: object) -> FactorSpec:
    values = {
        "factor_id": "momentum_1p",
        "version": "1.0.0",
        "algorithm_id": "momentum",
        "input_profile": "bar",
        "dependencies": (FactorDependency("market", "close_price", "available_at"),),
        "window_periods": 1,
        "dtype": "float64",
        "annualized": False,
        "missing_policy": "null",
    }
    values.update(changes)
    return FactorSpec(**values)


def golden_output_schema(factor_id: str = "momentum_1p") -> tuple[OutputField, ...]:
    return (
        OutputField("instrument_id", "utf8", False),
        OutputField("event_time", "timestamp[ns,UTC]", False),
        OutputField("sequence", "int64", False),
        OutputField("event_id", "utf8", False),
        OutputField("source_available_at", "timestamp[ns,UTC]", False),
        OutputField(factor_id, "float64", True),
        OutputField(f"{factor_id}__available_at", "timestamp[ns,UTC]", True),
    )


def golden_manifest(**changes: object) -> FactorFrameManifest:
    values = {
        "schema_id": "puresaber.factor-frame-manifest@1.0.0",
        "certification_scope": "full-frequency-certified",
        "frequency": golden_frequency(),
        "factor_specs": (golden_factor(),),
        "as_of": AsOfSpec("source_available_at", None),
        "source_lineage": (SourceLineage("market", ZERO_SNAPSHOT, ZERO_HASH, ZERO_HASH),),
        "input_event_schemas": ({"schema_id": "puresaber.bar-event", "schema_version": "2.0.0"},),
        "auxiliary_sources": (),
        "code_version": f"commit:{'0' * 40}",
        "input_rows": "1",
        "output_rows": "1",
        "output_schema": golden_output_schema(),
        "logical_content_sha256": GOLDEN_LOGICAL_HASH,
        "physical_sha256": ZERO_HASH,
    }
    values.update(changes)
    return FactorFrameManifest(**values)


def golden_record(**changes: object) -> dict[str, object]:
    values = {
        "instrument_id": "BTC-USDT",
        "event_time": GOLDEN_TIMESTAMP_NS,
        "sequence": 7,
        "event_id": "event-0007",
        "source_available_at": GOLDEN_TIMESTAMP_NS,
        "momentum_1p": 0.1,
        "momentum_1p__available_at": GOLDEN_TIMESTAMP_NS,
    }
    values.update(changes)
    return values


def assert_code(code: str, function, *args, **kwargs) -> None:
    with pytest.raises(ContractViolation) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code


def test_golden_manifest_projection_envelope_and_logical_hash() -> None:
    manifest = golden_manifest()
    envelope = factor_frame_canonical_envelope(manifest, [golden_record()])

    assert manifest_projection_sha256(manifest) == GOLDEN_PROJECTION_HASH
    assert factor_frame_logical_sha256(manifest, [golden_record()]) == GOLDEN_LOGICAL_HASH
    assert verify_factor_frame_logical_sha256(manifest, envelope) == GOLDEN_LOGICAL_HASH
    assert envelope["records"][0][5] == {"t": "f64", "v": "3fb999999999999a"}
    assert jcs_bytes(envelope).hex().startswith("7b226d6574616461746122")


def test_hash_is_deterministic_and_semantic_changes_are_bound() -> None:
    manifest = golden_manifest()
    baseline = factor_frame_logical_sha256(manifest, [golden_record()])

    assert {factor_frame_logical_sha256(manifest, [golden_record()]) for _ in range(3)} == {
        baseline
    }
    assert factor_frame_logical_sha256(manifest, [golden_record(momentum_1p=0.2)]) != baseline
    assert (
        factor_frame_logical_sha256(
            manifest,
            [
                golden_record(
                    source_available_at=GOLDEN_TIMESTAMP_NS + 1,
                    momentum_1p__available_at=GOLDEN_TIMESTAMP_NS + 1,
                )
            ],
        )
        != baseline
    )
    frequency_change = replace(
        manifest,
        frequency=replace(manifest.frequency, periods_per_year="98281"),
    )
    assert factor_frame_logical_sha256(frequency_change, [golden_record()]) != baseline
    lineage_change = replace(
        manifest,
        source_lineage=(replace(manifest.source_lineage[0], selection_sha256=ONE_HASH),),
    )
    assert factor_frame_logical_sha256(lineage_change, [golden_record()]) != baseline
    code_change = replace(manifest, code_version=f"commit:{'1' * 40}")
    assert factor_frame_logical_sha256(code_change, [golden_record()]) != baseline


def test_contract_objects_are_frozen_and_to_contract_is_closed() -> None:
    frequency = golden_frequency()
    with pytest.raises(FrozenInstanceError):
        frequency.frequency_id = "changed"
    assert frequency.to_contract() == {
        "frequency_id": "bar-1m",
        "kind": "fixed_time_bar",
        "periods_per_year": "98280",
        "calendar_id": "crypto-24x7-v1",
        "session_policy_version": "v1",
        "interval_ns": "60000000000",
        "session_rollup": None,
        "event_bar_basis": None,
        "event_bar_threshold": None,
        "market_event_types": None,
    }
    with pytest.raises(TypeError):
        FrequencySpec(**frequency.to_contract(), unknown=True)


@pytest.mark.parametrize("periods", ["0", "1.0", "+1", "1e3", "01", "-0"])
def test_frequency_rejects_noncanonical_periods_per_year(periods: str) -> None:
    assert_code(
        "FREQUENCY_PERIODS_PER_YEAR_INVALID",
        golden_frequency,
        periods_per_year=periods,
    )


def test_frequency_conditional_fields_int64_sorting_and_uniqueness() -> None:
    assert_code(
        "FREQUENCY_INTERVAL_OUT_OF_RANGE",
        golden_frequency,
        interval_ns=str(INT64_MAX + 1),
    )
    assert_code(
        "FREQUENCY_INTERVAL_FORBIDDEN",
        golden_frequency,
        kind="session_bar",
        session_rollup="session",
    )
    event_bar = golden_frequency(
        kind="event_bar",
        interval_ns=None,
        event_bar_basis="trade_count",
        event_bar_threshold=FixedPoint("10", 0),
    )
    assert event_bar.to_contract()["event_bar_threshold"] == {"units": "10", "scale": 0}
    assert_code(
        "FREQUENCY_THRESHOLD_OUT_OF_RANGE",
        golden_frequency,
        kind="event_bar",
        interval_ns=None,
        event_bar_basis="trade_count",
        event_bar_threshold=FixedPoint("0", 0),
    )
    assert_code(
        "FREQUENCY_EVENT_TYPES_NOT_SORTED",
        golden_frequency,
        kind="market_event",
        interval_ns=None,
        market_event_types=("puresaber.trade-event", "puresaber.bbo-event"),
    )
    assert_code(
        "FREQUENCY_DUPLICATE_EVENT_TYPE",
        golden_frequency,
        kind="market_event",
        interval_ns=None,
        market_event_types=("puresaber.trade-event", "puresaber.trade-event"),
    )


def test_factor_dependency_order_and_uniqueness() -> None:
    alpha = FactorDependency("alpha", "value", "available_at")
    market = FactorDependency("market", "close", "available_at")
    assert_code(
        "FACTOR_DEPENDENCIES_NOT_SORTED",
        golden_factor,
        dependencies=(market, alpha),
    )
    assert_code(
        "DUPLICATE_FACTOR_DEPENDENCY",
        golden_factor,
        dependencies=(alpha, alpha),
    )
    assert_code("FACTOR_WINDOW_PERIODS_OUT_OF_RANGE", golden_factor, window_periods=0)


def test_as_of_int64_and_fixed_scope_constraints() -> None:
    assert AsOfSpec("fixed", str(INT64_MAX)).to_contract()["fixed_at_ns"] == str(INT64_MAX)
    assert_code("AS_OF_FIXED_AT_FORBIDDEN", AsOfSpec, "source_available_at", "0")
    assert_code("AS_OF_OUT_OF_RANGE", AsOfSpec, "fixed", str(INT64_MAX + 1))
    assert_code(
        "FIXED_AS_OF_REQUIRES_RESEARCH_RESTATED",
        golden_manifest,
        as_of=AsOfSpec("fixed", str(GOLDEN_TIMESTAMP_NS)),
    )
    restated = golden_manifest(
        certification_scope="research-restated",
        as_of=AsOfSpec("fixed", str(GOLDEN_TIMESTAMP_NS)),
    )
    assert restated.certification_scope == "research-restated"


def test_factor_input_ref_curated_and_normalized_contracts() -> None:
    curated = FactorInputRef(
        layer="curated",
        root="lake/curated",
        dataset="bars-1m",
        snapshot_id=ZERO_SNAPSHOT,
        event_schemas=None,
        market_context_snapshot_id=None,
    )
    assert curated.to_contract() == {
        "layer": "curated",
        "root": "lake/curated",
        "dataset": "bars-1m",
        "snapshot_id": ZERO_SNAPSHOT,
        "event_schemas": None,
        "market_context_snapshot_id": None,
    }
    normalized = FactorInputRef(
        layer="normalized",
        root="lake/normalized",
        dataset=None,
        snapshot_id=ZERO_SNAPSHOT,
        event_schemas=(
            {"schema_id": "puresaber.bbo-event", "schema_version": "2.0.0"},
            {"schema_id": "puresaber.trade-event", "schema_version": "2.0.0"},
        ),
        market_context_snapshot_id=ONE_SNAPSHOT,
    )
    assert normalized.to_contract()["event_schemas"] == [
        {"schema_id": "puresaber.bbo-event", "schema_version": "2.0.0"},
        {"schema_id": "puresaber.trade-event", "schema_version": "2.0.0"},
    ]


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"dataset": None}, "INPUT_REF_CURATED_DATASET_REQUIRED"),
        ({"event_schemas": ()}, "INPUT_REF_CURATED_EVENT_SCHEMAS_FORBIDDEN"),
        ({"market_context_snapshot_id": ZERO_SNAPSHOT}, "INPUT_REF_CURATED_CONTEXT_FORBIDDEN"),
        ({"snapshot_id": ZERO_HASH}, "INPUT_REF_SNAPSHOT_ID_INVALID"),
    ],
)
def test_factor_input_ref_curated_rejects_invalid_fields(changes: dict, code: str) -> None:
    values = {
        "layer": "curated",
        "root": "lake",
        "dataset": "bars",
        "snapshot_id": ZERO_SNAPSHOT,
        "event_schemas": None,
        "market_context_snapshot_id": None,
    }
    values.update(changes)
    assert_code(code, FactorInputRef, **values)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"dataset": "bars"}, "INPUT_REF_NORMALIZED_DATASET_FORBIDDEN"),
        ({"event_schemas": None}, "INPUT_REF_NORMALIZED_EVENT_SCHEMAS_REQUIRED"),
        ({"event_schemas": ()}, "INPUT_REF_NORMALIZED_EVENT_SCHEMAS_EMPTY"),
        ({"market_context_snapshot_id": None}, "INPUT_REF_NORMALIZED_CONTEXT_REQUIRED"),
        (
            {
                "event_schemas": (
                    {"schema_id": "puresaber.trade-event", "schema_version": "2.0.0"},
                    {"schema_id": "puresaber.bbo-event", "schema_version": "2.0.0"},
                )
            },
            "INPUT_REF_EVENT_SCHEMAS_NOT_SORTED",
        ),
        (
            {
                "event_schemas": (
                    {"schema_id": "puresaber.trade-event", "schema_version": "2.0.0"},
                    {"schema_id": "puresaber.trade-event", "schema_version": "2.0.0"},
                )
            },
            "INPUT_REF_DUPLICATE_EVENT_SCHEMA",
        ),
        (
            {"event_schemas": ({"schema_id": "puresaber.bar-event", "schema_version": "2.0.0"},)},
            "INPUT_REF_NORMALIZED_BAR_FORBIDDEN",
        ),
    ],
)
def test_factor_input_ref_normalized_rejects_invalid_fields(changes: dict, code: str) -> None:
    values = {
        "layer": "normalized",
        "root": "lake",
        "dataset": None,
        "snapshot_id": ZERO_SNAPSHOT,
        "event_schemas": ({"schema_id": "puresaber.trade-event", "schema_version": "2.0.0"},),
        "market_context_snapshot_id": ONE_SNAPSHOT,
    }
    values.update(changes)
    assert_code(code, FactorInputRef, **values)


def auxiliary_source(**changes: object) -> AuxiliarySource:
    values = {
        "role": "fundamental",
        "schema_id": "puresaber.fundamental",
        "schema_version": "1.0.0",
        "snapshot_id": ONE_SNAPSHOT,
        "physical_sha256": ONE_HASH,
        "logical_sha256": ONE_HASH,
        "business_key_columns": ("instrument_id",),
        "observation_time_column": "observation_time",
        "effective_from_column": "effective_from",
        "effective_to_column": "effective_to",
        "available_at_column": "available_at",
        "superseded_at_column": "superseded_at",
        "revision_column": "revision",
        "value_availability": {"book_value": "book_value_available_at"},
        "join_recipe": "latest-known-v1",
    }
    values.update(changes)
    return AuxiliarySource(**values)


def test_auxiliary_lineage_dependency_and_mapping_are_cross_bound() -> None:
    auxiliary = auxiliary_source()
    with pytest.raises(TypeError):
        auxiliary.value_availability["book_value"] = "changed"
    factor = golden_factor(
        dependencies=(FactorDependency("fundamental", "book_value", "book_value_available_at"),)
    )
    manifest = golden_manifest(
        factor_specs=(factor,),
        source_lineage=(SourceLineage("fundamental", ONE_SNAPSHOT, ONE_HASH, ZERO_HASH),),
        auxiliary_sources=(auxiliary,),
    )
    assert manifest.to_contract()["auxiliary_sources"][0]["join_recipe"] == "latest-known-v1"

    assert_code(
        "AUXILIARY_LINEAGE_MISSING",
        golden_manifest,
        factor_specs=(factor,),
        source_lineage=(SourceLineage("fundamental", ZERO_SNAPSHOT, ZERO_HASH, ZERO_HASH),),
        auxiliary_sources=(auxiliary,),
    )
    mismatched_factor = replace(
        factor,
        dependencies=(FactorDependency("fundamental", "book_value", "wrong_available_at"),),
    )
    assert_code(
        "FACTOR_AUXILIARY_MAPPING_MISMATCH",
        golden_manifest,
        factor_specs=(mismatched_factor,),
        source_lineage=(SourceLineage("fundamental", ONE_SNAPSHOT, ONE_HASH, ZERO_HASH),),
        auxiliary_sources=(auxiliary,),
    )


def test_manifest_rejects_frequency_profile_schema_and_output_mismatches() -> None:
    market_frequency = FrequencySpec(
        frequency_id="trade",
        kind="market_event",
        periods_per_year="1",
        calendar_id="crypto-24x7-v1",
        session_policy_version="v1",
        interval_ns=None,
        session_rollup=None,
        event_bar_basis=None,
        event_bar_threshold=None,
        market_event_types=("puresaber.trade-event",),
    )
    assert_code(
        "MARKET_EVENT_REQUIRES_EVENT_FACTORS",
        golden_manifest,
        frequency=market_frequency,
        input_event_schemas=(("puresaber.trade-event", "2.0.0"),),
    )
    event_factor = golden_factor(input_profile="market_event")
    assert_code(
        "MARKET_EVENT_SCHEMA_SET_MISMATCH",
        golden_manifest,
        frequency=market_frequency,
        factor_specs=(event_factor,),
        input_event_schemas=(("puresaber.bbo-event", "2.0.0"),),
    )
    fields = list(golden_output_schema())
    fields[-1], fields[-2] = fields[-2], fields[-1]
    assert_code("OUTPUT_SCHEMA_ORDER_MISMATCH", golden_manifest, output_schema=tuple(fields))
    duplicate = list(golden_output_schema())
    duplicate[-1] = replace(duplicate[-1], name="momentum_1p")
    assert_code("DUPLICATE_OUTPUT_COLUMN", golden_manifest, output_schema=tuple(duplicate))


def test_typed_cells_normalize_negative_zero_and_reject_nonfinite_values() -> None:
    assert typed_cell(-0.0, arrow_type="float64") == {
        "t": "f64",
        "v": "0000000000000000",
    }
    assert_code(
        "CELL_NEGATIVE_ZERO",
        validate_typed_cell,
        {"t": "f64", "v": "8000000000000000"},
    )
    for value in (float("nan"), float("inf"), float("-inf")):
        assert_code("CELL_NON_FINITE_FLOAT", typed_cell, value, arrow_type="float64")
    assert_code("CELL_NON_FINITE_FLOAT", typed_cell, 10**10_000, arrow_type="float64")


def test_typed_cells_cover_exact_scalar_list_struct_and_binary_types() -> None:
    cells = (
        typed_cell(None),
        typed_cell(True),
        typed_cell(7),
        typed_cell(0.5),
        typed_cell(Decimal("12.30")),
        typed_cell(date(2026, 8, 31)),
        typed_cell(""),
        typed_cell(b"\x00\xff"),
        typed_cell([1, "two"]),
        typed_cell({"flag": False, "value": 3}),
        typed_cell(255, arrow_type="uint8"),
        typed_cell(date(2026, 8, 31), arrow_type="date32"),
        typed_cell(b"\x00\xff", arrow_type="binary"),
        typed_cell(Decimal("12.30"), arrow_type="fixed"),
        typed_cell("", arrow_type="utf8"),
    )
    for cell in cells:
        validate_typed_cell(cell)
    assert cells[4] == {"t": "fixed", "u": "1230", "s": "2"}
    assert cells[7] == {"t": "binary", "v": "AP8"}
    assert cells[8]["t"] == "list"
    assert cells[9]["t"] == "struct"


def test_typed_cell_validation_rejects_noncanonical_or_unknown_encodings() -> None:
    assert_code("CELL_TIMESTAMP_TYPE", typed_cell, "2026-08-31", arrow_type="timestamp[ns,UTC]")
    assert_code("CELL_U8_OUT_OF_RANGE", typed_cell, 256, arrow_type="uint8")
    assert_code("CELL_FIXED_NON_FINITE", typed_cell, Decimal("NaN"), arrow_type="fixed")
    assert_code("CELL_UNSUPPORTED_ARROW_TYPE", typed_cell, "x", arrow_type="large_utf8")
    assert_code("MANIFEST_UNSUPPORTED_TYPE", typed_cell, object())
    assert_code("CELL_INVALID_DATE", validate_typed_cell, {"t": "date", "v": "2026-02-30"})
    assert_code("CELL_INVALID_BASE64URL", validate_typed_cell, {"t": "binary", "v": "a"})
    assert_code(
        "CELL_DUPLICATE_STRUCT_FIELD",
        validate_typed_cell,
        {"t": "struct", "v": [["x", {"t": "null"}], ["x", {"t": "null"}]]},
    )
    assert_code("CELL_UNKNOWN_TAG", validate_typed_cell, {"t": "decimal"})


def test_timestamp_preserves_nanoseconds_and_rejects_naive_or_out_of_range() -> None:
    assert typed_cell(GOLDEN_TIMESTAMP_NS, arrow_type="timestamp[ns,UTC]") == {
        "t": "ts_ns",
        "v": str(GOLDEN_TIMESTAMP_NS),
    }
    assert typed_cell(
        datetime(1970, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc),
        arrow_type="timestamp[ns,UTC]",
    ) == {"t": "ts_ns", "v": "1000"}
    assert_code(
        "CELL_NAIVE_TIMESTAMP",
        typed_cell,
        datetime(2026, 1, 1),  # noqa: DTZ001 - the contract must reject naive values
        arrow_type="timestamp[ns,UTC]",
    )
    assert_code(
        "CELL_TIMESTAMP_OUT_OF_RANGE",
        typed_cell,
        INT64_MAX + 1,
        arrow_type="timestamp[ns,UTC]",
    )
    assert_code(
        "CELL_TIMESTAMP_NOT_UTC",
        typed_cell,
        datetime.fromisoformat("2026-08-31T08:00:00+08:00"),
        arrow_type="timestamp[ns,UTC]",
    )
    assert_code("CELL_I64_OUT_OF_RANGE", typed_cell, INT64_MAX + 1, arrow_type="int64")


def test_utf16_key_order_and_number_free_jcs() -> None:
    supplementary = "\U00010000"
    private_use = "\ue000"
    cell = typed_cell({private_use: "private", supplementary: "supplementary"})
    assert [pair[0] for pair in cell["v"]] == [supplementary, private_use]
    encoded = jcs_bytes({private_use: "private", supplementary: "supplementary"}).decode()
    assert encoded.index(supplementary) < encoded.index(private_use)
    assert_code("JCS_RAW_NUMBER_FORBIDDEN", jcs_bytes, {"raw": 1})
    assert_code("JCS_INVALID_UNICODE", jcs_bytes, {"invalid": chr(0xD800)})


def test_records_require_identity_order_uniqueness_and_schema_binding() -> None:
    manifest = replace(golden_manifest(), input_rows="2", output_rows="2")
    first = golden_record(sequence=7, event_id="event-0007")
    second = golden_record(sequence=8, event_id="event-0008")
    factor_frame_canonical_envelope(manifest, [first, second])
    assert_code(
        "RECORDS_NOT_SORTED",
        factor_frame_canonical_envelope,
        manifest,
        [second, first],
    )
    assert_code(
        "DUPLICATE_RECORD_IDENTITY",
        factor_frame_canonical_envelope,
        manifest,
        [first, dict(first)],
    )
    missing = golden_record()
    missing.pop("event_id")
    assert_code(
        "RECORD_SCHEMA_MISMATCH",
        factor_frame_canonical_envelope,
        golden_manifest(),
        [missing],
    )
    wrong_schema = list(golden_output_schema())
    wrong_schema[-1] = replace(wrong_schema[-1], arrow_type="utf8")
    assert_code(
        "EXPLICIT_SCHEMA_BINDING_MISMATCH",
        factor_frame_canonical_envelope,
        golden_manifest(),
        [golden_record()],
        output_schema=wrong_schema,
    )


def test_table_schema_is_inferred_and_cross_bound() -> None:
    schema = pa.schema(
        [
            pa.field("instrument_id", pa.string(), nullable=False),
            pa.field("event_time", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("sequence", pa.int64(), nullable=False),
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("source_available_at", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("momentum_1p", pa.float64(), nullable=True),
            pa.field(
                "momentum_1p__available_at",
                pa.timestamp("ns", tz="UTC"),
                nullable=True,
            ),
        ]
    )
    table = pa.Table.from_pylist([golden_record()], schema=schema)
    assert factor_frame_logical_sha256(golden_manifest(), table) == GOLDEN_LOGICAL_HASH
    bad_table = table.set_column(0, "instrument_id", pa.array(["BTC-USDT"]))
    assert_code(
        "TABLE_SCHEMA_BINDING_MISMATCH",
        factor_frame_canonical_envelope,
        golden_manifest(),
        bad_table,
    )


def test_factor_availability_and_as_of_are_enforced() -> None:
    assert_code(
        "FACTOR_AVAILABILITY_MISSING",
        factor_frame_canonical_envelope,
        golden_manifest(),
        [golden_record(momentum_1p__available_at=None)],
    )
    assert_code(
        "FACTOR_AVAILABLE_BEFORE_SOURCE",
        factor_frame_canonical_envelope,
        golden_manifest(),
        [golden_record(momentum_1p__available_at=GOLDEN_TIMESTAMP_NS - 1)],
    )
    fixed = golden_manifest(
        certification_scope="research-restated",
        as_of=AsOfSpec("fixed", str(GOLDEN_TIMESTAMP_NS - 1)),
    )
    assert_code(
        "NON_NULL_FACTOR_AFTER_AS_OF",
        factor_frame_canonical_envelope,
        fixed,
        [golden_record()],
    )


def test_manifest_projection_and_output_schema_tampering_are_rejected() -> None:
    manifest = golden_manifest()
    envelope = factor_frame_canonical_envelope(manifest, [golden_record()])
    wrong_projection = copy.deepcopy(envelope)
    wrong_projection["metadata"]["v"][0][1]["v"] = ZERO_HASH
    assert_code(
        "ENVELOPE_MANIFEST_PROJECTION_MISMATCH",
        validate_factor_frame_envelope,
        manifest,
        wrong_projection,
    )
    wrong_schema = copy.deepcopy(envelope)
    wrong_schema["output_schema"]["v"][-1]["v"][1][1]["v"] = "utf8"
    assert_code(
        "ENVELOPE_OUTPUT_SCHEMA_MISMATCH",
        validate_factor_frame_envelope,
        manifest,
        wrong_schema,
    )
    wrong_manifest_hash = replace(manifest, logical_content_sha256=ZERO_HASH)
    assert_code(
        "MANIFEST_LOGICAL_HASH_MISMATCH",
        verify_factor_frame_logical_sha256,
        wrong_manifest_hash,
        envelope,
    )
