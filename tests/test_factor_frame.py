from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from quant_data_kit import (
    AssetClass,
    EventSchemaRef,
    InstrumentSpec,
    MarginMode,
    SessionPhase,
    StoragePolicy,
    TradingSession,
    create_market_context_snapshot,
    curate_session_bars_from_snapshot,
    curate_trade_bars_from_snapshot,
    curate_trade_event_bars_from_snapshot,
    write_normalized_events,
    write_raw_bytes,
)
from quant_data_kit import (
    FixedPoint as QdkFixedPoint,
)
from quant_data_kit.research_contracts_v2 import CuratedAggregation
from quant_data_kit.schemas_v2 import (
    BAR_EVENT_SCHEMA_ID,
    TRADE_EVENT_SCHEMA_ID,
    get_arrow_schema,
)

import quant_factors.factor_frame as factor_frame_module
from quant_factors.contracts_v2 import (
    AsOfSpec,
    AuxiliarySource,
    FactorDependency,
    FactorInputRef,
    FactorSpec,
    FixedPoint,
    FrequencySpec,
)
from quant_factors.factor_frame import (
    FactorFrameError,
    _validate_frequency_binding,
    compute_factor_frame,
    compute_factor_frame_from_fixture,
)
from quant_factors.pit_v2 import (
    arrow_table_logical_sha256,
    load_verified_auxiliary_source,
)

ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64
ZERO_SNAPSHOT = "sha256-" + ZERO_HASH
ONE_SNAPSHOT = "sha256-" + ONE_HASH
UTC = timezone.utc
TEST_POLICY = StoragePolicy(
    hot_quota_bytes=1024**3,
    minimum_free_bytes=1,
    minimum_free_fraction=0.000001,
)


@pytest.fixture(autouse=True)
def _certified_build_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(factor_frame_module.CODE_VERSION_ENV, "a" * 40)


def _bar_table() -> pa.Table:
    base = 1_704_067_200_000_000_000
    minute = 60_000_000_000
    rows = []
    for index, price in enumerate((100, 110, 121)):
        bar_start = base + index * minute
        bar_end = bar_start + minute
        fixed = {"units": price, "scale": 0}
        rows.append(
            {
                "event_type": "bar",
                "event_id": f"bar-{index + 1}",
                "instrument_id": "BTC-USDT",
                "event_time": bar_end,
                "received_at": bar_end,
                "available_at": bar_end,
                "source": "fixture",
                "trading_day": date(2024, 1, 1),
                "session_id": "crypto-2024-01-01",
                "sequence": index + 1,
                "bar_start": bar_start,
                "bar_end": bar_end,
                "open_price": fixed,
                "high_price": fixed,
                "low_price": fixed,
                "close_price": fixed,
                "volume": {"units": 1, "scale": 0},
                "is_complete": True,
            }
        )
    return pa.Table.from_pylist(rows, schema=get_arrow_schema(BAR_EVENT_SCHEMA_ID))


def _aggregation() -> CuratedAggregation:
    return CuratedAggregation(
        calendar_id="crypto-24x7-v1",
        session_policy_version="v1",
        kind="fixed_time_bar",
        recipe_version="fixture-1m-v1",
        market_context_snapshot_id=ONE_SNAPSHOT,
        market_context_logical_sha256=ONE_HASH,
        source_event_schemas=(EventSchemaRef("puresaber.trade-event", "2.0.0"),),
        interval_ns=60_000_000_000,
    )


def _fixture_manifest(table: pa.Table) -> dict:
    aggregation = _aggregation()
    return {
        "schema_id": "puresaber.verified-factor-input@1.0.0",
        "layer": "curated",
        "source_snapshot_id": ZERO_SNAPSHOT,
        "source_logical_sha256": ZERO_HASH,
        "selection_logical_sha256": arrow_table_logical_sha256(table),
        "event_schemas": [{"schema_id": "puresaber.bar-event", "schema_version": "2.0.0"}],
        "calendar_id": "crypto-24x7-v1",
        "session_policy_version": "v1",
        "market_context_snapshot_id": ONE_SNAPSHOT,
        "market_context_logical_sha256": ONE_HASH,
        "lineage": [
            {
                "role": "market",
                "snapshot_id": ZERO_SNAPSHOT,
                "logical_sha256": ZERO_HASH,
            },
            {
                "role": "market_context",
                "snapshot_id": ONE_SNAPSHOT,
                "logical_sha256": ONE_HASH,
            },
        ],
        "rows": str(table.num_rows),
        "arrow_schema_sha256": hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest(),
        "aggregation": aggregation.to_contract(),
    }


def _frequency(*, interval_ns: str = "60000000000") -> FrequencySpec:
    return FrequencySpec(
        frequency_id="bar-1m",
        kind="fixed_time_bar",
        periods_per_year="525600",
        calendar_id="crypto-24x7-v1",
        session_policy_version="v1",
        interval_ns=interval_ns,
        session_rollup=None,
        event_bar_basis=None,
        event_bar_threshold=None,
        market_event_types=None,
    )


def _factor() -> FactorSpec:
    return FactorSpec(
        factor_id="momentum_1p",
        version="1.0.0",
        algorithm_id="momentum",
        input_profile="bar",
        dependencies=(FactorDependency("market", "close_price", "available_at"),),
        window_periods=1,
        dtype="float64",
        annualized=False,
        missing_policy="null",
    )


def test_fixture_factor_frame_is_deterministic_and_never_market_certified() -> None:
    table = _bar_table()
    values = [
        compute_factor_frame_from_fixture(
            table,
            _fixture_manifest(table),
            _frequency(),
            [_factor()],
            as_of=AsOfSpec("source_available_at", None),
        )
        for _ in range(3)
    ]
    assert {item.manifest.logical_content_sha256 for item in values} == {
        values[0].manifest.logical_content_sha256
    }
    assert values[0].manifest.certification_scope == "fixture-certified"
    assert values[0].table.column("momentum_1p").to_pylist()[1:] == pytest.approx([0.1, 0.1])
    assert hashlib.sha256(values[0].parquet_bytes).hexdigest() == (
        values[0].manifest.physical_sha256
    )
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_PHYSICAL_HASH_MISMATCH"):
        replace(
            values[0],
            parquet_bytes=b"forged",
            _factory_token=factor_frame_module._FIXTURE_FRAME_TOKEN,
        )


def test_factor_frame_factory_scope_and_physical_logical_binding_are_unforgeable() -> None:
    table = _bar_table()
    frame = compute_factor_frame_from_fixture(
        table,
        _fixture_manifest(table),
        _frequency(),
        [_factor()],
        as_of=AsOfSpec("source_available_at", None),
    )
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_FACTORY_REQUIRED"):
        factor_frame_module.FactorFrame(
            table=frame.table,
            manifest=frame.manifest,
            _canonical_envelope=frame.canonical_envelope,
            parquet_bytes=frame.parquet_bytes,
        )
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_PARQUET_INVALID"):
        replace(
            frame,
            parquet_bytes=memoryview(frame.parquet_bytes),  # type: ignore[arg-type]
            _factory_token=factor_frame_module._FIXTURE_FRAME_TOKEN,
        )
    malformed = b"not-parquet"
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_PARQUET_INVALID"):
        replace(
            frame,
            parquet_bytes=malformed,
            manifest=replace(
                frame.manifest,
                physical_sha256=hashlib.sha256(malformed).hexdigest(),
            ),
            _factory_token=factor_frame_module._FIXTURE_FRAME_TOKEN,
        )
    forged_scope = replace(frame.manifest, certification_scope="full-frequency-certified")
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_FIXTURE_SCOPE_ESCALATION"):
        replace(
            frame,
            manifest=forged_scope,
            _factory_token=factor_frame_module._FIXTURE_FRAME_TOKEN,
        )

    changed_rows = table.to_pylist()
    changed_rows[-1]["close_price"] = {"units": 130, "scale": 0}
    changed_rows[-1]["high_price"] = {"units": 130, "scale": 0}
    changed = pa.Table.from_pylist(changed_rows, schema=table.schema)
    other = compute_factor_frame_from_fixture(
        changed,
        _fixture_manifest(changed),
        _frequency(),
        [_factor()],
        as_of=AsOfSpec("source_available_at", None),
    )
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_PHYSICAL_LOGICAL_MISMATCH"):
        factor_frame_module.FactorFrame(
            table=frame.table,
            manifest=other.manifest,
            _canonical_envelope=other.canonical_envelope,
            parquet_bytes=other.parquet_bytes,
            _factory_token=factor_frame_module._FIXTURE_FRAME_TOKEN,
        )


def test_formal_entry_uses_qdk_loader_and_refuses_arbitrary_table(monkeypatch) -> None:
    table = _bar_table()
    fixture = factor_frame_module._verified_fixture(table, _fixture_manifest(table))
    calls: list[tuple[str, str, str]] = []

    def verified_loader(root: str, dataset: str, snapshot_id: str):
        calls.append((root, dataset, snapshot_id))
        return fixture

    monkeypatch.setattr(factor_frame_module, "load_verified_curated_bars", verified_loader)
    ref = FactorInputRef(
        layer="curated",
        root="frozen-lake",
        dataset="bars-1m",
        snapshot_id=ZERO_SNAPSHOT,
        event_schemas=None,
        market_context_snapshot_id=None,
    )
    frame = compute_factor_frame(
        ref,
        _frequency(),
        [_factor()],
        as_of=AsOfSpec("source_available_at", None),
    )
    assert calls == [("frozen-lake", "bars-1m", ZERO_SNAPSHOT)]
    assert frame.manifest.certification_scope == "full-frequency-certified"
    with pytest.raises(FactorFrameError, match="FACTOR_INPUT_REF_REQUIRED"):
        compute_factor_frame(  # type: ignore[arg-type]
            table,
            _frequency(),
            [_factor()],
            as_of=AsOfSpec("source_available_at", None),
        )


def test_frequency_and_fixture_hash_mismatches_fail_closed() -> None:
    table = _bar_table()
    with pytest.raises(FactorFrameError, match="FACTOR_FREQUENCY_INTERVAL_MISMATCH"):
        compute_factor_frame_from_fixture(
            table,
            _fixture_manifest(table),
            _frequency(interval_ns="30000000000"),
            [_factor()],
            as_of=AsOfSpec("source_available_at", None),
        )
    manifest = _fixture_manifest(table)
    manifest["selection_logical_sha256"] = ZERO_HASH
    with pytest.raises(FactorFrameError, match="FIXTURE_SELECTION_HASH_MISMATCH"):
        compute_factor_frame_from_fixture(
            table,
            manifest,
            _frequency(),
            [_factor()],
            as_of=AsOfSpec("source_available_at", None),
        )


def test_fixed_fixture_is_research_restated_and_parquet_write_is_immutable(
    tmp_path: Path,
) -> None:
    table = _bar_table()
    fixed_at = str(table.column("available_at").to_pylist()[-1].value)
    frame = compute_factor_frame_from_fixture(
        table,
        _fixture_manifest(table),
        _frequency(),
        [_factor()],
        as_of=AsOfSpec("fixed", fixed_at),
    )
    assert frame.manifest.certification_scope == "research-restated"
    target = tmp_path / "frame.parquet"
    frame.write_parquet(str(target))
    assert target.read_bytes() == frame.parquet_bytes
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_OUTPUT_EXISTS"):
        frame.write_parquet(str(target))


def test_fixture_lineage_and_schema_are_cross_bound() -> None:
    table = _bar_table()
    manifest = _fixture_manifest(table)
    manifest["source_logical_sha256"] = ONE_HASH
    with pytest.raises(FactorFrameError, match="FIXTURE_MARKET_LINEAGE_MISMATCH"):
        compute_factor_frame_from_fixture(
            table,
            manifest,
            _frequency(),
            [_factor()],
            as_of=AsOfSpec("source_available_at", None),
        )


def test_real_qdk_curated_factory_runs_end_to_end(tmp_path: Path) -> None:
    session_id = "CFFEX-IF-2026-01-05-DAY"
    records = []
    for index, price in enumerate((4000, 4400, 4840), start=1):
        timestamp = f"2026-01-05T01:{29 + index:02d}:01Z"
        records.append(
            {
                "event_type": "trade",
                "event_id": f"trade-{index}",
                "instrument_id": "IF-CONT",
                "event_time": timestamp,
                "received_at": timestamp,
                "available_at": timestamp,
                "source": "cn-fixture",
                "trading_day": "2026-01-05",
                "session_id": session_id,
                "sequence": index,
                "price": {"units": price, "scale": 1},
                "quantity": {"units": 1, "scale": 0},
                "aggressor_side": "unknown",
            }
        )
    raw = write_raw_bytes(
        tmp_path,
        source="cn-fixture",
        request={"fixture": "quant-factors"},
        collected_at="2026-01-05T01:00:00Z",
        payload=b"quant-factors-m8",
        idempotency_key="quant-factors-m8",
        policy=TEST_POLICY,
    )
    normalized = write_normalized_events(
        tmp_path,
        records,
        provider="cn-fixture",
        venue="CFFEX",
        upstream_raw_references=[raw.reference()],
        policy=TEST_POLICY,
    )
    assert normalized.snapshot is not None
    context = create_market_context_snapshot(
        tmp_path,
        calendar_id="cffex-v1",
        session_policy_version="cffex-session-v1",
        instruments=[
            InstrumentSpec(
                instrument_id="IF-CONT",
                asset_class=AssetClass.FUTURE,
                product_type="index-future",
                venue="CFFEX",
                native_symbol="IF",
                settlement_currency="CNY",
                price_tick=QdkFixedPoint(2, 1),
                quantity_step=QdkFixedPoint(1, 0),
                contract_multiplier=QdkFixedPoint(300, 0),
                calendar_id="cffex-v1",
                margin_mode=MarginMode.CROSS,
                effective_from=datetime(2025, 1, 1, tzinfo=UTC),
                available_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
        ],
        sessions=[
            TradingSession(
                session_id=session_id,
                calendar_id="cffex-v1",
                venue="CFFEX",
                trading_day=date(2026, 1, 5),
                phase=SessionPhase.CONTINUOUS,
                opens_at=datetime(2026, 1, 5, 1, 30, tzinfo=UTC),
                closes_at=datetime(2026, 1, 5, 2, 0, tzinfo=UTC),
                available_at=datetime(2025, 12, 1, tzinfo=UTC),
            )
        ],
        policy=TEST_POLICY,
    )
    curated = curate_trade_bars_from_snapshot(
        tmp_path,
        normalized_snapshot_id=normalized.snapshot.snapshot_id,
        dataset="factor-bars",
        revision_id="r1",
        recipe_version="fixed-v1",
        interval=timedelta(minutes=1),
        session_starts={session_id: datetime(2026, 1, 5, 1, 30, tzinfo=UTC)},
        market_context_snapshot_id=context.snapshot_id,
        policy=TEST_POLICY,
    )
    session_curated = curate_session_bars_from_snapshot(
        tmp_path,
        normalized_snapshot_id=normalized.snapshot.snapshot_id,
        dataset="factor-session-bars",
        revision_id="r1",
        recipe_version="session-v1",
        session_rollup="session",
        market_context_snapshot_id=context.snapshot_id,
        policy=TEST_POLICY,
    )
    event_curated = curate_trade_event_bars_from_snapshot(
        tmp_path,
        normalized_snapshot_id=normalized.snapshot.snapshot_id,
        dataset="factor-event-bars",
        revision_id="r1",
        recipe_version="event-v1",
        basis="trade_count",
        threshold=QdkFixedPoint(1, 0),
        market_context_snapshot_id=context.snapshot_id,
        policy=TEST_POLICY,
    )
    frame = compute_factor_frame(
        FactorInputRef(
            layer="curated",
            root=str(tmp_path),
            dataset="factor-bars",
            snapshot_id=curated.snapshot_id,
            event_schemas=None,
            market_context_snapshot_id=None,
        ),
        FrequencySpec(
            frequency_id="bar-1m-cffex",
            kind="fixed_time_bar",
            periods_per_year="60000",
            calendar_id="cffex-v1",
            session_policy_version="cffex-session-v1",
            interval_ns="60000000000",
            session_rollup=None,
            event_bar_basis=None,
            event_bar_threshold=None,
            market_event_types=None,
        ),
        [_factor()],
        as_of=AsOfSpec("source_available_at", None),
    )
    assert frame.manifest.certification_scope == "full-frequency-certified"
    assert frame.table.column("momentum_1p").to_pylist()[1:] == pytest.approx([0.1, 0.1])

    session_frame = compute_factor_frame(
        FactorInputRef(
            layer="curated",
            root=str(tmp_path),
            dataset="factor-session-bars",
            snapshot_id=session_curated.snapshot_id,
            event_schemas=None,
            market_context_snapshot_id=None,
        ),
        FrequencySpec(
            frequency_id="session-cffex",
            kind="session_bar",
            periods_per_year="252",
            calendar_id="cffex-v1",
            session_policy_version="cffex-session-v1",
            interval_ns=None,
            session_rollup="session",
            event_bar_basis=None,
            event_bar_threshold=None,
            market_event_types=None,
        ),
        [_factor()],
        as_of=AsOfSpec("source_available_at", None),
    )
    assert session_frame.manifest.certification_scope == "full-frequency-certified"
    assert session_frame.table.num_rows == 1

    event_bar_frame = compute_factor_frame(
        FactorInputRef(
            layer="curated",
            root=str(tmp_path),
            dataset="factor-event-bars",
            snapshot_id=event_curated.snapshot_id,
            event_schemas=None,
            market_context_snapshot_id=None,
        ),
        FrequencySpec(
            frequency_id="trade-count-1-cffex",
            kind="event_bar",
            periods_per_year="60000",
            calendar_id="cffex-v1",
            session_policy_version="cffex-session-v1",
            interval_ns=None,
            session_rollup=None,
            event_bar_basis="trade_count",
            event_bar_threshold=FixedPoint("1", 0),
            market_event_types=None,
        ),
        [_factor()],
        as_of=AsOfSpec("source_available_at", None),
    )
    assert event_bar_frame.manifest.certification_scope == "full-frequency-certified"
    assert event_bar_frame.table.column("momentum_1p").to_pylist()[1:] == pytest.approx([0.1, 0.1])

    event_factor = FactorSpec(
        factor_id="trade_price_1p",
        version="1.0.0",
        algorithm_id="last_value",
        input_profile="market_event",
        dependencies=(FactorDependency("market", "price", "available_at"),),
        window_periods=1,
        dtype="float64",
        annualized=False,
        missing_policy="null",
    )
    event_frame = compute_factor_frame(
        FactorInputRef(
            layer="normalized",
            root=str(tmp_path),
            dataset=None,
            snapshot_id=normalized.snapshot.snapshot_id,
            event_schemas=({"schema_id": "puresaber.trade-event", "schema_version": "2.0.0"},),
            market_context_snapshot_id=context.snapshot_id,
        ),
        FrequencySpec(
            frequency_id="trade-event-cffex",
            kind="market_event",
            periods_per_year="60000",
            calendar_id="cffex-v1",
            session_policy_version="cffex-session-v1",
            interval_ns=None,
            session_rollup=None,
            event_bar_basis=None,
            event_bar_threshold=None,
            market_event_types=("puresaber.trade-event",),
        ),
        [event_factor],
        as_of=AsOfSpec("source_available_at", None),
    )
    assert event_frame.manifest.input_event_schemas == (("puresaber.trade-event", "2.0.0"),)
    assert event_frame.table.column("trade_price_1p").to_pylist() == pytest.approx(
        [400.0, 440.0, 484.0]
    )


def test_auxiliary_pit_is_bound_into_values_availability_and_manifest(tmp_path: Path) -> None:
    market = _bar_table()
    base = market.column("event_time").to_pylist()[0].value - 60_000_000_000
    auxiliary_table = pa.Table.from_pylist(
        [
            {
                "instrument_id": "BTC-USDT",
                "observation_time": base,
                "effective_from": base,
                "effective_to": None,
                "available_at": base,
                "superseded_at": None,
                "revision": 1,
                "pe_ratio": 10.0,
                "pe_available_at": base + 120_000_000_000,
            }
        ],
        schema=pa.schema(
            [
                pa.field("instrument_id", pa.string(), nullable=False),
                pa.field("observation_time", pa.timestamp("ns", tz="UTC"), nullable=False),
                pa.field("effective_from", pa.timestamp("ns", tz="UTC"), nullable=False),
                pa.field("effective_to", pa.timestamp("ns", tz="UTC")),
                pa.field("available_at", pa.timestamp("ns", tz="UTC"), nullable=False),
                pa.field("superseded_at", pa.timestamp("ns", tz="UTC")),
                pa.field("revision", pa.int64(), nullable=False),
                pa.field("pe_ratio", pa.float64()),
                pa.field("pe_available_at", pa.timestamp("ns", tz="UTC")),
            ]
        ),
    )
    path = tmp_path / "fundamental.parquet"
    pq.write_table(auxiliary_table, path)
    auxiliary_contract = AuxiliarySource(
        role="fundamental",
        schema_id="puresaber.fundamental-point-in-time",
        schema_version="1.0.0",
        snapshot_id="sha256-" + "2" * 64,
        physical_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        logical_sha256=arrow_table_logical_sha256(auxiliary_table),
        business_key_columns=("instrument_id",),
        observation_time_column="observation_time",
        effective_from_column="effective_from",
        effective_to_column="effective_to",
        available_at_column="available_at",
        superseded_at_column="superseded_at",
        revision_column="revision",
        value_availability={"pe_ratio": "pe_available_at"},
        join_recipe="instrument-asof-v1",
    )
    verified_auxiliary = load_verified_auxiliary_source(path, auxiliary_contract)
    factor = FactorSpec(
        factor_id="pe_1p",
        version="1.0.0",
        algorithm_id="last_value",
        input_profile="bar",
        dependencies=(FactorDependency("fundamental", "pe_ratio", "pe_available_at"),),
        window_periods=1,
        dtype="float64",
        annualized=False,
        missing_policy="null",
    )
    frame = compute_factor_frame_from_fixture(
        market,
        _fixture_manifest(market),
        _frequency(),
        [factor],
        as_of=AsOfSpec("source_available_at", None),
        auxiliary_sources=[verified_auxiliary],
    )
    assert frame.table.column("pe_1p").to_pylist() == [None, 10.0, 10.0]
    assert frame.table.column("pe_1p__available_at").to_pylist()[0].value == (
        base + 120_000_000_000
    )
    assert frame.manifest.auxiliary_sources == (auxiliary_contract,)
    assert any(item.role == "fundamental" for item in frame.manifest.source_lineage)


def test_frequency_binding_rejects_every_layer_and_aggregation_mismatch() -> None:
    table = _bar_table()
    curated = factor_frame_module._verified_fixture(table, _fixture_manifest(table))
    with pytest.raises(FactorFrameError, match="FACTOR_FREQUENCY_CONTEXT_MISMATCH"):
        _validate_frequency_binding(replace(curated, calendar_id="wrong"), _frequency())

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
        market_event_types=(TRADE_EVENT_SCHEMA_ID,),
    )
    with pytest.raises(FactorFrameError, match="MARKET_EVENT_REQUIRES_NORMALIZED_INPUT"):
        _validate_frequency_binding(curated, market_frequency)

    normalized = SimpleNamespace(
        layer="normalized",
        aggregation=None,
        calendar_id="crypto-24x7-v1",
        session_policy_version="v1",
        event_schemas=(EventSchemaRef(TRADE_EVENT_SCHEMA_ID, "2.0.0"),),
    )
    with pytest.raises(FactorFrameError, match="BAR_FREQUENCY_REQUIRES_CURATED_INPUT"):
        _validate_frequency_binding(normalized, _frequency())
    mismatched_market = replace(market_frequency, market_event_types=("puresaber.quote-event",))
    with pytest.raises(FactorFrameError, match="MARKET_EVENT_SCHEMA_SET_MISMATCH"):
        _validate_frequency_binding(normalized, mismatched_market)

    wrong_kind = replace(curated, aggregation=SimpleNamespace(kind="session_bar"))
    with pytest.raises(FactorFrameError, match="FACTOR_FREQUENCY_KIND_MISMATCH"):
        _validate_frequency_binding(wrong_kind, _frequency())

    session_frequency = FrequencySpec(
        frequency_id="session",
        kind="session_bar",
        periods_per_year="252",
        calendar_id="crypto-24x7-v1",
        session_policy_version="v1",
        interval_ns=None,
        session_rollup="session",
        event_bar_basis=None,
        event_bar_threshold=None,
        market_event_types=None,
    )
    session_input = replace(
        curated,
        aggregation=SimpleNamespace(kind="session_bar", session_rollup="trading_day"),
    )
    with pytest.raises(FactorFrameError, match="FACTOR_FREQUENCY_SESSION_ROLLUP_MISMATCH"):
        _validate_frequency_binding(session_input, session_frequency)

    event_frequency = FrequencySpec(
        frequency_id="event-2",
        kind="event_bar",
        periods_per_year="1000",
        calendar_id="crypto-24x7-v1",
        session_policy_version="v1",
        interval_ns=None,
        session_rollup=None,
        event_bar_basis="trade_count",
        event_bar_threshold=FixedPoint("2", 0),
        market_event_types=None,
    )
    event_input = replace(
        curated,
        aggregation=SimpleNamespace(
            kind="event_bar",
            event_bar_basis="trade_count",
            event_bar_threshold=None,
        ),
    )
    with pytest.raises(FactorFrameError, match="FACTOR_FREQUENCY_EVENT_THRESHOLD_MISMATCH"):
        _validate_frequency_binding(event_input, event_frequency)
    event_input = replace(
        event_input,
        aggregation=SimpleNamespace(
            kind="event_bar",
            event_bar_basis="base_volume",
            event_bar_threshold=QdkFixedPoint(3, 0),
        ),
    )
    with pytest.raises(FactorFrameError, match="FACTOR_FREQUENCY_EVENT_THRESHOLD_MISMATCH"):
        _validate_frequency_binding(event_input, event_frequency)


def test_factor_frame_post_init_cross_checks_envelope_table_and_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _bar_table()
    frame = compute_factor_frame_from_fixture(
        table,
        _fixture_manifest(table),
        _frequency(),
        [_factor()],
        as_of=AsOfSpec("source_available_at", None),
    )
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_FACTORY_REQUIRED"):
        replace(frame, table="bad")  # type: ignore[arg-type]
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_TABLE_INVALID"):
        replace(
            frame,
            table="bad",  # type: ignore[arg-type]
            _factory_token=factor_frame_module._FIXTURE_FRAME_TOKEN,
        )
    envelope = deepcopy(frame.canonical_envelope)
    envelope["metadata"]["v"][0][1]["v"] = ZERO_HASH
    assert frame.canonical_envelope["metadata"]["v"][0][1]["v"] != ZERO_HASH
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_LOGICAL_BINDING_INVALID"):
        replace(
            frame,
            _canonical_envelope=envelope,
            _factory_token=factor_frame_module._FIXTURE_FRAME_TOKEN,
        )
    changed = frame.table.set_column(
        frame.table.schema.get_field_index("momentum_1p"),
        "momentum_1p",
        pa.array([None, 0.2, 0.2], type=pa.float64()),
    )
    changed_bytes = factor_frame_module._parquet_bytes(changed)
    monkeypatch.setattr(factor_frame_module, "verify_factor_frame_logical_sha256", lambda *_: None)
    with pytest.raises(FactorFrameError, match="FACTOR_FRAME_TABLE_HASH_MISMATCH"):
        replace(
            frame,
            table=changed,
            parquet_bytes=changed_bytes,
            manifest=replace(
                frame.manifest,
                physical_sha256=hashlib.sha256(changed_bytes).hexdigest(),
            ),
            _factory_token=factor_frame_module._FIXTURE_FRAME_TOKEN,
        )


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value.pop("rows"), "FIXTURE_MANIFEST_FIELDS_INVALID"),
        (
            lambda value: value.update({"schema_id": "wrong"}),
            "FIXTURE_SCHEMA_ID_INVALID",
        ),
        (lambda value: value.update({"rows": "999"}), "FIXTURE_ROW_COUNT_MISMATCH"),
        (
            lambda value: value.update({"arrow_schema_sha256": ZERO_HASH}),
            "FIXTURE_ARROW_SCHEMA_HASH_MISMATCH",
        ),
        (
            lambda value: value.update({"event_schemas": []}),
            "FIXTURE_EVENT_SCHEMAS_NOT_CANONICAL",
        ),
        (
            lambda value: value.update(
                {"event_schemas": [{"schema_id": TRADE_EVENT_SCHEMA_ID, "schema_version": "2.0.0"}]}
            ),
            "FIXTURE_CURATED_BAR_SCHEMA_REQUIRED",
        ),
        (
            lambda value: value["lineage"].reverse(),
            "FIXTURE_LINEAGE_NOT_CANONICAL",
        ),
        (
            lambda value: value.update({"market_context_logical_sha256": ZERO_HASH}),
            "FIXTURE_CONTEXT_LINEAGE_MISMATCH",
        ),
        (
            lambda value: value.update({"aggregation": None}),
            "FIXTURE_AGGREGATION_CONTEXT_MISMATCH",
        ),
    ],
)
def test_fixture_manifest_negative_matrix(mutate, code: str) -> None:
    table = _bar_table()
    manifest = _fixture_manifest(table)
    mutate(manifest)
    with pytest.raises(FactorFrameError, match=code):
        factor_frame_module._verified_fixture(table, manifest)


def _trade_union_table() -> pa.Table:
    schema = get_arrow_schema(TRADE_EVENT_SCHEMA_ID)
    row = {
        "event_type": "trade",
        "event_id": "trade-1",
        "instrument_id": "BTC-USDT",
        "event_time": 1_704_067_260_000_000_000,
        "received_at": 1_704_067_260_000_000_000,
        "available_at": 1_704_067_260_000_000_000,
        "source": "fixture",
        "trading_day": date(2024, 1, 1),
        "session_id": "crypto-2024-01-01",
        "sequence": 1,
        "price": {"units": 100, "scale": 0},
        "quantity": {"units": 1, "scale": 0},
        "aggressor_side": "unknown",
    }
    table = pa.Table.from_pylist([row], schema=schema)
    return table.append_column(
        "event_schema_id", pa.array([TRADE_EVENT_SCHEMA_ID], type=pa.string())
    )


def _normalized_fixture_manifest(table: pa.Table) -> dict:
    manifest = _fixture_manifest(_bar_table())
    manifest.update(
        {
            "layer": "normalized",
            "selection_logical_sha256": arrow_table_logical_sha256(table),
            "event_schemas": [{"schema_id": TRADE_EVENT_SCHEMA_ID, "schema_version": "2.0.0"}],
            "rows": str(table.num_rows),
            "arrow_schema_sha256": hashlib.sha256(
                table.schema.serialize().to_pybytes()
            ).hexdigest(),
            "aggregation": None,
        }
    )
    return manifest


def test_normalized_fixture_schema_and_aggregation_fail_closed() -> None:
    table = _trade_union_table()
    manifest = _normalized_fixture_manifest(table)
    fixture = factor_frame_module._verified_fixture(table, manifest)
    assert fixture.layer == "normalized"

    no_schema_id = table.drop(["event_schema_id"])
    missing_manifest = _normalized_fixture_manifest(no_schema_id)
    with pytest.raises(FactorFrameError, match="FIXTURE_NORMALIZED_SCHEMA_ID_COLUMN_MISSING"):
        factor_frame_module._verified_fixture(no_schema_id, missing_manifest)

    wrong_ids = table.set_column(
        table.schema.get_field_index("event_schema_id"),
        "event_schema_id",
        pa.array(["puresaber.quote-event"]),
    )
    wrong_manifest = _normalized_fixture_manifest(wrong_ids)
    with pytest.raises(FactorFrameError, match="FIXTURE_NORMALIZED_SCHEMA_SET_MISMATCH"):
        factor_frame_module._verified_fixture(wrong_ids, wrong_manifest)

    no_price = table.drop(["price"])
    no_price_manifest = _normalized_fixture_manifest(no_price)
    with pytest.raises(FactorFrameError, match="FIXTURE_NORMALIZED_ARROW_SCHEMA_MISMATCH"):
        factor_frame_module._verified_fixture(no_price, no_price_manifest)

    aggregation_manifest = _normalized_fixture_manifest(table)
    aggregation_manifest["aggregation"] = _aggregation().to_contract()
    with pytest.raises(FactorFrameError, match="FIXTURE_NORMALIZED_AGGREGATION_FORBIDDEN"):
        factor_frame_module._verified_fixture(table, aggregation_manifest)

    invalid_schemas = _normalized_fixture_manifest(table)
    invalid_schemas["event_schemas"] = [
        {"schema_id": BAR_EVENT_SCHEMA_ID, "schema_version": "2.0.0"}
    ]
    with pytest.raises(FactorFrameError, match="FIXTURE_NORMALIZED_EVENT_SCHEMAS_INVALID"):
        factor_frame_module._verified_fixture(table, invalid_schemas)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda rows: rows[0].update({"is_complete": False}), "FIXTURE_BAR_INCOMPLETE"),
        (
            lambda rows: rows[0].update({"bar_start": rows[0]["bar_end"]}),
            "FIXTURE_BAR_INTERVAL_INVALID",
        ),
        (
            lambda rows: rows[0].update({"event_time": rows[0]["bar_start"]}),
            "FIXTURE_BAR_EVENT_TIME_MISMATCH",
        ),
        (
            lambda rows: rows[0].update({"received_at": rows[0]["event_time"].value - 1}),
            "FIXTURE_EVENT_TIME_ORDER_INVALID",
        ),
        (
            lambda rows: rows[0].update(
                {
                    "bar_end": rows[0]["bar_end"].value + 1,
                    "event_time": rows[0]["event_time"].value + 1,
                    "received_at": rows[0]["received_at"].value + 1,
                    "available_at": rows[0]["available_at"].value + 1,
                }
            ),
            "FIXTURE_BAR_DURATION_MISMATCH",
        ),
        (lambda rows: rows.reverse(), "FIXTURE_STREAM_NOT_ORDERED"),
        (lambda rows: rows.append(dict(rows[0])), "FIXTURE_EVENT_IDENTITY_DUPLICATE"),
        (lambda rows: rows[0].update({"event_id": ""}), "FIXTURE_EVENT_IDENTITY_INVALID"),
    ],
)
def test_fixture_bar_semantics_fail_closed(mutation, code: str) -> None:
    rows = _bar_table().to_pylist()
    mutation(rows)
    table = pa.Table.from_pylist(rows, schema=get_arrow_schema(BAR_EVENT_SCHEMA_ID))
    with pytest.raises(FactorFrameError, match=code):
        factor_frame_module._verified_fixture(table, _fixture_manifest(table))


def test_fixture_non_utc_arrow_time_is_rejected() -> None:
    table = _bar_table()
    fields = list(table.schema)
    index = table.schema.get_field_index("event_time")
    fields[index] = pa.field("event_time", pa.timestamp("ns", tz="Asia/Shanghai"), nullable=False)
    non_utc = table.cast(pa.schema(fields))
    with pytest.raises(FactorFrameError, match="FIXTURE_CURATED_ARROW_SCHEMA_MISMATCH"):
        factor_frame_module._verified_fixture(non_utc, _fixture_manifest(non_utc))


def test_fixture_normalized_sequence_and_non_self_contained_aggregations_fail_closed() -> None:
    rows = _trade_union_table().drop(["event_schema_id"]).to_pylist()
    later = dict(rows[0])
    later.update(
        {
            "event_id": "trade-2",
            "event_time": rows[0]["event_time"].value + 1,
            "received_at": rows[0]["received_at"].value + 1,
            "available_at": rows[0]["available_at"].value + 1,
            "sequence": 0,
        }
    )
    ordered_time_bad_sequence = pa.Table.from_pylist(
        [rows[0], later], schema=get_arrow_schema(TRADE_EVENT_SCHEMA_ID)
    ).append_column(
        "event_schema_id",
        pa.array([TRADE_EVENT_SCHEMA_ID, TRADE_EVENT_SCHEMA_ID], type=pa.string()),
    )
    with pytest.raises(FactorFrameError, match="FIXTURE_SEQUENCE_NOT_INCREASING"):
        factor_frame_module._verified_fixture(
            ordered_time_bad_sequence,
            _normalized_fixture_manifest(ordered_time_bad_sequence),
        )

    table = _bar_table()
    manifest = _fixture_manifest(table)
    manifest["aggregation"] = CuratedAggregation(
        calendar_id="crypto-24x7-v1",
        session_policy_version="v1",
        kind="session_bar",
        recipe_version="fixture-session-v1",
        market_context_snapshot_id=ONE_SNAPSHOT,
        market_context_logical_sha256=ONE_HASH,
        source_event_schemas=(EventSchemaRef(TRADE_EVENT_SCHEMA_ID, "2.0.0"),),
        session_rollup="session",
    ).to_contract()
    with pytest.raises(FactorFrameError, match="FIXTURE_AGGREGATION_NOT_SELF_CONTAINED"):
        factor_frame_module._verified_fixture(table, manifest)


def test_code_version_uses_real_commit_provenance_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_git_root = factor_frame_module._git_root
    actual_git_text = factor_frame_module._git_text
    monkeypatch.setenv(factor_frame_module.CODE_VERSION_ENV, "invalid")
    with pytest.raises(FactorFrameError, match="CODE_VERSION_BUILD_COMMIT_INVALID"):
        factor_frame_module._resolve_code_version()

    monkeypatch.delenv(factor_frame_module.CODE_VERSION_ENV)
    monkeypatch.setattr(factor_frame_module, "_git_root", lambda: Path("repo"))
    monkeypatch.setattr(
        factor_frame_module,
        "_git_text",
        lambda _root, *args: "b" * 40 if args[0] == "rev-parse" else "",
    )
    assert factor_frame_module._resolve_code_version() == "commit:" + "b" * 40

    monkeypatch.setattr(
        factor_frame_module,
        "_git_text",
        lambda _root, *args: "b" * 40 if args[0] == "rev-parse" else " M source.py",
    )
    with pytest.raises(FactorFrameError, match="CODE_VERSION_DIRTY"):
        factor_frame_module._resolve_code_version()

    monkeypatch.setattr(
        factor_frame_module,
        "_git_text",
        lambda _root, *_args: "not-a-commit",
    )
    with pytest.raises(FactorFrameError, match="CODE_VERSION_GIT_COMMIT_INVALID"):
        factor_frame_module._resolve_code_version()

    monkeypatch.setattr(factor_frame_module, "_git_root", lambda: None)
    distribution = SimpleNamespace(
        read_text=lambda _name: '{"vcs_info":{"commit_id":"' + "c" * 40 + '"}}'
    )
    monkeypatch.setattr(factor_frame_module.metadata, "distribution", lambda _name: distribution)
    assert factor_frame_module._resolve_code_version() == "commit:" + "c" * 40

    distribution.read_text = lambda _name: None
    with pytest.raises(FactorFrameError, match="CODE_VERSION_UNAVAILABLE"):
        factor_frame_module._resolve_code_version()

    def missing_distribution(_name: str):
        raise factor_frame_module.metadata.PackageNotFoundError

    monkeypatch.setattr(factor_frame_module.metadata, "distribution", missing_distribution)
    with pytest.raises(FactorFrameError, match="CODE_VERSION_UNAVAILABLE"):
        factor_frame_module._resolve_code_version()

    monkeypatch.setattr(factor_frame_module, "_git_root", actual_git_root)
    monkeypatch.setattr(factor_frame_module, "_git_text", actual_git_text)
    root = factor_frame_module._git_root()
    assert root is not None
    assert len(factor_frame_module._git_text(root, "rev-parse", "HEAD")) == 40


def test_git_provenance_command_failures_have_stable_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(factor_frame_module.subprocess, "run", unavailable)
    with pytest.raises(FactorFrameError, match="CODE_VERSION_GIT_UNAVAILABLE"):
        factor_frame_module._git_text(Path("repo"), "rev-parse", "HEAD")

    failed = SimpleNamespace(returncode=1, stderr="bad repository", stdout="")
    monkeypatch.setattr(factor_frame_module.subprocess, "run", lambda *_args, **_kwargs: failed)
    with pytest.raises(FactorFrameError, match="CODE_VERSION_GIT_FAILED:bad repository"):
        factor_frame_module._git_text(Path("repo"), "rev-parse", "HEAD")
