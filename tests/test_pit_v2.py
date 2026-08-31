from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_factors.contracts_v2 import AuxiliarySource
from quant_factors.pit_v2 import (
    PitError,
    VerifiedAuxiliaryInput,
    _freeze,
    _revision,
    _utc_timestamp,
    arrow_table_logical_sha256,
    load_verified_auxiliary_source,
    select_auxiliary_version,
    validate_auxiliary_inputs,
)

UTC_NS = pa.timestamp("ns", tz="UTC")


def _table(rows: list[dict] | None = None) -> pa.Table:
    values = rows or [
        {
            "instrument_id": "AAA",
            "observation_time": 1_704_067_200_000_000_000,
            "effective_from": 1_704_067_200_000_000_000,
            "effective_to": None,
            "available_at": 1_704_153_600_000_000_000,
            "superseded_at": None,
            "revision": 1,
            "pe_ratio": 10.0,
            "pe_available_at": 1_704_153_600_000_000_000,
        }
    ]
    schema = pa.schema(
        [
            pa.field("instrument_id", pa.string(), nullable=False),
            pa.field("observation_time", UTC_NS, nullable=False),
            pa.field("effective_from", UTC_NS, nullable=False),
            pa.field("effective_to", UTC_NS),
            pa.field("available_at", UTC_NS, nullable=False),
            pa.field("superseded_at", UTC_NS),
            pa.field("revision", pa.int64(), nullable=False),
            pa.field("pe_ratio", pa.float64()),
            pa.field("pe_available_at", UTC_NS),
        ]
    )
    return pa.Table.from_pylist(values, schema=schema)


def _source(path: Path, table: pa.Table, *, role: str = "fundamental") -> AuxiliarySource:
    pq.write_table(table, path)
    return AuxiliarySource(
        role=role,
        schema_id="puresaber.fundamental-point-in-time",
        schema_version="1.0.0",
        snapshot_id="sha256-" + "1" * 64,
        physical_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        logical_sha256=arrow_table_logical_sha256(table),
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


def test_auxiliary_factory_binds_same_bytes_and_rejects_forgery(tmp_path: Path) -> None:
    table = _table()
    path = tmp_path / "fundamental.parquet"
    source = _source(path, table)
    verified = load_verified_auxiliary_source(path, source)
    assert verified.table.equals(table)
    with pytest.raises(PitError, match="AUX_FACTORY_REQUIRED"):
        VerifiedAuxiliaryInput(source=source, table=table, path=str(path))
    with pytest.raises(PitError, match="AUX_FACTORY_REQUIRED"):
        replace(verified, path="forged")

    tampered = replace(source, physical_sha256="0" * 64)
    with pytest.raises(PitError, match="AUX_PHYSICAL_HASH_MISMATCH"):
        load_verified_auxiliary_source(path, tampered)
    wrong_logical = replace(source, logical_sha256="0" * 64)
    with pytest.raises(PitError, match="AUX_LOGICAL_HASH_MISMATCH"):
        load_verified_auxiliary_source(path, wrong_logical)


def test_pit_selects_latest_effective_then_revision_and_honours_superseded(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "instrument_id": "AAA",
            "observation_time": 1_704_067_200_000_000_000,
            "effective_from": 1_704_067_200_000_000_000,
            "effective_to": None,
            "available_at": 1_704_153_600_000_000_000,
            "superseded_at": 1_704_326_400_000_000_000,
            "revision": 1,
            "pe_ratio": 10.0,
            "pe_available_at": 1_704_153_600_000_000_000,
        },
        {
            "instrument_id": "AAA",
            "observation_time": 1_704_067_200_000_000_000,
            "effective_from": 1_704_067_200_000_000_000,
            "effective_to": None,
            "available_at": 1_704_240_000_000_000_000,
            "superseded_at": None,
            "revision": 2,
            "pe_ratio": 11.0,
            "pe_available_at": 1_704_240_000_000_000_000,
        },
    ]
    table = _table(rows)
    path = tmp_path / "revisions.parquet"
    source = _source(path, table)
    verified = load_verified_auxiliary_source(path, source)
    early = select_auxiliary_version(
        [verified],
        role="fundamental",
        business_key={"instrument_id": "AAA"},
        observation_time=1_704_200_000_000_000_000,
        row_as_of=1_704_200_000_000_000_000,
    )
    assert early is not None and early.row["pe_ratio"] == 10.0
    revised = select_auxiliary_version(
        [verified],
        role="fundamental",
        business_key={"instrument_id": "AAA"},
        observation_time=1_704_300_000_000_000_000,
        row_as_of=1_704_300_000_000_000_000,
    )
    assert revised is not None and revised.row["pe_ratio"] == 11.0
    assert (
        select_auxiliary_version(
            [verified],
            role="fundamental",
            business_key={"instrument_id": "BBB"},
            observation_time=1_704_300_000_000_000_000,
            row_as_of=1_704_300_000_000_000_000,
        )
        is None
    )


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"effective_to": 1_704_067_200_000_000_000}, "AUX_INVALID_INTERVAL"),
        ({"superseded_at": 1_704_153_600_000_000_000}, "AUX_INVALID_INTERVAL"),
    ],
)
def test_auxiliary_invalid_intervals_fail_closed(tmp_path: Path, mutation: dict, code: str) -> None:
    row = _table().to_pylist()[0]
    row.update(mutation)
    table = _table([row])
    path = tmp_path / f"{code}.parquet"
    source = _source(path, table)
    with pytest.raises(PitError, match=code):
        load_verified_auxiliary_source(path, source)


def test_duplicate_and_conflicting_revisions_have_stable_codes(tmp_path: Path) -> None:
    base = _table().to_pylist()[0]
    duplicate_table = _table([base, dict(base)])
    duplicate_path = tmp_path / "duplicate.parquet"
    duplicate_source = _source(duplicate_path, duplicate_table)
    with pytest.raises(PitError, match="AUX_DUPLICATE_VERSION"):
        load_verified_auxiliary_source(duplicate_path, duplicate_source)

    conflict = dict(base)
    conflict["pe_ratio"] = 12.0
    conflict_table = _table([base, conflict])
    conflict_path = tmp_path / "conflict.parquet"
    conflict_source = _source(conflict_path, conflict_table)
    with pytest.raises(PitError, match="AUX_REVISION_CONFLICT"):
        load_verified_auxiliary_source(conflict_path, conflict_source)


def test_pit_scalar_validators_reject_ambiguous_values() -> None:
    with pytest.raises(PitError, match="AUX_NOT_ARROW_TABLE"):
        arrow_table_logical_sha256("not-arrow")  # type: ignore[arg-type]
    for value in (True, "bad", 2**63):
        with pytest.raises(PitError, match="AUX_REVISION_OUT_OF_RANGE"):
            _revision(value)
    for value in ("bad", "2026-01-01T00:00:00", "2026-01-01T08:00:00+08:00"):
        with pytest.raises(PitError, match="AUX_NAIVE_TIME"):
            _utc_timestamp(value, "AUX_NAIVE_TIME")
    assert _freeze([pa.scalar(1), None]) == (1, ("null",))


def test_auxiliary_schema_empty_and_value_availability_fail_closed(tmp_path: Path) -> None:
    table = _table()
    missing = table.drop(["pe_available_at"])
    missing_path = tmp_path / "missing.parquet"
    missing_source = _source(missing_path, missing)
    with pytest.raises(PitError, match="AUX_MISSING_COLUMNS"):
        load_verified_auxiliary_source(missing_path, missing_source)

    empty = table.slice(0, 0)
    empty_path = tmp_path / "empty.parquet"
    empty_source = _source(empty_path, empty)
    with pytest.raises(PitError, match="AUX_EMPTY_TABLE"):
        load_verified_auxiliary_source(empty_path, empty_source)

    row = table.to_pylist()[0]
    row["pe_available_at"] = None
    no_availability = _table([row])
    unavailable_path = tmp_path / "unavailable.parquet"
    unavailable_source = _source(unavailable_path, no_availability)
    with pytest.raises(PitError, match="AUX_VALUE_AVAILABILITY_MISSING"):
        load_verified_auxiliary_source(unavailable_path, unavailable_source)


def test_malformed_parquet_and_both_toctou_windows_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = tmp_path / "malformed.parquet"
    malformed.write_bytes(b"not-parquet")
    malformed_source = AuxiliarySource(
        role="fundamental",
        schema_id="puresaber.fundamental-point-in-time",
        schema_version="1.0.0",
        snapshot_id="sha256-" + "3" * 64,
        physical_sha256=hashlib.sha256(malformed.read_bytes()).hexdigest(),
        logical_sha256="0" * 64,
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
    with pytest.raises(PitError, match="AUX_PARQUET_INVALID"):
        load_verified_auxiliary_source(malformed, malformed_source)

    table = _table()
    during_read = tmp_path / "during-read.parquet"
    during_source = _source(during_read, table)
    original_read_bytes = Path.read_bytes

    def changing_read(path: Path) -> bytes:
        payload = original_read_bytes(path)
        path.write_bytes(payload + b"changed")
        return payload

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    with pytest.raises(PitError, match="AUX_SOURCE_CHANGED_DURING_READ"):
        load_verified_auxiliary_source(during_read, during_source)
    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)

    after_decode = tmp_path / "after-decode.parquet"
    after_source = _source(after_decode, table)
    original_read_table = pq.read_table

    def changing_decode(*args, **kwargs):
        decoded = original_read_table(*args, **kwargs)
        after_decode.write_bytes(after_decode.read_bytes() + b"changed")
        return decoded

    monkeypatch.setattr(pq, "read_table", changing_decode)
    with pytest.raises(PitError, match="AUX_SOURCE_CHANGED_DURING_READ"):
        load_verified_auxiliary_source(after_decode, after_source)


def test_selection_role_key_window_and_cross_snapshot_conflicts(tmp_path: Path) -> None:
    table = _table()
    first_path = tmp_path / "first.parquet"
    first_source = _source(first_path, table)
    first = load_verified_auxiliary_source(first_path, first_source)
    assert (
        select_auxiliary_version(
            [first],
            role="other",
            business_key={"instrument_id": "AAA"},
            observation_time=1_704_200_000_000_000_000,
            row_as_of=1_704_200_000_000_000_000,
        )
        is None
    )
    with pytest.raises(PitError, match="AUX_BUSINESS_KEY_MISSING"):
        select_auxiliary_version(
            [first],
            role="fundamental",
            business_key={},
            observation_time=1_704_200_000_000_000_000,
            row_as_of=1_704_200_000_000_000_000,
        )
    assert (
        select_auxiliary_version(
            [first],
            role="fundamental",
            business_key={"instrument_id": "AAA"},
            observation_time=1_700_000_000_000_000_000,
            row_as_of=1_704_200_000_000_000_000,
        )
        is None
    )

    second_path = tmp_path / "second.parquet"
    second_source = replace(_source(second_path, table), snapshot_id="sha256-" + "4" * 64)
    second = load_verified_auxiliary_source(second_path, second_source)
    with pytest.raises(PitError, match="AUX_DUPLICATE_VERSION"):
        select_auxiliary_version(
            [first, second],
            role="fundamental",
            business_key={"instrument_id": "AAA"},
            observation_time=1_704_200_000_000_000_000,
            row_as_of=1_704_200_000_000_000_000,
        )

    changed_row = table.to_pylist()[0]
    changed_row["pe_ratio"] = 12.0
    changed_table = _table([changed_row])
    conflict_path = tmp_path / "cross-conflict.parquet"
    conflict_source = replace(
        _source(conflict_path, changed_table), snapshot_id="sha256-" + "5" * 64
    )
    conflict = load_verified_auxiliary_source(conflict_path, conflict_source)
    with pytest.raises(PitError, match="AUX_REVISION_CONFLICT"):
        select_auxiliary_version(
            [first, conflict],
            role="fundamental",
            business_key={"instrument_id": "AAA"},
            observation_time=1_704_200_000_000_000_000,
            row_as_of=1_704_200_000_000_000_000,
        )


def test_auxiliary_set_is_validated_before_any_market_row_queries(tmp_path: Path) -> None:
    table = _table()
    first_path = tmp_path / "first-global.parquet"
    first = load_verified_auxiliary_source(first_path, _source(first_path, table))

    second_path = tmp_path / "second-global.parquet"
    second_source = replace(
        _source(second_path, table),
        snapshot_id="sha256-" + "6" * 64,
    )
    second = load_verified_auxiliary_source(second_path, second_source)
    with pytest.raises(PitError, match="AUX_DUPLICATE_VERSION"):
        validate_auxiliary_inputs([first, second])

    mismatched_path = tmp_path / "mismatched-role.parquet"
    mismatched_source = replace(
        _source(mismatched_path, table),
        snapshot_id="sha256-" + "7" * 64,
        join_recipe="different-asof-v2",
    )
    mismatched = load_verified_auxiliary_source(mismatched_path, mismatched_source)
    with pytest.raises(PitError, match="AUX_ROLE_CONTRACT_MISMATCH"):
        validate_auxiliary_inputs([first, mismatched])

    assert validate_auxiliary_inputs([first]) == (first,)
    with pytest.raises(PitError, match="AUX_VERIFIED_INPUT_REQUIRED"):
        validate_auxiliary_inputs([object()])  # type: ignore[list-item]
