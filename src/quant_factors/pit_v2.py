"""Verified auxiliary snapshots and deterministic point-in-time selection."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import ipc

from quant_factors.contracts_v2 import AuxiliarySource

_FACTORY_TOKEN = object()
_HASH = re.compile(r"^[0-9a-f]{64}$")


class PitError(ValueError):
    """Stable M8 auxiliary-data failure."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def arrow_table_logical_sha256(table: pa.Table) -> str:
    """Hash one Arrow table independently of its Parquet encoding."""
    if not isinstance(table, pa.Table):
        raise PitError("AUX_NOT_ARROW_TABLE")
    combined = table.combine_chunks()
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, combined.schema) as writer:
        writer.write_table(combined)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _utc_timestamp(value: Any, code: str) -> pd.Timestamp:
    try:
        timestamp = (
            pd.Timestamp(int(value), unit="ns", tz="UTC")
            if isinstance(value, Integral) and not isinstance(value, bool)
            else pd.Timestamp(value)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise PitError(code) from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise PitError(code)
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise PitError(code)
    try:
        return timestamp.tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise PitError(code) from exc


def _nullable_utc_timestamp(value: Any, code: str) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return _utc_timestamp(value, code)


def _revision(value: Any) -> int:
    if isinstance(value, bool):
        raise PitError("AUX_REVISION_OUT_OF_RANGE")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PitError("AUX_REVISION_OUT_OF_RANGE") from exc
    if parsed != value or not -(2**63) <= parsed <= 2**63 - 1:
        raise PitError("AUX_REVISION_OUT_OF_RANGE")
    return parsed


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, pd.Timestamp):
        return ("timestamp", value.value, str(value.tz))
    if hasattr(value, "as_py"):
        return _freeze(value.as_py())
    try:
        if pd.isna(value):
            return ("null",)
    except (TypeError, ValueError):
        pass
    return value


def _required_columns(spec: AuxiliarySource) -> tuple[str, ...]:
    columns = [
        *spec.business_key_columns,
        spec.observation_time_column,
        spec.effective_from_column,
        spec.effective_to_column,
        spec.available_at_column,
        spec.superseded_at_column,
        spec.revision_column,
        *spec.value_availability.keys(),
        *spec.value_availability.values(),
    ]
    return tuple(dict.fromkeys(columns))


def _validate_rows(spec: AuxiliarySource, table: pa.Table) -> None:
    missing = sorted(set(_required_columns(spec)) - set(table.column_names))
    if missing:
        raise PitError("AUX_MISSING_COLUMNS", ",".join(missing))
    identities: dict[tuple[Any, ...], Any] = {}
    for row in table.to_pylist():
        _utc_timestamp(row[spec.observation_time_column], "AUX_NAIVE_TIME")
        effective_from = _utc_timestamp(row[spec.effective_from_column], "AUX_NAIVE_TIME")
        effective_to = _nullable_utc_timestamp(row[spec.effective_to_column], "AUX_NAIVE_TIME")
        available_at = _utc_timestamp(row[spec.available_at_column], "AUX_NAIVE_TIME")
        superseded_at = _nullable_utc_timestamp(row[spec.superseded_at_column], "AUX_NAIVE_TIME")
        if effective_to is not None and effective_from >= effective_to:
            raise PitError("AUX_INVALID_INTERVAL")
        if superseded_at is not None and superseded_at <= available_at:
            raise PitError("AUX_INVALID_INTERVAL")
        revision = _revision(row[spec.revision_column])
        business_key = tuple(_freeze(row[name]) for name in spec.business_key_columns)
        identity = (spec.role, business_key, effective_from.value, revision)
        content = _freeze(row)
        if identity in identities:
            if identities[identity] == content:
                raise PitError("AUX_DUPLICATE_VERSION")
            raise PitError("AUX_REVISION_CONFLICT")
        identities[identity] = content
        for value_column, availability_column in spec.value_availability.items():
            value_available_at = _nullable_utc_timestamp(row[availability_column], "AUX_NAIVE_TIME")
            if row[value_column] is not None and value_available_at is None:
                raise PitError("AUX_VALUE_AVAILABILITY_MISSING")


@dataclass(frozen=True, eq=False)
class VerifiedAuxiliaryInput:
    """Factory-only auxiliary table bound to one frozen source contract."""

    source: AuxiliarySource
    table: pa.Table = field(repr=False)
    path: str
    _factory_token: InitVar[object] = None

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise PitError("AUX_FACTORY_REQUIRED")
        if not isinstance(self.table, pa.Table) or self.table.num_rows <= 0:
            raise PitError("AUX_EMPTY_TABLE")
        if not _HASH.fullmatch(self.source.logical_sha256):
            raise PitError("AUX_LOGICAL_HASH_INVALID")
        if arrow_table_logical_sha256(self.table) != self.source.logical_sha256:
            raise PitError("AUX_LOGICAL_HASH_MISMATCH")
        _validate_rows(self.source, self.table)

    @classmethod
    def _from_factory(
        cls, source: AuxiliarySource, table: pa.Table, path: str
    ) -> VerifiedAuxiliaryInput:
        return cls(source=source, table=table, path=path, _factory_token=_FACTORY_TOKEN)


def load_verified_auxiliary_source(
    path: str | Path, source: AuxiliarySource
) -> VerifiedAuxiliaryInput:
    """Read one immutable Parquet object once and bind both physical and logical hashes."""
    resolved = Path(path).resolve(strict=True)
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    before_stamp = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_stamp = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_stamp != after_stamp or len(payload) != before.st_size:
        raise PitError("AUX_SOURCE_CHANGED_DURING_READ")
    if hashlib.sha256(payload).hexdigest() != source.physical_sha256:
        raise PitError("AUX_PHYSICAL_HASH_MISMATCH")
    try:
        table = pq.read_table(pa.BufferReader(payload)).combine_chunks()
    except (pa.ArrowException, OSError) as exc:
        raise PitError("AUX_PARQUET_INVALID") from exc
    current = resolved.stat()
    current_stamp = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if current_stamp != after_stamp:
        raise PitError("AUX_SOURCE_CHANGED_DURING_READ")
    return VerifiedAuxiliaryInput._from_factory(source, table, str(resolved))


@dataclass(frozen=True)
class AuxiliarySelection:
    source: AuxiliarySource
    row: Mapping[str, Any]


def _role_contract(source: AuxiliarySource) -> tuple[Any, ...]:
    return (
        source.schema_id,
        source.schema_version,
        source.business_key_columns,
        source.observation_time_column,
        source.effective_from_column,
        source.effective_to_column,
        source.available_at_column,
        source.superseded_at_column,
        source.revision_column,
        tuple(source.value_availability.items()),
        source.join_recipe,
    )


def validate_auxiliary_inputs(
    inputs: Iterable[VerifiedAuxiliaryInput],
) -> tuple[VerifiedAuxiliaryInput, ...]:
    """Validate role contracts and version identity across every supplied snapshot."""
    verified_inputs = tuple(inputs)
    role_contracts: dict[str, tuple[Any, ...]] = {}
    identities: dict[tuple[Any, ...], Any] = {}
    for verified in verified_inputs:
        if not isinstance(verified, VerifiedAuxiliaryInput):
            raise PitError("AUX_VERIFIED_INPUT_REQUIRED")
        spec = verified.source
        contract = _role_contract(spec)
        if spec.role in role_contracts and role_contracts[spec.role] != contract:
            raise PitError("AUX_ROLE_CONTRACT_MISMATCH", spec.role)
        role_contracts[spec.role] = contract
        for row in verified.table.to_pylist():
            business_key = tuple(_freeze(row[name]) for name in spec.business_key_columns)
            effective_from = _utc_timestamp(row[spec.effective_from_column], "AUX_NAIVE_TIME")
            revision = _revision(row[spec.revision_column])
            identity = (spec.role, business_key, effective_from.value, revision)
            content = _freeze(row)
            if identity in identities:
                if identities[identity] == content:
                    raise PitError("AUX_DUPLICATE_VERSION")
                raise PitError("AUX_REVISION_CONFLICT")
            identities[identity] = content
    return verified_inputs


def select_auxiliary_version(
    inputs: Iterable[VerifiedAuxiliaryInput],
    *,
    role: str,
    business_key: Mapping[str, Any],
    observation_time: Any,
    row_as_of: Any,
) -> AuxiliarySelection | None:
    """Apply the frozen bitemporal ordering across every snapshot for one role."""
    observation = _utc_timestamp(observation_time, "AUX_NAIVE_TIME")
    as_of = _utc_timestamp(row_as_of, "AUX_NAIVE_TIME")
    candidates: list[tuple[int, int, AuxiliarySource, Mapping[str, Any]]] = []
    identities: dict[tuple[Any, ...], Any] = {}
    for verified in inputs:
        spec = verified.source
        if spec.role != role:
            continue
        missing_keys = [name for name in spec.business_key_columns if name not in business_key]
        if missing_keys:
            raise PitError("AUX_BUSINESS_KEY_MISSING", ",".join(missing_keys))
        expected_key = tuple(_freeze(business_key[name]) for name in spec.business_key_columns)
        for row in verified.table.to_pylist():
            actual_key = tuple(_freeze(row[name]) for name in spec.business_key_columns)
            if actual_key != expected_key:
                continue
            effective_from = _utc_timestamp(row[spec.effective_from_column], "AUX_NAIVE_TIME")
            effective_to = _nullable_utc_timestamp(row[spec.effective_to_column], "AUX_NAIVE_TIME")
            available_at = _utc_timestamp(row[spec.available_at_column], "AUX_NAIVE_TIME")
            superseded_at = _nullable_utc_timestamp(
                row[spec.superseded_at_column], "AUX_NAIVE_TIME"
            )
            revision = _revision(row[spec.revision_column])
            identity = (role, expected_key, effective_from.value, revision)
            content = _freeze(row)
            if identity in identities:
                if identities[identity] == content:
                    raise PitError("AUX_DUPLICATE_VERSION")
                raise PitError("AUX_REVISION_CONFLICT")
            identities[identity] = content
            if not (
                effective_from <= observation
                and (effective_to is None or observation < effective_to)
                and available_at <= as_of
                and (superseded_at is None or as_of < superseded_at)
            ):
                continue
            candidates.append((effective_from.value, revision, spec, row))
    if not candidates:
        return None
    _, _, source, row = max(candidates, key=lambda item: (item[0], item[1]))
    return AuxiliarySelection(source=source, row=row)
