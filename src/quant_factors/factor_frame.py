"""Certified M8 FactorFrame orchestration over quant-data-kit verified inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from importlib import metadata
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, NoReturn

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from quant_data_kit import (
    EventSchemaRef,
    LineageRef,
    VerifiedFactorInput,
    load_verified_curated_bars,
    load_verified_normalized_events,
)
from quant_data_kit.research_contracts_v2 import CuratedAggregation
from quant_data_kit.schemas_v2 import (
    BAR_EVENT_SCHEMA_ID,
    BOOK_DELTA_EVENT_SCHEMA_ID,
    BOOK_SNAPSHOT_EVENT_SCHEMA_ID,
    SCHEMA_VERSION_V2,
    get_arrow_schema,
)

from quant_factors.contracts_v2 import (
    AsOfSpec,
    ContractViolation,
    FactorFrameManifest,
    FactorInputRef,
    FactorSpec,
    FixedPoint,
    FrequencySpec,
    OutputField,
    SourceLineage,
    factor_frame_canonical_envelope,
    factor_frame_logical_sha256,
    verify_factor_frame_logical_sha256,
)
from quant_factors.engine_v2 import compute_factor_table
from quant_factors.pit_v2 import VerifiedAuxiliaryInput, arrow_table_logical_sha256

FACTOR_FRAME_MANIFEST_SCHEMA_ID = "puresaber.factor-frame-manifest@1.0.0"
_ZERO_HASH = "0" * 64
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class FactorFrameError(ValueError):
    """Stable orchestration failure before a FactorFrame can be certified."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


def _git_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_text(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        raise FactorFrameError("CODE_VERSION_GIT_UNAVAILABLE") from exc
    if result.returncode != 0:
        raise FactorFrameError("CODE_VERSION_GIT_FAILED", result.stderr.strip())
    return result.stdout.strip()


def _resolve_code_version() -> str:
    """Resolve immutable code provenance; never infer a release tag from package metadata."""
    root = _git_root()
    if root is not None:
        commit = _git_text(root, "rev-parse", "HEAD")
        if not _COMMIT.fullmatch(commit):
            raise FactorFrameError("CODE_VERSION_GIT_COMMIT_INVALID")
        if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
            raise FactorFrameError("CODE_VERSION_DIRTY")
        return f"commit:{commit}"

    try:
        direct_url = metadata.distribution("quant-factors").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct_url = None
    if direct_url:
        try:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except (AttributeError, json.JSONDecodeError, TypeError) as exc:
            raise FactorFrameError("CODE_VERSION_DIRECT_URL_INVALID") from exc
        if isinstance(commit, str) and _COMMIT.fullmatch(commit):
            return f"commit:{commit}"
    raise FactorFrameError("CODE_VERSION_UNAVAILABLE")


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    )
    return sink.getvalue().to_pybytes()


def _output_fields(table: pa.Table) -> tuple[OutputField, ...]:
    aliases = {
        "string": "utf8",
        "double": "float64",
        "int64": "int64",
        "timestamp[ns, tz=UTC]": "timestamp[ns,UTC]",
    }
    fields: list[OutputField] = []
    for arrow_field in table.schema:
        arrow_type = aliases.get(str(arrow_field.type), str(arrow_field.type))
        fields.append(OutputField(arrow_field.name, arrow_type, arrow_field.nullable))
    return tuple(fields)


def _validate_frequency_binding(verified: VerifiedFactorInput, frequency: FrequencySpec) -> None:
    if (
        verified.calendar_id != frequency.calendar_id
        or verified.session_policy_version != frequency.session_policy_version
    ):
        raise FactorFrameError("FACTOR_FREQUENCY_CONTEXT_MISMATCH")
    if frequency.kind == "market_event":
        if verified.layer != "normalized" or verified.aggregation is not None:
            raise FactorFrameError("MARKET_EVENT_REQUIRES_NORMALIZED_INPUT")
        actual = tuple(item.schema_id for item in verified.event_schemas)
        if actual != tuple(frequency.market_event_types or ()):
            raise FactorFrameError("MARKET_EVENT_SCHEMA_SET_MISMATCH")
        return
    if verified.layer != "curated" or verified.aggregation is None:
        raise FactorFrameError("BAR_FREQUENCY_REQUIRES_CURATED_INPUT")
    aggregation = verified.aggregation
    if aggregation.kind != frequency.kind:
        raise FactorFrameError("FACTOR_FREQUENCY_KIND_MISMATCH")
    if frequency.kind == "fixed_time_bar":
        if str(aggregation.interval_ns) != frequency.interval_ns:
            raise FactorFrameError("FACTOR_FREQUENCY_INTERVAL_MISMATCH")
    elif frequency.kind == "session_bar":
        if aggregation.session_rollup != frequency.session_rollup:
            raise FactorFrameError("FACTOR_FREQUENCY_SESSION_ROLLUP_MISMATCH")
    else:
        threshold = frequency.event_bar_threshold
        upstream = aggregation.event_bar_threshold
        if not isinstance(threshold, FixedPoint) or upstream is None:
            raise FactorFrameError("FACTOR_FREQUENCY_EVENT_THRESHOLD_MISMATCH")
        if (
            aggregation.event_bar_basis != frequency.event_bar_basis
            or upstream.units != int(threshold.units)
            or upstream.scale != threshold.scale
        ):
            raise FactorFrameError("FACTOR_FREQUENCY_EVENT_THRESHOLD_MISMATCH")


def _load_verified_input(input_ref: FactorInputRef) -> VerifiedFactorInput:
    contract = input_ref.to_contract()
    if input_ref.layer == "curated":
        return load_verified_curated_bars(
            contract["root"], contract["dataset"], contract["snapshot_id"]
        )
    refs = tuple(
        EventSchemaRef(item["schema_id"], item["schema_version"])
        for item in contract["event_schemas"]
    )
    return load_verified_normalized_events(
        contract["root"],
        contract["snapshot_id"],
        refs,
        contract["market_context_snapshot_id"],
    )


def _source_lineage(
    verified: VerifiedFactorInput, auxiliaries: tuple[VerifiedAuxiliaryInput, ...]
) -> tuple[SourceLineage, ...]:
    values = [
        SourceLineage(
            role=item.role,
            snapshot_id=item.snapshot_id,
            logical_sha256=item.logical_sha256,
            selection_sha256=(
                verified.selection_logical_sha256 if item.role == "market" else item.logical_sha256
            ),
        )
        for item in verified.lineage
    ]
    values.extend(
        SourceLineage(
            role=item.source.role,
            snapshot_id=item.source.snapshot_id,
            logical_sha256=item.source.logical_sha256,
            selection_sha256=item.source.logical_sha256,
        )
        for item in auxiliaries
    )
    return tuple(sorted(values, key=lambda item: (item.role, item.snapshot_id)))


def _event_schemas(verified: VerifiedFactorInput) -> tuple[tuple[str, str], ...]:
    return tuple((item.schema_id, item.schema_version) for item in verified.event_schemas)


def _build_factor_frame(
    verified: VerifiedFactorInput,
    frequency: FrequencySpec,
    factors: Iterable[FactorSpec],
    *,
    as_of: AsOfSpec,
    auxiliary_sources: Iterable[VerifiedAuxiliaryInput],
) -> FactorFrame:
    certification_scope = _certification_scope(verified, as_of)
    _validate_frequency_binding(verified, frequency)
    factor_specs = tuple(factors)
    auxiliaries = tuple(auxiliary_sources)
    output = compute_factor_table(
        verified.table,
        frequency,
        factor_specs,
        as_of=as_of,
        auxiliary_sources=auxiliaries,
    )
    physical_bytes = _parquet_bytes(output)
    physical_hash = hashlib.sha256(physical_bytes).hexdigest()
    provisional = FactorFrameManifest(
        schema_id=FACTOR_FRAME_MANIFEST_SCHEMA_ID,
        certification_scope=certification_scope,
        frequency=frequency,
        factor_specs=factor_specs,
        as_of=as_of,
        source_lineage=_source_lineage(verified, auxiliaries),
        input_event_schemas=_event_schemas(verified),
        auxiliary_sources=tuple(item.source for item in auxiliaries),
        code_version=_resolve_code_version(),
        input_rows=str(verified.table.num_rows),
        output_rows=str(output.num_rows),
        output_schema=_output_fields(output),
        logical_content_sha256=_ZERO_HASH,
        physical_sha256=physical_hash,
    )
    logical_hash = factor_frame_logical_sha256(provisional, output)
    manifest = replace(provisional, logical_content_sha256=logical_hash)
    envelope = factor_frame_canonical_envelope(manifest, output)
    verify_factor_frame_logical_sha256(manifest, envelope)
    return _seal_factor_frame(
        table=output,
        manifest=manifest,
        canonical_envelope=envelope,
        parquet_bytes=physical_bytes,
        verified=verified,
        as_of=as_of,
    )


class FactorFrame:
    """Opaque immutable result; only verified formal and fixture entry points can create it."""

    __slots__ = ("_canonical_envelope", "_manifest", "_parquet_bytes", "_table")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> NoReturn:
        raise FactorFrameError("FACTOR_FRAME_FACTORY_REQUIRED")

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise FactorFrameError("FACTOR_FRAME_IMMUTABLE")

    @property
    def table(self) -> pa.Table:
        return self._table

    @property
    def manifest(self) -> FactorFrameManifest:
        return self._manifest

    @property
    def parquet_bytes(self) -> bytes:
        return self._parquet_bytes

    @property
    def canonical_envelope(self) -> dict[str, Any]:
        """Return a defensive copy so callers cannot mutate certified logical content."""
        return deepcopy(self._canonical_envelope)

    def write_parquet(self, path: str) -> None:
        """Persist exactly the bytes whose physical hash is recorded in the manifest."""
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as stream:
                stream.write(self.parquet_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            raise FactorFrameError("FACTOR_FRAME_OUTPUT_EXISTS", str(target))
        except OSError:
            target.unlink(missing_ok=True)
            raise


def _certification_scope(verified: Any, as_of: AsOfSpec) -> str:
    if type(verified) is VerifiedFactorInput:
        base_scope = "full-frequency-certified"
    elif type(verified) is _FixtureInput:
        base_scope = "fixture-certified"
    else:
        raise FactorFrameError("FACTOR_VERIFIED_INPUT_REQUIRED")
    return "research-restated" if as_of.mode == "fixed" else base_scope


def _seal_factor_frame(
    *,
    table: pa.Table,
    manifest: FactorFrameManifest,
    canonical_envelope: Mapping[str, Any],
    parquet_bytes: bytes,
    verified: Any,
    as_of: AsOfSpec,
) -> FactorFrame:
    if manifest.certification_scope != _certification_scope(verified, as_of):
        raise FactorFrameError("FACTOR_FRAME_SCOPE_MISMATCH")
    if manifest.as_of != as_of:
        raise FactorFrameError("FACTOR_FRAME_AS_OF_MISMATCH")
    if not isinstance(table, pa.Table):
        raise FactorFrameError("FACTOR_FRAME_TABLE_INVALID")
    if not isinstance(parquet_bytes, bytes):
        raise FactorFrameError("FACTOR_FRAME_PARQUET_INVALID")
    if hashlib.sha256(parquet_bytes).hexdigest() != manifest.physical_sha256:
        raise FactorFrameError("FACTOR_FRAME_PHYSICAL_HASH_MISMATCH")
    try:
        decoded = pq.read_table(pa.BufferReader(parquet_bytes)).combine_chunks()
    except (OSError, pa.ArrowException) as exc:
        raise FactorFrameError("FACTOR_FRAME_PARQUET_INVALID") from exc
    expected = table.combine_chunks()
    if decoded.schema != expected.schema or not decoded.equals(expected, check_metadata=True):
        raise FactorFrameError("FACTOR_FRAME_PHYSICAL_LOGICAL_MISMATCH")
    try:
        verify_factor_frame_logical_sha256(manifest, canonical_envelope)
    except ContractViolation as exc:
        raise FactorFrameError("FACTOR_FRAME_LOGICAL_BINDING_INVALID", exc.code) from exc
    if factor_frame_logical_sha256(manifest, table) != manifest.logical_content_sha256:
        raise FactorFrameError("FACTOR_FRAME_TABLE_HASH_MISMATCH")
    frame = object.__new__(FactorFrame)
    object.__setattr__(frame, "_table", table)
    object.__setattr__(frame, "_manifest", manifest)
    object.__setattr__(frame, "_canonical_envelope", deepcopy(canonical_envelope))
    object.__setattr__(frame, "_parquet_bytes", parquet_bytes)
    return frame


def compute_factor_frame(
    input_ref: FactorInputRef,
    frequency: FrequencySpec,
    factors: Iterable[FactorSpec],
    *,
    as_of: AsOfSpec,
    auxiliary_sources: Iterable[VerifiedAuxiliaryInput] = (),
) -> FactorFrame:
    """Certified entry point; caller-provided market tables are intentionally impossible."""
    if not isinstance(input_ref, FactorInputRef):
        raise FactorFrameError("FACTOR_INPUT_REF_REQUIRED")
    verified = _load_verified_input(input_ref)
    return _build_factor_frame(
        verified,
        frequency,
        factors,
        as_of=as_of,
        auxiliary_sources=auxiliary_sources,
    )


@dataclass(frozen=True)
class _FixtureInput:
    layer: Literal["curated", "normalized"]
    source_snapshot_id: str
    source_logical_sha256: str
    selection_logical_sha256: str
    event_schemas: tuple[EventSchemaRef, ...]
    table: pa.Table = field(repr=False)
    calendar_id: str
    session_policy_version: str
    market_context_snapshot_id: str
    market_context_logical_sha256: str
    lineage: tuple[LineageRef, ...]
    aggregation: CuratedAggregation | None


def _fixture_time_ns(table: pa.Table, row: Mapping[str, Any], column: str) -> int:
    if column not in table.column_names:
        raise FactorFrameError("FIXTURE_TIME_COLUMN_MISSING", column)
    field_type = table.schema.field(column).type
    if not pa.types.is_timestamp(field_type) or field_type.unit != "ns" or field_type.tz != "UTC":
        raise FactorFrameError("FIXTURE_TIME_NOT_UTC", column)
    try:
        timestamp = pd.Timestamp(row[column])
    except (TypeError, ValueError, OverflowError) as exc:
        raise FactorFrameError("FIXTURE_TIME_INVALID", column) from exc
    if pd.isna(timestamp) or timestamp.tz is None or str(timestamp.tz) != "UTC":
        raise FactorFrameError("FIXTURE_TIME_NOT_UTC", column)
    return int(timestamp.value)


def _validate_fixture_market_semantics(
    table: pa.Table,
    *,
    layer: Literal["curated", "normalized"],
    aggregation: CuratedAggregation | None,
) -> None:
    if (
        layer == "curated"
        and aggregation is not None
        and aggregation.kind
        in {
            "session_bar",
            "event_bar",
        }
    ):
        raise FactorFrameError("FIXTURE_AGGREGATION_NOT_SELF_CONTAINED")

    rows = table.to_pylist()
    identities: set[tuple[str, int, int, str]] = set()
    previous: dict[tuple[str, str, str, str], tuple[int, int, str]] = {}
    previous_bar_end: dict[tuple[str, str, str], int] = {}
    l2_schemas = {BOOK_SNAPSHOT_EVENT_SCHEMA_ID, BOOK_DELTA_EVENT_SCHEMA_ID}
    if layer == "normalized" and l2_schemas.intersection(
        set(table.column("event_schema_id").to_pylist())
    ):
        raise FactorFrameError("FIXTURE_NORMALIZED_SCHEMA_NOT_SELF_CONTAINED")

    for row in rows:
        event_time = _fixture_time_ns(table, row, "event_time")
        received_at = _fixture_time_ns(table, row, "received_at")
        available_at = _fixture_time_ns(table, row, "available_at")
        if not event_time <= received_at <= available_at:
            raise FactorFrameError("FIXTURE_EVENT_TIME_ORDER_INVALID")
        sequence = row.get("sequence")
        if not isinstance(sequence, Integral) or isinstance(sequence, bool):
            raise FactorFrameError("FIXTURE_SEQUENCE_INVALID")
        event_id = row.get("event_id")
        instrument_id = row.get("instrument_id")
        source = row.get("source")
        session_id = row.get("session_id")
        if not all(
            isinstance(value, str) and value
            for value in (event_id, instrument_id, source, session_id)
        ):
            raise FactorFrameError("FIXTURE_EVENT_IDENTITY_INVALID")
        identity = (instrument_id, event_time, int(sequence), event_id)
        if identity in identities:
            raise FactorFrameError("FIXTURE_EVENT_IDENTITY_DUPLICATE")
        identities.add(identity)
        schema_id = row.get("event_schema_id", BAR_EVENT_SCHEMA_ID)
        stream = (str(schema_id), source, instrument_id, session_id)
        ordering = (event_time, int(sequence), event_id)
        if stream in previous:
            if ordering <= previous[stream]:
                raise FactorFrameError("FIXTURE_STREAM_NOT_ORDERED")
            if int(sequence) <= previous[stream][1]:
                raise FactorFrameError("FIXTURE_SEQUENCE_NOT_INCREASING")
        previous[stream] = ordering

        if layer == "curated":
            if row.get("is_complete") is not True:
                raise FactorFrameError("FIXTURE_BAR_INCOMPLETE")
            bar_start = _fixture_time_ns(table, row, "bar_start")
            bar_end = _fixture_time_ns(table, row, "bar_end")
            if bar_start >= bar_end:
                raise FactorFrameError("FIXTURE_BAR_INTERVAL_INVALID")
            if event_time != bar_end:
                raise FactorFrameError("FIXTURE_BAR_EVENT_TIME_MISMATCH")
            if (
                aggregation is not None
                and aggregation.kind == "fixed_time_bar"
                and (
                    aggregation.interval_ns is None
                    or bar_end - bar_start != aggregation.interval_ns
                )
            ):
                raise FactorFrameError("FIXTURE_BAR_DURATION_MISMATCH")
            bar_stream = (source, instrument_id, session_id)
            if bar_stream in previous_bar_end and bar_start < previous_bar_end[bar_stream]:
                raise FactorFrameError("FIXTURE_BAR_OVERLAP")
            previous_bar_end[bar_stream] = bar_end


def _verified_fixture(table: pa.Table, manifest: Mapping[str, Any]) -> _FixtureInput:
    expected = {
        "schema_id",
        "layer",
        "source_snapshot_id",
        "source_logical_sha256",
        "selection_logical_sha256",
        "event_schemas",
        "calendar_id",
        "session_policy_version",
        "market_context_snapshot_id",
        "market_context_logical_sha256",
        "lineage",
        "rows",
        "arrow_schema_sha256",
        "aggregation",
    }
    if set(manifest) != expected:
        raise FactorFrameError("FIXTURE_MANIFEST_FIELDS_INVALID")
    if manifest["schema_id"] != "puresaber.verified-factor-input@1.0.0":
        raise FactorFrameError("FIXTURE_SCHEMA_ID_INVALID")
    if not isinstance(table, pa.Table) or table.num_rows <= 0:
        raise FactorFrameError("FIXTURE_TABLE_INVALID")
    if str(table.num_rows) != manifest["rows"]:
        raise FactorFrameError("FIXTURE_ROW_COUNT_MISMATCH")
    selection_hash = arrow_table_logical_sha256(table)
    if selection_hash != manifest["selection_logical_sha256"]:
        raise FactorFrameError("FIXTURE_SELECTION_HASH_MISMATCH")
    schema_hash = hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest()
    if schema_hash != manifest["arrow_schema_sha256"]:
        raise FactorFrameError("FIXTURE_ARROW_SCHEMA_HASH_MISMATCH")
    event_schemas = tuple(
        EventSchemaRef(item["schema_id"], item["schema_version"])
        for item in manifest["event_schemas"]
    )
    if not event_schemas or event_schemas != tuple(sorted(set(event_schemas))):
        raise FactorFrameError("FIXTURE_EVENT_SCHEMAS_NOT_CANONICAL")
    if manifest["layer"] == "curated":
        if event_schemas != (EventSchemaRef(BAR_EVENT_SCHEMA_ID, SCHEMA_VERSION_V2),):
            raise FactorFrameError("FIXTURE_CURATED_BAR_SCHEMA_REQUIRED")
        if table.schema != get_arrow_schema(BAR_EVENT_SCHEMA_ID):
            raise FactorFrameError("FIXTURE_CURATED_ARROW_SCHEMA_MISMATCH")
    elif manifest["layer"] == "normalized":
        if not event_schemas or any(
            item.schema_id == BAR_EVENT_SCHEMA_ID or item.schema_version != SCHEMA_VERSION_V2
            for item in event_schemas
        ):
            raise FactorFrameError("FIXTURE_NORMALIZED_EVENT_SCHEMAS_INVALID")
        if "event_schema_id" not in table.column_names:
            raise FactorFrameError("FIXTURE_NORMALIZED_SCHEMA_ID_COLUMN_MISSING")
        if set(table.column("event_schema_id").to_pylist()) != {
            item.schema_id for item in event_schemas
        }:
            raise FactorFrameError("FIXTURE_NORMALIZED_SCHEMA_SET_MISMATCH")
        rows = table.to_pylist()
        for item in event_schemas:
            expected_schema = get_arrow_schema(item.schema_id, item.schema_version)
            for expected_field in expected_schema:
                if expected_field.name not in table.column_names or (
                    table.schema.field(expected_field.name).type != expected_field.type
                ):
                    raise FactorFrameError("FIXTURE_NORMALIZED_ARROW_SCHEMA_MISMATCH")
            selected = [row for row in rows if row["event_schema_id"] == item.schema_id]
            if any(
                row[field.name] is None
                for row in selected
                for field in expected_schema
                if not field.nullable
            ):
                raise FactorFrameError("FIXTURE_NORMALIZED_REQUIRED_VALUE_MISSING")
    else:
        raise FactorFrameError("FIXTURE_LAYER_INVALID")
    lineage = tuple(
        LineageRef(item["role"], item["snapshot_id"], item["logical_sha256"])
        for item in manifest["lineage"]
    )
    if lineage != tuple(sorted(lineage)) or len(
        {(item.role, item.snapshot_id) for item in lineage}
    ) != len(lineage):
        raise FactorFrameError("FIXTURE_LINEAGE_NOT_CANONICAL")
    market_lineage = [item for item in lineage if item.role == "market"]
    if len(market_lineage) != 1 or (
        market_lineage[0].snapshot_id,
        market_lineage[0].logical_sha256,
    ) != (manifest["source_snapshot_id"], manifest["source_logical_sha256"]):
        raise FactorFrameError("FIXTURE_MARKET_LINEAGE_MISMATCH")
    context_lineage = [item for item in lineage if item.role == "market_context"]
    if len(context_lineage) != 1 or (
        context_lineage[0].snapshot_id,
        context_lineage[0].logical_sha256,
    ) != (
        manifest["market_context_snapshot_id"],
        manifest["market_context_logical_sha256"],
    ):
        raise FactorFrameError("FIXTURE_CONTEXT_LINEAGE_MISMATCH")
    aggregation = (
        CuratedAggregation.from_contract(manifest["aggregation"])
        if manifest["aggregation"] is not None
        else None
    )
    if manifest["layer"] == "curated":
        if aggregation is None or (
            aggregation.calendar_id,
            aggregation.session_policy_version,
            aggregation.market_context_snapshot_id,
            aggregation.market_context_logical_sha256,
        ) != (
            manifest["calendar_id"],
            manifest["session_policy_version"],
            manifest["market_context_snapshot_id"],
            manifest["market_context_logical_sha256"],
        ):
            raise FactorFrameError("FIXTURE_AGGREGATION_CONTEXT_MISMATCH")
    elif aggregation is not None:
        raise FactorFrameError("FIXTURE_NORMALIZED_AGGREGATION_FORBIDDEN")
    _validate_fixture_market_semantics(
        table,
        layer=manifest["layer"],
        aggregation=aggregation,
    )
    return _FixtureInput(
        layer=manifest["layer"],
        source_snapshot_id=manifest["source_snapshot_id"],
        source_logical_sha256=manifest["source_logical_sha256"],
        selection_logical_sha256=selection_hash,
        event_schemas=event_schemas,
        table=table,
        calendar_id=manifest["calendar_id"],
        session_policy_version=manifest["session_policy_version"],
        market_context_snapshot_id=manifest["market_context_snapshot_id"],
        market_context_logical_sha256=manifest["market_context_logical_sha256"],
        lineage=lineage,
        aggregation=aggregation,
    )


def compute_factor_frame_from_fixture(
    table: pa.Table,
    fixture_manifest: Mapping[str, Any],
    frequency: FrequencySpec,
    factors: Iterable[FactorSpec],
    *,
    as_of: AsOfSpec,
    auxiliary_sources: Iterable[VerifiedAuxiliaryInput] = (),
) -> FactorFrame:
    """Explicit non-market fixture path; output can never claim real-data certification."""
    fixture = _verified_fixture(table, fixture_manifest)
    return _build_factor_frame(
        fixture,  # type: ignore[arg-type] - immutable fixture mirrors the verified public view
        frequency,
        factors,
        as_of=as_of,
        auxiliary_sources=auxiliary_sources,
    )


__all__ = [
    "FACTOR_FRAME_MANIFEST_SCHEMA_ID",
    "FactorFrame",
    "FactorFrameError",
    "compute_factor_frame",
    "compute_factor_frame_from_fixture",
]
