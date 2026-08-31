"""PureSaber M8 factor contracts and canonical logical hashing.

This module deliberately contains no loaders.  It accepts already verified
lineage and explicit records (or a table-shaped object) and binds them to the
closed M8 contracts and their deterministic logical representation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
_INT32_MAX = 2**31 - 1
_INTEGER_RANGES = {
    "i8": (-(2**7), 2**7 - 1),
    "i16": (-(2**15), 2**15 - 1),
    "i32": (-(2**31), 2**31 - 1),
    "i64": (INT64_MIN, INT64_MAX),
    "u8": (0, 2**8 - 1),
    "u16": (0, 2**16 - 1),
    "u32": (0, 2**32 - 1),
    "u64": (0, 2**64 - 1),
}
_IDENTITY_COLUMNS = (
    "instrument_id",
    "event_time",
    "sequence",
    "event_id",
    "source_available_at",
)
_MANIFEST_SCHEMA_ID = "puresaber.factor-frame-manifest@1.0.0"
_CANONICAL_SCHEMA_ID = "puresaber.factor-frame-canonical@1.0.0"
_SNAPSHOT_PATTERN = re.compile(r"sha256-[0-9a-f]{64}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SCHEMA_ID_PATTERN = re.compile(r"puresaber\.[a-z0-9._-]+")
_SEMVER_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_FACTOR_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_MAPPING_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_CANONICAL_INTEGER_PATTERN = re.compile(r"(?:0|[1-9][0-9]*|-[1-9][0-9]*)")
_POSITIVE_INTEGER_PATTERN = re.compile(r"[1-9][0-9]*")
_NONNEGATIVE_INTEGER_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)")
_POSITIVE_DECIMAL_PATTERN = re.compile(r"(?:[1-9][0-9]*(?:\.[0-9]*[1-9])?|0\.[0-9]*[1-9])")

EventSchemaInput = Mapping[str, str] | tuple[str, str]
ValueAvailabilityInput = Mapping[str, str] | Sequence[tuple[str, str]]


class ContractViolation(ValueError):
    """A versioned M8 invariant failed with a stable machine-readable code."""

    def __init__(self, code: str, detail: object | None = None) -> None:
        self.code = code
        self.detail = detail
        message = code if detail is None else f"{code}:{detail}"
        super().__init__(message)


def _require(condition: bool, code: str, detail: object | None = None) -> None:
    if not condition:
        raise ContractViolation(code, detail)


def _require_string(value: object, code: str) -> str:
    _require(isinstance(value, str) and bool(value), code)
    return value


def _require_pattern(value: object, pattern: re.Pattern[str], code: str) -> str:
    text = _require_string(value, code)
    _require(pattern.fullmatch(text) is not None, code)
    return text


def _parse_integer_text(value: object, minimum: int, maximum: int, code: str) -> int:
    _require(isinstance(value, str), code)
    _require(_CANONICAL_INTEGER_PATTERN.fullmatch(value) is not None, code)
    parsed = int(value)
    _require(minimum <= parsed <= maximum, code)
    return parsed


def _parse_nonnegative_count(value: object, code: str) -> int:
    _require(isinstance(value, str), code)
    _require(_NONNEGATIVE_INTEGER_PATTERN.fullmatch(value) is not None, code)
    parsed = int(value)
    _require(parsed <= INT64_MAX, code)
    return parsed


def _utf16_sort_key(value: str) -> bytes:
    return value.encode("utf-16-be", errors="surrogatepass")


def _canonical_order_key(value: str | tuple[str, ...]) -> tuple[object, ...]:
    if isinstance(value, str):
        return (_utf16_sort_key(value),)
    return tuple(_utf16_sort_key(item) for item in value)


def _require_unique_sorted(
    values: Sequence[Any],
    *,
    key,
    duplicate_code: str,
    sorted_code: str,
) -> None:
    identities = [key(value) for value in values]
    _require(len(identities) == len(set(identities)), duplicate_code)
    _require(
        identities == sorted(identities, key=_canonical_order_key),
        sorted_code,
    )


def _event_schema_ref(value: EventSchemaInput, *, code: str) -> tuple[str, str]:
    if isinstance(value, Mapping):
        _require(set(value) == {"schema_id", "schema_version"}, f"{code}_FIELDS")
        schema_id = value["schema_id"]
        version = value["schema_version"]
    else:
        _require(
            isinstance(value, tuple) and len(value) == 2,
            f"{code}_TYPE",
        )
        schema_id, version = value
    return (
        _require_pattern(schema_id, _SCHEMA_ID_PATTERN, f"{code}_SCHEMA_ID"),
        _require_pattern(version, _SEMVER_PATTERN, f"{code}_SCHEMA_VERSION"),
    )


def _event_schema_contract(value: tuple[str, str]) -> dict[str, str]:
    return {"schema_id": value[0], "schema_version": value[1]}


@dataclass(frozen=True, slots=True)
class FixedPoint:
    """Exact int64 units and scale used by event-frequency thresholds."""

    units: str
    scale: int

    def __post_init__(self) -> None:
        _parse_integer_text(self.units, INT64_MIN, INT64_MAX, "FIXED_UNITS_OUT_OF_RANGE")
        _require(
            isinstance(self.scale, int)
            and not isinstance(self.scale, bool)
            and 0 <= self.scale <= 18,
            "FIXED_SCALE_OUT_OF_RANGE",
        )

    def to_contract(self) -> dict[str, object]:
        return {"units": self.units, "scale": self.scale}


@dataclass(frozen=True, slots=True)
class FrequencySpec:
    frequency_id: str
    kind: Literal["fixed_time_bar", "session_bar", "event_bar", "market_event"]
    periods_per_year: str
    calendar_id: str
    session_policy_version: str
    interval_ns: str | None
    session_rollup: Literal["session", "trading_day"] | None
    event_bar_basis: Literal["trade_count", "base_volume", "quote_notional"] | None
    event_bar_threshold: FixedPoint | Mapping[str, object] | None
    market_event_types: tuple[str, ...] | Sequence[str] | None

    def __post_init__(self) -> None:
        _require_string(self.frequency_id, "FREQUENCY_ID_EMPTY")
        _require(
            self.kind in {"fixed_time_bar", "session_bar", "event_bar", "market_event"},
            "FREQUENCY_KIND_INVALID",
        )
        _require_pattern(
            self.periods_per_year,
            _POSITIVE_DECIMAL_PATTERN,
            "FREQUENCY_PERIODS_PER_YEAR_INVALID",
        )
        _require_string(self.calendar_id, "FREQUENCY_CALENDAR_ID_EMPTY")
        _require_string(
            self.session_policy_version,
            "FREQUENCY_SESSION_POLICY_VERSION_EMPTY",
        )
        threshold = self.event_bar_threshold
        if isinstance(threshold, Mapping):
            _require(set(threshold) == {"units", "scale"}, "FREQUENCY_THRESHOLD_FIELDS")
            threshold = FixedPoint(units=threshold["units"], scale=threshold["scale"])
            object.__setattr__(self, "event_bar_threshold", threshold)
        _require(
            threshold is None or isinstance(threshold, FixedPoint),
            "FREQUENCY_THRESHOLD_TYPE",
        )
        event_types = self.market_event_types
        if event_types is not None:
            _require(
                not isinstance(event_types, (str, bytes)) and isinstance(event_types, Sequence),
                "FREQUENCY_EVENT_TYPES_TYPE",
            )
            event_types = tuple(event_types)
            object.__setattr__(self, "market_event_types", event_types)
            for schema_id in event_types:
                _require_pattern(
                    schema_id,
                    _SCHEMA_ID_PATTERN,
                    "FREQUENCY_EVENT_SCHEMA_ID_INVALID",
                )
            _require(len(event_types) > 0, "FREQUENCY_EVENT_TYPES_EMPTY")
            _require_unique_sorted(
                event_types,
                key=lambda item: item,
                duplicate_code="FREQUENCY_DUPLICATE_EVENT_TYPE",
                sorted_code="FREQUENCY_EVENT_TYPES_NOT_SORTED",
            )
        if self.kind == "fixed_time_bar":
            _parse_integer_text(
                self.interval_ns,
                1,
                INT64_MAX,
                "FREQUENCY_INTERVAL_OUT_OF_RANGE",
            )
            _require(self.session_rollup is None, "FREQUENCY_SESSION_ROLLUP_FORBIDDEN")
            _require(self.event_bar_basis is None, "FREQUENCY_EVENT_BASIS_FORBIDDEN")
            _require(threshold is None, "FREQUENCY_THRESHOLD_FORBIDDEN")
            _require(event_types is None, "FREQUENCY_EVENT_TYPES_FORBIDDEN")
        elif self.kind == "session_bar":
            _require(self.interval_ns is None, "FREQUENCY_INTERVAL_FORBIDDEN")
            _require(
                self.session_rollup in {"session", "trading_day"},
                "FREQUENCY_SESSION_ROLLUP_REQUIRED",
            )
            _require(self.event_bar_basis is None, "FREQUENCY_EVENT_BASIS_FORBIDDEN")
            _require(threshold is None, "FREQUENCY_THRESHOLD_FORBIDDEN")
            _require(event_types is None, "FREQUENCY_EVENT_TYPES_FORBIDDEN")
        elif self.kind == "event_bar":
            _require(self.interval_ns is None, "FREQUENCY_INTERVAL_FORBIDDEN")
            _require(self.session_rollup is None, "FREQUENCY_SESSION_ROLLUP_FORBIDDEN")
            _require(
                self.event_bar_basis in {"trade_count", "base_volume", "quote_notional"},
                "FREQUENCY_EVENT_BASIS_REQUIRED",
            )
            _require(threshold is not None, "FREQUENCY_THRESHOLD_REQUIRED")
            _require(
                _parse_integer_text(
                    threshold.units,
                    INT64_MIN,
                    INT64_MAX,
                    "FREQUENCY_THRESHOLD_OUT_OF_RANGE",
                )
                > 0,
                "FREQUENCY_THRESHOLD_OUT_OF_RANGE",
            )
            _require(event_types is None, "FREQUENCY_EVENT_TYPES_FORBIDDEN")
        else:
            _require(self.interval_ns is None, "FREQUENCY_INTERVAL_FORBIDDEN")
            _require(self.session_rollup is None, "FREQUENCY_SESSION_ROLLUP_FORBIDDEN")
            _require(self.event_bar_basis is None, "FREQUENCY_EVENT_BASIS_FORBIDDEN")
            _require(threshold is None, "FREQUENCY_THRESHOLD_FORBIDDEN")
            _require(event_types is not None, "FREQUENCY_EVENT_TYPES_REQUIRED")

    def to_contract(self) -> dict[str, object]:
        threshold = self.event_bar_threshold
        return {
            "frequency_id": self.frequency_id,
            "kind": self.kind,
            "periods_per_year": self.periods_per_year,
            "calendar_id": self.calendar_id,
            "session_policy_version": self.session_policy_version,
            "interval_ns": self.interval_ns,
            "session_rollup": self.session_rollup,
            "event_bar_basis": self.event_bar_basis,
            "event_bar_threshold": threshold.to_contract() if threshold is not None else None,
            "market_event_types": (
                list(self.market_event_types) if self.market_event_types is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class FactorDependency:
    role: str
    value_column: str
    availability_column: str

    def __post_init__(self) -> None:
        _require_string(self.role, "FACTOR_DEPENDENCY_ROLE_EMPTY")
        _require_string(self.value_column, "FACTOR_DEPENDENCY_VALUE_COLUMN_EMPTY")
        _require_string(
            self.availability_column,
            "FACTOR_DEPENDENCY_AVAILABILITY_COLUMN_EMPTY",
        )

    def to_contract(self) -> dict[str, str]:
        return {
            "role": self.role,
            "value_column": self.value_column,
            "availability_column": self.availability_column,
        }


@dataclass(frozen=True, slots=True)
class FactorSpec:
    factor_id: str
    version: str
    algorithm_id: str
    input_profile: Literal["bar", "market_event"]
    dependencies: tuple[FactorDependency, ...] | Sequence[FactorDependency]
    window_periods: int
    dtype: Literal["float64"]
    annualized: bool
    missing_policy: Literal["null", "error"]

    def __post_init__(self) -> None:
        _require_pattern(self.factor_id, _FACTOR_ID_PATTERN, "FACTOR_ID_INVALID")
        _require_pattern(self.version, _SEMVER_PATTERN, "FACTOR_VERSION_INVALID")
        _require_string(self.algorithm_id, "FACTOR_ALGORITHM_ID_EMPTY")
        _require(self.input_profile in {"bar", "market_event"}, "FACTOR_INPUT_PROFILE_INVALID")
        dependencies = tuple(self.dependencies)
        object.__setattr__(self, "dependencies", dependencies)
        _require(len(dependencies) > 0, "FACTOR_DEPENDENCIES_EMPTY")
        _require(
            all(isinstance(item, FactorDependency) for item in dependencies),
            "FACTOR_DEPENDENCY_TYPE",
        )
        _require_unique_sorted(
            dependencies,
            key=lambda item: (item.role, item.value_column),
            duplicate_code="DUPLICATE_FACTOR_DEPENDENCY",
            sorted_code="FACTOR_DEPENDENCIES_NOT_SORTED",
        )
        _require(
            isinstance(self.window_periods, int)
            and not isinstance(self.window_periods, bool)
            and 1 <= self.window_periods <= _INT32_MAX,
            "FACTOR_WINDOW_PERIODS_OUT_OF_RANGE",
        )
        _require(self.dtype == "float64", "FACTOR_DTYPE_INVALID")
        _require(isinstance(self.annualized, bool), "FACTOR_ANNUALIZED_TYPE")
        _require(self.missing_policy in {"null", "error"}, "FACTOR_MISSING_POLICY_INVALID")

    def to_contract(self) -> dict[str, object]:
        return {
            "factor_id": self.factor_id,
            "version": self.version,
            "algorithm_id": self.algorithm_id,
            "input_profile": self.input_profile,
            "dependencies": [item.to_contract() for item in self.dependencies],
            "window_periods": self.window_periods,
            "dtype": self.dtype,
            "annualized": self.annualized,
            "missing_policy": self.missing_policy,
        }


@dataclass(frozen=True, slots=True)
class AsOfSpec:
    mode: Literal["source_available_at", "fixed"]
    fixed_at_ns: str | None

    def __post_init__(self) -> None:
        _require(self.mode in {"source_available_at", "fixed"}, "AS_OF_MODE_INVALID")
        if self.mode == "source_available_at":
            _require(self.fixed_at_ns is None, "AS_OF_FIXED_AT_FORBIDDEN")
        else:
            _parse_integer_text(
                self.fixed_at_ns,
                INT64_MIN,
                INT64_MAX,
                "AS_OF_OUT_OF_RANGE",
            )

    def to_contract(self) -> dict[str, str | None]:
        return {"mode": self.mode, "fixed_at_ns": self.fixed_at_ns}


@dataclass(frozen=True, slots=True)
class FactorInputRef:
    """Closed discriminated union for curated or normalized verified input."""

    layer: Literal["curated", "normalized"]
    root: str
    dataset: str | None
    snapshot_id: str
    event_schemas: tuple[EventSchemaInput, ...] | Sequence[EventSchemaInput] | None
    market_context_snapshot_id: str | None

    def __post_init__(self) -> None:
        _require(self.layer in {"curated", "normalized"}, "INPUT_REF_LAYER_INVALID")
        _require_string(self.root, "INPUT_REF_ROOT_EMPTY")
        _require_pattern(self.snapshot_id, _SNAPSHOT_PATTERN, "INPUT_REF_SNAPSHOT_ID_INVALID")
        if self.layer == "curated":
            _require_string(self.dataset, "INPUT_REF_CURATED_DATASET_REQUIRED")
            _require(self.event_schemas is None, "INPUT_REF_CURATED_EVENT_SCHEMAS_FORBIDDEN")
            _require(
                self.market_context_snapshot_id is None,
                "INPUT_REF_CURATED_CONTEXT_FORBIDDEN",
            )
            return
        _require(self.dataset is None, "INPUT_REF_NORMALIZED_DATASET_FORBIDDEN")
        _require(self.event_schemas is not None, "INPUT_REF_NORMALIZED_EVENT_SCHEMAS_REQUIRED")
        schemas = tuple(
            _event_schema_ref(value, code="INPUT_REF_EVENT") for value in self.event_schemas or ()
        )
        object.__setattr__(self, "event_schemas", schemas)
        _require(len(schemas) > 0, "INPUT_REF_NORMALIZED_EVENT_SCHEMAS_EMPTY")
        _require_unique_sorted(
            schemas,
            key=lambda item: item,
            duplicate_code="INPUT_REF_DUPLICATE_EVENT_SCHEMA",
            sorted_code="INPUT_REF_EVENT_SCHEMAS_NOT_SORTED",
        )
        _require(
            all(schema_id != "puresaber.bar-event" for schema_id, _ in schemas),
            "INPUT_REF_NORMALIZED_BAR_FORBIDDEN",
        )
        _require_pattern(
            self.market_context_snapshot_id,
            _SNAPSHOT_PATTERN,
            "INPUT_REF_NORMALIZED_CONTEXT_REQUIRED",
        )

    def to_contract(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "root": self.root,
            "dataset": self.dataset,
            "snapshot_id": self.snapshot_id,
            "event_schemas": (
                [_event_schema_contract(value) for value in self.event_schemas]
                if self.event_schemas is not None
                else None
            ),
            "market_context_snapshot_id": self.market_context_snapshot_id,
        }


def _normalize_value_availability(value: ValueAvailabilityInput) -> Mapping[str, str]:
    pairs = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    _require(len(pairs) > 0, "AUX_VALUE_AVAILABILITY_EMPTY")
    normalized: list[tuple[str, str]] = []
    for pair in pairs:
        _require(isinstance(pair, tuple) and len(pair) == 2, "AUX_VALUE_AVAILABILITY_TYPE")
        key, availability = pair
        normalized.append(
            (
                _require_pattern(key, _MAPPING_KEY_PATTERN, "AUX_VALUE_COLUMN_INVALID"),
                _require_string(availability, "AUX_AVAILABILITY_COLUMN_EMPTY"),
            )
        )
    _require(len({key for key, _ in normalized}) == len(normalized), "AUX_DUPLICATE_VALUE_COLUMN")
    ordered = sorted(normalized, key=lambda item: _utf16_sort_key(item[0]))
    return MappingProxyType(dict(ordered))


@dataclass(frozen=True, slots=True)
class AuxiliarySource:
    role: str
    schema_id: str
    schema_version: str
    snapshot_id: str
    physical_sha256: str
    logical_sha256: str
    business_key_columns: tuple[str, ...] | Sequence[str]
    observation_time_column: str
    effective_from_column: str
    effective_to_column: str
    available_at_column: str
    superseded_at_column: str
    revision_column: str
    value_availability: ValueAvailabilityInput
    join_recipe: str

    def __post_init__(self) -> None:
        _require_string(self.role, "AUX_ROLE_EMPTY")
        _require_string(self.schema_id, "AUX_SCHEMA_ID_EMPTY")
        _require_string(self.schema_version, "AUX_SCHEMA_VERSION_EMPTY")
        _require_pattern(self.snapshot_id, _SNAPSHOT_PATTERN, "AUX_SNAPSHOT_ID_INVALID")
        _require_pattern(self.physical_sha256, _SHA256_PATTERN, "AUX_PHYSICAL_HASH_INVALID")
        _require_pattern(self.logical_sha256, _SHA256_PATTERN, "AUX_LOGICAL_HASH_INVALID")
        keys = tuple(self.business_key_columns)
        object.__setattr__(self, "business_key_columns", keys)
        _require(len(keys) > 0, "AUX_BUSINESS_KEYS_EMPTY")
        for key in keys:
            _require_string(key, "AUX_BUSINESS_KEY_EMPTY")
        _require(len(keys) == len(set(keys)), "AUX_DUPLICATE_BUSINESS_KEY")
        for field_name, field_value in (
            ("observation_time", self.observation_time_column),
            ("effective_from", self.effective_from_column),
            ("effective_to", self.effective_to_column),
            ("available_at", self.available_at_column),
            ("superseded_at", self.superseded_at_column),
            ("revision", self.revision_column),
        ):
            _require_string(field_value, f"AUX_{field_name.upper()}_COLUMN_EMPTY")
        object.__setattr__(
            self,
            "value_availability",
            _normalize_value_availability(self.value_availability),
        )
        _require_string(self.join_recipe, "AUX_JOIN_RECIPE_EMPTY")

    def to_contract(self) -> dict[str, object]:
        return {
            "role": self.role,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "physical_sha256": self.physical_sha256,
            "logical_sha256": self.logical_sha256,
            "business_key_columns": list(self.business_key_columns),
            "observation_time_column": self.observation_time_column,
            "effective_from_column": self.effective_from_column,
            "effective_to_column": self.effective_to_column,
            "available_at_column": self.available_at_column,
            "superseded_at_column": self.superseded_at_column,
            "revision_column": self.revision_column,
            "value_availability": dict(self.value_availability),
            "join_recipe": self.join_recipe,
        }


@dataclass(frozen=True, slots=True)
class SourceLineage:
    role: str
    snapshot_id: str
    logical_sha256: str
    selection_sha256: str

    def __post_init__(self) -> None:
        _require_string(self.role, "SOURCE_LINEAGE_ROLE_EMPTY")
        _require_pattern(
            self.snapshot_id,
            _SNAPSHOT_PATTERN,
            "SOURCE_LINEAGE_SNAPSHOT_ID_INVALID",
        )
        _require_pattern(
            self.logical_sha256,
            _SHA256_PATTERN,
            "SOURCE_LINEAGE_LOGICAL_HASH_INVALID",
        )
        _require_pattern(
            self.selection_sha256,
            _SHA256_PATTERN,
            "SOURCE_LINEAGE_SELECTION_HASH_INVALID",
        )

    def to_contract(self) -> dict[str, str]:
        return {
            "role": self.role,
            "snapshot_id": self.snapshot_id,
            "logical_sha256": self.logical_sha256,
            "selection_sha256": self.selection_sha256,
        }


@dataclass(frozen=True, slots=True)
class OutputField:
    name: str
    arrow_type: str
    nullable: bool

    def __post_init__(self) -> None:
        _require_string(self.name, "OUTPUT_FIELD_NAME_EMPTY")
        _require_string(self.arrow_type, "OUTPUT_FIELD_ARROW_TYPE_EMPTY")
        _require(isinstance(self.nullable, bool), "OUTPUT_FIELD_NULLABLE_TYPE")

    def to_contract(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arrow_type": self.arrow_type,
            "nullable": self.nullable,
        }


@dataclass(frozen=True, slots=True)
class FactorFrameManifest:
    schema_id: str
    certification_scope: Literal[
        "full-frequency-certified",
        "research-restated",
        "fixture-certified",
        "legacy-not-certified",
    ]
    frequency: FrequencySpec
    factor_specs: tuple[FactorSpec, ...] | Sequence[FactorSpec]
    as_of: AsOfSpec
    source_lineage: tuple[SourceLineage, ...] | Sequence[SourceLineage]
    input_event_schemas: tuple[EventSchemaInput, ...] | Sequence[EventSchemaInput]
    auxiliary_sources: tuple[AuxiliarySource, ...] | Sequence[AuxiliarySource]
    code_version: str
    input_rows: str
    output_rows: str
    output_schema: tuple[OutputField, ...] | Sequence[OutputField]
    logical_content_sha256: str
    physical_sha256: str

    def __post_init__(self) -> None:
        _require(self.schema_id == _MANIFEST_SCHEMA_ID, "MANIFEST_SCHEMA_ID_INVALID")
        _require(
            self.certification_scope
            in {
                "full-frequency-certified",
                "research-restated",
                "fixture-certified",
                "legacy-not-certified",
            },
            "MANIFEST_CERTIFICATION_SCOPE_INVALID",
        )
        _require(isinstance(self.frequency, FrequencySpec), "MANIFEST_FREQUENCY_TYPE")
        _require(isinstance(self.as_of, AsOfSpec), "MANIFEST_AS_OF_TYPE")
        factors = tuple(self.factor_specs)
        lineage = tuple(self.source_lineage)
        event_schemas = tuple(
            _event_schema_ref(value, code="MANIFEST_INPUT_EVENT")
            for value in self.input_event_schemas
        )
        auxiliaries = tuple(self.auxiliary_sources)
        output = tuple(self.output_schema)
        object.__setattr__(self, "factor_specs", factors)
        object.__setattr__(self, "source_lineage", lineage)
        object.__setattr__(self, "input_event_schemas", event_schemas)
        object.__setattr__(self, "auxiliary_sources", auxiliaries)
        object.__setattr__(self, "output_schema", output)
        _require(len(factors) > 0, "MANIFEST_FACTOR_SPECS_EMPTY")
        _require(all(isinstance(item, FactorSpec) for item in factors), "MANIFEST_FACTOR_TYPE")
        _require(len(lineage) > 0, "MANIFEST_SOURCE_LINEAGE_EMPTY")
        _require(
            all(isinstance(item, SourceLineage) for item in lineage),
            "MANIFEST_SOURCE_LINEAGE_TYPE",
        )
        _require(len(event_schemas) > 0, "MANIFEST_INPUT_EVENT_SCHEMAS_EMPTY")
        _require(
            all(isinstance(item, AuxiliarySource) for item in auxiliaries),
            "MANIFEST_AUXILIARY_TYPE",
        )
        _require(len(output) > 0, "MANIFEST_OUTPUT_SCHEMA_EMPTY")
        _require(all(isinstance(item, OutputField) for item in output), "MANIFEST_OUTPUT_TYPE")
        _require_unique_sorted(
            factors,
            key=lambda item: item.factor_id,
            duplicate_code="DUPLICATE_FACTOR",
            sorted_code="FACTOR_SPECS_NOT_SORTED",
        )
        _require_unique_sorted(
            event_schemas,
            key=lambda item: item,
            duplicate_code="DUPLICATE_INPUT_EVENT_SCHEMA",
            sorted_code="INPUT_EVENT_SCHEMAS_NOT_SORTED",
        )
        _require_unique_sorted(
            lineage,
            key=lambda item: (item.role, item.snapshot_id),
            duplicate_code="DUPLICATE_SOURCE_LINEAGE",
            sorted_code="SOURCE_LINEAGE_NOT_SORTED",
        )
        _require_unique_sorted(
            auxiliaries,
            key=lambda item: (item.role, item.snapshot_id),
            duplicate_code="DUPLICATE_AUXILIARY_SOURCE",
            sorted_code="AUXILIARY_SOURCES_NOT_SORTED",
        )
        names = [field.name for field in output]
        _require(len(names) == len(set(names)), "DUPLICATE_OUTPUT_COLUMN")
        _require_string(self.code_version, "MANIFEST_CODE_VERSION_EMPTY")
        _parse_nonnegative_count(self.input_rows, "INPUT_ROWS_OUT_OF_RANGE")
        _parse_nonnegative_count(self.output_rows, "OUTPUT_ROWS_OUT_OF_RANGE")
        _require_pattern(
            self.logical_content_sha256,
            _SHA256_PATTERN,
            "MANIFEST_LOGICAL_HASH_INVALID",
        )
        _require_pattern(
            self.physical_sha256,
            _SHA256_PATTERN,
            "MANIFEST_PHYSICAL_HASH_INVALID",
        )
        if self.as_of.mode == "fixed":
            _require(
                self.certification_scope == "research-restated",
                "FIXED_AS_OF_REQUIRES_RESEARCH_RESTATED",
            )
        self._validate_frequency_schema_binding(event_schemas, factors)
        self._validate_lineage_binding(lineage, auxiliaries, factors)
        self._validate_output_schema(output, factors)

    def _validate_frequency_schema_binding(
        self,
        event_schemas: tuple[tuple[str, str], ...],
        factors: tuple[FactorSpec, ...],
    ) -> None:
        if self.frequency.kind == "market_event":
            _require(
                all(factor.input_profile == "market_event" for factor in factors),
                "MARKET_EVENT_REQUIRES_EVENT_FACTORS",
            )
            _require(
                [item[0] for item in event_schemas] == list(self.frequency.market_event_types),
                "MARKET_EVENT_SCHEMA_SET_MISMATCH",
            )
            return
        _require(
            all(factor.input_profile == "bar" for factor in factors),
            "BAR_FREQUENCY_REQUIRES_BAR_FACTORS",
        )
        _require(
            event_schemas == (("puresaber.bar-event", "2.0.0"),),
            "BAR_INPUT_SCHEMA_MISMATCH",
        )

    @staticmethod
    def _validate_lineage_binding(
        lineage: tuple[SourceLineage, ...],
        auxiliaries: tuple[AuxiliarySource, ...],
        factors: tuple[FactorSpec, ...],
    ) -> None:
        lineage_keys = {(item.role, item.snapshot_id, item.logical_sha256) for item in lineage}
        lineage_roles = {item.role for item in lineage}
        auxiliary_by_role: dict[str, list[AuxiliarySource]] = {}
        for auxiliary in auxiliaries:
            auxiliary_by_role.setdefault(auxiliary.role, []).append(auxiliary)
            _require(
                (auxiliary.role, auxiliary.snapshot_id, auxiliary.logical_sha256) in lineage_keys,
                "AUXILIARY_LINEAGE_MISSING",
            )
        referenced_auxiliary_roles: set[str] = set()
        for factor in factors:
            for dependency in factor.dependencies:
                _require(
                    dependency.role in lineage_roles,
                    "FACTOR_DEPENDENCY_LINEAGE_MISSING",
                    dependency.role,
                )
                if dependency.role in auxiliary_by_role:
                    referenced_auxiliary_roles.add(dependency.role)
                    for auxiliary in auxiliary_by_role[dependency.role]:
                        mapping = dict(auxiliary.value_availability)
                        _require(
                            mapping.get(dependency.value_column) == dependency.availability_column,
                            "FACTOR_AUXILIARY_MAPPING_MISMATCH",
                            dependency.role,
                        )
        unused_roles = set(auxiliary_by_role) - referenced_auxiliary_roles
        _require(
            not unused_roles,
            "AUXILIARY_FACTOR_DEPENDENCY_MISSING",
            min(unused_roles) if unused_roles else None,
        )

    @staticmethod
    def _validate_output_schema(
        output: tuple[OutputField, ...], factors: tuple[FactorSpec, ...]
    ) -> None:
        expected_names = _IDENTITY_COLUMNS + tuple(
            name
            for factor in factors
            for name in (factor.factor_id, f"{factor.factor_id}__available_at")
        )
        _require(
            tuple(field.name for field in output) == expected_names, "OUTPUT_SCHEMA_ORDER_MISMATCH"
        )
        expected_identity = (
            ("instrument_id", "utf8", False),
            ("event_time", "timestamp[ns,UTC]", False),
            ("sequence", "int64", False),
            ("event_id", "utf8", False),
            ("source_available_at", "timestamp[ns,UTC]", False),
        )
        actual_identity = tuple(
            (field.name, field.arrow_type, field.nullable)
            for field in output[: len(_IDENTITY_COLUMNS)]
        )
        _require(actual_identity == expected_identity, "OUTPUT_IDENTITY_SCHEMA_MISMATCH")
        by_name = {field.name: field for field in output}
        for factor in factors:
            _require(
                by_name[factor.factor_id] == OutputField(factor.factor_id, "float64", True),
                "OUTPUT_FACTOR_SCHEMA_MISMATCH",
                factor.factor_id,
            )
            availability_name = f"{factor.factor_id}__available_at"
            _require(
                by_name[availability_name]
                == OutputField(availability_name, "timestamp[ns,UTC]", True),
                "OUTPUT_FACTOR_AVAILABILITY_SCHEMA_MISMATCH",
                factor.factor_id,
            )

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "certification_scope": self.certification_scope,
            "frequency": self.frequency.to_contract(),
            "factor_specs": [item.to_contract() for item in self.factor_specs],
            "as_of": self.as_of.to_contract(),
            "source_lineage": [item.to_contract() for item in self.source_lineage],
            "input_event_schemas": [
                _event_schema_contract(item) for item in self.input_event_schemas
            ],
            "auxiliary_sources": [item.to_contract() for item in self.auxiliary_sources],
            "code_version": self.code_version,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "output_schema": [item.to_contract() for item in self.output_schema],
            "logical_content_sha256": self.logical_content_sha256,
            "physical_sha256": self.physical_sha256,
        }


def _float64_cell(value: object) -> dict[str, str]:
    _require(isinstance(value, Real) and not isinstance(value, bool), "CELL_F64_TYPE")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ContractViolation("CELL_NON_FINITE_FLOAT") from exc
    _require(math.isfinite(number), "CELL_NON_FINITE_FLOAT")
    if number == 0.0:
        number = 0.0
    return {"t": "f64", "v": struct.pack(">d", number).hex()}


def _timestamp_ns(value: object) -> int:
    if isinstance(value, Integral) and not isinstance(value, bool):
        timestamp = int(value)
    elif isinstance(value, datetime):
        _require(value.tzinfo is not None and value.utcoffset() is not None, "CELL_NAIVE_TIMESTAMP")
        _require(value.utcoffset() == timezone.utc.utcoffset(value), "CELL_TIMESTAMP_NOT_UTC")
        exact_value = getattr(value, "value", None)
        if isinstance(exact_value, Integral):
            timestamp = int(exact_value)
        else:
            utc_value = value.astimezone(timezone.utc)
            delta = utc_value - datetime(1970, 1, 1, tzinfo=timezone.utc)
            timestamp = (
                delta.days * 86_400 + delta.seconds
            ) * 1_000_000_000 + delta.microseconds * 1_000
    else:
        raise ContractViolation("CELL_TIMESTAMP_TYPE")
    _require(INT64_MIN <= timestamp <= INT64_MAX, "CELL_TIMESTAMP_OUT_OF_RANGE")
    return timestamp


def _fixed_from_decimal(value: Decimal) -> FixedPoint:
    _require(value.is_finite(), "CELL_FIXED_NON_FINITE")
    exponent = value.as_tuple().exponent
    scale = max(0, -exponent)
    _require(scale <= 18, "CELL_FIXED_SCALE_OUT_OF_RANGE")
    units = int(value.scaleb(scale))
    _require(Decimal(units).scaleb(-scale) == value, "CELL_FIXED_INEXACT")
    return FixedPoint(str(units), scale)


def typed_cell(value: Any, *, arrow_type: str | None = None) -> dict[str, Any]:
    """Convert a domain value to its unique M8 typed-cell representation."""

    if value is None:
        return {"t": "null"}
    if arrow_type in {"utf8", "string"}:
        _require(isinstance(value, str), "CELL_UTF8_TYPE")
        return {"t": "utf8", "v": value}
    if arrow_type == "timestamp[ns,UTC]":
        return {"t": "ts_ns", "v": str(_timestamp_ns(value))}
    if arrow_type == "float64":
        return _float64_cell(value)
    integer_tag = {
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
    }.get(arrow_type or "")
    if integer_tag is not None:
        _require(
            isinstance(value, Integral) and not isinstance(value, bool),
            f"CELL_{integer_tag.upper()}_TYPE",
        )
        minimum, maximum = _INTEGER_RANGES[integer_tag]
        integer = int(value)
        _require(minimum <= integer <= maximum, f"CELL_{integer_tag.upper()}_OUT_OF_RANGE")
        return {"t": integer_tag, "v": str(integer)}
    if arrow_type == "date32":
        _require(isinstance(value, date) and not isinstance(value, datetime), "CELL_DATE_TYPE")
        return {"t": "date", "v": value.isoformat()}
    if arrow_type == "binary":
        _require(isinstance(value, bytes), "CELL_BINARY_TYPE")
        return {
            "t": "binary",
            "v": base64.urlsafe_b64encode(value).decode("ascii").rstrip("="),
        }
    if arrow_type == "fixed":
        if isinstance(value, Decimal):
            value = _fixed_from_decimal(value)
        _require(isinstance(value, FixedPoint), "CELL_FIXED_TYPE")
        return {"t": "fixed", "u": value.units, "s": str(value.scale)}
    _require(arrow_type is None, "CELL_UNSUPPORTED_ARROW_TYPE", arrow_type)
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, Integral):
        integer = int(value)
        _require(INT64_MIN <= integer <= INT64_MAX, "MANIFEST_INTEGER_OUT_OF_RANGE")
        return {"t": "i64", "v": str(integer)}
    if isinstance(value, Real):
        return _float64_cell(value)
    if isinstance(value, Decimal):
        fixed = _fixed_from_decimal(value)
        return {"t": "fixed", "u": fixed.units, "s": str(fixed.scale)}
    if isinstance(value, datetime):
        return {"t": "ts_ns", "v": str(_timestamp_ns(value))}
    if isinstance(value, date):
        return {"t": "date", "v": value.isoformat()}
    if isinstance(value, str):
        return {"t": "utf8", "v": value}
    if isinstance(value, bytes):
        return {
            "t": "binary",
            "v": base64.urlsafe_b64encode(value).decode("ascii").rstrip("="),
        }
    if isinstance(value, Mapping):
        for key in value:
            _require(isinstance(key, str) and bool(key), "CELL_STRUCT_FIELD_INVALID")
        return {
            "t": "struct",
            "v": [[key, typed_cell(value[key])] for key in sorted(value, key=_utf16_sort_key)],
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {"t": "list", "v": [typed_cell(item) for item in value]}
    raise ContractViolation("MANIFEST_UNSUPPORTED_TYPE", type(value).__name__)


def validate_typed_cell(cell: Mapping[str, Any]) -> None:
    """Validate a typed cell independently of JSON Schema tooling."""

    _require(isinstance(cell, Mapping) and "t" in cell, "CELL_TYPE")
    tag = cell["t"]
    if tag == "null":
        _require(set(cell) == {"t"}, "CELL_FIELDS_MISMATCH")
        return
    if tag == "bool":
        _require(set(cell) == {"t", "v"} and isinstance(cell["v"], bool), "CELL_BOOL_INVALID")
        return
    if tag in _INTEGER_RANGES:
        _require(set(cell) == {"t", "v"}, "CELL_FIELDS_MISMATCH")
        minimum, maximum = _INTEGER_RANGES[tag]
        _parse_integer_text(cell["v"], minimum, maximum, f"CELL_{tag.upper()}_OUT_OF_RANGE")
        return
    if tag == "f64":
        _require(set(cell) == {"t", "v"}, "CELL_FIELDS_MISMATCH")
        _require(
            isinstance(cell["v"], str) and re.fullmatch(r"[0-9a-f]{16}", cell["v"]),
            "CELL_F64_HEX_INVALID",
        )
        bits = int(cell["v"], 16)
        _require(bits != 0x8000000000000000, "CELL_NEGATIVE_ZERO")
        number = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
        _require(math.isfinite(number), "CELL_NON_FINITE_FLOAT")
        return
    if tag == "fixed":
        _require(set(cell) == {"t", "u", "s"}, "CELL_FIELDS_MISMATCH")
        _parse_integer_text(cell["u"], INT64_MIN, INT64_MAX, "CELL_FIXED_OUT_OF_RANGE")
        _require(
            isinstance(cell["s"], str) and re.fullmatch(r"(?:[0-9]|1[0-8])", cell["s"]) is not None,
            "CELL_FIXED_SCALE_OUT_OF_RANGE",
        )
        return
    if tag in {"ts_ns", "date", "utf8", "binary"}:
        _require(set(cell) == {"t", "v"} and isinstance(cell["v"], str), "CELL_TEXT_INVALID")
        if tag == "ts_ns":
            _parse_integer_text(cell["v"], INT64_MIN, INT64_MAX, "CELL_TIMESTAMP_OUT_OF_RANGE")
        elif tag == "date":
            try:
                parsed = date.fromisoformat(cell["v"])
            except ValueError as exc:
                raise ContractViolation("CELL_INVALID_DATE") from exc
            _require(parsed.isoformat() == cell["v"], "CELL_NON_CANONICAL_DATE")
        elif tag == "binary":
            encoded = cell["v"]
            _require(len(encoded) % 4 != 1, "CELL_INVALID_BASE64URL")
            try:
                decoded = base64.b64decode(
                    encoded + "=" * ((4 - len(encoded) % 4) % 4),
                    altchars=b"-_",
                    validate=True,
                )
            except ValueError as exc:
                raise ContractViolation("CELL_INVALID_BASE64URL") from exc
            _require(
                base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") == encoded,
                "CELL_NON_CANONICAL_BASE64URL",
            )
        return
    if tag == "list":
        _require(set(cell) == {"t", "v"} and isinstance(cell["v"], list), "CELL_LIST_INVALID")
        for item in cell["v"]:
            validate_typed_cell(item)
        return
    if tag == "struct":
        _require(set(cell) == {"t", "v"} and isinstance(cell["v"], list), "CELL_STRUCT_INVALID")
        names: list[str] = []
        for pair in cell["v"]:
            _require(isinstance(pair, list) and len(pair) == 2, "CELL_STRUCT_FIELD_INVALID")
            name, value = pair
            names.append(_require_string(name, "CELL_STRUCT_FIELD_INVALID"))
            validate_typed_cell(value)
        _require(len(names) == len(set(names)), "CELL_DUPLICATE_STRUCT_FIELD")
        return
    raise ContractViolation("CELL_UNKNOWN_TAG", tag)


def _jcs_order(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in value:
            _require(isinstance(key, str), "JCS_OBJECT_KEY_TYPE")
        return {key: _jcs_order(value[key]) for key in sorted(value, key=_utf16_sort_key)}
    if isinstance(value, list):
        return [_jcs_order(item) for item in value]
    _require(
        value is None or isinstance(value, (str, bool)),
        "JCS_RAW_NUMBER_FORBIDDEN" if isinstance(value, (int, float)) else "JCS_VALUE_TYPE",
    )
    return value


def jcs_bytes(value: Any) -> bytes:
    """Serialize the number-free typed envelope as RFC 8785 JCS UTF-8."""

    ordered = _jcs_order(value)
    try:
        text = json.dumps(
            ordered,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ContractViolation("JCS_INVALID_UNICODE") from exc


def manifest_projection_sha256(manifest: FactorFrameManifest) -> str:
    """Hash the typed manifest after removing physical and logical hashes."""

    _require(isinstance(manifest, FactorFrameManifest), "MANIFEST_TYPE")
    projection = {
        key: value
        for key, value in manifest.to_contract().items()
        if key not in {"logical_content_sha256", "physical_sha256"}
    }
    return hashlib.sha256(jcs_bytes(typed_cell(projection))).hexdigest()


def _output_schema_cell(fields: Sequence[OutputField]) -> dict[str, Any]:
    return {
        "t": "list",
        "v": [
            {
                "t": "struct",
                "v": [
                    ["name", typed_cell(field.name)],
                    ["arrow_type", typed_cell(field.arrow_type)],
                    ["nullable", typed_cell(field.nullable)],
                ],
            }
            for field in fields
        ],
    }


def _normalize_arrow_type(value: object) -> str:
    text = str(value)
    aliases = {"string": "utf8", "double": "float64", "date32[day]": "date32"}
    if text in aliases:
        return aliases[text]
    match = re.fullmatch(r"timestamp\[ns, tz=([^]]+)\]", text)
    if match:
        _require(match.group(1) == "UTC", "TABLE_TIMESTAMP_TIMEZONE_INVALID")
        return "timestamp[ns,UTC]"
    return text


def _table_records_and_schema(
    value: object,
) -> tuple[list[Mapping[str, Any]], tuple[OutputField, ...]] | None:
    if not (hasattr(value, "schema") and callable(getattr(value, "to_pylist", None))):
        return None
    fields = tuple(
        OutputField(field.name, _normalize_arrow_type(field.type), bool(field.nullable))
        for field in value.schema
    )
    records = value.to_pylist()
    _require(isinstance(records, list), "TABLE_RECORDS_TYPE")
    return records, fields


def _normalize_explicit_schema(
    value: Sequence[OutputField | Mapping[str, object]] | None,
) -> tuple[OutputField, ...] | None:
    if value is None:
        return None
    fields: list[OutputField] = []
    for item in value:
        if isinstance(item, OutputField):
            fields.append(item)
        else:
            _require(set(item) == {"name", "arrow_type", "nullable"}, "OUTPUT_SCHEMA_FIELDS")
            fields.append(OutputField(item["name"], item["arrow_type"], item["nullable"]))
    return tuple(fields)


def _record_values(record: object, fields: Sequence[OutputField]) -> list[Any]:
    if isinstance(record, Mapping):
        names = [field.name for field in fields]
        _require(set(record) == set(names), "RECORD_SCHEMA_MISMATCH")
        return [record[name] for name in names]
    _require(
        isinstance(record, Sequence) and not isinstance(record, (str, bytes)),
        "RECORD_TYPE",
    )
    values = list(record)
    _require(len(values) == len(fields), "RECORD_COLUMN_COUNT_MISMATCH")
    return values


def _validate_record_cells(
    records: Sequence[Sequence[Mapping[str, Any]]],
    manifest: FactorFrameManifest,
) -> None:
    fields = manifest.output_schema
    _require(len(records) == int(manifest.output_rows), "OUTPUT_ROW_COUNT_MISMATCH")
    index = {field.name: position for position, field in enumerate(fields)}
    expected_tags = {
        "utf8": "utf8",
        "timestamp[ns,UTC]": "ts_ns",
        "int64": "i64",
        "float64": "f64",
    }
    factor_pairs = [
        (index[factor.factor_id], index[f"{factor.factor_id}__available_at"])
        for factor in manifest.factor_specs
    ]
    identities: list[tuple[str, int, int, str]] = []
    for row in records:
        _require(len(row) == len(fields), "RECORD_COLUMN_COUNT_MISMATCH")
        for cell, field in zip(row, fields, strict=True):
            validate_typed_cell(cell)
            if cell["t"] == "null":
                _require(field.nullable, "NULL_IN_REQUIRED_COLUMN", field.name)
                continue
            _require(
                expected_tags.get(field.arrow_type) == cell["t"],
                "OUTPUT_TYPE_MISMATCH",
                field.name,
            )
        event_time = int(row[index["event_time"]]["v"])
        source_available_at = int(row[index["source_available_at"]]["v"])
        _require(event_time <= source_available_at, "SOURCE_AVAILABLE_BEFORE_EVENT")
        row_as_of = (
            source_available_at
            if manifest.as_of.mode == "source_available_at"
            else int(manifest.as_of.fixed_at_ns)
        )
        for factor_index, availability_index in factor_pairs:
            factor_cell = row[factor_index]
            availability_cell = row[availability_index]
            if factor_cell["t"] != "null":
                _require(
                    availability_cell["t"] != "null",
                    "FACTOR_AVAILABILITY_MISSING",
                )
            if availability_cell["t"] != "null":
                availability = int(availability_cell["v"])
                _require(
                    availability >= source_available_at,
                    "FACTOR_AVAILABLE_BEFORE_SOURCE",
                )
            if factor_cell["t"] != "null":
                _require(source_available_at <= row_as_of, "NON_NULL_FACTOR_AFTER_AS_OF")
                _require(
                    int(availability_cell["v"]) <= row_as_of,
                    "NON_NULL_FACTOR_AFTER_AS_OF",
                )
        identities.append(
            (
                row[index["instrument_id"]]["v"],
                event_time,
                int(row[index["sequence"]]["v"]),
                row[index["event_id"]]["v"],
            )
        )
    _require(len(identities) == len(set(identities)), "DUPLICATE_RECORD_IDENTITY")
    _require(identities == sorted(identities), "RECORDS_NOT_SORTED")


def factor_frame_canonical_envelope(
    manifest: FactorFrameManifest,
    records_or_table: object,
    *,
    output_schema: Sequence[OutputField | Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Build and validate the canonical FactorFrame envelope.

    ``records_or_table`` may be a pyarrow.Table-like object exposing ``schema``
    and ``to_pylist()``, or an explicit sequence of mappings/rows.  A supplied
    output schema is cross-checked against both the table and the manifest.
    """

    _require(isinstance(manifest, FactorFrameManifest), "MANIFEST_TYPE")
    explicit_schema = _normalize_explicit_schema(output_schema)
    table_data = _table_records_and_schema(records_or_table)
    if table_data is not None:
        records, table_schema = table_data
        _require(table_schema == manifest.output_schema, "TABLE_SCHEMA_BINDING_MISMATCH")
        if explicit_schema is not None:
            _require(explicit_schema == table_schema, "EXPLICIT_SCHEMA_BINDING_MISMATCH")
    else:
        _require(
            isinstance(records_or_table, Sequence)
            and not isinstance(records_or_table, (str, bytes)),
            "RECORDS_TYPE",
        )
        records = list(records_or_table)
        if explicit_schema is not None:
            _require(explicit_schema == manifest.output_schema, "EXPLICIT_SCHEMA_BINDING_MISMATCH")
    typed_records: list[list[dict[str, Any]]] = []
    for record in records:
        values = _record_values(record, manifest.output_schema)
        typed_row: list[dict[str, Any]] = []
        for value, field in zip(values, manifest.output_schema, strict=True):
            if value is None:
                _require(field.nullable, "NULL_IN_REQUIRED_COLUMN", field.name)
            typed_row.append(typed_cell(value, arrow_type=field.arrow_type))
        typed_records.append(typed_row)
    envelope = {
        "schema": _CANONICAL_SCHEMA_ID,
        "metadata": {
            "t": "struct",
            "v": [
                [
                    "manifest_projection_sha256",
                    {"t": "utf8", "v": manifest_projection_sha256(manifest)},
                ],
                ["manifest_schema_id", {"t": "utf8", "v": manifest.schema_id}],
            ],
        },
        "output_schema": _output_schema_cell(manifest.output_schema),
        "records": typed_records,
    }
    validate_factor_frame_envelope(manifest, envelope)
    return envelope


def validate_factor_frame_envelope(
    manifest: FactorFrameManifest,
    envelope: Mapping[str, Any],
) -> None:
    """Prove manifest projection, output schema, typed records and PIT binding."""

    _require(set(envelope) == {"schema", "metadata", "output_schema", "records"}, "ENVELOPE_FIELDS")
    _require(envelope["schema"] == _CANONICAL_SCHEMA_ID, "ENVELOPE_SCHEMA_ID_INVALID")
    expected_metadata = {
        "t": "struct",
        "v": [
            [
                "manifest_projection_sha256",
                {"t": "utf8", "v": manifest_projection_sha256(manifest)},
            ],
            ["manifest_schema_id", {"t": "utf8", "v": manifest.schema_id}],
        ],
    }
    _require(
        envelope["metadata"] == expected_metadata,
        "ENVELOPE_MANIFEST_PROJECTION_MISMATCH",
    )
    expected_schema = _output_schema_cell(manifest.output_schema)
    _require(envelope["output_schema"] == expected_schema, "ENVELOPE_OUTPUT_SCHEMA_MISMATCH")
    _require(isinstance(envelope["records"], list), "ENVELOPE_RECORDS_TYPE")
    _validate_record_cells(envelope["records"], manifest)


def factor_frame_logical_sha256(
    manifest: FactorFrameManifest,
    records_or_table: object,
    *,
    output_schema: Sequence[OutputField | Mapping[str, object]] | None = None,
) -> str:
    envelope = factor_frame_canonical_envelope(
        manifest,
        records_or_table,
        output_schema=output_schema,
    )
    return hashlib.sha256(jcs_bytes(envelope)).hexdigest()


def verify_factor_frame_logical_sha256(
    manifest: FactorFrameManifest,
    envelope: Mapping[str, Any],
) -> str:
    """Validate an envelope and require its SHA-256 to equal the manifest."""

    validate_factor_frame_envelope(manifest, envelope)
    logical_hash = hashlib.sha256(jcs_bytes(envelope)).hexdigest()
    _require(
        logical_hash == manifest.logical_content_sha256,
        "MANIFEST_LOGICAL_HASH_MISMATCH",
    )
    return logical_hash


__all__ = [
    "AsOfSpec",
    "AuxiliarySource",
    "ContractViolation",
    "FactorDependency",
    "FactorFrameManifest",
    "FactorInputRef",
    "FactorSpec",
    "FixedPoint",
    "FrequencySpec",
    "OutputField",
    "SourceLineage",
    "factor_frame_canonical_envelope",
    "factor_frame_logical_sha256",
    "jcs_bytes",
    "manifest_projection_sha256",
    "typed_cell",
    "validate_factor_frame_envelope",
    "validate_typed_cell",
    "verify_factor_frame_logical_sha256",
]
