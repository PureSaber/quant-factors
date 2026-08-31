"""Frequency-explicit, point-in-time factor computation primitives."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from itertools import pairwise
from typing import Any

import pandas as pd
import pyarrow as pa

from quant_factors.contracts_v2 import AsOfSpec, FactorSpec, FrequencySpec
from quant_factors.pit_v2 import (
    PitError,
    VerifiedAuxiliaryInput,
    select_auxiliary_version,
    validate_auxiliary_inputs,
)

IDENTITY_COLUMNS = (
    "instrument_id",
    "event_time",
    "sequence",
    "event_id",
    "source_available_at",
)
UTC_NS = pa.timestamp("ns", tz="UTC")


class FactorComputationError(ValueError):
    """Stable failure from the certified factor engine."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}:{detail}" if detail else code)


@dataclass(frozen=True)
class _DependencyValue:
    value: float | None
    available_at: pd.Timestamp | None


def _utc(value: Any, code: str = "FACTOR_TIME_NOT_UTC") -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FactorComputationError(code) from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise FactorComputationError(code)
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise FactorComputationError(code)
    return timestamp.tz_convert("UTC")


def _fixed_or_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping) and set(value) == {"units", "scale"}:
        units = value["units"]
        scale = value["scale"]
        if isinstance(units, bool) or isinstance(scale, bool):
            raise FactorComputationError("FACTOR_FIXED_POINT_INVALID")
        try:
            units_value = int(units)
            scale_value = int(scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FactorComputationError("FACTOR_FIXED_POINT_INVALID") from exc
        if units_value != units or scale_value != scale or not 0 <= scale_value <= 18:
            raise FactorComputationError("FACTOR_FIXED_POINT_INVALID")
        if not -(2**63) <= units_value <= 2**63 - 1:
            raise FactorComputationError("FACTOR_FIXED_POINT_INVALID")
        number = float(Decimal(units_value).scaleb(-scale_value))
    else:
        if isinstance(value, bool):
            raise FactorComputationError("FACTOR_VALUE_INVALID")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FactorComputationError("FACTOR_VALUE_INVALID") from exc
    if not math.isfinite(number):
        raise FactorComputationError("FACTOR_VALUE_NON_FINITE")
    return 0.0 if number == 0.0 else number


def annualization_multiplier(frequency: FrequencySpec) -> float:
    """Use the frozen decimal square-root rule without inferred calendar constants."""
    try:
        periods = Decimal(frequency.periods_per_year)
    except Exception as exc:  # pragma: no cover - FrequencySpec normally rejects this first
        raise FactorComputationError("FREQUENCY_PERIODS_PER_YEAR_INVALID") from exc
    if not periods.is_finite() or periods <= 0:
        raise FactorComputationError("FREQUENCY_PERIODS_PER_YEAR_INVALID")
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        result = periods.sqrt(context=context)
    return float(result)


def _row_as_of(as_of: AsOfSpec, source_available_at: pd.Timestamp) -> pd.Timestamp:
    if as_of.mode == "source_available_at":
        return source_available_at
    try:
        value = int(as_of.fixed_at_ns)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FactorComputationError("AS_OF_OUT_OF_RANGE") from exc
    if not -(2**63) <= value <= 2**63 - 1:
        raise FactorComputationError("AS_OF_OUT_OF_RANGE")
    return pd.Timestamp(value, unit="ns", tz="UTC")


def _market_dependency(row: Mapping[str, Any], dependency: Any) -> _DependencyValue:
    if dependency.value_column not in row or dependency.availability_column not in row:
        raise FactorComputationError("FACTOR_DEPENDENCY_COLUMN_MISSING", dependency.value_column)
    value = _fixed_or_float(row[dependency.value_column])
    raw_available = row[dependency.availability_column]
    available = None if raw_available is None else _utc(raw_available)
    if value is not None and available is None:
        raise FactorComputationError("FACTOR_DEPENDENCY_AVAILABILITY_MISSING")
    return _DependencyValue(value, available)


def _auxiliary_dependency(
    row: Mapping[str, Any],
    dependency: Any,
    auxiliaries: Sequence[VerifiedAuxiliaryInput],
    observation_time: pd.Timestamp,
    row_as_of: pd.Timestamp,
    missing_policy: str,
) -> _DependencyValue:
    role_inputs = [item for item in auxiliaries if item.source.role == dependency.role]
    if not role_inputs:
        if missing_policy == "error":
            raise PitError("AUX_NOT_FOUND", dependency.role)
        return _DependencyValue(None, None)
    business_columns = {name for item in role_inputs for name in item.source.business_key_columns}
    missing = sorted(business_columns - set(row))
    if missing:
        raise FactorComputationError("AUX_BUSINESS_KEY_MISSING", ",".join(missing))
    selection = select_auxiliary_version(
        role_inputs,
        role=dependency.role,
        business_key={name: row[name] for name in business_columns},
        observation_time=observation_time,
        row_as_of=row_as_of,
    )
    if selection is None:
        if missing_policy == "error":
            raise PitError("AUX_NOT_FOUND", dependency.role)
        return _DependencyValue(None, None)
    source = selection.source
    expected_availability = source.value_availability.get(dependency.value_column)
    if expected_availability != dependency.availability_column:
        raise FactorComputationError("FACTOR_AUXILIARY_MAPPING_MISMATCH", dependency.role)
    selected = selection.row
    value = _fixed_or_float(selected.get(dependency.value_column))
    raw_value_available = selected.get(dependency.availability_column)
    raw_source_available = selected.get(source.available_at_column)
    value_available = None if raw_value_available is None else _utc(raw_value_available)
    source_available = _utc(raw_source_available)
    available = (
        source_available if value_available is None else max(source_available, value_available)
    )
    if value is not None and value_available is None:
        raise FactorComputationError("FACTOR_DEPENDENCY_AVAILABILITY_MISSING")
    return _DependencyValue(value, available)


def _factor_window(
    history: Sequence[tuple[_DependencyValue, ...]], factor: FactorSpec
) -> tuple[tuple[_DependencyValue, ...], ...] | None:
    window = factor.window_periods
    algorithm = factor.algorithm_id
    required = (
        window + 1 if algorithm in {"momentum", "volatility", "downside_volatility"} else window
    )
    if algorithm in {"last_value", "spread", "ratio"}:
        required = 1
    if len(history) < required:
        return None
    if algorithm == "momentum":
        return (history[-required], history[-1])
    return tuple(history[-required:])


def _compute_algorithm(
    window: tuple[tuple[_DependencyValue, ...], ...], factor: FactorSpec
) -> float | None:
    values = [[item.value for item in period] for period in window]
    if any(value is None for period in values for value in period):
        return None
    numeric = [[float(value) for value in period] for period in values]
    algorithm = factor.algorithm_id
    if algorithm == "momentum":
        start, end = numeric[0][0], numeric[-1][0]
        return None if start == 0 else end / start - 1.0
    if algorithm in {"volatility", "downside_volatility"}:
        prices = [period[0] for period in numeric]
        returns = [
            current / previous - 1.0 for previous, current in pairwise(prices) if previous != 0
        ]
        if len(returns) != factor.window_periods:
            return None
        if algorithm == "downside_volatility":
            returns = [value if value < 0 else 0.0 for value in returns]
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        return math.sqrt(variance)
    if algorithm == "mean_reversion_z":
        series = [period[0] for period in numeric]
        if len(series) < 2:
            return None
        mean = sum(series) / len(series)
        variance = sum((value - mean) ** 2 for value in series) / (len(series) - 1)
        deviation = math.sqrt(variance)
        return None if deviation == 0 else (series[-1] - mean) / deviation
    if algorithm == "rolling_mean":
        series = [period[0] for period in numeric]
        return sum(series) / len(series)
    if algorithm == "rolling_sum":
        return sum(period[0] for period in numeric)
    if algorithm == "last_value":
        return numeric[-1][0]
    if algorithm == "spread":
        if len(numeric[-1]) != 2:
            raise FactorComputationError("FACTOR_ALGORITHM_ARITY", factor.factor_id)
        return numeric[-1][0] - numeric[-1][1]
    if algorithm == "ratio":
        if len(numeric[-1]) != 2:
            raise FactorComputationError("FACTOR_ALGORITHM_ARITY", factor.factor_id)
        return None if numeric[-1][1] == 0 else numeric[-1][0] / numeric[-1][1]
    raise FactorComputationError("FACTOR_ALGORITHM_UNSUPPORTED", algorithm)


def _output_schema(factors: Sequence[FactorSpec]) -> pa.Schema:
    fields = [
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("event_time", UTC_NS, nullable=False),
        pa.field("sequence", pa.int64(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("source_available_at", UTC_NS, nullable=False),
    ]
    for factor in factors:
        fields.extend(
            (
                pa.field(factor.factor_id, pa.float64()),
                pa.field(f"{factor.factor_id}__available_at", UTC_NS),
            )
        )
    return pa.schema(fields)


def compute_factor_table(
    table: pa.Table,
    frequency: FrequencySpec,
    factors: Iterable[FactorSpec],
    *,
    as_of: AsOfSpec,
    auxiliary_sources: Iterable[VerifiedAuxiliaryInput] = (),
) -> pa.Table:
    """Compute one deterministic FactorFrame table from an already verified market input."""
    if not isinstance(table, pa.Table) or table.num_rows <= 0:
        raise FactorComputationError("FACTOR_INPUT_EMPTY")
    factor_specs = tuple(factors)
    if not factor_specs or factor_specs != tuple(
        sorted(factor_specs, key=lambda item: item.factor_id)
    ):
        raise FactorComputationError("FACTOR_SPECS_NOT_SORTED")
    if len({item.factor_id for item in factor_specs}) != len(factor_specs):
        raise FactorComputationError("DUPLICATE_FACTOR")
    required_columns = {
        "instrument_id",
        "event_time",
        "sequence",
        "event_id",
        "received_at",
        "available_at",
    }
    missing = sorted(required_columns - set(table.column_names))
    if missing:
        raise FactorComputationError("FACTOR_INPUT_COLUMNS_MISSING", ",".join(missing))
    auxiliaries = validate_auxiliary_inputs(auxiliary_sources)
    histories: dict[tuple[str, str], list[tuple[_DependencyValue, ...]]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    identities: set[tuple[str, int, int, str]] = set()
    prior_by_instrument: dict[str, tuple[int, int, str]] = {}
    annualizer = annualization_multiplier(frequency)
    for source_row in table.to_pylist():
        row = dict(source_row)
        instrument_id = str(row["instrument_id"])
        event_time = _utc(row["event_time"])
        received_at = _utc(row["received_at"])
        source_available_at = _utc(row["available_at"])
        if not event_time <= received_at <= source_available_at:
            raise FactorComputationError("FACTOR_INPUT_TIME_ORDER_INVALID")
        sequence = row["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise FactorComputationError("FACTOR_SEQUENCE_INVALID")
        identity = (instrument_id, event_time.value, sequence, str(row["event_id"]))
        if identity in identities:
            raise FactorComputationError("FACTOR_DUPLICATE_IDENTITY")
        identities.add(identity)
        ordering = identity[1:]
        if instrument_id in prior_by_instrument and ordering <= prior_by_instrument[instrument_id]:
            raise FactorComputationError("FACTOR_INPUT_NOT_SORTED")
        prior_by_instrument[instrument_id] = ordering
        row_as_of = _row_as_of(as_of, source_available_at)
        result: dict[str, Any] = {
            "instrument_id": instrument_id,
            "event_time": event_time,
            "sequence": sequence,
            "event_id": str(row["event_id"]),
            "source_available_at": source_available_at,
        }
        for factor in factor_specs:
            dependencies: list[_DependencyValue] = []
            for dependency in factor.dependencies:
                if dependency.role == "market":
                    selected = _market_dependency(row, dependency)
                else:
                    selected = _auxiliary_dependency(
                        row,
                        dependency,
                        auxiliaries,
                        event_time,
                        row_as_of,
                        factor.missing_policy,
                    )
                if selected.value is None and factor.missing_policy == "error":
                    raise FactorComputationError("FACTOR_DEPENDENCY_MISSING", factor.factor_id)
                dependencies.append(selected)
            history = histories[(factor.factor_id, instrument_id)]
            history.append(tuple(dependencies))
            window = _factor_window(history, factor)
            available_at: pd.Timestamp | None = None
            value: float | None = None
            if window is not None:
                availability = [
                    item.available_at
                    for period in window
                    for item in period
                    if item.available_at is not None
                ]
                available_at = max([source_available_at, *availability])
                all_available = all(
                    item.available_at is not None and item.available_at <= row_as_of
                    for period in window
                    for item in period
                )
                if source_available_at <= row_as_of and all_available:
                    value = _compute_algorithm(window, factor)
                    if value is not None and factor.annualized:
                        value *= annualizer
                    if value is not None and not math.isfinite(value):
                        raise FactorComputationError("FACTOR_OUTPUT_NON_FINITE")
            result[factor.factor_id] = value
            result[f"{factor.factor_id}__available_at"] = available_at
        output.append(result)
    output.sort(
        key=lambda row: (
            row["instrument_id"],
            row["event_time"].value,
            row["sequence"],
            row["event_id"],
        )
    )
    return pa.Table.from_pylist(output, schema=_output_schema(factor_specs)).combine_chunks()
