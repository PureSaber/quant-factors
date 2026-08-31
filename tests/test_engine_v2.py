from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from quant_factors.contracts_v2 import (
    AsOfSpec,
    FactorDependency,
    FactorSpec,
    FrequencySpec,
)
from quant_factors.engine_v2 import (
    FactorComputationError,
    _auxiliary_dependency,
    _compute_algorithm,
    _DependencyValue,
    _fixed_or_float,
    _market_dependency,
    _row_as_of,
    _utc,
    annualization_multiplier,
    compute_factor_table,
)
from quant_factors.pit_v2 import PitError

UTC_NS = pa.timestamp("ns", tz="UTC")
PRICE = pa.struct([pa.field("units", pa.int64()), pa.field("scale", pa.int8())])


def _frequency(periods: str = "98280") -> FrequencySpec:
    return FrequencySpec(
        frequency_id="bar-1m",
        kind="fixed_time_bar",
        periods_per_year=periods,
        calendar_id="crypto-24x7-v1",
        session_policy_version="v1",
        interval_ns="60000000000",
        session_rollup=None,
        event_bar_basis=None,
        event_bar_threshold=None,
        market_event_types=None,
    )


def _factor(
    factor_id: str,
    algorithm: str,
    window: int,
    *,
    annualized: bool = False,
) -> FactorSpec:
    return FactorSpec(
        factor_id=factor_id,
        version="1.0.0",
        algorithm_id=algorithm,
        input_profile="bar",
        dependencies=(
            FactorDependency(
                role="market",
                value_column="close_price",
                availability_column="available_at",
            ),
        ),
        window_periods=window,
        dtype="float64",
        annualized=annualized,
        missing_policy="null",
    )


def _bars(prices: list[int], *, scale: int = 0) -> pa.Table:
    base = 1_704_067_200_000_000_000
    minute = 60_000_000_000
    rows = []
    for index, price in enumerate(prices):
        event_time = base + (index + 1) * minute
        rows.append(
            {
                "instrument_id": "BTC-USDT",
                "event_time": event_time,
                "received_at": event_time,
                "available_at": event_time,
                "sequence": index + 1,
                "event_id": f"bar-{index + 1}",
                "close_price": {"units": price, "scale": scale},
            }
        )
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("instrument_id", pa.string(), nullable=False),
                pa.field("event_time", UTC_NS, nullable=False),
                pa.field("received_at", UTC_NS, nullable=False),
                pa.field("available_at", UTC_NS, nullable=False),
                pa.field("sequence", pa.int64(), nullable=False),
                pa.field("event_id", pa.string(), nullable=False),
                pa.field("close_price", PRICE, nullable=False),
            ]
        ),
    )


def test_period_momentum_is_frequency_agnostic_and_fixed_point_is_exact() -> None:
    factor = _factor("momentum_1p", "momentum", 1)
    first = compute_factor_table(
        _bars([1000, 1100, 1210], scale=1),
        _frequency("98280"),
        [factor],
        as_of=AsOfSpec(mode="source_available_at", fixed_at_ns=None),
    )
    second = compute_factor_table(
        _bars([1000, 1100, 1210], scale=1),
        _frequency("252"),
        [factor],
        as_of=AsOfSpec(mode="source_available_at", fixed_at_ns=None),
    )
    assert first.column("momentum_1p").to_pylist() == second.column("momentum_1p").to_pylist()
    assert first.column("momentum_1p").to_pylist()[0] is None
    assert first.column("momentum_1p").to_pylist()[1:] == pytest.approx([0.1, 0.1])


def test_annualization_uses_only_explicit_decimal_periods_per_year() -> None:
    factor = _factor("volatility_2p", "volatility", 2, annualized=True)
    low = compute_factor_table(
        _bars([100, 110, 99]),
        _frequency("4"),
        [factor],
        as_of=AsOfSpec(mode="source_available_at", fixed_at_ns=None),
    )
    high = compute_factor_table(
        _bars([100, 110, 99]),
        _frequency("9"),
        [factor],
        as_of=AsOfSpec(mode="source_available_at", fixed_at_ns=None),
    )
    low_value = low.column("volatility_2p").to_pylist()[-1]
    high_value = high.column("volatility_2p").to_pylist()[-1]
    assert low_value is not None and high_value is not None
    assert high_value / low_value == pytest.approx(1.5)
    assert annualization_multiplier(_frequency("2")) == float(
        __import__("decimal").Decimal(2).sqrt()
    )


def test_fixed_as_of_nulls_future_source_and_preserves_actual_availability() -> None:
    factor = _factor("momentum_1p", "momentum", 1)
    table = _bars([100, 110])
    fixed_ns = table.column("available_at").to_pylist()[0].value
    result = compute_factor_table(
        table,
        _frequency(),
        [factor],
        as_of=AsOfSpec(mode="fixed", fixed_at_ns=str(fixed_ns)),
    )
    assert result.column("momentum_1p").to_pylist() == [None, None]
    assert (
        result.column("momentum_1p__available_at").to_pylist()[1]
        == table.column("available_at").to_pylist()[1]
    )


def test_input_identity_and_time_order_fail_closed() -> None:
    factor = _factor("momentum_1p", "momentum", 1)
    rows = _bars([100, 110]).to_pylist()
    rows[1]["event_id"] = rows[0]["event_id"]
    rows[1]["event_time"] = rows[0]["event_time"]
    rows[1]["sequence"] = rows[0]["sequence"]
    duplicate = pa.Table.from_pylist(rows, schema=_bars([100, 110]).schema)
    with pytest.raises(FactorComputationError, match="FACTOR_DUPLICATE_IDENTITY"):
        compute_factor_table(
            duplicate,
            _frequency(),
            [factor],
            as_of=AsOfSpec(mode="source_available_at", fixed_at_ns=None),
        )

    bad_rows = _bars([100]).to_pylist()
    bad_rows[0]["received_at"] = bad_rows[0]["event_time"].value - 1
    bad = pa.Table.from_pylist(bad_rows, schema=_bars([100]).schema)
    with pytest.raises(FactorComputationError, match="FACTOR_INPUT_TIME_ORDER_INVALID"):
        compute_factor_table(
            bad,
            _frequency(),
            [factor],
            as_of=AsOfSpec(mode="source_available_at", fixed_at_ns=None),
        )


def test_non_finite_values_are_rejected() -> None:
    table = _bars([100])
    rows = table.to_pylist()
    schema = table.schema.set(
        table.schema.get_field_index("close_price"), pa.field("close_price", pa.float64())
    )
    rows[0]["close_price"] = math.inf
    with pytest.raises(FactorComputationError, match="FACTOR_VALUE_NON_FINITE"):
        compute_factor_table(
            pa.Table.from_pylist(rows, schema=schema),
            _frequency(),
            [_factor("last_1p", "last_value", 1)],
            as_of=AsOfSpec(mode="source_available_at", fixed_at_ns=None),
        )


@pytest.mark.parametrize(
    "value",
    [None, True, {"units": True, "scale": 0}, {"units": "bad", "scale": 0}],
)
def test_numeric_adapter_rejects_non_contract_values(value) -> None:
    if value is None:
        assert _fixed_or_float(value) is None
        return
    with pytest.raises(FactorComputationError):
        _fixed_or_float(value)
    with pytest.raises(FactorComputationError, match="FACTOR_FIXED_POINT_INVALID"):
        _fixed_or_float({"units": 2**63, "scale": 0})
    assert math.copysign(1, _fixed_or_float(-0.0)) == 1


def test_time_and_as_of_helpers_reject_invalid_values() -> None:
    with pytest.raises(FactorComputationError, match="FACTOR_TIME_NOT_UTC"):
        _utc("not-a-time")
    with pytest.raises(FactorComputationError, match="FACTOR_TIME_NOT_UTC"):
        _utc("2026-01-01T00:00:00")
    with pytest.raises(FactorComputationError, match="FACTOR_TIME_NOT_UTC"):
        _utc("2026-01-01T08:00:00+08:00")
    with pytest.raises(FactorComputationError, match="AS_OF_OUT_OF_RANGE"):
        _row_as_of(SimpleNamespace(mode="fixed", fixed_at_ns="bad"), pd.Timestamp.now(tz="UTC"))
    with pytest.raises(FactorComputationError, match="AS_OF_OUT_OF_RANGE"):
        _row_as_of(
            SimpleNamespace(mode="fixed", fixed_at_ns=str(2**63)),
            pd.Timestamp.now(tz="UTC"),
        )
    with pytest.raises(FactorComputationError, match="FREQUENCY_PERIODS_PER_YEAR_INVALID"):
        annualization_multiplier(SimpleNamespace(periods_per_year="NaN"))


def test_dependency_helpers_fail_closed(monkeypatch) -> None:
    dependency = FactorDependency("market", "close_price", "available_at")
    with pytest.raises(FactorComputationError, match="FACTOR_DEPENDENCY_COLUMN_MISSING"):
        _market_dependency({}, dependency)
    with pytest.raises(FactorComputationError, match="FACTOR_DEPENDENCY_AVAILABILITY_MISSING"):
        _market_dependency({"close_price": 1.0, "available_at": None}, dependency)

    auxiliary_dependency = FactorDependency("fundamental", "pe_ratio", "pe_available_at")
    now = pd.Timestamp("2026-01-01T00:00:00Z")
    assert _auxiliary_dependency({}, auxiliary_dependency, (), now, now, "null").value is None
    with pytest.raises(PitError, match="AUX_NOT_FOUND"):
        _auxiliary_dependency({}, auxiliary_dependency, (), now, now, "error")

    fake_source = SimpleNamespace(
        role="fundamental",
        business_key_columns=("instrument_id",),
        value_availability={"pe_ratio": "pe_available_at"},
        available_at_column="available_at",
    )
    fake_input = SimpleNamespace(source=fake_source)
    with pytest.raises(FactorComputationError, match="AUX_BUSINESS_KEY_MISSING"):
        _auxiliary_dependency({}, auxiliary_dependency, (fake_input,), now, now, "null")

    monkeypatch.setattr(
        "quant_factors.engine_v2.select_auxiliary_version",
        lambda *_args, **_kwargs: None,
    )
    row = {"instrument_id": "AAA"}
    with pytest.raises(PitError, match="AUX_NOT_FOUND"):
        _auxiliary_dependency(row, auxiliary_dependency, (fake_input,), now, now, "error")

    selection = SimpleNamespace(
        source=SimpleNamespace(
            value_availability={"pe_ratio": "wrong"}, available_at_column="available_at"
        ),
        row={"pe_ratio": 10.0, "pe_available_at": now, "available_at": now},
    )
    monkeypatch.setattr(
        "quant_factors.engine_v2.select_auxiliary_version",
        lambda *_args, **_kwargs: selection,
    )
    with pytest.raises(FactorComputationError, match="FACTOR_AUXILIARY_MAPPING_MISMATCH"):
        _auxiliary_dependency(row, auxiliary_dependency, (fake_input,), now, now, "null")


def _algorithm_factor(algorithm: str, dependencies: int = 1, window: int = 2) -> FactorSpec:
    return FactorSpec(
        factor_id=f"factor_{algorithm}",
        version="1.0.0",
        algorithm_id=algorithm,
        input_profile="bar",
        dependencies=tuple(
            FactorDependency("market", f"value_{index}", "available_at")
            for index in range(dependencies)
        ),
        window_periods=window,
        dtype="float64",
        annualized=False,
        missing_policy="null",
    )


def _window(*periods: tuple[float | None, ...]):
    now = pd.Timestamp("2026-01-01T00:00:00Z")
    return tuple(tuple(_DependencyValue(value, now) for value in period) for period in periods)


def test_all_certified_algorithms_and_edges() -> None:
    assert _compute_algorithm(_window((None,), (1.0,)), _algorithm_factor("rolling_mean")) is None
    assert _compute_algorithm(_window((0.0,), (1.0,)), _algorithm_factor("momentum")) is None
    assert (
        _compute_algorithm(
            _window((100.0,), (90.0,), (99.0,)), _algorithm_factor("downside_volatility")
        )
        is not None
    )
    assert _compute_algorithm(
        _window((1.0,), (2.0,)), _algorithm_factor("mean_reversion_z")
    ) == pytest.approx(2**-0.5)
    assert (
        _compute_algorithm(_window((1.0,), (1.0,)), _algorithm_factor("mean_reversion_z")) is None
    )
    assert _compute_algorithm(_window((1.0,), (3.0,)), _algorithm_factor("rolling_mean")) == 2.0
    assert _compute_algorithm(_window((1.0,), (3.0,)), _algorithm_factor("rolling_sum")) == 4.0
    assert _compute_algorithm(_window((3.0,)), _algorithm_factor("last_value")) == 3.0
    assert (
        _compute_algorithm(_window((3.0, 1.0)), _algorithm_factor("spread", dependencies=2)) == 2.0
    )
    assert (
        _compute_algorithm(_window((3.0, 2.0)), _algorithm_factor("ratio", dependencies=2)) == 1.5
    )
    assert (
        _compute_algorithm(_window((3.0, 0.0)), _algorithm_factor("ratio", dependencies=2)) is None
    )
    with pytest.raises(FactorComputationError, match="FACTOR_ALGORITHM_ARITY"):
        _compute_algorithm(_window((3.0,)), _algorithm_factor("spread"))
    with pytest.raises(FactorComputationError, match="FACTOR_ALGORITHM_UNSUPPORTED"):
        _compute_algorithm(_window((3.0,)), _algorithm_factor("unknown"))


def test_compute_table_governance_branches(monkeypatch) -> None:
    factor = _factor("momentum_1p", "momentum", 1)
    with pytest.raises(FactorComputationError, match="FACTOR_INPUT_EMPTY"):
        compute_factor_table(
            pa.table({"x": []}),
            _frequency(),
            [factor],
            as_of=AsOfSpec("source_available_at", None),
        )
    with pytest.raises(FactorComputationError, match="FACTOR_SPECS_NOT_SORTED"):
        compute_factor_table(
            _bars([100, 110]),
            _frequency(),
            [],
            as_of=AsOfSpec("source_available_at", None),
        )
    with pytest.raises(FactorComputationError, match="DUPLICATE_FACTOR"):
        compute_factor_table(
            _bars([100, 110]),
            _frequency(),
            [factor, factor],
            as_of=AsOfSpec("source_available_at", None),
        )
    with pytest.raises(FactorComputationError, match="FACTOR_INPUT_COLUMNS_MISSING"):
        compute_factor_table(
            pa.table({"instrument_id": ["AAA"]}),
            _frequency(),
            [factor],
            as_of=AsOfSpec("source_available_at", None),
        )

    rows = _bars([100, 110]).to_pylist()[::-1]
    with pytest.raises(FactorComputationError, match="FACTOR_INPUT_NOT_SORTED"):
        compute_factor_table(
            pa.Table.from_pylist(rows, schema=_bars([100, 110]).schema),
            _frequency(),
            [factor],
            as_of=AsOfSpec("source_available_at", None),
        )

    error_factor = replace(factor, missing_policy="error")
    missing_rows = _bars([100]).to_pylist()
    missing_rows[0]["close_price"] = None
    nullable_schema = _bars([100]).schema.set(
        _bars([100]).schema.get_field_index("close_price"),
        pa.field("close_price", PRICE),
    )
    with pytest.raises(FactorComputationError, match="FACTOR_DEPENDENCY_MISSING"):
        compute_factor_table(
            pa.Table.from_pylist(missing_rows, schema=nullable_schema),
            _frequency(),
            [error_factor],
            as_of=AsOfSpec("source_available_at", None),
        )

    monkeypatch.setattr("quant_factors.engine_v2._compute_algorithm", lambda *_args: math.inf)
    with pytest.raises(FactorComputationError, match="FACTOR_OUTPUT_NON_FINITE"):
        compute_factor_table(
            _bars([100]),
            _frequency(),
            [_factor("last_1p", "last_value", 1)],
            as_of=AsOfSpec("source_available_at", None),
        )
