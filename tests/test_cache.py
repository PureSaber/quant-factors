import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pytest

from quant_factors import cache as module
from quant_factors.cache import FactorCache
from quant_factors.contracts_v2 import AsOfSpec, FactorDependency, FactorSpec, FrequencySpec
from quant_factors.engine_v2 import compute_factor_table
from quant_factors.pit_v2 import arrow_table_logical_sha256


def bars(n=100):
    rows = []
    for symbol in ("A", "B"):
        for i in range(n):
            stamp = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i)
            rows.append(
                {
                    "instrument_id": symbol,
                    "event_time": stamp,
                    "received_at": stamp,
                    "available_at": stamp,
                    "sequence": i,
                    "event_id": f"{symbol}-{i}",
                    "close": float(100 + i + i % 7),
                }
            )
    return pa.Table.from_pylist(rows)


def frequency():
    return FrequencySpec(
        "daily", "session_bar", "252", "synthetic-v1", "v1", None, "trading_day", None, None, None
    )


def spec(algorithm="momentum", window=5):
    return FactorSpec(
        "factor",
        "1.0.0",
        algorithm,
        "bar",
        [FactorDependency("market", "close", "available_at")],
        window,
        "float64",
        False,
        "null",
    )


def refs(table):
    return [
        {
            "schema_version": "quant.artifact-ref/v1",
            "sha256": arrow_table_logical_sha256(table),
            "size_bytes": table.nbytes,
            "uri": "memory:verified-test",
        }
    ]


def run(cache, table, factor=None, **kwargs):
    options = {
        "as_of": AsOfSpec("source_available_at", None),
        "input_identities": refs(table),
        "universe": ["A", "B"],
        "run_id": "test",
    }
    options.update(kwargs)
    return cache.compute(table, frequency(), [factor or spec()], **options)


@pytest.mark.parametrize("algorithm", sorted(module.SUPPORTED_ALGORITHMS))
@pytest.mark.parametrize("change", ["append", "revise", "insert", "delete", "late"])
def test_incremental_is_exactly_full_and_evaluates_only_dirty_windows(tmp_path, algorithm, change):
    cache, factor = FactorCache(tmp_path / "cache.db"), spec(algorithm)
    original = bars()
    if change == "insert":
        before = pa.Table.from_pylist([r for r in original.to_pylist() if r["sequence"] != 60])
        after = original
    else:
        before = original
        rows = (bars(103) if change == "append" else original).to_pylist()
        if change == "revise":
            rows[60]["close"] += 13
        elif change == "delete":
            del rows[60]
        elif change == "late":
            rows[60]["available_at"] += pd.Timedelta(days=7)
        after = pa.Table.from_pylist(rows)
    run(cache, before, factor)
    actual = run(cache, after, factor)
    expected = compute_factor_table(
        after, frequency(), [factor], as_of=AsOfSpec("source_available_at", None)
    )
    assert actual.table.equals(expected)
    assert actual.metadata["execution_mode"] == "incremental"
    assert actual.metadata["engine_rows"] < 40
    assert actual.metadata["recomputed_rows"] < 22
    assert actual.metadata["reused_rows"] > 160


def test_repeat_survives_restart_and_does_not_invoke_engine(tmp_path, monkeypatch):
    path, table = tmp_path / "cache.db", bars()
    first = run(FactorCache(path), table)

    def fail(*args, **kwargs):
        raise AssertionError("cache hit must not execute factors")

    monkeypatch.setattr(module.engine_v2, "compute_factor_table", fail)
    second = run(FactorCache(path), table, run_id="another-run")
    assert second.table.equals(first.table)
    assert second.computation_id == first.computation_id
    assert second.metadata["execution_mode"] == "cache_hit"
    assert second.metadata["run_id"] == "another-run"
    relocated = refs(table)
    relocated[0]["uri"] = "store:other-machine"
    assert (
        run(FactorCache(path), table, input_identities=relocated).computation_id
        == first.computation_id
    )


def test_parameters_source_input_time_universe_and_calendar_invalidate(tmp_path, monkeypatch):
    cache, table = FactorCache(tmp_path / "cache.db"), bars()
    baseline = run(cache, table)
    changed = [
        run(cache, table, spec(window=7)),
        run(cache, table, as_of=AsOfSpec("fixed", str(pd.Timestamp("2025", tz="UTC").value))),
        run(cache, table, universe=["A"]),
        run(cache, table, input_identities=[{**refs(table)[0], "sha256": "a" * 64}]),
        cache.compute(
            table,
            replace(frequency(), calendar_id="another-calendar"),
            [spec()],
            as_of=AsOfSpec("source_available_at", None),
            input_identities=refs(table),
            universe=["A", "B"],
            run_id="calendar",
        ),
    ]
    source = module.source_identity()
    monkeypatch.setattr(module, "source_identity", lambda: {**source, "revision": "changed"})
    changed.append(run(cache, table))
    assert all(r.computation_id != baseline.computation_id for r in changed)
    assert all(r.metadata["execution_mode"] == "full" for r in changed[:3] + changed[4:])
    assert changed[3].metadata["engine_rows"] == 0


def test_training_isolation_and_no_future_price_or_availability(tmp_path):
    cache, table = FactorCache(tmp_path / "cache.db"), bars()
    start = str(pd.Timestamp("2024-01-01", tz="UTC").value)
    end = str(pd.Timestamp("2024-02-19", tz="UTC").value)
    training = {"start_ns": start, "end_ns": end, "fold_id": "fold-1", "state_sha256": None}
    first = run(cache, table, as_of=AsOfSpec("fixed", end), training=training)
    assert first.table.num_rows == 100
    changed = table.to_pylist()
    for row in changed:
        if pd.Timestamp(row["event_time"]).value > int(end):
            row["close"] *= 99
    second = run(
        cache, pa.Table.from_pylist(changed), as_of=AsOfSpec("fixed", end), training=training
    )
    assert second.table.equals(first.table)
    assert second.computation_id != first.computation_id  # full input version remains bound
    assert second.metadata["engine_rows"] == 0
    other_fold = run(
        cache, table, as_of=AsOfSpec("fixed", end), training={**training, "fold_id": "fold-2"}
    )
    assert other_fold.metadata["execution_mode"] == "full"
    with pytest.raises(ValueError, match="fixed as-of"):
        run(cache, table, training=training)
    with pytest.raises(ValueError, match="fitting unsupported"):
        run(
            cache,
            table,
            as_of=AsOfSpec("fixed", end),
            training={**training, "state_sha256": "f" * 64},
        )
    changed = table.to_pylist()
    changed[49]["available_at"] += pd.Timedelta(days=1)
    result = run(
        cache, pa.Table.from_pylist(changed), as_of=AsOfSpec("fixed", end), training=training
    )
    assert result.table["factor"][49].as_py() is None


def test_prefix_is_causal_new_future_rows_do_not_change_history(tmp_path):
    cache = FactorCache(tmp_path / "cache.db")
    first = run(cache, bars(30))
    second = run(cache, bars(100))
    for symbol in ("A", "B"):
        old = first.table.filter(pa.compute.equal(first.table["instrument_id"], symbol))
        new = second.table.filter(pa.compute.equal(second.table["instrument_id"], symbol)).slice(
            0, 30
        )
        assert new.equals(old)


def test_unary_boundary_windows_removal_and_multiple_factors(tmp_path):
    cache, table = FactorCache(tmp_path / "cache.db"), bars(12)
    for algorithm in module.SUPPORTED_ALGORITHMS:
        factor = spec(algorithm, 1)
        run(cache, table, factor)
        shorter = pa.Table.from_pylist(table.to_pylist()[:8])
        result = run(cache, shorter, factor)
        assert result.table.equals(
            compute_factor_table(
                shorter, frequency(), [factor], as_of=AsOfSpec("source_available_at", None)
            )
        )
    factors = [spec(window=2), replace(spec("rolling_mean", 9), factor_id="mean")]
    kwargs = {
        "as_of": AsOfSpec("source_available_at", None),
        "universe": ["A", "B"],
        "run_id": "multi",
    }
    cache.compute(table, frequency(), factors, input_identities=refs(table), **kwargs)
    later = bars(15)
    actual = cache.compute(later, frequency(), factors, input_identities=refs(later), **kwargs)
    assert actual.table.equals(
        compute_factor_table(later, frequency(), factors, as_of=kwargs["as_of"])
    )


@pytest.mark.parametrize(
    "factor",
    [
        spec("last_value"),
        replace(spec(), dependencies=[FactorDependency("fundamental", "close", "available_at")]),
        replace(
            spec(),
            dependencies=[
                FactorDependency("market", "close", "available_at"),
                FactorDependency("market", "other", "available_at"),
            ],
        ),
    ],
)
def test_unsupported_is_rejected_not_silently_recomputed(tmp_path, factor):
    with pytest.raises(ValueError, match="unary market"):
        run(FactorCache(tmp_path / "cache.db"), bars(), factor)


def test_integrity_and_concurrent_publish(tmp_path):
    path, table = tmp_path / "cache.db", bars()
    FactorCache(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: run(FactorCache(path), table), range(8)))
    assert len({r.computation_id for r in results}) == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM factor_cache_v1").fetchone()[0] == 1
        db.execute("UPDATE factor_cache_v1 SET output_blob=?", (b"corrupt",))
    with pytest.raises(ValueError, match="checksum"):
        run(FactorCache(path), table)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"universe": []},
        {"universe": ["A", "A"]},
        {"run_id": ""},
        {"input_identities": []},
        {"input_identities": [{}]},
        {"universe": ["missing"]},
    ],
)
def test_invalid_context_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run(FactorCache(tmp_path / "cache.db"), bars(), **kwargs)


def test_invalid_table_and_duplicate_specs_rejected(tmp_path):
    cache, table = FactorCache(tmp_path / "cache.db"), bars(3)
    with pytest.raises(ValueError, match="Duplicate"):
        run(cache, pa.concat_tables([table, table]))
    with pytest.raises(ValueError, match="Missing"):
        run(cache, table.drop(["close"]))
    with pytest.raises(ValueError, match="Nonempty"):
        run(cache, table.slice(0, 0))
    with pytest.raises(ValueError, match="Distinct"):
        cache.compute(
            table,
            frequency(),
            [spec(), spec()],
            as_of=AsOfSpec("fixed", "0"),
            input_identities=refs(table),
            universe=["A"],
            run_id="bad",
        )
