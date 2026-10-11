"""Transactional local cache over the existing finite-window PIT engine.

Input comparison and storage are O(N). Only factor evaluation is incremental.
No fitted transform or auxiliary dependency is supported by this first version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import ipc

from quant_factors import contracts_v2, engine_v2, pit_v2
from quant_factors.contracts_v2 import AsOfSpec, FactorSpec, FrequencySpec

SUPPORTED_ALGORITHMS = frozenset(
    {
        "momentum",
        "volatility",
        "downside_volatility",
        "mean_reversion_z",
        "rolling_mean",
        "rolling_sum",
    }
)
_KEYS = ("instrument_id", "event_time", "sequence", "event_id")
_LOG = logging.getLogger(__name__)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def artifact_identity(ref: dict) -> dict:
    """Validate the public ArtifactRef and exclude its machine-specific locator."""
    required = {"schema_version", "sha256", "size_bytes", "uri"}
    if (
        not isinstance(ref, dict)
        or not required <= ref.keys()
        or ref.keys() - required - {"media_type"}
        or ref["schema_version"] != "quant.artifact-ref/v1"
        or not isinstance(ref["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None
        or type(ref["size_bytes"]) is not int
        or ref["size_bytes"] < 0
        or not isinstance(ref["uri"], str)
        or not ref["uri"].strip()
        or ("media_type" in ref and not isinstance(ref["media_type"], str))
    ):
        raise ValueError("Invalid quant.artifact-ref/v1")
    return {key: ref[key] for key in ("schema_version", "sha256", "size_bytes")}


def source_identity() -> dict:
    """Bind actual source bytes and numerical runtime, without absolute paths."""
    files = [Path(__file__), *[Path(m.__file__) for m in (contracts_v2, engine_v2, pit_v2)]]
    return {
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
    }


def _encode(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table.combine_chunks())
    return sink.getvalue().to_pybytes()


def _decode(blob: bytes, expected: str) -> pa.Table:
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("Factor cache checksum mismatch")
    return ipc.open_stream(blob).read_all()


class _Connection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


@dataclass(frozen=True)
class CachedFactors:
    table: pa.Table
    computation_id: str
    metadata: dict


class FactorCache:
    """Append-only results; SQLite commits input, output and identity atomically."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS factor_cache_inputs_v1 (
                    input_hash TEXT PRIMARY KEY, input_blob BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS factor_cache_v1 (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    computation_id TEXT UNIQUE NOT NULL, family_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL REFERENCES factor_cache_inputs_v1(input_hash),
                    output_blob BLOB NOT NULL, output_hash TEXT NOT NULL,
                    metadata TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS factor_cache_family_v1
                    ON factor_cache_v1(family_id, sequence);
            """)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30, factory=_Connection)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def compute(
        self,
        table: pa.Table,
        frequency: FrequencySpec,
        factors: list[FactorSpec],
        *,
        as_of: AsOfSpec,
        input_identities: list[dict],
        universe: list[str],
        run_id: str,
        training: dict | None = None,
    ) -> CachedFactors:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id is required")
        specs = tuple(sorted(factors, key=lambda f: f.factor_id))
        if not specs or len({s.factor_id for s in specs}) != len(specs):
            raise ValueError("Distinct factor specs required")
        for spec in specs:
            if (
                spec.algorithm_id not in SUPPORTED_ALGORITHMS
                or len(spec.dependencies) != 1
                or spec.dependencies[0].role != "market"
            ):
                raise ValueError("Incremental cache supports listed unary market factors only")
        if (
            not isinstance(universe, list)
            or not universe
            or any(not isinstance(x, str) or not x.strip() for x in universe)
            or len(set(universe)) != len(universe)
        ):
            raise ValueError("Explicit distinct universe required")
        if not input_identities:
            raise ValueError("Immutable input identities required")
        identities = sorted([artifact_identity(r) for r in input_identities], key=canonical)
        if not isinstance(table, pa.Table) or not table.num_rows:
            raise ValueError("Nonempty Arrow input required")
        if len(set(table.column_names)) != len(table.column_names):
            raise ValueError("Duplicate input columns")
        required = {*_KEYS, "received_at", "available_at"}
        required.update(
            c
            for s in specs
            for c in (
                s.dependencies[0].value_column,
                s.dependencies[0].availability_column,
            )
        )
        if not required <= set(table.column_names):
            raise ValueError("Missing input columns")
        # Project only genuine dependencies. Column order/unused fields cannot alter arithmetic.
        selected = table.select(sorted(required)).replace_schema_metadata(None)
        selected = selected.filter(pc.is_in(selected["instrument_id"], pa.array(universe)))
        selected = selected.sort_by([(key, "ascending") for key in _KEYS]).combine_chunks()
        if training is not None:
            self._validate_training(training, as_of)
            cutoff = pa.scalar(pd.Timestamp(int(training["end_ns"]), unit="ns", tz="UTC"))
            selected = selected.filter(pc.less_equal(selected["event_time"], cutoff))
        if not selected.num_rows:
            raise ValueError("Empty selected input")
        rows = selected.to_pylist()
        seen = set()
        for row in rows:
            key = tuple(row[c] for c in _KEYS)
            if key in seen or any(row[c] is None for c in _KEYS):
                raise ValueError("Duplicate or null event identity")
            if not all(isinstance(row[c], str) and row[c] for c in ("instrument_id", "event_id")):
                raise ValueError("Nonempty event identity required")
            seen.add(key)
        family = {
            "operation": "quant-factors/finite-window-cache",
            "version": "1.0.0",
            "source": source_identity(),
            "parameters": [s.to_contract() for s in specs],
            "frequency": frequency.to_contract(),
            "as_of": as_of.to_contract(),
            "universe": sorted(universe),
            "training": training,
        }
        identity = {
            **family,
            "inputs": identities,
            "selected_input_sha256": pit_v2.arrow_table_logical_sha256(selected),
        }
        computation_id, family_id = digest(identity), digest(family)
        with self._connect() as db:
            hit = db.execute(
                "SELECT output_blob,output_hash,metadata FROM factor_cache_v1 "
                "WHERE computation_id=?",
                (computation_id,),
            ).fetchone()
            prior = (
                None
                if hit
                else db.execute(
                    "SELECT c.computation_id,i.input_blob,c.input_hash,c.output_blob,c.output_hash "
                    "FROM factor_cache_v1 c JOIN factor_cache_inputs_v1 i "
                    "ON i.input_hash=c.input_hash WHERE c.family_id=? "
                    "ORDER BY c.sequence DESC LIMIT 1",
                    (family_id,),
                ).fetchone()
            )
        if hit:
            output = _decode(hit[0], hit[1])
            metadata = {
                **json.loads(hit[2]),
                "execution_mode": "cache_hit",
                "engine_rows": 0,
                "recomputed_rows": 0,
                "reused_rows": output.num_rows,
                "run_id": run_id,
            }
            return self._result(output, computation_id, metadata)
        lookback = max(
            s.window_periods
            if s.algorithm_id
            in {
                "momentum",
                "volatility",
                "downside_volatility",
            }
            else s.window_periods - 1
            for s in specs
        )
        old_input = _decode(prior[1], prior[2]) if prior else None
        old_output = _decode(prior[3], prior[4]) if prior else None
        if old_input is not None and selected.schema.equals(old_input.schema):
            output, engine_rows, recomputed = self._incremental(
                selected,
                old_input,
                old_output,
                frequency,
                specs,
                as_of,
                lookback,
            )
            mode = "incremental"
        else:
            output = engine_v2.compute_factor_table(selected, frequency, specs, as_of=as_of)
            engine_rows = recomputed = selected.num_rows
            mode = "full"
        metadata = {
            "identity": identity,
            "execution_mode": mode,
            "engine_rows": engine_rows,
            "recomputed_rows": recomputed,
            "reused_rows": selected.num_rows - recomputed,
            "input_rows": selected.num_rows,
            "parent_computation_id": prior[0] if prior else None,
            "run_id": run_id,
            "output_sha256": pit_v2.arrow_table_logical_sha256(output),
        }
        input_blob, output_blob = _encode(selected), _encode(output)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT OR IGNORE INTO factor_cache_inputs_v1 VALUES (?,?)",
                (hashlib.sha256(input_blob).hexdigest(), input_blob),
            )
            db.execute(
                "INSERT OR IGNORE INTO factor_cache_v1 "
                "(computation_id,family_id,input_hash,output_blob,output_hash,metadata) "
                "VALUES (?,?,?,?,?,?)",
                (
                    computation_id,
                    family_id,
                    hashlib.sha256(input_blob).hexdigest(),
                    output_blob,
                    hashlib.sha256(output_blob).hexdigest(),
                    canonical(metadata),
                ),
            )
            existing = db.execute(
                "SELECT output_hash FROM factor_cache_v1 WHERE computation_id=?",
                (computation_id,),
            ).fetchone()[0]
            if existing != hashlib.sha256(output_blob).hexdigest():
                raise ValueError("Non-deterministic factor result for one computation identity")
        return self._result(output, computation_id, metadata)

    @staticmethod
    def _validate_training(training, as_of):
        if (
            not isinstance(training, dict)
            or set(training) != {"start_ns", "end_ns", "fold_id", "state_sha256"}
            or not isinstance(training["fold_id"], str)
            or not training["fold_id"].strip()
            or training["state_sha256"] is not None
        ):
            raise ValueError(
                "Training needs start_ns/end_ns/fold_id and null state; fitting unsupported"
            )
        start = AsOfSpec("fixed", training["start_ns"])
        end = AsOfSpec("fixed", training["end_ns"])
        if (
            int(start.fixed_at_ns) > int(end.fixed_at_ns)
            or as_of.mode != "fixed"
            or int(as_of.fixed_at_ns) > int(end.fixed_at_ns)
        ):
            raise ValueError("Training requires a fixed as-of no later than its end")

    @staticmethod
    def _incremental(table, old, old_output, frequency, specs, as_of, lookback):
        pieces, engine_rows, recomputed = [], 0, 0
        for instrument in dict.fromkeys(table["instrument_id"].to_pylist()):
            current = table.filter(pc.equal(table["instrument_id"], instrument))
            previous = old.filter(pc.equal(old["instrument_id"], instrument))
            previous_output = old_output.filter(pc.equal(old_output["instrument_id"], instrument))
            new_rows, old_rows = current.to_pylist(), previous.to_pylist()
            prefix = 0
            while (
                prefix < min(len(new_rows), len(old_rows)) and new_rows[prefix] == old_rows[prefix]
            ):
                prefix += 1
            suffix = 0
            while (
                suffix < min(len(new_rows), len(old_rows)) - prefix
                and new_rows[-1 - suffix] == old_rows[-1 - suffix]
            ):
                suffix += 1
            # Row-count changes shift rolling windows. A suffix is reusable only after
            # a complete new lookback lies inside the common suffix.
            stop = min(len(new_rows), len(new_rows) - suffix + lookback)
            if prefix == len(new_rows) == len(old_rows):
                stop = prefix
            if prefix:
                pieces.append(previous_output.slice(0, prefix))
            if stop > prefix:
                begin = max(0, prefix - lookback)
                computed = engine_v2.compute_factor_table(
                    current.slice(begin, stop - begin),
                    frequency,
                    specs,
                    as_of=as_of,
                )
                pieces.append(computed.slice(prefix - begin))
                engine_rows += stop - begin
                recomputed += stop - prefix
            if stop < len(new_rows):
                pieces.append(previous_output.slice(len(old_rows) - (len(new_rows) - stop)))
        return pa.concat_tables(pieces).combine_chunks(), engine_rows, recomputed

    @staticmethod
    def _result(output, computation_id, metadata):
        _LOG.info(
            canonical(
                {
                    "event": "factor_cache_result",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": metadata["run_id"],
                    "computation_id": computation_id,
                    "execution_mode": metadata["execution_mode"],
                    "engine_rows": metadata["engine_rows"],
                }
            )
        )
        return CachedFactors(output, computation_id, metadata)
