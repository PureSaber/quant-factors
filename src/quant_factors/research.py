"""Descriptive factor evidence; labels are used only after the specified cutoff."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from quant_factors.expressions import (
    compute_research_factors,
    expression_requirements,
)
from quant_factors.neutralize import neutralize_cross_section

_SOURCE_ORDER = "__quant_factor_source_order"
_EOD_UTC = pd.Timedelta(hours=23, minutes=59)


def _session_dates(values, *, name: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.tz is not None:
        raise ValueError(f"{name} must contain timezone-naive session dates")
    if not dates.equals(dates.normalize()):
        raise ValueError(f"{name} must contain normalized session dates")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"{name} must contain unique ordered session dates")
    return dates


def _dense_inputs(
    original: pd.DataFrame,
    symbols: pd.Index,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    keys = pd.MultiIndex.from_product([symbols, sessions], names=["symbol", "date"]).to_frame(
        index=False
    )
    return (
        keys.merge(original, on=["symbol", "date"], how="left", validate="one_to_one")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def _mask_incomplete_windows(
    computed: pd.DataFrame,
    visible: pd.Series,
    names: list[str],
    requirements: dict[str, dict],
) -> None:
    def complete_window(values: pd.Series, *, window: int) -> pd.Series:
        return values.rolling(window, min_periods=window).sum().eq(window)

    for name in names:
        window = int(requirements[name]["warmup_bars"])
        complete = visible.groupby(computed.symbol, sort=False).transform(
            complete_window, window=window
        )
        computed.loc[~complete, name] = np.nan


def _historical_factor_panel(
    panel: pd.DataFrame,
    names: list[str],
    expressions: dict[str, str] | None,
    requirements: dict[str, dict],
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute each signal from data visible at its historical decision time."""

    if panel.duplicated(["symbol", "date"]).any():
        raise ValueError("symbol/date rows must be unique")
    symbols = pd.Index(panel.symbol.unique())
    expected_rows = len(symbols) * len(sessions)
    complete = (
        len(panel) == expected_rows
        and panel.date.isin(sessions).all()
        and panel.groupby("symbol").date.nunique().eq(len(sessions)).all()
    )
    original = panel.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    original[_SOURCE_ORDER] = np.arange(len(original))
    has_availability = "available_at" in original
    late_rows = False
    if has_availability:
        original["available_at"] = pd.to_datetime(original.available_at, utc=True)
        decision_at = original.date.dt.tz_localize("UTC") + _EOD_UTC
        late_rows = bool(
            (original.available_at.isna() | original.available_at.gt(decision_at)).any()
        )

    if not late_rows:
        inputs = original if complete else _dense_inputs(original, symbols, sessions)
        computed = compute_research_factors(inputs, names, expressions)
        if not complete:
            _mask_incomplete_windows(
                computed,
                computed[_SOURCE_ORDER].notna(),
                names,
                requirements,
            )
        return (
            computed.loc[computed[_SOURCE_ORDER].notna()]
            .sort_values(_SOURCE_ORDER)
            .drop(columns=_SOURCE_ORDER)
            .reset_index(drop=True)
        )

    maximum_warmup = max(int(item["warmup_bars"]) for item in requirements.values())
    value_columns = [
        column
        for column in original
        if column not in {"symbol", "date", "available_at", _SOURCE_ORDER}
    ]
    factor_rows = []
    for evaluation_date in pd.DatetimeIndex(sorted(original.date.unique())):
        position = sessions.get_loc(evaluation_date)
        window_dates = sessions[max(0, position - maximum_warmup + 1) : position + 1]
        window_source = original[original.date.isin(window_dates)]
        inputs = _dense_inputs(window_source, symbols, window_dates)
        decision_at = evaluation_date.tz_localize("UTC") + _EOD_UTC
        visible = (
            inputs[_SOURCE_ORDER].notna()
            & inputs.available_at.notna()
            & inputs.available_at.le(decision_at)
        )
        for column in value_columns:
            inputs[column] = inputs[column].where(visible)
        computed = compute_research_factors(inputs, names, expressions)
        _mask_incomplete_windows(computed, visible, names, requirements)
        current = computed[computed.date.eq(evaluation_date) & computed[_SOURCE_ORDER].notna()]
        factor_rows.append(current[["symbol", "date", *names]])
    factors = pd.concat(factor_rows, ignore_index=True)
    base = original.drop(columns=[name for name in names if name in original])
    return (
        base.merge(factors, on=["symbol", "date"], how="left", validate="one_to_one")
        .sort_values(_SOURCE_ORDER)
        .drop(columns=_SOURCE_ORDER)
        .reset_index(drop=True)
    )


def _delay_signal_panel(
    frame: pd.DataFrame,
    names: list[str],
    signal_delay: int,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Rank then delay signals on the complete session grid."""

    result = frame.copy()
    signals = frame[["symbol", "date", *names]].copy()
    signals[_SOURCE_ORDER] = np.arange(len(signals))
    signals[names] = signals.groupby("date")[names].rank(pct=True)
    dense = _dense_inputs(signals, pd.Index(frame.symbol.unique()), sessions)
    dense[names] = dense.groupby("symbol", sort=False)[names].shift(signal_delay)
    delayed = (
        dense.loc[dense[_SOURCE_ORDER].notna()]
        .sort_values(_SOURCE_ORDER)[names]
        .reset_index(drop=True)
    )
    result[names] = delayed.to_numpy()
    return result


def factor_requirements(names: list[str]) -> dict[str, dict]:
    """Expose required columns and completed-bar warmup without executing a factor."""
    requirements = expression_requirements(names)
    return {
        name: {
            key: value
            for key, value in requirement.items()
            if key not in {"dependencies", "pit_columns"}
        }
        for name, requirement in requirements.items()
    }


def _corr(a: pd.Series, b: pd.Series, *, rank: bool = True) -> float | None:
    pair = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    if rank:
        pair = pair.rank()
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return float(value) if np.isfinite(value) else None


def factor_report(
    prices: pd.DataFrame,
    names: list[str],
    *,
    expressions: dict[str, str] | None = None,
    horizons: tuple[int, ...] = (1, 5, 20),
    cutoff: str,
    start: str | None = None,
    end: str | None = None,
    baseline_names: tuple[str, ...] = (),
    neutralize_by: tuple[str, ...] = (),
    signal_delay: int = 0,
    terminal_labels: pd.DataFrame | None = None,
    sessions=None,
) -> dict:
    """Coverage, IC decay, redundancy, yearly and optional industry/regime evidence.

    No direction is selected from these diagnostics. A label endpoint on the cutoff
    is excluded. Missing warmup values stay missing, and sparse cross sections are
    reported as unavailable instead of zero correlation.
    """
    requirements = expression_requirements(names, expressions)
    if type(signal_delay) is not int or signal_delay < 0:
        raise ValueError("signal_delay must be a non-negative integer")
    if baseline_names:
        expression_requirements(list(baseline_names), expressions)
        if set(baseline_names) - set(names):
            raise ValueError("baseline_names must be included in names")
    if not names or not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Factors and positive integer horizons are required")
    panel = prices.copy()
    panel["date"] = pd.to_datetime(panel.date)
    if panel.date.dt.tz is not None or not panel.date.equals(panel.date.dt.normalize()):
        raise ValueError("date must contain timezone-naive normalized session dates")
    cutoff_date = pd.Timestamp(cutoff)
    if cutoff_date.tzinfo is not None or cutoff_date != cutoff_date.normalize():
        raise ValueError("cutoff must be a timezone-naive session date")
    panel = panel[panel.date < cutoff_date].copy()
    grid = (
        _session_dates(sorted(panel.date.unique()), name="date")
        if sessions is None
        else _session_dates(sessions, name="sessions")
    )
    grid = grid[grid < cutoff_date]
    if not panel.date.isin(grid).all():
        raise ValueError("sessions must contain every price date before cutoff")
    panel = _historical_factor_panel(panel, names, expressions, requirements, grid).sort_values(
        ["symbol", "date"]
    )
    neutralized = (
        neutralize_cross_section(panel, cols=names, by=list(neutralize_by))
        if neutralize_by
        else None
    )
    if signal_delay:
        # Execution delays per-date percentile ranks, after neutralization.
        # Lag the same representation before cutting the evaluation interval.
        panel = _delay_signal_panel(panel, names, signal_delay, grid)
        if neutralized is not None:
            neutralized = _delay_signal_panel(neutralized, names, signal_delay, grid)
    from quant_data_kit.financial.labels import forward_labels

    selected = panel[["date", "symbol"]].rename(columns={"symbol": "instrument_id"})
    selected["sample_id"] = (
        selected.instrument_id.astype(str) + ":" + selected.date.dt.strftime("%Y-%m-%d")
    )
    levels = selected[["date", "instrument_id"]].copy()
    levels["value"] = panel.get("return_close", panel.close)
    levels["available_at"] = (
        pd.to_datetime(panel.available_at, utc=True)
        if "available_at" in panel
        else panel.date.dt.tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59)
    )
    levels = levels.loc[np.isfinite(levels.value) & levels.value.gt(0)]
    label_records = []
    for horizon in horizons:
        terminal = None
        if terminal_labels is not None:
            if "horizon" not in terminal_labels:
                raise ValueError("terminal label evidence must declare its horizon")
            terminal = terminal_labels.loc[terminal_labels.horizon.eq(horizon)].drop(
                columns="horizon"
            )
            terminal = terminal.loc[terminal.sample_id.isin(selected.sample_id)]
        labels = forward_labels(
            levels,
            selected,
            grid,
            horizon=horizon,
            as_of=cutoff_date.tz_localize("UTC"),
            terminals=terminal,
        )
        panel[f"return_{horizon}"] = labels["return"].to_numpy(dtype=float)
        panel[f"end_{horizon}"] = pd.to_datetime(labels.label_end).to_numpy()
        for row in labels.to_dict("records"):
            if (
                start
                and row["date"] < pd.Timestamp(start)
                or end
                and row["date"] > pd.Timestamp(end)
            ):
                continue
            label_records.append(
                {
                    "sample_id": row["sample_id"],
                    "date": str(row["date"].date()),
                    "symbol": row["instrument_id"],
                    "horizon": horizon,
                    "status": row["status"],
                    "return": None if pd.isna(row["return"]) else row["return"],
                }
            )
    if start:
        panel = panel[panel.date >= pd.Timestamp(start)]
    if end:
        panel = panel[panel.date <= pd.Timestamp(end)]
    panel["year"] = panel.date.dt.year.astype(str)
    evidence, coverage, segments, redundancy = [], [], [], []
    for name in names:
        valid = np.isfinite(pd.to_numeric(panel[name], errors="coerce"))
        coverage.append(
            {
                "factor": name,
                "rows": len(panel),
                "valid_rows": int(valid.sum()),
                "coverage": float(valid.mean()) if len(panel) else None,
            }
        )
        for horizon in horizons:
            mature = panel[panel[f"end_{horizon}"] < cutoff_date]
            ics, ranks = [], []
            for _, day in mature.groupby("date"):
                ic = _corr(day[name], day[f"return_{horizon}"], rank=False)
                rank_ic = _corr(day[name], day[f"return_{horizon}"])
                if ic is not None:
                    ics.append(ic)
                if rank_ic is not None:
                    ranks.append(rank_ic)
            sd = float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0
            evidence.append(
                {
                    "factor": name,
                    "horizon": horizon,
                    "sessions": len(ranks),
                    "mean_ic": float(np.mean(ics)) if ics else None,
                    "rank_ic": float(np.mean(ranks)) if ranks else None,
                    "rank_ic_ir": float(np.mean(ranks) / sd) if sd > 0 else None,
                    "positive_ratio": float(np.mean(np.array(ranks) > 0)) if ranks else None,
                }
            )
            for dimension in ("year", "industry", "regime"):
                if dimension not in mature:
                    continue
                for value, group in mature.groupby(dimension):
                    values = [
                        _corr(day[name], day[f"return_{horizon}"])
                        for _, day in group.groupby("date")
                    ]
                    values = [v for v in values if v is not None]
                    segments.append(
                        {
                            "factor": name,
                            "horizon": horizon,
                            "dimension": dimension,
                            "value": str(value),
                            "sessions": len(values),
                            "rank_ic": float(np.mean(values)) if values else None,
                        }
                    )
    for left, right in combinations(names, 2):
        values = [_corr(day[left], day[right]) for _, day in panel.groupby("date")]
        values = [v for v in values if v is not None]
        redundancy.append(
            {
                "left": left,
                "right": right,
                "sessions": len(values),
                "rank_correlation": float(np.mean(values)) if values else None,
            }
        )

    incremental = []
    candidates = [name for name in names if name not in baseline_names]
    if baseline_names:
        for candidate in candidates:
            residual = pd.Series(np.nan, index=panel.index, dtype=float)
            for _, day in panel.groupby("date"):
                columns = [candidate, *baseline_names]
                valid = day[columns].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) <= len(baseline_names) + 1:
                    continue
                design = np.column_stack(
                    [np.ones(len(valid)), *(valid[name].to_numpy() for name in baseline_names)]
                )
                coefficients, *_ = np.linalg.lstsq(design, valid[candidate].to_numpy(), rcond=None)
                residual.loc[valid.index] = valid[candidate].to_numpy() - design @ coefficients
            for horizon in horizons:
                mature = panel[panel[f"end_{horizon}"] < cutoff_date]
                raw_values = [
                    _corr(day[candidate], day[f"return_{horizon}"])
                    for _, day in mature.groupby("date")
                ]
                residual_values = [
                    _corr(residual.loc[day.index], day[f"return_{horizon}"])
                    for _, day in mature.groupby("date")
                ]
                raw_values = [value for value in raw_values if value is not None]
                residual_values = [value for value in residual_values if value is not None]
                incremental.append(
                    {
                        "factor": candidate,
                        "baseline_factors": list(baseline_names),
                        "horizon": horizon,
                        "raw_sessions": len(raw_values),
                        "residual_sessions": len(residual_values),
                        "raw_rank_ic": float(np.mean(raw_values)) if raw_values else None,
                        "residual_rank_ic": (
                            float(np.mean(residual_values)) if residual_values else None
                        ),
                    }
                )

    neutralization = []
    if neutralize_by:
        neutralized = neutralized.loc[panel.index]
        applied = [name for name in neutralize_by if name in panel.columns]
        for name in names:
            for horizon in horizons:
                mature = panel[panel[f"end_{horizon}"] < cutoff_date]
                neutral_mature = neutralized.loc[mature.index]
                raw_values = [
                    _corr(day[name], day[f"return_{horizon}"]) for _, day in mature.groupby("date")
                ]
                adjusted_values = [
                    _corr(day[name], mature.loc[day.index, f"return_{horizon}"])
                    for _, day in neutral_mature.groupby("date")
                ]
                raw_values = [value for value in raw_values if value is not None]
                adjusted_values = [value for value in adjusted_values if value is not None]
                neutralization.append(
                    {
                        "factor": name,
                        "horizon": horizon,
                        "requested_by": list(neutralize_by),
                        "applied_by": applied,
                        "raw_rank_ic": float(np.mean(raw_values)) if raw_values else None,
                        "neutralized_rank_ic": (
                            float(np.mean(adjusted_values)) if adjusted_values else None
                        ),
                        "sessions": len(adjusted_values),
                    }
                )
    return {
        "schema_version": "quant.factor-research/v1",
        "scope": "descriptive-retrospective",
        "signal_delay": signal_delay,
        "signal_representation": "lagged-cross-sectional-percentile-rank"
        if signal_delay
        else "factor-value",
        "cutoff": str(cutoff_date.date()),
        "requirements": requirements,
        "coverage": coverage,
        "label_samples": label_records,
        "label_coverage": [
            {
                "horizon": h,
                "selected_samples": sum(r["horizon"] == h for r in label_records),
                "status_counts": pd.Series(
                    [r["status"] for r in label_records if r["horizon"] == h]
                )
                .value_counts()
                .to_dict(),
            }
            for h in horizons
        ],
        "ic_decay": evidence,
        "segments": segments,
        "correlations": redundancy,
        "incremental": incremental,
        "neutralization": neutralization,
        "limitations": [
            "overlapping labels; ICIR is descriptive, not a significance test",
            "incremental strategy value requires paired out-of-sample replay",
        ],
    }
