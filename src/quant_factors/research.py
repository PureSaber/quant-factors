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
) -> dict:
    """Coverage, IC decay, redundancy, yearly and optional industry/regime evidence.

    No direction is selected from these diagnostics. A label endpoint on the cutoff
    is excluded. Missing warmup values stay missing, and sparse cross sections are
    reported as unavailable instead of zero correlation.
    """
    requirements = expression_requirements(names, expressions)
    if baseline_names:
        expression_requirements(list(baseline_names), expressions)
        if set(baseline_names) - set(names):
            raise ValueError("baseline_names must be included in names")
    if not names or not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Factors and positive integer horizons are required")
    panel = prices.copy()
    panel["date"] = pd.to_datetime(panel.date)
    cutoff_date = pd.Timestamp(cutoff)
    panel = panel[panel.date < cutoff_date].copy()
    panel = compute_research_factors(panel, names, expressions).sort_values(["symbol", "date"])
    for horizon in horizons:
        panel[f"return_{horizon}"] = panel.groupby("symbol").close.transform(
            lambda x, h=horizon: x.shift(-h) / x - 1
        )
        panel[f"end_{horizon}"] = panel.groupby("symbol").date.shift(-horizon)
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
        neutralized = neutralize_cross_section(panel, cols=names, by=list(neutralize_by))
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
        "cutoff": str(cutoff_date.date()),
        "requirements": requirements,
        "coverage": coverage,
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
