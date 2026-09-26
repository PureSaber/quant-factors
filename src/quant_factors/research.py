"""Descriptive factor evidence; labels are used only after the specified cutoff."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from quant_factors.core import FACTOR_REGISTRY, compute_factors


def factor_requirements(names: list[str]) -> dict[str, dict]:
    """Expose required columns and completed-bar warmup without executing a factor."""
    unknown = set(names) - set(FACTOR_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown factors: {sorted(unknown)}")
    requirements = {}
    for name in names:
        tokens = name.split("_")
        window = next((int(t[:-1]) for t in tokens if t.endswith("d") and t[:-1].isdigit()), 0)
        columns = ["close"]
        if name.startswith(("volume_", "turnover_")):
            columns = ["volume"]
        elif name.startswith("amihud_"):
            columns = ["close", "volume"]
        elif name in {"pe_inv", "pb_inv"}:
            columns = ["pe_ratio" if name == "pe_inv" else "pb_ratio"]
        if name == "volume_surge_5d":
            window = 20
        requirements[name] = {
            "columns": columns,
            "warmup_bars": window + 1,
            "pit_required": name in {"pe_inv", "pb_inv"},
        }
    return requirements


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
    horizons: tuple[int, ...] = (1, 5, 20),
    cutoff: str,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Coverage, IC decay, redundancy, yearly and optional industry/regime evidence.

    No direction is selected from these diagnostics. A label endpoint on the cutoff
    is excluded. Missing warmup values stay missing, and sparse cross sections are
    reported as unavailable instead of zero correlation.
    """
    factor_requirements(names)
    if not names or not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Factors and positive integer horizons are required")
    panel = prices.copy()
    panel["date"] = pd.to_datetime(panel.date)
    cutoff_date = pd.Timestamp(cutoff)
    panel = panel[panel.date < cutoff_date].copy()
    panel = compute_factors(panel, names).sort_values(["symbol", "date"])
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
    return {
        "schema_version": "quant.factor-research/v1",
        "scope": "descriptive-retrospective",
        "cutoff": str(cutoff_date.date()),
        "requirements": factor_requirements(names),
        "coverage": coverage,
        "ic_decay": evidence,
        "segments": segments,
        "correlations": redundancy,
        "limitations": [
            "overlapping labels; ICIR is descriptive, not a significance test",
            "incremental strategy value requires paired out-of-sample replay",
        ],
    }
