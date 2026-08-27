"""Time-series research validation primitives.

All splitters operate on ordered observation timestamps. ``purged_kfold`` also
accepts label end times, so training labels overlapping the test interval are
removed rather than merely shuffling rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationSplit:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _ordered_dates(values: Iterable) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(list(values)))
    if dates.hasnans:
        raise ValueError("validation dates contain missing values")
    if not dates.is_monotonic_increasing:
        raise ValueError("validation dates must be sorted")
    return dates


def walk_forward_splits(
    dates: Iterable,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    embargo_size: int = 0,
    expanding: bool = True,
) -> list[ValidationSplit]:
    """Create deterministic rolling or expanding out-of-sample windows."""
    ordered = _ordered_dates(dates)
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if embargo_size < 0:
        raise ValueError("embargo_size cannot be negative")
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    splits: list[ValidationSplit] = []
    test_start_idx = train_size + embargo_size
    fold = 0
    while test_start_idx + test_size <= len(ordered):
        train_end_idx = test_start_idx - embargo_size
        train_start_idx = 0 if expanding else train_end_idx - train_size
        if train_start_idx < 0:
            break
        train_idx = np.arange(train_start_idx, train_end_idx, dtype=int)
        test_idx = np.arange(test_start_idx, test_start_idx + test_size, dtype=int)
        splits.append(
            ValidationSplit(
                fold=fold,
                train_indices=train_idx,
                test_indices=test_idx,
                train_start=ordered[train_idx[0]],
                train_end=ordered[train_idx[-1]],
                test_start=ordered[test_idx[0]],
                test_end=ordered[test_idx[-1]],
            )
        )
        fold += 1
        test_start_idx += step
    return splits


def purged_kfold_splits(
    event_starts: Iterable,
    event_ends: Iterable,
    *,
    n_splits: int = 5,
    embargo_size: int = 0,
) -> list[ValidationSplit]:
    """K-fold splits that purge overlapping label intervals and embargo successors."""
    starts = _ordered_dates(event_starts)
    ends = pd.DatetimeIndex(pd.to_datetime(list(event_ends)))
    if len(starts) != len(ends):
        raise ValueError("event_starts and event_ends must have equal length")
    if (ends < starts).any():
        raise ValueError("event end precedes event start")
    if not 2 <= n_splits <= len(starts):
        raise ValueError("n_splits must be between 2 and sample count")
    if embargo_size < 0:
        raise ValueError("embargo_size cannot be negative")

    test_blocks = [
        block for block in np.array_split(np.arange(len(starts)), n_splits) if len(block)
    ]
    splits: list[ValidationSplit] = []
    all_idx = np.arange(len(starts))
    for fold, test_idx in enumerate(test_blocks):
        test_start = starts[test_idx].min()
        test_end = ends[test_idx].max()
        overlaps = (starts <= test_end) & (ends >= test_start)
        train_mask = ~overlaps
        embargo_stop = min(int(test_idx[-1]) + 1 + embargo_size, len(starts))
        train_mask[int(test_idx[-1]) + 1 : embargo_stop] = False
        train_idx = all_idx[train_mask]
        if not len(train_idx):
            raise ValueError(f"fold {fold} has no training observations after purging")
        splits.append(
            ValidationSplit(
                fold=fold,
                train_indices=train_idx,
                test_indices=test_idx,
                train_start=starts[train_idx].min(),
                train_end=starts[train_idx].max(),
                test_start=test_start,
                test_end=test_end,
            )
        )
    return splits


def benjamini_hochberg(p_values: Iterable[float], alpha: float = 0.05) -> pd.DataFrame:
    """Control false discovery rate across a family of factor tests."""
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if ((values < 0) | (values > 1) | ~np.isfinite(values)).any():
        raise ValueError("p_values must be finite values in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    order = np.argsort(values)
    ranked = values[order]
    m = len(values)
    adjusted_sorted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return pd.DataFrame(
        {
            "test": np.arange(m),
            "p_value": values,
            "adjusted_p_value": adjusted,
            "reject": adjusted <= alpha,
        }
    )


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    observations: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that observed Sharpe exceeds a benchmark under non-normal returns."""
    if observations < 2:
        raise ValueError("observations must be at least 2")
    variance = 1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2
    if variance <= 0:
        raise ValueError("invalid Sharpe sampling variance")
    z_score = (observed_sharpe - benchmark_sharpe) * sqrt(observations - 1) / sqrt(variance)
    return 0.5 * (1 + erf(z_score / sqrt(2)))


def audit_feature_availability(
    frame: pd.DataFrame,
    feature_availability: dict[str, str],
    *,
    observation_col: str = "date",
) -> pd.DataFrame:
    """Report features whose source timestamp is later than the model observation."""
    if observation_col not in frame.columns:
        raise ValueError(f"missing observation column: {observation_col}")
    observation = pd.to_datetime(frame[observation_col], errors="coerce")
    if observation.isna().any():
        raise ValueError("observation timestamps contain invalid values")
    rows: list[dict[str, int | str | float]] = []
    for feature, availability_col in feature_availability.items():
        if feature not in frame.columns or availability_col not in frame.columns:
            raise ValueError(f"missing feature availability pair: {feature}, {availability_col}")
        available = pd.to_datetime(frame[availability_col], errors="coerce")
        populated = frame[feature].notna()
        missing_evidence = populated & available.isna()
        future = populated & available.notna() & (available > observation)
        rows.append(
            {
                "feature": feature,
                "populated_rows": int(populated.sum()),
                "missing_availability_rows": int(missing_evidence.sum()),
                "future_rows": int(future.sum()),
                "future_ratio": float(future.sum() / max(populated.sum(), 1)),
            }
        )
    return pd.DataFrame(rows)


def summarize_fold_stability(metrics: pd.DataFrame, metric_col: str) -> dict[str, float | int]:
    """Summarize out-of-sample direction and dispersion across validation folds."""
    if metric_col not in metrics.columns or metrics.empty:
        raise ValueError(f"missing fold metric: {metric_col}")
    values = pd.to_numeric(metrics[metric_col], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"fold metric has no finite values: {metric_col}")
    mean = float(values.mean())
    return {
        "folds": len(values),
        "mean": mean,
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "positive_fold_ratio": float((values > 0).mean()),
        "worst_fold": float(values.min()),
    }
