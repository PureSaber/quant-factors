"""Purged combinatorial splits: a distinct estimand from original CSCV/PBO."""

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def purged_combinatorial_splits(
    starts, ends, *, blocks=6, test_blocks=2, embargo=0, max_splits=20000
):
    starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    if (
        len(starts) != len(ends)
        or starts.hasnans
        or ends.hasnans
        or not starts.is_monotonic_increasing
        or (ends < starts).any()
    ):
        raise ValueError("ordered starts and valid observed label end times required")
    if (
        type(blocks) is not int
        or type(test_blocks) is not int
        or not 2 <= blocks <= len(starts)
        or not 1 <= test_blocks < blocks
        or type(embargo) is not int
        or embargo < 0
        or comb(blocks, test_blocks) > max_splits
    ):
        raise ValueError("invalid combinatorial split/embargo limits")
    partitions = np.array_split(np.arange(len(starts)), blocks)
    records = []
    for selected in combinations(range(blocks), test_blocks):
        test = np.concatenate([partitions[i] for i in selected])
        blocked = np.zeros(len(starts), dtype=bool)
        # Exact interval overlap, not one min/max interval spanning disjoint test blocks.
        for i in test:
            blocked |= (starts <= ends[i]) & (ends >= starts[i])
        purged = np.flatnonzero(blocked & ~np.isin(np.arange(len(starts)), test))
        embargoed = np.zeros(len(starts), dtype=bool)
        for block in selected:
            last = partitions[block][-1]
            # Embargo starts after test labels end, not just their observation start.
            stop_time = ends[partitions[block]].max()
            first = max(last + 1, starts.searchsorted(stop_time, side="right"))
            embargoed[first : first + embargo] = True
        train = np.flatnonzero(~(blocked | embargoed))
        if not len(train):
            raise ValueError("no training data after purge/embargo")
        records.append(
            {
                "train_indices": train,
                "test_indices": test,
                "test_blocks": selected,
                "purged_indices": purged,
                "embargoed_indices": np.flatnonzero(embargoed),
                "estimand": "purged-combinatorial-validation-not-original-CSCV",
            }
        )
    return records


def family_statistics(returns, *, periods_per_year, blocks=8):
    """Optional shared research layer; pip install quant-factors[research]."""
    from quant_lab.selection import cscv_pbo, deflated_sharpe

    return {
        "dsr": deflated_sharpe(returns, periods_per_year=periods_per_year),
        "cscv": cscv_pbo(returns, blocks=blocks),
    }


def dependent_mean_evidence(values, *, block_lengths, lags, seed=17, repetitions=1000):
    from quant_lab.selection import bootstrap_means, hac_mean

    return {
        "bootstrap": bootstrap_means(
            values, block_lengths=block_lengths, repetitions=repetitions, seed=seed
        ),
        "hac": {name: hac_mean(values[name], lags=lags) for name in values},
    }


def nested_factor_selection(data, candidates, **settings):
    """Fit direction, windows and neutralization inside callbacks, never globally.

    One row per decision date; panel-aware callbacks may use a bounded per-date
    cross section. Return values are descriptive scores, not a funded OOS account.
    Requires the optional ``research`` dependency group.
    """
    from quant_lab.nested import nested_selection

    return nested_selection(data, candidates, **settings)
