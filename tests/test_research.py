import json

import numpy as np
import pandas as pd
import pytest

from quant_factors.core import compute_factors
from quant_factors.research import factor_report, factor_requirements


def panel():
    return pd.DataFrame(
        [
            {
                "date": d,
                "symbol": str(s),
                "close": 10 * (1 + 0.001 * s) ** i,
                "volume": 100 + i,
                "industry": "example",
            }
            for s in range(1, 6)
            for i, d in enumerate(pd.bdate_range("2024-01-01", periods=100))
        ]
    )


def test_all_factors_discovered_and_unknown_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        compute_factors(panel(), ["momentun_20d"])
    assert factor_requirements(["volume_surge_5d"])["volume_surge_5d"]["warmup_bars"] == 21
    with pytest.raises(ValueError, match="unique"):
        compute_factors(pd.concat([panel(), panel()]), ["momentum_20d"])


def test_cutoff_blocks_future_and_keeps_warmup_missing():
    p = panel()
    kwargs = {"names": ["momentum_20d", "reversal_5d"], "cutoff": "2024-04-01"}
    first = factor_report(p, **kwargs)
    p.loc[p.date >= "2024-04-01", "close"] = np.nan
    assert factor_report(p, **kwargs) == first
    assert 0 < first["coverage"][0]["coverage"] < 1
    assert first["ic_decay"][0]["rank_ic"] > 0.99
    json.dumps(first, allow_nan=False)


def test_sparse_cross_section_is_unavailable():
    p = panel().query("symbol == '1'")
    report = factor_report(p, ["momentum_20d"], cutoff="2025-01-01")
    assert all(row["rank_ic"] is None for row in report["ic_decay"])
