import numpy as np
import pandas as pd
import pytest

from quant_factors.combinatorial import purged_combinatorial_splits
from quant_factors.research import factor_report


def test_disjoint_test_blocks_purge_only_real_label_overlap_and_keep_audit():
    starts = pd.date_range("2024-01-01", periods=30)
    ends = starts + pd.Timedelta(days=2)
    folds = purged_combinatorial_splits(starts, ends, blocks=5, test_blocks=2, embargo=1)
    assert len(folds) == 10
    for fold in folds:
        assert not set(fold["train_indices"]) & set(fold["test_indices"])
        for train in fold["train_indices"]:
            assert all(
                ends[train] < starts[test] or starts[train] > ends[test]
                for test in fold["test_indices"]
            )
    spaced = next(f for f in folds if f["test_blocks"] == (0, 4))
    assert 15 in spaced["train_indices"]
    with pytest.raises(ValueError):
        purged_combinatorial_splits(starts, ends, embargo=-1)


def test_factor_report_preserves_delisted_sample_and_evidenced_zero(monkeypatch):
    days = pd.date_range("2024-01-01", periods=8)
    data = pd.DataFrame(
        [
            {"date": day, "symbol": symbol, "close": 10 + i, "signal": i + ord(symbol)}
            for symbol in "ABC"
            for i, day in enumerate(days)
            if symbol != "A" or i < 5
        ]
    )
    monkeypatch.setattr("quant_factors.research.compute_research_factors", lambda p, *_: p)
    monkeypatch.setattr("quant_factors.research.expression_requirements", lambda *_: {})
    terminal = pd.DataFrame(
        [
            {
                "sample_id": "A:2024-01-05",
                "date": days[5],
                "realized_return": -1.0,
                "available_at": "2024-01-06T23:00:00Z",
                "source": "zero",
                "horizon": 2,
            }
        ]
    )
    report = factor_report(
        data, ["signal"], cutoff="2024-01-10", horizons=(2,), terminal_labels=terminal
    )
    sample = next(r for r in report["label_samples"] if r["sample_id"] == "A:2024-01-05")
    assert sample["return"] == -1 and sample["status"] == "terminal"
    assert any(r["status"] == "missing_price" for r in report["label_samples"])
    assert len(report["label_samples"]) == len(data)


def test_future_perturbation_prefix_invariant_for_v2_factors():
    from quant_factors.core import compute_factors

    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60),
            "symbol": "A",
            "close": np.linspace(10, 20, 60),
            "return_close": np.linspace(100, 200, 60),
            "volume": 100,
            "free_float_shares": 10000,
            "amount": 1000,
            "volume_unit": "shares",
            "share_basis": "raw",
            "amount_unit": "currency",
            "currency": "CNY",
        }
    )
    names = ["turnover_rate_20d_v2", "amihud_illiq_20d_v2"]
    original = compute_factors(data, names)
    data.loc[40:, ["close", "return_close", "amount", "volume", "free_float_shares"]] *= 100
    changed = compute_factors(data, names)
    pd.testing.assert_frame_equal(original.iloc[:40], changed.iloc[:40])


def test_optional_shared_family_dependency_and_nested_adapters():
    from quant_factors.combinatorial import (
        dependent_mean_evidence,
        family_statistics,
        nested_factor_selection,
    )

    dates = pd.date_range("2024-01-01", periods=64)
    data = pd.DataFrame(
        np.random.default_rng(19).normal(0, 0.01, (64, 3)), index=dates, columns=list("abc")
    )
    result = family_statistics(data, periods_per_year=252, blocks=4)
    assert result["dsr"] and result["cscv"]
    evidence = dependent_mean_evidence(data, block_lengths=(3, 6), lags=6, repetitions=100)
    assert evidence["bootstrap"] and evidence["hac"]
    nested = nested_factor_selection(
        data,
        {"positive": {"direction": 1}, "negative": {"direction": -1}},
        fit=lambda frame, recipe: recipe["direction"],
        evaluate=lambda model, frame: model * frame.a.mean(),
        outer_train=30,
        outer_test=10,
        inner_train=10,
        inner_test=5,
    )
    assert nested["folds"]
