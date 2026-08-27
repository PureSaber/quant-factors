import numpy as np
import pandas as pd

from quant_factors.validation import (
    audit_feature_availability,
    benjamini_hochberg,
    probabilistic_sharpe_ratio,
    purged_kfold_splits,
    summarize_fold_stability,
    walk_forward_splits,
)


def test_walk_forward_has_strict_oos_and_embargo() -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    splits = walk_forward_splits(
        dates, train_size=8, test_size=3, step_size=3, embargo_size=2, expanding=False
    )
    assert len(splits) == 3
    assert splits[0].train_end < splits[0].test_start
    assert splits[0].test_indices[0] - splits[0].train_indices[-1] == 3


def test_purged_kfold_removes_overlapping_labels_and_embargo() -> None:
    starts = pd.date_range("2024-01-01", periods=12, freq="D")
    ends = starts + pd.Timedelta(days=2)
    splits = purged_kfold_splits(starts, ends, n_splits=3, embargo_size=1)
    for split in splits:
        train_starts = starts[split.train_indices]
        train_ends = ends[split.train_indices]
        assert not ((train_starts <= split.test_end) & (train_ends >= split.test_start)).any()
        assert not np.intersect1d(split.train_indices, split.test_indices).size


def test_benjamini_hochberg_controls_false_discoveries() -> None:
    result = benjamini_hochberg([0.001, 0.02, 0.2, 0.8], alpha=0.05)
    assert result["reject"].tolist() == [True, True, False, False]
    assert (result["adjusted_p_value"] >= result["p_value"]).all()


def test_feature_availability_audit_detects_future_value() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "pe": [10.0, 11.0],
            "pe_available_at": ["2024-01-01", "2024-01-03"],
        }
    )
    report = audit_feature_availability(frame, {"pe": "pe_available_at"})
    assert report.loc[0, "future_rows"] == 1


def test_sharpe_probability_and_stability_summary() -> None:
    assert probabilistic_sharpe_ratio(1.0, 0.0, 252) > 0.99
    summary = summarize_fold_stability(pd.DataFrame({"ic": [0.03, 0.01, -0.01]}), "ic")
    assert summary["folds"] == 3
    assert summary["positive_fold_ratio"] == 2 / 3
