from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quant_factors.cli import main
from quant_factors.core import compute_factors, factor_requires_fundamental, list_factors
from quant_factors.neutralize import _market_cap_bins, neutralize_cross_section
from quant_factors.validation import (
    _ordered_dates,
    audit_feature_availability,
    benjamini_hochberg,
    probabilistic_sharpe_ratio,
    purged_kfold_splits,
    summarize_fold_stability,
    walk_forward_splits,
)


def _panel(rows: int = 25) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "symbol": ["AAA"] * rows,
            "close": np.arange(rows, dtype=float) + 10,
            "volume": np.arange(rows, dtype=float) + 100,
            "pe_ratio": np.arange(rows, dtype=float) + 10,
            "pb_ratio": np.arange(rows, dtype=float) + 2,
            "industry": ["A" if i % 2 else "B" for i in range(rows)],
            "market_cap": np.arange(rows, dtype=float) + 1,
        }
    )


def test_cli_direct_entrypoints_and_parquet_paths(tmp_path: Path, capsys) -> None:
    assert main(["list"]) == 0
    assert "momentum_20d" in capsys.readouterr().out
    assert main(["list", "--json"]) == 0
    assert "momentum_20d" in json.loads(capsys.readouterr().out)

    data = _panel()
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "out.csv"
    data.to_csv(input_csv, index=False)
    assert (
        main(
            [
                "compute",
                "--input",
                str(input_csv),
                "--output",
                str(output_csv),
                "--factors",
                "momentum_5d",
            ]
        )
        == 0
    )
    assert "momentum_5d" in pd.read_csv(output_csv)

    config = tmp_path / "compute.yaml"
    output_parquet = tmp_path / "nested" / "out.parquet"
    config.write_text(
        yaml.safe_dump(
            {"input": str(input_csv), "output": str(output_parquet), "factors": ["pe_inv"]}
        ),
        encoding="utf-8",
    )
    assert main(["compute", "--config", str(config)]) == 0
    assert "pe_inv" in pd.read_parquet(output_parquet)

    neutral_config = tmp_path / "neutralize.yaml"
    neutral_output = tmp_path / "nested" / "neutral.parquet"
    neutral_config.write_text(
        yaml.safe_dump(
            {
                "input": str(input_csv),
                "output": str(neutral_output),
                "cols": ["pe_ratio", "not_present"],
                "by": ["industry", "market_cap"],
            }
        ),
        encoding="utf-8",
    )
    assert main(["neutralize", "--config", str(neutral_config)]) == 0
    assert neutral_output.is_file()

    with pytest.raises(SystemExit):
        main(["compute"])


def test_core_governance_edges_and_fundamental_contract() -> None:
    data = _panel()
    result = compute_factors(data, factors=list(list_factors()))
    assert {"pe_inv", "pb_inv", "amihud_illiq_20d"}.issubset(result.columns)
    no_volume = data.drop(columns=["volume", "pe_ratio", "pb_ratio"])
    result = compute_factors(no_volume, factors=["volume_surge_5d", "pe_inv", "pb_inv", "unknown"])
    assert result["pe_inv"].isna().all() and result["pb_inv"].isna().all()
    assert factor_requires_fundamental("pe_inv")
    assert not factor_requires_fundamental("momentum_20d")
    with pytest.raises(ValueError, match="Missing columns"):
        compute_factors(pd.DataFrame({"date": [], "symbol": []}), factors=[])


def test_neutralize_governance_edges() -> None:
    series = pd.Series([np.nan, np.nan])
    assert _market_cap_bins(series).isna().all()
    assert _market_cap_bins(pd.Series([1.0, 1.0])).isna().all()
    plain = pd.DataFrame({"date": ["2024-01-01"], "factor": [1.0]})
    assert neutralize_cross_section(plain, ["factor"], ["industry"]).equals(plain)
    grouped = neutralize_cross_section(
        _panel(6), cols=["close", "missing"], by=["industry", "market_cap"]
    )
    assert "_mcap_bin" not in grouped.columns


def test_validation_rejects_invalid_inputs_and_empty_results() -> None:
    with pytest.raises(ValueError, match="missing values"):
        _ordered_dates(["2024-01-01", None])
    with pytest.raises(ValueError, match="sorted"):
        _ordered_dates(["2024-01-02", "2024-01-01"])
    with pytest.raises(ValueError, match="positive"):
        walk_forward_splits([], train_size=0, test_size=1)
    with pytest.raises(ValueError, match="embargo"):
        walk_forward_splits([], train_size=1, test_size=1, embargo_size=-1)
    with pytest.raises(ValueError, match="step_size"):
        walk_forward_splits([], train_size=1, test_size=1, step_size=-1)
    assert (
        walk_forward_splits(pd.date_range("2024-01-01", periods=2), train_size=2, test_size=1) == []
    )

    starts = pd.date_range("2024-01-01", periods=4, freq="D")
    with pytest.raises(ValueError, match="equal length"):
        purged_kfold_splits(starts, starts[:2])
    with pytest.raises(ValueError, match="precedes"):
        purged_kfold_splits(starts, starts - pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="between"):
        purged_kfold_splits(starts, starts, n_splits=1)
    with pytest.raises(ValueError, match="embargo"):
        purged_kfold_splits(starts, starts, n_splits=2, embargo_size=-1)
    with pytest.raises(ValueError, match="no training"):
        purged_kfold_splits(starts, starts + pd.Timedelta(days=10), n_splits=2)

    with pytest.raises(ValueError, match="non-empty"):
        benjamini_hochberg([])
    with pytest.raises(ValueError, match="finite"):
        benjamini_hochberg([0.1, np.nan])
    with pytest.raises(ValueError, match="finite"):
        benjamini_hochberg([-0.1])
    with pytest.raises(ValueError, match="one-dimensional"):
        benjamini_hochberg([[-0.1]])
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg([0.1], alpha=1)
    with pytest.raises(ValueError, match="at least"):
        probabilistic_sharpe_ratio(1, 0, 1)
    with pytest.raises(ValueError, match="variance"):
        probabilistic_sharpe_ratio(1, 0, 10, kurtosis=-10)


def test_validation_audits_and_fold_summary_edges() -> None:
    frame = pd.DataFrame({"when": ["2024-01-01"], "feature": [1.0], "available": [None]})
    with pytest.raises(ValueError, match="observation"):
        audit_feature_availability(frame, {}, observation_col="date")
    with pytest.raises(ValueError, match="invalid"):
        audit_feature_availability(
            pd.DataFrame({"date": ["bad"], "feature": [1.0], "available": ["2024-01-01"]}),
            {"feature": "available"},
        )
    with pytest.raises(ValueError, match="pair"):
        audit_feature_availability(
            pd.DataFrame({"date": ["2024-01-01"], "feature": [1.0]}),
            {"feature": "available"},
        )
    report = audit_feature_availability(
        frame.rename(columns={"when": "date"}), {"feature": "available"}
    )
    assert report.loc[0, "missing_availability_rows"] == 1

    with pytest.raises(ValueError, match="missing fold"):
        summarize_fold_stability(pd.DataFrame(), "ic")
    with pytest.raises(ValueError, match="no finite"):
        summarize_fold_stability(pd.DataFrame({"ic": ["bad"]}), "ic")
    assert summarize_fold_stability(pd.DataFrame({"ic": [0.1, -0.2]}), "ic")["worst_fold"] == -0.2
