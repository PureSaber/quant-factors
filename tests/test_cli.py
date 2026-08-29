import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cli_list_json() -> None:
    out = subprocess.check_output(
        [sys.executable, "-m", "quant_factors.cli", "list", "--json"], text=True
    )
    data = json.loads(out)
    assert "momentum_20d" in data
    assert len(data) >= 12


def test_cli_compute_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="B"),
            "symbol": ["C"] * 30,
            "close": range(30),
            "volume": 1.0,
        }
    )
    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    df.to_csv(inp, index=False)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "quant_factors.cli",
            "compute",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--factors",
            "momentum_5d",
        ]
    )
    result = pd.read_csv(out)
    assert "momentum_5d" in result.columns
