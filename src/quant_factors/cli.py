from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from quant_factors.core import compute_factors, list_factors


def cmd_list(_: argparse.Namespace) -> int:
    print(json.dumps(list_factors(), indent=2, ensure_ascii=False))
    return 0


def cmd_compute(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    input_path = Path(cfg["input"])
    output_path = Path(cfg.get("output", "data/factors.parquet"))
    factors = cfg.get("factors") or list(list_factors())

    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    result = compute_factors(df, factors=factors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        result.to_parquet(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
    print(f"wrote {output_path} rows={len(result)} factors={factors}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quant-factors")
    sub = p.add_subparsers(dest="command", required=True)

    lst = sub.add_parser("list", help="List built-in factors")
    lst.set_defaults(func=cmd_list)

    compute = sub.add_parser("compute", help="Compute factors from YAML config")
    compute.add_argument("--config", required=True)
    compute.set_defaults(func=cmd_compute)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
