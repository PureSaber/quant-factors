from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from quant_factors.core import compute_factors, list_factors
from quant_factors.neutralize import neutralize_cross_section


def cmd_list(args: argparse.Namespace) -> int:
    factors = list_factors()
    if args.json:
        print(json.dumps(factors, indent=2, ensure_ascii=False))
    else:
        for name, desc in factors.items():
            print(f"{name:24} {desc}")
    return 0


def cmd_compute(args: argparse.Namespace) -> int:
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        input_path = Path(cfg["input"])
        output_path = Path(cfg.get("output", "data/factors.parquet"))
        factors = cfg.get("factors") or list(list_factors())
    else:
        input_path = Path(args.input)
        output_path = Path(args.output)
        factors = args.factors.split(",") if args.factors else list(list_factors())

    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    result = compute_factors(df, factors=factors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        result.to_parquet(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
    print(f"wrote {output_path} rows={len(result)} factors={factors}")
    return 0


def cmd_neutralize(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    input_path = Path(cfg["input"])
    output_path = Path(cfg.get("output", "data/factors_neutral.parquet"))
    cols = cfg.get("cols") or []
    by = cfg.get("by") or ["industry", "market_cap"]

    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    result = neutralize_cross_section(df, cols=cols, by=by)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        result.to_parquet(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
    print(f"wrote {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quant-factors")
    sub = p.add_subparsers(dest="command", required=True)

    lst = sub.add_parser("list", help="List built-in factors")
    lst.add_argument("--json", action="store_true")
    lst.set_defaults(func=cmd_list)

    compute = sub.add_parser("compute", help="Compute factors from panel data")
    compute.add_argument("--config", help="YAML config with input/output/factors")
    compute.add_argument("--input", help="Input parquet/csv (when no --config)")
    compute.add_argument("--output", default="data/factors.parquet")
    compute.add_argument("--factors", help="Comma-separated factor names")
    compute.set_defaults(func=cmd_compute)

    neutralize = sub.add_parser("neutralize", help="Cross-sectional neutralize")
    neutralize.add_argument("--config", required=True)
    neutralize.set_defaults(func=cmd_neutralize)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compute" and not args.config and not args.input:
        parser.error("compute requires --config or --input")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
