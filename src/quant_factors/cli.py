from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from quant_factors.core import list_factors
from quant_factors.expressions import compute_research_factors, validate_expressions
from quant_factors.neutralize import neutralize_cross_section
from quant_factors.research import factor_report


def _load_mapping(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


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
        expressions = cfg.get("factor_expressions") or {}
    else:
        input_path = Path(args.input)
        output_path = Path(args.output)
        factors = args.factors.split(",") if args.factors else list(list_factors())
        expressions = _load_mapping(args.expressions) if args.expressions else {}

    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    result = compute_research_factors(df, names=factors, expressions=expressions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        result.to_parquet(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
    print(f"wrote {output_path} rows={len(result)} factors={factors}")
    return 0


def cmd_validate_expressions(args: argparse.Namespace) -> int:
    expressions = validate_expressions(_load_mapping(args.expressions))
    print(json.dumps(expressions, indent=2, ensure_ascii=False))
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    cfg = _load_mapping(args.config)
    input_path = Path(cfg["input"])
    output_path = Path(cfg.get("output", "data/factor-report.json"))
    frame = (
        pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    )
    report = factor_report(
        frame,
        cfg["factors"],
        expressions=cfg.get("factor_expressions") or {},
        horizons=tuple(cfg.get("horizons") or (1, 5, 20)),
        cutoff=cfg["cutoff"],
        start=cfg.get("start"),
        end=cfg.get("end"),
        baseline_names=tuple(cfg.get("baseline_factors") or ()),
        neutralize_by=tuple(cfg.get("neutralize_by") or ()),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")
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
    compute.add_argument("--expressions", help="YAML/JSON custom expression mapping")
    compute.set_defaults(func=cmd_compute)

    validate = sub.add_parser("validate-expressions", help="Validate a custom expression mapping")
    validate.add_argument("--expressions", required=True)
    validate.set_defaults(func=cmd_validate_expressions)

    screen = sub.add_parser("screen", help="Write a descriptive factor screening report")
    screen.add_argument("--config", required=True)
    screen.set_defaults(func=cmd_screen)

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
