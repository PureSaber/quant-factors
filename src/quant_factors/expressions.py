"""Validated, causal factor expressions for research-only screening.

The expression language is deliberately small. It is parsed and interpreted from
an AST; Python ``eval`` and arbitrary object access are never used.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quant_factors.core import FACTOR_REGISTRY, compute_factors

MAX_EXPRESSIONS = 32
MAX_EXPRESSION_LENGTH = 500
MAX_AST_NODES = 100
MAX_AST_DEPTH = 12
MAX_WINDOW = 252

RAW_INPUTS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover_rate",
        "market_cap",
        "pe_ratio",
        "pb_ratio",
    }
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_UNARY_FUNCTIONS = frozenset({"abs", "log", "sqrt", "sign", "rank"})
_BINARY_FUNCTIONS = frozenset({"minimum", "maximum"})
_WINDOW_FUNCTIONS = frozenset(
    {"rolling_mean", "rolling_std", "rolling_min", "rolling_max", "rolling_sum", "zscore"}
)
_LAG_FUNCTIONS = frozenset({"lag", "shift", "delta", "pct_change"})
_FUNCTIONS = (
    _UNARY_FUNCTIONS
    | _BINARY_FUNCTIONS
    | _WINDOW_FUNCTIONS
    | _LAG_FUNCTIONS
    | {
        "clip",
        "where",
    }
)
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}
_COMPARE_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class ExpressionError(ValueError):
    """Raised when a factor expression is invalid or cannot be computed."""


def _constant_int(node: ast.AST, *, minimum: int) -> int:
    if not isinstance(node, ast.Constant) or type(node.value) is not int:
        raise ExpressionError("Window and lag arguments must be literal integers")
    value = node.value
    if value < minimum or value > MAX_WINDOW:
        raise ExpressionError(f"Window and lag arguments must be in [{minimum}, {MAX_WINDOW}]")
    return value


def _constant_number(node: ast.AST, *, allow_none: bool = False) -> float | None:
    if allow_none and isinstance(node, ast.Constant) and node.value is None:
        return None
    sign = 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        node = node.operand
    if not isinstance(node, ast.Constant) or type(node.value) not in {int, float}:
        raise ExpressionError("This function argument must be a literal finite number")
    value = sign * float(node.value)
    if not math.isfinite(value) or abs(value) > 1e12:
        raise ExpressionError("Numeric literals must be finite and no larger than 1e12")
    return value


def _depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + (max((_depth(child) for child in children), default=0))


class _ExpressionValidator(ast.NodeVisitor):
    def __init__(self, allowed_names: set[str]) -> None:
        self.allowed_names = allowed_names
        self.dependencies: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Compare,
            ast.BoolOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.BitAnd,
            ast.BitOr,
            ast.UAdd,
            ast.USub,
            ast.Invert,
            ast.And,
            ast.Or,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Eq,
            ast.NotEq,
        )
        if not isinstance(node, allowed):
            raise ExpressionError(f"Unsupported expression syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names:
            raise ExpressionError(f"Unknown expression name: {node.id}")
        self.dependencies.add(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value is None:
            raise ExpressionError("null is allowed only as a clip bound")
        if type(node.value) is bool:
            return
        if type(node.value) not in {int, float}:
            raise ExpressionError("Only numeric, boolean and null literals are allowed")
        _constant_number(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if type(node.op) not in _BIN_OPS:
            raise ExpressionError(f"Unsupported binary operator: {type(node.op).__name__}")
        if isinstance(node.op, ast.Pow):
            exponent = _constant_number(node.right)
            if exponent is None or exponent != int(exponent) or not -4 <= exponent <= 4:
                raise ExpressionError("Power exponent must be a literal integer in [-4, 4]")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
            raise ExpressionError(f"Unsupported unary operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARE_OPS:
            raise ExpressionError("Only a single simple comparison is allowed")
        self.visit(node.left)
        self.visit(node.comparators[0])

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, (ast.And, ast.Or)) or len(node.values) < 2:
            raise ExpressionError("Boolean expressions require and/or with at least two values")
        for value in node.values:
            self.visit(value)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ExpressionError("Only documented expression functions may be called")
        if node.keywords:
            raise ExpressionError("Expression functions accept positional arguments only")
        name = node.func.id
        if name in _UNARY_FUNCTIONS and len(node.args) != 1:
            raise ExpressionError(f"{name} requires one argument")
        if name in _BINARY_FUNCTIONS and len(node.args) != 2:
            raise ExpressionError(f"{name} requires two arguments")
        if name in _WINDOW_FUNCTIONS:
            if len(node.args) != 2:
                raise ExpressionError(f"{name} requires a value and a window")
            _constant_int(node.args[1], minimum=2)
        if name in _LAG_FUNCTIONS:
            if len(node.args) != 2:
                raise ExpressionError(f"{name} requires a value and a non-negative lag")
            _constant_int(node.args[1], minimum=0 if name in {"lag", "shift"} else 1)
        if name == "clip":
            if len(node.args) != 3:
                raise ExpressionError("clip requires a value, lower bound and upper bound")
            lower = _constant_number(node.args[1], allow_none=True)
            upper = _constant_number(node.args[2], allow_none=True)
            if lower is None and upper is None:
                raise ExpressionError("clip requires at least one finite bound")
            if lower is not None and upper is not None and lower > upper:
                raise ExpressionError("clip lower bound cannot exceed upper bound")
        if name == "where" and len(node.args) != 3:
            raise ExpressionError("where requires a condition, true value and false value")
        for index, argument in enumerate(node.args):
            if name in _WINDOW_FUNCTIONS | _LAG_FUNCTIONS and index == 1:
                continue
            if name == "clip" and index in {1, 2}:
                continue
            self.visit(argument)


def _parse_expressions(
    expressions: Mapping[str, str] | None,
) -> tuple[dict[str, str], dict[str, ast.Expression], dict[str, set[str]]]:
    if expressions is None:
        return {}, {}, {}
    if not isinstance(expressions, Mapping):
        raise ExpressionError("factor_expressions must be a mapping of names to expressions")
    if len(expressions) > MAX_EXPRESSIONS:
        raise ExpressionError(f"At most {MAX_EXPRESSIONS} custom factor expressions are allowed")
    normalized: dict[str, str] = {}
    for name, expression in expressions.items():
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise ExpressionError(f"Invalid custom factor name: {name!r}")
        if name in FACTOR_REGISTRY or name in RAW_INPUTS or name in _FUNCTIONS:
            raise ExpressionError(f"Custom factor name is reserved: {name}")
        if not isinstance(expression, str) or not expression.strip():
            raise ExpressionError(f"Expression for {name} must be a nonempty string")
        expression = expression.strip()
        if len(expression) > MAX_EXPRESSION_LENGTH:
            raise ExpressionError(
                f"Expression for {name} exceeds {MAX_EXPRESSION_LENGTH} characters"
            )
        normalized[name] = expression

    allowed = set(RAW_INPUTS) | set(FACTOR_REGISTRY) | set(normalized)
    trees: dict[str, ast.Expression] = {}
    dependencies: dict[str, set[str]] = {}
    for name, expression in normalized.items():
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ExpressionError(f"Invalid expression for {name}: {exc.msg}") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_AST_NODES or _depth(tree) > MAX_AST_DEPTH:
            raise ExpressionError(f"Expression for {name} exceeds the complexity limit")
        validator = _ExpressionValidator(allowed)
        validator.visit(tree)
        if name in validator.dependencies:
            raise ExpressionError(f"Expression for {name} references itself")
        trees[name] = tree
        dependencies[name] = validator.dependencies

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ExpressionError("Custom factor expressions contain a dependency cycle")
        if name in visited:
            return
        visiting.add(name)
        for dependency in dependencies[name] & set(normalized):
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in normalized:
        visit(name)
    return normalized, trees, dependencies


def validate_expressions(mapping: Mapping[str, str] | None) -> dict[str, str]:
    """Validate and normalize a custom factor expression mapping.

    Validation is fail-closed: unknown names, calls, attributes, subscripts,
    comprehensions, imports and dependency cycles are rejected.
    """

    normalized, _, _ = _parse_expressions(mapping)
    return normalized


def _builtin_requirement(name: str) -> dict:
    tokens = name.split("_")
    window = next(
        (int(token[:-1]) for token in tokens if token.endswith("d") and token[:-1].isdigit()), 0
    )
    columns = ["close"]
    if name.startswith(("volume_", "turnover_")):
        columns = ["volume"]
    elif name.startswith("amihud_"):
        columns = ["close", "volume"]
    elif name in {"pe_inv", "pb_inv"}:
        columns = ["pe_ratio" if name == "pe_inv" else "pb_ratio"]
    if name == "volume_surge_5d":
        window = 20
    return {
        "columns": columns,
        "warmup_bars": window + 1,
        "pit_required": name in {"pe_inv", "pb_inv"},
    }


def _temporal_extra(node: ast.AST) -> int:
    own = 0
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in _WINDOW_FUNCTIONS:
            own = _constant_int(node.args[1], minimum=2) - 1
        elif node.func.id in _LAG_FUNCTIONS:
            own = _constant_int(node.args[1], minimum=0 if node.func.id in {"lag", "shift"} else 1)
    children = [
        child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.operator)
    ]
    return own + max((_temporal_extra(child) for child in children), default=0)


def expression_requirements(
    names: Sequence[str], expressions: Mapping[str, str] | None = None
) -> dict[str, dict]:
    """Return transitive source columns, warmup and PIT requirements."""

    normalized, trees, dependencies = _parse_expressions(expressions)
    requested = list(names)
    if any(not isinstance(name, str) for name in requested) or len(set(requested)) != len(
        requested
    ):
        raise ExpressionError("Factor names must be unique strings")
    unknown = set(requested) - set(FACTOR_REGISTRY) - set(normalized)
    if unknown:
        raise ExpressionError(f"Unknown factors: {sorted(unknown)}")
    memo: dict[str, dict] = {}

    def resolve(name: str) -> dict:
        if name in memo:
            return memo[name]
        if name in FACTOR_REGISTRY:
            result = {**_builtin_requirement(name), "dependencies": [name]}
        elif name in RAW_INPUTS:
            result = {
                "columns": [name],
                "warmup_bars": 1,
                "pit_required": name in {"pe_ratio", "pb_ratio"},
                "dependencies": [name],
            }
        else:
            parts = [resolve(dependency) for dependency in dependencies[name]]
            base_warmup = max((part["warmup_bars"] for part in parts), default=1)
            result = {
                "columns": sorted({column for part in parts for column in part["columns"]}),
                "warmup_bars": base_warmup + _temporal_extra(trees[name].body),
                "pit_required": any(part["pit_required"] for part in parts),
                "dependencies": sorted(dependencies[name]),
                "expression": normalized[name],
            }
        memo[name] = result
        return result

    return {name: resolve(name) for name in requested}


class _Interpreter:
    def __init__(self, context: dict[str, object], symbols: pd.Series, dates: pd.Series) -> None:
        self.context = context
        self.symbols = symbols
        self.dates = dates

    def evaluate(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.context[node.id]
        if isinstance(node, ast.UnaryOp):
            value = self.evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
            return ~value
        if isinstance(node, ast.BinOp):
            return _BIN_OPS[type(node.op)](self.evaluate(node.left), self.evaluate(node.right))
        if isinstance(node, ast.Compare):
            return _COMPARE_OPS[type(node.ops[0])](
                self.evaluate(node.left), self.evaluate(node.comparators[0])
            )
        if isinstance(node, ast.BoolOp):
            values = [self.evaluate(value) for value in node.values]
            operation = operator.and_ if isinstance(node.op, ast.And) else operator.or_
            result = values[0]
            for value in values[1:]:
                result = operation(result, value)
            return result
        if isinstance(node, ast.Call):
            return self._call(node.func.id, [self.evaluate(argument) for argument in node.args])
        raise ExpressionError(f"Unsupported expression syntax: {type(node).__name__}")

    def _series(self, value) -> pd.Series:
        if isinstance(value, pd.Series):
            return value
        return pd.Series(value, index=self.symbols.index)

    def _call(self, name: str, args: list):
        value = self._series(args[0]) if args else None
        if name == "abs":
            return value.abs()
        if name == "log":
            return np.log(value)
        if name == "sqrt":
            return np.sqrt(value)
        if name == "sign":
            return np.sign(value)
        if name == "rank":
            return value.groupby(self.dates, sort=False).rank(pct=True)
        if name == "minimum":
            return np.minimum(value, args[1])
        if name == "maximum":
            return np.maximum(value, args[1])
        if name == "clip":
            return value.clip(lower=args[1], upper=args[2])
        if name == "where":
            condition = self._series(args[0]).fillna(False).astype(bool)
            return self._series(args[1]).where(condition, self._series(args[2]))
        if name in {"lag", "shift"}:
            return value.groupby(self.symbols, sort=False).shift(args[1])
        if name == "delta":
            return value.groupby(self.symbols, sort=False).diff(args[1])
        if name == "pct_change":
            return value.groupby(self.symbols, sort=False).pct_change(args[1], fill_method=None)
        if name in _WINDOW_FUNCTIONS:
            window = args[1]
            grouped = value.groupby(self.symbols, sort=False)
            if name == "zscore":
                mean = grouped.transform(
                    lambda series: series.rolling(window, min_periods=window).mean()
                )
                std = grouped.transform(
                    lambda series: series.rolling(window, min_periods=window).std()
                )
                return (value - mean) / std.replace(0, np.nan)
            method = name.removeprefix("rolling_")
            return grouped.transform(
                lambda series: getattr(series.rolling(window, min_periods=window), method)()
            )
        raise ExpressionError(f"Unsupported expression function: {name}")


def compute_research_factors(
    frame: pd.DataFrame,
    names: Sequence[str],
    expressions: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compute built-in and custom factors through one causal research entry point."""

    normalized, trees, dependencies = _parse_expressions(expressions)
    requested = list(names)
    expression_requirements(requested, normalized)
    if not requested:
        return frame.copy()
    custom_needed: set[str] = set()

    def collect(name: str) -> None:
        if name not in normalized or name in custom_needed:
            return
        custom_needed.add(name)
        for dependency in dependencies[name]:
            collect(dependency)

    for name in requested:
        collect(name)
    builtin_needed = sorted(
        {
            dependency
            for name in custom_needed
            for dependency in dependencies[name]
            if dependency in FACTOR_REGISTRY
        }
        | (set(requested) & set(FACTOR_REGISTRY))
    )
    direct_inputs = {
        dependency
        for name in custom_needed
        for dependency in dependencies[name]
        if dependency in RAW_INPUTS
    }
    required_columns = {"date", "symbol"} | direct_inputs
    if builtin_needed:
        # The existing built-in core always requires close and treats missing volume,
        # P/E and P/B as optional all-missing inputs.
        required_columns.add("close")
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ExpressionError(f"Missing columns: {missing}")
    result = compute_factors(frame, builtin_needed) if builtin_needed else frame.copy()
    result = result.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    context: dict[str, object] = {
        name: result[name]
        for name in set(RAW_INPUTS) | set(builtin_needed)
        if name in result.columns
    }
    interpreter = _Interpreter(context, result["symbol"], result["date"])
    remaining = set(custom_needed)
    while remaining:
        ready = sorted(name for name in remaining if not (dependencies[name] & remaining))
        if not ready:  # Defensive; cycles are rejected during validation.
            raise ExpressionError("Custom factor expressions contain a dependency cycle")
        for name in ready:
            value = interpreter.evaluate(trees[name].body)
            series = interpreter._series(value).replace([np.inf, -np.inf], np.nan)
            result[name] = pd.to_numeric(series, errors="coerce")
            context[name] = result[name]
            remaining.remove(name)
    helper_columns = (set(builtin_needed) | custom_needed) - set(requested) - set(frame.columns)
    if helper_columns:
        result = result.drop(columns=sorted(helper_columns))
    return result
