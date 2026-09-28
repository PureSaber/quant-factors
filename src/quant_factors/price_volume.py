"""Opt-in price-volume formulas.

These factors are not part of the default registry. Callers must name them.
Windows that only rescale an existing formula are stored once. Formulas that need
a benchmark index or Fama-French factors are omitted, as are formula strings that
do not parse. Missing values stay missing; infinities are not rewritten as sentinels.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"(\d+\.\d+|\d+|[A-Z][A-Z0-9_]*|>=|<=|==|[+\-*/^(),?:<>=&|])")
_FIELDS = frozenset({"OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMOUNT", "VWAP"})
_DERIVED = frozenset({"RET", "DTM", "DBM", "TR", "HD", "LD", "SELF"})
_CALLS = frozenset(
    {
        "DELAY",
        "DELTA",
        "SUM",
        "MEAN",
        "STD",
        "CORR",
        "COVARIANCE",
        "RANK",
        "TSRANK",
        "TSMAX",
        "TSMIN",
        "MAX",
        "MIN",
        "SMA",
        "WMA",
        "DECAYLINEAR",
        "COUNT",
        "SUMIF",
        "REGBETA",
        "REGRESI",
        "PROD",
        "FILTER",
        "HIGHDAY",
        "LOWDAY",
        "LOG",
        "ABS",
        "SIGN",
        "SQRT",
        "SEQUENCE",
    }
)


class FormulaError(ValueError):
    """A price-volume formula cannot be parsed or evaluated."""


def normalize_formula(formula: str) -> str:
    """Collapse cosmetic differences without changing coefficients or windows."""

    text = formula.strip().rstrip(";").upper()
    text = (
        text.replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("–", "-")
        .replace("—", "-")
        .replace("－", "-")
    )
    text = re.sub(r"\s+", "", text)
    text = text.replace("./", "/").replace(".*", "*").replace("||", "|")
    text = re.sub(r"(?<![A-Z])OR(?![A-Z])", "|", text)
    text = text.replace("SMEAN", "SMA").replace("HGIH", "HIGH").replace("DELAT", "DELTA")
    text = text.replace("COVIANCE", "COVARIANCE").replace("-L)", "-LOW)")
    text = text.replace("STD(CLOSE:20),0", "STD(CLOSE,20):0")
    text = re.sub(r"SEQUENCE,(\d+)", r"SEQUENCE(\1)", text)
    text = re.sub(r"(?<![A-Z])MA\(", "MEAN(", text)
    text = re.sub(r"(?<![A-Z])VOL(?![A-Z])", "VOLUME", text)
    return text


def _tokenize(formula: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(formula):
        match = _TOKEN.match(formula, index)
        if match is None:
            raise FormulaError(f"Cannot read formula at: {formula[index : index + 12]}")
        tokens.append(match.group(1))
        index = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def pop(self) -> str:
        token = self.peek()
        if token is None:
            raise FormulaError("Formula ended early")
        self.index += 1
        return token

    def parse(self):
        node = self._ternary()
        if self.peek() is not None:
            raise FormulaError(f"Unexpected token {self.peek()}")
        return node

    def _ternary(self):
        condition = self._or()
        if self.peek() != "?":
            return condition
        self.pop()
        yes = self._ternary()
        if self.pop() != ":":
            raise FormulaError("Ternary expression is missing ':'")
        return ("ternary", condition, yes, self._ternary())

    def _or(self):
        node = self._and()
        while self.peek() == "|":
            op = self.pop()
            node = ("bin", op, node, self._and())
        return node

    def _and(self):
        node = self._cmp()
        while self.peek() == "&":
            op = self.pop()
            node = ("bin", op, node, self._cmp())
        return node

    def _cmp(self):
        node = self._add()
        while self.peek() in {">", "<", ">=", "<=", "=", "=="}:
            op = self.pop()
            node = ("bin", "==" if op == "=" else op, node, self._add())
        return node

    def _add(self):
        node = self._mul()
        while self.peek() in {"+", "-"}:
            op = self.pop()
            node = ("bin", op, node, self._mul())
        return node

    def _mul(self):
        node = self._pow()
        while self.peek() in {"*", "/"}:
            op = self.pop()
            node = ("bin", op, node, self._pow())
        return node

    def _pow(self):
        node = self._unary()
        if self.peek() == "^":
            self.pop()
            return ("bin", "^", node, self._pow())
        return node

    def _unary(self):
        if self.peek() == "-":
            self.pop()
            return ("unary", "-", self._unary())
        return self._primary()

    def _primary(self):
        token = self.pop()
        if token == "(":
            node = self._ternary()
            if self.pop() != ")":
                raise FormulaError("Missing ')'")
            return node
        if re.fullmatch(r"\d+\.\d+|\d+", token):
            return ("num", float(token))
        if token not in _FIELDS | _DERIVED | _CALLS and token != "SELF":
            raise FormulaError(f"Unknown name {token}")
        if self.peek() == "(":
            if token not in _CALLS:
                raise FormulaError(f"{token} is not a function")
            self.pop()
            args = []
            if self.peek() != ")":
                args.append(self._ternary())
                while self.peek() == ",":
                    self.pop()
                    args.append(self._ternary())
            if self.pop() != ")":
                raise FormulaError(f"Missing ')' after {token}")
            return ("call", token, tuple(args))
        if token in _CALLS:
            raise FormulaError(f"{token} is missing its arguments")
        return ("name", token)


def parse_formula(formula: str):
    """Parse one normalized formula into a call tree."""

    parser = _Parser(_tokenize(normalize_formula(formula)))
    return parser.parse()


def _names_of(node) -> set[str]:
    kind = node[0]
    if kind == "name":
        return {node[1]}
    if kind == "num":
        return set()
    if kind == "unary":
        return _names_of(node[2])
    if kind == "bin":
        return _names_of(node[2]) | _names_of(node[3])
    if kind == "ternary":
        return _names_of(node[1]) | _names_of(node[2]) | _names_of(node[3])
    found = set()
    for arg in node[2]:
        found |= _names_of(arg)
    return found


def required_columns(formula: str) -> list[str]:
    """Columns a formula reads. VWAP may be supplied directly or derived later."""

    names = _names_of(parse_formula(formula))
    columns = {name.lower() for name in names if name in _FIELDS and name != "VWAP"}
    if "RET" in names or "SELF" in names:
        columns.add("close")
    if "HD" in names:
        columns.add("high")
    if "LD" in names:
        columns.add("low")
    if "TR" in names or "DTM" in names or "DBM" in names:
        columns.update({"open", "high", "low", "close"})
    return sorted(columns)


def _max_window(node) -> int:
    kind = node[0]
    if kind == "num":
        return int(node[1]) if node[1].is_integer() else 0
    if kind == "name":
        return 0
    if kind == "unary":
        return _max_window(node[2])
    if kind == "bin":
        return max(_max_window(node[2]), _max_window(node[3]))
    if kind == "ternary":
        return max(_max_window(node[1]), _max_window(node[2]), _max_window(node[3]))
    return max((_max_window(arg) for arg in node[2]), default=0)


class _Panel:
    def __init__(self, frame: pd.DataFrame) -> None:
        if frame.duplicated(["symbol", "date"]).any():
            raise FormulaError("Price-volume factors require unique symbol/date rows")
        ordered = frame.sort_values(["symbol", "date"], kind="stable")
        self.index = ordered.index
        self.symbol = ordered["symbol"]
        self.date = ordered["date"]
        self.columns = ordered
        self._self = pd.Series(np.nan, index=ordered.index)

    def field(self, name: str) -> pd.Series:
        column = name.lower()
        if name == "VWAP":
            if "vwap" in self.columns:
                return self.columns["vwap"].astype(float)
            if {"amount", "volume"} <= set(self.columns.columns):
                volume = self.columns["volume"].astype(float).replace(0, np.nan)
                return self.columns["amount"].astype(float) / volume
            raise FormulaError("VWAP requires a vwap column or amount and volume")
        if column not in self.columns.columns:
            raise FormulaError(f"Missing column {column}")
        return self.columns[column].astype(float)

    def series(self, name: str) -> pd.Series:
        if name == "SELF":
            return self._self
        if name in _FIELDS:
            return self.field(name)
        if name == "RET":
            close = self.field("CLOSE")
            return close / self._shift(close, 1) - 1
        if name == "HD":
            high = self.field("HIGH")
            return high - self._shift(high, 1)
        if name == "LD":
            low = self.field("LOW")
            return self._shift(low, 1) - low
        if name == "TR":
            high, low, close = self.field("HIGH"), self.field("LOW"), self.field("CLOSE")
            previous = self._shift(close, 1)
            return _maximum(_maximum(high - low, (high - previous).abs()), (low - previous).abs())
        if name == "DTM":
            return self._directional_move(up=True)
        if name == "DBM":
            return self._directional_move(up=False)
        raise FormulaError(f"Unknown series {name}")

    def _directional_move(self, *, up: bool) -> pd.Series:
        open_, high, low = self.field("OPEN"), self.field("HIGH"), self.field("LOW")
        previous = self._shift(open_, 1)
        if up:
            reach = _maximum(high - open_, open_ - previous)
            return reach.where(open_ > previous, 0.0)
        reach = _maximum(open_ - low, previous - open_)
        return reach.where(open_ < previous, 0.0)

    def _shift(self, values: pd.Series, periods: int) -> pd.Series:
        return values.groupby(self.symbol, sort=False).shift(periods)

    def grouped(self, values: pd.Series):
        return values.groupby(self.symbol, sort=False)


def _number(node) -> float | None:
    if node[0] == "num":
        return node[1]
    if node[0] == "unary" and node[1] == "-" and node[2][0] == "num":
        return -node[2][1]
    return None


def _window(node, name: str) -> int:
    value = _number(node)
    if value is None or not value.is_integer() or value < 1:
        raise FormulaError(f"{name} needs a positive integer window")
    return int(value)


def _as_series(value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=index, dtype=float)


def _aligned_index(*values) -> pd.Index:
    for value in values:
        if isinstance(value, pd.Series):
            return value.index
    raise FormulaError("Comparison needs at least one series")


def _maximum(left, right):
    if not isinstance(left, pd.Series) and not isinstance(right, pd.Series):
        return max(left, right)
    index = _aligned_index(left, right)
    return _as_series(left, index).combine(_as_series(right, index), np.fmax)


def _minimum(left, right):
    if not isinstance(left, pd.Series) and not isinstance(right, pd.Series):
        return min(left, right)
    index = _aligned_index(left, right)
    return _as_series(left, index).combine(_as_series(right, index), np.fmin)


def _pair_rolling(
    panel: _Panel, left: pd.Series, right: pd.Series, window: int, method: str
) -> pd.Series:
    pieces = []
    for idx in panel.symbol.groupby(panel.symbol, sort=False).groups.values():
        rolling = left.loc[idx].rolling(window, min_periods=window)
        pieces.append(getattr(rolling, method)(right.loc[idx]))
    return pd.concat(pieces).reindex(left.index)


def _rolling_stat(panel: _Panel, values: pd.Series, window: int, method: str) -> pd.Series:
    rolling = panel.grouped(values).rolling(window, min_periods=window)
    if method == "std":
        result = rolling.std(ddof=1)
    elif method == "prod":
        result = rolling.apply(lambda window_values: np.prod(window_values), raw=True)
    else:
        result = getattr(rolling, method)()
    return result.reset_index(level=0, drop=True).reindex(values.index)


def _weighted(panel: _Panel, values: pd.Series, window: int, weights: np.ndarray) -> pd.Series:
    def dot(window_values: np.ndarray) -> float:
        if not np.isfinite(window_values).all():
            return np.nan
        return float(np.dot(window_values, weights))

    result = panel.grouped(values).rolling(window, min_periods=window).apply(dot, raw=True)
    return result.reset_index(level=0, drop=True).reindex(values.index)


def _ts_rank(panel: _Panel, values: pd.Series, window: int) -> pd.Series:
    def last_rank(window_values: np.ndarray) -> float:
        if not np.isfinite(window_values).all():
            return np.nan
        return float(pd.Series(window_values).rank(pct=True, method="average").iloc[-1])

    result = panel.grouped(values).rolling(window, min_periods=window).apply(last_rank, raw=True)
    return result.reset_index(level=0, drop=True).reindex(values.index)


def _days_since_extreme(panel: _Panel, values: pd.Series, window: int, *, high: bool) -> pd.Series:
    def ago(window_values: np.ndarray) -> float:
        if not np.isfinite(window_values).all():
            return np.nan
        position = int(np.argmax(window_values) if high else np.argmin(window_values))
        return float(window - 1 - position)

    result = panel.grouped(values).rolling(window, min_periods=window).apply(ago, raw=True)
    return result.reset_index(level=0, drop=True).reindex(values.index)


def _sma(panel: _Panel, values: pd.Series, length: int, weight: int) -> pd.Series:
    def smooth(series: pd.Series) -> pd.Series:
        previous = np.nan
        output = []
        for value in series.to_numpy(dtype=float):
            if not np.isfinite(value):
                output.append(np.nan)
                continue
            previous = (
                value
                if not np.isfinite(previous)
                else (weight * value + (length - weight) * previous) / length
            )
            output.append(previous)
        return pd.Series(output, index=series.index)

    return panel.grouped(values).apply(smooth).reset_index(level=0, drop=True).reindex(values.index)


def _regression(
    panel: _Panel, left: pd.Series, right: pd.Series | None, window: int, *, residual: bool
):
    def slope(window_y: np.ndarray, window_x: np.ndarray) -> float:
        if not np.isfinite(window_y).all() or not np.isfinite(window_x).all():
            return np.nan
        x_centered = window_x - window_x.mean()
        variance = float(np.dot(x_centered, x_centered))
        if variance == 0:
            return np.nan
        beta = float(np.dot(x_centered, window_y - window_y.mean()) / variance)
        if not residual:
            return beta
        alpha = float(window_y.mean() - beta * window_x.mean())
        return float(window_y[-1] - (alpha + beta * window_x[-1]))

    pieces = []
    sequence = np.arange(1, window + 1, dtype=float)
    for idx in panel.symbol.groupby(panel.symbol, sort=False).groups.values():
        y = left.loc[idx].to_numpy(dtype=float)
        x = sequence if right is None else right.loc[idx].to_numpy(dtype=float)
        values = np.full(len(idx), np.nan)
        for end in range(window - 1, len(idx)):
            window_x = sequence if right is None else x[end - window + 1 : end + 1]
            values[end] = slope(y[end - window + 1 : end + 1], window_x)
        pieces.append(pd.Series(values, index=idx))
    return pd.concat(pieces).reindex(left.index)


class _Evaluator:
    def __init__(self, panel: _Panel) -> None:
        self.panel = panel

    def eval(self, node):
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "name":
            return self.panel.series(node[1])
        if kind == "unary":
            return -self.eval(node[2])
        if kind == "ternary":
            condition = self._bool(self.eval(node[1]))
            yes = _as_series(self.eval(node[2]), self.panel.index)
            no = _as_series(self.eval(node[3]), self.panel.index)
            chosen = yes.where(condition, no)
            return chosen.mask(condition.isna())
        if kind == "bin":
            return self._bin(node[1], self.eval(node[2]), self.eval(node[3]))
        return self._call(node[1], node[2])

    def _bool(self, value) -> pd.Series:
        series = _as_series(value, self.panel.index)
        if str(series.dtype) == "boolean":
            return series
        if series.dtype == bool:
            return series.astype("boolean")
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.ne(0).mask(numeric.isna()).astype("boolean")

    def _bin(self, op: str, left, right):
        if op in {"&", "|"}:
            combined = (
                self._bool(left).__and__(self._bool(right))
                if op == "&"
                else self._bool(left).__or__(self._bool(right))
            )
            return combined
        if op in {">", "<", ">=", "<=", "=="}:
            method = {"==": "eq", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}[op]
            left_series = _as_series(left, self.panel.index)
            compared = getattr(left_series, method)(right)
            right_missing = right.isna() if isinstance(right, pd.Series) else False
            return compared.mask(left_series.isna() | right_missing)
        if op == "^":
            return _as_series(left, self.panel.index) ** right
        if op == "/":
            denominator = _as_series(right, self.panel.index).replace(0, np.nan)
            return _as_series(left, self.panel.index) / denominator
        if op == "+":
            return _as_series(left, self.panel.index) + right
        if op == "-":
            return _as_series(left, self.panel.index) - right
        if op == "*":
            return _as_series(left, self.panel.index) * right
        raise FormulaError(f"Unsupported operator {op}")

    def _call(self, name: str, args: tuple):
        if name in {"LOG", "ABS", "SIGN", "SQRT"}:
            if len(args) != 1:
                raise FormulaError(f"{name} expects one argument")
            values = _as_series(self.eval(args[0]), self.panel.index)
            if name == "LOG":
                return np.log(values.where(values > 0))
            if name == "ABS":
                return values.abs()
            if name == "SQRT":
                return np.sqrt(values.where(values >= 0))
            return np.sign(values)
        if name == "RANK":
            if len(args) != 1:
                raise FormulaError("RANK expects one argument")
            values = _as_series(self.eval(args[0]), self.panel.index)
            return values.groupby(self.panel.date, sort=False).rank(pct=True, method="average")
        if name == "DELAY":
            periods = 1 if len(args) == 1 else _window(args[1], name)
            return self.panel._shift(_as_series(self.eval(args[0]), self.panel.index), periods)
        if name == "DELTA":
            values = _as_series(self.eval(args[0]), self.panel.index)
            return values - self.panel._shift(values, _window(args[1], name))
        if name in {"SUM", "MEAN", "STD", "TSMAX", "TSMIN", "PROD"}:
            method = {
                "SUM": "sum",
                "MEAN": "mean",
                "STD": "std",
                "TSMAX": "max",
                "TSMIN": "min",
                "PROD": "prod",
            }[name]
            return _rolling_stat(
                self.panel,
                _as_series(self.eval(args[0]), self.panel.index),
                _window(args[1], name),
                method,
            )
        if name in {"MAX", "MIN"}:
            if len(args) != 2:
                raise FormulaError(f"{name} expects two arguments")
            window = _number(args[1])
            if window is not None and window.is_integer():
                return _rolling_stat(
                    self.panel,
                    _as_series(self.eval(args[0]), self.panel.index),
                    int(window),
                    "max" if name == "MAX" else "min",
                )
            left, right = self.eval(args[0]), self.eval(args[1])
            return _maximum(left, right) if name == "MAX" else _minimum(left, right)
        if name in {"CORR", "COVARIANCE"}:
            window = _window(args[2], name)
            return _pair_rolling(
                self.panel,
                _as_series(self.eval(args[0]), self.panel.index),
                _as_series(self.eval(args[1]), self.panel.index),
                window,
                "corr" if name == "CORR" else "cov",
            )
        if name == "TSRANK":
            return _ts_rank(
                self.panel, _as_series(self.eval(args[0]), self.panel.index), _window(args[1], name)
            )
        if name == "SMA":
            return _sma(
                self.panel,
                _as_series(self.eval(args[0]), self.panel.index),
                _window(args[1], name),
                _window(args[2], name),
            )
        if name == "DECAYLINEAR":
            window = _window(args[1], name)
            weights = np.arange(1, window + 1, dtype=float)
            return _weighted(
                self.panel,
                _as_series(self.eval(args[0]), self.panel.index),
                window,
                weights / weights.sum(),
            )
        if name == "WMA":
            window = _window(args[1], name)
            distance = np.arange(window - 1, -1, -1, dtype=float)
            weights = 0.9**distance
            return _weighted(
                self.panel,
                _as_series(self.eval(args[0]), self.panel.index),
                window,
                weights / weights.sum(),
            )
        if name in {"COUNT", "SUMIF"}:
            window = _window(args[1], name)
            if name == "COUNT":
                condition = self._bool(self.eval(args[0])).astype(float)
                return _rolling_stat(self.panel, condition, window, "sum")
            values = _as_series(self.eval(args[0]), self.panel.index)
            condition = self._bool(self.eval(args[2]))
            selected = values.where(condition, 0.0).mask(condition.isna())
            return _rolling_stat(self.panel, selected, window, "sum")
        if name in {"HIGHDAY", "LOWDAY"}:
            return _days_since_extreme(
                self.panel,
                _as_series(self.eval(args[0]), self.panel.index),
                _window(args[1], name),
                high=name == "HIGHDAY",
            )
        if name == "FILTER":
            values = _as_series(self.eval(args[0]), self.panel.index)
            return values.where(self._bool(self.eval(args[1])))
        if name in {"REGBETA", "REGRESI"}:
            left = _as_series(self.eval(args[0]), self.panel.index)
            if args[1][0] == "call" and args[1][1] == "SEQUENCE":
                window = _window(args[1][2][0], "SEQUENCE")
                right = None
            elif args[1][0] == "name" and args[1][1] == "SEQUENCE":
                window = _window(args[2], name)
                right = None
            else:
                window = _window(args[2], name)
                right = _as_series(self.eval(args[1]), self.panel.index)
            return _regression(self.panel, left, right, window, residual=name == "REGRESI")
        if name == "SEQUENCE":
            raise FormulaError("SEQUENCE is only valid inside REGBETA or REGRESI")
        raise FormulaError(f"Unsupported function {name}")


def evaluate_formula(formula: str, frame: pd.DataFrame) -> pd.Series:
    """Evaluate one formula on a long price panel."""

    normalized = normalize_formula(formula)
    tree = parse_formula(normalized)
    if "SELF" in normalized:
        return _evaluate_self(tree, frame)
    panel = _Panel(frame)
    values = _Evaluator(panel).eval(tree)
    numeric = pd.to_numeric(_as_series(values, panel.index), errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).reindex(frame.index)


def _evaluate_self(tree, frame: pd.DataFrame) -> pd.Series:
    panel = _Panel(frame)
    output = pd.Series(np.nan, index=panel.index)
    for idx in panel.symbol.groupby(panel.symbol, sort=False).groups.values():
        previous = 1.0
        for row in idx:
            panel._self.loc[row] = previous
            value = _Evaluator(panel).eval(tree).loc[row]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            previous = float(value) if np.isfinite(value) else previous
            output.loc[row] = previous
    numeric = pd.to_numeric(output, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).reindex(frame.index)


def warmup_bars(formula: str) -> int:
    return max(_max_window(parse_formula(formula)), 1)


def list_price_volume_factors() -> dict[str, str]:
    from quant_factors.price_volume_catalog import PRICE_VOLUME_CATALOG

    return {name: entry["description"] for name, entry in PRICE_VOLUME_CATALOG.items()}


def price_volume_requirements(name: str) -> dict:
    from quant_factors.price_volume_catalog import PRICE_VOLUME_CATALOG

    entry = PRICE_VOLUME_CATALOG[name]
    return {
        "columns": list(entry["columns"]),
        "warmup_bars": int(entry["warmup_bars"]),
        "pit_required": False,
        "pit_columns": [],
        "dependencies": [name],
    }


def compute_price_volume_factors(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Compute named price-volume formulas. Names must come from the opt-in catalog."""

    from quant_factors.price_volume_catalog import PRICE_VOLUME_CATALOG

    unknown = sorted(set(names) - set(PRICE_VOLUME_CATALOG))
    if unknown:
        raise FormulaError(f"Unknown price-volume factors: {unknown}")
    result = frame.copy()
    for name in names:
        result[name] = evaluate_formula(PRICE_VOLUME_CATALOG[name]["formula"], result)
    return result
