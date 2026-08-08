"""Shared factor computations."""

from importlib.metadata import PackageNotFoundError, version

from quant_factors.core import compute_factors, list_factors

try:
    __version__ = version("quant-factors")
except PackageNotFoundError:  # pragma: no cover - editable/source tree
    __version__ = "0.1.0"

__all__ = ["__version__", "compute_factors", "list_factors"]
