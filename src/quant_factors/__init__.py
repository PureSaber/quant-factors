"""Shared factor computations."""

from importlib.metadata import PackageNotFoundError, version

from quant_factors.core import compute_factors, list_factors

try:
    __version__ = version("quant-factors")
except PackageNotFoundError:  # pragma: no cover - editable/source tree
    __version__ = "0.2.0"

from quant_factors.validation import (
    ValidationSplit,
    audit_feature_availability,
    benjamini_hochberg,
    purged_kfold_splits,
    walk_forward_splits,
)

__all__ = [
    "ValidationSplit",
    "__version__",
    "audit_feature_availability",
    "benjamini_hochberg",
    "compute_factors",
    "list_factors",
    "purged_kfold_splits",
    "walk_forward_splits",
]
