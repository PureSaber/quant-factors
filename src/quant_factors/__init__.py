"""Shared factor computations."""

from importlib.metadata import PackageNotFoundError, version

from quant_factors.contracts_v2 import (
    AsOfSpec,
    AuxiliarySource,
    ContractViolation,
    FactorDependency,
    FactorFrameManifest,
    FactorInputRef,
    FactorSpec,
    FixedPoint,
    FrequencySpec,
    OutputField,
    SourceLineage,
)
from quant_factors.core import compute_factors, list_factors
from quant_factors.factor_frame import (
    FactorFrame,
    FactorFrameError,
    compute_factor_frame,
    compute_factor_frame_from_fixture,
)
from quant_factors.pit_v2 import (
    PitError,
    VerifiedAuxiliaryInput,
    load_verified_auxiliary_source,
)

try:
    __version__ = version("quant-factors")
except PackageNotFoundError:  # pragma: no cover - editable/source tree
    __version__ = "0.3.0"

from quant_factors.validation import (
    ValidationSplit,
    audit_feature_availability,
    benjamini_hochberg,
    purged_kfold_splits,
    walk_forward_splits,
)

__all__ = [
    "AsOfSpec",
    "AuxiliarySource",
    "ContractViolation",
    "FactorDependency",
    "FactorFrame",
    "FactorFrameError",
    "FactorFrameManifest",
    "FactorInputRef",
    "FactorSpec",
    "FixedPoint",
    "FrequencySpec",
    "OutputField",
    "PitError",
    "SourceLineage",
    "ValidationSplit",
    "VerifiedAuxiliaryInput",
    "__version__",
    "audit_feature_availability",
    "benjamini_hochberg",
    "compute_factor_frame",
    "compute_factor_frame_from_fixture",
    "compute_factors",
    "list_factors",
    "load_verified_auxiliary_source",
    "purged_kfold_splits",
    "walk_forward_splits",
]
