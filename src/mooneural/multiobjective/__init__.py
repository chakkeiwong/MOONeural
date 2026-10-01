"""TensorFlow-native multiobjective gradient aggregation utilities.

This package exposes the runtime reusable-library surface for shared
multiobjective aggregation utilities plus explicit state helpers.  Runtime
exposure is not by itself equivalent to reviewed-canonical promotion; use the
reusable-library status-audit artifacts for promotion status.
"""

from mooneural.multiobjective.api import (
    BLOCKED_METHODS,
    IMPLEMENTED_METHODS,
    SUPPORTED_METHODS,
    UnsupportedMethodError,
    aggregate,
    create_gradnorm_state,
    create_famo_state,
    famo_after_step,
    gradnorm_update,
)
from mooneural.multiobjective.types import (
    AggregationResult,
    FAMOState,
    FAMOUpdateResult,
    GradNormState,
    GradNormUpdateResult,
)

__all__ = [
    "AggregationResult",
    "BLOCKED_METHODS",
    "FAMOState",
    "FAMOUpdateResult",
    "GradNormState",
    "GradNormUpdateResult",
    "IMPLEMENTED_METHODS",
    "SUPPORTED_METHODS",
    "UnsupportedMethodError",
    "aggregate",
    "create_gradnorm_state",
    "create_famo_state",
    "famo_after_step",
    "gradnorm_update",
]
