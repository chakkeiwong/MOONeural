"""Host arithmetic for supplied, design-bound replication bank vectors.

Callers qualify the frozen design, endpoints, banks and estimator before using
these helpers. Identity agreement here does not measure banks, verify their
anchors/providers, or establish their independence or the normal approximation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.stats import chi2
from scipy.stats import t as student_t

from .generic_training_contracts import canonical_json, stable_hash

PHASE5R_ACCEPTED_TARGET_SHA256 = "da608984cbd2397e3a2256d3c28250286249d8427745de4f8da4554624f54622"
PHASE5R_METRIC_PATHS = tuple(
    ("global_quality", "cells", cell, task, "upper_normalized_mse")
    for cell in ("default", "interior-policy", "interior-preference", "interior-structure")
    for task in ("derivative.euler", "derivative.phillips", "residual.euler", "residual.phillips")
) + tuple(
    ("local_quality", "tasks", task, "upper_normalized_mse")
    for task in ("parameter_derivative", "state_derivative", "value")
)
PHASE5R_METRIC_ORDER = tuple(
    f"global.{path[2]}.{path[3]}" if path[0] == "global_quality" else f"local.{path[2]}"
    for path in PHASE5R_METRIC_PATHS
)

_BANK_COUNT = 20
_SOURCE_THRESHOLD = 0.04
_SD_MULTIPLIER = 0.1
_MAXIMUM_CV = 0.5
_FAMILY_ALPHA = 0.05
_VARIANCE_BOUND_PROBABILITY = 0.95
_NORMAL_ASSUMPTION = "predeclared independent-normal source bank approximation"
_INTERVAL_DISPOSITION_RULE = (
    "With valid statistics, inclusive containment is PASS; strict disjointness is "
    "NOT_EQUIVALENT; intersection without containment is INCONCLUSIVE. "
    "Any defined profile/Boolean failure or strictly disjoint interval takes "
    "precedence over another valid crossing."
)
_NONCLAIMS = (
    "Supplied bank-vector arithmetic only; no endpoint measurement or scientific admission.",
    (
        "Caller must qualify bank/provider completeness, 20 anchors and per-unit contributions, "
        "actual seed/role disjointness, target arrays, replacements, and frozen endpoint identity."
    ),
)


def _digest(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifiers(name: str, values, *, count: int | None = None) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of IDs")
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate IDs")
    if count is not None and len(normalized) != count:
        raise ValueError(f"{name} requires exactly {count} entries")
    return normalized


def _json_numbers(value):
    if isinstance(value, np.ndarray):
        return _json_numbers(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_numbers(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


@dataclass(frozen=True)
class ReplicationStatisticsSpec:
    """Immutable declared identities and metric family for the fixed 20/20 design.

    Coordinate hashes must bind each bank's inputs, targets, metric coordinates,
    anchor identities and estimator definition. IDs are globally qualified seed
    registry bank IDs; excluded IDs cover training, validation, selection and
    untouched certification. Their underlying contents are the caller's duty.
    """

    design_sha256: str
    source_endpoint_sha256: str
    accepted_target_sha256: str
    bank_registry_sha256: str
    metric_order: tuple[str, ...]
    metric_paths: tuple[tuple[str, ...], ...]
    boolean_paths: tuple[tuple[str, ...], ...]
    accepted_values: tuple[float, ...]
    accepted_booleans: tuple[bool, ...]
    calibration_bank_ids: tuple[str, ...]
    comparison_bank_ids: tuple[str, ...]
    excluded_bank_ids: tuple[str, ...]
    calibration_coordinate_sha256s: tuple[str, ...]
    comparison_coordinate_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("design_sha256", "source_endpoint_sha256", "accepted_target_sha256", "bank_registry_sha256"):
            _digest(name, getattr(self, name))
        order = _identifiers("metric order", self.metric_order)
        if not order:
            raise ValueError("metric family must not be empty")
        object.__setattr__(self, "metric_order", order)
        for name in ("metric_paths", "boolean_paths"):
            supplied_paths = tuple(getattr(self, name))
            if any(isinstance(path, str) for path in supplied_paths):
                raise TypeError(f"{name} requires key sequences, not dotted strings")
            paths = tuple(tuple(path) for path in supplied_paths)
            if len(paths) != len(order) or len(set(paths)) != len(paths) or any(
                not path or any(not isinstance(part, str) or not part for part in path)
                for path in paths
            ):
                raise ValueError(f"{name} must bind each metric to a unique explicit key path")
            object.__setattr__(self, name, paths)
        accepted = np.asarray(self.accepted_values)
        if accepted.dtype.kind not in "fiu" or accepted.shape != (len(order),) or not np.all(
            np.isfinite(accepted)
        ):
            raise ValueError("accepted values must be a complete finite numeric vector")
        booleans = tuple(self.accepted_booleans)
        if len(booleans) != len(order) or any(type(value) is not bool for value in booleans):
            raise ValueError("accepted Booleans must be a complete Boolean vector")
        if booleans != tuple(bool(value <= _SOURCE_THRESHOLD) for value in accepted):
            raise ValueError("accepted Booleans disagree with the fixed 0.04 source threshold")
        object.__setattr__(self, "accepted_values", tuple(float(value) for value in accepted))
        object.__setattr__(self, "accepted_booleans", booleans)
        for name in ("calibration_bank_ids", "comparison_bank_ids", "excluded_bank_ids"):
            count = None if name == "excluded_bank_ids" else _BANK_COUNT
            object.__setattr__(self, name, _identifiers(name, getattr(self, name), count=count))
        groups = (self.calibration_bank_ids, self.comparison_bank_ids, self.excluded_bank_ids)
        if len(set().union(*groups)) != sum(len(group) for group in groups):
            raise ValueError("calibration, comparison and excluded bank IDs overlap")
        for name in ("calibration_coordinate_sha256s", "comparison_coordinate_sha256s"):
            hashes = _identifiers(name, getattr(self, name), count=_BANK_COUNT)
            for digest in hashes:
                _digest(name, digest)
            object.__setattr__(self, name, hashes)
        if set(self.calibration_coordinate_sha256s).intersection(self.comparison_coordinate_sha256s):
            raise ValueError("calibration and comparison coordinates overlap")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            schema="dsge_hmc.generic_replication_statistics_spec.v1",
            calibration_count=_BANK_COUNT,
            comparison_count=_BANK_COUNT,
            ddof=1,
            source_threshold=_SOURCE_THRESHOLD,
            sd_multiplier=_SD_MULTIPLIER,
            maximum_cv=_MAXIMUM_CV,
            family_alpha=_FAMILY_ALPHA,
            variance_bound_probability=_VARIANCE_BOUND_PROBABILITY,
            normality_assumption=_NORMAL_ASSUMPTION,
            interval_disposition_rule=_INTERVAL_DISPOSITION_RULE,
        )
        return json.loads(canonical_json(payload))

    def binding_hash(self) -> str:
        return stable_hash(self.to_dict())


def phase5r_statistics_spec(
    *,
    accepted_target_bytes: bytes,
    design_sha256: str,
    source_endpoint_sha256: str,
    bank_registry_sha256: str,
    calibration_bank_ids: tuple[str, ...],
    comparison_bank_ids: tuple[str, ...],
    excluded_bank_ids: tuple[str, ...],
    calibration_coordinate_sha256s: tuple[str, ...],
    comparison_coordinate_sha256s: tuple[str, ...],
) -> ReplicationStatisticsSpec:
    """Bind Phase 5R's exact accepted artifact and canonical 19-coordinate family.

    This adapter only extracts a target; all statistical arithmetic is generic.
    Agreement with recovery provenance and design qualification remain external.
    """
    if not isinstance(accepted_target_bytes, bytes) or hashlib.sha256(
        accepted_target_bytes
    ).hexdigest() != PHASE5R_ACCEPTED_TARGET_SHA256:
        raise ValueError("Phase 5R accepted target bytes do not match the controlling artifact")
    target = json.loads(accepted_target_bytes)

    def at_path(path):
        value = target
        for key in path:
            value = value[key]
        return value

    boolean_paths = tuple(path[:-1] + ("passed",) for path in PHASE5R_METRIC_PATHS)
    accepted_values = tuple(at_path(path) for path in PHASE5R_METRIC_PATHS)
    accepted_booleans = tuple(at_path(path) for path in boolean_paths)
    if accepted_values[16] != 0.05371185695178999 or accepted_booleans != (
        (True,) * 16 + (False, True, True)
    ):
        raise ValueError("Phase 5R accepted miss or Boolean profile changed")
    return ReplicationStatisticsSpec(
        design_sha256=design_sha256,
        source_endpoint_sha256=source_endpoint_sha256,
        accepted_target_sha256=PHASE5R_ACCEPTED_TARGET_SHA256,
        bank_registry_sha256=bank_registry_sha256,
        metric_order=PHASE5R_METRIC_ORDER,
        metric_paths=PHASE5R_METRIC_PATHS,
        boolean_paths=boolean_paths,
        accepted_values=accepted_values,
        accepted_booleans=accepted_booleans,
        calibration_bank_ids=calibration_bank_ids,
        comparison_bank_ids=comparison_bank_ids,
        excluded_bank_ids=excluded_bank_ids,
        calibration_coordinate_sha256s=calibration_coordinate_sha256s,
        comparison_coordinate_sha256s=comparison_coordinate_sha256s,
    )


@dataclass(frozen=True)
class ReplicationBankVectors:
    """A frozen snapshot of ordered endpoint vectors and supplied identities."""

    spec_sha256: str
    endpoint_sha256: str
    metric_order: tuple[str, ...]
    bank_ids: tuple[str, ...]
    coordinate_sha256s: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        _digest("statistics spec", self.spec_sha256)
        _digest("endpoint", self.endpoint_sha256)
        for name in ("metric_order", "bank_ids"):
            object.__setattr__(self, name, _identifiers(name, getattr(self, name)))
        coordinates = tuple(self.coordinate_sha256s)
        for digest in coordinates:
            _digest("bank coordinates", digest)
        object.__setattr__(self, "coordinate_sha256s", coordinates)
        values = np.asarray(self.values)
        if values.dtype.kind not in "fiu" or values.shape != (len(self.bank_ids), len(self.metric_order)):
            raise ValueError("bank vectors must have one numeric row per ID and one column per metric")
        if len(coordinates) != len(self.bank_ids):
            raise ValueError("bank coordinate inventory is incomplete")
        object.__setattr__(self, "values", tuple(tuple(float(value) for value in row) for row in values))

    def to_dict(self) -> dict[str, Any]:
        return _json_numbers(asdict(self))


def _bound_vectors(spec, banks, *, calibration, endpoint_sha256):
    if banks.spec_sha256 != spec.binding_hash():
        raise ValueError("bank statistics spec binding mismatch")
    if banks.endpoint_sha256 != endpoint_sha256:
        raise ValueError("bank frozen endpoint binding mismatch")
    if banks.metric_order != spec.metric_order:
        raise ValueError("bank metric order mismatch")
    expected_ids = spec.calibration_bank_ids if calibration else spec.comparison_bank_ids
    coordinates = spec.calibration_coordinate_sha256s if calibration else spec.comparison_coordinate_sha256s
    if banks.bank_ids != expected_ids:
        raise ValueError("bank IDs/order/count differ from the fixed registry")
    if banks.coordinate_sha256s != coordinates:
        raise ValueError("bank input/target coordinates differ from the fixed registry")
    return np.asarray(banks.values, dtype=np.float64)


def _mean_and_centered(values):
    shifted = values - values[0]
    mean_shift = shifted.mean(axis=0)
    return values[0] + mean_shift, shifted - mean_shift


def _profile(mean, spec, sigma_gate, stable):
    finite = np.isfinite(mean)
    booleans = [bool(value <= _SOURCE_THRESHOLD) if valid else None for value, valid in zip(mean, finite)]
    boolean_matches = [
        actual == expected if actual is not None else None
        for actual, expected in zip(booleans, spec.accepted_booleans)
    ]
    with np.errstate(all="ignore"):
        deviations = np.abs(mean - spec.accepted_values)
    profile_matches = [
        bool(deviation <= _SD_MULTIPLIER * sigma) if valid and sigma_valid else None
        for deviation, sigma, valid, sigma_valid in zip(deviations, sigma_gate, finite, stable)
    ]
    return {
        "booleans": booleans,
        "boolean_matches": boolean_matches,
        "boolean_passed": all(boolean_matches) if all(finite) else None,
        "absolute_profile_deviation": deviations,
        "profile_matches": profile_matches,
        "profile_passed": all(profile_matches) if all(finite & stable) else None,
    }


def _source_statistics(spec, banks):
    values = _bound_vectors(spec, banks, calibration=True, endpoint_sha256=spec.source_endpoint_sha256)
    critical = float(chi2.ppf(_VARIANCE_BOUND_PROBABILITY, _BANK_COUNT - 1))
    with np.errstate(all="ignore"):
        mean, centered = _mean_and_centered(values)
        sigma_point = np.sqrt(np.sum(centered ** 2, axis=0) / (_BANK_COUNT - 1))
        sigma_gate = sigma_point * math.sqrt((_BANK_COUNT - 1) / critical)
        coefficient_of_variation = sigma_point / np.abs(mean)
    stable = (
        np.isfinite(values).all(axis=0) & np.isfinite(mean)
        & np.isfinite(sigma_point) & (sigma_point > 0)
        & np.isfinite(sigma_gate) & (sigma_gate > 0)
        & np.isfinite(coefficient_of_variation) & (coefficient_of_variation > 0)
        & (coefficient_of_variation <= _MAXIMUM_CV)
    )
    profile = _profile(mean, spec, sigma_gate, stable)
    if not stable.all():
        status = "INCONCLUSIVE"
    elif not profile["profile_passed"] or not profile["boolean_passed"]:
        status = "NOT_EQUIVALENT"
    else:
        status = "SOURCE_PROFILE_PASS"
    return _json_numbers({
        "schema": "dsge_hmc.generic_source_calibration_statistics.v1",
        "status": status,
        "scope": "supplied_bank_vector_arithmetic",
        "spec": spec.to_dict(),
        "spec_sha256": spec.binding_hash(),
        "source_banks": banks.to_dict(),
        "U_accepted": spec.accepted_values,
        "B_accepted": spec.accepted_booleans,
        "U_source_current": mean,
        "B_source_current": profile["booleans"],
        "sigma_point": sigma_point,
        "sigma_gate": sigma_gate,
        "coefficient_of_variation": coefficient_of_variation,
        "sigma_stable": stable,
        "statistics_valid": bool(stable.all()),
        "chi2_critical": critical,
        "degrees_of_freedom": _BANK_COUNT - 1,
        "equivalence_half_width": _SD_MULTIPLIER * sigma_gate,
        "source_profile": profile,
        "nonclaims": _NONCLAIMS,
    })


@dataclass(frozen=True)
class SourceCalibrationResult:
    """Reusable source-only result computed from immutable supplied vectors."""

    spec: ReplicationStatisticsSpec
    banks: ReplicationBankVectors
    _payload_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_payload_json", canonical_json(_source_statistics(self.spec, self.banks)))

    def to_dict(self) -> dict[str, Any]:
        payload = json.loads(self._payload_json)
        payload["result_sha256"] = stable_hash(payload)
        return payload

    @property
    def status(self) -> str:
        return json.loads(self._payload_json)["status"]


def evaluate_source_calibration(
    spec: ReplicationStatisticsSpec, source_banks: ReplicationBankVectors
) -> SourceCalibrationResult:
    """Evaluate the existing source predicate after external design qualification.

    No generic checkpoint or completed training is required. Invalid sigma/CV
    gives INCONCLUSIVE. A defined profile/Boolean failure gives NOT_EQUIVALENT.
    Structural identity/count errors raise ValueError without dropping banks.
    """
    return SourceCalibrationResult(spec, source_banks)


def _interval_geometry(lower, upper, bound, sigma_valid):
    if not sigma_valid or not math.isfinite(lower) or not math.isfinite(upper):
        return "UNDEFINED"
    if lower >= -bound and upper <= bound:
        return "WITHIN_ENVELOPE"
    if upper < -bound or lower > bound:
        return "WHOLLY_OUTSIDE"
    return "CROSSES_BOUNDARY"


def evaluate_paired_replication(
    source_calibration: SourceCalibrationResult,
    source_pairs: ReplicationBankVectors,
    generic_pairs: ReplicationBankVectors,
    *,
    generic_endpoint_sha256: str,
) -> dict[str, Any]:
    """Evaluate 20 separately supplied pairs against the preserved calibration.

    EQUIVALENT is an arithmetic result only; authority/integrity and bank
    completeness are external. Singular covariance and exactly zero difference
    variance need no inverse, regularization or extra banks. Crossing intervals
    are INCONCLUSIVE; strictly disjoint intervals give NOT_EQUIVALENT when
    statistics are valid, taking precedence over another valid crossing.
    """
    spec = source_calibration.spec
    calibration = source_calibration.to_dict()
    source = _bound_vectors(spec, source_pairs, calibration=False, endpoint_sha256=spec.source_endpoint_sha256)
    generic = _bound_vectors(
        spec, generic_pairs, calibration=False,
        endpoint_sha256=_digest("generic endpoint", generic_endpoint_sha256),
    )
    critical = float(student_t.ppf(1.0 - _FAMILY_ALPHA / (2 * len(spec.metric_order)), _BANK_COUNT - 1))
    with np.errstate(all="ignore"):
        differences = generic - source
        mean_difference, centered = _mean_and_centered(differences)
        covariance = centered.T @ centered / (_BANK_COUNT - 1)
        standard_deviation = np.sqrt(np.diag(covariance))
        half_width = critical * standard_deviation / math.sqrt(_BANK_COUNT)
        intervals = np.column_stack((mean_difference - half_width, mean_difference + half_width))
        generic_mean, _ = _mean_and_centered(generic)
        source_pair_mean, _ = _mean_and_centered(source)
    pair_valid = bool(all(np.isfinite(values).all() for values in (
        source, generic, differences, covariance, intervals, generic_mean, source_pair_mean,
    )))
    sigma_gate = np.asarray(calibration["sigma_gate"], dtype=np.float64)
    stable = np.asarray(calibration["sigma_stable"], dtype=bool)
    profile = _profile(generic_mean, spec, sigma_gate, stable)
    envelope = _SD_MULTIPLIER * sigma_gate
    geometries = [
        _interval_geometry(lower, upper, bound, sigma_valid)
        for (lower, upper), bound, sigma_valid in zip(intervals, envelope, stable)
    ]
    paired_passed = pair_valid and all(item == "WITHIN_ENVELOPE" for item in geometries)
    paired_failed = calibration["statistics_valid"] and pair_valid and "WHOLLY_OUTSIDE" in geometries
    if not calibration["statistics_valid"] or not pair_valid:
        status = "INCONCLUSIVE"
    elif paired_failed or calibration["status"] == "NOT_EQUIVALENT" or any(
        profile[name] is False for name in ("profile_passed", "boolean_passed")
    ):
        status = "NOT_EQUIVALENT"
    elif paired_passed:
        status = "EQUIVALENT"
    else:
        status = "INCONCLUSIVE"
    result = _json_numbers({
        "schema": "dsge_hmc.generic_paired_replication_statistics.v1",
        "status": status,
        "scope": "supplied_bank_vector_arithmetic",
        "spec_sha256": spec.binding_hash(),
        "source_calibration": calibration,
        "source_calibration_sha256": calibration["result_sha256"],
        "source_pairs": source_pairs.to_dict(),
        "generic_pairs": generic_pairs.to_dict(),
        "generic_endpoint_sha256": generic_endpoint_sha256,
        "U_source_pair_mean": source_pair_mean,
        "U_replication": generic_mean,
        "B_replication": profile["booleans"],
        "generic_profile": profile,
        "D": differences,
        "mean_D": mean_difference,
        "covariance_D": covariance,
        "sd_D": standard_deviation,
        "degrees_of_freedom": _BANK_COUNT - 1,
        "family_alpha": _FAMILY_ALPHA,
        "per_coordinate_alpha": _FAMILY_ALPHA / len(spec.metric_order),
        "student_t_critical": critical,
        "confidence_intervals": intervals,
        "equivalence_half_width": envelope,
        "interval_geometry": geometries,
        "paired_passed": paired_passed,
        "paired_failed": paired_failed,
        "paired_statistics_valid": pair_valid,
        "interval_disposition_rule": _INTERVAL_DISPOSITION_RULE,
        "nonclaims": _NONCLAIMS,
    })
    result["result_sha256"] = stable_hash(result)
    return result


__all__ = [
    "PHASE5R_ACCEPTED_TARGET_SHA256",
    "PHASE5R_METRIC_ORDER",
    "PHASE5R_METRIC_PATHS",
    "ReplicationBankVectors",
    "ReplicationStatisticsSpec",
    "SourceCalibrationResult",
    "evaluate_paired_replication",
    "evaluate_source_calibration",
    "phase5r_statistics_spec",
]
