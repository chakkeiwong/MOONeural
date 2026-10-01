"""Bounded method screens, not statistical tests of endpoint equivalence."""

from __future__ import annotations

import math

import numpy as np

PASSED = "METHOD_CONFORMANCE_SCREEN_PASSED"
INCONCLUSIVE = "INCONCLUSIVE_DIAGNOSTIC"
INVALID = "INVALID_EVIDENCE"
RTOL = 1e-12
ATOL = 1e-14


def compare_arrays(candidate, source):
    """Keep shape/finiteness defects separate from numerical disagreement."""
    if not candidate or set(candidate) != set(source):
        return {"decision": INVALID, "reason": "array inventory mismatch", "fields": {}}
    fields = {}
    for name in sorted(source):
        actual, reference = np.asarray(candidate[name]), np.asarray(source[name])
        valid = (actual.shape == reference.shape and actual.dtype == reference.dtype
                 and actual.dtype.kind in "biuf" and np.isfinite(actual).all()
                 and np.isfinite(reference).all())
        if not valid:
            fields[name] = {"valid": False, "within_envelope": False,
                            "actual_shape": list(actual.shape), "source_shape": list(reference.shape),
                            "actual_dtype": str(actual.dtype), "source_dtype": str(reference.dtype)}
            continue
        difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
        allowance = ATOL + RTOL * np.abs(reference.astype(np.float64))
        fields[name] = {
            "valid": True, "within_envelope": bool(np.all(difference <= allowance)),
            "max_abs": float(np.max(difference, initial=0)),
            "max_scaled": float(np.max(difference / allowance, initial=0)),
            "max_relative": float(np.max(difference / np.maximum(np.abs(reference), ATOL), initial=0)),
            "shape": list(actual.shape), "dtype": str(actual.dtype),
        }
    decision = (INVALID if any(not field["valid"] for field in fields.values()) else
                PASSED if all(field["within_envelope"] for field in fields.values()) else INCONCLUSIVE)
    return {"decision": decision, "fields": fields, "rtol": RTOL, "atol": ATOL}


def disposition(comparison, *, structural, decisions_equal):
    if not structural or any(value is not True for value in structural.values()):
        return INVALID
    if comparison["decision"] == INVALID:
        return INVALID
    if not decisions_equal or comparison["decision"] != PASSED:
        return INCONCLUSIVE
    return PASSED


def required_cases(stage):
    if stage == "one":
        return {f"arm-{arm}-update-27000-{kind}" for arm in range(5)
                for kind in ("shadow", "repeat-1", "repeat-2")}
    if stage == "ten":
        return ({f"arm-{arm}-update-{update}-{kind}" for arm in range(5)
                 for update in range(27000, 27010) for kind in ("shadow", "free")}
                | {f"arm-{arm}-reload" for arm in range(5)})
    raise ValueError("method screen stage must be one or ten")


def summarize(stage, cases):
    names = [case["case_id"] for case in cases]
    expected = required_cases(stage)
    complete = len(names) == len(set(names)) and set(names) == expected
    if not complete:
        return {"decision": INVALID, "complete": False, "missing": sorted(expected - set(names)),
                "unexpected": sorted(set(names) - expected), "case_count": len(names)}
    required = [case for case in cases if not case["case_id"].endswith("-free")]
    valid_statuses = {PASSED, INCONCLUSIVE, INVALID}
    decisions = [case.get("decision") for case in required]
    decision = (INVALID if any(case.get("decision") not in valid_statuses or case["decision"] == INVALID for case in cases) else
                INCONCLUSIVE if INCONCLUSIVE in decisions else PASSED)
    return {"decision": decision, "complete": True, "case_count": len(cases),
            "required_case_count": len(required),
            "diagnostic_case_count": len(cases) - len(required),
            "inconclusive_cases": [case["case_id"] for case in required if case.get("decision") == INCONCLUSIVE],
            "invalid_cases": [case["case_id"] for case in required if case.get("decision") not in {PASSED, INCONCLUSIVE}]}


def remaining_budget(attempts, *, maximum_seconds=7200, maximum_launches=4):
    spent = 0.0
    for attempt in attempts:
        seconds = attempt.get("seconds", attempt["limit_seconds"])
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid campaign accounting")
        spent += seconds
    return {"seconds": max(0.0, maximum_seconds - spent),
            "launches": max(0, maximum_launches - len(attempts)), "spent_seconds": spent}
