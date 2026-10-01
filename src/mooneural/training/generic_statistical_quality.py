"""Host-side, reusable upper-confidence estimators for held-out MSE samples."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _finite_nonnegative(name: str, value: Any) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a numeric scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _task_order(task_order: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(task_order)
    if not normalized or any(not isinstance(task, str) or not task for task in normalized):
        raise ValueError("task order must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("task order must be unique")
    return normalized


def _sample_contract(
    anchor_records: Sequence[Mapping[str, Any]],
    anchor_ids: Sequence[str],
    task_order: tuple[str, ...],
    *,
    minimum_anchors: int,
    record_schema: str | None,
) -> tuple[tuple[str, ...], dict[str, list[float]], dict[str, list[float]]]:
    records = tuple(anchor_records)
    identifiers = tuple(str(value) for value in anchor_ids)
    if type(minimum_anchors) is not int or minimum_anchors < 2:
        raise ValueError("minimum anchor count must be at least two")
    if (
        len(records) < minimum_anchors
        or len(records) != len(identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError("anchor inventory is incomplete or has duplicate IDs")
    central_samples = {task: [] for task in task_order}
    conservative_samples = {task: [] for task in task_order}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("anchor records must be mappings")
        if record_schema is not None and record.get("schema") != record_schema:
            raise ValueError("anchor record schema mismatch")
        central = record.get("central_normalized_mse")
        conservative = record.get("conservative_normalized_mse")
        if (
            not isinstance(central, Mapping)
            or not isinstance(conservative, Mapping)
            or tuple(central) != task_order
            or tuple(conservative) != task_order
        ):
            raise ValueError("anchor task inventory or order mismatch")
        for task in task_order:
            central_value = _finite_nonnegative(
                f"central {task}", central[task]
            )
            conservative_value = _finite_nonnegative(
                f"conservative {task}", conservative[task]
            )
            if conservative_value < central_value:
                raise ValueError("conservative MSE is below central MSE")
            central_samples[task].append(central_value)
            conservative_samples[task].append(conservative_value)
    return identifiers, central_samples, conservative_samples


def evaluate_upper_mean_mse(
    anchor_records: Sequence[Mapping[str, Any]],
    *,
    anchor_ids: Sequence[str],
    task_order: Sequence[str],
    minimum_anchors: int = 20,
    alpha: float = 0.05,
    threshold_mse: float = 0.04,
    record_schema: str | None = None,
    contract_id: str = "dsge_hmc.generic_upper_mean_mse.v1",
) -> dict[str, Any]:
    """Compute Bonferroni one-sided Student-t upper bounds from complete anchors."""
    tasks = _task_order(task_order)
    if type(alpha) not in (int, float) or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    threshold = _finite_nonnegative("threshold MSE", threshold_mse)
    identifiers, central_samples, conservative_samples = _sample_contract(
        anchor_records,
        anchor_ids,
        tasks,
        minimum_anchors=minimum_anchors,
        record_schema=record_schema,
    )
    from scipy.stats import t as student_t

    count = len(identifiers)
    critical = float(student_t.ppf(1.0 - float(alpha) / len(tasks), count - 1))
    if not math.isfinite(critical):
        raise ValueError("Student-t critical value is non-finite")
    task_results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        central = np.asarray(central_samples[task], dtype=np.float64)
        conservative = np.asarray(conservative_samples[task], dtype=np.float64)
        mean_mse = float(np.mean(central))
        conservative_mean = float(np.mean(conservative))
        standard_error = float(np.std(conservative, ddof=1) / math.sqrt(count))
        upper_mse = conservative_mean + critical * standard_error
        task_results[task] = {
            "sample_count": count,
            "mean_normalized_mse": mean_mse,
            "point_normalized_rms": math.sqrt(mean_mse),
            "conservative_mean_normalized_mse": conservative_mean,
            "standard_error_mse": standard_error,
            "critical_value": critical,
            "upper_normalized_mse": upper_mse,
            "upper_normalized_rms": math.sqrt(max(0.0, upper_mse)),
            "threshold_mse": threshold,
            "threshold_rms": math.sqrt(threshold),
            "passed": upper_mse <= threshold,
            "central_anchor_samples": central.tolist(),
            "conservative_anchor_samples": conservative.tolist(),
        }
    return {
        "schema": "dsge_hmc.generic_upper_mean_mse_result.v1",
        "contract_id": contract_id,
        "task_order": list(tasks),
        "alpha": float(alpha),
        "minimum_anchors": int(minimum_anchors),
        "confidence_adjustment": "Bonferroni across task order",
        "replication_unit": "complete independently seeded anchor",
        "anchor_ids": list(identifiers),
        "threshold_definition": {
            "normalized_mse": threshold,
            "normalized_rms": math.sqrt(threshold),
            "mse_matches_training_objective_coordinate": True,
        },
        "tasks": task_results,
        "passed": all(item["passed"] for item in task_results.values()),
        "nonclaims": [
            "This estimator does not establish equation, trajectory, posterior, HMC, or production validity.",
            "Rows within an anchor are not independent replications.",
        ],
    }


__all__ = ["evaluate_upper_mean_mse"]
