"""Population orchestration for replicas with local method switching.

This module deliberately does not train TensorFlow variables.  It owns the
pure state-selection part of a population-based training loop so that policy,
optimizer, and controller transfer semantics can be tested without a DSGE
model or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc


METHODS = (
    "cagrad",
    "pcgrad",
    "mgda",
    "gradnorm",
    "weighted_normalized_sum",
)


def _finite_objectives(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("population_objectives_invalid")
    return result


@dataclass(frozen=True)
class PopulationConfig:
    replica_count: int = 4
    synchronization_update_cap: int = 300
    synchronization_count: int = 2
    archive_size: int = 4
    absolute_tolerance: float = 1.0e-12
    selection_seed: int = 20260723
    movement_floor: float = 1.0e-7
    movement_patience: int = 3

    def validate(self) -> None:
        for value, name in (
            (self.replica_count, "replica_count"),
            (self.synchronization_update_cap, "synchronization_update_cap"),
            (self.synchronization_count, "synchronization_count"),
            (self.archive_size, "archive_size"),
            (self.movement_patience, "movement_patience"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"population_{name}_invalid")
        if self.archive_size != self.replica_count:
            raise ValueError("population_archive_must_equal_population")
        if not math.isfinite(float(self.absolute_tolerance)) or self.absolute_tolerance < 0.0:
            raise ValueError("population_tolerance_invalid")
        if type(self.selection_seed) is not int or self.selection_seed < 0:
            raise ValueError("population_selection_seed_invalid")
        if not math.isfinite(float(self.movement_floor)) or self.movement_floor < 0.0:
            raise ValueError("population_movement_floor_invalid")

    def record(self) -> dict[str, Any]:
        self.validate()
        return {
            "replica_count": self.replica_count,
            "synchronization_update_cap": self.synchronization_update_cap,
            "synchronization_count": self.synchronization_count,
            "archive_size": self.archive_size,
            "absolute_tolerance": self.absolute_tolerance,
            "selection_seed": self.selection_seed,
            "movement_floor": self.movement_floor,
            "movement_patience": self.movement_patience,
        }


def relative_parameter_movement(
    before: Sequence[Any], after: Sequence[Any]
) -> float:
    before_values = tuple(np.asarray(value, dtype=np.float64) for value in before)
    after_values = tuple(np.asarray(value, dtype=np.float64) for value in after)
    if not before_values or len(before_values) != len(after_values):
        raise ValueError("population_movement_inventory_invalid")
    numerator_squared = 0.0
    denominator_squared = 0.0
    for left, right in zip(before_values, after_values, strict=True):
        if left.shape != right.shape or not np.all(
            np.isfinite(left) & np.isfinite(right)
        ):
            raise ValueError("population_movement_values_invalid")
        numerator_squared += float(np.sum(np.square(right - left)))
        denominator_squared += float(np.sum(np.square(left)))
    return math.sqrt(numerator_squared) / max(1.0, math.sqrt(denominator_squared))


def next_stagnation_streak(
    previous: int, movement: float, *, movement_floor: float = 1.0e-7
) -> int:
    if type(previous) is not int or previous < 0:
        raise ValueError("population_stagnation_streak_invalid")
    if not math.isfinite(float(movement)) or movement < 0.0:
        raise ValueError("population_movement_invalid")
    if not math.isfinite(float(movement_floor)) or movement_floor < 0.0:
        raise ValueError("population_movement_floor_invalid")
    return previous + 1 if movement <= movement_floor else 0


def pairwise_objective_distances(
    objective_vectors: Mapping[int, Sequence[float]],
) -> dict[str, Any]:
    """Summarize objective-space diversity without ranking replicas."""
    checked = {
        int(replica_id): np.asarray(_finite_objectives(values), dtype=np.float64)
        for replica_id, values in objective_vectors.items()
    }
    if len(checked) < 2 or any(value.shape != next(iter(checked.values())).shape for value in checked.values()):
        raise ValueError("population_diversity_objectives_invalid")
    rows = []
    identifiers = sorted(checked)
    for left_index, left in enumerate(identifiers):
        for right in identifiers[left_index + 1 :]:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "distance": float(np.linalg.norm(checked[left] - checked[right])),
                }
            )
    distances = [row["distance"] for row in rows]
    return {
        "pairs": rows,
        "minimum_distance": min(distances),
        "maximum_distance": max(distances),
        "identical_pair_count": sum(value == 0.0 for value in distances),
    }


def summarize_method_progress(
    progress: Sequence[Mapping[str, Any]],
    *,
    replica_count: int,
    methods: Sequence[str] = METHODS,
) -> list[dict[str, Any]]:
    """Aggregate explanatory method and fallback telemetry by replica."""
    method_order = tuple(str(method) for method in methods)
    if type(replica_count) is not int or replica_count < 1 or not method_order:
        raise ValueError("population_progress_summary_config_invalid")
    summaries = []
    for replica_id in range(replica_count):
        occupancy = {method: 0 for method in method_order}
        rate_minima = {method: None for method in method_order}
        fallback_count = 0
        fallback_streak = 0
        longest_fallback_streak = 0
        errors = {method: 0 for method in method_order}
        for sweep in progress:
            matching = [
                record
                for record in sweep.get("records", [])
                if int(record.get("replica", -1)) == replica_id
            ]
            if len(matching) > 1:
                raise ValueError("population_progress_duplicate_replica_record")
            if not matching or "method" not in matching[0]:
                fallback_streak = 0
                continue
            record = matching[0]
            method = str(record["method"])
            if method not in occupancy:
                raise ValueError("population_progress_method_invalid")
            occupancy[method] += 1
            rate = float(record["rate"])
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError("population_progress_rate_invalid")
            current_minimum = rate_minima[method]
            rate_minima[method] = rate if current_minimum is None else min(current_minimum, rate)
            failed = bool(record.get("fallback", False))
            fallback_count += int(failed)
            fallback_streak = fallback_streak + 1 if failed else 0
            longest_fallback_streak = max(longest_fallback_streak, fallback_streak)
            for failed_method in record.get("method_errors", {}):
                if failed_method not in errors:
                    raise ValueError("population_progress_error_method_invalid")
                errors[failed_method] += 1
        summaries.append(
            {
                "replica": replica_id,
                "method_occupancy": occupancy,
                "rate_minima": rate_minima,
                "fallback_count": fallback_count,
                "longest_fallback_streak": longest_fallback_streak,
                "method_error_counts": errors,
                "methods_never_visited": [
                    method for method, count in occupancy.items() if count == 0
                ],
            }
        )
    return summaries


def synchronization_reason(
    update_counts: Sequence[int],
    stagnation_streaks: Sequence[int],
    *,
    update_cap: int = 300,
    movement_patience: int = 3,
    population_count: int | None = None,
) -> str | None:
    counts = tuple(update_counts)
    streaks = tuple(stagnation_streaks)
    if (
        not counts
        or len(counts) != len(streaks)
        or type(update_cap) is not int
        or update_cap < 1
        or type(movement_patience) is not int
        or movement_patience < 1
        or (
            population_count is not None
            and (type(population_count) is not int or population_count < len(streaks))
        )
        or any(type(value) is not int or value < 0 for value in (*counts, *streaks))
    ):
        raise ValueError("population_synchronization_state_invalid")
    if all(value >= update_cap for value in counts):
        return "all_replicas_reached_update_cap"
    stagnant = sum(value >= movement_patience for value in streaks)
    denominator = len(streaks) if population_count is None else population_count
    if stagnant > denominator / 2.0:
        return "strict_majority_stagnant"
    return None


@dataclass
class ContinuousMethodController:
    """Validation-free method cycling with per-method learning rates."""

    learning_rates: dict[str, float]
    current_method: str = "cagrad"
    method_order: tuple[str, ...] = METHODS
    local_stagnation_streak: int = 0
    switch_count: int = 0
    method_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.learning_rates = {
            str(method): float(rate) for method, rate in self.learning_rates.items()
        }
        if not self.method_counts:
            self.method_counts = {method: 0 for method in self.method_order}
        self.validate()

    def validate(self) -> None:
        if (
            tuple(self.method_order) != METHODS
            or self.current_method not in self.method_order
            or set(self.learning_rates) != set(self.method_order)
            or any(
                not math.isfinite(rate) or rate <= 0.0
                for rate in self.learning_rates.values()
            )
            or set(self.method_counts) != set(self.method_order)
            or any(type(value) is not int or value < 0 for value in self.method_counts.values())
            or type(self.local_stagnation_streak) is not int
            or self.local_stagnation_streak < 0
            or type(self.switch_count) is not int
            or self.switch_count < 0
        ):
            raise ValueError("population_continuous_controller_invalid")

    def next_update(self) -> dict[str, Any]:
        self.validate()
        return {
            "method": self.current_method,
            "learning_rate": self.learning_rates[self.current_method],
        }

    def record_update(
        self,
        movement: float,
        *,
        movement_floor: float = 1.0e-7,
        movement_patience: int = 3,
    ) -> dict[str, Any]:
        method = self.current_method
        self.method_counts[method] += 1
        self.local_stagnation_streak = next_stagnation_streak(
            self.local_stagnation_streak,
            movement,
            movement_floor=movement_floor,
        )
        switched = False
        if self.local_stagnation_streak >= movement_patience:
            self.learning_rates[method] /= 2.0
            position = self.method_order.index(method)
            self.current_method = self.method_order[(position + 1) % len(self.method_order)]
            self.local_stagnation_streak = 0
            self.switch_count += 1
            switched = True
        self.validate()
        return {
            "updated_method": method,
            "movement": float(movement),
            "switched": switched,
            "next_method": self.current_method,
            "learning_rates": dict(self.learning_rates),
        }

    def record(self) -> dict[str, Any]:
        self.validate()
        return {
            "learning_rates": dict(self.learning_rates),
            "current_method": self.current_method,
            "method_order": list(self.method_order),
            "local_stagnation_streak": self.local_stagnation_streak,
            "switch_count": self.switch_count,
            "method_counts": dict(self.method_counts),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ContinuousMethodController":
        required = {
            "learning_rates",
            "current_method",
            "method_order",
            "local_stagnation_streak",
            "switch_count",
            "method_counts",
        }
        if set(record) != required:
            raise ValueError("population_continuous_controller_schema_invalid")
        return cls(
            learning_rates=dict(record["learning_rates"]),
            current_method=str(record["current_method"]),
            method_order=tuple(str(value) for value in record["method_order"]),
            local_stagnation_streak=int(record["local_stagnation_streak"]),
            switch_count=int(record["switch_count"]),
            method_counts={
                str(method): int(value)
                for method, value in dict(record["method_counts"]).items()
            },
        )


@dataclass
class ReplicaState:
    """Serializable state owned by one independently trained replica."""

    policy: tuple[np.ndarray, ...]
    optimizer: tuple[np.ndarray, ...]
    controller: Mapping[str, Any]
    method: str
    seed: int
    # Neural update rates keyed by method identity.  These are hyperparameter
    # state, not TensorFlow variables, and are copied across PBT boundaries.
    learning_rates: Mapping[str, float] = field(default_factory=dict)
    recipe_counter: int = 0

    def clone(
        self,
        *,
        child_seed: int | None = None,
        child_method: str | None = None,
        initial_controller: Mapping[str, Any] | None = None,
    ) -> "ReplicaState":
        policy = tuple(np.asarray(value).copy() for value in self.policy)
        optimizer = tuple(np.asarray(value).copy() for value in self.optimizer)
        if not policy or not optimizer:
            raise ValueError("population_state_inventory_invalid")
        rates = {
            str(method): float(rate) for method, rate in self.learning_rates.items()
        }
        if not rates or any(not math.isfinite(rate) or rate <= 0.0 for rate in rates.values()):
            raise ValueError("population_learning_rates_invalid")
        optimizer = tuple(np.zeros_like(value) for value in optimizer)
        # The strict SGU adapter carries a six-entry functional optimizer
        # schema. Its GradNorm weights are required to be strictly positive,
        # even for a fresh state; all-one weights are the identity sentinel,
        # while moments, initial losses, and counters reset to zero. Keep the
        # historical all-zero behavior for shorter/reference state tuples.
        if (
            len(optimizer) == 6
            and np.asarray(optimizer[2]).shape == ()
            and np.asarray(optimizer[3]).ndim == 1
            and np.asarray(optimizer[4]).shape == np.asarray(optimizer[3]).shape
            and np.asarray(optimizer[5]).shape == ()
        ):
            reset = list(optimizer)
            reset[3] = np.ones_like(reset[3])
            optimizer = tuple(reset)
        return ReplicaState(
            policy=policy,
            optimizer=optimizer,
            controller=copy.deepcopy(dict(initial_controller or {})),
            method=str(self.method if child_method is None else child_method),
            seed=int(self.seed if child_seed is None else child_seed),
            learning_rates=rates,
            recipe_counter=0,
        )

    def fingerprint(self) -> tuple[tuple[str, tuple[int, ...], bytes], ...]:
        values = self.policy + self.optimizer
        return tuple(
            (value.dtype.str, tuple(value.shape), value.tobytes(order="C"))
            for value in values
        )


@dataclass(frozen=True)
class ReplicaEvaluation:
    replica_id: int
    objectives: tuple[float, ...]
    state: ReplicaState
    valid: bool = True
    method: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    # Stable identity for archived checkpoints.  ``replica_id`` identifies the
    # worker; this field distinguishes multiple checkpoints from that worker.
    candidate_id: str | None = None

    def checked(self) -> "ReplicaEvaluation":
        if type(self.replica_id) is not int or self.replica_id < 0:
            raise ValueError("population_replica_id_invalid")
        if not self.valid:
            return self
        objectives = _finite_objectives(self.objectives)
        return ReplicaEvaluation(
            replica_id=self.replica_id,
            objectives=objectives,
            state=self.state,
            valid=True,
            method=self.method or self.state.method,
            diagnostics=dict(self.diagnostics),
            candidate_id=self.candidate_id,
        )


def dominates(
    candidate: Sequence[float], other: Sequence[float], *, absolute_tolerance: float = 0.0
) -> bool:
    left = _finite_objectives(candidate)
    right = _finite_objectives(other)
    if len(left) != len(right):
        raise ValueError("population_objective_dimension_mismatch")
    if absolute_tolerance < 0.0 or not math.isfinite(float(absolute_tolerance)):
        raise ValueError("population_tolerance_invalid")
    no_worse = all(a <= b + absolute_tolerance for a, b in zip(left, right))
    strictly_better = any(a < b - absolute_tolerance for a, b in zip(left, right))
    return no_worse and strictly_better


def nondominated(
    entries: Sequence[ReplicaEvaluation], *, absolute_tolerance: float = 0.0
) -> list[ReplicaEvaluation]:
    checked = [entry.checked() for entry in entries if entry.valid]
    result: list[ReplicaEvaluation] = []
    for index, candidate in enumerate(checked):
        if any(
            dominates(other.objectives, candidate.objectives, absolute_tolerance=absolute_tolerance)
            for other_index, other in enumerate(checked)
            if other_index != index
        ):
            continue
        result.append(candidate)
    return sorted(result, key=lambda entry: (entry.objectives, _candidate_id(entry), entry.replica_id))


def _candidate_fingerprint(entry: ReplicaEvaluation) -> tuple[Any, ...]:
    return tuple(
        (value.dtype.str, tuple(value.shape), value.tobytes(order="C"))
        for value in entry.state.policy
    )


def _candidate_id(entry: ReplicaEvaluation) -> str:
    return str(entry.candidate_id or entry.diagnostics.get("candidate_id") or f"replica:{entry.replica_id}")


def _decision_view(
    entry: ReplicaEvaluation,
    values: Sequence[float],
    *,
    source: str = "paired_replay_mean",
) -> ReplicaEvaluation:
    """Copy an entry with decision coordinates without mutating evidence."""

    checked = _finite_objectives(values)
    diagnostics = dict(entry.diagnostics)
    diagnostics["decision_objectives"] = list(checked)
    diagnostics["decision_objective_source"] = str(source)
    return replace(entry, objectives=checked, diagnostics=diagnostics)


def deduplicate_evaluations(
    entries: Sequence[ReplicaEvaluation],
) -> list[ReplicaEvaluation]:
    """Keep one exact policy state while retaining deterministic lineage."""
    selected: dict[tuple[Any, ...], ReplicaEvaluation] = {}
    for entry in entries:
        checked = entry.checked()
        if not checked.valid:
            continue
        fingerprint = _candidate_fingerprint(checked)
        prior = selected.get(fingerprint)
        if prior is None:
            selected[fingerprint] = checked
            continue
        prior_origin = str(prior.diagnostics.get("origin", ""))
        current_origin = str(checked.diagnostics.get("origin", ""))
        if (not (current_origin == "round_start"), checked.replica_id) < (
            not (prior_origin == "round_start"), prior.replica_id
        ):
            selected[fingerprint] = checked
    return sorted(selected.values(), key=lambda entry: (entry.objectives, _candidate_id(entry), entry.replica_id))


def _sobol_points(*, dimensions: int, sample_count: int, seed: int) -> np.ndarray:
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("population_hypervolume_dimensions_invalid")
    if type(sample_count) is not int or sample_count < 2 or sample_count & (sample_count - 1):
        raise ValueError("population_hypervolume_sample_count_must_be_power_of_two")
    if type(seed) is not int or seed < 0:
        raise ValueError("population_hypervolume_seed_invalid")
    exponent = int(math.log2(sample_count))
    return qmc.Sobol(d=dimensions, scramble=True, seed=seed).random_base2(exponent)


def hypervolume_context(
    entries: Sequence[ReplicaEvaluation],
    *,
    scales: Sequence[float],
    seed: int = 20260725,
    sample_count: int = 4096,
    reference_margin: float = 1.0,
) -> dict[str, Any]:
    """Create the fixed finite coverage universe used for one synchronization."""
    checked = deduplicate_evaluations(entries)
    if not checked:
        raise ValueError("population_hypervolume_empty_candidates")
    scale = np.asarray(scales, dtype=np.float64)
    if scale.shape != (len(checked[0].objectives),) or not np.all(
        np.isfinite(scale) & (scale > 0.0)
    ):
        raise ValueError("population_hypervolume_scales_invalid")
    if not math.isfinite(float(reference_margin)) or reference_margin <= 0.0:
        raise ValueError("population_hypervolume_reference_margin_invalid")
    normalized = np.asarray([entry.objectives for entry in checked], dtype=np.float64) / scale
    reference = np.max(normalized, axis=0) + float(reference_margin)
    return {
        "scales": scale,
        "reference": reference,
        "sobol": _sobol_points(
            dimensions=normalized.shape[1], sample_count=sample_count, seed=seed
        ),
        "seed": seed,
        "sample_count": sample_count,
        "reference_margin": float(reference_margin),
    }


def sampled_hypervolume(
    entries: Sequence[ReplicaEvaluation], *, context: Mapping[str, Any]
) -> float:
    """Evaluate the fixed sampled dominated-coverage objective."""
    checked = deduplicate_evaluations(entries)
    if not checked:
        return 0.0
    scale = np.asarray(context["scales"], dtype=np.float64)
    reference = np.asarray(context["reference"], dtype=np.float64)
    sobol = np.asarray(context["sobol"], dtype=np.float64)
    matrix = np.asarray([entry.objectives for entry in checked], dtype=np.float64) / scale
    if matrix.shape[1] != reference.size or sobol.shape[1] != reference.size:
        raise ValueError("population_hypervolume_context_dimension_mismatch")
    covered = np.any(np.all(matrix[:, None, :] <= reference[None, None, :] * sobol[None, :, :], axis=2), axis=0)
    return float(np.mean(covered))


def _is_round_start(entry: ReplicaEvaluation) -> bool:
    return str(entry.diagnostics.get("origin", "")) == "round_start"


def greedy_hypervolume_subset(
    entries: Sequence[ReplicaEvaluation],
    size: int,
    *,
    context: Mapping[str, Any],
) -> tuple[list[ReplicaEvaluation], dict[str, Any]]:
    """Select a diverse bounded Pareto subset by greedy sampled coverage."""
    if type(size) is not int or size < 1:
        raise ValueError("population_hypervolume_subset_size_invalid")
    frontier = deduplicate_evaluations(nondominated(entries))
    selected: list[ReplicaEvaluation] = []
    remaining = list(frontier)
    marginal_gains: list[float] = []
    while remaining and len(selected) < size:
        base = sampled_hypervolume(selected, context=context)
        scored = []
        for candidate in remaining:
            gain = sampled_hypervolume(selected + [candidate], context=context) - base
            scored.append(
                (
                    -gain,
                    not _is_round_start(candidate),
                    tuple(candidate.objectives),
                    _candidate_id(candidate),
                    int(candidate.replica_id),
                    repr(candidate.state.fingerprint()),
                    candidate,
                    gain,
                )
            )
        scored.sort(key=lambda value: value[:-2])
        _, _, _, _, _, _, winner, gain = scored[0]
        selected.append(winner)
        remaining.remove(winner)
        marginal_gains.append(float(gain))
    return selected, {
        "frontier_size": len(frontier),
        "selected_size": len(selected),
        "marginal_gains": marginal_gains,
        "frontier_replica_ids": [entry.replica_id for entry in frontier],
        "frontier_candidate_ids": [_candidate_id(entry) for entry in frontier],
        "selected_replica_ids": [entry.replica_id for entry in selected],
        "selected_candidate_ids": [_candidate_id(entry) for entry in selected],
    }


def worst_normalized_score(
    entry: ReplicaEvaluation, *, scales: Sequence[float]
) -> float:
    """Return the largest scaled objective for feasibility-first selection."""
    checked = entry.checked()
    scale = np.asarray(scales, dtype=np.float64)
    objectives = np.asarray(checked.objectives, dtype=np.float64)
    if scale.shape != objectives.shape or not np.all(
        np.isfinite(scale) & (scale > 0.0)
    ):
        raise ValueError("population_robust_score_scales_invalid")
    return float(np.max(objectives / scale))


def constraint_violation_score(
    entry: ReplicaEvaluation,
    *,
    global_objective_count: int,
    constraint_limits: Sequence[float],
) -> tuple[float, float]:
    """Return maximum and squared aggregate positive constraint violations."""
    objectives = np.asarray(entry.checked().objectives, dtype=np.float64)
    limits = np.asarray(constraint_limits, dtype=np.float64)
    if (
        type(global_objective_count) is not int
        or global_objective_count < 1
        or objectives.size != global_objective_count + limits.size
        or limits.size < 1
        or not np.all(np.isfinite(limits) & (limits > 0.0))
    ):
        raise ValueError("population_constraint_contract_invalid")
    violations = np.maximum(
        objectives[global_objective_count:] / limits - 1.0, 0.0
    )
    return float(np.max(violations)), float(np.sum(np.square(violations)))


def _project_objectives(
    entry: ReplicaEvaluation, *, objective_count: int
) -> ReplicaEvaluation:
    return ReplicaEvaluation(
        replica_id=entry.replica_id,
        objectives=tuple(entry.objectives[:objective_count]),
        state=entry.state,
        valid=entry.valid,
        method=entry.method,
        diagnostics=entry.diagnostics,
        candidate_id=entry.candidate_id,
    )


def select_constrained_parents(
    entries: Sequence[ReplicaEvaluation],
    size: int,
    *,
    global_scales: Sequence[float],
    constraint_limits: Sequence[float],
    seed: int,
    sample_count: int,
    reference_margin: float,
) -> tuple[list[ReplicaEvaluation], dict[str, Any]]:
    """Select feasible global performers, then least-violating candidates."""
    candidates = deduplicate_evaluations(entries)
    global_count = len(tuple(global_scales))
    scored = [
        (
            *constraint_violation_score(
                entry,
                global_objective_count=global_count,
                constraint_limits=constraint_limits,
            ),
            worst_normalized_score(
                _project_objectives(entry, objective_count=global_count),
                scales=global_scales,
            ),
            _candidate_id(entry),
            entry,
        )
        for entry in candidates
    ]
    feasible = [item[-1] for item in scored if item[0] == 0.0]
    selected: list[ReplicaEvaluation] = []
    hypervolume_metadata: dict[str, Any] | None = None
    if feasible:
        projected = [
            _project_objectives(entry, objective_count=global_count)
            for entry in feasible
        ]
        context = hypervolume_context(
            projected,
            scales=global_scales,
            seed=seed,
            sample_count=sample_count,
            reference_margin=reference_margin,
        )
        robust = min(
            feasible,
            key=lambda entry: (
                worst_normalized_score(
                    _project_objectives(entry, objective_count=global_count),
                    scales=global_scales,
                ),
                _candidate_id(entry),
            ),
        )
        projected_selected, gains = _greedy_hypervolume_fill(
            projected,
            [_project_objectives(robust, objective_count=global_count)],
            min(size, len(feasible)),
            context=context,
        )
        selected_ids = {_candidate_id(entry) for entry in projected_selected}
        selected.extend(
            entry for entry in feasible if _candidate_id(entry) in selected_ids
        )
        hypervolume_metadata = {
            "global_hypervolume": sampled_hypervolume(
                projected_selected, context=context
            ),
            "global_reference": np.asarray(context["reference"]).tolist(),
            "global_fill_marginal_gains": gains,
        }
    if len(selected) < size:
        selected_ids = {_candidate_id(entry) for entry in selected}
        remaining = sorted(
            (item for item in scored if item[3] not in selected_ids),
            key=lambda item: item[:-1],
        )
        selected.extend(item[-1] for item in remaining[: size - len(selected)])
    return selected, {
        "selection_mode": "four_objective_three_constraint_feasibility_first",
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "global_objective_count": global_count,
        "constraint_limits": [float(value) for value in constraint_limits],
        "selected_candidate_ids": [_candidate_id(entry) for entry in selected],
        "selected_replica_ids": [entry.replica_id for entry in selected],
        "selected_constraint_scores": [
            list(
                constraint_violation_score(
                    entry,
                    global_objective_count=global_count,
                    constraint_limits=constraint_limits,
                )
            )
            for entry in selected
        ],
        "hypervolume": hypervolume_metadata,
        "candidate_ids": [_candidate_id(entry) for entry in candidates],
        "frontier_candidate_ids": [
            _candidate_id(entry) for entry in nondominated(candidates)
        ],
    }


def select_rotating_protected_parents(
    entries: Sequence[ReplicaEvaluation],
    size: int,
    *,
    active_indices: Sequence[int],
    protected_indices: Sequence[int],
    limits: Sequence[float],
    incumbent_candidate_id: str,
    all_task_guard: float,
    all_task_margins: Sequence[float] | None = None,
    guard_reference_candidate_id: str | None = None,
    fill_to_size: bool = True,
    decision_objectives: Mapping[str, Sequence[float]] | None = None,
    require_complete_decision_objectives: bool = False,
) -> tuple[list[ReplicaEvaluation], dict[str, Any]]:
    """Select only feasible rotating-incumbent parents and reserve its slot.

    The default preserves the Rotemberg source behavior: any unfilled parent
    slots are filled with the incumbent.  The SGU transaction can explicitly
    request a distinct archive so that its coordinator records the subsequent
    with-replacement refill as an auditable lifecycle event.
    """
    candidates = deduplicate_evaluations(entries)
    if type(size) is not int or size < 1:
        raise ValueError("rotating_parent_size_invalid")
    active = tuple(int(index) for index in active_indices)
    protected = tuple(int(index) for index in protected_indices)
    if (
        not active
        or not protected
        or len(set(active)) != len(active)
        or len(set(protected)) != len(protected)
        or set(active) & set(protected)
    ):
        raise ValueError("rotating_parent_partition_invalid")
    task_count = len(candidates[0].objectives) if candidates else 0
    if task_count == 7 and (len(active) != 4 or len(protected) != 3):
        raise ValueError("rotating_parent_partition_invalid")
    if set(active) | set(protected) != set(range(task_count)):
        raise ValueError("rotating_parent_partition_incomplete")
    limit_values = tuple(float(value) for value in limits)
    if len(limit_values) != len(protected) or any(
        not math.isfinite(value) or value < 0.0 for value in limit_values
    ):
        raise ValueError("rotating_parent_limits_invalid")
    if not math.isfinite(float(all_task_guard)) or all_task_guard < 0.0:
        raise ValueError("rotating_parent_guard_invalid")
    if type(fill_to_size) is not bool:
        raise ValueError("rotating_parent_fill_mode_invalid")
    if type(require_complete_decision_objectives) is not bool:
        raise ValueError("rotating_parent_decision_map_mode_invalid")
    by_id = {_candidate_id(entry): entry for entry in candidates}
    decision_values: dict[str, np.ndarray] = {}
    decision_sources: dict[str, str] = {}
    if decision_objectives is not None:
        if not isinstance(decision_objectives, Mapping):
            raise ValueError("rotating_parent_decision_values_invalid")
        for candidate_id, raw_values in decision_objectives.items():
            candidate_key = str(candidate_id)
            if candidate_key not in by_id and not require_complete_decision_objectives:
                raise ValueError("rotating_parent_decision_candidate_missing")
            if candidate_key in decision_values:
                raise ValueError("rotating_parent_decision_candidate_duplicate")
            values = np.asarray(raw_values, dtype=np.float64)
            if values.shape != (task_count,) or not np.all(
                np.isfinite(values) & (values >= 0.0)
            ):
                raise ValueError("rotating_parent_decision_values_invalid")
            decision_values[candidate_key] = values
            decision_sources[candidate_key] = "paired_replay_mean"

    candidate_ids = set(by_id)
    if require_complete_decision_objectives:
        if decision_objectives is None:
            raise ValueError("rotating_parent_decision_values_required")
        decision_ids = set(decision_values)
        if candidate_ids - decision_ids:
            raise ValueError("rotating_parent_decision_candidate_missing")
        if decision_ids - candidate_ids:
            raise ValueError("rotating_parent_decision_candidate_unknown")

    def effective_values(entry: ReplicaEvaluation) -> np.ndarray:
        candidate_id = _candidate_id(entry)
        values = decision_values.get(candidate_id)
        if values is None:
            if require_complete_decision_objectives:
                raise ValueError("rotating_parent_decision_candidate_missing")
            values = np.asarray(entry.checked().objectives, dtype=np.float64)
            decision_sources.setdefault(candidate_id, "selection_objectives")
        return values

    # Frontier membership must be computed in one estimator coordinate system.
    # In strict SGU mode those coordinates are replay means for every entry;
    # the original entries are retained for policy/state lineage below.
    decision_entries = [
        _decision_view(
            entry,
            effective_values(entry),
            source=decision_sources.get(_candidate_id(entry), "selection_objectives"),
        )
        for entry in candidates
    ]
    decision_by_id = {_candidate_id(entry): entry for entry in decision_entries}

    # Return replay-coordinate views whenever an explicit map is supplied.
    # Their states and candidate IDs are unchanged, so policy lineage remains
    # intact while downstream consumers cannot read stale selection values.
    output_by_id = decision_by_id if decision_objectives is not None else by_id
    incumbent = output_by_id.get(str(incumbent_candidate_id))
    if incumbent is None:
        raise ValueError("rotating_parent_incumbent_missing")
    guard_reference_id = str(guard_reference_candidate_id or incumbent_candidate_id)
    guard_reference = output_by_id.get(guard_reference_id)
    if guard_reference is None:
        raise ValueError("rotating_parent_guard_reference_missing")

    def feasible(entry: ReplicaEvaluation) -> bool:
        values = effective_values(entry)
        return bool(
            np.all(np.isfinite(values))
            and np.all(values[list(protected)] <= np.asarray(limit_values))
        )

    guard_reference_values = effective_values(guard_reference)
    guard_reference_all = float(np.max(guard_reference_values))
    if all_task_margins is None:
        margin_values = np.full(task_count, float(all_task_guard), dtype=np.float64)
    else:
        margin_values = np.asarray(tuple(all_task_margins), dtype=np.float64)
        if margin_values.shape != (task_count,) or not np.all(
            np.isfinite(margin_values) & (margin_values >= 0.0)
        ):
            raise ValueError("rotating_parent_margins_invalid")
    if not feasible(incumbent):
        raise ValueError("rotating_parent_incumbent_infeasible")
    if not feasible(guard_reference):
        raise ValueError("rotating_parent_guard_reference_infeasible")
    feasible_entries = []
    for entry in candidates:
        output_entry = output_by_id[_candidate_id(entry)]
        if not feasible(entry):
            continue
        values = effective_values(entry)
        all_score = float(np.max(values))
        componentwise_guard = bool(
            np.all(values <= guard_reference_values + margin_values)
        )
        if componentwise_guard and all_score <= guard_reference_all + float(all_task_guard):
            feasible_entries.append(
                (
                    float(np.max(values[list(active)])),
                    all_score,
                    _candidate_id(entry),
                    output_entry,
                )
            )
    feasible_entries.sort(key=lambda item: item[:-1])
    selected = [incumbent]
    selected_ids = {_candidate_id(incumbent)}
    for _, _, candidate_id, entry in feasible_entries:
        if len(selected) >= size:
            break
        if candidate_id not in selected_ids:
            selected.append(entry)
            selected_ids.add(candidate_id)
    if fill_to_size:
        while len(selected) < size:
            selected.append(incumbent)
    return selected, {
        "selection_mode": "rotating_protected_feasible_incumbent_first",
        "candidate_count": len(candidates),
        "feasible_guarded_count": len(feasible_entries),
        "active_indices": list(active),
        "protected_indices": list(protected),
        "limits": list(limit_values),
        "all_task_guard": float(all_task_guard),
        "all_task_margins": margin_values.tolist(),
        "incumbent_candidate_id": _candidate_id(incumbent),
        "guard_reference_candidate_id": _candidate_id(guard_reference),
        "fill_to_size": bool(fill_to_size),
        "decision_objective_sources": {
            candidate_id: decision_sources[candidate_id]
            for candidate_id in sorted(decision_sources)
        },
        "decision_objectives": {
            candidate_id: values.tolist()
            for candidate_id, values in sorted(decision_values.items())
        },
        "require_complete_decision_objectives": bool(
            require_complete_decision_objectives
        ),
        "incumbent_decision_objectives": effective_values(incumbent).tolist(),
        "guard_reference_decision_objectives": guard_reference_values.tolist(),
        "selected_candidate_ids": [_candidate_id(entry) for entry in selected],
        "selected_replica_ids": [entry.replica_id for entry in selected],
        "infeasible_candidate_ids": [
            _candidate_id(entry) for entry in candidates if not feasible(entry)
        ],
        "candidate_ids": [_candidate_id(entry) for entry in candidates],
        "frontier_candidate_ids": [
            _candidate_id(entry) for entry in nondominated(decision_entries)
        ],
        "decision_objective_sources_complete": {
            _candidate_id(entry): decision_sources[_candidate_id(entry)]
            for entry in candidates
        },
        "decision_map_extra_candidate_ids": sorted(
            set(decision_values) - set(_candidate_id(entry) for entry in candidates)
        ),
    }


def _greedy_hypervolume_fill(
    entries: Sequence[ReplicaEvaluation],
    selected: Sequence[ReplicaEvaluation],
    size: int,
    *,
    context: Mapping[str, Any],
) -> tuple[list[ReplicaEvaluation], list[float]]:
    result = list(selected)
    selected_ids = {_candidate_id(entry) for entry in result}
    remaining = [
        entry
        for entry in deduplicate_evaluations(nondominated(entries))
        if _candidate_id(entry) not in selected_ids
    ]
    marginal_gains: list[float] = []
    while remaining and len(result) < size:
        base = sampled_hypervolume(result, context=context)
        scored = []
        for candidate in remaining:
            gain = sampled_hypervolume(result + [candidate], context=context) - base
            scored.append(
                (
                    -gain,
                    not _is_round_start(candidate),
                    tuple(candidate.objectives),
                    _candidate_id(candidate),
                    int(candidate.replica_id),
                    repr(candidate.state.fingerprint()),
                    candidate,
                    gain,
                )
            )
        scored.sort(key=lambda value: value[:-2])
        _, _, _, _, _, _, winner, gain = scored[0]
        result.append(winner)
        remaining.remove(winner)
        marginal_gains.append(float(gain))
    return result, marginal_gains


def select_active_parents(
    entries: Sequence[ReplicaEvaluation],
    starts: Sequence[ReplicaEvaluation],
    size: int,
    *,
    scales: Sequence[float],
    seed: int = 20260725,
    sample_count: int = 4096,
    reference_margin: float = 1.0,
    robust_slots: int = 0,
) -> tuple[list[ReplicaEvaluation], dict[str, Any]]:
    """Choose active parents while keeping the round-start population valid."""
    if type(robust_slots) is not int or not 0 <= robust_slots <= size:
        raise ValueError("population_robust_slots_invalid")
    candidates = deduplicate_evaluations(entries)
    context = hypervolume_context(
        candidates,
        scales=scales,
        seed=seed,
        sample_count=sample_count,
        reference_margin=reference_margin,
    )
    greedy, greedy_meta = greedy_hypervolume_subset(candidates, size, context=context)
    baseline = deduplicate_evaluations(starts)
    if not baseline:
        raise ValueError("population_hypervolume_start_baseline_empty")
    baseline = baseline[:size]
    greedy_score = sampled_hypervolume(greedy, context=context)
    baseline_score = sampled_hypervolume(baseline, context=context)
    robust_candidates = sorted(
        candidates,
        key=lambda entry: (
            worst_normalized_score(entry, scales=scales),
            not _is_round_start(entry),
            tuple(entry.objectives),
            _candidate_id(entry),
            entry.replica_id,
        ),
    )[:robust_slots]
    if robust_candidates:
        selected, robust_marginal_gains = _greedy_hypervolume_fill(
            candidates, robust_candidates, size, context=context
        )
        selection_mode = "robust_incumbent_then_greedy_sampled_hypervolume"
    elif baseline_score > greedy_score + 1.0e-15:
        selected = baseline
        selection_mode = "round_start_baseline"
        robust_marginal_gains = []
    else:
        selected = greedy
        selection_mode = "greedy_sampled_hypervolume"
        robust_marginal_gains = []
    return selected, {
        **greedy_meta,
        "selected_replica_ids": [entry.replica_id for entry in selected],
        "selected_candidate_ids": [_candidate_id(entry) for entry in selected],
        "selected_size": len(selected),
        "selection_mode": selection_mode,
        "greedy_hypervolume": greedy_score,
        "round_start_hypervolume": baseline_score,
        "selected_hypervolume": sampled_hypervolume(selected, context=context),
        "robust_slots": robust_slots,
        "robust_candidate_ids": [
            _candidate_id(entry) for entry in robust_candidates
        ],
        "robust_candidate_scores": [
            worst_normalized_score(entry, scales=scales)
            for entry in robust_candidates
        ],
        "selected_robust_scores": [
            worst_normalized_score(entry, scales=scales) for entry in selected
        ],
        "robust_fill_marginal_gains": robust_marginal_gains,
        "candidate_count": len(candidates),
        "candidate_cap": 2 * size * size,
        "hypervolume_seed": seed,
        "hypervolume_sample_count": sample_count,
        "reference": np.asarray(context["reference"]).tolist(),
        "frontier": [
            {
                "replica_id": entry.replica_id,
                "candidate_id": _candidate_id(entry),
                "objectives": list(entry.objectives),
                "origin": str(entry.diagnostics.get("origin", "")),
            }
            for entry in deduplicate_evaluations(nondominated(candidates))
        ],
        "candidate_ids": [_candidate_id(entry) for entry in candidates],
        "discarded_candidate_ids": [
            _candidate_id(entry)
            for entry in candidates
            if _candidate_id(entry)
            not in {_candidate_id(frontier_entry) for frontier_entry in nondominated(candidates)}
        ],
    }


def lexicographic_key(entry: ReplicaEvaluation) -> tuple[float, ...]:
    return tuple(entry.objectives)


def _minimum_distance_pick(
    entries: Sequence[ReplicaEvaluation], selected: Sequence[ReplicaEvaluation]
) -> ReplicaEvaluation:
    if not entries:
        raise ValueError("population_empty_candidate_set")
    if not selected:
        return min(entries, key=lambda entry: (lexicographic_key(entry), entry.replica_id))
    matrix = np.asarray([entry.objectives for entry in selected], dtype=np.float64)
    candidate_matrix = np.asarray([entry.objectives for entry in entries], dtype=np.float64)
    scale = np.ptp(np.vstack((matrix, candidate_matrix)), axis=0)
    scale[scale == 0.0] = 1.0
    best = None
    for entry in entries:
        distances = np.linalg.norm((matrix - np.asarray(entry.objectives)) / scale, axis=1)
        score = float(np.min(distances))
        key = (-score, lexicographic_key(entry), entry.replica_id)
        if best is None or key < best[0]:
            best = (key, entry)
    assert best is not None
    return best[1]


def truncate_archive(
    entries: Sequence[ReplicaEvaluation],
    size: int,
    *,
    absolute_tolerance: float = 0.0,
) -> list[ReplicaEvaluation]:
    if type(size) is not int or size < 1:
        raise ValueError("population_archive_size_invalid")
    candidates = nondominated(entries, absolute_tolerance=absolute_tolerance)
    if len(candidates) <= size:
        return sorted(candidates, key=lambda entry: (entry.objectives, entry.replica_id))
    selected = [min(candidates, key=lambda entry: (entry.objectives, entry.replica_id))]
    remaining = [entry for entry in candidates if entry is not selected[0]]
    while len(selected) < size:
        pick = _minimum_distance_pick(remaining, selected)
        selected.append(pick)
        remaining.remove(pick)
    return sorted(selected, key=lambda entry: (entry.objectives, entry.replica_id))


@dataclass
class PopulationCoordinator:
    config: PopulationConfig
    method_order: tuple[str, ...] = METHODS
    archive: list[ReplicaEvaluation] = field(default_factory=list)
    round_index: int = 0
    evidence_archive: list[ReplicaEvaluation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.config.validate()
        if self.round_index < 0:
            raise ValueError("population_round_invalid")
        if not self.method_order or len(set(self.method_order)) != len(self.method_order):
            raise ValueError("population_method_order_invalid")

    def finish_round(
        self,
        evaluations: Sequence[ReplicaEvaluation],
        *,
        local_archives: Mapping[int, Sequence[ReplicaEvaluation]] | None = None,
        round_starts: Sequence[ReplicaEvaluation] | None = None,
        selection_scales: Sequence[float] | None = None,
        hypervolume_seed: int = 20260725,
        hypervolume_sample_count: int = 4096,
        robust_slots: int = 0,
        constraint_limits: Sequence[float] | None = None,
        rotating_active_indices: Sequence[int] | None = None,
        rotating_protected_indices: Sequence[int] | None = None,
        rotating_incumbent_candidate_id: str | None = None,
        rotating_guard_reference_candidate_id: str | None = None,
        rotating_all_task_guard: float = 0.0,
        rotating_all_task_margins: Sequence[float] | None = None,
        rotating_fill_to_size: bool = True,
        rotating_decision_objectives: Mapping[str, Sequence[float]] | None = None,
        rotating_require_complete_decision_objectives: bool = False,
    ) -> dict[str, Any]:
        if len(evaluations) != self.config.replica_count:
            raise ValueError("population_replica_count_mismatch")
        legacy_mode = local_archives is None and round_starts is None
        checked = [entry.checked() for entry in evaluations]
        valid = [entry for entry in checked if entry.valid]
        starts = list(round_starts or [])
        if local_archives is None:
            local_archives = {
                entry.replica_id: [entry]
                for entry in starts
            }
        # A live endpoint may be vetoed after a valid round-start checkpoint
        # was archived.  In that case the archived state is the valid parent
        # candidate; retain the all-invalid veto only when no archive exists.
        if not valid and local_archives:
            valid = [
                entry.checked()
                for entries in local_archives.values()
                for entry in entries
                if entry.valid
            ]
        if not valid:
            raise ValueError("population_no_valid_replica")
        if selection_scales is None:
            selection_scales = [1.0] * len(valid[0].objectives)
        archive_entries = []
        for replica_id, local_entries in local_archives.items():
            if len(local_entries) > 2 * self.config.replica_count:
                raise ValueError("population_local_archive_capacity_exceeded")
            archive_entries.extend(local_entries)
        if not archive_entries:
            archive_entries = valid + starts
        candidates = deduplicate_evaluations(archive_entries + valid)
        if len(candidates) > 2 * self.config.replica_count * self.config.replica_count:
            raise ValueError("population_global_archive_capacity_exceeded")
        selection_evidence_archive = nondominated(
            candidates, absolute_tolerance=self.config.absolute_tolerance
        )
        decision_evidence_archive = selection_evidence_archive
        if rotating_active_indices is not None:
            if (
                rotating_protected_indices is None
                or rotating_incumbent_candidate_id is None
                or constraint_limits is None
            ):
                raise ValueError("rotating_parent_contract_incomplete")
            if type(rotating_require_complete_decision_objectives) is not bool:
                raise ValueError("rotating_parent_decision_map_mode_invalid")
            if rotating_require_complete_decision_objectives and rotating_decision_objectives is None:
                raise ValueError("rotating_parent_decision_values_required")
            # The candidate union is archived in selection coordinates, but
            # every frontier/parent comparison in strict R7 mode is performed
            # on a replay-coordinate view.  Keep both frontiers available for
            # audit and never mutate the original evidence entries.
            raw_decision_map = dict(rotating_decision_objectives or {})
            normalized_decision_keys = [str(key) for key in raw_decision_map]
            if len(normalized_decision_keys) != len(set(normalized_decision_keys)):
                raise ValueError("rotating_parent_decision_candidate_duplicate")
            decision_map = {
                str(key): value for key, value in raw_decision_map.items()
            }
            # ``candidates`` is an exact-policy-deduplicated view of the
            # persisted union.  The decision map, however, is keyed by every
            # persisted candidate ID.  A duplicate policy observation may
            # therefore legitimately have a map entry even though it is not
            # passed to the frontier selector.  Require exact coverage of the
            # pre-deduplication union, rather than accidentally rejecting (or
            # silently dropping) that replay evidence.
            candidate_ids = {_candidate_id(entry) for entry in candidates}
            union_ids = {
                _candidate_id(entry)
                for entry in (*archive_entries, *valid)
            }
            if rotating_require_complete_decision_objectives and set(
                str(key) for key in decision_map
            ) != union_ids:
                raise ValueError("rotating_parent_complete_replay_map_invalid")
            decision_entries = [
                _decision_view(
                    entry,
                    decision_map[_candidate_id(entry)]
                    if _candidate_id(entry) in decision_map
                    else entry.objectives,
                    source=(
                        "paired_replay_mean"
                        if _candidate_id(entry) in decision_map
                        else "selection_objectives"
                    ),
                )
                for entry in candidates
            ]
            decision_by_id = {
                _candidate_id(entry): entry for entry in decision_entries
            }
            decision_evidence_archive = nondominated(
                decision_entries, absolute_tolerance=self.config.absolute_tolerance
            )
            # Keep the complete frontier for evidence, but restrict rotating
            # parent selection to that replay-valued frontier.  A dominated
            # incumbent or guard reference is retained as an explicit anchor
            # because the round contract needs those identities for replay.
            rotating_frontier = decision_evidence_archive
            frontier_ids = {_candidate_id(entry) for entry in rotating_frontier}
            required_ids = {
                str(rotating_incumbent_candidate_id),
                str(rotating_guard_reference_candidate_id or rotating_incumbent_candidate_id),
            }
            by_candidate_id = decision_by_id
            rotating_pool = list(rotating_frontier)
            for required_id in sorted(required_ids):
                required_entry = by_candidate_id.get(required_id)
                if required_entry is not None and required_id not in frontier_ids:
                    rotating_pool.append(required_entry)
            selected, selection_meta = select_rotating_protected_parents(
                rotating_pool,
                self.config.archive_size,
                active_indices=rotating_active_indices,
                protected_indices=rotating_protected_indices,
                limits=constraint_limits,
                incumbent_candidate_id=rotating_incumbent_candidate_id,
                guard_reference_candidate_id=rotating_guard_reference_candidate_id,
                all_task_guard=rotating_all_task_guard,
                all_task_margins=rotating_all_task_margins,
                fill_to_size=rotating_fill_to_size,
                decision_objectives={
                    candidate_id: decision_map[candidate_id]
                    for candidate_id in (
                        _candidate_id(entry) for entry in rotating_pool
                    )
                    if candidate_id in decision_map
                },
                require_complete_decision_objectives=(
                    rotating_require_complete_decision_objectives
                ),
            )
            selection_meta = {
                **selection_meta,
                "evidence_frontier_candidate_ids": [
                    _candidate_id(entry) for entry in rotating_frontier
                ],
                "selection_pool_candidate_ids": [
                    _candidate_id(entry) for entry in rotating_pool
                ],
                "selection_frontier_candidate_ids": [
                    _candidate_id(entry) for entry in selection_evidence_archive
                ],
                "decision_frontier_candidate_ids": [
                    _candidate_id(entry) for entry in decision_evidence_archive
                ],
                "decision_objective_source": (
                    "paired_replay_mean"
                    if rotating_require_complete_decision_objectives
                    else "explicit_map_with_compatibility_fallback"
                ),
                # The selector receives only the replay frontier plus required
                # anchors so dominated candidates cannot re-enter the parent
                # archive.  Retain the complete candidate-union map here for
                # transaction/reload validation and audit provenance.
                # Keep the complete persisted-union map in the transaction
                # metadata, even when exact-policy deduplication removed an
                # observation from the selector input.
                "decision_objectives": {
                    candidate_id: list(decision_map[candidate_id])
                    for candidate_id in sorted(set(decision_map))
                },
                "decision_objective_sources": {
                    candidate_id: "paired_replay_mean"
                    for candidate_id in sorted(set(decision_map))
                },
                "decision_objective_sources_complete": {
                    candidate_id: "paired_replay_mean"
                    for candidate_id in sorted(set(decision_map))
                },
                "decision_map_complete": bool(
                    set(decision_map) == union_ids
                ),
                "decision_map_required": bool(
                    rotating_require_complete_decision_objectives
                ),
                "selector_decision_candidate_ids": [
                    _candidate_id(entry) for entry in rotating_pool
                ],
            }
        elif constraint_limits is not None:
            selected, selection_meta = select_constrained_parents(
                candidates,
                self.config.archive_size,
                global_scales=selection_scales[: len(candidates[0].objectives) - len(constraint_limits)],
                constraint_limits=constraint_limits,
                seed=hypervolume_seed,
                sample_count=hypervolume_sample_count,
                reference_margin=1.0,
            )
        elif legacy_mode:
            selected = truncate_archive(
                valid,
                self.config.archive_size,
                absolute_tolerance=self.config.absolute_tolerance,
            )
            selection_meta = {
                "selection_mode": "legacy_endpoint_only",
                "candidate_count": len(valid),
                "frontier_size": len(selected),
            }
        else:
            selected, selection_meta = select_active_parents(
                candidates,
                starts or valid,
                self.config.archive_size,
                scales=selection_scales,
                seed=hypervolume_seed,
                sample_count=hypervolume_sample_count,
                robust_slots=robust_slots,
            )
        # In the rotating strict path this is the replay-valued frontier; for
        # legacy paths it remains the historical selection frontier.
        self.evidence_archive = decision_evidence_archive
        self.archive = selected
        parents = list(self.archive)
        rng = np.random.default_rng(self.config.selection_seed + self.round_index)
        while len(parents) < self.config.replica_count:
            parents.append(self.archive[int(rng.integers(0, len(self.archive)))])
        child_seeds = [
            int(rng.integers(0, np.iinfo(np.int32).max))
            for _ in range(self.config.replica_count)
        ]
        child_methods = [
            self.method_order[(self.round_index + slot) % len(self.method_order)]
            for slot in range(self.config.replica_count)
        ]
        self.round_index += 1
        return {
            "round": self.round_index,
            "valid_replica_ids": [entry.replica_id for entry in valid],
            "invalid_replica_ids": [entry.replica_id for entry in checked if not entry.valid],
            "archive": [
                {"replica_id": entry.replica_id, "candidate_id": _candidate_id(entry), "objectives": list(entry.objectives)}
                for entry in self.archive
            ],
            "evidence_archive": [
                {"replica_id": entry.replica_id, "candidate_id": _candidate_id(entry), "objectives": list(entry.objectives)}
                for entry in self.evidence_archive
            ],
            "selection_evidence_archive": [
                {
                    "replica_id": entry.replica_id,
                    "candidate_id": _candidate_id(entry),
                    "objectives": list(entry.objectives),
                }
                for entry in selection_evidence_archive
            ],
            "parent_replica_ids": [entry.replica_id for entry in parents],
            "parent_candidate_ids": [_candidate_id(entry) for entry in parents],
            "selector": "pareto_elites_then_seeded_uniform_refill",
            "selection": selection_meta,
            "selection_seed": self.config.selection_seed + self.round_index - 1,
            "child_training_seeds": child_seeds,
            "child_starting_methods": child_methods,
            "state_transfer_mode": "policy_and_method_rates_fresh_training",
            "next_states": [
                parent.state.clone(
                    child_seed=child_seed,
                    child_method=child_method,
                )
                for parent, child_seed, child_method in zip(
                    parents, child_seeds, child_methods, strict=True
                )
            ],
        }

    def record(self) -> dict[str, Any]:
        return {
            "config": self.config.record(),
            "round_index": self.round_index,
            "archive": [
                {
                    "replica_id": entry.replica_id,
                    "candidate_id": _candidate_id(entry),
                    "objectives": list(entry.objectives),
                }
                for entry in self.archive
            ],
            "evidence_archive": [
                {"replica_id": entry.replica_id, "candidate_id": _candidate_id(entry), "objectives": list(entry.objectives)}
                for entry in self.evidence_archive
            ],
        }


__all__ = [
    "ContinuousMethodController",
    "METHODS",
    "PopulationConfig",
    "PopulationCoordinator",
    "ReplicaEvaluation",
    "ReplicaState",
    "dominates",
    "deduplicate_evaluations",
    "greedy_hypervolume_subset",
    "hypervolume_context",
    "lexicographic_key",
    "nondominated",
    "next_stagnation_streak",
    "pairwise_objective_distances",
    "relative_parameter_movement",
    "sampled_hypervolume",
    "select_active_parents",
    "summarize_method_progress",
    "synchronization_reason",
    "truncate_archive",
    "worst_normalized_score",
    "constraint_violation_score",
    "select_constrained_parents",
    "select_rotating_protected_parents",
]
