"""Reversible working-set control and variable-constraint projected Adam."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from tests.support.task_order import SEVEN_TASK_ORDER


WORKING_SET_ENTRY_THRESHOLD = 0.032
WORKING_SET_EXIT_THRESHOLD = 0.04
WORKING_SET_ENTRY_STREAK = 2
PERMANENT_PASS_THRESHOLD = 0.04


@dataclass(frozen=True)
class WorkingSetState:
    protected: tuple[str, ...]
    safe_streaks: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            len(set(self.protected)) != len(self.protected)
            or any(task not in SEVEN_TASK_ORDER for task in self.protected)
            or len(self.safe_streaks) != len(SEVEN_TASK_ORDER)
            or any(type(value) is not int or value < 0 for value in self.safe_streaks)
        ):
            raise ValueError("rotemberg_working_set_state_invalid")

    @property
    def active(self) -> tuple[str, ...]:
        return tuple(task for task in SEVEN_TASK_ORDER if task not in self.protected)

    def record(self) -> dict[str, Any]:
        return {
            "active": list(self.active),
            "protected": list(self.protected),
            "safe_streaks": dict(zip(SEVEN_TASK_ORDER, self.safe_streaks, strict=True)),
        }


def initial_working_set_state() -> WorkingSetState:
    return WorkingSetState(protected=(), safe_streaks=(0,) * len(SEVEN_TASK_ORDER))


@dataclass(frozen=True)
class PermanentPassState:
    permanent: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(set(self.permanent)) != len(self.permanent)
            or any(task not in SEVEN_TASK_ORDER for task in self.permanent)
            or self.permanent
            != tuple(task for task in SEVEN_TASK_ORDER if task in self.permanent)
        ):
            raise ValueError("rotemberg_permanent_pass_state_invalid")

    @property
    def unresolved(self) -> tuple[str, ...]:
        return tuple(task for task in SEVEN_TASK_ORDER if task not in self.permanent)

    def record(self) -> dict[str, Any]:
        return {
            "permanent_constraints": list(self.permanent),
            "rotation_pool": list(self.unresolved),
        }


def _checked_upper_mse(upper_mse: Mapping[str, float]) -> dict[str, float]:
    if tuple(upper_mse) != SEVEN_TASK_ORDER:
        raise ValueError("rotemberg_permanent_pass_control_order_invalid")
    values = {task: float(upper_mse[task]) for task in SEVEN_TASK_ORDER}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("rotemberg_permanent_pass_control_value_invalid")
    return values


def initialize_permanent_pass_state(
    upper_mse: Mapping[str, float],
    *,
    threshold: float = PERMANENT_PASS_THRESHOLD,
) -> tuple[PermanentPassState, dict[str, Any]]:
    """Permanently constrain every objective already satisfying the gate."""

    if not math.isfinite(float(threshold)) or threshold <= 0.0:
        raise ValueError("rotemberg_permanent_pass_threshold_invalid")
    values = _checked_upper_mse(upper_mse)
    permanent = tuple(
        task for task in SEVEN_TASK_ORDER if values[task] <= float(threshold)
    )
    state = PermanentPassState(permanent=permanent)
    return state, {
        "schema": "dsge_hmc.rotemberg_permanent_pass_initialization.v1",
        "upper_normalized_mse": values,
        "threshold": float(threshold),
        "state": state.record(),
        "entered_permanent": list(permanent),
    }


def promote_permanent_passes(
    state: PermanentPassState,
    upper_mse: Mapping[str, float],
    *,
    threshold: float = PERMANENT_PASS_THRESHOLD,
) -> tuple[PermanentPassState, dict[str, Any]]:
    """Monotonically remove newly passing objectives from the rotation pool."""

    if not math.isfinite(float(threshold)) or threshold <= 0.0:
        raise ValueError("rotemberg_permanent_pass_threshold_invalid")
    values = _checked_upper_mse(upper_mse)
    permanent = set(state.permanent)
    entered = tuple(
        task
        for task in SEVEN_TASK_ORDER
        if task not in permanent and values[task] <= float(threshold)
    )
    permanent.update(entered)
    result = PermanentPassState(
        permanent=tuple(task for task in SEVEN_TASK_ORDER if task in permanent)
    )
    return result, {
        "schema": "dsge_hmc.rotemberg_permanent_pass_transition.v1",
        "upper_normalized_mse": values,
        "threshold": float(threshold),
        "before": state.record(),
        "after": result.record(),
        "entered_permanent": list(entered),
    }


def permanent_pass_rotation_indices(
    state: PermanentPassState, update_index: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Rotate at most three unresolved objectives with exact block balance."""

    if type(update_index) is not int or update_index < 0:
        raise ValueError("rotemberg_permanent_pass_update_index_invalid")
    unresolved = tuple(
        index
        for index, task in enumerate(SEVEN_TASK_ORDER)
        if task in state.unresolved
    )
    active_count = min(3, len(unresolved))
    if active_count == 0:
        return (), ()
    combinations = tuple(itertools.combinations(unresolved, active_count))
    active = combinations[update_index % len(combinations)]
    temporary = tuple(index for index in unresolved if index not in active)
    return active, temporary


def update_working_set(
    state: WorkingSetState,
    upper_mse: Mapping[str, float],
    *,
    entry_threshold: float = WORKING_SET_ENTRY_THRESHOLD,
    exit_threshold: float = WORKING_SET_EXIT_THRESHOLD,
    entry_streak: int = WORKING_SET_ENTRY_STREAK,
) -> tuple[WorkingSetState, dict[str, Any]]:
    """Update membership with entry hysteresis and reversible protection."""

    if (
        tuple(upper_mse) != SEVEN_TASK_ORDER
        or not math.isfinite(float(entry_threshold))
        or not math.isfinite(float(exit_threshold))
        or not 0.0 < float(entry_threshold) < float(exit_threshold)
        or type(entry_streak) is not int
        or entry_streak < 1
    ):
        raise ValueError("rotemberg_working_set_control_input_invalid")
    values = {task: float(upper_mse[task]) for task in SEVEN_TASK_ORDER}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("rotemberg_working_set_control_value_invalid")
    old_protected = set(state.protected)
    streaks = dict(zip(SEVEN_TASK_ORDER, state.safe_streaks, strict=True))
    next_protected = set(old_protected)
    entered = []
    exited = []
    for task in SEVEN_TASK_ORDER:
        value = values[task]
        if task in old_protected:
            if value > exit_threshold:
                next_protected.remove(task)
                streaks[task] = 0
                exited.append(task)
            elif value <= entry_threshold:
                streaks[task] += 1
            else:
                streaks[task] = 0
            continue
        streaks[task] = streaks[task] + 1 if value <= entry_threshold else 0
        if streaks[task] >= entry_streak:
            next_protected.add(task)
            entered.append(task)
    result = WorkingSetState(
        protected=tuple(task for task in SEVEN_TASK_ORDER if task in next_protected),
        safe_streaks=tuple(streaks[task] for task in SEVEN_TASK_ORDER),
    )
    return result, {
        "schema": "dsge_hmc.rotemberg_working_set_transition.v1",
        "upper_normalized_mse": values,
        "entry_threshold": float(entry_threshold),
        "exit_threshold": float(exit_threshold),
        "entry_streak": entry_streak,
        "before": state.record(),
        "after": result.record(),
        "entered_protection": entered,
        "reactivated": exited,
    }


def _project_intersection_tf(
    direction: Any, protected_rows: Any, *, constraint_count: int, eta: float
) -> Any:
    import tensorflow as tf

    if constraint_count == 0:
        return direction
    tolerance = tf.constant(1.0e-10, tf.float64)
    norms = tf.linalg.norm(protected_rows, axis=1, keepdims=True)
    normalized = protected_rows / norms
    candidates = [direction]
    for size in range(1, constraint_count + 1):
        for indices in itertools.combinations(range(constraint_count), size):
            selected = tf.gather(normalized, indices)
            gram = tf.linalg.matmul(selected, selected, transpose_b=True)
            rhs = -tf.linalg.matvec(selected, direction)
            multipliers = tf.linalg.matvec(
                tf.linalg.pinv(gram, rcond=tf.constant(eta, tf.float64)), rhs
            )
            candidates.append(
                direction + tf.linalg.matvec(selected, multipliers, transpose_a=True)
            )
    candidates.append(tf.zeros_like(direction))
    stacked = tf.stack(candidates, axis=0)
    directional = tf.linalg.matmul(stacked, normalized, transpose_b=True)
    feasible = tf.reduce_all(directional >= -tolerance, axis=1)
    distance = tf.reduce_sum(tf.square(stacked - direction[None, :]), axis=1)
    distance = tf.where(
        feasible,
        distance,
        tf.fill(tf.shape(distance), tf.constant(np.inf, tf.float64)),
    )
    return stacked[tf.argmin(distance, output_type=tf.int32)]


def prepare_working_set_projected_adam_updates(
    variables: Sequence[Any],
    optimizer: Any,
    *,
    clip_norm: float = 10.0,
    eta: float = 1.0e-12,
) -> dict[int, Any]:
    """Build one stable update graph for each possible protected-row count."""

    import tensorflow as tf

    variables = tuple(variables)
    if (
        not variables
        or not math.isfinite(float(clip_norm))
        or clip_norm <= 0.0
        or not math.isfinite(float(eta))
        or eta <= 0.0
    ):
        raise ValueError("rotemberg_working_set_update_config_invalid")
    optimizer.build(variables)
    dimension = sum(int(variable.shape.num_elements()) for variable in variables)

    def build(constraint_count: int) -> Any:
        @tf.function(
            input_signature=(
                tf.TensorSpec([7, dimension], tf.float64, name="seven_task_rows"),
                tf.TensorSpec([dimension], tf.float64, name="active_direction"),
                tf.TensorSpec(
                    [constraint_count], tf.int32, name="protected_indices"
                ),
            ),
            autograph=False,
            jit_compile=False,
        )
        def update(
            task_rows: Any, active_direction: Any, protected_indices: Any
        ) -> Any:
            protected = tf.gather(task_rows, protected_indices)
            protected_norms = tf.linalg.norm(protected, axis=1, keepdims=True)
            descent = _project_intersection_tf(
                active_direction,
                protected,
                constraint_count=constraint_count,
                eta=eta,
            )
            normalized = (
                protected / protected_norms
                if constraint_count
                else tf.zeros([0, dimension], tf.float64)
            )
            submitted = tf.linalg.matvec(normalized, descent)
            submitted_raw = tf.linalg.matvec(protected, descent)
            pieces = []
            cursor = 0
            for variable in variables:
                size = int(variable.shape.num_elements())
                pieces.append(
                    tf.reshape(descent[cursor : cursor + size], variable.shape)
                )
                cursor += size
            clipped, raw_norm = tf.clip_by_global_norm(pieces, clip_norm)
            before = tuple(tf.identity(variable) for variable in variables)
            optimizer_before = tuple(
                tf.identity(variable) for variable in optimizer.variables
            )
            checks = [
                tf.debugging.assert_all_finite(
                    descent, "rotemberg_working_set_descent_nonfinite"
                )
            ]
            if constraint_count:
                checks.extend(
                    (
                        tf.debugging.assert_greater(
                            protected_norms,
                            tf.zeros_like(protected_norms),
                            message="rotemberg_working_set_protected_gradient_degenerate",
                        ),
                        tf.debugging.assert_greater_equal(
                            submitted,
                            tf.fill(
                                [constraint_count],
                                tf.constant(-1.0e-10, tf.float64),
                            ),
                            message="rotemberg_working_set_submitted_constraint_invalid",
                        ),
                    )
                )
            with tf.control_dependencies(checks):
                applied = optimizer.apply_gradients(
                    zip(clipped, variables, strict=True)
                )
            with tf.control_dependencies([tf.identity(applied)]):
                proposed_delta = tf.concat(
                    [
                        tf.reshape(variable - old, [-1])
                        for variable, old in zip(variables, before, strict=True)
                    ],
                    axis=0,
                )
            projected_descent = _project_intersection_tf(
                -proposed_delta,
                protected,
                constraint_count=constraint_count,
                eta=eta,
            )
            final_delta = -projected_descent
            assignments = []
            cursor = 0
            for variable, old in zip(variables, before, strict=True):
                size = int(variable.shape.num_elements())
                piece = tf.reshape(
                    final_delta[cursor : cursor + size], variable.shape
                )
                assignments.append(variable.assign(old + piece))
                cursor += size
            with tf.control_dependencies(assignments):
                committed_delta = tf.concat(
                    [
                        tf.reshape(variable - old, [-1])
                        for variable, old in zip(variables, before, strict=True)
                    ],
                    axis=0,
                )
                actual = tf.linalg.matvec(normalized, committed_delta)
                actual_raw = tf.linalg.matvec(protected, committed_delta)
                finite = tf.reduce_all(
                    tf.math.is_finite(
                        tf.concat(
                            [
                                descent,
                                proposed_delta,
                                committed_delta,
                                submitted,
                                actual,
                                submitted_raw,
                                actual_raw,
                                *[
                                    tf.reshape(variable, [-1])
                                    for variable in variables
                                ],
                                *[
                                    tf.cast(tf.reshape(variable, [-1]), tf.float64)
                                    for variable in optimizer.variables
                                ],
                            ],
                            axis=0,
                        )
                    )
                )
                constraints = tf.reduce_all(
                    actual <= tf.constant(1.0e-10, tf.float64)
                )

            def accepted() -> tuple[Any, ...]:
                return (
                    raw_norm,
                    tf.linalg.global_norm(clipped),
                    tf.linalg.norm(proposed_delta),
                    tf.linalg.norm(committed_delta),
                    submitted,
                    actual,
                    descent,
                    submitted_raw,
                    actual_raw,
                )

            def rollback() -> tuple[Any, ...]:
                restored = tf.group(
                    *[
                        variable.assign(old)
                        for variable, old in zip(variables, before, strict=True)
                    ],
                    *[
                        variable.assign(old)
                        for variable, old in zip(
                            optimizer.variables, optimizer_before, strict=True
                        )
                    ],
                )
                with tf.control_dependencies([restored]):
                    failure = tf.debugging.assert_equal(
                        finite & constraints,
                        True,
                        message="rotemberg_working_set_commit_invalid_rolled_back",
                    )
                with tf.control_dependencies([failure]):
                    nan = tf.constant(np.nan, tf.float64)
                    return (
                        nan,
                        nan,
                        nan,
                        nan,
                        tf.fill([constraint_count], nan),
                        tf.fill([constraint_count], nan),
                        tf.fill([dimension], nan),
                        tf.fill([constraint_count], nan),
                        tf.fill([constraint_count], nan),
                    )

            return tf.cond(finite & constraints, accepted, rollback)

        return update

    return {count: build(count) for count in range(len(SEVEN_TASK_ORDER))}


__all__ = [
    "PERMANENT_PASS_THRESHOLD",
    "WORKING_SET_ENTRY_STREAK",
    "WORKING_SET_ENTRY_THRESHOLD",
    "WORKING_SET_EXIT_THRESHOLD",
    "PermanentPassState",
    "WorkingSetState",
    "initialize_permanent_pass_state",
    "initial_working_set_state",
    "permanent_pass_rotation_indices",
    "prepare_working_set_projected_adam_updates",
    "promote_permanent_passes",
    "update_working_set",
]
