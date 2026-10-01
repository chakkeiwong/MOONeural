"""Rollback-oriented controller for adaptive multi-objective training blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


METHODS = (
    "cagrad",
    "pcgrad",
    "mgda",
    "gradnorm",
    "weighted_normalized_sum",
)


@dataclass(frozen=True)
class AdaptiveControllerConfig:
    method_order: tuple[str, ...] = METHODS
    initial_learning_rates: Mapping[str, float] = field(
        default_factory=lambda: {method: 3.0e-4 for method in METHODS}
    )
    block_size: int = 25
    max_accepted_updates: int = 1000
    max_accepted_blocks: int = 40
    max_decisions: int = 200

    def validate(self) -> None:
        if (
            not self.method_order
            or len(set(self.method_order)) != len(self.method_order)
            or set(self.method_order) != set(METHODS)
        ):
            raise ValueError("adaptive_controller_method_order_invalid")
        if set(self.initial_learning_rates) != set(self.method_order) or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.initial_learning_rates.values()
        ):
            raise ValueError("adaptive_controller_learning_rates_invalid")
        for value, error in (
            (self.block_size, "adaptive_controller_block_size_invalid"),
            (
                self.max_accepted_updates,
                "adaptive_controller_update_cap_invalid",
            ),
            (self.max_accepted_blocks, "adaptive_controller_block_cap_invalid"),
            (self.max_decisions, "adaptive_controller_decision_cap_invalid"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(error)
        if self.block_size > self.max_accepted_updates:
            raise ValueError("adaptive_controller_block_exceeds_update_cap")

    def record(self) -> dict[str, Any]:
        self.validate()
        return {
            "method_order": list(self.method_order),
            "initial_learning_rates": {
                method: float(self.initial_learning_rates[method])
                for method in self.method_order
            },
            "block_size": self.block_size,
            "max_accepted_updates": self.max_accepted_updates,
            "max_accepted_blocks": self.max_accepted_blocks,
            "max_decisions": self.max_decisions,
        }


@dataclass
class AdaptiveControllerState:
    episode_order: tuple[str, ...]
    current_method: str
    learning_rates: dict[str, float]
    episode_start_learning_rates: dict[str, float]
    episode_failures: dict[str, int]
    no_improvement_round: int = 0
    method_position: int = 0
    attempt_in_visit: int = 0
    accepted_updates: int = 0
    accepted_blocks: int = 0
    decision_index: int = 0
    last_accepted_method: str | None = None
    terminal_reason: str | None = None

    def record(self) -> dict[str, Any]:
        return {
            "episode_order": list(self.episode_order),
            "current_method": self.current_method,
            "learning_rates": dict(self.learning_rates),
            "episode_start_learning_rates": dict(
                self.episode_start_learning_rates
            ),
            "episode_failures": dict(self.episode_failures),
            "no_improvement_round": self.no_improvement_round,
            "method_position": self.method_position,
            "attempt_in_visit": self.attempt_in_visit,
            "accepted_updates": self.accepted_updates,
            "accepted_blocks": self.accepted_blocks,
            "decision_index": self.decision_index,
            "last_accepted_method": self.last_accepted_method,
            "terminal_reason": self.terminal_reason,
        }


class AdaptiveMethodController:
    """Schedule cyclic block trials and retain per-method learning rates."""

    def __init__(self, config: AdaptiveControllerConfig | None = None):
        self.config = config or AdaptiveControllerConfig()
        self.config.validate()
        order = tuple(self.config.method_order)
        rates = {
            method: float(self.config.initial_learning_rates[method])
            for method in order
        }
        self.state = AdaptiveControllerState(
            episode_order=order,
            current_method=order[0],
            learning_rates=rates,
            episode_start_learning_rates=dict(rates),
            episode_failures={method: 0 for method in order},
        )
        self._validate_state()

    @staticmethod
    def _cyclic_order(order: Sequence[str], first: str) -> tuple[str, ...]:
        index = tuple(order).index(first)
        values = tuple(order)
        return values[index:] + values[:index]

    @property
    def stopped(self) -> bool:
        return self.state.terminal_reason is not None

    def next_trial(self) -> dict[str, Any]:
        self._validate_state()
        if self.stopped:
            raise RuntimeError("adaptive_controller_already_stopped")
        method = self.state.episode_order[self.state.method_position]
        return {
            "decision_index": self.state.decision_index,
            "method": method,
            "learning_rate": self.state.learning_rates[method],
            "round": self.state.no_improvement_round + 1,
            "attempt": self.state.attempt_in_visit + 1,
            "method_failure_ordinal": self.state.episode_failures[method] + 1,
            "accepted_update_start": self.state.accepted_updates,
            "block_size": self.config.block_size,
        }

    def record_outcome(self, *, improved: bool) -> dict[str, Any]:
        trial = self.next_trial()
        method = str(trial["method"])
        self.state.decision_index += 1
        transition = "accepted" if improved else "rejected"

        if improved:
            self.state.accepted_updates += self.config.block_size
            self.state.accepted_blocks += 1
            self.state.current_method = method
            self.state.last_accepted_method = method
            self.state.episode_order = self._cyclic_order(
                self.config.method_order, method
            )
            self.state.episode_start_learning_rates = dict(
                self.state.learning_rates
            )
            self.state.episode_failures = {
                candidate: 0 for candidate in self.config.method_order
            }
            self.state.no_improvement_round = 0
            self.state.method_position = 0
            self.state.attempt_in_visit = 0
            if (
                self.state.accepted_updates >= self.config.max_accepted_updates
                or self.state.accepted_updates + self.config.block_size
                > self.config.max_accepted_updates
            ):
                self.state.terminal_reason = "accepted_update_cap_reached"
            elif self.state.accepted_blocks >= self.config.max_accepted_blocks:
                self.state.terminal_reason = "accepted_block_cap_reached"
        else:
            self.state.episode_failures[method] += 1
            if self.state.attempt_in_visit == 0:
                self.state.learning_rates[method] /= 2.0
                self.state.attempt_in_visit = 1
                transition = "rejected_retry_same_method"
            else:
                self.state.attempt_in_visit = 0
                self.state.method_position += 1
                transition = "rejected_advance_method"
                if self.state.method_position == len(self.state.episode_order):
                    self.state.method_position = 0
                    if self.state.no_improvement_round == 0:
                        self.state.no_improvement_round = 1
                        for candidate in self.state.learning_rates:
                            self.state.learning_rates[candidate] /= 2.0
                        transition = "rejected_start_second_round"
                    else:
                        self.state.terminal_reason = "all_methods_exhausted"
                        transition = "rejected_all_methods_exhausted"

        if (
            self.state.terminal_reason is None
            and self.state.decision_index >= self.config.max_decisions
        ):
            self.state.terminal_reason = "decision_cap_reached"
            transition = f"{transition}_decision_cap"
        self._validate_state()
        return {
            **trial,
            "improved": bool(improved),
            "transition": transition,
            "state": self.state_record(),
        }

    def state_record(self) -> dict[str, Any]:
        self._validate_state()
        return self.state.record()

    def restore_state(self, record: Mapping[str, Any]) -> None:
        required = set(self.state.record())
        if set(record) != required:
            raise ValueError("adaptive_controller_state_schema_invalid")
        self.state = AdaptiveControllerState(
            episode_order=tuple(str(value) for value in record["episode_order"]),
            current_method=str(record["current_method"]),
            learning_rates={
                str(key): float(value)
                for key, value in dict(record["learning_rates"]).items()
            },
            episode_start_learning_rates={
                str(key): float(value)
                for key, value in dict(
                    record["episode_start_learning_rates"]
                ).items()
            },
            episode_failures={
                str(key): int(value)
                for key, value in dict(record["episode_failures"]).items()
            },
            no_improvement_round=int(record["no_improvement_round"]),
            method_position=int(record["method_position"]),
            attempt_in_visit=int(record["attempt_in_visit"]),
            accepted_updates=int(record["accepted_updates"]),
            accepted_blocks=int(record["accepted_blocks"]),
            decision_index=int(record["decision_index"]),
            last_accepted_method=(
                None
                if record["last_accepted_method"] is None
                else str(record["last_accepted_method"])
            ),
            terminal_reason=(
                None
                if record["terminal_reason"] is None
                else str(record["terminal_reason"])
            ),
        )
        self._validate_state()

    def _validate_state(self) -> None:
        state = self.state
        methods = set(self.config.method_order)
        if (
            len(state.episode_order) != len(methods)
            or set(state.episode_order) != methods
            or state.current_method not in methods
            or (
                state.last_accepted_method is not None
                and state.last_accepted_method not in methods
            )
        ):
            raise ValueError("adaptive_controller_state_methods_invalid")
        for values in (
            state.learning_rates,
            state.episode_start_learning_rates,
        ):
            if set(values) != methods or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in values.values()
            ):
                raise ValueError("adaptive_controller_state_rates_invalid")
        if set(state.episode_failures) != methods or any(
            type(value) is not int or not 0 <= value <= 4
            for value in state.episode_failures.values()
        ):
            raise ValueError("adaptive_controller_state_failures_invalid")
        counters = (
            state.accepted_updates,
            state.accepted_blocks,
            state.decision_index,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("adaptive_controller_state_counters_invalid")
        if state.no_improvement_round not in (0, 1):
            raise ValueError("adaptive_controller_state_round_invalid")
        if not 0 <= state.method_position < len(state.episode_order):
            raise ValueError("adaptive_controller_state_position_invalid")
        if state.attempt_in_visit not in (0, 1):
            raise ValueError("adaptive_controller_state_attempt_invalid")
        terminal = {
            None,
            "all_methods_exhausted",
            "accepted_update_cap_reached",
            "accepted_block_cap_reached",
            "decision_cap_reached",
        }
        if state.terminal_reason not in terminal:
            raise ValueError("adaptive_controller_state_terminal_invalid")


@dataclass(frozen=True)
class DirectionConfig:
    cagrad_c: float = 0.5
    cagrad_rescale: int = 1
    pcgrad_seed: int = 20260722
    dormant_absolute_norm: float = 1.0e-14
    dormant_relative_norm: float = 1.0e-12
    lambda_max_ratio: float = 100.0
    gradnorm_alpha: float = 1.5
    gradnorm_learning_rate: float = 1.0e-4
    gradnorm_weight_ratio_veto: float = 100.0

    def validate(self) -> None:
        numeric = (
            self.cagrad_c,
            self.dormant_absolute_norm,
            self.dormant_relative_norm,
            self.lambda_max_ratio,
            self.gradnorm_alpha,
            self.gradnorm_learning_rate,
            self.gradnorm_weight_ratio_veto,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("adaptive_direction_setting_nonfinite")
        if not 0.0 <= self.cagrad_c < 1.0 or self.cagrad_rescale not in (0, 1, 2):
            raise ValueError("adaptive_direction_cagrad_invalid")
        if type(self.pcgrad_seed) is not int:
            raise ValueError("adaptive_direction_pcgrad_seed_invalid")
        if min(self.dormant_absolute_norm, self.dormant_relative_norm) < 0.0:
            raise ValueError("adaptive_direction_dormant_cutoff_invalid")
        if min(self.lambda_max_ratio, self.gradnorm_weight_ratio_veto) < 1.0:
            raise ValueError("adaptive_direction_ratio_invalid")
        if self.gradnorm_alpha <= 0.0 or self.gradnorm_learning_rate <= 0.0:
            raise ValueError("adaptive_direction_gradnorm_invalid")


class AdaptiveDirectionSet:
    """Five independent direction kernels with a fixed objective count."""

    def __init__(
        self,
        gradient_dim: int,
        config: DirectionConfig | None = None,
        *,
        objective_count: int = 4,
    ):
        if type(gradient_dim) is not int or gradient_dim < 1:
            raise ValueError("adaptive_direction_gradient_dim_invalid")
        if type(objective_count) is not int or not 1 <= objective_count <= 8:
            raise ValueError("adaptive_direction_objective_count_invalid")
        self.gradient_dim = gradient_dim
        self.objective_count = objective_count
        self.config = config or DirectionConfig()
        self.config.validate()
        self._functions = self._build_functions()

    def _build_functions(self) -> dict[str, Any]:
        import tensorflow as tf

        matrix = tf.TensorSpec(
            [self.objective_count, self.gradient_dim],
            tf.float64,
            name="gradients",
        )
        vector = tf.TensorSpec([self.objective_count], tf.float64)

        @tf.function(input_signature=(matrix,), autograph=False, jit_compile=False)
        def weighted_normalized_sum(flat: Any) -> tuple[Any, Any, Any]:
            norms = tf.linalg.norm(flat, axis=1)
            maximum = tf.reduce_max(norms)
            cutoff = tf.maximum(
                tf.constant(self.config.dormant_absolute_norm, tf.float64),
                tf.constant(self.config.dormant_relative_norm, tf.float64)
                * maximum,
            )
            active = norms > cutoff
            active_count = tf.reduce_sum(tf.cast(active, tf.float64))
            floor = maximum / tf.constant(self.config.lambda_max_ratio, tf.float64)
            safe_norms = tf.maximum(norms, floor)
            inverse = tf.where(active, tf.math.reciprocal(safe_norms), 0.0)
            denominator = tf.reduce_sum(inverse)
            weights = tf.where(
                active_count > 0.0,
                active_count * inverse / tf.maximum(denominator, 1.0e-300),
                tf.zeros([self.objective_count], tf.float64),
            )
            combined = tf.linalg.matvec(flat, weights, transpose_a=True)
            return weights, combined, active

        @tf.function(input_signature=(matrix,), autograph=False, jit_compile=False)
        def cagrad(flat: Any) -> tuple[Any, Any]:
            from mooneural.multiobjective.cagrad import cagrad_flat

            coefficients, combined, _ = cagrad_flat(
                flat,
                c=self.config.cagrad_c,
                rescale=self.config.cagrad_rescale,
                eps=1.0e-12,
                max_objectives=8,
                stationarity_tol=1.0e-6,
            )
            return coefficients, combined

        @tf.function(
            input_signature=(matrix, tf.TensorSpec([2], tf.int32, name="seed")),
            autograph=False,
            jit_compile=False,
        )
        def pcgrad(flat: Any, seed: Any) -> tuple[Any, Any]:
            from mooneural.multiobjective.pcgrad import pcgrad_flat

            coefficients, combined, _ = pcgrad_flat(
                flat, seed=seed, reduction="sum", eps=1.0e-12
            )
            return coefficients, combined

        @tf.function(input_signature=(matrix,), autograph=False, jit_compile=False)
        def mgda(flat: Any) -> tuple[Any, Any]:
            from mooneural.multiobjective.mgda import mgda_flat

            coefficients, combined, _ = mgda_flat(
                flat, eps=1.0e-12, max_objectives=8
            )
            return coefficients, combined

        @tf.function(
            input_signature=(
                matrix,
                vector,
                vector,
                vector,
                tf.TensorSpec([], tf.int64, name="step"),
            ),
            autograph=False,
            jit_compile=False,
        )
        def gradnorm(
            flat: Any,
            losses: Any,
            weights: Any,
            initial_losses: Any,
            step: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            from mooneural.multiobjective.gradnorm import gradnorm_update
            from mooneural.multiobjective.types import GradNormState

            update = gradnorm_update(
                flat,
                losses,
                GradNormState(
                    weights=weights,
                    initial_losses=initial_losses,
                    step=step,
                ),
                alpha=self.config.gradnorm_alpha,
                learning_rate=self.config.gradnorm_learning_rate,
                eps=1.0e-12,
            )
            combined = tf.linalg.matvec(flat, weights, transpose_a=True)
            return (
                weights,
                combined,
                update.state.weights,
                update.state.initial_losses,
                update.state.step,
            )

        return {
            "weighted_normalized_sum": weighted_normalized_sum,
            "cagrad": cagrad,
            "pcgrad": pcgrad,
            "mgda": mgda,
            "gradnorm": gradnorm,
        }

    def initial_gradnorm_state(self, losses: Any) -> dict[str, Any]:
        values = np.asarray(losses, dtype=np.float64)
        if values.shape != (self.objective_count,) or not np.all(
            np.isfinite(values) & (values > 0.0)
        ):
            raise ValueError("adaptive_direction_gradnorm_losses_invalid")
        return {
            "weights": np.ones(self.objective_count, dtype=np.float64).tolist(),
            "initial_losses": values.tolist(),
            "step": 0,
        }

    def combine(
        self,
        method: str,
        flat_gradients: Any,
        task_losses: Any,
        *,
        seed_index: int,
        method_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        import tensorflow as tf

        if method not in METHODS:
            raise ValueError("adaptive_direction_method_invalid")
        flat = tf.convert_to_tensor(flat_gradients, dtype=tf.float64)
        losses = np.asarray(task_losses, dtype=np.float64)
        if flat.shape != (self.objective_count, self.gradient_dim) or not bool(
            np.asarray(tf.reduce_all(tf.math.is_finite(flat)))
        ):
            raise ValueError("adaptive_direction_gradients_invalid")
        if losses.shape != (self.objective_count,) or not np.all(
            np.isfinite(losses) & (losses >= 0.0)
        ):
            raise ValueError("adaptive_direction_losses_invalid")

        next_state = None
        active = None
        if method == "pcgrad":
            output = self._functions[method](
                flat,
                tf.constant([self.config.pcgrad_seed, int(seed_index)], tf.int32),
            )
        elif method == "gradnorm":
            state = (
                self.initial_gradnorm_state(losses)
                if method_state is None
                else dict(method_state)
            )
            output = self._functions[method](
                flat,
                tf.convert_to_tensor(losses, tf.float64),
                tf.convert_to_tensor(state["weights"], tf.float64),
                tf.convert_to_tensor(state["initial_losses"], tf.float64),
                tf.constant(int(state["step"]), tf.int64),
            )
            next_weights = np.asarray(output[2], np.float64)
            if not np.all(np.isfinite(next_weights) & (next_weights > 0.0)):
                raise ValueError("adaptive_direction_gradnorm_weights_nonfinite")
            ratio = float(np.max(next_weights) / np.min(next_weights))
            if ratio > self.config.gradnorm_weight_ratio_veto:
                raise ValueError("adaptive_direction_gradnorm_weight_ratio_veto")
            next_state = {
                "weights": next_weights.tolist(),
                "initial_losses": np.asarray(output[3], np.float64).tolist(),
                "step": int(np.asarray(output[4])),
            }
        elif method == "weighted_normalized_sum":
            output = self._functions[method](flat)
            active = np.asarray(output[2], bool).tolist()
        else:
            output = self._functions[method](flat)

        coefficients = np.asarray(output[0], np.float64)
        combined = output[1]
        if coefficients.shape != (self.objective_count,) or not np.all(
            np.isfinite(coefficients)
        ):
            raise ValueError("adaptive_direction_coefficients_invalid")
        if not bool(np.asarray(tf.reduce_all(tf.math.is_finite(combined)))):
            raise ValueError("adaptive_direction_combined_nonfinite")
        return {
            "method": method,
            "coefficients": coefficients.tolist(),
            "combined_gradient": combined,
            "next_method_state": next_state,
            "active": active,
            "trace_count": int(
                self._functions[method].experimental_get_tracing_count()
            ),
        }

    def trace_counts(self) -> dict[str, int]:
        return {
            method: int(function.experimental_get_tracing_count())
            for method, function in self._functions.items()
        }


def validation_key(metrics: Mapping[str, Any]) -> tuple[int, float, float, float]:
    values = (
        int(metrics["failed_cells"]),
        float(metrics["worst_normalized_rms"]),
        float(metrics["median_normalized_rms"]),
        float(metrics["pooled_loss"]),
    )
    if values[0] < 0 or any(not math.isfinite(value) for value in values[1:]):
        raise ValueError("adaptive_validation_key_invalid")
    return values


def strictly_improves(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1.0e-12,
) -> bool:
    if not math.isfinite(float(absolute_tolerance)) or absolute_tolerance < 0.0:
        raise ValueError("adaptive_validation_tolerance_invalid")
    left = validation_key(candidate)
    right = validation_key(incumbent)
    if left[0] != right[0]:
        return left[0] < right[0]
    for candidate_value, incumbent_value in zip(left[1:], right[1:], strict=True):
        if candidate_value < incumbent_value - absolute_tolerance:
            return True
        if candidate_value > incumbent_value + absolute_tolerance:
            return False
    return False


def snapshot_variables(variables: Sequence[Any]) -> tuple[np.ndarray, ...]:
    values = tuple(np.asarray(value).copy() for value in variables)
    if not values:
        raise ValueError("adaptive_snapshot_variables_empty")
    return values


def restore_variables(variables: Sequence[Any], snapshot: Sequence[np.ndarray]) -> None:
    variables = tuple(variables)
    snapshot = tuple(snapshot)
    if len(variables) != len(snapshot) or not variables:
        raise ValueError("adaptive_snapshot_inventory_invalid")
    for variable, value in zip(variables, snapshot, strict=True):
        if tuple(variable.shape) != value.shape:
            raise ValueError("adaptive_snapshot_shape_invalid")
        variable.assign(value)


__all__ = [
    "AdaptiveControllerConfig",
    "AdaptiveDirectionSet",
    "AdaptiveMethodController",
    "DirectionConfig",
    "METHODS",
    "restore_variables",
    "snapshot_variables",
    "strictly_improves",
    "validation_key",
]
