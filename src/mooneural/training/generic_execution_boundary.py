"""Host-owned fixed-signature execution boundary; no membership policy here."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from itertools import combinations

import tensorflow as tf

from .generic_xla_kernels import (
    evaluate_impl,
    method_direction_impl,
    project_cone_impl,
    projected_adam_impl,
)
from .adam import packed_adam_impl

METHODS = ("cagrad", "pcgrad", "mgda", "gradnorm", "weighted_normalized_sum")
SOURCE_CONSTRAINT_COUNTS = tuple(range(7))
REACHABLE_CONSTRAINT_COUNTS = (4, 5, 6)
SUPPORTED_CONSTRAINT_COUNTS = tuple(range(9))
logger = logging.getLogger(__name__)


def _dimension(name, value, minimum, maximum=None):
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"invalid {name}")


def subset_masks(constraint_count):
    """Source combinations order, resolved once on the host, not while tracing."""
    _dimension("constraint_count", constraint_count, 0, 8)
    masks = [
        [index in subset for index in range(constraint_count)]
        for size in range(1, constraint_count + 1)
        for subset in combinations(range(constraint_count), size)
    ]
    return tf.reshape(
        tf.constant(masks, tf.bool), [(1 << constraint_count) - 1, constraint_count]
    )


def jacobi_schedule():
    positions = list(range(8))
    rounds = []
    for _round in range(7):
        rounds.append([[positions[index], positions[7 - index]] for index in range(4)])
        positions = [positions[0], positions[-1], *positions[1:-1]]
    return tf.constant(rounds, tf.int32)


def pcgrad_permutations(active_count, update_index, seed=20260722):
    """Resolve the source stateless shuffle on the host as an explicit input."""
    _dimension("active_count", active_count, 1, 8)
    _dimension("update_index", update_index, 0, 2147483647 - active_count)
    _dimension("seed", seed, 0, 2147483647)
    return tf.stack(
        [
            tf.random.experimental.stateless_shuffle(
                tf.range(active_count, dtype=tf.int32),
                tf.constant([seed, update_index + index + 1], tf.int32),
            )
            for index in range(active_count)
        ]
    )


@dataclass(frozen=True)
class ExecutionSpec:
    task_count: int
    parameter_dim: int
    batch_size: int
    hard_check_count: int
    device: str = "/CPU:0"

    def __post_init__(self):
        _dimension("task_count", self.task_count, 1)
        _dimension("parameter_dim", self.parameter_dim, 1)
        _dimension("batch_size", self.batch_size, 1)
        _dimension("hard_check_count", self.hard_check_count, 0)
        if self.device != "/CPU:0":
            raise ValueError(
                "Phase 2 qualifies explicit CPU XLA only; other routes need qualification"
            )


@dataclass(frozen=True)
class AdamConfig:
    clip_norm: float = 10.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-7
    epsilon_coordinate: str = "keras_epsilon_hat"
    bias_power_coordinate: str = "keras_float32_cast"

    def __post_init__(self):
        for name in ("clip_norm", "beta1", "beta2", "epsilon"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid {name}")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("Adam beta outside [0,1)")
        if self.clip_norm <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("Adam clip/epsilon must be positive")
        if self.epsilon_coordinate not in ("keras_epsilon_hat", "bias_corrected"):
            raise ValueError("unknown Adam epsilon coordinate")
        if self.bias_power_coordinate not in ("keras_float32_cast", "float64"):
            raise ValueError("unknown Adam bias-power coordinate")


DEFAULT_ADAM = AdamConfig()


def make_evaluation_function(adapter, spec):
    if (
        len(adapter.registry.tasks) != spec.task_count
        or len(adapter.hard_check_ids) != spec.hard_check_count
    ):
        raise ValueError("adapter registries do not match fixed signature")
    if (
        tuple(item.task_id for item in adapter.normalizations)
        != adapter.registry.task_ids
    ):
        raise ValueError("normalization task order mismatch")
    factors = tf.constant(
        [item.training.factor for item in adapter.normalizations], tf.float64
    )
    signature = [
        tf.TensorSpec([spec.parameter_dim], tf.float64, "parameters"),
        tf.TensorSpec(
            [spec.task_count, spec.batch_size, spec.parameter_dim],
            tf.float64,
            "features",
        ),
        tf.TensorSpec([spec.task_count, spec.batch_size], tf.float64, "targets"),
        tf.TensorSpec([spec.task_count, spec.batch_size], tf.float64, "sample_weights"),
        tf.TensorSpec([], tf.int64, "update_index"),
    ]

    @tf.function(input_signature=signature, autograph=False, jit_compile=True)
    def evaluate(parameters, features, targets, sample_weights, update_index):
        with tf.device(spec.device):
            return evaluate_impl(
                adapter,
                parameters,
                features,
                targets,
                sample_weights,
                factors,
                update_index,
                spec.task_count,
                spec.parameter_dim,
                spec.hard_check_count,
            )

    return evaluate


def make_method_function(
    method, active_count, parameter_dim, *, reset_method_state=True
):
    if method not in METHODS or type(reset_method_state) is not bool:
        raise ValueError("invalid method profile")
    _dimension("active_count", active_count, 1, 8)
    _dimension("parameter_dim", parameter_dim, 1)
    signature = [
        tf.TensorSpec([active_count, parameter_dim], tf.float64, "rows"),
        tf.TensorSpec([active_count], tf.float64, "losses"),
        tf.TensorSpec([active_count, active_count], tf.int32, "permutations"),
        tf.TensorSpec([active_count], tf.float64, "method_weights"),
        tf.TensorSpec([active_count], tf.float64, "initial_losses"),
        tf.TensorSpec([], tf.int64, "method_step"),
    ]

    @tf.function(input_signature=signature, autograph=False, jit_compile=True)
    def direction(
        rows, losses, permutations, method_weights, initial_losses, method_step
    ):
        with tf.device("/CPU:0"):
            return method_direction_impl(
                rows,
                losses,
                permutations,
                method_weights,
                initial_losses,
                method_step,
                method,
                active_count,
                parameter_dim,
                reset_method_state,
            )

    return direction


def make_projected_adam_function(parameter_dim, constraint_count, config=DEFAULT_ADAM):
    _dimension("constraint_count", constraint_count, 0, 6)
    return _make_projected_adam_function(
        parameter_dim, constraint_count, config,
        projector_impl=project_cone_impl,
        packed_impl=packed_adam_impl,
        jit_compile=True,
    )


def _make_projected_adam_function(
    parameter_dim, constraint_count, config, *, projector_impl, packed_impl, jit_compile
):
    _dimension("parameter_dim", parameter_dim, 1)
    masks = subset_masks(constraint_count)
    schedule = jacobi_schedule()
    constants = [
        tf.constant(getattr(config, name), tf.float64)
        for name in ("clip_norm", "beta1", "beta2", "epsilon")
    ]
    power_dtype = (
        tf.float32
        if config.bias_power_coordinate == "keras_float32_cast"
        else tf.float64
    )
    power_betas = [
        tf.cast(tf.constant(getattr(config, name), power_dtype), tf.float64)
        for name in ("beta1", "beta2")
    ]
    signature = [
        tf.TensorSpec([parameter_dim], tf.float64, "parameters"),
        tf.TensorSpec([parameter_dim], tf.float64, "first_moment"),
        tf.TensorSpec([parameter_dim], tf.float64, "second_moment"),
        tf.TensorSpec([], tf.int64, "iteration"),
        tf.TensorSpec([parameter_dim], tf.float64, "direction"),
        tf.TensorSpec([constraint_count, parameter_dim], tf.float64, "protected_rows"),
        tf.TensorSpec([], tf.float64, "learning_rate"),
    ]

    @tf.function(input_signature=signature, autograph=False, jit_compile=jit_compile)
    def update(
        parameters,
        first_moment,
        second_moment,
        iteration,
        direction,
        protected_rows,
        learning_rate,
    ):
        with tf.device("/CPU:0"):
            return projected_adam_impl(
                parameters,
                first_moment,
                second_moment,
                iteration,
                direction,
                protected_rows,
                learning_rate,
                masks,
                schedule,
                *constants,
                config.epsilon_coordinate,
                *power_betas,
                projector_impl=projector_impl,
                packed_impl=packed_impl,
            )

    return update


def make_partition_function(spec, active_indices, constraint_indices):
    active, constrained = tuple(active_indices), tuple(constraint_indices)
    if any(type(index) is not int for index in active + constrained):
        raise ValueError("partition indices must be integers")
    if not 1 <= len(active) <= 8 or len(constrained) not in SUPPORTED_CONSTRAINT_COUNTS:
        raise ValueError("unsupported partition specialization")
    if len(set(active + constrained)) != spec.task_count or sorted(
        active + constrained
    ) != list(range(spec.task_count)):
        raise ValueError(
            "active and constrained indices must partition the task registry"
        )
    active_tensor = tf.constant(active, tf.int32)
    constraint_tensor = tf.constant(constrained, tf.int32)

    @tf.function(
        input_signature=[
            tf.TensorSpec([spec.task_count], tf.float64, "values"),
            tf.TensorSpec([spec.task_count, spec.parameter_dim], tf.float64, "rows"),
        ],
        autograph=False,
        jit_compile=True,
    )
    def partition(values, rows):
        with tf.device(spec.device):
            return (
                tf.gather(rows, active_tensor),
                tf.gather(values, active_tensor),
                tf.gather(rows, constraint_tensor),
            )

    return partition


def compiled_evidence(function, arguments):
    """Host-only inventories include called FunctionDefs and optimized HLO."""
    concrete = function.get_concrete_function()
    graph = concrete.graph.as_graph_def()
    nodes = list(graph.node)
    for definition in graph.library.function:
        nodes.extend(definition.node_def)
    operations = sorted({node.op for node in nodes})
    forbidden = [
        operation
        for operation in operations
        if any(
            token in operation.lower()
            for token in (
                "pyfunc",
                "while",
                "mapdefun",
                "pfor",
                "variable",
                "assign",
                "random",
                "rng",
            )
        )
    ]
    hlo = function.experimental_get_compiler_ir(*arguments)(stage="optimized_hlo")
    if re.search(
        r"\bwhile\s*\(|\bmap\s*\(|host_callback|python_callback", hlo, re.IGNORECASE
    ):
        forbidden.append("forbidden optimized HLO operation")
    if forbidden:
        raise ValueError(f"forbidden compiled operations: {forbidden}")
    if function.experimental_get_tracing_count() != 1:
        raise ValueError("kernel must have exactly one trace")
    return {
        "operations": operations,
        "node_count": len(nodes),
        "trace_count": 1,
        "input_signature": [
            {"name": item.name, "shape": item.shape.as_list(), "dtype": item.dtype.name}
            for item in function.input_signature
        ],
        "jit_compile": True,
        "device": "/CPU:0",
        "forbidden_operations": forbidden,
        "graph_def": graph.SerializeToString(),
        "optimized_hlo": hlo,
    }


class MethodFallback:
    """Source event-local cyclic fallback; preferred method never silently changes."""

    def __init__(self, active_count, parameter_dim, rates, preferred="cagrad", *, method_factory=None):
        if preferred not in METHODS or set(rates) != set(METHODS):
            raise ValueError("method/rate inventory mismatch")
        if any(
            type(value) not in (float, int) or not math.isfinite(value) or value <= 0.0
            for value in rates.values()
        ):
            raise ValueError("method rates must be finite and positive")
        self.active_count = active_count
        self.preferred = preferred
        self.rates = dict(rates)
        self.events = []
        self.functions = {
            method: (method_factory or make_method_function)(method, active_count, parameter_dim)
            for method in METHODS
        }

    def choose(self, rows, losses, update_index):
        permutations = pcgrad_permutations(self.active_count, update_index)
        arguments = (
            rows,
            losses,
            permutations,
            tf.ones([self.active_count], tf.float64),
            losses,
            tf.constant(0, tf.int64),
        )
        start = METHODS.index(self.preferred)
        for offset in range(len(METHODS)):
            method = METHODS[(start + offset) % len(METHODS)]
            try:
                output = self.functions[method](*arguments)
                accepted = bool(output[-1].numpy())
                reason = (
                    "valid_direction" if accepted else "invalid_numerical_direction"
                )
            except Exception as error:
                logger.exception(
                    "Compiled method %s failed at update %d", method, update_index
                )
                accepted = False
                reason = f"{type(error).__name__}:{error}"
            previous_rate = self.rates[method]
            if not accepted:
                self.rates[method] *= 0.5
            self.events.append(
                {
                    "update_index": update_index,
                    "method": method,
                    "accepted": accepted,
                    "previous_rate": previous_rate,
                    "next_rate": self.rates[method],
                    "reason": reason,
                }
            )
            if accepted:
                return method, output, tf.constant(self.rates[method], tf.float64)
        raise ValueError("all compiled methods failed; no update is authorized")


class PostproposalRejected(ValueError):
    """An optional proposal failed without changing the source-method verdict."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def numerical_batch_binding(batch, update_index):
    """Identity of the exact tensor inputs supplied to a direct boundary call."""
    return {"update": update_index, "batch": [value.numpy().tolist() for value in batch]}


class CompiledBoundary:
    """One resource-free numerical transaction with an externally supplied partition."""

    def __init__(
        self,
        adapter,
        spec,
        active_indices,
        constraint_indices,
        rates,
        *,
        preferred="cagrad",
        adam=DEFAULT_ADAM,
        kernel_cache=None,
        evaluation_factory=None,
        update_factory=None,
        method_factory=None,
        postproposal=None,
    ):
        cache = {} if kernel_cache is None else kernel_cache
        selected_factory = (
            make_projected_adam_function if update_factory is None else update_factory
        )
        if not callable(selected_factory):
            raise TypeError("update factory must be callable")
        binding = (id(adapter), spec, adam, evaluation_factory)
        if cache.get("binding", binding) != binding:
            raise ValueError("kernel cache adapter/spec/Adam binding mismatch")
        if ("binding" in cache or "update_factory" in cache) and cache.get(
            "update_factory", make_projected_adam_function
        ) is not selected_factory:
            raise ValueError("kernel cache update factory binding mismatch")
        cache.setdefault("binding", binding)
        cache.setdefault("update_factory", selected_factory)
        selected_method_factory = method_factory or make_method_function
        if cache.get("method_factory", selected_method_factory) is not selected_method_factory:
            raise ValueError("kernel cache method factory binding mismatch")
        cache.setdefault("method_factory", selected_method_factory)
        if cache.get("postproposal", postproposal) is not postproposal:
            raise ValueError("kernel cache postproposal binding mismatch")
        if postproposal is not None:
            cache.setdefault("postproposal", postproposal)
        self.postproposal = postproposal
        self.active_indices = tuple(active_indices)
        self.constraint_indices = tuple(constraint_indices)

        def cached(key, build):
            if key not in cache:
                cache[key] = build()
            return cache[key]

        self.evaluate = cached(
            "evaluation",
            lambda: (evaluation_factory or make_evaluation_function)(adapter, spec),
        )
        self.partition = cached(
            ("partition", tuple(active_indices), tuple(constraint_indices)),
            lambda: make_partition_function(spec, active_indices, constraint_indices),
        )
        self.methods = MethodFallback(
            len(active_indices), spec.parameter_dim, rates, preferred,
            method_factory=selected_method_factory,
        )
        self.methods.functions = dict(
            cached(
                ("method_functions", len(active_indices)),
                lambda: self.methods.functions,
            )
        )
        self.update = cached(
            ("adam", len(constraint_indices)),
            lambda: selected_factory(
                spec.parameter_dim, len(constraint_indices), adam
            ),
        )
        self.last_arguments = {}

    def step(self, state, batch, update_index, *, objective_context=None):
        _dimension("update_index", update_index, 0, 2147483639)
        evaluation_args = (state[0], *batch, tf.constant(update_index, tf.int64))
        evaluation = self.evaluate(*evaluation_args)
        if not bool(evaluation[-1].numpy()):
            raise ValueError("invalid task evaluation; no method or optimizer call")
        partition_args = (evaluation[2], evaluation[3])
        rows, losses, constraints = self.partition(*partition_args)
        method, direction, learning_rate = self.methods.choose(
            rows, losses, update_index
        )
        update_args = (*state, direction[1], constraints, learning_rate)
        result = self.update(*update_args)
        self.last_arguments = {
            "evaluation": evaluation_args,
            "partition": partition_args,
            "adam": update_args,
        }
        if not bool(result[-1].numpy()):
            raise ValueError(
                "invalid projected optimizer proposal; all tensor state rolled back"
            )
        receipt = None
        if self.postproposal is not None:
            context = numerical_batch_binding(batch, update_index) if objective_context is None else objective_context
            result, receipt = self.postproposal(
                parameters=state[0], batch=batch if objective_context is None else objective_context,
                update_index=update_index, gradient_batch_binding=context,
                raw_losses=evaluation[0], normalized_losses=evaluation[2],
                gradient_rows=evaluation[3], optimizer=result,
                active_indices=self.active_indices,
                constraint_indices=self.constraint_indices,
            )
        evidence = {
            "method": method,
            "evaluation": evaluation,
            "optimizer": result,
        }
        if receipt is not None:
            evidence["postproposal"] = receipt
        return result[:4], evidence
