"""Bounded CPU-XLA qualification; eager/NumPy are independent test oracles only."""

from functools import lru_cache
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow as tf

from mooneural.multiobjective.adaptive_controller import AdaptiveDirectionSet
from mooneural.training import generic_execution_boundary as boundary_module
from mooneural.training.generic_execution_boundary import (
    DEFAULT_ADAM,
    METHODS,
    CompiledBoundary,
    ExecutionSpec,
    MethodFallback,
    compiled_evidence,
    jacobi_schedule,
    make_evaluation_function,
    make_method_function,
    make_projected_adam_function,
    pcgrad_permutations,
    subset_masks,
)
from mooneural.training.generic_tensor_fixture import (
    QuadraticTensorFixture,
    fixture_inputs,
)
from mooneural.training.generic_xla_kernels import (
    fixed_linear_solve_impl,
    fixed_symmetric_pinv_impl,
    project_cone_impl,
)
from tests.support.working_set import (
    _project_intersection_tf,
    prepare_working_set_projected_adam_updates,
)


@lru_cache(None)
def projection_function(count):
    masks, schedule = subset_masks(count), jacobi_schedule()

    @tf.function(
        input_signature=[
            tf.TensorSpec([8], tf.float64),
            tf.TensorSpec([count, 8], tf.float64),
        ],
        autograph=False,
        jit_compile=True,
    )
    def project(direction, rows):
        return project_cone_impl(
            direction, rows, masks, schedule, tf.constant(1.0e-12, tf.float64)
        )

    return project


@lru_cache(None)
def adam_function(count, config=DEFAULT_ADAM):
    return make_projected_adam_function(8, count, config)


@lru_cache(None)
def method_function(method, count):
    return make_method_function(method, count, 8)


@lru_cache(None)
def source_methods(count):
    return AdaptiveDirectionSet(8, objective_count=count)


def method_arguments(rows, losses, index=5):
    count = rows.shape[0]
    return (
        rows,
        losses,
        pcgrad_permutations(count, index),
        tf.ones([count], tf.float64),
        losses,
        tf.constant(0, tf.int64),
    )


@pytest.fixture
def update_backend_factories(monkeypatch):
    factories = {
        name: Mock(side_effect=lambda *_args, **_kwargs: object())
        for name in ("make_evaluation_function", "make_partition_function", "make_projected_adam_function")
    }
    for name, factory in factories.items():
        monkeypatch.setattr(boundary_module, name, factory)
    monkeypatch.setattr(
        boundary_module, "MethodFallback",
        Mock(side_effect=lambda *_args, **_kwargs: SimpleNamespace(functions={"cagrad": object()})),
    )
    return factories


def test_update_backend_cache_reuses_only_the_same_factory(update_backend_factories):
    adapter, cache = object(), {}
    spec = ExecutionSpec(7, 8, 2, 1)
    custom = Mock(side_effect=lambda *_args: object())
    for count in (4, 5, 6):
        arguments = (adapter, spec, tuple(range(7 - count)), tuple(range(7 - count, 7)), {})
        first = CompiledBoundary(*arguments, kernel_cache=cache, update_factory=custom)
        second = CompiledBoundary(*arguments, kernel_cache=cache, update_factory=custom)
        assert first.update is second.update
        assert first.evaluate is second.evaluate
    assert [call.args for call in custom.call_args_list] == [
        (8, count, DEFAULT_ADAM) for count in (4, 5, 6)
    ]
    update_backend_factories["make_projected_adam_function"].assert_not_called()
    before = dict(cache)
    different = Mock(side_effect=AssertionError("factory must not be built"))
    with pytest.raises(ValueError, match="update factory binding"):
        CompiledBoundary(*arguments, kernel_cache=cache, update_factory=different)
    assert cache == before
    different.assert_not_called()


def test_update_backend_default_and_explicit_default_share_cache(update_backend_factories):
    cache = {}
    arguments = (object(), ExecutionSpec(7, 8, 2, 1), (0, 1, 2), (3, 4, 5, 6), {})
    default = update_backend_factories["make_projected_adam_function"]
    first = CompiledBoundary(*arguments, kernel_cache=cache)
    second = CompiledBoundary(*arguments, kernel_cache=cache, update_factory=default)
    assert first.update is second.update
    default.assert_called_once_with(8, 4, DEFAULT_ADAM)
    before = dict(cache)
    with pytest.raises(ValueError, match="update factory binding"):
        CompiledBoundary(*arguments, kernel_cache=cache, update_factory=Mock())
    assert cache == before
    assert CompiledBoundary(*arguments, kernel_cache=cache).update is first.update


def test_update_backend_cache_refusals_leave_cache_unchanged(update_backend_factories):
    adapter, cache = object(), {}
    arguments = (adapter, ExecutionSpec(7, 8, 2, 1), (0, 1, 2), (3, 4, 5, 6), {})
    with pytest.raises(TypeError, match="update factory must be callable"):
        CompiledBoundary(*arguments, kernel_cache=cache, update_factory=7)
    assert cache == {}
    CompiledBoundary(*arguments, kernel_cache=cache)
    before = dict(cache)
    with pytest.raises(ValueError, match="adapter/spec/Adam"):
        CompiledBoundary(object(), *arguments[1:], kernel_cache=cache, update_factory=Mock())
    assert cache == before


def test_jacobi_all_pairs_once_and_rank_deficient_pinv():
    schedule = jacobi_schedule()
    pairs = schedule.numpy().reshape(-1, 2)
    assert len({tuple(sorted(pair)) for pair in pairs}) == 28
    random = np.random.default_rng(1201)
    rows = random.normal(size=(12, 6, 8))
    rows[::2, 3:, :] = rows[::2, :3, :]
    gram = np.zeros((12, 8, 8))
    gram[:, :6, :6] = rows @ np.transpose(rows, (0, 2, 1))
    inverse, residual, valid = fixed_symmetric_pinv_impl(
        tf.constant(gram), schedule, tf.constant(1.0e-12, tf.float64)
    )
    assert bool(valid)
    assert np.max(residual) < 1.0e-12
    np.testing.assert_allclose(
        inverse, np.linalg.pinv(gram, rcond=1.0e-12), rtol=1.0e-10, atol=1.0e-10
    )


@pytest.mark.parametrize("count", range(7))
def test_projection_source_parity_and_compiler(count):
    random = np.random.default_rng(912 + count)
    direction = tf.constant(random.normal(size=8))
    rows = tf.constant(random.normal(size=(count, 8)))
    function = projection_function(count)
    result = function(direction, rows)
    assert bool(result[-1])
    expected = _project_intersection_tf(
        direction, rows, constraint_count=count, eta=1.0e-12
    )
    np.testing.assert_allclose(result[0], expected, rtol=1.0e-10, atol=1.0e-10)
    assert np.all(result[1].numpy() >= -1.0e-10)
    evidence = compiled_evidence(function, (direction, rows))
    assert evidence["trace_count"] == 1


@pytest.mark.parametrize("count", [2, 7])
def test_evaluation_derivatives_and_normalization(count):
    factors = tuple(10.0 ** (index - count // 2) for index in range(count))
    adapter = QuadraticTensorFixture(count, 3, factors)
    function = make_evaluation_function(adapter, ExecutionSpec(count, 8, 4, 3))
    inputs = fixture_inputs(count)
    output = function(*inputs)
    eager = function.python_function(*inputs)
    assert bool(output[-1])
    for compiled_value, eager_value in zip(output[:-2], eager[:-2]):
        np.testing.assert_allclose(
            compiled_value, eager_value, rtol=1.0e-12, atol=1.0e-12
        )
    parameters, features, targets, weights, _index = inputs
    expected_rows = []
    for index in range(count):
        with tf.GradientTape() as tape:
            tape.watch(parameters)
            loss = tf.reduce_mean(
                weights[index]
                * tf.square(
                    tf.linalg.matvec(features[index], parameters) - targets[index]
                )
            )
        expected_rows.append(tape.gradient(loss, parameters))
    np.testing.assert_allclose(output[1], expected_rows, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(output[2], output[0] * factors, rtol=1.0e-12)
    np.testing.assert_allclose(
        output[3], output[1] * np.array(factors)[:, None], rtol=1.0e-12
    )
    assert compiled_evidence(function, inputs)["trace_count"] == 1


def test_unconstrained_adam_keras_parity():
    function = adam_function(0)
    parameters = tf.Variable(np.linspace(-0.5, 0.5, 8), dtype=tf.float64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3.0e-4)
    optimizer.build([parameters])
    state = (
        tf.identity(parameters),
        tf.zeros([8], tf.float64),
        tf.zeros([8], tf.float64),
        tf.constant(0, tf.int64),
    )
    for index in range(5):
        gradient = tf.constant(np.linspace(-3.0, 7.0, 8) * (index + 1))
        arguments = (
            *state,
            gradient,
            tf.zeros([0, 8], tf.float64),
            tf.constant(float(optimizer.learning_rate.numpy()), tf.float64),
        )
        result = function(*arguments)
        assert bool(result[-1])
        clipped, _norm = tf.clip_by_global_norm([gradient], 10.0)
        optimizer.apply_gradients(zip(clipped, [parameters]))
        np.testing.assert_allclose(result[0], parameters, rtol=0, atol=1.0e-10)
        np.testing.assert_allclose(
            result[1], optimizer.variables[2], rtol=1.0e-12, atol=1.0e-12
        )
        np.testing.assert_allclose(
            result[2], optimizer.variables[3], rtol=1.0e-12, atol=1.0e-12
        )
        state = result[:4]
    assert compiled_evidence(function, arguments)["trace_count"] == 1


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("method", METHODS)
def test_moo_frozen_source_parity(count, method):
    random = np.random.default_rng(110 + count)
    rows = tf.constant(random.normal(size=(count, 8)))
    losses = tf.constant(np.linspace(0.1, 1.0, count))
    arguments = method_arguments(rows, losses)
    function = method_function(method, count)
    output = function(*arguments)
    expected = source_methods(count).combine(
        method, rows, losses.numpy(), seed_index=5, method_state=None
    )
    assert bool(output[-1])
    np.testing.assert_allclose(
        output[0], expected["coefficients"], rtol=1.0e-9, atol=1.0e-10
    )
    np.testing.assert_allclose(
        output[1], expected["combined_gradient"], rtol=1.0e-9, atol=1.0e-10
    )
    if method == "gradnorm":
        np.testing.assert_allclose(
            output[2],
            expected["next_method_state"]["weights"],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    assert compiled_evidence(function, arguments)["trace_count"] == 1


@pytest.mark.parametrize("count", [4, 5, 6])
def test_projection_rank_deficient_and_contradictory(count):
    direction = tf.constant([1.0, -2.0, 3.0, -1.0, 2.0, 1.0, 1.0, 0.0], tf.float64)
    base = np.eye(8)[:count]
    base[1] = -base[0]
    base[-1] = base[0]
    rows = tf.constant(base)
    output = projection_function(count)(direction, rows)
    assert bool(output[-1])
    expected = _project_intersection_tf(
        direction, rows, constraint_count=count, eta=1.0e-12
    )
    np.testing.assert_allclose(output[0], expected, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fault", ["zero", "nonfinite", "negative_loss"])
def test_methods_fail_closed(method, fault):
    rows = np.ones((3, 8))
    losses = np.ones(3)
    if fault == "zero":
        rows[:] = 0.0
    elif fault == "nonfinite":
        rows[0, 0] = np.nan
    else:
        losses[0] = -1.0
    output = method_function(method, 3)(
        *method_arguments(tf.constant(rows), tf.constant(losses))
    )
    assert not bool(output[-1])


def test_fallback_is_event_local_and_exhaustion_is_not_a_zero_update():
    fallback = MethodFallback(
        3, 8, dict.fromkeys(METHODS, 3.0e-4), preferred="gradnorm"
    )
    fallback.functions = {method: method_function(method, 3) for method in METHODS}
    rows = tf.constant(np.eye(8)[:3])
    method, _output, rate = fallback.choose(rows, tf.zeros([3], tf.float64), 0)
    assert method == "weighted_normalized_sum"
    assert fallback.preferred == "gradnorm"
    assert fallback.rates["gradnorm"] == 1.5e-4
    assert float(rate) == 3.0e-4
    method, _output, _rate = fallback.choose(rows, tf.ones([3], tf.float64), 1)
    assert method == "gradnorm"
    with pytest.raises(ValueError, match="all compiled methods failed"):
        fallback.choose(tf.zeros([3, 8], tf.float64), tf.ones([3], tf.float64), 2)
    assert len(fallback.events) == 8


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6])
def test_double_projection_source_optimizer_and_moment_parity(count):
    parameters = tf.Variable(np.linspace(-0.5, 0.5, 8), dtype=tf.float64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3.0e-4)
    source = prepare_working_set_projected_adam_updates([parameters], optimizer)[count]
    optimizer.variables[2].assign(np.linspace(0.4, -0.4, 8))
    optimizer.variables[3].assign(np.linspace(0.01, 0.1, 8))
    optimizer.iterations.assign(9)
    state = (
        tf.identity(parameters),
        tf.identity(optimizer.variables[2]),
        tf.identity(optimizer.variables[3]),
        tf.constant(9, tf.int64),
    )
    random = np.random.default_rng(778 + count)
    rows = tf.constant(random.normal(size=(7, 8)))
    direction = tf.constant(random.normal(size=8))
    indices = tf.constant(list(range(count)), tf.int32)
    arguments = (
        *state,
        direction,
        tf.gather(rows, indices),
        tf.constant(float(optimizer.learning_rate.numpy()), tf.float64),
    )
    output = adam_function(count)(*arguments)
    source(rows, direction, indices)
    assert bool(output[-1])
    np.testing.assert_allclose(output[0], parameters, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        output[1], optimizer.variables[2], rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        output[2], optimizer.variables[3], rtol=1.0e-10, atol=1.0e-10
    )
    assert int(output[3]) == 10
    assert np.all(output[7].numpy() >= -1.0e-10)
    assert np.all(output[8].numpy() <= 1.0e-10)
    assert compiled_evidence(adam_function(count), arguments)["trace_count"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "zero_constraint",
        "nonfinite_direction",
        "negative_second_moment",
        "negative_iteration",
        "iteration_overflow",
        "invalid_rate",
    ],
)
def test_optimizer_atomic_rollback(fault):
    state = [
        tf.ones([8], tf.float64),
        tf.zeros([8], tf.float64),
        tf.zeros([8], tf.float64),
        tf.constant(0, tf.int64),
    ]
    direction = tf.ones([8], tf.float64)
    constraints = tf.constant(np.eye(8)[:4])
    rate = tf.constant(3.0e-4, tf.float64)
    if fault == "zero_constraint":
        constraints = tf.zeros([4, 8], tf.float64)
    elif fault == "nonfinite_direction":
        direction = tf.fill([8], tf.constant(float("nan"), tf.float64))
    elif fault == "negative_second_moment":
        state[2] = -tf.ones([8], tf.float64)
    elif fault == "negative_iteration":
        state[3] = tf.constant(-1, tf.int64)
    elif fault == "iteration_overflow":
        state[3] = tf.constant(9223372036854775807, tf.int64)
    else:
        rate = -rate
    output = adam_function(4)(*state, direction, constraints, rate)
    assert not bool(output[-1])
    for actual, expected in zip(output[:4], state):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(output[6], np.zeros(8))


@pytest.mark.parametrize(
    "fault", ["domain", "nonfinite", "disconnected", "negative_value"]
)
def test_adapter_failure_prevents_method_and_optimizer(fault):
    class FaultyAdapter(QuadraticTensorFixture):
        def compute_task_tensors(self, *arguments):
            values, rows, hard, domain, connected = super().compute_task_tensors(
                *arguments
            )
            if fault == "disconnected":
                connected = tf.constant([True, False])
            if fault == "negative_value":
                values = -values
            return values, rows, hard, domain, connected

    inputs = list(fixture_inputs(2))
    if fault == "domain":
        inputs[3] = -inputs[3]
    elif fault == "nonfinite":
        inputs[1] = tf.fill([2, 4, 8], tf.constant(float("inf"), tf.float64))
    boundary = CompiledBoundary(
        FaultyAdapter(),
        ExecutionSpec(2, 8, 4, 1),
        [0],
        [1],
        dict.fromkeys(METHODS, 3.0e-4),
    )
    state = (
        inputs[0],
        tf.zeros([8], tf.float64),
        tf.zeros([8], tf.float64),
        tf.constant(0, tf.int64),
    )
    with pytest.raises(ValueError, match="invalid task evaluation"):
        boundary.step(state, inputs[1:4], 0)
    assert not boundary.methods.events
    assert boundary.update.experimental_get_tracing_count() == 0


def test_fixed_solve_pivoting_and_singular_refusal():
    random = np.random.default_rng(88)
    matrix = random.normal(size=(4, 9, 9))
    rhs = random.normal(size=(4, 9, 2))
    solved = fixed_linear_solve_impl(tf.constant(matrix), tf.constant(rhs))
    np.testing.assert_allclose(
        solved, np.linalg.solve(matrix, rhs), rtol=1.0e-10, atol=1.0e-10
    )
    matrix[0] = 0.0
    solved = fixed_linear_solve_impl(tf.constant(matrix), tf.constant(rhs))
    assert not np.any(np.isfinite(solved[0]))
    np.testing.assert_allclose(
        solved[1:], np.linalg.solve(matrix[1:], rhs[1:]), rtol=1.0e-10, atol=1.0e-10
    )


@pytest.mark.parametrize("scale", [1.0e-10, 1.0, 1.0e10])
def test_weighted_dormant_floor_and_scale_parity(scale):
    rows = np.zeros((3, 8))
    rows[:, 0] = np.array([1.0, 1.0e-4, 1.0e-14]) * scale
    losses = tf.ones([3], tf.float64)
    output = method_function("weighted_normalized_sum", 3)(
        *method_arguments(tf.constant(rows), losses)
    )
    expected = source_methods(3).combine(
        "weighted_normalized_sum", rows, losses.numpy(), seed_index=5
    )
    np.testing.assert_allclose(
        output[0], expected["coefficients"], rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        output[1], expected["combined_gradient"], rtol=1.0e-12, atol=1.0e-12
    )


def test_fixed_signature_refuses_shape_dtype_and_bad_partition():
    function = make_evaluation_function(
        QuadraticTensorFixture(), ExecutionSpec(2, 8, 4, 1)
    )
    inputs = list(fixture_inputs(2))
    inputs[0] = tf.ones([9], tf.float64)
    with pytest.raises((TypeError, ValueError)):
        function(*inputs)
    inputs[0] = tf.ones([8], tf.float32)
    with pytest.raises((TypeError, ValueError)):
        function(*inputs)
    with pytest.raises(ValueError, match="partition"):
        CompiledBoundary(
            QuadraticTensorFixture(),
            ExecutionSpec(2, 8, 4, 1),
            [0],
            [0],
            dict.fromkeys(METHODS, 3.0e-4),
        )


def test_compiler_gate_rejects_dynamic_loop():
    @tf.function(
        input_signature=[tf.TensorSpec([], tf.float64)],
        autograph=False,
        jit_compile=True,
    )
    def forbidden(value):
        return tf.while_loop(
            lambda current: current < 3.0, lambda current: (current + 1.0,), (value,)
        )

    with pytest.raises(ValueError, match="forbidden"):
        compiled_evidence(forbidden, (tf.constant(0.0, tf.float64),))


@pytest.mark.parametrize(
    "losses",
    [[1.0e-10, 1.0e-9], [1.0e-20, 1.0e-19], [1.0e-7, 1.0e-6], [1.0e-10, 1.0e-10]],
)
def test_gradnorm_source_floor_acceptance_and_refusal(losses):
    rows = tf.constant(np.eye(8)[:2] * 10000.0)
    output = method_function("gradnorm", 2)(
        *method_arguments(rows, tf.constant(losses, tf.float64))
    )
    try:
        expected = source_methods(2).combine("gradnorm", rows, losses, seed_index=5)
    except ValueError as error:
        assert "weight_ratio_veto" in str(error)
        assert not bool(output[-1])
    else:
        assert bool(output[-1])
        np.testing.assert_allclose(
            output[2],
            expected["next_method_state"]["weights"],
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_rank_ambiguity_at_unchanged_source_cutoff_is_refused():
    rows = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-np.cos(0.7), -np.sin(0.7), 2.0e-6 * 1.00001],
        ]
    )
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    gram = np.zeros((1, 8, 8))
    gram[0, :3, :3] = rows @ rows.T
    _inverse, residual, valid = fixed_symmetric_pinv_impl(
        tf.constant(gram), jacobi_schedule(), tf.constant(1.0e-12, tf.float64)
    )
    assert float(residual[0]) < 1.0e-13
    assert not bool(valid)


def test_fixed_solve_indefinite_zero_diagonal_pivot():
    matrix = tf.constant(
        [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], tf.float64
    )
    rhs = tf.constant([[[2.0], [3.0]], [[1.0], [1.0]]], tf.float64)
    solution = fixed_linear_solve_impl(matrix, rhs)
    np.testing.assert_allclose(solution[0], [[3.0], [2.0]], rtol=0.0, atol=0.0)
    assert not np.any(np.isfinite(solution[1]))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("profile", ["duplicate", "opposite", "identical"])
def test_degenerate_objective_geometry_matches_source(method, profile):
    rows = np.eye(8)[:3]
    if profile == "duplicate":
        rows[1] = rows[0]
    elif profile == "opposite":
        rows[1] = -rows[0]
    else:
        rows[:] = rows[0]
    losses = tf.ones([3], tf.float64)
    output = method_function(method, 3)(*method_arguments(tf.constant(rows), losses))
    try:
        expected = source_methods(3).combine(method, rows, losses.numpy(), seed_index=5)
    except (ValueError, tf.errors.InvalidArgumentError):
        assert not bool(output[-1])
    else:
        expected_valid = np.linalg.norm(expected["combined_gradient"]) > 1.0e-12
        assert bool(output[-1]) == expected_valid
        if expected_valid:
            np.testing.assert_allclose(
                output[1], expected["combined_gradient"], rtol=1.0e-9, atol=1.0e-10
            )
