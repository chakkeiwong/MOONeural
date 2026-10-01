"""NumPy-only qualification of fixed-population weighted accumulation.

Compare unequal partitions against the unpartitioned empirical objective with
nonunit denominators and analytic/finite-difference derivatives. Check complete
ordered traversal, loss-only dispatch, dimensions, finiteness and invalid
weights. This tests host accumulation, not a model, optimizer or tensor backend.
Run only this file with CPU devices hidden, --confcutdir=tests/contracts to
exclude the repository's TensorFlow autouse fixtures, and at most120 CPU seconds.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from mooneural.training import generic_weighted_objective as weighted

DENOMINATORS = np.array([.3, 7.], dtype=np.float64)


def batch_evaluate(parameters, batch):
    inputs, targets = batch
    residual = inputs @ parameters - targets
    raw = np.array([np.mean(residual**2), np.mean(residual**4)])
    rows = np.stack([2. * residual, 4. * residual**3], axis=1)
    gradients = (rows.T @ inputs) / len(inputs) / DENOMINATORS[:, None]
    return raw, raw / DENOMINATORS, gradients


def batch_values(parameters, batch):
    inputs, targets = batch
    residual = inputs @ parameters - targets
    raw = np.array([np.mean(residual**2), np.mean(residual**4)])
    return raw, raw / DENOMINATORS


def problem():
    inputs = np.array([[1., 2., -.5], [-.3, 4., 1.], [2., .5, -1.],
                       [-2., -1., 1.5], [3., .5, .7], [.8, -1., -.2]])
    targets = np.array([.5, 1., -.5, 1.4, -.7, .3])
    parameters = np.array([.3, -.2, .4])
    batches = [(inputs[:1], targets[:1]), (inputs[1:3], targets[1:3]), (inputs[3:], targets[3:])]
    return parameters, (inputs, targets), batches, np.array([1., 2., 3.]) / 6.


def test_unequal_partitions_match_direct_empirical_objective_and_derivatives():
    parameters, whole, batches, weights = problem()
    callbacks = weighted.make_weighted_objective(batch_evaluate, batch_values, batches, weights)
    actual = callbacks["evaluate"](parameters)
    expected = batch_evaluate(parameters, whole)
    for observed, direct in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, direct, rtol=5e-15, atol=1e-15)
        assert observed.dtype == np.float64
    np.testing.assert_allclose(actual[1], actual[0] / DENOMINATORS, rtol=5e-15, atol=0.)
    for observed, direct in zip(callbacks["losses"](parameters), expected[:2], strict=True):
        np.testing.assert_allclose(observed, direct, rtol=5e-15, atol=1e-15)
    derivative = np.empty_like(actual[2])
    for index in range(len(parameters)):
        displacement = np.zeros_like(parameters)
        displacement[index] = 1e-6
        derivative[:, index] = (callbacks["losses"](parameters + displacement)[1]
                                - callbacks["losses"](parameters - displacement)[1]) / 2e-6
    np.testing.assert_allclose(actual[2], derivative, rtol=2e-9, atol=1e-9)
    unweighted = np.mean([batch_values(parameters, batch)[0] for batch in batches], axis=0)
    assert not np.allclose(unweighted, expected[0], rtol=1e-3, atol=1e-3)


def test_factory_does_not_evaluate_and_each_callback_visits_all_batches_in_order():
    parameters, _whole, batches, weights = problem()
    calls = []

    def evaluate(policy, batch):
        calls.append(("evaluate", batch, policy.copy()))
        return batch_evaluate(policy, batches[batch])

    def values(policy, batch):
        calls.append(("values", batch, policy.copy()))
        return batch_values(policy, batches[batch])

    callbacks = weighted.make_weighted_objective(evaluate, values, range(3), weights)
    assert calls == []
    callbacks["losses"](parameters)
    callbacks["evaluate"](parameters)
    callbacks["evaluate"](parameters)
    assert [(kind, batch) for kind, batch, _policy in calls] == [
        (kind, batch) for kind in ("values", "evaluate", "evaluate") for batch in range(3)]
    for _kind, _batch, policy in calls:
        np.testing.assert_array_equal(policy, parameters)


def test_weights_and_batch_order_are_snapshots_and_results_have_no_running_state():
    parameters, whole, batches, weights = problem()
    callbacks = weighted.make_weighted_objective(batch_evaluate, batch_values, batches, weights)
    weights[:] = 0.
    batches.reverse()
    first = callbacks["evaluate"](parameters)
    first[0][:] = 99.
    second = callbacks["evaluate"](parameters)
    for observed, direct in zip(second, batch_evaluate(parameters, whole), strict=True):
        np.testing.assert_allclose(observed, direct, rtol=5e-15, atol=1e-15)


def test_returned_function_source_is_the_new_generic_module():
    _parameters, _whole, batches, weights = problem()
    callbacks = weighted.make_weighted_objective(batch_evaluate, batch_values, batches, weights)
    assert set(callbacks) == {"evaluate", "losses"}
    for callback in callbacks.values():
        assert Path(inspect.getsourcefile(callback)).resolve() == Path(weighted.__file__).resolve()


@pytest.mark.parametrize("weights", ([.2, .2, .2], [0., .5, .5], [-.1, .5, .6],
                                    [np.nan, .5, .5], [np.inf, .5, .5],
                                    [1e308, 1e308, 1e308], [True, False, False],
                                    [[.1, .2, .7]], [.5, .5], [.1j, .2, .7]))
def test_invalid_weights_refuse_without_calling_callbacks(weights):
    def forbidden(*_args):
        raise AssertionError("construction called a batch callback")

    with pytest.raises(ValueError):
        weighted.make_weighted_objective(forbidden, forbidden, (0, 1, 2), weights)


def test_empty_population_and_noncallable_callbacks_refuse():
    with pytest.raises(ValueError):
        weighted.make_weighted_objective(batch_evaluate, batch_values, (), [])
    with pytest.raises(TypeError):
        weighted.make_weighted_objective(None, batch_values, (0,), [1.])
    with pytest.raises(TypeError):
        weighted.make_weighted_objective(batch_evaluate, None, (0,), [1.])


@pytest.mark.parametrize("parameters", ([], [[1., 2.]], [np.nan, 2.], [1., np.inf],
                                      np.array([1., 2.], dtype=np.float32), np.array([1, 2])))
def test_invalid_parameters_refuse_before_batch_evaluation(parameters):
    def forbidden(*_args):
        raise AssertionError("invalid parameters reached callback")

    callbacks = weighted.make_weighted_objective(forbidden, forbidden, (0,), [1.])
    with pytest.raises(ValueError):
        callbacks["evaluate"](parameters)


@pytest.mark.parametrize("kind", ("arity", "raw_rank", "normalized_rank", "task_count", "empty", "gradient_rank",
                                 "gradient_tasks", "gradient_parameters", "raw_nan", "normalized_inf", "gradient_nan",
                                 "float32", "array_required", "later_task_change"))
def test_invalid_batch_outputs_refuse(kind):
    parameters, _whole, batches, weights = problem()

    def invalid(policy, batch):
        raw, normalized, gradients = batch_evaluate(policy, batches[batch])
        if kind == "arity":
            return raw, normalized
        if kind == "raw_rank":
            return raw[None, :], normalized, gradients
        if kind == "normalized_rank":
            return raw, normalized[None, :], gradients
        if kind == "task_count":
            return raw, normalized[:1], gradients
        if kind == "empty":
            return raw[:0], normalized[:0], gradients[:0]
        if kind == "gradient_rank":
            return raw, normalized, gradients[None, :, :]
        if kind == "gradient_tasks":
            return raw, normalized, gradients[:1]
        if kind == "gradient_parameters":
            return raw, normalized, gradients[:, :1]
        if kind == "raw_nan":
            return raw * np.nan, normalized, gradients
        if kind == "normalized_inf":
            return raw, normalized * np.inf, gradients
        if kind == "gradient_nan":
            return raw, normalized, gradients * np.nan
        if kind == "float32":
            return raw, normalized, gradients.astype(np.float32)
        if kind == "array_required":
            return raw.tolist(), normalized, gradients
        return (raw, normalized, gradients) if batch == 0 else (raw[:1], normalized[:1], gradients[:1])

    callbacks = weighted.make_weighted_objective(invalid, batch_values, range(3), weights)
    with pytest.raises(ValueError):
        callbacks["evaluate"](parameters)


def test_loss_only_shape_validation_and_shared_signature():
    parameters, _whole, batches, weights = problem()

    def values(policy, batch):
        return tuple(value[:1] for value in batch_values(policy, batch))

    callbacks = weighted.make_weighted_objective(batch_evaluate, values, batches, weights)
    callbacks["losses"](parameters)
    with pytest.raises(ValueError, match="task counts"):
        callbacks["evaluate"](parameters)
    with pytest.raises(ValueError, match="parameter count changed"):
        callbacks["losses"](parameters[:1])


@pytest.mark.parametrize("kind", ("arity", "rank", "nonfinite"))
def test_invalid_loss_only_output_refuses(kind):
    parameters, _whole, batches, weights = problem()

    def invalid(policy, batch):
        raw, normalized = batch_values(policy, batch)
        if kind == "arity":
            return raw, normalized, raw
        if kind == "rank":
            return raw, normalized[:, None]
        return raw, normalized * np.nan

    callbacks = weighted.make_weighted_objective(batch_evaluate, invalid, batches, weights)
    with pytest.raises(ValueError):
        callbacks["losses"](parameters)


def test_original_callback_error_propagates_and_traversal_stops():
    error = RuntimeError("manufactured batch failure")
    calls = []

    def broken(_parameters, batch):
        calls.append(batch)
        raise error

    callbacks = weighted.make_weighted_objective(broken, broken, (0, 1), [.4, .6])
    with pytest.raises(RuntimeError) as observed:
        callbacks["evaluate"](np.array([1.]))
    assert observed.value is error
    assert calls == [0]


def test_finite_inputs_cannot_return_nonfinite_accumulation():
    maximum = np.finfo(np.float64).max

    def large(parameters, _batch):
        raw = np.array([maximum])
        return raw, raw.copy(), np.full((1, len(parameters)), maximum)

    callbacks = weighted.make_weighted_objective(large, batch_values, (0, 1), [.5, .5 + 5e-13])
    with pytest.raises(ValueError, match="weighted accumulation must be finite"):
        callbacks["evaluate"](np.array([1.]))
