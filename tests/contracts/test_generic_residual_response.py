"""Manufactured response checks; no SGU kernels or scientific qualification."""

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_residual_response import (
    make_residual_response,
    residual_losses,
)

PARAMETERS = np.array([.4, -.3, .2], dtype=np.float64)
DIRECTION = np.array([.3, -.2, .5], dtype=np.float64)
DATA = np.array([[-.7, .2], [.1, -.4], [.8, .6], [-.2, .9], [1.1, -.3]], dtype=np.float64)
PROBABILITIES = np.array([.05, .25, .1, .2, .4], dtype=np.float64)
SIGNATURE = (tf.TensorSpec([3], tf.float64), tf.TensorSpec([None, 2], tf.float64))


def coupled_residuals(parameters, data):
    state, structural = data[:, 0], data[:, 1]
    current = parameters[0] * state + parameters[1] * structural**2
    current_jet = 2. * parameters[1] * structural
    successor = state + current**2 + structural * current
    successor_jet = (2. * current + structural) * current_jet + current
    future = parameters[2] * successor**2 + parameters[1] * structural
    future_jet = 2. * parameters[2] * successor * successor_jet + parameters[1]
    residual = current + future - state * structural
    structural_jet = current_jet + future_jet - state
    return (tf.stack([residual, current], axis=1) / np.sqrt(6.),
            structural_jet[:, None] / np.sqrt(5.), tf.identity(data))


def analytic_response(parameters, direction, data):
    state, structural = data[:, 0], data[:, 1]
    current = parameters[0] * state + parameters[1] * structural**2
    current_jet = 2. * parameters[1] * structural
    current_response = direction[0] * state + direction[1] * structural**2
    current_mixed = 2. * direction[1] * structural
    successor = state + current**2 + structural * current
    successor_jet = (2. * current + structural) * current_jet + current
    successor_response = (2. * current + structural) * current_response
    successor_mixed = (2. * current_response * current_jet
                       + (2. * current + structural) * current_mixed + current_response)
    future = parameters[2] * successor**2 + parameters[1] * structural
    future_jet = 2. * parameters[2] * successor * successor_jet + parameters[1]
    future_response = (direction[2] * successor**2
                       + 2. * parameters[2] * successor * successor_response + direction[1] * structural)
    future_mixed = (2. * direction[2] * successor * successor_jet
                    + 2. * parameters[2] * (successor_response * successor_jet + successor * successor_mixed)
                    + direction[1])
    values = (np.column_stack([current + future - state * structural, current]) / np.sqrt(6.),
              (current_jet + future_jet - state)[:, None] / np.sqrt(5.), data)
    derivatives = (np.column_stack([current_response + future_response, current_response]) / np.sqrt(6.),
                   (current_mixed + future_mixed)[:, None] / np.sqrt(5.), np.zeros_like(data))
    return values, derivatives


@pytest.fixture(scope="module")
def response():
    return make_residual_response(coupled_residuals, SIGNATURE)


@pytest.mark.parametrize("direction", (DIRECTION, np.array([1., 0., 0.]), np.array([0., 0., 1.])))
def test_moving_successor_and_mixed_structural_response_match_analytic_chain(response, direction):
    actual = response(PARAMETERS, direction, DATA)
    expected = analytic_response(PARAMETERS, direction, DATA)
    for actual_tasks, expected_tasks in zip(actual, expected, strict=True):
        for observed, direct in zip(actual_tasks, expected_tasks, strict=True):
            assert observed.dtype == tf.float64
            np.testing.assert_allclose(observed, direct, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("step", (2e-5, 1e-5))
def test_individual_signed_responses_match_direct_weight_finite_differences(response, step):
    _values, derivatives = response(PARAMETERS, DIRECTION, DATA)
    upper = coupled_residuals(tf.constant(PARAMETERS + step * DIRECTION), tf.constant(DATA))
    lower = coupled_residuals(tf.constant(PARAMETERS - step * DIRECTION), tf.constant(DATA))
    for derivative, upper_task, lower_task in zip(derivatives, upper, lower, strict=True):
        np.testing.assert_allclose(derivative, (upper_task - lower_task) / (2. * step), rtol=3e-6, atol=3e-9)


def test_analytic_structural_jet_matches_direct_structural_finite_difference(response):
    values, derivatives = response(PARAMETERS, DIRECTION, DATA)
    step = 1e-5
    upper_data, lower_data = DATA.copy(), DATA.copy()
    upper_data[:, 1] += step
    lower_data[:, 1] -= step
    upper_values, upper_derivatives = response(PARAMETERS, DIRECTION, upper_data)
    lower_values, lower_derivatives = response(PARAMETERS, DIRECTION, lower_data)
    for expected, upper, lower in ((values[1][:, 0], upper_values[0][:, 0], lower_values[0][:, 0]),
                                   (derivatives[1][:, 0], upper_derivatives[0][:, 0], lower_derivatives[0][:, 0])):
        np.testing.assert_allclose(expected, np.sqrt(6. / 5.) * (upper - lower) / (2. * step),
                                   rtol=3e-6, atol=3e-9)


def test_nonuniform_probabilities_preserve_loss_gradient_identity_and_unequal_chunk_masses(response):
    values, derivatives = response(PARAMETERS, DIRECTION, DATA)
    losses = residual_losses(values, PROBABILITIES)
    expected = np.array([np.einsum("b,bc,bc->", PROBABILITIES, value, value) for value in values])
    directional_losses = np.array([2. * np.einsum("b,bc,bc->", PROBABILITIES, value, derivative)
                                   for value, derivative in zip(values, derivatives, strict=True)])
    np.testing.assert_allclose(losses, expected, rtol=2e-13, atol=2e-13)
    parameters = tf.constant(PARAMETERS)
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(parameters)
        scalar_losses = tf.unstack(residual_losses(coupled_residuals(parameters, tf.constant(DATA)), PROBABILITIES))
    gradients = [tape.gradient(loss, parameters, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                 for loss in scalar_losses]
    del tape
    np.testing.assert_allclose(np.asarray(gradients) @ DIRECTION, directional_losses, rtol=2e-13, atol=2e-13)
    chunk_losses, chunk_responses, masses = [], [], []
    for start, stop in ((0, 2), (2, 5)):
        mass = PROBABILITIES[start:stop].sum()
        probabilities = PROBABILITIES[start:stop] / mass
        chunk_values, chunk_derivatives = response(PARAMETERS, DIRECTION, DATA[start:stop])
        chunk_losses.append(residual_losses(chunk_values, probabilities))
        chunk_responses.append([2. * np.einsum("b,bc,bc->", probabilities, value, derivative)
                                for value, derivative in zip(chunk_values, chunk_derivatives, strict=True)])
        masses.append(mass)
    np.testing.assert_allclose(np.asarray(masses) @ np.asarray(chunk_losses), losses, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(np.asarray(masses) @ np.asarray(chunk_responses), directional_losses,
                               rtol=2e-13, atol=2e-13)
    assert not np.allclose(np.mean(chunk_losses, axis=0), losses)


def test_zero_direction_and_unconnected_residual_have_exact_zero_responses(response):
    values, derivatives = response(PARAMETERS, np.zeros_like(PARAMETERS), DATA)
    for value, derivative in zip(values, derivatives, strict=True):
        np.testing.assert_array_equal(derivative, np.zeros(value.shape))
    values, derivatives = response(PARAMETERS, DIRECTION, DATA)
    np.testing.assert_array_equal(values[-1], DATA)
    np.testing.assert_array_equal(derivatives[-1], np.zeros_like(DATA))


def test_varying_rows_and_parameters_share_one_trace_without_stale_values():
    response = make_residual_response(coupled_residuals, SIGNATURE)
    for count in (1, 5, 2, 4):
        parameters = PARAMETERS + count * .01
        actual = response(parameters, DIRECTION, DATA[:count])
        expected = analytic_response(parameters, DIRECTION, DATA[:count])
        for actual_tasks, expected_tasks in zip(actual, expected, strict=True):
            for observed, direct in zip(actual_tasks, expected_tasks, strict=True):
                assert observed.shape[0] == count
                np.testing.assert_allclose(observed, direct, rtol=2e-13, atol=2e-13)
    assert response.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("signature", ((), ("parameters",), (tf.TensorSpec([3], tf.float32),),
    (tf.TensorSpec([], tf.float64),), (tf.TensorSpec([1, 3], tf.float64),),
    (tf.TensorSpec([None], tf.float64),), (tf.TensorSpec(None, tf.float64),)))
def test_invalid_parameter_signatures_refuse(signature):
    with pytest.raises(ValueError, match="fixed float64 parameter-vector signature"):
        make_residual_response(coupled_residuals, signature)


@pytest.mark.parametrize("field", ("parameters", "direction", "data"))
@pytest.mark.parametrize("nonfinite", (np.nan, np.inf))
def test_nonfinite_parameters_directions_and_data_refuse(response, field, nonfinite):
    inputs = {"parameters": PARAMETERS.copy(), "direction": DIRECTION.copy(), "data": DATA.copy()}
    inputs[field].flat[0] = nonfinite
    with pytest.raises(tf.errors.InvalidArgumentError, match="nonfinite"):
        response(inputs["parameters"], inputs["direction"], inputs["data"])


@pytest.mark.parametrize("kind", ("empty", "tensor", "rank", "dtype", "empty_rows", "empty_components"))
def test_invalid_residual_contract_refuses(kind):
    def invalid(_parameters, data):
        if kind == "empty":
            return ()
        if kind == "tensor":
            return data
        if kind == "rank":
            return (data[:, 0],)
        if kind == "dtype":
            return (tf.cast(data, tf.float32),)
        if kind == "empty_rows":
            return (data[:0],)
        return (data[:, :0],)

    response = make_residual_response(invalid, SIGNATURE)
    with pytest.raises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
        response(PARAMETERS, DIRECTION, DATA)


def test_finite_residual_with_nonfinite_directional_response_refuses():
    def singular(parameters, data):
        return (tf.ones_like(data[:, :1]) * tf.sqrt(parameters[0]),)

    response = make_residual_response(singular, SIGNATURE)
    with pytest.raises(tf.errors.InvalidArgumentError, match="nonfinite directional response"):
        response(np.array([0., .2, .3]), np.array([1., 0., 0.]), DATA)


@pytest.mark.parametrize("probabilities", ([.1, .2, .3, .1, .1], [0., .2, .2, .2, .4],
    [-.1, .2, .2, .2, .5], [np.nan, .2, .2, .2, .2], [np.inf, .2, .2, .2, .2], [.5, .5]))
def test_invalid_outer_probabilities_refuse(probabilities):
    with pytest.raises(tf.errors.InvalidArgumentError):
        residual_losses((tf.constant(DATA),), probabilities)


@pytest.mark.parametrize("residuals", ((), (np.ones(5),), (np.ones((5, 0)),),
    (np.full((5, 1), np.nan),), (np.full((5, 1), np.inf),), (np.full((5, 1), 1e308),)))
def test_invalid_residual_losses_and_overflow_refuse(residuals):
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        residual_losses(residuals, PROBABILITIES)
