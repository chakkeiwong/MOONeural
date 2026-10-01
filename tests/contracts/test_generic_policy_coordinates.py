"""Independent network and finite-difference checks of coordinate changes."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_policy_coordinates import (
    AffinePolicyCoordinates,
    TanhCoordinateMap,
)


def profile():
    return AffinePolicyCoordinates((2., -3., .5), (.02, 5., .7), (1., -.2), (.3, 2.), "a" * 64)


def network(parameters, inputs, width=4):
    offset = 0
    blocks = []
    for size in (3 * width, width, width * width, width, width * 2, 2):
        blocks.append(parameters[offset:offset + size])
        offset += size
    hidden0 = np.tanh(inputs @ blocks[0].reshape(3, width) + blocks[1])
    hidden1 = np.tanh(hidden0 @ blocks[2].reshape(width, width) + blocks[3])
    return hidden1 @ blocks[4].reshape(width, 2) + blocks[5]


def test_roundtrip_values_and_raw_input_chain_rule():
    coordinates = profile()
    mapping = TanhCoordinateMap(coordinates, 4)
    raw = np.random.default_rng(44).normal(size=mapping.parameter_dim) * .1
    changed = mapping.from_raw(raw).numpy()
    np.testing.assert_allclose(mapping.to_raw(changed), raw, rtol=1e-13, atol=1e-14)
    inputs = np.random.default_rng(45).normal(size=(3, 3))
    standardized = (inputs - coordinates.input_center) / coordinates.input_scale
    np.testing.assert_allclose(network(raw, inputs), coordinates.output_center + network(changed, standardized) * coordinates.output_scale, rtol=1e-12, atol=1e-14)
    raw_derivative = np.stack([(network(raw, inputs + 1e-5 * basis) - network(raw, inputs - 1e-5 * basis)) / 2e-5 for basis in np.eye(3)], axis=-1)
    standard_derivative = np.stack([(network(changed, standardized + 1e-4 * basis) - network(changed, standardized - 1e-4 * basis)) / 2e-4 for basis in np.eye(3)], axis=-1)
    expected = standard_derivative * np.asarray(coordinates.output_scale)[None, :, None] / np.asarray(coordinates.input_scale)[None, None, :]
    np.testing.assert_allclose(raw_derivative, expected, rtol=2e-6, atol=2e-9)


def test_gradient_pullback_against_finite_difference_and_adjoint_identity():
    mapping = TanhCoordinateMap(profile(), 4)
    rng = np.random.default_rng(80)
    coordinates = rng.normal(size=mapping.parameter_dim) * .02
    rows = rng.normal(size=(7, mapping.parameter_dim))
    direction = rng.normal(size=mapping.parameter_dim)
    delta = mapping.to_raw(coordinates + 1e-5 * direction) - mapping.to_raw(coordinates - 1e-5 * direction)
    observed = rows @ (delta.numpy() / 2e-5)
    np.testing.assert_allclose(mapping.pullback(rows).numpy() @ direction, observed, rtol=1e-10, atol=1e-8)
    parameters = mapping.to_raw(coordinates).numpy()
    quadratic_gradient = mapping.pullback((2 * parameters)[None]).numpy()[0]
    values = [np.sum(mapping.to_raw(coordinates + sign * 1e-6 * direction).numpy() ** 2) for sign in (1, -1)]
    assert (values[0] - values[1]) / 2e-6 == pytest.approx(quadratic_gradient @ direction, rel=1e-8)
    mapping.pullback(rows[:2])
    assert mapping.pullback.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("name,value", [("input_scale", (0., 1., 1.)), ("input_scale", (-1., 1., 1.)),
                                      ("input_center", (1., 2.)), ("output_center", (float("nan"), 1.)),
                                      ("output_scale", (float("inf"), 1.)), ("training_binding", "")])
def test_invalid_profile_is_refused(name, value):
    with pytest.raises(ValueError):
        replace(profile(), **{name: value})


def test_profile_is_immutable_and_serialized():
    coordinates = profile()
    assert AffinePolicyCoordinates.from_dict(coordinates.to_dict()) == coordinates
    with pytest.raises(FrozenInstanceError):
        coordinates.input_scale = (1., 1., 1.)


@pytest.mark.parametrize("width", (0, -1, True, 1.5))
def test_invalid_network_width_is_refused(width):
    with pytest.raises(ValueError):
        TanhCoordinateMap(profile(), width)


def test_identity_profile_preserves_every_parameter_and_gradient():
    identity = AffinePolicyCoordinates((0.,) * 3, (1.,) * 3, (0.,) * 2, (1.,) * 2, "b" * 64)
    mapping = TanhCoordinateMap(identity, 4)
    parameters = tf.range(mapping.parameter_dim, dtype=tf.float64)
    np.testing.assert_array_equal(mapping.to_raw(parameters), parameters)
    np.testing.assert_array_equal(mapping.from_raw(parameters), parameters)
    np.testing.assert_array_equal(mapping.pullback(parameters[None]), parameters[None])
