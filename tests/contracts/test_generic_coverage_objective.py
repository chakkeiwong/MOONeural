"""Bounded engineering checks for trainable coverage features, not model runs.

Compare raw-coordinate NumPy losses and finite differences of every neural
parameter block, alongside the frozen least-squares objective. Unequal masses,
coordinate scales and reductions expose missing chain factors or extra means.
No structural autodiff, economic reference solve or scientific promotion occurs.
"""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_coverage_initialization import (
    CoverageGroup,
    CoverageTrainingData,
    fit_coverage_warm_start,
)
from mooneural.training.generic_coverage_objective import (
    make_coverage_objective,
    make_coverage_objective_function,
)
from mooneural.training.generic_policy_coordinates import (
    AffinePolicyCoordinates,
    TanhCoordinateMap,
)


def _problem(input_dim=5, output_dim=2, width=4):
    generator = np.random.default_rng(19041)
    center = np.linspace(-0.7, 0.9, input_dim)
    scale = np.geomspace(0.3, 2.0, input_dim)
    groups = []
    for location, count, mass in (("current", 4, 2.0), ("successor", 7, 3.0)):
        groups.append(CoverageGroup(
            group_id=location, location=location, role="training",
            inputs=center + scale * generator.normal(size=(count, input_dim)),
            values=generator.normal(size=(count, output_dim)) + 0.3,
            raw_jacobians=generator.normal(size=(count, output_dim, input_dim)) + 0.1,
            row_weights=np.linspace(0.5, 2, count), mass=mass,
        ))
    data = CoverageTrainingData(tuple(groups))
    profile = AffinePolicyCoordinates(tuple(center), tuple(scale),
                                      tuple(np.linspace(-0.8, 1.1, output_dim)),
                                      tuple(np.geomspace(0.4, 2.2, output_dim)), data.binding_hash())
    mapping = TanhCoordinateMap(profile, width)
    parameters = generator.normal(size=mapping.parameter_dim) * 0.3
    return profile, data, parameters


def test_runner_factory_name_is_the_same_factory():
    assert make_coverage_objective_function is make_coverage_objective


def _numpy_prediction(profile, width, parameters, inputs):
    input_dim, output_dim = len(profile.input_center), len(profile.output_center)
    sizes = (input_dim * width, width, width**2, width, width * output_dim, output_dim)
    first, bias0, second, bias1, last, bias2 = np.split(parameters, np.cumsum(sizes)[:-1])
    first = first.reshape(input_dim, width) / np.asarray(profile.input_scale)[:, None]
    bias0 = bias0 - np.asarray(profile.input_center) @ first
    second = second.reshape(width, width)
    last = last.reshape(width, output_dim) * np.asarray(profile.output_scale)[None, :]
    bias2 = profile.output_center + bias2 * np.asarray(profile.output_scale)
    hidden0 = np.tanh(inputs @ first + bias0)
    hidden1 = np.tanh(hidden0 @ second + bias1)
    jacobians = []
    for coordinate in range(input_dim):
        tangent0 = (1 - hidden0**2) * first[coordinate]
        tangent1 = (tangent0 @ second) * (1 - hidden1**2)
        jacobians.append(tangent1 @ last)
    return hidden1 @ last + bias2, np.stack(jacobians, axis=-1)


def _numpy_losses(profile, width, parameters, data, split, state_reduction, parameter_reduction):
    raw = np.zeros(3)
    total_mass = sum(group.mass for group in data.groups)
    for group in data.groups:
        values, jacobians = _numpy_prediction(profile, width, parameters, group.inputs)
        losses = [np.mean((values - group.values)**2, axis=1)]
        for start, stop, reduction in ((0, split, state_reduction), (split, group.inputs.shape[1], parameter_reduction)):
            squared = (jacobians[:, :, start:stop] - group.raw_jacobians[:, :, start:stop])**2
            reduced = squared.mean(axis=2) if reduction == "mean" else squared.sum(axis=2)
            losses.append(reduced.mean(axis=1))
        raw += group.mass / total_mass * np.array([
            np.average(loss, weights=group.row_weights) for loss in losses
        ])
    return raw


@pytest.mark.parametrize("state_reduction,parameter_reduction", [("mean", "mean"), ("sum", "sum"), ("mean", "sum"), ("sum", "mean")])
def test_loss_and_parameter_block_gradients_match_independent_finite_differences(state_reduction, parameter_reduction, record_property):
    profile, data, parameters = _problem()
    width, split = 4, 2
    denominators = np.array([0.25, 3.0, 0.7])
    mapping = TanhCoordinateMap(profile, width)
    objective = make_coverage_objective(profile, width, data, denominators, split,
                                        state_reduction=state_reduction, parameter_reduction=parameter_reduction)
    raw, normalized, gradients = [value.numpy() for value in objective(parameters)]
    expected = _numpy_losses(profile, width, parameters, data, split, state_reduction, parameter_reduction)
    np.testing.assert_allclose(raw, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(normalized, expected / denominators, rtol=1e-13, atol=1e-14)
    assert raw.shape == normalized.shape == (3,) and gradients.shape == (3, mapping.parameter_dim)
    assert np.all(raw > 0) and np.isfinite(gradients).all()
    generator = np.random.default_rng(911)
    offsets = np.concatenate(([0], np.cumsum(mapping.sizes)))
    directions = []
    for start, stop in pairwise(offsets):
        direction = np.zeros(mapping.parameter_dim)
        direction[start:stop] = generator.normal(size=stop - start)
        directions.append(direction / np.linalg.norm(direction))
        if stop <= sum(mapping.sizes[:4]):
            assert np.all(np.linalg.norm(gradients[:, start:stop], axis=1) > 1e-7)
    directions.append(generator.normal(size=mapping.parameter_dim) / np.sqrt(mapping.parameter_dim))
    max_error = 0.0
    for direction in directions:
        upper = _numpy_losses(profile, width, parameters + 1e-6 * direction, data, split, state_reduction, parameter_reduction)
        lower = _numpy_losses(profile, width, parameters - 1e-6 * direction, data, split, state_reduction, parameter_reduction)
        difference = (upper - lower) / (2e-6 * denominators)
        observed = gradients @ direction
        max_error = max(max_error, float(np.max(np.abs(difference - observed))))
        np.testing.assert_allclose(observed, difference, rtol=2e-5, atol=2e-8)
    np.testing.assert_array_equal(gradients[1:, -len(profile.output_center):], 0)
    unchanged = objective(parameters)
    for actual, expected_array in zip(unchanged, (raw, normalized, gradients)):
        np.testing.assert_array_equal(actual, expected_array)
    assert objective.experimental_get_tracing_count() == 1
    assert not objective.get_concrete_function().variables
    record_property("max_directional_gradient_error", max_error)


@pytest.mark.parametrize("input_dim,output_dim,width,split,reduction", [(2, 1, 3, 1, "mean"), (5, 3, 5, 2, "sum"), (14, 2, 4, 7, "mean")])
def test_objective_matches_frozen_svd_loss_with_arbitrary_dimensions(input_dim, output_dim, width, split, reduction):
    profile, data, initial = _problem(input_dim, output_dim, width)
    denominators = np.array([0.5, 2.0, 0.3])
    fitted = fit_coverage_warm_start(profile, width, initial, data, denominators, split,
                                    state_reduction=reduction, parameter_reduction=reduction)
    assert fitted.valid
    objective = make_coverage_objective(profile, width, data, denominators, split,
                                        state_reduction=reduction, parameter_reduction=reduction)
    raw, normalized, gradients = [value.numpy() for value in objective(fitted.parameters)]
    np.testing.assert_allclose(normalized, fitted.task_losses, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(raw, fitted.task_losses * denominators, rtol=1e-10, atol=1e-11)
    mapping = TanhCoordinateMap(profile, width)
    np.testing.assert_allclose(gradients.sum(axis=0)[sum(mapping.sizes[:4]):], 0, atol=1e-8)
    assert np.linalg.norm(gradients.sum(axis=0)[:sum(mapping.sizes[:4])]) > 1e-7


def test_row_replication_and_group_mass_splitting_preserve_losses_and_gradients():
    profile, data, parameters = _problem()
    group = data.groups[1]
    repeated = replace(group, **{
        name: np.tile(getattr(group, name), (3,) + (1,) * (getattr(group, name).ndim - 1))
        for name in ("inputs", "values", "raw_jacobians", "row_weights")
    })
    candidates = (
        CoverageTrainingData((data.groups[0], repeated)),
        CoverageTrainingData((data.groups[0], replace(group, group_id="future-a", mass=group.mass / 3),
                               replace(group, group_id="future-b", mass=2 * group.mass / 3))),
    )
    expected = make_coverage_objective(profile, 4, data, [1, 2, 3], 2)(parameters)
    for candidate in candidates:
        rebound = replace(profile, training_binding=candidate.binding_hash())
        observed = make_coverage_objective(rebound, 4, candidate, [1, 2, 3], 2)(parameters)
        for actual, reference in zip(observed, expected):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-13)


def test_denominator_snapshot_and_rescaling_preserve_raw_losses():
    profile, data, parameters = _problem()
    scales = np.array([0.5, 2, 3])
    objective = make_coverage_objective(profile, 4, data, scales, 2)
    scales[:] *= 7
    original = [value.numpy() for value in objective(parameters)]
    rescaled = make_coverage_objective(profile, 4, data, scales, 2)(parameters)
    np.testing.assert_array_equal(rescaled[0], original[0])
    np.testing.assert_allclose(rescaled[1], original[1] / 7, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(rescaled[2], original[2] / 7, rtol=1e-13, atol=1e-14)


def test_exact_targets_and_zero_output_layer_have_expected_gradient_connectivity():
    profile, data, parameters = _problem()
    groups = []
    for group in data.groups:
        values, jacobians = _numpy_prediction(profile, 4, parameters, group.inputs)
        groups.append(replace(group, values=values, raw_jacobians=jacobians))
    exact_data = CoverageTrainingData(tuple(groups))
    exact_profile = replace(profile, training_binding=exact_data.binding_hash())
    objective = make_coverage_objective(exact_profile, 4, exact_data, [1, 2, 3], 2)
    raw, normalized, gradients = objective(parameters)
    np.testing.assert_allclose(raw, 0, atol=1e-25)
    np.testing.assert_allclose(normalized, 0, atol=1e-25)
    np.testing.assert_allclose(gradients, 0, atol=1e-12)
    mapping = TanhCoordinateMap(profile, 4)
    initial = parameters.copy()
    start = sum(mapping.sizes[:4])
    initial[start:] = 0
    _, _, initial_gradients = objective(initial)
    np.testing.assert_array_equal(initial_gradients[:, :start], 0)
    assert np.linalg.norm(initial_gradients[:, start:]) > 1e-7


def test_invalid_configuration_bindings_and_nonfinite_parameters_are_rejected():
    profile, data, parameters = _problem()
    arguments = {"profile": profile, "width": 4, "data": data,
                 "denominators": [1, 2, 3], "derivative_split": 2}
    for changed in ({"denominators": [1, 0, 1]}, {"denominators": [1, -2, 1]},
                    {"denominators": [1, np.nan, 1]}, {"denominators": [1, np.inf, 1]},
                    {"denominators": [1, 2]}, {"denominators": ["1", "2", "3"]},
                    {"denominators": [True, True, True]}, {"denominators": [1e308] * 3},
                    {"derivative_split": 0}, {"derivative_split": 5}, {"derivative_split": True},
                    {"state_reduction": "median"}, {"parameter_reduction": "none"}, {"width": 0}):
        with pytest.raises(ValueError):
            make_coverage_objective(**{**arguments, **changed})
    with pytest.raises(TypeError, match="CoverageTrainingData"):
        make_coverage_objective(**{**arguments, "data": data.groups})
    with pytest.raises(ValueError, match="binding"):
        make_coverage_objective(**{**arguments, "profile": replace(profile, training_binding="f" * 64)})
    other_profile, _unused_data, _unused_parameters = _problem(3, 1)
    with pytest.raises(ValueError, match="dimensions"):
        make_coverage_objective(**{**arguments, "profile": other_profile})
    objective = make_coverage_objective(**arguments)
    invalid = parameters.copy()
    invalid[0] = np.nan
    with pytest.raises(tf.errors.InvalidArgumentError, match="nonfinite"):
        objective(invalid)
