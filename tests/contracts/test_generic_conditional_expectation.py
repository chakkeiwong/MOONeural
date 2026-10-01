"""Engineering checks for the shared expectation repair, R1.

Parent record: docs/plans/generic-neural-solver-expectation-coverage-repair-2026-09-18.md.
Pre-run audit: exact moments/enumeration and identical float64 tensors are the
comparators; bad axes, weights and bindings must fail. Stochastic moment checks
use loose multi-standard-error tolerances, not performance rankings. No training,
model promotion, independent-stream proof or quadrature error bound is inferred.
Worker cap 120s, CPU intentionally selected, BLAS/OpenMP/TensorFlow threads2.
F4 review adds full recursive-utility gamma derivatives against a lognormal
closed form and fixed-node finite differences, plus exact independent sign-cell
enumeration showing nonlinear plug-in bias survives an independent product.
"""

import itertools
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_conditional_expectation import (
    GaussianExpectationInputs,
    GaussianExpectationProfile,
    antithetic_from_iid,
    conditional_mean,
    independent_signed_audit,
    prepare_gaussian_inputs,
    reduce_conditional_losses,
)


def prepared(method="iid", *, rows=2, shock_dim=3, draws=4, order=None,
             seed=103, role="training", stream=0):
    profile = GaussianExpectationProfile(method, shock_dim, draws, order)
    return prepare_gaussian_inputs(profile, rows=rows, seed=seed, role=role,
                                   stream=stream, max_points=65536)


def loss_inputs():
    generator = np.random.default_rng(409)
    return {
        "integrand": generator.normal(size=(2, 6, 3)),
        "jacobian": generator.normal(size=(2, 6, 3, 5)),
        "projection": generator.normal(size=(2, 3, 4, 5)),
        "denominators": np.array([.03, 3., .5, 11., 1., 7.]),
    }


def test_profile_json_roundtrip_freeze_and_all_fields_bound():
    for profile in (GaussianExpectationProfile("iid", 5, 8),
                    GaussianExpectationProfile("antithetic", 5, 8),
                    GaussianExpectationProfile("gauss_hermite", 5, order=3)):
        encoded = json.loads(json.dumps(profile.to_dict(), allow_nan=False))
        restored = GaussianExpectationProfile.from_dict(encoded)
        assert restored == profile
        assert restored.binding_hash() == profile.binding_hash()
        assert len(profile.binding_hash()) == 64
        with pytest.raises(FrozenInstanceError):
            profile.shock_dim = 9
        assert replace(profile, shock_dim=7).binding_hash() != profile.binding_hash()
    assert GaussianExpectationProfile("iid", 5, 8).binding_hash() != GaussianExpectationProfile("antithetic", 5, 8).binding_hash()
    assert GaussianExpectationProfile("iid", 5, 8).binding_hash() != GaussianExpectationProfile("iid", 5, 10).binding_hash()
    assert GaussianExpectationProfile("gauss_hermite", 5, order=3).binding_hash() != GaussianExpectationProfile("gauss_hermite", 5, order=4).binding_hash()


def test_invalid_profiles_and_json_are_refused():
    for kwargs in ({"method": "sobol"}, {"method": []}, {"shock_dim": 0},
                   {"shock_dim": True}, {"shock_dim": 1.5}, {"draws": 0},
                   {"draws": True}, {"draws": 3.5}, {"draws": None},
                   {"order": 3}, {"method": "antithetic", "draws": 3},
                   {"method": "gauss_hermite"}):
        with pytest.raises(ValueError):
            GaussianExpectationProfile(**({"method": "iid", "shock_dim": 2, "draws": 4} | kwargs))
    for order in (0, True, 1.5, None):
        with pytest.raises(ValueError):
            GaussianExpectationProfile("gauss_hermite", 2, order=order)
    payload = GaussianExpectationProfile("iid", 2, 4).to_dict()
    for key, value in (("schema", "unknown"), ("distribution", "uniform"),
                       ("dtype", "float32"), ("unexpected", True)):
        with pytest.raises(ValueError):
            GaussianExpectationProfile.from_dict(payload | {key: value})
    with pytest.raises(ValueError):
        GaussianExpectationProfile.from_dict({key: value for key, value in payload.items() if key != "draws"})


def test_roles_seeds_streams_are_repeatable_and_distinct_without_global_rng():
    np.random.seed(76)
    before = np.random.get_state()
    original = prepared()
    repeated = prepared()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(original.nodes, repeated.nodes)
    assert original.binding_hash() == repeated.binding_hash()
    for changed in (prepared(role="validation"), prepared(seed=104), prepared(stream=1)):
        assert changed.profile_fingerprint == original.profile_fingerprint
        assert changed.stream_key != original.stream_key
        assert changed.binding_hash() != original.binding_hash()
        assert not np.array_equal(changed.nodes, original.nodes)
    original.require_profile(original.profile)
    with pytest.raises(ValueError, match="profile mismatch"):
        original.require_profile(replace(original.profile, draws=6))
    with pytest.raises(ValueError):
        original.nodes[0, 0, 0] = 0
    with pytest.raises(ValueError):
        original.nodes.setflags(write=True)
    with pytest.raises(ValueError):
        original.weights.setflags(write=True)


def test_preparation_rejects_bad_identifiers_or_budget_before_allocation():
    profile = GaussianExpectationProfile("iid", 2, 4)
    defaults = {"rows": 2, "seed": 3, "role": "training", "max_points": 4}
    for kwargs in ({"rows": 0}, {"rows": True}, {"seed": -1}, {"seed": True},
                   {"role": "  "}, {"role": 3}, {"stream": -1},
                   {"stream": 1.5}, {"max_points": 3}, {"max_points": True}):
        with pytest.raises(ValueError):
            prepare_gaussian_inputs(profile, **(defaults | kwargs))
    with pytest.raises(ValueError, match="before allocation"):
        prepare_gaussian_inputs(GaussianExpectationProfile("gauss_hermite", 20, order=10), **defaults)


def test_prepared_inputs_refuse_malformed_weights_and_nodes_and_copy_arrays():
    inputs = prepared()
    assert isinstance(inputs, GaussianExpectationInputs)
    for weights in ([.2] * 4, [0., .25, .25, .5], [-.25, .25, .5, .5],
                    [np.nan, .25, .25, .25], [np.inf, .25, .25, .25], [[.25] * 4], [.5, .5]):
        with pytest.raises(ValueError):
            replace(inputs, weights=weights)
    for nodes in (np.zeros((2, 4)), np.zeros((1, 4, 2)), np.zeros((0, 4, 3)),
                  np.full((2, 4, 3), np.nan), np.full((2, 4, 3), 1j)):
        with pytest.raises(ValueError):
            replace(inputs, nodes=nodes)
    nodes = np.array(inputs.nodes)
    weights = np.array(inputs.weights)
    copied = replace(inputs, nodes=nodes, weights=weights)
    assert copied.binding_hash() == inputs.binding_hash()
    nodes[:] = 100.
    weights[:] = 1.
    np.testing.assert_array_equal(copied.nodes, inputs.nodes)
    np.testing.assert_array_equal(copied.weights, inputs.weights)
    changed_nodes = replace(inputs, nodes=nodes)
    changed_weights = replace(inputs, weights=[.1, .2, .3, .4])
    assert changed_nodes.stream_key == changed_weights.stream_key == inputs.stream_key
    assert changed_nodes.binding_hash() != inputs.binding_hash()
    assert changed_weights.binding_hash() != inputs.binding_hash()


@pytest.mark.parametrize("shock_dim", [1, 3, 5])
def test_gauss_hermite_known_moments_arbitrary_dimension(shock_dim):
    inputs = prepared("gauss_hermite", rows=2, shock_dim=shock_dim, draws=None, order=3)
    nodes = inputs.nodes
    np.testing.assert_allclose(conditional_mean(nodes, inputs.weights), 0., atol=2e-15)
    np.testing.assert_allclose(conditional_mean(nodes**2, inputs.weights), 1., atol=2e-15)
    np.testing.assert_allclose(conditional_mean(nodes**4, inputs.weights), 3., atol=5e-15)
    products = nodes[:, :, :, None] * nodes[:, :, None, :]
    moments = conditional_mean(products.reshape(2, -1, shock_dim**2), inputs.weights)
    np.testing.assert_allclose(np.asarray(moments).reshape(2, shock_dim, shock_dim),
                               np.broadcast_to(np.eye(shock_dim), (2, shock_dim, shock_dim)), atol=3e-15)
    distinct_role = prepared("gauss_hermite", rows=2, shock_dim=shock_dim, draws=None,
                             order=3, role="validation")
    np.testing.assert_array_equal(inputs.nodes, distinct_role.nodes)


def test_iid_coordinates_have_standard_gaussian_moments():
    inputs = prepared(rows=1, shock_dim=4, draws=32768)
    nodes = inputs.nodes[0]
    np.testing.assert_allclose(nodes.mean(axis=0), 0., atol=.04)
    np.testing.assert_allclose(nodes.T @ nodes / len(nodes), np.eye(4), atol=.05)
    assert inputs.weights.dtype == inputs.nodes.dtype == np.float64


def test_antithetic_pairs_preserve_marginals_but_are_not_independent():
    inputs = prepared("antithetic", rows=16384, shock_dim=2, draws=8)
    base = inputs.nodes[:, :4]
    opposite = inputs.nodes[:, 4:]
    np.testing.assert_array_equal(base, -opposite)
    np.testing.assert_allclose(conditional_mean(inputs.nodes, inputs.weights), 0., atol=5e-16)
    np.testing.assert_allclose((base * opposite).mean(axis=(0, 1)), -1., atol=.04)
    squared_mean = np.asarray(conditional_mean(inputs.nodes**2, inputs.weights))
    np.testing.assert_allclose(squared_mean, (base**2).mean(axis=1), atol=1e-15)
    np.testing.assert_allclose(squared_mean.mean(axis=0), 1., atol=.03)
    np.testing.assert_allclose(squared_mean.var(axis=0), 2. / 4., atol=.05)
    assert np.all(squared_mean.var(axis=0) > 2. / 8. + .15)


def test_retained_iid_transformation_is_exact_and_does_not_mutate_or_use_rng():
    base = prepared(rows=3, shock_dim=5, draws=24).nodes
    untouched = base.copy()
    before = np.random.get_state()
    paired = antithetic_from_iid(base)
    after = np.random.get_state()
    np.testing.assert_array_equal(paired[:, :12], base[:, :12])
    np.testing.assert_array_equal(paired[:, 12:], -base[:, :12])
    np.testing.assert_array_equal(base, untouched)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert not paired.flags.writeable
    for bad in (np.zeros((2, 3, 1)), np.zeros((2, 4)), np.zeros((0, 4, 2)),
                np.zeros((2, 4, 0)), np.full((2, 4, 1), np.nan)):
        with pytest.raises(ValueError):
            antithetic_from_iid(bad)


def test_square_of_mean_is_not_mean_square_and_has_exact_iid_variance_term():
    innovations = np.array(list(itertools.product((-1., 1.), repeat=2)))[:, :, None]
    mean, scale = 2., 3.
    integrand = mean + scale * innovations
    losses = reduce_conditional_losses(integrand, np.ones((4, 2, 1, 1)),
                                        np.ones((4, 1, 1, 1)), [2., 4.])
    assert np.asarray(losses.raw)[:, 0].mean() == mean**2 + scale**2 / 2
    assert (integrand**2).mean() == mean**2 + scale**2
    np.testing.assert_array_equal(losses.normalized, np.asarray(losses.raw) / [2., 4.])
    cancel = reduce_conditional_losses(np.array([[[-1.], [1.]]]), np.ones((1, 2, 1, 1)),
                                        np.ones((1, 1, 1, 1)), [1., 1.])
    assert float(cancel.raw[0, 0]) == 0.


def test_uniform_reduction_matches_existing_float64_operation_order_exactly():
    inputs = loss_inputs()
    values = tf.constant(inputs["integrand"], tf.float64)
    jacobian = tf.constant(inputs["jacobian"], tf.float64)
    projection = tf.constant(inputs["projection"], tf.float64)
    tasks = []
    for equation in range(3):
        residual_mean = tf.reduce_mean(values[:, :, equation], axis=1)
        derivative_mean = tf.reduce_mean(jacobian[:, :, equation, :], axis=1)
        projected = tf.einsum("bpd,bd->bp", projection[:, equation], derivative_mean)
        tasks.extend([tf.square(residual_mean), tf.reduce_mean(tf.square(projected), axis=1)])
    expected = tf.stack(tasks, axis=1)
    for weights in (None, np.full(6, 1. / 6.)):
        observed = reduce_conditional_losses(**inputs, weights=weights)
        np.testing.assert_array_equal(observed.raw, expected)
        np.testing.assert_array_equal(observed.normalized, expected / inputs["denominators"])
    np.testing.assert_array_equal(conditional_mean(values), tf.reduce_mean(values, axis=1))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_weighted_interleaved_losses_against_explicit_scalar_calculation(reduction):
    inputs = loss_inputs()
    weights = np.array([.05, .1, .15, .2, .2, .3])
    expected = np.zeros((2, 6))
    for row in range(2):
        for equation in range(3):
            mean = sum(weights[draw] * inputs["integrand"][row, draw, equation] for draw in range(6))
            derivative = np.array([sum(weights[draw] * inputs["jacobian"][row, draw, equation, parameter]
                                       for draw in range(6)) for parameter in range(5)])
            projected = inputs["projection"][row, equation] @ derivative
            expected[row, 2 * equation] = mean**2
            expected[row, 2 * equation + 1] = np.sum(projected**2) / (4 if reduction == "mean" else 1)
    observed = reduce_conditional_losses(**inputs, weights=weights, derivative_reduction=reduction)
    np.testing.assert_allclose(observed.raw, expected, rtol=5e-14, atol=2e-15)
    np.testing.assert_allclose(observed.normalized, expected / inputs["denominators"], rtol=5e-14, atol=2e-15)


def test_nonlinear_recursive_utility_expectation_placement():
    inputs = prepared("gauss_hermite", rows=2, shock_dim=2, draws=None, order=12)
    location = np.array([.2, -.1])
    loading = np.array([[.2, .1], [.1, -.25]])
    continuation = np.exp(location[:, None] + np.sum(loading[:, None] * inputs.nodes, axis=2))
    power = -2.
    integrated = np.asarray(conditional_mean((continuation**power)[:, :, None], inputs.weights))[:, 0]
    analytic = np.exp(power * location + .5 * power**2 * np.sum(loading**2, axis=1))
    np.testing.assert_allclose(integrated, analytic, rtol=2e-14)
    wrong = np.asarray(conditional_mean(continuation[:, :, None], inputs.weights))[:, 0]**power
    assert np.all(np.abs(wrong - integrated) > .05)
    np.testing.assert_allclose(integrated**(1. / power), analytic**(1. / power), rtol=2e-14)


def test_complete_recursive_utility_gamma_derivative_including_continuation_and_outer_power():
    """Check U=[(1-beta)C**rho+beta(E[V**(1-gamma)])**(rho/(1-gamma))]**(1/rho)."""

    inputs = prepared("gauss_hermite", rows=2, shock_dim=2, draws=None, order=12)
    location_base = np.array([.15, -.08])
    location_slope = np.array([.025, -.02])
    loading_base = np.array([[.14, .03], [.05, .18]])
    loading_slope = np.array([[.008, -.005], [-.01, .012]])
    consumption = np.array([.9, 1.2])
    discount = .94

    def evaluate(gamma, elasticity):
        risk_power = 1. - gamma
        recursion_power = 1. - 1. / elasticity
        location = location_base + gamma * location_slope
        loading = loading_base + gamma * loading_slope
        log_continuation = location[:, None] + np.sum(loading[:, None, :] * inputs.nodes, axis=2)
        log_continuation_gamma = location_slope[:, None] + np.sum(loading_slope[:, None, :] * inputs.nodes, axis=2)
        integrand = np.exp(risk_power * log_continuation)
        direct_gamma = -log_continuation * integrand
        through_continuation = risk_power * log_continuation_gamma * integrand
        integrated = np.asarray(conditional_mean(np.stack((integrand, direct_gamma, through_continuation), axis=2), inputs.weights))
        moment, direct_mean, through_mean = integrated.T
        outer_power = recursion_power / risk_power
        outer_power_gamma = recursion_power / risk_power**2
        future_term = moment**outer_power
        aggregate = (1. - discount) * consumption**recursion_power + discount * future_term
        utility = aggregate**(1. / recursion_power)
        outer_factor = utility * discount * future_term / (recursion_power * aggregate)
        moment_contribution = outer_power * (direct_mean + through_mean) / moment
        exponent_contribution = outer_power_gamma * np.log(moment)
        derivative = outer_factor * (moment_contribution + exponent_contribution)
        incomplete = (
            outer_factor * (outer_power * through_mean / moment + exponent_contribution),
            outer_factor * (outer_power * direct_mean / moment + exponent_contribution),
            outer_factor * moment_contribution,
        )
        return utility, derivative, moment, direct_mean + through_mean, incomplete

    for gamma, elasticity in ((.4, .8), (2.6, 1.7), (4.2, 1.7)):
        observed, derivative, moment, moment_gamma, incomplete = evaluate(gamma, elasticity)
        risk_power = 1. - gamma
        recursion_power = 1. - 1. / elasticity
        location = location_base + gamma * location_slope
        loading = loading_base + gamma * loading_slope
        variance = np.sum(loading**2, axis=1)
        covariance_slope = np.sum(loading * loading_slope, axis=1)
        analytic_moment = np.exp(risk_power * location + .5 * risk_power**2 * variance)
        analytic_moment_gamma = analytic_moment * (-location + risk_power * location_slope
                                                   - risk_power * variance + risk_power**2 * covariance_slope)
        log_certainty_equivalent = location + .5 * risk_power * variance
        log_certainty_gamma = location_slope - .5 * variance + risk_power * covariance_slope
        future_term = np.exp(recursion_power * log_certainty_equivalent)
        aggregate = (1. - discount) * consumption**recursion_power + discount * future_term
        analytic_utility = aggregate**(1. / recursion_power)
        analytic_derivative = analytic_utility * discount * future_term * log_certainty_gamma / aggregate
        np.testing.assert_allclose(moment, analytic_moment, rtol=3e-13, atol=1e-15)
        np.testing.assert_allclose(moment_gamma, analytic_moment_gamma, rtol=3e-13, atol=1e-15)
        np.testing.assert_allclose(observed, analytic_utility, rtol=3e-13)
        np.testing.assert_allclose(derivative, analytic_derivative, rtol=3e-12, atol=2e-15)
        step = 1e-5
        difference = (evaluate(gamma + step, elasticity)[0] - evaluate(gamma - step, elasticity)[0]) / (2 * step)
        np.testing.assert_allclose(derivative, difference, rtol=3e-8, atol=2e-11)
        for missing_term in incomplete:
            assert np.max(np.abs(missing_term - analytic_derivative)) > 1e-4


def test_independent_product_does_not_debias_nonlinear_recursive_utility_residual():
    """Exactly enumerate four equiprobable cells of two independent Gaussian signs."""

    signs = np.array(list(itertools.product((-1., 1.), repeat=2)))
    continuation = np.where(signs < 0., 1., 4.)
    risk_power, recursion_power, discount = -1., .5, .8
    exact_moment = .5 * (1.**risk_power + 4.**risk_power)

    def recurse(moment):
        return ((1. - discount) + discount * moment**(recursion_power / risk_power))**(1. / recursion_power)

    exact_utility = recurse(exact_moment)
    estimated_moments = continuation**risk_power
    np.testing.assert_array_equal(estimated_moments.mean(axis=0), [exact_moment, exact_moment])
    residuals = recurse(estimated_moments) - exact_utility
    bias = .5 * (recurse(1.**risk_power) + recurse(4.**risk_power)) - exact_utility
    assert bias > .5
    left_inputs = replace(prepared(rows=4, shock_dim=1, draws=1, role="audit_left"),
                          nodes=signs[:, 0, None, None])
    right_inputs = replace(prepared(rows=4, shock_dim=1, draws=1, role="audit_right"),
                           nodes=signs[:, 1, None, None])
    diagnostic = independent_signed_audit(
        residuals[:, 0, None, None], np.zeros((4, 1, 1, 1)),
        residuals[:, 1, None, None], np.zeros((4, 1, 1, 1)),
        np.ones((4, 1, 1, 1)), [1., 1.], left_inputs=left_inputs, right_inputs=right_inputs,
    )
    independent_product = float(np.asarray(diagnostic.raw)[:, 0].mean())
    assert independent_product == pytest.approx(bias**2, rel=2e-14)
    assert independent_product > .25
    assert np.any(np.asarray(diagnostic.raw)[:, 0] < 0.)


def test_state_dependent_volatility_and_lagged_input_total_derivatives():
    inputs = prepared("gauss_hermite", rows=2, shock_dim=1, draws=None, order=3)
    state = np.array([.4, -.2])
    lagged = np.array([.7, -.5])
    slope, persistence, log_scale, loading = .8, .3, -.4, .6

    def evaluate(current):
        location = slope * current + persistence * lagged
        scale = np.exp(log_scale + loading * current)
        noise = scale[:, None] * inputs.nodes[:, :, 0]
        successor = location[:, None] + noise
        tangent = np.stack((np.broadcast_to(current[:, None], successor.shape),
                            np.broadcast_to(lagged[:, None], successor.shape),
                            noise, current[:, None] * noise), axis=-1)
        return successor**2, 2 * successor[:, :, None] * tangent

    integrand, jacobian = evaluate(state)
    location = slope * state + persistence * lagged
    variance = np.exp(2 * (log_scale + loading * state))
    expected_mean = location**2 + variance
    expected_jacobian = np.stack((2 * location * state, 2 * location * lagged,
                                 2 * variance, 2 * state * variance), axis=-1)
    means = conditional_mean(jacobian, inputs.weights)
    np.testing.assert_allclose(means, expected_jacobian, rtol=1e-14, atol=1e-15)
    projection = np.broadcast_to(np.eye(4), (2, 1, 4, 4))
    observed = reduce_conditional_losses(integrand[:, :, None], jacobian[:, :, None, :],
                                          projection, [2., 5.], weights=inputs.weights)
    expected = np.stack((expected_mean**2, np.mean(expected_jacobian**2, axis=1)), axis=1)
    np.testing.assert_allclose(observed.raw, expected, rtol=1e-14)
    step = 1e-5
    plus = conditional_mean(evaluate(state + step)[1], inputs.weights)
    minus = conditional_mean(evaluate(state - step)[1], inputs.weights)
    mixed = np.asarray((plus - minus) / (2 * step))[:, 2]
    np.testing.assert_allclose(mixed, 4 * loading * variance, rtol=1e-9)
    assert np.all(expected_jacobian[:, 2] > .5)


def test_signed_audit_preserves_negative_values_and_uses_product_of_means():
    left_inputs = prepared(rows=2, shock_dim=2, draws=2, role="audit_left")
    right_inputs = prepared(rows=2, shock_dim=2, draws=4, role="audit_right")
    left = np.array([[[-4.], [2.]], [[-2.], [-2.]]])
    right = np.array([[[1.], [5.], [1.], [5.]], [[3.], [3.], [3.], [3.]]])
    left_jacobian = np.tile([1., 2., 3.], (2, 2, 1, 1))
    right_jacobian = np.tile([-1., -2., -3.], (2, 4, 1, 1))
    projection = np.broadcast_to(np.eye(3), (2, 1, 3, 3))
    observed = independent_signed_audit(left, left_jacobian, right, right_jacobian,
                                        projection, [2., 7.], left_inputs=left_inputs,
                                        right_inputs=right_inputs)
    np.testing.assert_allclose(observed.raw, [[-3., -14. / 3.], [-6., -14. / 3.]])
    np.testing.assert_allclose(observed.normalized, np.asarray(observed.raw) / [2., 7.])
    assert observed.diagnostic_only


def test_independent_cross_means_remove_variance_in_finite_enumeration():
    combinations = np.array(list(itertools.product((-1., 1.), repeat=4)))
    left = 2. + 3. * combinations[:, :2, None]
    right = 2. + 3. * combinations[:, 2:, None]
    left_inputs = prepared(rows=16, shock_dim=1, draws=2, role="audit_left")
    right_inputs = prepared(rows=16, shock_dim=1, draws=2, role="audit_right")
    observed = independent_signed_audit(left, left[..., None], right, right[..., None],
                                        np.ones((16, 1, 1, 1)), [1., 1.],
                                        left_inputs=left_inputs, right_inputs=right_inputs)
    np.testing.assert_array_equal(np.asarray(observed.raw).mean(axis=0), [4., 4.])
    assert np.any(np.asarray(observed.raw) < 0.)


def test_signed_audit_refuses_reused_streams_quadrature_or_mismatched_batches():
    left = prepared(shock_dim=2)
    distinct = prepared(shock_dim=2, role="audit_right")

    def audit(right):
        return independent_signed_audit(np.ones((2, 4, 1)), np.ones((2, 4, 1, 2)),
                                        np.ones((2, 4, 1)), np.ones((2, 4, 1, 2)),
                                        np.ones((2, 1, 2, 2)), [1., 1.],
                                        left_inputs=left, right_inputs=right)

    audit(distinct)
    for bad in (left, prepared("antithetic", shock_dim=2), prepared(shock_dim=3, role="right"),
                prepared("gauss_hermite", shock_dim=2, draws=None, order=2),
                prepared(rows=1, shock_dim=2, role="right")):
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            audit(bad)


def test_tensor_reductions_refuse_invalid_weights_denominators_axes_and_nonfinite_values():
    inputs = loss_inputs()
    for weights in ([1. / 5.] * 5, [1.] * 6, [0., .2, .2, .2, .2, .2],
                    [-.1, .1, .2, .2, .3, .3], [np.nan] * 6, [np.inf] * 6,
                    np.ones((2, 6)) / 6):
        with pytest.raises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
            reduce_conditional_losses(**inputs, weights=weights)
    for denominators in (np.ones(5), np.zeros(6), -np.ones(6),
                         np.full(6, np.nan), np.full(6, np.inf), np.ones((2, 6))):
        with pytest.raises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
            reduce_conditional_losses(**(inputs | {"denominators": denominators}))
    for key, value in (("integrand", np.zeros((2, 0, 3))),
                       ("integrand", np.zeros((0, 6, 3))),
                       ("integrand", np.zeros((2, 6, 0))),
                       ("integrand", np.full((2, 6, 3), np.nan)),
                       ("jacobian", np.zeros((1, 6, 3, 5))),
                       ("jacobian", np.zeros((2, 5, 3, 5))),
                       ("jacobian", np.full((2, 6, 3, 5), np.inf)),
                       ("projection", np.zeros((1, 3, 4, 5))),
                       ("projection", np.zeros((2, 3, 4, 4))),
                       ("projection", np.full((2, 3, 4, 5), np.nan))):
        with pytest.raises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
            reduce_conditional_losses(**(inputs | {key: value}))
    with pytest.raises(ValueError):
        reduce_conditional_losses(**inputs, derivative_reduction="norm")
    with pytest.raises((ValueError, TypeError)):
        conditional_mean(tf.ones((2, 4, 1), tf.float32))


def test_finite_inputs_that_overflow_loss_or_normalization_are_refused():
    inputs = loss_inputs()
    for kwargs in ({"integrand": np.full((2, 6, 3), 1e160)},
                   {"integrand": np.full((2, 6, 3), 1e100), "denominators": np.full(6, 1e-200)}):
        with pytest.raises(tf.errors.InvalidArgumentError):
            reduce_conditional_losses(**(inputs | kwargs))


def test_public_reduction_runs_in_stable_graph_without_host_callbacks():
    @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float64),
                                  tf.TensorSpec([None, None, None, None], tf.float64),
                                  tf.TensorSpec([None, None, None, None], tf.float64),
                                  tf.TensorSpec([None], tf.float64)], autograph=False)
    def consumer(integrand, jacobian, projection, denominators):
        return reduce_conditional_losses(integrand, jacobian, projection, denominators)

    consumer(**loss_inputs())
    changed = {"integrand": np.zeros((4, 8, 2)), "jacobian": np.zeros((4, 8, 2, 7)),
               "projection": np.ones((4, 2, 3, 7)), "denominators": np.ones(4)}
    result = consumer(**changed)
    assert result.raw.shape == (4, 4)
    assert consumer.experimental_get_tracing_count() == 1
    graph = consumer.get_concrete_function().graph.as_graph_def()
    operations = {node.op for node in graph.node}
    operations.update(node.op for function in graph.library.function for node in function.node_def)
    assert not operations & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
    assert not any("Random" in operation for operation in operations)


def test_signed_audit_is_detached_in_graph_and_accepts_independent_antithetic_batches():
    left_inputs = prepared("antithetic", rows=2, shock_dim=1, role="audit_left")
    right_inputs = prepared("antithetic", rows=2, shock_dim=1, role="audit_right")

    @tf.function(input_signature=[tf.TensorSpec([2, 4, 1], tf.float64),
                                  tf.TensorSpec([2, 4, 1], tf.float64)], autograph=False)
    def consumer(left, right):
        return independent_signed_audit(left, left[..., None], right, right[..., None],
                                        tf.ones((2, 1, 1, 1), tf.float64), [1., 1.],
                                        left_inputs=left_inputs, right_inputs=right_inputs)

    observed = consumer(left_inputs.nodes + 1., right_inputs.nodes + 1.)
    np.testing.assert_allclose(observed.raw, np.ones((2, 2)), atol=5e-16)
    graph = consumer.get_concrete_function().graph.as_graph_def()
    operations = [node.op for function in graph.library.function for node in function.node_def]
    assert operations.count("StopGradient") >= 2
