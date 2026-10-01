"""Manufactured geometric cases for an optional, nondefault step guard."""

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_displacement_blend import blend_displacements


def guard(proposal, reference, rows=None, active=None, **kwargs):
    return blend_displacements(np.asarray(proposal, np.float64), np.asarray(reference, np.float64),
        np.eye(2, dtype=np.float64) if rows is None else np.asarray(rows, np.float64),
        np.array([True, False]) if active is None else np.asarray(active, bool), margin_fraction=.1, **kwargs)


def test_minimal_segment_mixing_repairs_adam_ascent():
    result = guard([1., -1.], [-1., -1.], relative_tolerance=0.)
    assert bool(result["valid"])
    np.testing.assert_allclose(result["fraction"], .55, atol=1e-15)
    np.testing.assert_allclose(result["displacement"], [-.1, -1.], atol=1e-15)
    assert float(result["lower"]) <= float(result["upper"])
    assert 1. - 2 * .549 > -.1


def test_already_qualified_proposal_is_preserved():
    result = guard([-.2, -1.], [-1., -1.])
    assert bool(result["valid"]) and float(result["fraction"]) == 0.
    np.testing.assert_array_equal(result["displacement"], [-.2, -1.])


def test_protected_conflict_refuses_only_optional_candidate():
    result = guard([1., -1.], [-1., 2.])
    assert not bool(result["valid"])
    assert float(result["lower"]) > float(result["upper"])
    np.testing.assert_array_equal(result["displacement"], [1., -1.])


def test_no_direction_on_segment_can_repair_constant_ascent():
    result = guard([1., -1.], [1., -2.])
    assert not bool(result["valid"])
    np.testing.assert_array_equal(result["displacement"], [1., -1.])


def test_zero_gradient_and_zero_step_are_finite():
    result = guard([0., 0.], [0., 0.], rows=np.zeros((2, 2)))
    assert bool(result["valid"])
    for name in ("displacement", "slopes", "limits", "slack"):
        assert np.isfinite(np.asarray(result[name])).all()


def test_positive_task_scaling_and_row_permutation_do_not_change_guard():
    original = guard([1., -1.], [-1., -1.])
    scaled = guard([1., -1.], [-1., -1.], rows=[[1e-150, 0.], [0., 1e150]])
    permuted = guard([1., -1.], [-1., -1.], rows=[[0., 1e150], [1e-150, 0.]], active=[False, True])
    for result in (scaled, permuted):
        assert bool(result["valid"])
        np.testing.assert_array_equal(result["displacement"], original["displacement"])


def test_guard_composes_with_one_stable_graph():
    function = tf.function(lambda proposal, reference, rows, active: blend_displacements(
        proposal, reference, rows, active, margin_fraction=.1), autograph=False, input_signature=[
            tf.TensorSpec([2], tf.float64), tf.TensorSpec([2], tf.float64),
            tf.TensorSpec([2, 2], tf.float64), tf.TensorSpec([2], tf.bool)])
    for proposal in ([1., -1.], [.5, -1.]):
        result = function(np.array(proposal), np.array([-1., -1.]), np.eye(2), np.array([True, False]))
        assert bool(result["valid"])
    assert function.experimental_get_tracing_count() == 1
    assert not function.get_concrete_function().variables


def test_near_tangent_protected_row_uses_same_declared_rounding_allowance():
    proposal = np.array([1., -1., 1e-13])
    reference = np.array([-1., -1., -1e-18])
    rows = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    result = blend_displacements(proposal, reference, rows, np.array([True, False, False]), margin_fraction=.1)
    assert bool(result["valid"])
    np.testing.assert_allclose(result["fraction"], .55, atol=1e-12)
    assert float(result["proposal_slopes"][2]) < float(result["slack"])
    assert np.all(np.asarray(result["slopes"]) <= np.asarray(result["limits"]) + float(result["slack"]))


@pytest.mark.parametrize("magnitude", [1e200, 1.7e308])
def test_large_finite_input_cannot_certify_ascent_through_infinite_slack(magnitude):
    result = guard([magnitude, magnitude], [magnitude, magnitude])
    assert not bool(result["valid"])
    np.testing.assert_array_equal(result["displacement"], [magnitude, magnitude])
    if magnitude == 1e200:
        assert np.isfinite(float(result["slack"]))


@pytest.mark.parametrize("fault", ["nan", "shape", "empty", "dtype", "mask", "margin", "tolerance"])
def test_invalid_input_refused(fault):
    arguments = [np.array([1., -1.]), np.array([-1., -1.]), np.eye(2), np.array([True, False])]
    options = {"margin_fraction": .1}
    if fault == "nan":
        arguments[0][0] = np.nan
    elif fault == "shape":
        arguments[2] = np.ones((3, 2))
    elif fault == "empty":
        arguments[0] = arguments[1] = np.zeros(0)
        arguments[2] = np.zeros((2, 0))
    elif fault == "dtype":
        arguments[0] = arguments[0].astype(np.float32)
    elif fault == "mask":
        arguments[3] = np.ones(2)
    elif fault == "margin":
        options["margin_fraction"] = 1.1
    else:
        options["relative_tolerance"] = -1.
    with pytest.raises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
        blend_displacements(*arguments, **options)
