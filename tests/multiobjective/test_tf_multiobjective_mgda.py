"""Model-independent TensorFlow MGDA tests."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest
import tensorflow as tf

from mooneural.multiobjective import aggregate
from mooneural.multiobjective.mgda import mgda_flat


def _var(dim=2):
    return (tf.Variable(tf.zeros([dim], dtype=tf.float64)),)


def _rows(rows):
    return tuple((tf.constant(row, dtype=tf.float64),) for row in rows)


def test_two_opposing_gradients_give_zero_direction():
    result = aggregate(
        "mgda",
        _rows(([1.0, 0.0], [-1.0, 0.0])),
        _var(),
    )

    tf.debugging.assert_near(result.coefficients, [0.5, 0.5], atol=1e-8)
    tf.debugging.assert_near(result.flat_gradient, [0.0, 0.0], atol=1e-8)
    assert int(result.diagnostics["mgda_active_set_size"].numpy()) == 2
    assert bool(result.diagnostics["coefficients_are_simplex"].numpy())


def test_mgda_direct_solver_rejects_non_matrix_flat_gradients():
    with pytest.raises(ValueError, match="shape \\[objectives, params\\]"):
        mgda_flat(tf.constant([1.0, 2.0], dtype=tf.float64))


def test_two_objective_formula_case_matches_expected_active_pair():
    result = aggregate(
        "mgda",
        _rows(([2.0, 0.0], [0.0, 1.0])),
        _var(),
    )

    expected_alpha_1 = 1.0 / 5.0
    tf.debugging.assert_near(
        result.coefficients,
        [expected_alpha_1, 1.0 - expected_alpha_1],
        atol=1e-8,
    )


def test_duplicate_objective_permutation_preserves_direction():
    variables = _var()
    result = aggregate(
        "mgda",
        _rows(([1.0, 0.0], [1.0, 0.0], [0.0, 1.0])),
        variables,
    )
    permuted = aggregate(
        "mgda",
        _rows(([0.0, 1.0], [1.0, 0.0], [1.0, 0.0])),
        variables,
    )

    tf.debugging.assert_near(result.flat_gradient, permuted.flat_gradient, atol=1e-8)
    assert bool(result.diagnostics["coefficients_are_simplex"].numpy())
    assert bool(permuted.diagnostics["coefficients_are_simplex"].numpy())


def test_singular_duplicate_active_set_does_not_mask_feasible_singleton():
    result = aggregate(
        "mgda",
        _rows(([1.0, 0.0], [1.0, 0.0])),
        _var(),
    )

    tf.debugging.assert_near(result.flat_gradient, [1.0, 0.0], atol=1e-8)
    assert int(result.diagnostics["mgda_active_set_size"].numpy()) == 1
    assert bool(result.diagnostics["coefficients_are_simplex"].numpy())


def test_zero_gradient_objective_can_be_selected():
    result = aggregate(
        "mgda",
        _rows(([1.0, 0.0], [0.0, 0.0])),
        _var(),
    )

    tf.debugging.assert_near(result.flat_gradient, [0.0, 0.0], atol=1e-8)
    assert float(result.coefficients[1].numpy()) > 0.999


def test_large_objective_guard_fails_closed():
    rows = [[1.0 if i == j else 0.0 for j in range(9)] for i in range(9)]

    with pytest.raises(ValueError, match="limited to 8 objectives"):
        aggregate("mgda", _rows(rows), _var(9), max_objectives=8)


def test_mgda_tf_function_static_small_k():
    variables = _var()
    grads = _rows(([1.0, 0.0], [-1.0, 0.0]))

    @tf.function
    def run():
        return aggregate("mgda", grads, variables).flat_gradient

    tf.debugging.assert_near(run(), [0.0, 0.0], atol=1e-8)
