"""Contract tests for TensorFlow-native bounded Aligned-MTL."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest
import tensorflow as tf

from mooneural.multiobjective import aggregate
from mooneural.multiobjective.aligned import aligned_flat


def _variables(dim=2, dtype=tf.float64):
    return (tf.Variable(tf.zeros([dim], dtype=dtype)),)


def _grads(rows, dtype=tf.float64):
    return tuple((tf.constant(row, dtype=dtype),) for row in rows)


def test_aligned_equal_orthogonal_gradients_preserve_sum():
    result = aggregate(
        "aligned",
        _grads(([1.0, 0.0], [0.0, 1.0])),
        _variables(),
    )

    tf.debugging.assert_near(result.coefficients, [1.0, 1.0], atol=1e-12)
    tf.debugging.assert_near(result.flat_gradient, [1.0, 1.0], atol=1e-12)
    assert result.diagnostics["aligned_operator_status"].numpy() == (
        b"bounded_tf_procrustes_operator"
    )
    assert int(result.diagnostics["aligned_rank"].numpy()) == 2
    assert result.diagnostics["aligned_scale_mode"].numpy() == b"min"
    assert not bool(result.diagnostics["coefficients_are_simplex"].numpy())


def test_aligned_min_scale_orthogonal_unequal_gradients_equalizes_axes():
    result = aggregate(
        "aligned",
        _grads(([2.0, 0.0], [0.0, 1.0])),
        _variables(),
    )

    tf.debugging.assert_near(result.coefficients, [0.5, 1.0], atol=1e-12)
    tf.debugging.assert_near(result.flat_gradient, [1.0, 1.0], atol=1e-12)
    tf.debugging.assert_near(
        result.diagnostics["aligned_retained_gram_eigenvalues"],
        [4.0, 1.0],
        atol=1e-12,
    )
    assert result.diagnostics["aligned_condition_before"].numpy() == pytest.approx(
        2.0
    )
    assert result.diagnostics["aligned_condition_after"].numpy() == pytest.approx(
        1.0
    )


def test_aligned_rmse_scale_uses_mean_retained_gram_eigenvalue():
    result = aggregate(
        "aligned",
        _grads(([2.0, 0.0], [0.0, 1.0])),
        _variables(),
        aligned_scale_mode="rmse",
    )

    expected = 2.5**0.5
    assert result.diagnostics["aligned_scale"].numpy() == pytest.approx(2.5)
    tf.debugging.assert_near(
        result.flat_gradient,
        [expected, expected],
        atol=1e-12,
    )


def test_aligned_condition_number_improves_for_nonsingular_case():
    coefficients, combined, diagnostics = aligned_flat(
        tf.constant([[2.0, 0.0], [0.5, 1.0]], dtype=tf.float64)
    )

    assert coefficients.shape == (2,)
    assert combined.shape == (2,)
    assert diagnostics["aligned_condition_before"].numpy() > 1.0
    assert diagnostics["aligned_condition_after"].numpy() == pytest.approx(1.0)
    assert bool(tf.reduce_all(tf.math.is_finite(coefficients)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(combined)).numpy())


def test_aligned_rank_deficient_nonzero_case_is_bounded_and_diagnostic():
    result = aggregate(
        "aligned",
        _grads(([1.0, 0.0], [2.0, 0.0])),
        _variables(),
    )

    assert int(result.diagnostics["aligned_rank"].numpy()) == 1
    assert result.flat_gradient.shape == (2,)
    assert bool(result.diagnostics["finite"].numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result.coefficients)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result.flat_gradient)).numpy())


def test_aligned_rank_zero_fails_closed():
    with pytest.raises(tf.errors.InvalidArgumentError, match="zero retained rank"):
        aligned_flat(tf.zeros([2, 2], dtype=tf.float64))


def test_aligned_invalid_shape_fails_closed():
    with pytest.raises(ValueError, match="shape"):
        aligned_flat(tf.zeros([2], dtype=tf.float64))


def test_aligned_invalid_mode_fails_closed():
    with pytest.raises(ValueError, match="scale_mode"):
        aligned_flat(tf.eye(2, dtype=tf.float64), scale_mode="max")


def test_aligned_nonfinite_input_fails_closed():
    with pytest.raises(tf.errors.InvalidArgumentError, match="finite"):
        aligned_flat(tf.constant([[1.0, 0.0], [float("nan"), 1.0]], dtype=tf.float64))


def test_aligned_tf_function_compatible():
    @tf.function
    def combine(flat):
        coefficients, combined, diagnostics = aligned_flat(flat)
        return coefficients, combined, diagnostics["aligned_rank"]

    coefficients, combined, rank = combine(
        tf.constant([[2.0, 0.0], [0.0, 1.0]], dtype=tf.float64)
    )

    tf.debugging.assert_near(coefficients, [0.5, 1.0], atol=1e-12)
    tf.debugging.assert_near(combined, [1.0, 1.0], atol=1e-12)
    assert int(rank.numpy()) == 2
