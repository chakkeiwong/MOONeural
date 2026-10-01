"""Model-independent TensorFlow IMTL-G tests."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest
import tensorflow as tf

from mooneural.multiobjective import aggregate
from mooneural.multiobjective.imtl import imtl_g_flat


def _var(dim=2):
    return (tf.Variable(tf.zeros([dim], dtype=tf.float64)),)


def _rows(rows):
    return tuple((tf.constant(row, dtype=tf.float64),) for row in rows)


def test_imtlg_two_task_linear_system_formula():
    result = aggregate(
        "imtl",
        _rows(([1.0, 0.0], [0.0, 2.0])),
        _var(),
    )

    expected_tail = 1.0 / 3.0
    tf.debugging.assert_near(
        result.coefficients,
        [1.0 - expected_tail, expected_tail],
        atol=1e-8,
    )
    tf.debugging.assert_near(
        result.flat_gradient,
        [1.0 - expected_tail, 2.0 * expected_tail],
        atol=1e-8,
    )


def test_imtlg_direct_solver_rejects_non_matrix_flat_gradients():
    with pytest.raises(ValueError, match="shape \\[objectives, params\\]"):
        imtl_g_flat(tf.constant([1.0, 2.0], dtype=tf.float64))


def test_imtlg_coefficients_need_not_be_simplex_nonnegative():
    result = aggregate(
        "imtl",
        _rows(([1.0, 0.2], [0.1, 1.0], [0.5, 0.7])),
        _var(),
    )

    assert result.coefficients.shape[0] == 3
    assert bool(result.diagnostics["finite"].numpy())
    assert result.diagnostics["reference_status"].numpy() == (
        b"bounded_reference_faithful_fail_closed")


def test_imtlg_zero_gradient_fails_closed():
    with pytest.raises(tf.errors.InvalidArgumentError, match="nonzero"):
        aggregate(
            "imtl",
            _rows(([1.0, 0.0], [0.0, 0.0])),
            _var(),
        )


def test_imtlg_singular_system_fails_closed():
    with pytest.raises(tf.errors.InvalidArgumentError, match="singular"):
        aggregate(
            "imtl",
            _rows(([1.0, 0.0], [0.5, 0.0])),
            _var(),
        )


def test_imtlg_tf_function_nonsingular():
    variables = _var()
    grads = _rows(([1.0, 0.0], [0.0, 2.0]))

    @tf.function
    def run():
        return aggregate("imtl", grads, variables).flat_gradient

    tf.debugging.assert_near(run(), [2.0 / 3.0, 2.0 / 3.0], atol=1e-8)
