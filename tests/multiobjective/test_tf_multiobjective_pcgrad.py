"""Model-independent TensorFlow PCGrad tests."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest
import tensorflow as tf

from mooneural.multiobjective import aggregate
from mooneural.multiobjective.pcgrad import pcgrad_flat


def _var(dim=2):
    return (tf.Variable(tf.zeros([dim], dtype=tf.float64)),)


def _rows(rows):
    return tuple((tf.constant(row, dtype=tf.float64),) for row in rows)


def test_pcgrad_missing_seed_fails_closed():
    with pytest.raises(ValueError, match="requires a TensorFlow stateless seed"):
        aggregate(
            "pcgrad",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
        )


def test_pcgrad_invalid_reduction_fails_closed():
    with pytest.raises(ValueError, match="pcgrad reduction"):
        aggregate(
            "pcgrad",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            seed=tf.constant([1, 2], dtype=tf.int32),
            pcgrad_reduction="median",
        )


def test_pcgrad_flat_requires_rank_two_gradients():
    with pytest.raises(ValueError, match=r"shape \[objectives, params\]"):
        pcgrad_flat(
            tf.ones([2, 2, 1], dtype=tf.float64),
            seed=tf.constant([1, 2], dtype=tf.int32),
        )


def test_pairwise_conflict_projection_removes_negative_component():
    result = aggregate(
        "pcgrad",
        _rows(([1.0, 0.0], [-1.0, 0.0])),
        _var(),
        seed=tf.constant([1, 2], dtype=tf.int32),
        pcgrad_reduction="mean",
    )

    tf.debugging.assert_near(result.flat_gradient, [0.0, 0.0], atol=1e-8)
    assert int(result.diagnostics["pcgrad_projection_count"].numpy()) >= 1


def test_near_zero_conflict_denominator_is_skipped_and_recorded():
    result = aggregate(
        "pcgrad",
        _rows(([-1e-13, 0.0], [1e-13, 0.0])),
        _var(),
        seed=tf.constant([2, 3], dtype=tf.int32),
    )

    assert int(result.diagnostics["pcgrad_projection_count"].numpy()) == 0
    assert int(result.diagnostics["pcgrad_zero_denom_skips"].numpy()) >= 1
    assert bool(result.diagnostics["finite"].numpy())


def test_no_conflict_gradients_are_reduced_by_sum():
    result = aggregate(
        "pcgrad",
        _rows(([1.0, 0.0], [0.0, 2.0])),
        _var(),
        seed=tf.constant([3, 4], dtype=tf.int32),
    )

    tf.debugging.assert_near(result.flat_gradient, [1.0, 2.0], atol=1e-8)
    assert int(result.diagnostics["pcgrad_projection_count"].numpy()) == 0
    assert not bool(result.diagnostics["coefficients_are_simplex"].numpy())


def test_stateless_seed_is_reproducible():
    grads = _rows(([1.0, 0.0], [-0.25, 1.0], [0.5, -0.5]))
    variables = _var()
    seed = tf.constant([9, 13], dtype=tf.int32)

    first = aggregate("pcgrad", grads, variables, seed=seed)
    second = aggregate("pcgrad", grads, variables, seed=seed)

    tf.debugging.assert_near(first.flat_gradient, second.flat_gradient, atol=1e-12)
    tf.debugging.assert_equal(
        first.diagnostics["pcgrad_projection_orders"],
        second.diagnostics["pcgrad_projection_orders"],
    )


def test_pcgrad_mean_reduction_reports_simplex_coefficients():
    result = aggregate(
        "pcgrad",
        _rows(([1.0, 0.0], [0.0, 2.0])),
        _var(),
        seed=tf.constant([5, 6], dtype=tf.int32),
        pcgrad_reduction="mean",
    )

    tf.debugging.assert_near(result.flat_gradient, [0.5, 1.0], atol=1e-8)
    assert bool(result.diagnostics["coefficients_are_simplex"].numpy())


def test_pcgrad_tf_function():
    variables = _var()
    grads = _rows(([1.0, 0.0], [0.0, 2.0]))

    @tf.function
    def run(seed):
        return aggregate("pcgrad", grads, variables, seed=seed).flat_gradient

    tf.debugging.assert_near(
        run(tf.constant([7, 8], dtype=tf.int32)), [1.0, 2.0], atol=1e-8)


def test_pcgrad_many_objective_high_parameter_smoke():
    rows = []
    for i in range(16):
        rows.append([1.0 if j == i % 32 else 0.0 for j in range(32)])

    result = aggregate(
        "pcgrad",
        _rows(rows),
        _var(32),
        seed=tf.constant([11, 12], dtype=tf.int32),
    )

    assert int(result.diagnostics["objective_count"].numpy()) == 16
    assert int(result.diagnostics["gradient_dim"].numpy()) == 32
    assert bool(result.diagnostics["finite"].numpy())
