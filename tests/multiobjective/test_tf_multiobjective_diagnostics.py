"""Diagnostics tests for TensorFlow multiobjective utilities."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest
import tensorflow as tf

from mooneural.multiobjective import aggregate, create_famo_state, create_gradnorm_state


def _var(dim=4):
    return (tf.Variable(tf.zeros([dim], dtype=tf.float64)),)


def _rows(rows):
    return tuple((tf.constant(row, dtype=tf.float64),) for row in rows)


def test_diagnostics_include_directional_derivatives_and_shapes():
    result = aggregate(
        "mgda",
        _rows(([1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0])),
        _var(),
    )
    diag = result.diagnostics

    assert diag["objective_directional_derivatives"].shape == (2,)
    assert diag["cosine"].shape == (2, 2)
    assert int(diag["gram_element_count"].numpy()) == 4
    assert int(diag["flat_gradient_element_count"].numpy()) == 8
    assert bool(diag["finite"].numpy())


@pytest.mark.parametrize(
    ("method", "kwargs", "expected_reference_status", "expect_simplex"),
    [
        (
            "mgda",
            {},
            b"bounded_reference_faithful_small_k",
            True,
        ),
        (
            "pcgrad",
            {"seed": tf.constant([2, 3], tf.int32)},
            b"bounded_reference_faithful_gradient_surgery",
            False,
        ),
        (
            "imtl",
            {},
            b"bounded_reference_faithful_fail_closed",
            None,
        ),
        (
            "famo",
            {
                "losses": tf.constant([1.0, 2.0], dtype=tf.float64),
                "state": create_famo_state(2),
            },
            b"bounded_reference_aligned_functional_state",
            True,
        ),
        (
            "cagrad",
            {"cagrad_c": 0.4},
            b"bounded_reference_faithful_small_k_strict_oracle",
            False,
        ),
        (
            "gradnorm",
            {
                "losses": tf.constant([1.0, 2.0], dtype=tf.float64),
                "state": create_gradnorm_state(2),
            },
            b"bounded_usable_selected_shared_gradients_not_official_code_parity",
            False,
        ),
        (
            "aligned",
            {},
            b"bounded_tf_procrustes_operator_not_benchmark_reproduction",
            False,
        ),
    ],
)
def test_common_diagnostics_cover_runtime_contract_surface(
    method,
    kwargs,
    expected_reference_status,
    expect_simplex,
):
    result = aggregate(
        method,
        _rows(([1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0])),
        _var(),
        **kwargs,
    )
    diag = result.diagnostics

    assert diag["method"].numpy() == method.encode()
    assert diag["reference_status"].numpy() == expected_reference_status
    assert int(diag["objective_count"].numpy()) == 2
    assert int(diag["gradient_dim"].numpy()) == 4
    assert diag["gram"].shape == (2, 2)
    assert diag["cosine"].shape == (2, 2)
    assert diag["coefficients"].shape == (2,)
    assert int(diag["flat_gradient_element_count"].numpy()) == 8
    assert int(diag["gram_element_count"].numpy()) == 4
    assert bool(diag["finite"].numpy())
    if expect_simplex is not None:
        assert bool(diag["coefficients_are_simplex"].numpy()) is expect_simplex
    if method == "cagrad":
        assert diag["cagrad_c_theorem_scope"].numpy() == (
            b"c>=1_default_blocked_a5_path_removed_explicit_non_theorem_opt_in")
        assert diag["cagrad_a5_status"].numpy() == (
            b"blocked_removed_not_implemented")
        assert bool(
            diag[
                "cagrad_fixed_step_average_loss_theorem_supported"].numpy())
        assert not bool(
            diag["cagrad_outside_fixed_step_theorem_opt_in"].numpy())
    else:
        assert "cagrad_c_theorem_scope" not in diag
        assert "cagrad_a5_status" not in diag
        assert "cagrad_fixed_step_average_loss_theorem_supported" not in diag
        assert "cagrad_outside_fixed_step_theorem_opt_in" not in diag


def test_none_gradients_are_zero_filled():
    variables = (
        tf.Variable([0.0, 0.0], dtype=tf.float64),
        tf.Variable([[0.0]], dtype=tf.float64),
    )
    grads = (
        (None, tf.constant([[1.0]], dtype=tf.float64)),
        (tf.constant([1.0, 0.0], dtype=tf.float64), None),
    )

    result = aggregate("pcgrad", grads, variables, seed=tf.constant([2, 3]))

    assert int(result.diagnostics["gradient_dim"].numpy()) == 3
    assert result.combined_gradients[0].shape == variables[0].shape
    assert result.combined_gradients[1].shape == variables[1].shape
