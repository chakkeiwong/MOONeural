"""Model-independent TensorFlow CAGrad tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest
import tensorflow as tf

from mooneural.multiobjective import aggregate
from mooneural.multiobjective.cagrad import (
    CAGRAD_ACTIVE_MIN_RELATIVE_TOL,
    CAGRAD_REFERENCE_EPS,
    CAGRAD_SIMPLEX_CONTRACT_TOL,
    _kkt_components,
    _stationarity_contract_pass,
    cagrad_flat,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAGRAD_GATE2_FIXTURE = (
    _REPO_ROOT
    / "results/neural/rotemberg/cagrad_gate2_mechanics_debug_20260527"
    / "medium_cagrad_target_0_epoch_0008_gradients.npz"
)
_CAGRAD_GATE2_EPOCH_0202_FIXTURE = (
    _REPO_ROOT
    / "results/neural/rotemberg/cagrad_gate2_c5_mechanics_20260530"
    # Historical filename from the C5 mechanics rerun that captured epoch 202;
    # the current production contract is scale-relative, not C5.
    / "medium_cagrad_target_0_epoch_0202_c5_failure.npz"
)


def _var(dim=2):
    return (tf.Variable(tf.zeros([dim], dtype=tf.float64)),)


def _rows(rows):
    return tuple((tf.constant(row, dtype=tf.float64),) for row in rows)


def _rows32(rows):
    return tuple((tf.constant(row, dtype=tf.float32),) for row in rows)


def _load_rotemberg_gate2_flat_gradients():
    if not _CAGRAD_GATE2_FIXTURE.exists():
        pytest.skip("Rotemberg CAGrad Gate-2 fixture not available")
    np = pytest.importorskip("numpy")
    with np.load(_CAGRAD_GATE2_FIXTURE) as fixture:
        return tf.constant(fixture["flat_gradients"], dtype=tf.float64)


def _load_rotemberg_gate2_epoch_0202_flat_gradients():
    if not _CAGRAD_GATE2_EPOCH_0202_FIXTURE.exists():
        pytest.skip("Rotemberg CAGrad Gate-2 epoch-202 fixture not available")
    np = pytest.importorskip("numpy")
    with np.load(_CAGRAD_GATE2_EPOCH_0202_FIXTURE) as fixture:
        return tf.constant(fixture["flat_gradients"], dtype=tf.float64)


def test_two_task_reference_formula_matches_official_toy_objective():
    grads = _rows(([1.0, 0.0], [-0.25, 1.0]))
    variables = _var()

    result = aggregate(
        "cagrad",
        grads,
        variables,
        cagrad_c=0.4,
        cagrad_rescale=1,
    )

    alpha = result.diagnostics["cagrad_subproblem_alpha"]
    tf.debugging.assert_near(tf.reduce_sum(alpha), 1.0, atol=1e-10)
    tf.debugging.assert_greater_equal(alpha, tf.zeros_like(alpha) - 1e-10)
    assert bool(result.diagnostics["cagrad_subproblem_converged"].numpy())
    assert float(result.diagnostics["cagrad_stationarity_residual"].numpy()) < 1e-7
    assert not bool(result.diagnostics["coefficients_are_simplex"].numpy())
    assert bool(
        result.diagnostics[
            "cagrad_coefficients_are_update_coefficients"].numpy())


def test_no_conflict_still_returns_finite_non_simplex_update_coefficients():
    result = aggregate(
        "cagrad",
        _rows(([1.0, 0.0], [0.0, 2.0])),
        _var(),
        cagrad_c=0.5,
    )

    assert bool(result.diagnostics["finite"].numpy())
    assert not bool(result.diagnostics["coefficients_are_simplex"].numpy())
    assert float(tf.linalg.norm(result.flat_gradient).numpy()) > 0.0


def test_duplicate_objectives_do_not_trigger_uniform_or_mgda_fallback():
    result = aggregate(
        "cagrad",
        _rows(([1.0, 0.0], [1.0, 0.0], [0.0, 1.0])),
        _var(),
        cagrad_c=0.3,
    )

    alpha = result.diagnostics["cagrad_subproblem_alpha"]
    tf.debugging.assert_near(tf.reduce_sum(alpha), 1.0, atol=1e-8)
    assert bool(result.diagnostics["cagrad_subproblem_converged"].numpy())
    assert bool(result.diagnostics["finite"].numpy())


def test_zero_gradient_objective_is_handled_without_nonfinite_output():
    result = aggregate(
        "cagrad",
        _rows(([1.0, 0.0], [0.0, 0.0])),
        _var(),
        cagrad_c=0.4,
    )

    assert bool(result.diagnostics["finite"].numpy())
    tf.debugging.assert_all_finite(result.flat_gradient, "finite CAGrad update")


def test_float32_ill_conditioned_subproblem_uses_stable_internal_solve():
    result = aggregate(
        "cagrad",
        _rows32((
            [1.0, 0.999999, 0.0],
            [0.999998, 1.000001, 0.0],
            [-0.2, 0.3, 1e-4],
            [0.05, -0.1, -2e-4],
        )),
        (tf.Variable(tf.zeros([3], dtype=tf.float32)),),
        cagrad_c=0.2,
    )

    assert result.flat_gradient.dtype == tf.float32
    assert result.coefficients.dtype == tf.float32
    assert (
        result.diagnostics["cagrad_internal_solve_dtype"].numpy()
        == b"float64"
    )
    assert bool(result.diagnostics["cagrad_subproblem_converged"].numpy())
    assert float(result.diagnostics["cagrad_stationarity_residual"].numpy()) <= 1e-6


def test_rotemberg_gate2_fixture_passes_scale_relative_contract_with_raw_residual_visible():
    flat = _load_rotemberg_gate2_flat_gradients()
    grads = tuple((row,) for row in tf.unstack(flat, axis=0))
    variables = (tf.Variable(tf.zeros([flat.shape[1]], dtype=tf.float64)),)

    result = aggregate(
        "cagrad",
        grads,
        variables,
        cagrad_c=0.05,
    )

    residual = float(result.diagnostics["cagrad_stationarity_residual"].numpy())
    assert residual == pytest.approx(1.52587890625e-05)
    assert residual > 1e-6
    assert bool(result.diagnostics["cagrad_stationarity_contract_pass"].numpy())
    assert bool(result.diagnostics["cagrad_subproblem_converged"].numpy())
    active_size = int(result.diagnostics["cagrad_active_set_size"].numpy())
    assert active_size == 2

    finite_diagnostics = (
        "cagrad_active_stationarity_residual",
        "cagrad_inactive_kkt_residual",
        "cagrad_simplex_residual",
        "cagrad_active_mean_relative_residual",
        "cagrad_active_min_scale",
        "cagrad_active_min_relative_residual",
        "cagrad_active_stationarity_ulps",
    )
    for name in finite_diagnostics:
        tf.debugging.assert_all_finite(
            result.diagnostics[name], f"finite {name}")

    assert (
        float(result.diagnostics["cagrad_inactive_kkt_residual"].numpy())
        <= 1e-6
    )
    assert (
        float(result.diagnostics["cagrad_simplex_residual"].numpy())
        <= CAGRAD_SIMPLEX_CONTRACT_TOL
    )
    assert (
        float(
            result.diagnostics[
                "cagrad_active_min_relative_residual"].numpy())
        <= CAGRAD_ACTIVE_MIN_RELATIVE_TOL
    )
    assert float(result.diagnostics["cagrad_active_stationarity_ulps"].numpy()) >= 0.0


def test_rotemberg_gate2_epoch_0202_fixture_passes_scale_relative_contract():
    flat = _load_rotemberg_gate2_epoch_0202_flat_gradients()
    grads = tuple((row,) for row in tf.unstack(flat, axis=0))
    variables = (tf.Variable(tf.zeros([flat.shape[1]], dtype=tf.float64)),)

    result = aggregate(
        "cagrad",
        grads,
        variables,
        cagrad_c=0.05,
    )

    residual = float(result.diagnostics["cagrad_stationarity_residual"].numpy())
    assert residual == pytest.approx(4.231929779052734e-06)
    assert residual > 1e-6
    assert bool(result.diagnostics["cagrad_stationarity_contract_pass"].numpy())
    assert bool(result.diagnostics["cagrad_subproblem_converged"].numpy())
    assert (
        float(result.diagnostics["cagrad_active_min_relative_residual"].numpy())
        <= CAGRAD_ACTIVE_MIN_RELATIVE_TOL
    )
    assert (
        float(result.diagnostics["cagrad_active_mean_relative_residual"].numpy())
        == pytest.approx(4.204923788008239e-14)
    )
    assert (
        float(result.diagnostics["cagrad_active_stationarity_ulps"].numpy())
        == pytest.approx(284.0)
    )
    assert (
        float(result.diagnostics["cagrad_inactive_kkt_residual"].numpy())
        <= 1e-6
    )
    assert (
        float(result.diagnostics["cagrad_simplex_residual"].numpy())
        <= CAGRAD_SIMPLEX_CONTRACT_TOL
    )


def test_rotemberg_gate2_uniform_alpha_negative_control_fails_scale_relative_contract():
    flat = _load_rotemberg_gate2_flat_gradients()
    dtype = tf.float64
    k = int(flat.shape[0])
    gram = tf.linalg.matmul(flat, flat, transpose_b=True)
    gram = 0.5 * (gram + tf.transpose(gram))
    uniform = tf.fill([k], tf.cast(1, dtype) / tf.cast(k, dtype))
    eps = tf.cast(CAGRAD_REFERENCE_EPS, dtype)
    g0_norm = tf.sqrt(tf.maximum(
        tf.tensordot(uniform, tf.linalg.matvec(gram, uniform), axes=1),
        tf.cast(0, dtype)) + eps)
    coef = tf.cast(0.05, dtype) * g0_norm + eps

    components = _kkt_components(gram, uniform, coef, uniform, eps)
    contract_pass = _stationarity_contract_pass(
        components, tf.cast(1e-6, dtype), dtype)

    assert not bool(contract_pass.numpy())
    assert float(components["max_residual"].numpy()) > 1e-6
    assert (
        float(components["active_min_relative_residual"].numpy())
        > CAGRAD_ACTIVE_MIN_RELATIVE_TOL
    )


def test_large_objective_guard_fails_closed():
    rows = [[1.0 if i == j else 0.0 for j in range(9)] for i in range(9)]

    with pytest.raises(ValueError, match="limited to 8 objectives"):
        aggregate("cagrad", _rows(rows), _var(9), max_objectives=8)


def test_negative_c_parameter_fails_closed():
    with pytest.raises(ValueError, match="must be nonnegative"):
        aggregate(
            "cagrad",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            cagrad_c=-0.1,
        )


@pytest.mark.parametrize("cagrad_c", [1.0, 1.5])
def test_c_ge_one_requires_explicit_non_theorem_opt_in(cagrad_c):
    with pytest.raises(ValueError, match="outside the CAGrad fixed-step"):
        aggregate(
            "cagrad",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            cagrad_c=cagrad_c,
        )


@pytest.mark.parametrize("c", [1.0, 1.5])
def test_low_level_cagrad_flat_c_ge_one_fails_closed_by_default(c):
    flat = tf.constant([[1.0, 0.0], [-0.25, 1.0]], dtype=tf.float64)

    with pytest.raises(ValueError, match="outside the CAGrad fixed-step"):
        cagrad_flat(flat, c=c)


def test_low_level_cagrad_flat_c_ge_one_requires_explicit_opt_in():
    flat = tf.constant([[1.0, 0.0], [-0.25, 1.0]], dtype=tf.float64)

    coefficients, combined, diagnostics = cagrad_flat(
        flat,
        c=1.5,
        allow_outside_fixed_step_theorem=True,
    )

    tf.debugging.assert_all_finite(coefficients, "finite CAGrad coefficients")
    tf.debugging.assert_all_finite(combined, "finite CAGrad update")
    assert bool(diagnostics["cagrad_subproblem_converged"].numpy())


def test_cagrad_tf_function_static_small_k():
    variables = _var()
    grads = _rows(([1.0, 0.0], [-0.25, 1.0]))

    @tf.function
    def run():
        result = aggregate(
            "cagrad",
            grads,
            variables,
            cagrad_c=0.4,
        )
        return (
            result.flat_gradient,
            result.diagnostics["cagrad_stationarity_residual"],
        )

    flat, residual = run()
    tf.debugging.assert_all_finite(flat, "finite tf.function CAGrad update")
    assert float(residual.numpy()) < 1e-7
