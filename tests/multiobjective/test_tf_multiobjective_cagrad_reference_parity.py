"""Independent reference checks for TensorFlow CAGrad evidence hardening."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar
import tensorflow as tf

from mooneural.multiobjective import aggregate


REFERENCE_EPS = 1e-8


def _variables(dim: int):
    return (tf.Variable(tf.zeros([dim], dtype=tf.float64)),)


def _objective_gradients(grads):
    return tuple((tf.constant(row, dtype=tf.float64),)
                 for row in np.asarray(grads, dtype=np.float64))


def _rescale_factor(c: float, rescale: int) -> float:
    if rescale == 0:
        return 1.0
    if rescale == 1:
        return 1.0 + c**2
    if rescale == 2:
        return 1.0 + c
    raise AssertionError(f"unreachable rescale mode: {rescale}")


def _reference_terms(grads, c: float):
    grads = np.asarray(grads, dtype=np.float64)
    k = grads.shape[0]
    gram = grads @ grads.T
    uniform = np.full(k, 1.0 / k, dtype=np.float64)
    g0 = uniform @ grads
    g0_norm = np.sqrt(max(float(uniform @ gram @ uniform), 0.0)
                      + REFERENCE_EPS)
    coef = c * g0_norm + REFERENCE_EPS
    return grads, gram, uniform, g0, coef


def _reference_objective(alpha, gram, uniform, coef):
    alpha = np.asarray(alpha, dtype=np.float64)
    quad = max(float(alpha @ gram @ alpha), 0.0) + REFERENCE_EPS
    return float(alpha @ gram @ uniform + coef * np.sqrt(quad))


def _reference_update_from_alpha(grads, c: float, alpha, *, rescale: int):
    grads, gram, uniform, g0, coef = _reference_terms(grads, c)
    alpha = np.asarray(alpha, dtype=np.float64)
    gw = alpha @ grads
    gw_norm = np.linalg.norm(gw)
    lambda_ = coef / (gw_norm + REFERENCE_EPS)
    unscaled = g0 + lambda_ * gw
    return {
        "alpha": alpha,
        "objective": _reference_objective(alpha, gram, uniform, coef),
        "gw": gw,
        "lambda": lambda_,
        "unscaled_update": unscaled,
        "rescaled_update": unscaled / _rescale_factor(c, rescale),
    }


def _solve_reference_scipy(grads, c: float):
    """Solve the CAGrad simplex objective without local active-set logic."""
    grads, gram, uniform, _, coef = _reference_terms(grads, c)
    k = grads.shape[0]

    def obj(alpha):
        return _reference_objective(alpha, gram, uniform, coef)

    starts = [np.full(k, 1.0 / k, dtype=np.float64)]
    starts.extend(np.eye(k, dtype=np.float64))
    rng = np.random.default_rng(12345 + 17 * k)
    starts.extend(rng.dirichlet(np.ones(k), size=8))

    best = None
    for start in starts:
        res = minimize(
            obj,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints=({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        candidate = np.clip(res.x, 0.0, 1.0)
        total = candidate.sum()
        if total <= 0.0:
            candidate = np.full(k, 1.0 / k, dtype=np.float64)
        else:
            candidate = candidate / total
        value = obj(candidate)
        if best is None or value < best["objective"]:
            best = {
                "alpha": candidate,
                "objective": value,
                "success": bool(res.success),
                "message": str(res.message),
            }
    assert best is not None
    return best


def _solve_reference_two_task_scalar(grads, c: float):
    """Independent 1D simplex solve matching official scalar objective form."""
    grads = np.asarray(grads, dtype=np.float64)
    assert grads.shape[0] == 2
    _, gram, uniform, _, coef = _reference_terms(grads, c)

    def obj(x):
        alpha = np.array([x, 1.0 - x], dtype=np.float64)
        return _reference_objective(alpha, gram, uniform, coef)

    res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded",
                          options={"xatol": 1e-13, "maxiter": 1000})
    candidates = [
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([float(res.x), 1.0 - float(res.x)], dtype=np.float64),
    ]
    alpha = min(candidates, key=lambda a: _reference_objective(
        a, gram, uniform, coef))
    return {
        "alpha": alpha,
        "objective": _reference_objective(alpha, gram, uniform, coef),
        "success": bool(res.success),
        "message": str(res.message),
    }


def _local_cagrad(grads, *, c: float, rescale: int):
    grads = np.asarray(grads, dtype=np.float64)
    result = aggregate(
        "cagrad",
        _objective_gradients(grads),
        _variables(grads.shape[1]),
        cagrad_c=c,
        cagrad_rescale=rescale,
        eps=REFERENCE_EPS,
        cagrad_allow_outside_fixed_step_theorem=(c >= 1.0),
        cagrad_stationarity_tol=1e-6,
    )
    alpha = result.diagnostics["cagrad_subproblem_alpha"].numpy()
    return {
        "alpha": alpha,
        "objective": float(result.diagnostics[
            "cagrad_subproblem_value"].numpy()),
        "residual": float(result.diagnostics[
            "cagrad_stationarity_residual"].numpy()),
        "flat_gradient": result.flat_gradient.numpy(),
        "unscaled_update": (
            result.flat_gradient.numpy() * _rescale_factor(c, rescale)
        ),
        "finite": bool(result.diagnostics["finite"].numpy()),
        "converged": bool(result.diagnostics[
            "cagrad_subproblem_converged"].numpy()),
    }


def _assert_simplex(alpha, *, atol=1e-8):
    assert np.all(np.isfinite(alpha))
    assert abs(float(np.sum(alpha)) - 1.0) <= atol
    assert np.min(alpha) >= -atol


def _assert_local_matches_reference(grads, *, c: float, rescale: int,
                                    reference_solver, objective_atol=2e-7,
                                    update_atol=2e-6):
    local = _local_cagrad(grads, c=c, rescale=rescale)
    reference = reference_solver(grads, c)
    local_from_alpha = _reference_update_from_alpha(
        grads, c, local["alpha"], rescale=rescale)
    reference_from_alpha = _reference_update_from_alpha(
        grads, c, reference["alpha"], rescale=rescale)

    _assert_simplex(local["alpha"], atol=2e-7)
    _assert_simplex(reference["alpha"], atol=2e-7)
    assert local["finite"]
    assert local["converged"]
    assert local["residual"] <= 1e-6

    assert np.isclose(
        local["objective"], local_from_alpha["objective"],
        atol=objective_atol, rtol=2e-7)
    assert np.isclose(
        local["objective"], reference["objective"],
        atol=objective_atol, rtol=2e-7)
    assert np.allclose(
        local["unscaled_update"], local_from_alpha["unscaled_update"],
        atol=update_atol, rtol=2e-6)
    assert np.allclose(
        local["flat_gradient"], local_from_alpha["rescaled_update"],
        atol=update_atol, rtol=2e-6)
    assert np.allclose(
        local["unscaled_update"], reference_from_alpha["unscaled_update"],
        atol=update_atol, rtol=2e-6)
    assert np.allclose(
        local["flat_gradient"], reference_from_alpha["rescaled_update"],
        atol=update_atol, rtol=2e-6)
    return local, reference


@pytest.mark.parametrize("rescale", [0, 1, 2])
def test_two_task_official_scalar_reference_matches_each_rescale(rescale):
    grads = np.array([
        [1.0, 0.0],
        [-0.25, 1.0],
    ], dtype=np.float64)

    _assert_local_matches_reference(
        grads,
        c=0.4,
        rescale=rescale,
        reference_solver=_solve_reference_two_task_scalar,
    )


@pytest.mark.parametrize("grads,c", [
    (
        np.array([
            [1.0, -0.5, 0.25],
            [-0.2, 0.75, 1.2],
            [0.4, 0.1, -0.7],
        ], dtype=np.float64),
        0.3,
    ),
    (
        np.array([
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 1.0, 0.0, -0.1],
            [0.0, 0.0, 1.0, 0.3],
            [-0.4, 0.2, 0.1, 1.0],
        ], dtype=np.float64),
        0.7,
    ),
    (
        np.array([
            [1.0, 0.2, -0.1],
            [-0.3, 0.9, 0.4],
            [0.5, -0.4, 0.8],
            [0.1, 0.3, -0.6],
            [-0.7, 0.2, 0.2],
        ], dtype=np.float64),
        0.5,
    ),
])
def test_k_task_scipy_simplex_reference_matches_objective_and_update(grads, c):
    _assert_local_matches_reference(
        grads,
        c=c,
        rescale=1,
        reference_solver=_solve_reference_scipy,
        objective_atol=5e-7,
        update_atol=5e-6,
    )


@pytest.mark.parametrize("name,grads,c,solver", [
    (
        "duplicate_nonzero",
        np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64),
        0.3,
        _solve_reference_scipy,
    ),
    (
        "all_zero",
        np.zeros((3, 4), dtype=np.float64),
        0.6,
        _solve_reference_scipy,
    ),
    (
        "one_zero_one_nonzero",
        np.array([
            [0.0, 0.0],
            [1.25, -0.5],
        ], dtype=np.float64),
        0.4,
        _solve_reference_two_task_scalar,
    ),
    (
        "opposite",
        np.array([
            [1.0, -0.5],
            [-1.0, 0.5],
        ], dtype=np.float64),
        0.4,
        _solve_reference_two_task_scalar,
    ),
    (
        "near_collinear_unequal_norm",
        np.array([
            [1.0, 0.0, 0.0],
            [1.000001, 1e-6, 0.0],
            [1.8, -2e-6, 1e-7],
        ], dtype=np.float64),
        0.2,
        _solve_reference_scipy,
    ),
])
def test_required_degeneracy_matrix_matches_reference(
        name, grads, c, solver):
    del name
    local, _ = _assert_local_matches_reference(
        grads,
        c=c,
        rescale=1,
        reference_solver=solver,
        objective_atol=1e-6,
        update_atol=1e-5,
    )
    assert np.all(np.isfinite(local["flat_gradient"]))


@pytest.mark.parametrize("c", [1.0, 1.5, 10.0])
def test_c_ge_one_two_task_reference_consistent_but_not_theorem_evidence(c):
    """Runtime/objective consistency for c>=1 does not extend the paper theorem."""
    grads = np.array([
        [1.0, 0.0],
        [-0.25, 1.0],
    ], dtype=np.float64)

    _assert_local_matches_reference(
        grads,
        c=c,
        rescale=1,
        reference_solver=_solve_reference_two_task_scalar,
        objective_atol=5e-7,
        update_atol=5e-6,
    )


@pytest.mark.parametrize("c", [1.0, 1.5, 2.5])
def test_c_ge_one_k_task_reference_consistent_but_not_theorem_evidence(c):
    """K-task c>=1 checks are runtime/reference evidence, not convergence evidence."""
    grads = np.array([
        [1.0, 0.5, -0.2],
        [-0.3, 1.2, 0.1],
        [0.4, -0.8, 0.9],
    ], dtype=np.float64)

    _assert_local_matches_reference(
        grads,
        c=c,
        rescale=1,
        reference_solver=_solve_reference_scipy,
        objective_atol=5e-7,
        update_atol=5e-6,
    )


def test_c_equal_one_zero_update_witness_is_not_average_loss_stationarity():
    """At c=1, a zero update need not mean the average gradient is stationary."""
    c = 1.0
    grads = np.array([
        [1.0],
        [-0.2],
    ], dtype=np.float64)

    local, reference = _assert_local_matches_reference(
        grads,
        c=c,
        rescale=0,
        reference_solver=_solve_reference_two_task_scalar,
        objective_atol=5e-7,
        update_atol=5e-6,
    )
    _, _, _, g0, _ = _reference_terms(grads, c)
    reference_from_alpha = _reference_update_from_alpha(
        grads, c, reference["alpha"], rescale=0)

    assert np.linalg.norm(g0) > 0.1
    assert np.linalg.norm(local["unscaled_update"]) <= 5e-6
    assert np.linalg.norm(reference_from_alpha["unscaled_update"]) <= 5e-6
