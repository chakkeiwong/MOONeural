"""TensorFlow-native bounded CAGrad active-set/KKT solver.

Invalid or unconverged subproblems fail closed rather than falling back to
MGDA, uniform weights, or stale coefficients.
"""

from __future__ import annotations

import itertools

import tensorflow as tf


CAGRAD_REFERENCE_EPS = 1e-8
CAGRAD_ACTIVE_MIN_RELATIVE_TOL = 1e-12
CAGRAD_SIMPLEX_CONTRACT_TOL = 1e-10


def cagrad_flat(
    flat_gradients,
    *,
    c=0.5,
    rescale=1,
    eps=CAGRAD_REFERENCE_EPS,
    max_objectives=8,
    stationarity_tol=1e-6,
    allow_outside_fixed_step_theorem=False,
):
    """Return CAGrad update coefficients and combined flat gradient.

    `c >= 1` is outside the CAGrad fixed-step average-loss theorem scope and
    requires an explicit non-theorem opt-in even for this low-level primitive.
    """
    flat = tf.convert_to_tensor(flat_gradients)
    dtype = flat.dtype
    solve_dtype = tf.float64 if dtype != tf.float64 else dtype
    flat_solve = tf.cast(flat, solve_dtype)
    if flat.shape.rank != 2:
        raise ValueError("flat_gradients must have shape [objectives, params]")
    k_static = flat.shape[0]
    if k_static is None:
        raise ValueError("CAGrad requires a statically known objective count")
    k = int(k_static)
    if k < 1:
        raise ValueError("at least one objective is required")
    if max_objectives is not None and k > int(max_objectives):
        raise ValueError(
            f"CAGrad active-set solver is limited to {int(max_objectives)} "
            f"objectives; got {k}")
    if rescale not in {0, 1, 2}:
        raise ValueError("cagrad rescale must be 0, 1, or 2")
    c_tensor = tf.cast(c, solve_dtype)
    if float(c) < 0.0:
        raise ValueError("cagrad c must be nonnegative")
    if float(c) >= 1.0 and not allow_outside_fixed_step_theorem:
        raise ValueError(
            "cagrad c >= 1 is outside the CAGrad fixed-step average-loss "
            "theorem scope 0 <= c < 1. Set "
            "allow_outside_fixed_step_theorem=True only for an explicit "
            "bounded runtime/non-theorem opt-in.")

    gram = tf.linalg.matmul(flat_solve, flat_solve, transpose_b=True)
    gram = 0.5 * (gram + tf.transpose(gram))
    eps_tensor = tf.cast(max(float(eps), CAGRAD_REFERENCE_EPS), solve_dtype)
    uniform = tf.fill(
        [k], tf.cast(1, solve_dtype) / tf.cast(k, solve_dtype))
    g0_norm = tf.sqrt(tf.maximum(
        tf.tensordot(uniform, tf.linalg.matvec(gram, uniform), axes=1),
        tf.cast(0, solve_dtype)) + eps_tensor)
    (
        alpha,
        subproblem_value,
        residual,
        active_size,
        residual_components,
    ) = _solve_cagrad_alpha(
        gram, uniform, c_tensor * g0_norm + eps_tensor, k, solve_dtype,
        eps_tensor, tf.cast(stationarity_tol, solve_dtype))
    gw = tf.linalg.matvec(flat_solve, alpha, transpose_a=True)
    gw_norm = tf.linalg.norm(gw)
    lambda_ = (c_tensor * g0_norm + eps_tensor) / (gw_norm + eps_tensor)
    combined_unscaled = (
        tf.linalg.matvec(flat_solve, uniform, transpose_a=True)
        + lambda_ * gw
    )
    if rescale == 0:
        rescale_factor = tf.cast(1, solve_dtype)
    elif rescale == 1:
        rescale_factor = tf.cast(1, solve_dtype) + tf.square(c_tensor)
    else:
        rescale_factor = tf.cast(1, solve_dtype) + c_tensor
    combined = tf.cast(combined_unscaled / rescale_factor, dtype)
    update_coefficients = tf.cast((uniform + lambda_ * alpha) / rescale_factor,
                                  dtype)
    contract_pass = _stationarity_contract_pass(
        residual_components, tf.cast(stationarity_tol, solve_dtype),
        solve_dtype)
    with tf.control_dependencies([
        tf.debugging.assert_equal(
            contract_pass,
            True,
            message="CAGrad simplex subproblem residual exceeded tolerance"),
        tf.debugging.assert_all_finite(
            update_coefficients, "CAGrad coefficients must be finite"),
        tf.debugging.assert_all_finite(
            combined, "CAGrad combined gradient must be finite"),
    ]):
        update_coefficients = tf.identity(update_coefficients)
        combined = tf.identity(combined)
    return update_coefficients, combined, {
        "cagrad_c": c_tensor,
        "cagrad_rescale": tf.constant(int(rescale), tf.int32),
        "cagrad_reference_epsilon": eps_tensor,
        "cagrad_internal_solve_dtype": tf.constant(solve_dtype.name),
        "cagrad_subproblem_alpha": alpha,
        "cagrad_subproblem_value": subproblem_value,
        "cagrad_stationarity_residual": residual,
        "cagrad_active_stationarity_residual": (
            residual_components["active_residual"]),
        "cagrad_inactive_kkt_residual": (
            residual_components["inactive_residual"]),
        "cagrad_simplex_residual": residual_components["simplex_residual"],
        "cagrad_active_mean_relative_residual": (
            residual_components["active_mean_relative_residual"]),
        "cagrad_active_min_scale": (
            residual_components["active_min_scale"]),
        "cagrad_active_min_relative_residual": (
            residual_components["active_min_relative_residual"]),
        "cagrad_active_stationarity_ulps": (
            residual_components["active_stationarity_ulps"]),
        "cagrad_stationarity_contract_pass": contract_pass,
        "cagrad_active_set_size": active_size,
        "cagrad_lambda": lambda_,
        "cagrad_g0_norm": g0_norm,
        "cagrad_coefficients_are_update_coefficients": tf.constant(True),
        "cagrad_subproblem_converged": contract_pass,
    }


def _solve_cagrad_alpha(gram, uniform, coef, k, dtype, eps, stationarity_tol):
    candidates = []
    values = []
    active_sizes = []
    for size in range(1, k + 1):
        for active in itertools.combinations(range(k), size):
            alpha, feasible = _solve_active_set(gram, uniform, coef, active, k,
                                                dtype, eps, stationarity_tol)
            value = _objective(gram, uniform, coef, alpha, eps)
            candidates.append(alpha)
            values.append(tf.where(feasible, value, tf.cast(float("inf"), dtype)))
            active_sizes.append(size)
    candidate_tensor = tf.stack(candidates, axis=0)
    value_tensor = tf.stack(values, axis=0)
    finite_values = tf.math.is_finite(value_tensor)
    with tf.control_dependencies([
        tf.debugging.assert_equal(
            tf.reduce_any(finite_values),
            True,
            message="no feasible CAGrad active set found",
        )
    ]):
        best = tf.argmin(value_tensor, axis=0, output_type=tf.int32)
    alpha = tf.gather(candidate_tensor, best)
    residual_components = _kkt_components(gram, uniform, coef, alpha, eps)
    return (
        alpha,
        tf.gather(value_tensor, best),
        residual_components["max_residual"],
        tf.gather(tf.constant(active_sizes, dtype=tf.int32), best),
        residual_components,
    )


def _solve_active_set(gram, uniform, coef, active, k, dtype, eps,
                      stationarity_tol):
    active_idx = tf.constant(active, dtype=tf.int32)
    if len(active) == 1:
        alpha = tf.scatter_nd(
            indices=tf.reshape(active_idx, [-1, 1]),
            updates=tf.ones([1], dtype=dtype),
            shape=[k],
        )
        return alpha, tf.constant(True)

    g_sub_original = tf.gather(
        tf.gather(gram, active_idx, axis=0), active_idx, axis=1)
    p_sub = tf.gather(tf.linalg.matvec(gram, uniform), active_idx)
    ridge = eps * tf.eye(len(active), dtype=dtype)
    g_sub = g_sub_original + ridge

    ones = tf.ones([len(active)], dtype=dtype)
    solve_ones = tf.linalg.solve(g_sub, ones[:, None])[:, 0]
    solve_p = tf.linalg.solve(g_sub, p_sub[:, None])[:, 0]
    a = tf.tensordot(ones, solve_ones, axes=1)
    b = tf.tensordot(ones, solve_p, axes=1)
    c_val = tf.tensordot(p_sub, solve_p, axes=1)
    qa = a + eps * tf.square(a)
    qb = -2.0 * b - 2.0 * eps * a * b
    qc = c_val + eps * tf.square(b) - tf.square(coef)
    disc = tf.square(qb) - 4.0 * qa * qc
    roots_feasible = disc >= tf.cast(0, dtype)
    sqrt_disc = tf.sqrt(tf.maximum(disc, tf.cast(0, dtype)))
    roots = tf.stack([
        (-qb - sqrt_disc) / (2.0 * qa),
        (-qb + sqrt_disc) / (2.0 * qa),
    ])
    alphas = []
    values = []
    feasibles = []
    for mu in tf.unstack(roots):
        t = a * mu - b
        weights = tf.linalg.solve(
            g_sub, (mu * ones - p_sub)[:, None])[:, 0] / t
        alpha = tf.scatter_nd(
            indices=tf.reshape(active_idx, [-1, 1]),
            updates=weights,
            shape=[k],
        )
        residual_components = _kkt_components(
            gram, uniform, coef, alpha, eps)
        contract_pass = _stationarity_contract_pass(
            residual_components, stationarity_tol, dtype)
        feasible = (
            roots_feasible
            & (t > eps)
            & tf.reduce_all(tf.math.is_finite(weights))
            & tf.reduce_all(weights >= tf.cast(-1e-8, dtype))
            & (tf.abs(tf.reduce_sum(weights) - tf.cast(1, dtype))
               <= tf.cast(1e-5, dtype))
            & contract_pass
        )
        value = _objective(gram, uniform, coef, alpha, eps)
        alphas.append(tf.where(feasible, alpha, tf.zeros([k], dtype=dtype)))
        values.append(tf.where(feasible, value, tf.cast(float("inf"), dtype)))
        feasibles.append(feasible)
    value_tensor = tf.stack(values)
    best = tf.argmin(value_tensor, output_type=tf.int32)
    return (
        tf.gather(tf.stack(alphas), best),
        tf.reduce_any(tf.stack(feasibles)),
    )


def _active_objective_and_grad(gram, uniform, coef, alpha, eps):
    gram_alpha = tf.linalg.matvec(gram, alpha)
    linear_grad = tf.linalg.matvec(gram, uniform)
    quad = tf.tensordot(alpha, gram_alpha, axes=1) + eps
    sqrt_quad = tf.sqrt(tf.maximum(quad, eps))
    value = tf.tensordot(alpha, linear_grad, axes=1) + coef * sqrt_quad
    grad = linear_grad + coef * gram_alpha / sqrt_quad
    return value, grad


def _objective(gram, uniform, coef, alpha, eps):
    value, _ = _active_objective_and_grad(gram, uniform, coef, alpha, eps)
    return value


def _kkt_components(gram, uniform, coef, alpha, eps):
    _, grad = _active_objective_and_grad(gram, uniform, coef, alpha, eps)
    active = alpha > tf.cast(1e-8, alpha.dtype)
    active_grad = tf.boolean_mask(grad, active)
    active_mean = tf.cond(
        tf.size(active_grad) > 0,
        lambda: tf.reduce_mean(active_grad),
        lambda: tf.cast(float("inf"), alpha.dtype),
    )
    active_residual = tf.cond(
        tf.size(active_grad) > 0,
        lambda: tf.reduce_max(tf.abs(active_grad - active_mean)),
        lambda: tf.cast(float("inf"), alpha.dtype),
    )
    inactive_grad = tf.boolean_mask(grad, ~active)
    inactive_residual = tf.cond(
        tf.size(inactive_grad) > 0,
        lambda: tf.reduce_max(tf.maximum(
            active_mean - inactive_grad,
            tf.cast(0, alpha.dtype))),
        lambda: tf.cast(0, alpha.dtype),
    )
    simplex_residual = tf.abs(tf.reduce_sum(alpha) - tf.cast(1, alpha.dtype))
    max_residual = tf.reduce_max(tf.stack([
        active_residual,
        inactive_residual,
        simplex_residual,
    ]))
    active_mean_scale = tf.maximum(tf.abs(active_mean), tf.cast(1, alpha.dtype))
    active_mean_relative_residual = active_residual / active_mean_scale
    active_min_abs = tf.cond(
        tf.size(active_grad) > 0,
        lambda: tf.reduce_min(tf.abs(active_grad)),
        lambda: tf.cast(float("inf"), alpha.dtype),
    )
    active_min_scale = tf.maximum(active_min_abs, tf.cast(1, alpha.dtype))
    active_min_relative_residual = active_residual / active_min_scale
    spacing = tf.abs(
        tf.math.nextafter(
            active_mean, tf.cast(float("inf"), alpha.dtype)) - active_mean)
    safe_spacing = tf.where(
        spacing > tf.cast(0, alpha.dtype),
        spacing,
        tf.cast(float("inf"), alpha.dtype),
    )
    active_stationarity_ulps = active_residual / safe_spacing
    return {
        "active_residual": active_residual,
        "inactive_residual": inactive_residual,
        "simplex_residual": simplex_residual,
        "max_residual": max_residual,
        "active_mean": active_mean,
        "active_mean_relative_residual": active_mean_relative_residual,
        "active_min_scale": active_min_scale,
        "active_min_relative_residual": active_min_relative_residual,
        "active_stationarity_ulps": active_stationarity_ulps,
    }


def _kkt_residual(gram, uniform, coef, alpha, eps):
    return _kkt_components(
        gram, uniform, coef, alpha, eps)["max_residual"]


def _stationarity_contract_pass(components, stationarity_tol, dtype):
    absolute_active_pass = components["active_residual"] <= stationarity_tol
    relative_pass = (
        components["active_min_relative_residual"]
        <= tf.cast(CAGRAD_ACTIVE_MIN_RELATIVE_TOL, dtype)
    )
    inactive_pass = components["inactive_residual"] <= stationarity_tol
    simplex_pass = (
        components["simplex_residual"]
        <= tf.cast(CAGRAD_SIMPLEX_CONTRACT_TOL, dtype)
    )
    finite_contract = tf.reduce_all(tf.stack([
        tf.math.is_finite(components["max_residual"]),
        tf.math.is_finite(components["active_mean_relative_residual"]),
        tf.math.is_finite(components["active_min_scale"]),
        tf.math.is_finite(components["active_min_relative_residual"]),
        tf.math.is_finite(components["active_stationarity_ulps"]),
        tf.math.is_finite(components["inactive_residual"]),
        tf.math.is_finite(components["simplex_residual"]),
    ]))
    active_pass = absolute_active_pass | relative_pass
    return finite_contract & active_pass & inactive_pass & simplex_pass
