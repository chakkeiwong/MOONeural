"""Strict fixed-shape TensorFlow/XLA multi-objective directions.

This module contains only numeric tensor algebra.  Method selection, failure
classification, learning-rate policy, and persistent controller state belong
to the host.  Every method returns a common numeric tuple:

``(coefficients, combined_gradient, diagnostics, valid)``

where ``diagnostics`` is itself a tuple of tensors (never a string-bearing
mapping).  The active-objective count is fixed when the functions are built;
the implementation supports one through eight rows without Python iteration.
"""

from __future__ import annotations

import tensorflow as tf


MAX_OBJECTIVES = 8
DEFAULT_EPS = 1.0e-12
DEFAULT_CAGRAD_C = 0.5
DEFAULT_CAGRAD_RESCALE = 1
DEFAULT_GRADNORM_ALPHA = 1.5
DEFAULT_GRADNORM_EPS = 1.0e-8
DEFAULT_DORMANT_ABSOLUTE = 1.0e-14
DEFAULT_DORMANT_RELATIVE = 1.0e-12
DEFAULT_LAMBDA_MAX_RATIO = 100.0

METHOD_WEIGHTED = 0
METHOD_CAGRAD = 1
METHOD_PCGRAD = 2
METHOD_MGDA = 3
METHOD_GRADNORM = 4


def _shape_checked(gradients, objective_count, parameter_dim):
    flat = tf.ensure_shape(gradients, [objective_count, parameter_dim])
    return flat


def _finite_matrix(values):
    return tf.reduce_all(tf.math.is_finite(values))


def _subset_masks(objective_count):
    """Return all nonempty active subsets as a fixed tensor."""

    codes = tf.range(1, 1 << objective_count, dtype=tf.int32)
    bits = tf.range(objective_count, dtype=tf.int32)
    return tf.not_equal(
        tf.bitwise.bitwise_and(codes[:, None], tf.bitwise.left_shift(1, bits[None, :])),
        0,
    )


def weighted_normalized_sum_impl(
    gradients,
    objective_count,
    parameter_dim,
    dormant_absolute,
    dormant_relative,
    lambda_max_ratio,
    eps,
):
    """Return the inverse-norm direction with an explicit dormant-row veto."""

    flat = _shape_checked(gradients, objective_count, parameter_dim)
    dtype = flat.dtype
    norms = tf.linalg.norm(flat, axis=1)
    maximum = tf.reduce_max(norms)
    cutoff = tf.maximum(
        tf.cast(dormant_absolute, dtype),
        tf.cast(dormant_relative, dtype) * maximum,
    )
    active = norms > cutoff
    active_count = tf.reduce_sum(tf.cast(active, dtype))
    floor = maximum / tf.cast(lambda_max_ratio, dtype)
    safe_norms = tf.maximum(
        tf.maximum(norms, floor), tf.cast(eps, dtype)
    )
    inverse = tf.where(
        active,
        tf.math.reciprocal(safe_norms),
        tf.zeros([objective_count], dtype),
    )
    denominator = tf.reduce_sum(inverse)
    safe_denominator = tf.maximum(denominator, tf.cast(eps, dtype))
    coefficients = tf.where(
        active_count > tf.zeros([], dtype),
        active_count * inverse / safe_denominator,
        tf.zeros([objective_count], dtype),
    )
    combined = tf.linalg.matvec(flat, coefficients, transpose_a=True)
    valid = (
        _finite_matrix(flat)
        & tf.reduce_all(tf.math.is_finite(norms))
        & tf.math.is_finite(maximum)
        & tf.reduce_all(tf.math.is_finite(coefficients))
        & tf.reduce_all(tf.math.is_finite(combined))
        & (active_count > tf.zeros([], dtype))
    )
    return (
        coefficients,
        combined,
        (active, norms, cutoff, denominator, active_count),
        valid,
    )


def _pc_stage(adjusted, original, stage, order, objective_count, eps):
    indices = tf.range(objective_count, dtype=tf.int32)
    partners = tf.gather(original, order)
    dots = tf.reduce_sum(adjusted * partners, axis=1)
    denominators = tf.reduce_sum(tf.square(partners), axis=1)
    safe_denominators = tf.where(
        denominators > eps, denominators, tf.ones_like(denominators)
    )
    should_project = (
        tf.not_equal(order, indices)
        & (dots < tf.zeros([], adjusted.dtype))
        & (denominators > eps)
    )
    zero_skip = (
        tf.not_equal(order, indices)
        & (dots < tf.zeros([], adjusted.dtype))
        & tf.logical_not(denominators > eps)
    )
    candidate = adjusted - (
        dots / safe_denominators
    )[:, None] * partners
    next_adjusted = tf.where(should_project[:, None], candidate, adjusted)
    next_adjusted = tf.where(
        tf.cast(stage < objective_count, tf.bool), next_adjusted, adjusted
    )
    return (
        next_adjusted,
        tf.reduce_sum(tf.cast(should_project, tf.int32)),
        tf.reduce_sum(tf.cast(zero_skip, tf.int32)),
        order,
    )


def pcgrad_impl(gradients, permutations, objective_count, parameter_dim, eps):
    """Apply supplied deterministic PCGrad permutations with eight stages.

    The permutation table is generated at the host boundary with the legacy
    stateless-shuffle convention.  Supplying it as an integer tensor keeps the
    compiled numerical graph free of the unsupported ``StatelessShuffle`` op.
    """

    flat = _shape_checked(gradients, objective_count, parameter_dim)
    permutations = tf.ensure_shape(permutations, [objective_count, objective_count])
    expected = tf.broadcast_to(
        tf.range(objective_count, dtype=tf.int32)[None, :],
        [objective_count, objective_count],
    )
    valid_permutations = tf.reduce_all(
        tf.equal(tf.sort(permutations, axis=1), expected)
    )
    padded = tf.concat(
        [permutations, tf.zeros([objective_count, 8 - objective_count], tf.int32)],
        axis=1,
    )
    adjusted0, count0, skip0, order0 = _pc_stage(
        flat, flat, 0, padded[:, 0], objective_count, eps
    )
    adjusted1, count1, skip1, order1 = _pc_stage(
        adjusted0, flat, 1, padded[:, 1], objective_count, eps
    )
    adjusted2, count2, skip2, order2 = _pc_stage(
        adjusted1, flat, 2, padded[:, 2], objective_count, eps
    )
    adjusted3, count3, skip3, order3 = _pc_stage(
        adjusted2, flat, 3, padded[:, 3], objective_count, eps
    )
    adjusted4, count4, skip4, order4 = _pc_stage(
        adjusted3, flat, 4, padded[:, 4], objective_count, eps
    )
    adjusted5, count5, skip5, order5 = _pc_stage(
        adjusted4, flat, 5, padded[:, 5], objective_count, eps
    )
    adjusted6, count6, skip6, order6 = _pc_stage(
        adjusted5, flat, 6, padded[:, 6], objective_count, eps
    )
    adjusted7, count7, skip7, order7 = _pc_stage(
        adjusted6, flat, 7, padded[:, 7], objective_count, eps
    )
    # Only the first m stages are semantically active.  The extra fixed stages
    # make the graph shape-independent for every supported m <= 8.
    active_stages = tf.range(8, dtype=tf.int32) < objective_count
    counts = tf.stack([count0, count1, count2, count3, count4, count5, count6, count7])
    skips = tf.stack([skip0, skip1, skip2, skip3, skip4, skip5, skip6, skip7])
    count = tf.reduce_sum(tf.where(active_stages, counts, tf.zeros_like(counts)))
    skip = tf.reduce_sum(tf.where(active_stages, skips, tf.zeros_like(skips)))
    orders = tf.stack([order0, order1, order2, order3, order4, order5, order6, order7])
    orders = tf.transpose(orders, [1, 0])
    orders = orders[:, :objective_count]
    coefficients = tf.ones([objective_count], flat.dtype)
    combined = tf.reduce_sum(adjusted7, axis=0)
    valid = (
        _finite_matrix(flat)
        & _finite_matrix(adjusted7)
        & tf.math.is_finite(combined)
        & valid_permutations
    )
    valid = tf.reduce_all(valid)
    return coefficients, combined, (orders, count, skip), valid


def _batched_kkt(gram, masks, rhs_top, eps, linear_solve=None):
    """Solve all masked simplex KKT systems in parallel."""

    dtype = gram.dtype
    mask_float = tf.cast(masks, dtype)
    outer = mask_float[:, :, None] * mask_float[:, None, :]
    eye = tf.eye(tf.shape(gram)[0], dtype=dtype)
    top_left = 2.0 * gram[None, :, :] * outer + tf.linalg.diag(
        1.0 - mask_float
    )
    top_right = mask_float[:, :, None]
    bottom_left = mask_float[:, None, :]
    bottom_right = tf.zeros([tf.shape(masks)[0], 1, 1], dtype=dtype)
    matrix = tf.concat(
        [
            tf.concat([top_left, top_right], axis=2),
            tf.concat([bottom_left, bottom_right], axis=2),
        ],
        axis=1,
    )
    rhs = tf.concat(
        [rhs_top, tf.ones([tf.shape(masks)[0], 1], dtype=dtype)], axis=1
    )
    # Inactive diagonal entries make every padded system nonsingular while
    # leaving the active KKT equations unchanged.
    solve = tf.linalg.solve if linear_solve is None else linear_solve
    solution = solve(matrix, rhs[:, :, None])[:, :, 0]
    alpha = solution[:, :-1] * mask_float
    residual = tf.linalg.matvec(matrix, solution) - rhs
    # Do not use ``tf.linalg.svd`` here.  On XLA CPU its Jacobi
    # implementation lowers to an HLO ``while`` computation, violating the
    # strict fixed-shape kernel boundary.  A solve that is finite and closes
    # its KKT residual is the relevant acceptance test for this active-set
    # candidate; the inactive identity rows keep the padded systems stable.
    residual_norm = tf.linalg.norm(residual, axis=1)
    nonsingular = (
        tf.reduce_all(tf.math.is_finite(solution), axis=1)
        & tf.math.is_finite(residual_norm)
        & (residual_norm <= tf.cast(1.0e-7, dtype))
    )
    return alpha, residual, nonsingular


def _mgda_feasible(gram, masks, alpha, residual, nonsingular, eps):
    dtype = gram.dtype
    mask_float = tf.cast(masks, dtype)
    sums = tf.reduce_sum(alpha, axis=1)
    grad = 2.0 * tf.einsum("ij,nj->ni", gram, alpha)
    active_count = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
    active_mean = tf.reduce_sum(grad * mask_float, axis=1) / tf.cast(
        active_count, dtype
    )
    inactive_gap = tf.where(
        masks,
        tf.zeros_like(grad),
        tf.maximum(active_mean[:, None] - grad, tf.zeros_like(grad)),
    )
    return (
        nonsingular
        & tf.reduce_all(tf.math.is_finite(alpha), axis=1)
        & tf.reduce_all(tf.math.is_finite(residual), axis=1)
        & (tf.abs(sums - tf.ones_like(sums)) <= tf.cast(1.0e-7, dtype))
        & tf.reduce_all(tf.where(masks, alpha >= tf.cast(-1.0e-8, dtype), True), axis=1)
        & (tf.reduce_max(inactive_gap, axis=1) <= tf.cast(1.0e-7, dtype))
    )


def mgda_impl(gradients, objective_count, parameter_dim, eps, linear_solve=None):
    """Solve the bounded MGDA simplex problem by a batched active-set table."""

    flat = _shape_checked(gradients, objective_count, parameter_dim)
    gram = tf.linalg.matmul(flat, flat, transpose_b=True)
    gram = 0.5 * (gram + tf.transpose(gram))
    masks = _subset_masks(objective_count)
    rhs_top = tf.zeros([tf.shape(masks)[0], objective_count], dtype=flat.dtype)
    alpha_all, residual, nonsingular = _batched_kkt(gram, masks, rhs_top, eps, linear_solve)
    feasible = _mgda_feasible(gram, masks, alpha_all, residual, nonsingular, eps)
    values = tf.einsum("ni,ij,nj->n", alpha_all, gram, alpha_all)
    values = tf.where(feasible, values, tf.fill(tf.shape(values), tf.constant(float("inf"), flat.dtype)))
    best = tf.argmin(values, axis=0, output_type=tf.int32)
    alpha = tf.gather(alpha_all, best)
    best_value = tf.gather(values, best)
    has_solution = tf.reduce_any(feasible)
    safe_alpha = tf.where(has_solution, alpha, tf.zeros_like(alpha))
    combined = tf.linalg.matvec(flat, safe_alpha, transpose_a=True)
    valid = (
        _finite_matrix(flat)
        & has_solution
        & tf.reduce_all(tf.math.is_finite(safe_alpha))
        & tf.reduce_all(tf.math.is_finite(combined))
    )
    return safe_alpha, combined, (best_value, tf.reduce_sum(tf.cast(feasible, tf.int32))), valid


def _cagrad_candidates(gram, masks, coefficient, eps, linear_solve=None):
    dtype = gram.dtype
    mask_float = tf.cast(masks, dtype)
    outer = mask_float[:, :, None] * mask_float[:, None, :]
    # The reference CAGrad solver floors its working epsilon at 1e-8 and
    # adds that ridge only on active coordinates.  Keep the same numerical
    # problem while evaluating every subset in one fixed tensor batch.
    effective_eps = tf.maximum(eps, tf.cast(1.0e-8, dtype))
    active_eye = tf.linalg.diag(mask_float)
    g_sub = (
        gram[None, :, :] * outer
        + tf.linalg.diag(1.0 - mask_float)
        + effective_eps * active_eye
    )
    ones = mask_float
    p = tf.linalg.matvec(gram, tf.ones([tf.shape(gram)[0]], dtype=dtype) / tf.cast(tf.shape(gram)[0], dtype))
    p_sub = p[None, :] * mask_float
    solve = tf.linalg.solve if linear_solve is None else linear_solve
    solve_ones = solve(g_sub, ones[:, :, None])[:, :, 0]
    solve_p = solve(g_sub, p_sub[:, :, None])[:, :, 0]
    a = tf.reduce_sum(ones * solve_ones, axis=1)
    b = tf.reduce_sum(ones * solve_p, axis=1)
    c_value = tf.reduce_sum(p_sub * solve_p, axis=1)
    qa = a + effective_eps * tf.square(a)
    qb = -2.0 * b - 2.0 * effective_eps * a * b
    qc = c_value + effective_eps * tf.square(b) - tf.square(coefficient)
    discriminant = tf.square(qb) - 4.0 * qa * qc
    root_ok = discriminant >= tf.zeros([], dtype)
    safe_qa = tf.where(tf.abs(qa) > effective_eps, qa, tf.ones_like(qa))
    sqrt_discriminant = tf.sqrt(tf.maximum(discriminant, tf.zeros([], dtype)))
    roots = tf.stack(
        [
            (-qb - sqrt_discriminant) / (2.0 * safe_qa),
            (-qb + sqrt_discriminant) / (2.0 * safe_qa),
        ],
        axis=1,
    )
    rhs0 = roots[:, 0:1] * ones - p_sub
    rhs1 = roots[:, 1:2] * ones - p_sub
    weights0 = solve(g_sub, rhs0[:, :, None])[:, :, 0]
    weights1 = solve(g_sub, rhs1[:, :, None])[:, :, 0]
    t0 = a * roots[:, 0] - b
    t1 = a * roots[:, 1] - b
    alpha0 = weights0 / tf.where(
        tf.abs(t0) > effective_eps, t0, tf.ones_like(t0)
    )[:, None]
    alpha1 = weights1 / tf.where(
        tf.abs(t1) > effective_eps, t1, tf.ones_like(t1)
    )[:, None]
    alpha0 = alpha0 * mask_float
    alpha1 = alpha1 * mask_float
    alpha_single = mask_float
    size = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
    simplex0 = tf.abs(tf.reduce_sum(alpha0, axis=1) - 1.0) <= 1.0e-6
    simplex1 = tf.abs(tf.reduce_sum(alpha1, axis=1) - 1.0) <= 1.0e-6
    root_feasible0 = (
        root_ok
        & (t0 > effective_eps)
        & tf.reduce_all(tf.math.is_finite(alpha0), axis=1)
        & tf.reduce_all(tf.where(masks, alpha0 >= -1.0e-8, True), axis=1)
        & simplex0
    )
    root_feasible1 = (
        root_ok
        & (t1 > effective_eps)
        & tf.reduce_all(tf.math.is_finite(alpha1), axis=1)
        & tf.reduce_all(tf.where(masks, alpha1 >= -1.0e-8, True), axis=1)
        & simplex1
    )
    alpha0 = tf.where((size == 1)[:, None], alpha_single, alpha0)
    alpha1 = tf.where((size == 1)[:, None], alpha_single, alpha1)
    root_feasible0 = tf.where(size == 1, tf.ones_like(root_feasible0), root_feasible0)
    root_feasible1 = tf.where(size == 1, tf.zeros_like(root_feasible1), root_feasible1)
    alpha_candidates = tf.stack([alpha0, alpha1], axis=1)
    feasible_candidates = tf.stack([root_feasible0, root_feasible1], axis=1)
    return alpha_candidates, feasible_candidates


def _cagrad_kkt_feasible(gram, uniform, coefficient, alpha, eps):
    dtype = gram.dtype
    mask = alpha > tf.cast(1.0e-8, dtype)
    q = tf.einsum("ni,ij,nj->n", alpha, gram, alpha)
    effective_eps = tf.maximum(eps, tf.cast(1.0e-8, dtype))
    denominator = tf.sqrt(tf.maximum(q + effective_eps, effective_eps))
    grad = tf.linalg.matvec(gram, uniform)[None, :] + coefficient * tf.einsum(
        "ij,nj->ni", gram, alpha
    ) / denominator[:, None]
    active_float = tf.cast(mask, dtype)
    active_count = tf.reduce_sum(active_float, axis=1)
    safe_count = tf.maximum(active_count, tf.ones_like(active_count))
    active_mean = tf.reduce_sum(grad * active_float, axis=1) / safe_count
    active_residual = tf.reduce_max(
        tf.where(mask, tf.abs(grad - active_mean[:, None]), tf.zeros_like(grad)),
        axis=1,
    )
    inactive_residual = tf.reduce_max(
        tf.where(
            mask,
            tf.zeros_like(grad),
            tf.maximum(active_mean[:, None] - grad, tf.zeros_like(grad)),
        ),
        axis=1,
    )
    active_min_abs = tf.reduce_min(
        tf.where(mask, tf.abs(grad), tf.fill(tf.shape(grad), tf.constant(float("inf"), dtype))),
        axis=1,
    )
    active_min_scale = tf.maximum(active_min_abs, tf.ones_like(active_min_abs))
    active_relative_residual = active_residual / active_min_scale
    simplex_residual = tf.abs(tf.reduce_sum(alpha, axis=1) - tf.ones_like(active_count))
    finite_contract = (
        tf.reduce_all(tf.math.is_finite(grad), axis=1)
        & tf.math.is_finite(active_residual)
        & tf.math.is_finite(inactive_residual)
        & tf.math.is_finite(simplex_residual)
        & tf.math.is_finite(active_relative_residual)
    )
    active_pass = (
        (active_residual <= tf.cast(1.0e-6, dtype))
        | (active_relative_residual <= tf.cast(1.0e-12, dtype))
    )
    inactive_violation = tf.where(
        mask,
        tf.zeros_like(grad),
        tf.maximum(active_mean[:, None] - grad, tf.zeros_like(grad)),
    )
    return (
        finite_contract
        & (active_count > tf.zeros_like(active_count))
        & active_pass
        & (tf.reduce_max(inactive_violation, axis=1) <= tf.cast(1.0e-6, dtype))
        & (simplex_residual <= tf.cast(1.0e-10, dtype))
    )


def cagrad_impl(
    gradients,
    objective_count,
    parameter_dim,
    c,
    rescale,
    eps,
    linear_solve=None,
):
    """Solve the bounded CAGrad active-set problem without host dispatch."""

    flat = _shape_checked(gradients, objective_count, parameter_dim)
    gram = tf.linalg.matmul(flat, flat, transpose_b=True)
    gram = 0.5 * (gram + tf.transpose(gram))
    dtype = flat.dtype
    uniform = tf.ones([objective_count], dtype=dtype) / tf.cast(objective_count, dtype)
    effective_eps = tf.maximum(eps, tf.cast(1.0e-8, dtype))
    g0_norm = tf.sqrt(
        tf.maximum(
            tf.tensordot(uniform, tf.linalg.matvec(gram, uniform), axes=1)
            + effective_eps,
            effective_eps,
        )
    )
    coefficient = tf.cast(c, dtype) * g0_norm + effective_eps
    masks = _subset_masks(objective_count)
    alpha_candidates, root_feasible = _cagrad_candidates(
        gram, masks, coefficient, eps, linear_solve
    )
    flat_alpha = tf.reshape(alpha_candidates, [-1, objective_count])
    kkt = _cagrad_kkt_feasible(
        gram,
        uniform,
        coefficient,
        flat_alpha,
        eps,
    )
    feasible = tf.reshape(root_feasible, [-1]) & kkt
    q = tf.einsum("ni,ij,nj->n", flat_alpha, gram, flat_alpha)
    effective_eps = tf.maximum(eps, tf.cast(1.0e-8, dtype))
    value = tf.einsum(
        "i,ni->n", tf.linalg.matvec(gram, uniform), flat_alpha
    ) + coefficient * tf.sqrt(tf.maximum(q + effective_eps, effective_eps))
    value = tf.where(
        feasible,
        value,
        tf.fill(tf.shape(value), tf.constant(float("inf"), dtype)),
    )
    best = tf.argmin(value, axis=0, output_type=tf.int32)
    alpha = tf.gather(flat_alpha, best)
    has_solution = tf.reduce_any(feasible)
    alpha = tf.where(has_solution, alpha, tf.zeros_like(alpha))
    gw = tf.linalg.matvec(flat, alpha, transpose_a=True)
    gw_norm = tf.linalg.norm(gw)
    lambda_value = coefficient / (
        gw_norm + tf.maximum(eps, tf.cast(1.0e-8, dtype))
    )
    if rescale == 0:
        factor = tf.ones([], dtype)
    elif rescale == 1:
        factor = 1.0 + tf.square(tf.cast(c, dtype))
    else:
        factor = 1.0 + tf.cast(c, dtype)
    base = tf.linalg.matvec(flat, uniform, transpose_a=True)
    combined = (base + lambda_value * gw) / factor
    coefficients = (uniform + lambda_value * alpha) / factor
    valid = (
        _finite_matrix(flat)
        & has_solution
        & tf.reduce_all(tf.math.is_finite(coefficients))
        & tf.reduce_all(tf.math.is_finite(combined))
    )
    return coefficients, combined, (lambda_value, g0_norm, tf.gather(value, best)), valid


def gradnorm_impl(
    gradients,
    losses,
    weights,
    initial_losses,
    step,
    objective_count,
    parameter_dim,
    alpha,
    learning_rate,
    eps,
    floor_mean_ratio=True,
):
    """Return one functional GradNorm direction and the next state."""

    flat = _shape_checked(gradients, objective_count, parameter_dim)
    dtype = flat.dtype
    loss_values = tf.ensure_shape(losses, [objective_count])
    current_weights = tf.ensure_shape(weights, [objective_count])
    prior_initial = tf.ensure_shape(initial_losses, [objective_count])
    finite_inputs = (
        _finite_matrix(flat)
        & tf.reduce_all(tf.math.is_finite(loss_values))
        & tf.reduce_all(tf.math.is_finite(current_weights))
        & tf.reduce_all(tf.math.is_finite(prior_initial))
    )
    nonnegative = tf.reduce_all(loss_values >= tf.zeros([], dtype))
    positive_weights = tf.reduce_all(current_weights > tf.zeros([], dtype))
    nonnegative_prior = tf.reduce_all(prior_initial >= tf.zeros([], dtype))
    initialized = tf.reduce_all(tf.equal(prior_initial, tf.zeros([], dtype)))
    initial = tf.where(initialized, loss_values, prior_initial)
    safe_initial = tf.maximum(initial, tf.cast(eps, dtype))
    positive_initial = tf.reduce_all(initial > tf.zeros([], dtype))
    nonnegative_step = tf.reshape(step, []) >= tf.constant(0, tf.int64)
    safe_losses = tf.maximum(loss_values, tf.zeros_like(loss_values))
    norms = tf.linalg.norm(flat, axis=1)
    weighted_norms = current_weights * norms
    ratios = safe_losses / safe_initial
    mean_ratio = tf.reduce_mean(ratios)
    if floor_mean_ratio:
        mean_ratio = tf.maximum(mean_ratio, tf.cast(eps, dtype))
    inverse_rates = ratios / mean_ratio
    targets = tf.reduce_mean(weighted_norms) * tf.pow(
        inverse_rates, tf.cast(alpha, dtype)
    )
    weight_gradient = tf.sign(weighted_norms - tf.stop_gradient(targets)) * norms
    updated = tf.maximum(
        current_weights - tf.cast(learning_rate, dtype) * weight_gradient,
        tf.cast(eps, dtype),
    )
    updated = updated * (tf.cast(objective_count, dtype) / tf.reduce_sum(updated))
    combined = tf.reduce_sum(current_weights[:, None] * flat, axis=0)
    next_step = tf.cast(step, tf.int64) + tf.constant(1, tf.int64)
    valid = (
        finite_inputs
        & nonnegative
        & nonnegative_prior
        & positive_weights
        & positive_initial
        & nonnegative_step
        & tf.reduce_all(tf.math.is_finite(initial))
        & tf.reduce_all(tf.math.is_finite(combined))
        & tf.reduce_all(tf.math.is_finite(updated))
    )
    coefficients = tf.where(valid, current_weights, tf.zeros_like(current_weights))
    combined = tf.where(valid, combined, tf.zeros_like(combined))
    updated = tf.where(valid, updated, tf.zeros_like(updated))
    return coefficients, combined, (updated, initial, next_step, norms, targets), valid


def make_moo_functions(
    *, objective_count: int, parameter_dim: int, cagrad_c: float = DEFAULT_CAGRAD_C,
    cagrad_rescale: int = DEFAULT_CAGRAD_RESCALE, eps: float = DEFAULT_EPS,
    gradnorm_learning_rate: float = 1.0e-4,
    dormant_absolute: float = DEFAULT_DORMANT_ABSOLUTE,
    dormant_relative: float = DEFAULT_DORMANT_RELATIVE,
    lambda_max_ratio: float = DEFAULT_LAMBDA_MAX_RATIO,
):
    """Create fixed-signature XLA functions for all five directions."""

    m = int(objective_count)
    d = int(parameter_dim)
    if m < 1 or m > MAX_OBJECTIVES or d < 1:
        raise ValueError("moo_shape_invalid")
    if cagrad_rescale not in (0, 1, 2):
        raise ValueError("sgu_xla_cagrad_rescale_invalid")
    if not (0.0 <= float(cagrad_c) < float("inf")):
        raise ValueError("moo_config_invalid")
    if not (0.0 < float(eps) < float("inf")):
        raise ValueError("moo_config_invalid")
    if not (0.0 < float(gradnorm_learning_rate) < float("inf")):
        raise ValueError("moo_config_invalid")
    if not (0.0 <= float(dormant_absolute) < float("inf")):
        raise ValueError("moo_config_invalid")
    if not (0.0 <= float(dormant_relative) < float("inf")):
        raise ValueError("moo_config_invalid")
    if not (1.0 <= float(lambda_max_ratio) < float("inf")):
        raise ValueError("moo_config_invalid")
    matrix_spec = tf.TensorSpec([m, d], tf.float64, name="task_gradients")
    vector_spec = tf.TensorSpec([m], tf.float64, name="task_losses")
    state_spec = tf.TensorSpec([m], tf.float64, name="method_state")
    step_spec = tf.TensorSpec([], tf.int64, name="method_step")
    permutation_spec = tf.TensorSpec(
        [m, m], tf.int32, name="pcgrad_permutations"
    )

    @tf.function(input_signature=(matrix_spec,), autograph=False, jit_compile=True)
    def weighted(gradients):
        return weighted_normalized_sum_impl(
            gradients,
            m,
            d,
            tf.constant(dormant_absolute, tf.float64),
            tf.constant(dormant_relative, tf.float64),
            tf.constant(lambda_max_ratio, tf.float64),
            tf.constant(eps, tf.float64),
        )

    @tf.function(input_signature=(matrix_spec,), autograph=False, jit_compile=True)
    def mgda(gradients):
        return mgda_impl(gradients, m, d, tf.constant(eps, tf.float64))

    @tf.function(input_signature=(matrix_spec,), autograph=False, jit_compile=True)
    def cagrad(gradients):
        return cagrad_impl(
            gradients,
            m,
            d,
            tf.constant(cagrad_c, tf.float64),
            cagrad_rescale,
            tf.constant(eps, tf.float64),
        )

    @tf.function(
        input_signature=(matrix_spec, permutation_spec),
        autograph=False,
        jit_compile=True,
    )
    def pcgrad(gradients, permutations):
        return pcgrad_impl(
            gradients, permutations, m, d, tf.cast(eps, tf.float64)
        )

    @tf.function(
        input_signature=(matrix_spec, vector_spec, state_spec, state_spec, step_spec),
        autograph=False,
        jit_compile=True,
    )
    def gradnorm(gradients, losses, weights, initial_losses, step):
        return gradnorm_impl(
            gradients,
            losses,
            weights,
            initial_losses,
            step,
            m,
            d,
            tf.constant(DEFAULT_GRADNORM_ALPHA, tf.float64),
            tf.constant(gradnorm_learning_rate, tf.float64),
            tf.constant(DEFAULT_GRADNORM_EPS, tf.float64),
        )

    return {
        "weighted_normalized_sum": weighted,
        "fixed_normalized_sum": weighted,
        "cagrad": cagrad,
        "pcgrad": pcgrad,
        "mgda": mgda,
        "gradnorm": gradnorm,
    }


__all__ = [
    "DEFAULT_CAGRAD_C",
    "DEFAULT_CAGRAD_RESCALE",
    "DEFAULT_EPS",
    "MAX_OBJECTIVES",
    "METHOD_CAGRAD",
    "METHOD_GRADNORM",
    "METHOD_MGDA",
    "METHOD_PCGRAD",
    "METHOD_WEIGHTED",
    "cagrad_impl",
    "gradnorm_impl",
    "make_moo_functions",
    "mgda_impl",
    "pcgrad_impl",
    "weighted_normalized_sum_impl",
]
