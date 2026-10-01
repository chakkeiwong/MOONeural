"""Fixed-shape tensor arithmetic; no host iteration, I/O or runtime callbacks."""

from __future__ import annotations

import tensorflow as tf

from .moo import (
    cagrad_impl,
    gradnorm_impl,
    mgda_impl,
    pcgrad_impl,
    weighted_normalized_sum_impl,
)
from .adam import packed_adam_impl


def _finite(value):
    return tf.reduce_all(tf.math.is_finite(value))


def elimination_stage_impl(matrix, rhs, valid, stage):
    dimension = matrix.shape[-1]
    if stage >= dimension:
        return matrix, rhs, valid
    coordinates = tf.range(dimension, dtype=tf.int32)
    scores = tf.where(coordinates[None, :] >= stage, tf.abs(matrix[:, :, stage]), -1.0)
    pivot_index = tf.argmax(scores, axis=1, output_type=tf.int32)
    permutation = tf.where(
        coordinates[None, :] == stage,
        pivot_index[:, None],
        tf.where(
            coordinates[None, :] == pivot_index[:, None], stage, coordinates[None, :]
        ),
    )
    matrix = tf.gather(matrix, permutation, batch_dims=1)
    rhs = tf.gather(rhs, permutation, batch_dims=1)
    pivot = matrix[:, stage, stage]
    valid = valid & tf.math.is_finite(pivot) & (tf.abs(pivot) > 0.0)
    safe_pivot = tf.where(tf.abs(pivot) > 0.0, pivot, 1.0)
    pivot_row = matrix[:, stage, :] / safe_pivot[:, None]
    pivot_rhs = rhs[:, stage, :] / safe_pivot[:, None]
    multiplier = tf.where(
        coordinates[None, :] == stage,
        tf.zeros_like(matrix[:, :, stage]),
        matrix[:, :, stage],
    )
    matrix = matrix - multiplier[:, :, None] * pivot_row[:, None, :]
    rhs = rhs - multiplier[:, :, None] * pivot_rhs[:, None, :]
    matrix = tf.where(
        coordinates[None, :, None] == stage, pivot_row[:, None, :], matrix
    )
    rhs = tf.where(coordinates[None, :, None] == stage, pivot_rhs[:, None, :], rhs)
    return matrix, rhs, valid


def fixed_linear_solve_impl(matrix, rhs):
    """Bounded pivoted Gauss-Jordan solve; singular candidates return nonfinite."""
    if matrix.shape[-1] is None or not 1 <= matrix.shape[-1] <= 9:
        raise ValueError("fixed solve supports dimensions 1..9")
    valid = tf.reduce_all(tf.math.is_finite(matrix), axis=[1, 2]) & tf.reduce_all(
        tf.math.is_finite(rhs), axis=[1, 2]
    )
    matrix, rhs, valid = elimination_stage_impl(matrix, rhs, valid, 0)
    matrix, rhs, valid = elimination_stage_impl(matrix, rhs, valid, 1)
    matrix, rhs, valid = elimination_stage_impl(matrix, rhs, valid, 2)
    matrix, rhs, valid = elimination_stage_impl(matrix, rhs, valid, 3)
    matrix, rhs, valid = elimination_stage_impl(matrix, rhs, valid, 4)
    matrix, rhs, valid = elimination_stage_impl(matrix, rhs, valid, 5)
    matrix, rhs, valid = elimination_stage_impl(matrix, rhs, valid, 6)
    matrix, rhs, valid = elimination_stage_impl(matrix, rhs, valid, 7)
    matrix, rhs, valid = elimination_stage_impl(matrix, rhs, valid, 8)
    return tf.where(valid[:, None, None], rhs, tf.constant(float("nan"), tf.float64))


def jacobi_round_impl(matrix, vectors, pairs):
    """One parallel round of disjoint symmetric Jacobi rotations."""
    dimension = matrix.shape[-1]
    left = tf.one_hot(pairs[:, 0], dimension, dtype=tf.float64)
    right = tf.one_hot(pairs[:, 1], dimension, dtype=tf.float64)
    diagonal_left = tf.einsum("ri,bij,rj->br", left, matrix, left)
    diagonal_right = tf.einsum("ri,bij,rj->br", right, matrix, right)
    offdiagonal = tf.einsum("ri,bij,rj->br", left, matrix, right)
    active = tf.abs(offdiagonal) > tf.constant(1.0e-30, tf.float64)
    denominator = tf.where(active, 2.0 * offdiagonal, tf.ones_like(offdiagonal))
    tau = (diagonal_right - diagonal_left) / denominator
    sign = tf.where(tau >= 0.0, tf.ones_like(tau), -tf.ones_like(tau))
    tangent = tf.where(
        active, sign / (tf.abs(tau) + tf.sqrt(1.0 + tf.square(tau))), 0.0
    )
    cosine = tf.math.rsqrt(1.0 + tf.square(tangent))
    sine = tangent * cosine
    rotation = (
        tf.eye(dimension, dtype=tf.float64)[None, :, :]
        + tf.einsum("br,ri,rj->bij", cosine - 1.0, left, left)
        + tf.einsum("br,ri,rj->bij", cosine - 1.0, right, right)
        + tf.einsum("br,ri,rj->bij", sine, left, right)
        - tf.einsum("br,ri,rj->bij", sine, right, left)
    )
    rotated = tf.matmul(tf.matmul(rotation, matrix, transpose_a=True), rotation)
    return 0.5 * (rotated + tf.transpose(rotated, [0, 2, 1])), tf.matmul(
        vectors, rotation
    )


def jacobi_sweep_impl(matrix, vectors, schedule):
    matrix, vectors = jacobi_round_impl(matrix, vectors, schedule[0])
    matrix, vectors = jacobi_round_impl(matrix, vectors, schedule[1])
    matrix, vectors = jacobi_round_impl(matrix, vectors, schedule[2])
    matrix, vectors = jacobi_round_impl(matrix, vectors, schedule[3])
    matrix, vectors = jacobi_round_impl(matrix, vectors, schedule[4])
    matrix, vectors = jacobi_round_impl(matrix, vectors, schedule[5])
    return jacobi_round_impl(matrix, vectors, schedule[6])


def fixed_symmetric_pinv_impl(matrix, schedule, rcond):
    """Eight fixed sweeps on padded 8x8 matrices, with a convergence veto."""
    original = matrix
    vectors = tf.broadcast_to(tf.eye(8, dtype=tf.float64), tf.shape(matrix))
    matrix, vectors = jacobi_sweep_impl(matrix, vectors, schedule)
    matrix, vectors = jacobi_sweep_impl(matrix, vectors, schedule)
    matrix, vectors = jacobi_sweep_impl(matrix, vectors, schedule)
    matrix, vectors = jacobi_sweep_impl(matrix, vectors, schedule)
    matrix, vectors = jacobi_sweep_impl(matrix, vectors, schedule)
    matrix, vectors = jacobi_sweep_impl(matrix, vectors, schedule)
    matrix, vectors = jacobi_sweep_impl(matrix, vectors, schedule)
    matrix, vectors = jacobi_sweep_impl(matrix, vectors, schedule)
    eigenvalues = tf.linalg.diag_part(matrix)
    cutoff = rcond * tf.reduce_max(tf.abs(eigenvalues), axis=1, keepdims=True)
    retained = tf.abs(eigenvalues) > cutoff
    reciprocal = tf.where(
        retained, tf.math.reciprocal(tf.where(retained, eigenvalues, 1.0)), 0.0
    )
    inverse = tf.matmul(vectors * reciprocal[:, None, :], vectors, transpose_b=True)
    reconstruction = tf.matmul(
        vectors * eigenvalues[:, None, :], vectors, transpose_b=True
    )
    residual = tf.reduce_max(tf.abs(original - reconstruction), axis=[1, 2])
    scale = tf.maximum(tf.reduce_max(tf.abs(original), axis=[1, 2]), 1.0)
    converged = _finite(inverse) & tf.reduce_all(
        residual <= tf.constant(1.0e-13, tf.float64) * scale
    )
    spectral_scale = tf.reduce_max(tf.abs(eigenvalues), axis=1)
    reconstruction_bound = tf.reduce_max(
        tf.reduce_sum(tf.abs(original - reconstruction), axis=2), axis=1
    )
    rounding_bound = (
        tf.constant(64.0 * 2.220446049250313e-16, tf.float64) * spectral_scale
    )
    uncertainty = (1.0 + rcond) * reconstruction_bound + rounding_bound
    rank_separated = tf.reduce_all(
        tf.abs(tf.abs(eigenvalues) - cutoff) > uncertainty[:, None], axis=1
    )
    rank_separated = rank_separated | (spectral_scale == 0.0)
    converged = converged & tf.reduce_all(rank_separated)
    return inverse, residual, converged


def project_cone_impl(direction, protected_rows, subset_masks, schedule, rcond):
    """Source-order pinv candidates, including unprojected and zero directions."""
    constraint_count = protected_rows.shape[0]
    if constraint_count == 0:
        return (
            direction,
            tf.zeros([0], tf.float64),
            tf.constant(0.0, tf.float64),
            _finite(direction),
        )
    norms = tf.linalg.norm(protected_rows, axis=1, keepdims=True)
    inputs_valid = (
        _finite(direction)
        & _finite(protected_rows)
        & _finite(norms)
        & tf.reduce_all(norms > 0.0)
    )
    normalized = protected_rows / tf.where(norms > 0.0, norms, 1.0)
    normalized = tf.where(tf.math.is_finite(normalized), normalized, 0.0)
    safe_direction = tf.where(tf.math.is_finite(direction), direction, 0.0)
    gram = tf.matmul(normalized, normalized, transpose_b=True)
    masks = tf.cast(subset_masks, tf.float64)
    selected_gram = gram[None, :, :] * masks[:, :, None] * masks[:, None, :]
    padded_gram = tf.pad(
        selected_gram, [[0, 0], [0, 8 - constraint_count], [0, 8 - constraint_count]]
    )
    inverse, residual, converged = fixed_symmetric_pinv_impl(
        padded_gram, schedule, rcond
    )
    inverse = inverse[:, :constraint_count, :constraint_count]
    rhs = -tf.linalg.matvec(normalized, safe_direction)[None, :] * masks
    multipliers = tf.linalg.matvec(inverse, rhs) * masks
    candidates = safe_direction[None, :] + tf.matmul(multipliers, normalized)
    candidates = tf.concat(
        [safe_direction[None, :], candidates, tf.zeros_like(safe_direction)[None, :]],
        axis=0,
    )
    directional = tf.matmul(candidates, normalized, transpose_b=True)
    feasible = tf.reduce_all(directional >= tf.constant(-1.0e-10, tf.float64), axis=1)
    distance = tf.reduce_sum(tf.square(candidates - safe_direction[None, :]), axis=1)
    scores = tf.where(feasible, distance, tf.constant(float("inf"), tf.float64))
    selected = tf.gather(candidates, tf.argmin(scores, output_type=tf.int32))
    valid = inputs_valid & converged & _finite(selected) & _finite(distance)
    return (
        selected,
        tf.linalg.matvec(normalized, selected),
        tf.reduce_max(residual),
        valid,
    )


def method_direction_impl(
    rows,
    losses,
    permutations,
    method_weights,
    initial_losses,
    method_step,
    method,
    objective_count,
    parameter_dim,
    reset_method_state,
):
    """Reuse strict MOO arithmetic with explicit source-profile qualifications."""
    next_weights, next_initial, next_step = method_weights, initial_losses, method_step
    if method == "weighted_normalized_sum":
        coefficients, direction, _diagnostics, valid = weighted_normalized_sum_impl(
            rows,
            objective_count,
            parameter_dim,
            tf.constant(1.0e-14, tf.float64),
            tf.constant(1.0e-12, tf.float64),
            tf.constant(100.0, tf.float64),
            tf.constant(1.0e-300, tf.float64),
        )
    elif method == "pcgrad":
        coefficients, direction, _diagnostics, valid = pcgrad_impl(
            rows,
            permutations,
            objective_count,
            parameter_dim,
            tf.constant(1.0e-12, tf.float64),
        )
        valid = valid & tf.reduce_all(
            tf.sort(permutations, axis=1) == tf.range(objective_count)[None, :]
        )
    elif method == "mgda":
        coefficients, direction, _diagnostics, valid = mgda_impl(
            rows,
            objective_count,
            parameter_dim,
            tf.constant(1.0e-12, tf.float64),
            fixed_linear_solve_impl,
        )
    elif method == "cagrad":
        coefficients, direction, _diagnostics, valid = cagrad_impl(
            rows,
            objective_count,
            parameter_dim,
            tf.constant(0.5, tf.float64),
            1,
            tf.constant(1.0e-12, tf.float64),
            fixed_linear_solve_impl,
        )
    elif method == "gradnorm":
        if reset_method_state:
            method_weights = tf.ones([objective_count], tf.float64)
            initial_losses = losses
            method_step = tf.constant(0, tf.int64)
        coefficients, direction, diagnostics, valid = gradnorm_impl(
            rows,
            losses,
            method_weights,
            initial_losses,
            method_step,
            objective_count,
            parameter_dim,
            tf.constant(1.5, tf.float64),
            tf.cast(tf.constant(1.0e-4, tf.float32), tf.float64),
            tf.cast(tf.constant(1.0e-8, tf.float32), tf.float64),
            floor_mean_ratio=False,
        )
        next_weights, next_initial, next_step = diagnostics[:3]
        valid = valid & (
            tf.reduce_max(next_weights)
            / tf.maximum(tf.reduce_min(next_weights), tf.constant(1.0e-300, tf.float64))
            <= 100.0
        )
    else:
        raise ValueError("unsupported method specialization")
    valid = (
        valid
        & _finite(losses)
        & tf.reduce_all(losses >= 0.0)
        & _finite(direction)
        & _finite(coefficients)
    )
    valid = valid & (tf.linalg.norm(direction) > tf.constant(1.0e-12, tf.float64))
    return coefficients, direction, next_weights, next_initial, next_step, valid


def projected_adam_impl(
    parameters,
    first_moment,
    second_moment,
    iteration,
    direction,
    protected_rows,
    learning_rate,
    subset_masks,
    schedule,
    clip_norm,
    beta1,
    beta2,
    epsilon,
    epsilon_coordinate,
    power_beta1,
    power_beta2,
    *,
    projector_impl=project_cone_impl,
    packed_impl=packed_adam_impl,
):
    descent, submitted, first_residual, direction_valid = projector_impl(
        direction,
        protected_rows,
        subset_masks,
        schedule,
        tf.constant(1.0e-12, tf.float64),
    )
    effective_epsilon = epsilon
    next_iteration = tf.cast(iteration + 1, tf.float64)
    correction1 = 1.0 - tf.pow(beta1, next_iteration)
    correction2 = 1.0 - tf.pow(beta2, next_iteration)
    source_correction1 = 1.0 - tf.pow(power_beta1, next_iteration)
    source_correction2 = 1.0 - tf.pow(power_beta2, next_iteration)
    effective_rate = (
        learning_rate
        * correction1
        / source_correction1
        * tf.sqrt(source_correction2 / correction2)
    )
    if epsilon_coordinate == "keras_epsilon_hat":
        effective_epsilon = epsilon / tf.sqrt(
            tf.maximum(correction2, tf.constant(1.0e-300, tf.float64))
        )
    proposal = packed_impl(
        parameters,
        first_moment,
        second_moment,
        iteration,
        descent,
        effective_rate,
        clip_norm,
        beta1,
        beta2,
        effective_epsilon,
    )
    proposed_delta = proposal[0] - parameters
    projected_step, _submitted_step, second_residual, step_valid = projector_impl(
        -proposed_delta,
        protected_rows,
        subset_masks,
        schedule,
        tf.constant(1.0e-12, tf.float64),
    )
    candidate = parameters - projected_step
    committed_delta = candidate - parameters
    norms = tf.linalg.norm(protected_rows, axis=1, keepdims=True)
    normalized = protected_rows / tf.where(norms > 0.0, norms, 1.0)
    actual = tf.linalg.matvec(normalized, committed_delta)
    valid = (
        direction_valid
        & step_valid
        & proposal[7]
        & _finite(candidate)
        & _finite(actual)
    )
    valid = (
        valid
        & tf.reduce_all(second_moment >= 0.0)
        & (iteration < tf.constant(9223372036854775807, tf.int64))
    )
    valid = (
        valid
        & tf.reduce_all(submitted >= tf.constant(-1.0e-10, tf.float64))
        & tf.reduce_all(actual <= tf.constant(1.0e-10, tf.float64))
    )
    return (
        tf.where(valid, candidate, parameters),
        tf.where(valid, proposal[1], first_moment),
        tf.where(valid, proposal[2], second_moment),
        tf.where(valid, proposal[3], iteration),
        descent,
        proposed_delta,
        tf.where(valid, committed_delta, tf.zeros_like(parameters)),
        submitted,
        actual,
        tf.maximum(first_residual, second_residual),
        valid,
    )


def evaluate_impl(
    adapter,
    parameters,
    features,
    targets,
    sample_weights,
    training_factors,
    update_index,
    task_count,
    parameter_dim,
    hard_check_count,
):
    raw_values, raw_rows, hard_max, domain_valid, connected = (
        adapter.compute_task_tensors(
            parameters,
            features,
            targets,
            sample_weights,
            update_index,
        )
    )
    raw_values = tf.ensure_shape(raw_values, [task_count])
    raw_rows = tf.ensure_shape(raw_rows, [task_count, parameter_dim])
    hard_max = tf.ensure_shape(hard_max, [hard_check_count])
    connected = tf.ensure_shape(connected, [task_count])
    values = raw_values * training_factors
    rows = raw_rows * training_factors[:, None]
    valid = tf.reshape(domain_valid, []) & tf.reduce_all(connected)
    valid = (
        valid
        & _finite(parameters)
        & _finite(features)
        & _finite(targets)
        & _finite(sample_weights)
    )
    valid = valid & _finite(training_factors) & tf.reduce_all(training_factors > 0.0)
    valid = (
        valid
        & _finite(values)
        & _finite(rows)
        & _finite(hard_max)
        & (update_index >= 0)
    )
    valid = valid & tf.reduce_all(raw_values >= 0.0)
    return raw_values, raw_rows, values, rows, hard_max, connected, valid
