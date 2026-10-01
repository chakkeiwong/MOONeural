"""Versioned native TensorFlow cone and CAGrad solvers with optimality checks."""

from itertools import combinations

import tensorflow as tf

from .generic_execution_boundary import (
    DEFAULT_ADAM,
    _dimension,
    _make_projected_adam_function,
    make_method_function,
)
from .moo import _cagrad_kkt_feasible
from .adam import packed_adam_impl


def numerical_binding():
    return {
        "schema": "generic_neural_solver.certified_numerics.v1",
        "profile": "direct_svd_cone__affine_face_cagrad.v1",
        "dtype": "float64",
        "jit_compile": False,
        "projection": {"primal_atol": 1e-10, "kkt_scaled_atol": 1e-12,
                       "rank_candidates": ["sqrt_source_rcond", "64_float64_epsilon"],
                       "zero_rows": "redundant", "pre_and_post_adam": True},
        "cagrad": {"objective": "unridged_smoothed_norm", "epsilon": "source_float32_1e-8",
                   "final_update": "source_norm_plus_epsilon", "rescale": 1, "c": 0.5,
                   "affine_anchor": "minimum_row_norm"},
        "learning_rate": "source_float32_roundtrip",
        "packed_adam_jit_compile": True,
    }


def _finite(value):
    return tf.reduce_all(tf.math.is_finite(value))


def _normalized_rows(rows):
    maximum = tf.reduce_max(tf.abs(rows), axis=1, keepdims=True)
    scaled = rows / tf.where(maximum > 0.0, maximum, 1.0)
    norms = tf.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / tf.where(norms > 0.0, norms, 1.0)


def certified_projection_impl(direction, protected_rows, subset_masks, schedule, rcond):
    """Return the closest full-cone KKT-certified subset candidate, or invalid."""
    count = protected_rows.shape[0]
    if count == 0:
        return direction, tf.zeros([0], tf.float64), tf.constant(0.0, tf.float64), _finite(direction)
    inputs_valid = _finite(direction) & _finite(protected_rows)
    normalized = _normalized_rows(tf.where(tf.math.is_finite(protected_rows), protected_rows, 0.0))
    safe_direction = tf.where(tf.math.is_finite(direction), direction, 0.0)
    candidates = [safe_direction]
    duals = [tf.zeros([count], tf.float64)]
    for size in range(1, count + 1):
        for indices in combinations(range(count), size):
            selected = tf.gather(normalized, indices)
            singular, left, right = tf.linalg.svd(selected, full_matrices=False)
            coordinates = tf.linalg.matvec(right, safe_direction, transpose_a=True)
            for cutoff in (tf.sqrt(rcond), tf.constant(64 * 2.220446049250313e-16, tf.float64)):
                retained = singular > cutoff * tf.reduce_max(singular)
                retained_coordinates = tf.where(retained, coordinates, 0.0)
                candidate = safe_direction - tf.linalg.matvec(right, retained_coordinates)
                leakage = tf.where(retained, tf.linalg.matvec(right, candidate, transpose_a=True), 0.0)
                candidate -= tf.linalg.matvec(right, leakage)
                multiplier = -tf.linalg.matvec(left, retained_coordinates / tf.where(retained, singular, 1.0))
                dual = tf.linalg.matvec(tf.one_hot(indices, count, dtype=tf.float64), multiplier, transpose_a=True)
                candidates.append(candidate)
                duals.append(tf.maximum(dual, 0.0))
    candidates = tf.stack(candidates)
    duals = tf.stack(duals)
    products = tf.matmul(candidates, normalized, transpose_b=True)
    correction = tf.matmul(duals, normalized)
    correction_scale = tf.linalg.norm(tf.matmul(duals, tf.abs(normalized)), axis=1)
    stationarity = tf.linalg.norm(candidates - safe_direction - correction, axis=1) / (
        1.0 + tf.linalg.norm(safe_direction) + correction_scale
    )
    complementarity = tf.reduce_sum(tf.abs(duals * products), axis=1) / (
        1.0 + tf.reduce_sum(tf.square(safe_direction))
    )
    residual = tf.maximum(stationarity, complementarity)
    feasible = (
        tf.reduce_all(products >= tf.constant(-1e-10, tf.float64), axis=1)
        & (residual <= tf.constant(1e-12, tf.float64))
        & tf.reduce_all(tf.math.is_finite(candidates), axis=1)
        & tf.reduce_all(tf.math.is_finite(duals), axis=1)
    )
    distances = tf.reduce_sum(tf.square(candidates - safe_direction), axis=1)
    scores = tf.where(feasible, distances, tf.constant(float("inf"), tf.float64))
    best = tf.argmin(scores, output_type=tf.int32)
    valid = inputs_valid & tf.reduce_any(feasible) & tf.math.is_finite(scores[best])
    return candidates[best], products[best], residual[best], valid


def make_certified_projection_function(parameter_dim, constraint_count):
    _dimension("parameter_dim", parameter_dim, 1)
    _dimension("constraint_count", constraint_count, 0, 8)
    return tf.function(
        certified_projection_impl, autograph=False, jit_compile=False,
        input_signature=[
            tf.TensorSpec([parameter_dim], tf.float64),
            tf.TensorSpec([constraint_count, parameter_dim], tf.float64),
            tf.TensorSpec([(1 << constraint_count) - 1, constraint_count], tf.bool),
            tf.TensorSpec([7, 4, 2], tf.int32), tf.TensorSpec([], tf.float64),
        ],
    )


def certified_cagrad_impl(rows, c=0.5, rescale=1):
    """Solve the unridged smoothed objective over independent convex-hull faces."""
    count = rows.shape[0]
    epsilon = tf.cast(tf.constant(1e-8, tf.float32), tf.float64)
    uniform = tf.ones([count], tf.float64) / count
    inputs_valid = _finite(rows)
    safe_rows = tf.where(tf.math.is_finite(rows), rows, 0.0)
    base = tf.reduce_mean(safe_rows, axis=0)
    base_norm = tf.sqrt(tf.reduce_sum(tf.square(base)) + epsilon)
    coefficient = tf.cast(c, tf.float64) * base_norm + epsilon
    weights = [tf.one_hot(index, count, dtype=tf.float64) for index in range(count)]
    available = [tf.constant(True) for _index in range(count)]
    for size in range(2, count + 1):
        for indices in combinations(range(count), size):
            selected = tf.gather(safe_rows, indices)
            anchor = tf.argmin(tf.linalg.norm(selected, axis=1), output_type=tf.int32)
            selected = tf.roll(selected, -anchor, axis=0)
            coordinates_indices = tf.roll(tf.constant(indices), -anchor, axis=0)
            difference = selected[1:] - selected[0]
            singular, left, right = tf.linalg.svd(difference, full_matrices=False)
            independent = (size - 1 <= rows.shape[1]) & tf.reduce_all(
                singular > tf.constant(64 * 2.220446049250313e-16, tf.float64) * tf.reduce_max(singular)
            )
            anchor_coordinates = tf.linalg.matvec(right, selected[0], transpose_a=True)
            perpendicular = selected[0] - tf.linalg.matvec(right, anchor_coordinates)
            base_coordinates = tf.linalg.matvec(right, base, transpose_a=True)
            denominator = tf.square(coefficient) - tf.reduce_sum(tf.square(base_coordinates))
            factor = tf.sqrt((tf.reduce_sum(tf.square(perpendicular)) + epsilon) / tf.where(denominator > 0.0, denominator, 1.0))
            coordinates = -factor * base_coordinates - anchor_coordinates
            barycentric = tf.linalg.matvec(left, coordinates / tf.where(singular > 0.0, singular, 1.0))
            local = tf.concat([[1.0 - tf.reduce_sum(barycentric)], barycentric], axis=0)
            positive = tf.maximum(local, 0.0)
            positive /= tf.reduce_sum(positive)
            weights.append(tf.linalg.matvec(tf.one_hot(coordinates_indices, count, dtype=tf.float64), positive, transpose_a=True))
            available.append(independent & (denominator > 0.0) & tf.reduce_all(local >= -1e-12))
    weights = tf.stack(weights)
    combined = tf.matmul(weights, safe_rows)
    norms = tf.sqrt(tf.reduce_sum(tf.square(combined), axis=1) + epsilon)
    values = tf.linalg.matvec(combined, base) + coefficient * norms
    gradient = tf.linalg.matvec(safe_rows, base)[None, :] + coefficient * tf.matmul(combined, safe_rows, transpose_b=True) / norms[:, None]
    gaps = tf.reduce_sum(weights * gradient, axis=1) - tf.reduce_min(gradient, axis=1)
    gram = tf.matmul(safe_rows, safe_rows, transpose_b=True)
    feasible = (
        tf.stack(available)
        & _cagrad_kkt_feasible(gram, uniform, coefficient, weights, epsilon)
        & (gaps <= tf.constant(1e-10, tf.float64) * tf.maximum(tf.abs(values), 1.0))
        & tf.math.is_finite(values)
    )
    best = tf.argmin(tf.where(feasible, values, tf.constant(float("inf"), tf.float64)), output_type=tf.int32)
    alpha = weights[best]
    weighted = combined[best]
    multiplier = coefficient / (tf.linalg.norm(weighted) + epsilon)
    if rescale not in (0, 1, 2):
        raise ValueError("CAGrad rescale must be 0, 1 or 2")
    divisor = (1.0, 1.0 + c * c, 1.0 + c)[rescale]
    coefficients = (uniform + multiplier * alpha) / divisor
    direction = (base + multiplier * weighted) / divisor
    valid = inputs_valid & tf.reduce_any(feasible) & _finite(direction) & _finite(coefficients)
    return coefficients, direction, (alpha, multiplier, base_norm, values[best], gaps[best]), valid


def make_certified_method_function(method, active_count, parameter_dim, *, reset_method_state=True):
    legacy = make_method_function(method, active_count, parameter_dim, reset_method_state=reset_method_state)
    if method != "cagrad":
        return legacy

    @tf.function(input_signature=legacy.input_signature, autograph=False, jit_compile=False)
    def direction(rows, losses, permutations, method_weights, initial_losses, method_step):
        coefficients, combined, _diagnostics, valid = certified_cagrad_impl(rows)
        valid = valid & _finite(losses) & tf.reduce_all(losses >= 0.0) & (tf.linalg.norm(combined) > 1e-12)
        return coefficients, combined, method_weights, initial_losses, method_step, valid

    return direction


def make_certified_projected_adam_function(parameter_dim, constraint_count, config=DEFAULT_ADAM):
    projector = make_certified_projection_function(parameter_dim, constraint_count)
    packed = tf.function(packed_adam_impl, autograph=False, jit_compile=True, input_signature=[
        tf.TensorSpec([parameter_dim], tf.float64) for _index in range(3)
    ] + [tf.TensorSpec([], tf.int64), tf.TensorSpec([parameter_dim], tf.float64)] + [
        tf.TensorSpec([], tf.float64) for _index in range(5)
    ])
    primitive = _make_projected_adam_function(parameter_dim, constraint_count, config,
                                            projector_impl=projector, packed_impl=packed, jit_compile=False)

    @tf.function(input_signature=primitive.input_signature, autograph=False, jit_compile=False)
    def update(parameters, first_moment, second_moment, iteration, direction, protected_rows, learning_rate):
        rate = tf.cast(tf.cast(learning_rate, tf.float32), tf.float64)
        return primitive(parameters, first_moment, second_moment, iteration, direction, protected_rows, rate)

    return update
