"""TensorFlow-native small-K MGDA solver."""

from __future__ import annotations

import itertools

import tensorflow as tf


def mgda_flat(flat_gradients, *, eps=1e-10, max_objectives=8):
    """Return MGDA coefficients and a combined flat gradient."""
    flat = tf.convert_to_tensor(flat_gradients)
    dtype = flat.dtype
    if flat.shape.rank != 2:
        raise ValueError("flat_gradients must have shape [objectives, params]")
    k_static = flat.shape[0]
    if k_static is None:
        raise ValueError("MGDA requires a statically known objective count")
    k = int(k_static)
    if k < 1:
        raise ValueError("at least one objective is required")
    if max_objectives is not None and k > int(max_objectives):
        raise ValueError(
            f"MGDA active-set solver is limited to {int(max_objectives)} "
            f"objectives; got {k}")
    if k == 1:
        alpha = tf.ones([1], dtype=dtype)
        return alpha, tf.reshape(flat[0], [-1]), {
            "mgda_active_set_size": tf.constant(1, tf.int32),
            "mgda_objective": tf.reduce_sum(tf.square(flat[0])),
        }

    gram = tf.linalg.matmul(flat, flat, transpose_b=True)
    gram = 0.5 * (gram + tf.transpose(gram))
    candidates = []
    values = []
    active_sizes = []
    for size in range(1, k + 1):
        for active in itertools.combinations(range(k), size):
            alpha, feasible = _solve_active_set(gram, active, k, dtype, eps)
            value = tf.tensordot(alpha, tf.linalg.matvec(gram, alpha), axes=1)
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
            message="no feasible MGDA active set found",
        )
    ]):
        best = tf.argmin(value_tensor, axis=0, output_type=tf.int32)
    alpha = tf.gather(candidate_tensor, best)
    alpha = alpha / tf.reduce_sum(alpha)
    combined = tf.linalg.matvec(flat, alpha, transpose_a=True)
    return alpha, combined, {
        "mgda_active_set_size": tf.gather(
            tf.constant(active_sizes, dtype=tf.int32), best),
        "mgda_objective": tf.gather(value_tensor, best),
    }


def _solve_active_set(gram, active, k, dtype, eps):
    active_idx = tf.constant(active, dtype=tf.int32)
    g_sub = tf.gather(tf.gather(gram, active_idx, axis=0), active_idx, axis=1)
    ones = tf.ones([len(active)], dtype=dtype)
    if len(active) == 1:
        active_weights = tf.ones([1], dtype=dtype)
        kkt_feasible = tf.constant(True)
    else:
        zeros = tf.zeros([len(active)], dtype=dtype)
        top = tf.concat([2.0 * g_sub, ones[:, None]], axis=1)
        bottom = tf.concat([
            ones[None, :],
            tf.zeros([1, 1], dtype=dtype),
        ], axis=1)
        kkt = tf.concat([top, bottom], axis=0)
        rhs = tf.concat([zeros, tf.ones([1], dtype=dtype)], axis=0)
        singular_values = tf.linalg.svd(kkt, compute_uv=False)
        kkt_feasible = tf.reduce_min(singular_values) > tf.cast(eps, dtype)

        def solve_kkt():
            return tf.linalg.solve(kkt, rhs[:, None])[:, 0][:len(active)]

        def inactive_candidate():
            return tf.zeros([len(active)], dtype=dtype)

        active_weights = tf.cond(kkt_feasible, solve_kkt, inactive_candidate)
    weight_sum = tf.reduce_sum(active_weights)
    feasible = (
        kkt_feasible
        & tf.reduce_all(tf.math.is_finite(active_weights))
        & (tf.abs(weight_sum - tf.cast(1, dtype)) <= tf.cast(1e-5, dtype))
        & tf.reduce_all(active_weights >= tf.cast(-1e-8, dtype))
    )

    def normalized_active_weights():
        clipped = tf.maximum(active_weights, tf.cast(0, dtype))
        return clipped / tf.reduce_sum(clipped)

    def inactive_active_weights():
        return tf.zeros([len(active)], dtype=dtype)

    active_weights = tf.cond(
        feasible, normalized_active_weights, inactive_active_weights)
    alpha = tf.scatter_nd(
        indices=tf.reshape(active_idx, [-1, 1]),
        updates=active_weights,
        shape=[k],
    )
    return alpha, feasible
