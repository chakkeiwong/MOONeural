"""TensorFlow-native PCGrad."""

from __future__ import annotations

import tensorflow as tf


def pcgrad_flat(flat_gradients, *, seed, reduction="sum", eps=1e-12):
    """Apply seeded PCGrad surgery to flat objective gradients."""
    if reduction not in {"sum", "mean"}:
        raise ValueError("pcgrad reduction must be 'sum' or 'mean'")
    flat = tf.convert_to_tensor(flat_gradients)
    dtype = flat.dtype
    if flat.shape.rank != 2:
        raise ValueError("flat_gradients must have shape [objectives, params]")
    k_static = flat.shape[0]
    if k_static is None:
        raise ValueError("PCGrad requires a statically known objective count")
    k = int(k_static)
    seed = tf.cast(tf.reshape(tf.convert_to_tensor(seed), [2]), tf.int32)
    adjusted_rows = []
    order_rows = []
    projection_count = tf.constant(0, dtype=tf.int32)
    skipped_zero_denoms = tf.constant(0, dtype=tf.int32)
    eps_tensor = tf.cast(eps, dtype)
    for i in range(k):
        order_seed = seed + tf.constant([0, i + 1], dtype=tf.int32)
        order = tf.random.experimental.stateless_shuffle(
            tf.range(k, dtype=tf.int32), seed=order_seed)
        order_rows.append(order)
        g_i = flat[i]
        for j in tf.unstack(order):
            g_j = tf.gather(flat, j)
            dot = tf.tensordot(g_i, g_j, axes=1)
            denom = tf.tensordot(g_j, g_j, axes=1)
            should_project = (j != i) & (dot < tf.cast(0, dtype)) & (
                denom > eps_tensor)
            zero_denom_conflict = (j != i) & (dot < tf.cast(0, dtype)) & (
                denom <= eps_tensor)
            skipped_zero_denoms += tf.cast(zero_denom_conflict, tf.int32)

            def project():
                return g_i - dot / denom * g_j

            g_i = tf.cond(should_project, project, lambda: g_i)
            projection_count += tf.cast(should_project, tf.int32)
        adjusted_rows.append(g_i)
    adjusted = tf.stack(adjusted_rows, axis=0)
    if reduction == "mean":
        coefficients = tf.fill([k], tf.cast(1, dtype) / tf.cast(k, dtype))
        combined = tf.reduce_mean(adjusted, axis=0)
    else:
        coefficients = tf.ones([k], dtype=dtype)
        combined = tf.reduce_sum(adjusted, axis=0)
    return coefficients, combined, {
        "pcgrad_projection_orders": tf.stack(order_rows, axis=0),
        "pcgrad_projection_count": projection_count,
        "pcgrad_zero_denom_skips": skipped_zero_denoms,
        "pcgrad_reduction_is_sum": tf.constant(reduction == "sum"),
    }
