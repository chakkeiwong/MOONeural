"""TensorFlow-native bounded Aligned-MTL Procrustes operator."""

from __future__ import annotations

import tensorflow as tf


_VALID_SCALE_MODES = ("min", "median", "rmse")


def aligned_flat(flat_gradients, *, scale_mode="min", eps=1e-12):
    """Return Aligned-MTL coefficients and a combined flat gradient.

    `flat_gradients` has shape `[objectives, params]`.  The Aligned-MTL paper
    and source operate on a gradient matrix with task gradients as columns, so
    the local paper matrix is `flat_gradients.T` and the task-space Gram matrix
    is `flat_gradients @ flat_gradients.T`.
    """
    if scale_mode not in _VALID_SCALE_MODES:
        raise ValueError(
            "Aligned-MTL scale_mode must be one of "
            f"{_VALID_SCALE_MODES}; got {scale_mode!r}")

    flat = tf.convert_to_tensor(flat_gradients)
    dtype = flat.dtype
    if flat.shape.rank != 2:
        raise ValueError("flat_gradients must have shape [objectives, params]")
    if not dtype.is_floating:
        raise ValueError("flat_gradients must have a floating dtype")
    k_static = flat.shape[0]
    p_static = flat.shape[1]
    if k_static is None or p_static is None:
        raise ValueError(
            "Aligned-MTL requires statically known objectives and params")
    if int(k_static) < 1:
        raise ValueError("at least one objective is required")
    if int(p_static) < 1:
        raise ValueError("at least one gradient parameter is required")

    with tf.control_dependencies([
        tf.debugging.assert_all_finite(
            flat, "Aligned-MTL gradients must be finite")
    ]):
        flat = tf.identity(flat)

    gram = tf.linalg.matmul(flat, flat, transpose_b=True)
    gram = 0.5 * (gram + tf.transpose(gram))
    eigenvalues, eigenvectors = tf.linalg.eigh(gram)
    order = tf.argsort(eigenvalues, direction="DESCENDING")
    eigenvalues = tf.gather(eigenvalues, order)
    eigenvectors = tf.gather(eigenvectors, order, axis=1)

    eps_tensor = tf.maximum(tf.cast(eps, dtype), _machine_epsilon(dtype))
    max_eigenvalue = tf.reduce_max(eigenvalues)
    rank_tolerance = (
        tf.maximum(max_eigenvalue, tf.cast(0, dtype))
        * tf.cast(int(k_static), dtype)
        * eps_tensor
    )
    retained = eigenvalues > rank_tolerance
    rank = tf.reduce_sum(tf.cast(retained, tf.int32))

    with tf.control_dependencies([
        tf.debugging.assert_greater(
            rank,
            tf.constant(0, tf.int32),
            message="Aligned-MTL task Gram matrix has zero retained rank",
        )
    ]):
        retained_eigenvalues = tf.boolean_mask(eigenvalues, retained)
        retained_basis = tf.boolean_mask(eigenvectors, retained, axis=1)
        retained_eigenvalues = tf.identity(retained_eigenvalues)

    scale = _select_scale(retained_eigenvalues, scale_mode, dtype)
    scale_factors = tf.sqrt(scale / retained_eigenvalues)
    scaled_basis = retained_basis * scale_factors[None, :]
    transform = tf.linalg.matmul(
        scaled_basis, retained_basis, transpose_b=True)
    coefficients = tf.reduce_sum(transform, axis=1)
    combined = tf.linalg.matvec(flat, coefficients, transpose_a=True)

    min_retained = tf.reduce_min(retained_eigenvalues)
    max_retained = tf.reduce_max(retained_eigenvalues)
    condition_before = tf.sqrt(max_retained / min_retained)
    condition_after = tf.ones([], dtype=dtype)
    with tf.control_dependencies([
        tf.debugging.assert_all_finite(
            coefficients, "Aligned-MTL coefficients must be finite"),
        tf.debugging.assert_all_finite(
            combined, "Aligned-MTL combined gradient must be finite"),
    ]):
        coefficients = tf.identity(coefficients)
        combined = tf.identity(combined)

    return coefficients, combined, {
        "aligned_operator_status": tf.constant(
            "bounded_tf_procrustes_operator"),
        "aligned_scale_mode": tf.constant(scale_mode),
        "aligned_rank": rank,
        "aligned_rank_tolerance": rank_tolerance,
        "aligned_scale": scale,
        "aligned_retained_gram_eigenvalues": retained_eigenvalues,
        "aligned_singular_values": tf.sqrt(retained_eigenvalues),
        "aligned_condition_before": condition_before,
        "aligned_condition_after": condition_after,
        "aligned_transform": transform,
    }


def _select_scale(retained_eigenvalues, scale_mode, dtype):
    if scale_mode == "min":
        return tf.reduce_min(retained_eigenvalues)
    if scale_mode == "rmse":
        return tf.reduce_mean(retained_eigenvalues)

    sorted_values = tf.sort(retained_eigenvalues, direction="ASCENDING")
    count = tf.shape(sorted_values)[0]
    lower_median_index = (count - 1) // 2
    return tf.cast(sorted_values[lower_median_index], dtype)


def _machine_epsilon(dtype):
    if dtype == tf.float64:
        return tf.constant(2.220446049250313e-16, dtype=dtype)
    if dtype == tf.float32:
        return tf.constant(1.1920928955078125e-7, dtype=dtype)
    if dtype == tf.float16:
        return tf.constant(9.765625e-4, dtype=dtype)
    if dtype == tf.bfloat16:
        return tf.constant(7.8125e-3, dtype=dtype)
    return tf.cast(1e-12, dtype)
