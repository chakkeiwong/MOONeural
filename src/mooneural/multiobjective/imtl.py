"""TensorFlow-native IMTL-G.

IMTL-G uses the paper's closed-form linear system built from task-gradient
differences and normalized-gradient differences.  The supported domain requires
nonzero task gradients and a full-rank, well-conditioned IMTL-G system.  Exact
duplicate, collinear, or otherwise rank-deficient task gradients are outside
this direct-solve contract and fail closed rather than receiving an implicit
regularized or pseudoinverse fallback.
"""

from __future__ import annotations

import tensorflow as tf


def imtl_g_flat(flat_gradients, *, eps=1e-12):
    """Return IMTL-G coefficients and combined flat gradient.

    Raises an InvalidArgumentError when the IMTL-G domain conditions are not
    met.  Callers that benchmark multiple methods should predeclare how such
    domain-ineligible IMTL cells are skipped instead of ranking them.
    """
    flat = tf.convert_to_tensor(flat_gradients)
    dtype = flat.dtype
    if flat.shape.rank != 2:
        raise ValueError("flat_gradients must have shape [objectives, params]")
    k_static = flat.shape[0]
    if k_static is None:
        raise ValueError("IMTL-G requires a statically known objective count")
    k = int(k_static)
    if k < 1:
        raise ValueError("at least one objective is required")
    if k == 1:
        alpha = tf.ones([1], dtype=dtype)
        return alpha, tf.reshape(flat[0], [-1]), {
            "imtl_solver_status": tf.constant("single_objective"),
        }
    norms = tf.linalg.norm(flat, axis=1)
    eps_tensor = tf.cast(eps, dtype)
    with tf.control_dependencies([
        tf.debugging.assert_greater(
            tf.reduce_min(norms),
            eps_tensor,
            message="IMTL-G requires nonzero task gradients",
        )
    ]):
        normalized = flat / norms[:, None]
    diff = flat[0][None, :] - flat[1:, :]
    normalized_diff = normalized[0][None, :] - normalized[1:, :]
    mat = tf.linalg.matmul(diff, normalized_diff, transpose_b=True)
    rhs = tf.linalg.matvec(normalized_diff, flat[0])
    singular_values = tf.linalg.svd(mat, compute_uv=False)
    with tf.control_dependencies([
        tf.debugging.assert_greater(
            tf.reduce_min(singular_values),
            eps_tensor,
            message="IMTL-G linear system is singular or ill-conditioned",
        )
    ]):
        tail = tf.linalg.solve(mat, rhs[:, None])[:, 0]
    alpha0 = tf.reshape(tf.cast(1, dtype) - tf.reduce_sum(tail), [1])
    alpha = tf.concat([alpha0, tail], axis=0)
    with tf.control_dependencies([
        tf.debugging.assert_all_finite(
            alpha, "IMTL-G coefficients must be finite")
    ]):
        alpha = tf.identity(alpha)
    combined = tf.linalg.matvec(flat, alpha, transpose_a=True)
    return alpha, combined, {
        "imtl_solver_status": tf.constant("direct"),
        "imtl_linear_system_matrix": mat,
        "imtl_linear_system_rhs": rhs,
        "imtl_min_singular_value": tf.reduce_min(singular_values),
    }
