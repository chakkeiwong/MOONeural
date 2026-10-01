"""TensorFlow diagnostics for common multiobjective aggregators.

These diagnostics describe runtime engineering contracts for exposed methods.
They do not by themselves certify reviewed-canonical promotion under the
reusable-library status taxonomy.
"""

from __future__ import annotations

import tensorflow as tf

from mooneural.multiobjective.unsupported import METHOD_CONTRACTS


def common_diagnostics(method, flat_gradients, coefficients, combined_flat, eps):
    """Build TensorFlow-only diagnostic tensors.

    The returned `reference_status` value is taken from `METHOD_CONTRACTS` and
    should be read as a bounded implementation-contract label for the runtime
    method implementation.  It is not, by itself, a reviewed-canonical
    promotion statement under the reusable-library status-audit taxonomy.
    """
    flat = tf.convert_to_tensor(flat_gradients)
    dtype = flat.dtype
    coeff = tf.cast(tf.reshape(coefficients, [-1]), dtype)
    combined = tf.cast(tf.reshape(combined_flat, [-1]), dtype)
    gram = tf.linalg.matmul(flat, flat, transpose_b=True)
    gram = 0.5 * (gram + tf.transpose(gram))
    norms = tf.sqrt(tf.maximum(tf.linalg.diag_part(gram), tf.cast(0, dtype)))
    denom = tf.maximum(
        norms[:, None] * norms[None, :],
        tf.cast(1e-30, dtype),
    )
    directional = tf.linalg.matvec(flat, combined)
    weight_sum = tf.reduce_sum(coeff)
    near_zero_tol = tf.cast(1e-8, dtype)
    finite = (
        tf.reduce_all(tf.math.is_finite(flat))
        & tf.reduce_all(tf.math.is_finite(coeff))
        & tf.reduce_all(tf.math.is_finite(combined))
        & tf.reduce_all(tf.math.is_finite(gram))
    )
    contract = METHOD_CONTRACTS[method]
    k = tf.shape(flat)[0]
    p = tf.shape(flat)[1]
    diagnostics = {
        "method": tf.constant(method),
        "contract_version": tf.constant(contract["contract_version"]),
        "reference_status": tf.constant(contract["reference_status"]),
        "objective_count": k,
        "gradient_dim": p,
        "gram": gram,
        "cosine": gram / denom,
        "coefficients": coeff,
        "coefficient_sum": weight_sum,
        "coefficient_min": tf.reduce_min(coeff),
        "coefficients_are_simplex": (
            tf.reduce_all(coeff >= -near_zero_tol)
            & (tf.abs(weight_sum - tf.cast(1, dtype)) <= tf.cast(1e-6, dtype))
        ),
        "near_zero_coefficient_count": tf.reduce_sum(
            tf.cast(coeff <= near_zero_tol, tf.int32)),
        "objective_grad_norms": norms,
        "combined_grad_norm": tf.linalg.norm(combined),
        "objective_directional_derivatives": directional,
        "min_objective_directional_derivative": tf.reduce_min(directional),
        "has_common_descent_direction_for_reported_gradients": tf.reduce_all(
            directional >= -near_zero_tol),
        "flat_gradient_element_count": tf.size(flat),
        "gram_element_count": tf.size(gram),
        "finite": finite,
    }
    if method == "cagrad" and "c_theorem_scope" in contract:
        diagnostics["cagrad_c_theorem_scope"] = tf.constant(
            contract["c_theorem_scope"])
    if method == "cagrad" and "a5_status" in contract:
        diagnostics["cagrad_a5_status"] = tf.constant(contract["a5_status"])
    return diagnostics
