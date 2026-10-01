"""TensorFlow-native paper-audited GradNorm helpers.

GradNorm requires non-negative task losses, explicit task-weight state,
initial-loss state, and a separate task-weight update.
"""

from __future__ import annotations

import tensorflow as tf

from mooneural.multiobjective.types import GradNormState, GradNormUpdateResult


GRADNORM_REFERENCE_EPS = 1e-8


def create_gradnorm_state(objective_count, *, dtype=tf.float64, initial_losses=None):
    """Create functional state for the paper-audited GradNorm contract."""
    k = int(objective_count)
    if k < 1:
        raise ValueError("objective_count must be positive")
    if initial_losses is None:
        initial = tf.zeros([k], dtype=dtype)
    else:
        initial = tf.cast(tf.reshape(initial_losses, [k]), dtype)
    return GradNormState(
        weights=tf.ones([k], dtype=dtype),
        initial_losses=initial,
        step=tf.constant(0, dtype=tf.int64),
    )


def gradnorm_flat(
    flat_gradients,
    losses,
    state,
    *,
    alpha=1.5,
    eps=GRADNORM_REFERENCE_EPS,
):
    """Return weighted GradNorm scalarization gradients without updating state."""
    flat = tf.convert_to_tensor(flat_gradients)
    dtype = flat.dtype
    if flat.shape.rank != 2:
        raise ValueError("flat_gradients must have shape [objectives, params]")
    if state is None:
        raise ValueError("GradNorm requires an explicit GradNormState")
    k_static = flat.shape[0]
    if k_static is None:
        raise ValueError("GradNorm requires a statically known objective count")
    k = int(k_static)
    losses = _loss_vector(losses, k, dtype)
    alpha = _positive_scalar(alpha, dtype, "GradNorm alpha must be positive")
    state = _ensure_initialized_state(state, losses, dtype)
    weights = _weights(state, k, dtype)
    weighted_flat = flat * weights[:, None]
    combined = tf.reduce_sum(weighted_flat, axis=0)
    diagnostics = _gradnorm_diagnostics(
        flat, losses, state, weights, alpha=alpha, eps=eps)
    return weights, combined, state, diagnostics


def gradnorm_update(
    flat_gradients,
    losses,
    state,
    *,
    alpha=1.5,
    learning_rate=0.01,
    eps=GRADNORM_REFERENCE_EPS,
):
    """Apply one GradNorm task-weight update."""
    flat = tf.convert_to_tensor(flat_gradients)
    dtype = flat.dtype
    if state is None:
        raise ValueError("GradNorm update requires an explicit GradNormState")
    if flat.shape.rank != 2:
        raise ValueError("flat_gradients must have shape [objectives, params]")
    k_static = flat.shape[0]
    if k_static is None:
        raise ValueError("GradNorm requires a statically known objective count")
    k = int(k_static)
    losses = _loss_vector(losses, k, dtype)
    alpha = _positive_scalar(alpha, dtype, "GradNorm alpha must be positive")
    learning_rate = _positive_scalar(
        learning_rate, dtype, "GradNorm learning_rate must be positive")
    state = _ensure_initialized_state(state, losses, dtype)
    weights = _weights(state, k, dtype)
    eps_tensor = tf.cast(max(float(eps), GRADNORM_REFERENCE_EPS), dtype)
    grad_norms = tf.linalg.norm(flat, axis=1)
    weighted_norms = weights * grad_norms
    initial_losses = _initial_losses(state, k, dtype)
    loss_ratios = losses / tf.maximum(initial_losses, eps_tensor)
    inverse_rates = loss_ratios / tf.reduce_mean(loss_ratios)
    target_norms = tf.stop_gradient(
        tf.reduce_mean(weighted_norms) * tf.pow(inverse_rates, tf.cast(alpha, dtype)))
    grad_loss_sign = tf.sign(weighted_norms - target_norms)
    weight_grad = grad_loss_sign * grad_norms
    updated_weights = weights - learning_rate * weight_grad
    updated_weights = tf.maximum(updated_weights, eps_tensor)
    updated_weights = updated_weights * (
        tf.cast(k, dtype) / tf.reduce_sum(updated_weights))
    new_state = GradNormState(
        weights=updated_weights,
        initial_losses=initial_losses,
        step=tf.cast(state.step + tf.constant(1, tf.int64), tf.int64),
    )
    diagnostics = _gradnorm_diagnostics(
        flat, losses, state, weights, alpha=alpha, eps=eps)
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "gradnorm_weight_gradient": weight_grad,
        "gradnorm_updated_weights": updated_weights,
        "gradnorm_step": new_state.step,
        "finite": (
            diagnostics["finite"]
            & tf.reduce_all(tf.math.is_finite(updated_weights))
            & tf.reduce_all(tf.math.is_finite(weight_grad))
        ),
    })
    return GradNormUpdateResult(state=new_state, diagnostics=diagnostics)


def _gradnorm_diagnostics(flat, losses, state, weights, *, alpha, eps):
    dtype = flat.dtype
    eps_tensor = tf.cast(max(float(eps), GRADNORM_REFERENCE_EPS), dtype)
    grad_norms = tf.linalg.norm(flat, axis=1)
    weighted_norms = weights * grad_norms
    initial_losses = _initial_losses(state, int(flat.shape[0]), dtype)
    loss_ratios = losses / tf.maximum(initial_losses, eps_tensor)
    inverse_rates = loss_ratios / tf.reduce_mean(loss_ratios)
    target_norms = tf.reduce_mean(weighted_norms) * tf.pow(
        inverse_rates, tf.cast(alpha, dtype))
    gradnorm_loss = tf.reduce_sum(tf.abs(weighted_norms - tf.stop_gradient(target_norms)))
    return {
        "gradnorm_alpha": tf.cast(alpha, dtype),
        "gradnorm_objective_grad_norms": grad_norms,
        "gradnorm_weighted_grad_norms": weighted_norms,
        "gradnorm_initial_losses": initial_losses,
        "gradnorm_loss_ratios": loss_ratios,
        "gradnorm_inverse_training_rates": inverse_rates,
        "gradnorm_target_grad_norms": target_norms,
        "gradnorm_loss": gradnorm_loss,
        "gradnorm_contract_status": tf.constant("paper_audited_gradnorm_v1"),
        "gradnorm_gradient_contract_status": tf.constant(
            "caller_supplied_selected_shared_gradients"),
        "finite": (
            tf.reduce_all(tf.math.is_finite(losses))
            & tf.reduce_all(tf.math.is_finite(weights))
            & tf.reduce_all(tf.math.is_finite(grad_norms))
            & tf.reduce_all(tf.math.is_finite(target_norms))
        ),
    }


def _ensure_initialized_state(state, losses, dtype):
    initial = tf.cast(tf.reshape(state.initial_losses, [-1]), dtype)
    zero_initial = tf.reduce_all(initial <= tf.cast(0, dtype))

    def from_losses():
        return tf.identity(losses)

    def from_state():
        return initial

    initial = tf.cond(zero_initial, from_losses, from_state)
    with tf.control_dependencies([
        tf.debugging.assert_non_negative(
            losses, message="GradNorm requires nonnegative task losses"),
        tf.debugging.assert_greater(
            initial,
            tf.cast(0, dtype),
            message="GradNorm initial losses must be positive"),
        tf.debugging.assert_all_finite(
            initial, "GradNorm initial losses must be finite"),
    ]):
        initial = tf.identity(initial)
    return GradNormState(
        weights=tf.cast(state.weights, dtype),
        initial_losses=initial,
        step=tf.cast(state.step, tf.int64),
    )


def _weights(state, expected_size, dtype):
    weights = tf.cast(tf.reshape(state.weights, [expected_size]), dtype)
    with tf.control_dependencies([
        tf.debugging.assert_greater(
            weights,
            tf.cast(0, dtype),
            message="GradNorm weights must be positive"),
        tf.debugging.assert_all_finite(weights, "GradNorm weights must be finite"),
    ]):
        return tf.identity(weights)


def _initial_losses(state, expected_size, dtype):
    return tf.cast(tf.reshape(state.initial_losses, [expected_size]), dtype)


def _loss_vector(losses, expected_size, dtype):
    losses_tensor = tf.cast(tf.reshape(tf.convert_to_tensor(losses), [-1]), dtype)
    if losses_tensor.shape[0] is not None and int(losses_tensor.shape[0]) != expected_size:
        raise ValueError(
            "GradNorm requires one loss value per objective gradient")
    with tf.control_dependencies([
        tf.debugging.assert_equal(
            tf.size(losses_tensor),
            expected_size,
            message="GradNorm requires one loss value per objective gradient"),
        tf.debugging.assert_all_finite(
            losses_tensor, "GradNorm losses must be finite"),
        tf.debugging.assert_non_negative(
            losses_tensor, message="GradNorm requires nonnegative task losses"),
    ]):
        return tf.identity(losses_tensor)


def _positive_scalar(value, dtype, message):
    scalar = tf.cast(tf.reshape(tf.convert_to_tensor(value), []), dtype)
    with tf.control_dependencies([
        tf.debugging.assert_greater(
            scalar,
            tf.cast(0, dtype),
            message=message),
        tf.debugging.assert_all_finite(scalar, message),
    ]):
        return tf.identity(scalar)
