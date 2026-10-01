"""TensorFlow-native functional FAMO."""

from __future__ import annotations

import math

import tensorflow as tf

from mooneural.multiobjective.types import FAMOState, FAMOUpdateResult


FAMO_REFERENCE_EPS = 1e-8


def create_famo_state(objective_count, *, dtype=tf.float64, min_losses=None):
    """Create a functional FAMO state."""
    k = int(objective_count)
    if k < 1:
        raise ValueError("objective_count must be positive")
    if min_losses is None:
        min_losses_tensor = tf.zeros([k], dtype=dtype)
    else:
        min_losses_tensor = tf.cast(tf.reshape(min_losses, [k]), dtype)
    return FAMOState(
        logits=tf.zeros([k], dtype=dtype),
        min_losses=min_losses_tensor,
        prev_losses=tf.zeros([k], dtype=dtype),
        adam_m=tf.zeros([k], dtype=dtype),
        adam_v=tf.zeros([k], dtype=dtype),
        step=tf.constant(0, dtype=tf.int64),
        pending=tf.constant(False),
    )


def famo_pre_step(flat_gradients, losses, state, *, eps=FAMO_REFERENCE_EPS):
    """Compute pre-step FAMO coefficients and return updated pending state."""
    flat = tf.convert_to_tensor(flat_gradients)
    dtype = flat.dtype
    if flat.shape.rank != 2:
        raise ValueError("flat_gradients must have shape [objectives, params]")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("FAMO eps must be positive and finite")
    if state is None:
        raise ValueError("FAMO requires an explicit FAMOState")
    k_static = flat.shape[0]
    if k_static is None:
        raise ValueError("FAMO requires a statically known objective count")
    k = int(k_static)
    losses = _loss_vector(losses, k, dtype)
    min_losses = tf.cast(tf.reshape(state.min_losses, [k]), dtype)
    eps_tensor = tf.cast(max(eps_value, FAMO_REFERENCE_EPS), dtype)
    with tf.control_dependencies([
        tf.debugging.assert_equal(
            state.pending,
            False,
            message="FAMO pre-step cannot overwrite a pending after-step"),
        tf.debugging.assert_non_negative(
            losses, message="FAMO requires nonnegative task losses"),
        tf.debugging.assert_greater_equal(
            losses - min_losses,
            -eps_tensor,
            message="FAMO min_losses exceed current losses"),
    ]):
        excess = losses - min_losses + eps_tensor
    logits = tf.cast(tf.reshape(state.logits, [k]), dtype)
    z = tf.nn.softmax(logits)
    inv_loss = z / excess
    weights = inv_loss / tf.reduce_sum(inv_loss)
    combined = tf.linalg.matvec(flat, weights, transpose_a=True)
    new_state = FAMOState(
        logits=logits,
        min_losses=min_losses,
        prev_losses=losses,
        adam_m=tf.cast(tf.reshape(state.adam_m, [k]), dtype),
        adam_v=tf.cast(tf.reshape(state.adam_v, [k]), dtype),
        step=tf.cast(state.step, tf.int64),
        pending=tf.constant(True),
    )
    return weights, combined, new_state, {
        "famo_softmax": z,
        "famo_excess_losses": excess,
        "famo_pending": tf.constant(True),
    }


def famo_after_step(
    losses_after,
    state,
    *,
    beta=0.01,
    weight_decay=0.001,
    eps=FAMO_REFERENCE_EPS,
):
    """Update FAMO task logits from same-batch after-step losses."""
    if state is None:
        raise ValueError("FAMO after-step requires an explicit FAMOState")
    dtype = state.logits.dtype
    beta_value = float(beta)
    weight_decay_value = float(weight_decay)
    eps_value = float(eps)
    if not math.isfinite(beta_value) or beta_value <= 0.0:
        raise ValueError("FAMO beta must be positive and finite")
    if not math.isfinite(weight_decay_value) or weight_decay_value < 0.0:
        raise ValueError("FAMO weight_decay must be nonnegative and finite")
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("FAMO eps must be positive and finite")
    k_static = state.logits.shape[0]
    if k_static is None:
        raise ValueError("FAMO state must have a static objective count")
    k = int(k_static)
    losses_after = _loss_vector(losses_after, k, dtype)
    min_losses = tf.cast(tf.reshape(state.min_losses, [k]), dtype)
    losses_before = tf.cast(tf.reshape(state.prev_losses, [k]), dtype)
    eps_tensor = tf.cast(max(eps_value, FAMO_REFERENCE_EPS), dtype)
    with tf.control_dependencies([
        tf.debugging.assert_equal(
            state.pending,
            True,
            message="FAMO after-step requires a pending pre-step"),
        tf.debugging.assert_non_negative(
            losses_after, message="FAMO requires nonnegative task losses"),
        tf.debugging.assert_greater_equal(
            losses_after - min_losses,
            -eps_tensor,
            message="FAMO min_losses exceed after-step losses"),
    ]):
        before_excess = losses_before - min_losses + eps_tensor
        after_excess = losses_after - min_losses + eps_tensor
    logits = tf.cast(tf.reshape(state.logits, [k]), dtype)
    z = tf.nn.softmax(logits)
    progress = tf.math.log(before_excess) - tf.math.log(after_excess)
    centered = progress - tf.reduce_sum(z * progress)
    logit_grad = z * centered
    step = tf.cast(state.step + tf.constant(1, tf.int64), tf.int64)
    grad = logit_grad + tf.cast(weight_decay_value, dtype) * logits
    adam_m = (
        tf.cast(0.9, dtype) * tf.cast(tf.reshape(state.adam_m, [k]), dtype)
        + tf.cast(0.1, dtype) * grad
    )
    adam_v = (
        tf.cast(0.999, dtype) * tf.cast(tf.reshape(state.adam_v, [k]), dtype)
        + tf.cast(0.001, dtype) * tf.square(grad)
    )
    step_float = tf.cast(step, dtype)
    m_hat = adam_m / (tf.cast(1, dtype) - tf.pow(tf.cast(0.9, dtype), step_float))
    v_hat = adam_v / (
        tf.cast(1, dtype) - tf.pow(tf.cast(0.999, dtype), step_float))
    update = tf.cast(beta_value, dtype) * m_hat / (
        tf.sqrt(v_hat) + eps_tensor)
    new_logits = logits - update
    new_state = FAMOState(
        logits=new_logits,
        min_losses=min_losses,
        prev_losses=losses_after,
        adam_m=adam_m,
        adam_v=adam_v,
        step=step,
        pending=tf.constant(False),
    )
    return FAMOUpdateResult(
        state=new_state,
        diagnostics={
            "method": tf.constant("famo"),
            "contract_version": tf.constant("famo_tf_functional_state_v1"),
            "famo_progress": progress,
            "famo_softmax": z,
            "famo_logit_gradient": logit_grad,
            "famo_logit_update": update,
            "famo_step": step,
            "famo_pending": tf.constant(False),
            "finite": (
                tf.reduce_all(tf.math.is_finite(new_logits))
                & tf.reduce_all(tf.math.is_finite(adam_m))
                & tf.reduce_all(tf.math.is_finite(adam_v))
            ),
        },
    )


def _loss_vector(losses, expected_size, dtype):
    losses_tensor = tf.cast(tf.reshape(tf.convert_to_tensor(losses), [-1]), dtype)
    if losses_tensor.shape[0] is not None and int(losses_tensor.shape[0]) != expected_size:
        raise ValueError(
            "FAMO requires one loss value per objective gradient")
    with tf.control_dependencies([
        tf.debugging.assert_equal(
            tf.size(losses_tensor),
            expected_size,
            message="FAMO requires one loss value per objective gradient"),
        tf.debugging.assert_all_finite(
            losses_tensor, "FAMO losses must be finite"),
    ]):
        return tf.identity(losses_tensor)
