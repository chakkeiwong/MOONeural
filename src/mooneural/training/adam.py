"""Functional packed FP64 Adam update for the strict SGU XLA lane."""

from __future__ import annotations

import tensorflow as tf


DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1.0e-8
DEFAULT_CLIP_NORM = 10.0

OUTPUT_PARAMETERS = 0
OUTPUT_FIRST_MOMENT = 1
OUTPUT_SECOND_MOMENT = 2
OUTPUT_ITERATION = 3
OUTPUT_RAW_NORM = 4
OUTPUT_CLIPPED_NORM = 5
OUTPUT_DELTA_NORM = 6
OUTPUT_VALID = 7


def packed_adam_impl(
    parameters,
    first_moment,
    second_moment,
    iteration,
    gradient,
    learning_rate,
    clip_norm,
    beta1,
    beta2,
    epsilon,
):
    """Apply one allocation-free packed Adam step and return numeric state."""

    parameters = tf.ensure_shape(parameters, [None])
    first_moment = tf.ensure_shape(first_moment, parameters.shape)
    second_moment = tf.ensure_shape(second_moment, parameters.shape)
    gradient = tf.ensure_shape(gradient, parameters.shape)
    dtype = parameters.dtype
    learning_rate_value = tf.cast(tf.reshape(learning_rate, []), dtype)
    clip_norm_value = tf.cast(tf.reshape(clip_norm, []), dtype)
    beta1_value = tf.cast(tf.reshape(beta1, []), dtype)
    beta2_value = tf.cast(tf.reshape(beta2, []), dtype)
    epsilon_value = tf.cast(tf.reshape(epsilon, []), dtype)
    finite_inputs = (
        tf.reduce_all(tf.math.is_finite(parameters))
        & tf.reduce_all(tf.math.is_finite(first_moment))
        & tf.reduce_all(tf.math.is_finite(second_moment))
        & tf.reduce_all(tf.math.is_finite(gradient))
        & tf.math.is_finite(learning_rate_value)
        & tf.math.is_finite(clip_norm_value)
        & tf.math.is_finite(beta1_value)
        & tf.math.is_finite(beta2_value)
        & tf.math.is_finite(epsilon_value)
        & tf.math.is_finite(tf.cast(tf.reshape(iteration, []), dtype))
    )
    positive_config = (
        (learning_rate_value > tf.zeros([], dtype))
        & (clip_norm_value > tf.zeros([], dtype))
        & (epsilon_value > tf.zeros([], dtype))
        & (beta1_value >= tf.zeros([], dtype))
        & (beta1_value < tf.ones([], dtype))
        & (beta2_value >= tf.zeros([], dtype))
        & (beta2_value < tf.ones([], dtype))
    )
    nonnegative_iteration = tf.reshape(iteration, []) >= tf.constant(0, tf.int64)
    raw_norm = tf.linalg.norm(gradient)
    safe_norm = tf.maximum(raw_norm, tf.cast(1.0e-300, dtype))
    scale = tf.minimum(
        tf.ones([], dtype), clip_norm_value / safe_norm
    )
    clipped_gradient = gradient * scale
    clipped_norm = tf.linalg.norm(clipped_gradient)
    next_first = beta1_value * first_moment + (
        tf.ones([], dtype) - beta1_value
    ) * clipped_gradient
    next_second = beta2_value * second_moment + (
        tf.ones([], dtype) - beta2_value
    ) * tf.square(clipped_gradient)
    next_iteration = tf.cast(iteration, tf.int64) + tf.constant(1, tf.int64)
    next_iteration_float = tf.cast(next_iteration, dtype)
    correction1 = tf.ones([], dtype) - tf.pow(beta1_value, next_iteration_float)
    correction2 = tf.ones([], dtype) - tf.pow(beta2_value, next_iteration_float)
    safe_correction1 = tf.where(
        tf.abs(correction1) > tf.cast(1.0e-300, dtype),
        correction1,
        tf.ones([], dtype),
    )
    safe_correction2 = tf.where(
        tf.abs(correction2) > tf.cast(1.0e-300, dtype),
        correction2,
        tf.ones([], dtype),
    )
    corrected_first = next_first / safe_correction1
    corrected_second = next_second / safe_correction2
    delta = learning_rate_value * corrected_first / (
        tf.sqrt(tf.maximum(corrected_second, tf.zeros_like(corrected_second)))
        + epsilon_value
    )
    next_parameters = parameters - delta
    finite_outputs = (
        tf.reduce_all(tf.math.is_finite(next_parameters))
        & tf.reduce_all(tf.math.is_finite(next_first))
        & tf.reduce_all(tf.math.is_finite(next_second))
        & tf.math.is_finite(raw_norm)
        & tf.math.is_finite(clipped_norm)
        & tf.reduce_all(tf.math.is_finite(delta))
    )
    valid = finite_inputs & positive_config & nonnegative_iteration & finite_outputs
    next_parameters = tf.where(valid, next_parameters, parameters)
    next_first = tf.where(valid, next_first, first_moment)
    next_second = tf.where(valid, next_second, second_moment)
    next_iteration = tf.where(valid, next_iteration, tf.cast(iteration, tf.int64))
    delta_norm = tf.linalg.norm(next_parameters - parameters)
    return (
        next_parameters,
        next_first,
        next_second,
        next_iteration,
        raw_norm,
        clipped_norm,
        delta_norm,
        valid,
    )


def make_optimizer_function(
    *,
    parameter_dim: int,
    clip_norm: float = DEFAULT_CLIP_NORM,
    beta1: float = DEFAULT_BETA1,
    beta2: float = DEFAULT_BETA2,
    epsilon: float = DEFAULT_EPSILON,
):
    """Create one fixed-signature, resource-free XLA Adam function."""

    dimension = int(parameter_dim)
    if dimension <= 0:
        raise ValueError("adam_parameter_dim_invalid")
    if not (0.0 < float(clip_norm) < float("inf")) or not (
        0.0 <= float(beta1) < 1.0
    ):
        raise ValueError("adam_beta1_or_clip_invalid")
    if not (0.0 <= float(beta2) < 1.0) or not (
        0.0 < float(epsilon) < float("inf")
    ):
        raise ValueError("adam_beta2_or_epsilon_invalid")
    vector_spec = tf.TensorSpec([dimension], tf.float64, name="packed_vector")
    scalar_spec = tf.TensorSpec([], tf.float64, name="learning_rate")
    iteration_spec = tf.TensorSpec([], tf.int64, name="iteration")

    @tf.function(
        input_signature=(
            vector_spec,
            vector_spec,
            vector_spec,
            iteration_spec,
            vector_spec,
            scalar_spec,
        ),
        autograph=False,
        jit_compile=True,
    )
    def update(parameters, first_moment, second_moment, iteration, gradient, learning_rate):
        return packed_adam_impl(
            parameters,
            first_moment,
            second_moment,
            iteration,
            gradient,
            learning_rate,
            tf.constant(clip_norm, tf.float64),
            tf.constant(beta1, tf.float64),
            tf.constant(beta2, tf.float64),
            tf.constant(epsilon, tf.float64),
        )

    return update


__all__ = [
    "DEFAULT_BETA1",
    "DEFAULT_BETA2",
    "DEFAULT_CLIP_NORM",
    "DEFAULT_EPSILON",
    "OUTPUT_CLIPPED_NORM",
    "OUTPUT_DELTA_NORM",
    "OUTPUT_FIRST_MOMENT",
    "OUTPUT_ITERATION",
    "OUTPUT_PARAMETERS",
    "OUTPUT_RAW_NORM",
    "OUTPUT_SECOND_MOMENT",
    "OUTPUT_VALID",
    "make_optimizer_function",
    "packed_adam_impl",
]
