"""Directional weight responses of signed, already normalized task residuals."""

import tensorflow as tf


def residual_losses(residuals, probabilities):
    """Reduce [rows, components] tasks using the supplied outer probabilities."""
    if not isinstance(residuals, (tuple, list)) or not residuals:
        raise ValueError("a nonempty sequence of residual tasks is required")
    probabilities = tf.ensure_shape(tf.convert_to_tensor(probabilities, tf.float64), [None])
    tf.debugging.assert_all_finite(probabilities, "nonfinite outer probabilities")
    tf.debugging.assert_positive(probabilities, message="positive outer probabilities required")
    tf.debugging.assert_near(tf.reduce_sum(probabilities), tf.constant(1., tf.float64),
                             rtol=0., atol=64 * 2.220446049250313e-16,
                             message="outer probabilities must sum to one")
    losses = []
    for residual in residuals:
        residual = tf.ensure_shape(tf.convert_to_tensor(residual, tf.float64), [None, None])
        tf.debugging.assert_positive(tf.shape(residual), message="residual dimensions must be nonempty")
        tf.debugging.assert_equal(tf.shape(residual)[0], tf.shape(probabilities)[0], message="outer row mismatch")
        tf.debugging.assert_all_finite(residual, "nonfinite signed residual")
        losses.append(tf.reduce_sum(probabilities * tf.reduce_sum(tf.square(residual), axis=1)))
    result = tf.stack(losses)
    tf.debugging.assert_all_finite(result, "nonfinite residual losses")
    return result


def make_residual_response(residuals, input_signature):
    """Return a stable graph taking parameters, one weight direction, then data.

    The callback returns one [rows, components] float64 matrix per task. It
    retains model hard checks and complete current/future dependencies. Only the
    packed weight vector is differentiated; model structural jets are inputs to
    this weight calculation, not structural autodiff requests.
    """
    signature = list(input_signature)
    if (not signature or not isinstance(signature[0], tf.TensorSpec)
            or signature[0].dtype != tf.float64 or signature[0].shape.rank != 1
            or not signature[0].shape.is_fully_defined()):
        raise ValueError("a fixed float64 parameter-vector signature is required")

    @tf.function(input_signature=[signature[0], signature[0], *signature[1:]], autograph=False)
    def response(parameters, direction, *inputs):
        tf.debugging.assert_all_finite(parameters, "nonfinite response parameters")
        tf.debugging.assert_all_finite(direction, "nonfinite response direction")
        with tf.autodiff.ForwardAccumulator(parameters, direction) as accumulator:
            values = residuals(parameters, *inputs)
        if not isinstance(values, (tuple, list)) or not values:
            raise ValueError("a nonempty sequence of residual tasks is required")
        derivatives = accumulator.jvp(values, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        for value, derivative in zip(values, derivatives, strict=True):
            tf.debugging.assert_rank(value, 2, message="residual task must have rank two")
            tf.debugging.assert_type(value, tf.float64)
            tf.debugging.assert_positive(tf.shape(value), message="residual dimensions must be nonempty")
            tf.debugging.assert_all_finite(value, "nonfinite signed response residual")
            tf.debugging.assert_all_finite(derivative, "nonfinite directional response")
        return tuple(values), tuple(derivatives)

    return response
