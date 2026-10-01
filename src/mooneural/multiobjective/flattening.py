"""TensorFlow gradient flattening helpers."""

from __future__ import annotations

import tensorflow as tf


def flatten_objective_gradients(objective_gradients, reference_variables):
    """Flatten a sequence of per-objective gradient structures."""
    if not objective_gradients:
        raise ValueError("at least one objective gradient structure is required")
    rows = [
        flatten_like_variables(grads, reference_variables)
        for grads in objective_gradients
    ]
    flat = tf.stack(rows, axis=0)
    with tf.control_dependencies([
        tf.debugging.assert_all_finite(
            flat, "objective gradients must be finite")
    ]):
        return tf.identity(flat)


def flatten_like_variables(grads, reference_variables):
    """Flatten one gradient structure, replacing `None` by zeros."""
    if len(grads) != len(reference_variables):
        raise ValueError(
            "gradient structure length must match reference_variables")
    pieces = []
    for grad, var in zip(grads, reference_variables):
        if grad is None:
            grad_tensor = tf.zeros_like(var)
        else:
            grad_tensor = tf.cast(tf.convert_to_tensor(grad), var.dtype)
        pieces.append(tf.reshape(grad_tensor, [-1]))
    if not pieces:
        raise ValueError("reference_variables must be nonempty")
    return tf.concat(pieces, axis=0)


def unflatten_like_variables(flat_gradient, reference_variables):
    """Unflatten a flat TensorFlow gradient into variable-shaped tensors."""
    pieces = []
    offset = 0
    for var in reference_variables:
        size = var.shape.num_elements()
        if size is None:
            raise ValueError(
                "reference variable shapes must be statically known")
        piece = flat_gradient[offset:offset + int(size)]
        offset += int(size)
        pieces.append(tf.reshape(tf.cast(piece, var.dtype), var.shape))
    if flat_gradient.shape.rank == 1 and flat_gradient.shape[0] is not None:
        if offset != int(flat_gradient.shape[0]):
            raise ValueError("flat gradient size does not match variables")
    return tuple(pieces)
