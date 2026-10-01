"""Three training-only coverage losses and neural-policy parameter gradients.

The frozen coverage initializer remains unchanged because its bytes bind prior
evidence. Its public predictor supplies values and analytic raw input Jacobians;
reverse mode here differentiates only with respect to neural policy weights.
There is no structural autodiff, input-Jacobian autodiff, pfor, or training loop.
Fixed successor inputs and their supplied targets remain constants throughout.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .generic_coverage_initialization import (
    CoverageTrainingData,
    make_coverage_predictor,
)
from .generic_policy_coordinates import TanhCoordinateMap


def make_coverage_objective(profile, width, data, denominators, derivative_split,
                            *, state_reduction="mean", parameter_reduction="mean"):
    """Build ``evaluate(parameters) -> (raw, normalized, gradients)``.

    Parameters are coordinate-space float64 values packed by TanhCoordinateMap.
    Losses have shape [3] in value/state/parameter order; gradients have shape
    [3,P] and differentiate the normalized losses with respect to all P policy
    parameters, including hidden weights and biases. Inputs and Jacobian targets
    use raw coordinates, with the state coordinates before ``derivative_split``.

    For row probability p and O outputs, the value loss is
    ``sum_rows p * sum_outputs error**2 / O``. Each derivative loss also reduces
    its coordinate axis by the declared mean or sum. Group probabilities come
    from CoverageTrainingData, so configured group mass does not depend on row
    count. Normalized losses divide raw losses by the three denominators once.
    This is the same objective as fit_coverage_warm_start, now with all weights
    differentiable. The captured data and normalization are immutable snapshots.
    """
    mapping = TanhCoordinateMap(profile, width)
    input_dim, output_dim = len(profile.input_center), len(profile.output_center)
    if not isinstance(data, CoverageTrainingData):
        raise TypeError("CoverageTrainingData required")
    if type(derivative_split) is not int or not 0 < derivative_split < input_dim:
        raise ValueError("derivative split must be inside the input dimension")
    if state_reduction not in ("mean", "sum") or parameter_reduction not in ("mean", "sum"):
        raise ValueError("derivative coordinate reductions must be mean or sum")
    scales = np.asarray(denominators)
    if scales.shape != (3,) or scales.dtype.kind not in "fiu":
        raise ValueError("three positive finite task denominators required")
    scales = np.array(scales, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(scales)) or not np.all(scales > 0):
        raise ValueError("three positive finite task denominators required")
    inputs, targets, jacobians, weights = data.assemble()
    if inputs.shape[1] != input_dim or targets.shape[1] != output_dim:
        raise ValueError("coordinate profile and coverage dimensions disagree")
    if profile.training_binding != data.binding_hash():
        raise ValueError("coordinate profile training binding mismatch")
    counts = [1, derivative_split if state_reduction == "mean" else 1,
              input_dim - derivative_split if parameter_reduction == "mean" else 1]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        divisors = scales * output_dim * np.array(counts)
        row_task_weights = weights[:, None] / divisors[None, :]
    if (not np.all(np.isfinite(divisors)) or not np.all(divisors > 0)
            or not np.all(np.isfinite(row_task_weights)) or not np.all(row_task_weights > 0)):
        raise ValueError("weighted task denominators exceed float64 range")

    input_tensor = tf.constant(inputs, tf.float64)
    target_tensor = tf.constant(targets, tf.float64)
    jacobian_tensor = tf.constant(jacobians, tf.float64)
    weight_tensor = tf.constant(weights, tf.float64)
    scale_tensor = tf.constant(scales, tf.float64)
    predict = make_coverage_predictor(profile, width)
    state_reduce = tf.reduce_mean if state_reduction == "mean" else tf.reduce_sum
    parameter_reduce = tf.reduce_mean if parameter_reduction == "mean" else tf.reduce_sum

    @tf.function(input_signature=[tf.TensorSpec([mapping.parameter_dim], tf.float64, name="parameters")],
                 autograph=False)
    def evaluate(parameters):
        tf.debugging.assert_all_finite(parameters, "nonfinite coverage policy parameters")
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(parameters)
            values, raw_jacobians = predict(parameters, input_tensor)
            value_rows = tf.reduce_mean(tf.square(values - target_tensor), axis=1)
            derivative_errors = tf.square(raw_jacobians - jacobian_tensor)
            state_rows = tf.reduce_mean(state_reduce(derivative_errors[:, :, :derivative_split], axis=2), axis=1)
            parameter_rows = tf.reduce_mean(parameter_reduce(derivative_errors[:, :, derivative_split:], axis=2), axis=1)
            raw = tf.stack([tf.reduce_sum(weight_tensor * rows)
                            for rows in (value_rows, state_rows, parameter_rows)])
            normalized = raw / scale_tensor
            scalar_losses = tf.unstack(normalized)
        gradients = tf.stack([tape.gradient(loss, parameters, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                              for loss in scalar_losses])
        del tape
        tf.debugging.assert_all_finite(raw, "nonfinite raw coverage losses")
        tf.debugging.assert_all_finite(normalized, "nonfinite normalized coverage losses")
        tf.debugging.assert_all_finite(gradients, "nonfinite coverage policy gradients")
        return raw, normalized, gradients

    return evaluate


make_coverage_objective_function = make_coverage_objective
