"""Fixed CPU tensor reductions; component probes, not an admitted model adapter."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

LOCAL_INPUT_NAMES = (
    "candidate_observables",
    "candidate_state_jacobian",
    "candidate_parameter_jacobian",
    "reference_observables",
    "reference_state_jacobian",
    "reference_parameter_jacobian",
    "state_factor",
    "parameter_factor",
    "observable_scales",
    "value_uncertainty",
    "state_uncertainty",
    "parameter_uncertainty",
    "parameter_reference_uncertainty_score",
    "scale_rounding_bounds",
)


@dataclass(frozen=True)
class LocalNormalizationSpec:
    row_count: int
    output_count: int
    state_dimension: int
    parameter_dimension: int

    def __post_init__(self):
        for name in (
            "row_count", "output_count", "state_dimension", "parameter_dimension"
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def signature(self):
        value_shape = [self.row_count, self.output_count]
        state_shape = [*value_shape, self.state_dimension]
        parameter_shape = [*value_shape, self.parameter_dimension]
        shapes = (
            value_shape, state_shape, parameter_shape,
            value_shape, state_shape, parameter_shape,
            [self.row_count, self.state_dimension, self.state_dimension],
            [self.parameter_dimension], [self.output_count],
            value_shape, state_shape, parameter_shape, [], [self.output_count],
        )
        return tuple(
            tf.TensorSpec(shape, tf.float64, name)
            for name, shape in zip(LOCAL_INPUT_NAMES, shapes, strict=True)
        )


def _all_finite(tensors):
    return tf.reduce_all(
        tf.stack([tf.reduce_all(tf.math.is_finite(value)) for value in tensors])
    )


def _mse_triplet(value, state, parameter):
    return tf.stack(
        (
            tf.reduce_mean(tf.square(value)),
            tf.reduce_mean(tf.reduce_sum(tf.square(state), axis=2)),
            tf.reduce_mean(tf.reduce_sum(tf.square(parameter), axis=2)),
        )
    )


def _vector_norm(value):
    return tf.sqrt(tf.reduce_sum(tf.square(value), axis=2))


def _mae_families(value_abs, state_abs, parameter_abs, parameter_norm):
    return tf.stack(
        (
            tf.reduce_mean(value_abs),
            tf.reduce_mean(state_abs),
            tf.reduce_mean(_vector_norm(state_abs)),
            tf.reduce_mean(parameter_abs),
            tf.reduce_mean(parameter_norm),
        )
    )


def monotonicity_guards(central_mse, conservative_mse, central_mae, conservative_mae):
    return (
        tf.reduce_all(conservative_mse >= central_mse),
        tf.reduce_all(conservative_mae >= central_mae),
    )


def local_normalization_impl(
    candidate_observables,
    candidate_state_jacobian,
    candidate_parameter_jacobian,
    reference_observables,
    reference_state_jacobian,
    reference_parameter_jacobian,
    state_factor,
    parameter_factor,
    observable_scales,
    value_uncertainty,
    state_uncertainty,
    parameter_uncertainty,
    parameter_reference_uncertainty_score,
    scale_rounding_bounds,
):
    """Standardize before squaring; physical MSE is a diagnostic, not a scale base."""
    value_delta = candidate_observables - reference_observables
    state_delta = candidate_state_jacobian - reference_state_jacobian
    parameter_delta = candidate_parameter_jacobian - reference_parameter_jacobian
    lower_scales = observable_scales - scale_rounding_bounds
    value_central = value_delta / observable_scales[None, :]
    value_upper = (tf.abs(value_delta) + value_uncertainty) / lower_scales[None, :]
    transformed_state = tf.einsum("noi,nij->noj", state_delta, state_factor)
    state_central = transformed_state / observable_scales[None, :, None]
    state_upper = (
        tf.abs(transformed_state)
        + tf.einsum("noi,nij->noj", state_uncertainty, tf.abs(state_factor))
    ) / lower_scales[None, :, None]
    parameter_central = (
        parameter_delta * parameter_factor[None, None, :]
        / observable_scales[None, :, None]
    )
    parameter_component_upper = (
        (tf.abs(parameter_delta) + parameter_uncertainty)
        * parameter_factor[None, None, :] / lower_scales[None, :, None]
    )
    rounding_factor = tf.reduce_max(observable_scales / lower_scales)
    reference_addend = parameter_reference_uncertainty_score * rounding_factor
    parameter_squared_norm = tf.reduce_sum(tf.square(parameter_component_upper), axis=2)
    parameter_norm = tf.sqrt(parameter_squared_norm)
    parameter_norm_upper = parameter_norm + reference_addend
    parameter_squared_norm_upper = (
        parameter_squared_norm + 2.0 * reference_addend * parameter_norm
        + tf.square(reference_addend)
    )
    central_mse = _mse_triplet(value_central, state_central, parameter_central)
    conservative_mse = tf.stack(
        (
            tf.reduce_mean(tf.square(value_upper)),
            tf.reduce_mean(tf.reduce_sum(tf.square(state_upper), axis=2)),
            tf.reduce_mean(parameter_squared_norm_upper),
        )
    )
    central_mae = _mae_families(
        tf.abs(value_central), tf.abs(state_central), tf.abs(parameter_central),
        _vector_norm(parameter_central),
    )
    conservative_mae = _mae_families(
        value_upper, state_upper, parameter_component_upper + reference_addend,
        parameter_norm_upper,
    )
    input_finite = _all_finite(
        (
            candidate_observables, candidate_state_jacobian,
            candidate_parameter_jacobian, reference_observables,
            reference_state_jacobian, reference_parameter_jacobian,
            state_factor, parameter_factor, observable_scales, value_uncertainty,
            state_uncertainty, parameter_uncertainty,
            parameter_reference_uncertainty_score, scale_rounding_bounds,
        )
    )
    input_valid = tf.reduce_all(tf.stack((
        input_finite,
        tf.reduce_all(observable_scales > 0.0),
        tf.reduce_all(parameter_factor > 0.0),
        tf.reduce_all(scale_rounding_bounds >= 0.0),
        tf.reduce_all(lower_scales > 0.0),
        tf.reduce_all(value_uncertainty >= 0.0),
        tf.reduce_all(state_uncertainty >= 0.0),
        tf.reduce_all(parameter_uncertainty >= 0.0),
        parameter_reference_uncertainty_score >= 0.0,
    )))
    result = {
        "physical_mse_diagnostic": _mse_triplet(value_delta, state_delta, parameter_delta),
        "value_central": value_central,
        "state_central": state_central,
        "parameter_central": parameter_central,
        "value_conservative_abs": value_upper,
        "state_conservative_abs": state_upper,
        "parameter_component_conservative_abs": parameter_component_upper,
        "parameter_norm_conservative": parameter_norm_upper,
        "parameter_squared_norm_central": tf.reduce_sum(tf.square(parameter_central), axis=2),
        "parameter_squared_norm_conservative": parameter_squared_norm_upper,
        "scale_rounding_factor": rounding_factor,
        "central_normalized_mse": central_mse,
        "conservative_normalized_mse": conservative_mse,
        "central_mae": central_mae,
        "conservative_mae": conservative_mae,
        "value_component_mae": tf.reduce_mean(tf.abs(value_central), axis=0),
        "state_component_mae": tf.reduce_mean(tf.abs(state_central), axis=0),
        "parameter_component_mae": tf.reduce_mean(tf.abs(parameter_central), axis=0),
    }
    output_finite = _all_finite(tuple(
        value for name, value in result.items()
        if name != "physical_mse_diagnostic" and "mae" not in name
    ))
    mae_finite = _all_finite(tuple(value for name, value in result.items() if "mae" in name))
    mse_monotone, mae_monotone = monotonicity_guards(
        central_mse, conservative_mse, central_mae, conservative_mae
    )
    result.update(
        inputs_valid=input_valid,
        outputs_finite=output_finite,
        physical_mse_finite=_all_finite((result["physical_mse_diagnostic"],)),
        mse_monotone=mse_monotone,
        mae_monotone=mae_monotone,
        mae_diagnostics_valid=input_valid & mae_finite & mae_monotone,
        valid=input_valid & output_finite & mse_monotone,
    )
    return result


def make_local_normalization_function(spec, *, jit_compile=False):
    """XLA is an explicit component experiment, not an admitted runtime default."""
    if not isinstance(spec, LocalNormalizationSpec):
        raise TypeError("a LocalNormalizationSpec is required")
    if type(jit_compile) is not bool:
        raise TypeError("jit_compile must be boolean")

    @tf.function(input_signature=spec.signature(), autograph=False, jit_compile=jit_compile)
    def normalize(*inputs):
        with tf.device("/CPU:0"):
            return local_normalization_impl(*inputs)

    return normalize


def make_global_mse_normalization_function(task_count, *, jit_compile=False):
    """The denominator is an MSE-scale scalar, not a residual standard deviation."""
    if type(task_count) is not int or task_count < 1:
        raise ValueError("task_count must be a positive integer")
    if type(jit_compile) is not bool:
        raise TypeError("jit_compile must be boolean")

    @tf.function(
        input_signature=(
            tf.TensorSpec([task_count], tf.float64, "raw_mse"),
            tf.TensorSpec([task_count], tf.float64, "mse_denominators"),
        ),
        autograph=False,
        jit_compile=jit_compile,
    )
    def normalize(raw_mse, mse_denominators):
        with tf.device("/CPU:0"):
            normalized = raw_mse / mse_denominators
            valid = (
                _all_finite((raw_mse, mse_denominators, normalized))
                & tf.reduce_all(raw_mse >= 0.0)
                & tf.reduce_all(mse_denominators > 0.0)
            )
            return {"training_normalized_mse": normalized, "valid": valid}

    return normalize


def require_valid_metrics(result):
    """Host refusal boundary: XLA assertions are not used as safety gates."""
    if not bool(result["valid"].numpy()):
        raise ValueError("normalization input, finiteness or monotonicity check failed")
    return result
