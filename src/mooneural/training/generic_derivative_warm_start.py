"""Training-only final-layer least squares with value and input-Jacobian targets."""

import tensorflow as tf

from .generic_policy_coordinates import TanhCoordinateMap


def make_derivative_warm_start(profile, hidden_width, sample_count, derivative_split):
    mapping = TanhCoordinateMap(profile, hidden_width)
    input_dim, output_dim = len(profile.input_center), len(profile.output_center)
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("positive construction sample count required")
    if type(derivative_split) is not int or not 0 < derivative_split < input_dim:
        raise ValueError("derivative split must be inside the input dimension")
    center = tf.constant(profile.input_center, tf.float64)
    scale = tf.constant(profile.input_scale, tf.float64)
    output_center = tf.constant(profile.output_center, tf.float64)
    output_scale = tf.constant(profile.output_scale, tf.float64)

    @tf.function(input_signature=[
        tf.TensorSpec([mapping.parameter_dim], tf.float64),
        tf.TensorSpec([sample_count, input_dim], tf.float64),
        tf.TensorSpec([sample_count, output_dim], tf.float64),
        tf.TensorSpec([sample_count, output_dim, input_dim], tf.float64),
        tf.TensorSpec([3], tf.float64),
    ], autograph=False)
    def fit(parameters, inputs, value_targets, jacobian_targets, denominators):
        weight0, bias0, weight1, bias1, _weight2, _bias2 = tf.split(parameters, mapping.sizes)
        first = tf.reshape(weight0, [input_dim, hidden_width])
        second = tf.reshape(weight1, [hidden_width, hidden_width])
        hidden0 = tf.tanh(tf.matmul((inputs - center) / scale, first) + bias0)
        hidden1 = tf.tanh(tf.matmul(hidden0, second) + bias1)
        derivative0 = (1.0 - tf.square(hidden0))[:, :, None] * tf.transpose(first)[None, :, :] / scale[None, None, :]
        derivative1 = tf.einsum("nwd,wh->nhd", derivative0, second) * (1.0 - tf.square(hidden1))[:, :, None]
        features = tf.concat([hidden1, tf.ones([sample_count, 1], tf.float64)], axis=1)
        feature_derivatives = tf.concat([derivative1, tf.zeros([sample_count, 1, input_dim], tf.float64)], axis=1)
        state_design = tf.reshape(tf.transpose(feature_derivatives[:, :, :derivative_split], [0, 2, 1]), [-1, hidden_width + 1])
        parameter_design = tf.reshape(tf.transpose(feature_derivatives[:, :, derivative_split:], [0, 2, 1]), [-1, hidden_width + 1])
        counts = tf.constant([sample_count * output_dim, sample_count * output_dim * derivative_split,
                              sample_count * output_dim * (input_dim - derivative_split)], tf.float64)
        weights = tf.math.rsqrt(denominators * counts)
        design = tf.concat([features * weights[0], state_design * weights[1], parameter_design * weights[2]], axis=0)
        target = tf.concat([(value_targets - output_center) * weights[0],
                            tf.reshape(tf.transpose(jacobian_targets[:, :, :derivative_split], [0, 2, 1]), [-1, output_dim]) * weights[1],
                            tf.reshape(tf.transpose(jacobian_targets[:, :, derivative_split:], [0, 2, 1]), [-1, output_dim]) * weights[2]], axis=0)
        singular, left, right = tf.linalg.svd(design, full_matrices=False)
        retained = singular > tf.constant(1e-12, tf.float64) * tf.reduce_max(singular)
        inverse = tf.where(retained, tf.math.reciprocal(tf.where(retained, singular, 1.0)), 0.0)
        raw_coefficients = tf.matmul(right * inverse[None, :], tf.matmul(left, target, transpose_a=True))
        coefficients = raw_coefficients / output_scale[None, :]
        fitted = tf.concat([weight0, bias0, weight1, bias1, tf.reshape(coefficients[:-1], [-1]), coefficients[-1]], axis=0)
        residual = tf.matmul(design, raw_coefficients) - target
        normal_residual = tf.linalg.norm(tf.matmul(design, residual, transpose_a=True)) / (1.0 + tf.linalg.norm(design) * tf.linalg.norm(target))
        valid = (tf.reduce_all(tf.math.is_finite(fitted)) & tf.reduce_all(tf.math.is_finite(residual))
                 & tf.reduce_all(denominators > 0.0) & (normal_residual < tf.constant(1e-10, tf.float64)))
        return fitted, design, target, raw_coefficients, singular, normal_residual, valid

    return fit
