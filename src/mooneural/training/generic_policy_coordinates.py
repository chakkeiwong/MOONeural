"""Affine coordinates and analytic pullbacks for a two-hidden-layer tanh policy."""

import math
from dataclasses import dataclass

import tensorflow as tf

from .generic_training_contracts import stable_hash


@dataclass(frozen=True)
class AffinePolicyCoordinates:
    input_center: tuple[float, ...]
    input_scale: tuple[float, ...]
    output_center: tuple[float, ...]
    output_scale: tuple[float, ...]
    training_binding: str

    def __post_init__(self):
        for name in ("input_center", "input_scale", "output_center", "output_scale"):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError("coordinate profile must be finite and nonempty")
            if name.endswith("scale") and min(values) <= 0:
                raise ValueError("coordinate scales must be positive")
            object.__setattr__(self, name, values)
        if len(self.input_center) != len(self.input_scale) or len(self.output_center) != len(self.output_scale):
            raise ValueError("coordinate profile dimensions disagree")
        if not isinstance(self.training_binding, str) or len(self.training_binding) != 64:
            raise ValueError("coordinate profile requires its training binding")
        int(self.training_binding, 16)

    def to_dict(self):
        return {"schema": "generic_neural_solver.affine_policy_coordinates.v1",
                "input_center": list(self.input_center), "input_scale": list(self.input_scale),
                "output_center": list(self.output_center), "output_scale": list(self.output_scale),
                "training_binding": self.training_binding}

    @classmethod
    def from_dict(cls, value):
        payload = dict(value)
        if payload.pop("schema", None) != "generic_neural_solver.affine_policy_coordinates.v1":
            raise ValueError("unknown coordinate schema")
        return cls(**payload)

    def binding_hash(self):
        return stable_hash(self.to_dict())


class TanhCoordinateMap:
    def __init__(self, profile, hidden_width):
        if not isinstance(profile, AffinePolicyCoordinates):
            raise TypeError("affine policy coordinates required")
        if type(hidden_width) is not int or hidden_width <= 0:
            raise ValueError("hidden width must be positive")
        self.profile = profile
        self.hidden_width = hidden_width
        input_dim, output_dim = len(profile.input_center), len(profile.output_center)
        self.sizes = (input_dim * hidden_width, hidden_width, hidden_width**2,
                      hidden_width, hidden_width * output_dim, output_dim)
        self.parameter_dim = sum(self.sizes)
        center = tf.constant(profile.input_center, tf.float64)
        scale = tf.constant(profile.input_scale, tf.float64)
        output_center = tf.constant(profile.output_center, tf.float64)
        output_scale = tf.constant(profile.output_scale, tf.float64)

        @tf.function(input_signature=[tf.TensorSpec([self.parameter_dim], tf.float64)], autograph=False)
        def to_raw(values):
            weight0, bias0, weight1, bias1, weight2, bias2 = tf.split(values, self.sizes)
            raw_weight0 = tf.reshape(weight0, [input_dim, hidden_width]) / scale[:, None]
            raw_bias0 = bias0 - tf.linalg.matvec(raw_weight0, center, transpose_a=True)
            raw_weight2 = tf.reshape(weight2, [hidden_width, output_dim]) * output_scale[None, :]
            return tf.concat([tf.reshape(raw_weight0, [-1]), raw_bias0, weight1, bias1,
                              tf.reshape(raw_weight2, [-1]), output_center + bias2 * output_scale], axis=0)

        @tf.function(input_signature=[tf.TensorSpec([self.parameter_dim], tf.float64)], autograph=False)
        def from_raw(values):
            weight0, bias0, weight1, bias1, weight2, bias2 = tf.split(values, self.sizes)
            raw_weight0 = tf.reshape(weight0, [input_dim, hidden_width])
            coordinate_bias0 = bias0 + tf.linalg.matvec(raw_weight0, center, transpose_a=True)
            coordinate_weight2 = tf.reshape(weight2, [hidden_width, output_dim]) / output_scale[None, :]
            return tf.concat([tf.reshape(raw_weight0 * scale[:, None], [-1]), coordinate_bias0, weight1, bias1,
                              tf.reshape(coordinate_weight2, [-1]), (bias2 - output_center) / output_scale], axis=0)

        @tf.function(input_signature=[tf.TensorSpec([None, self.parameter_dim], tf.float64)], autograph=False)
        def pullback(rows):
            weight0, bias0, weight1, bias1, weight2, bias2 = tf.split(rows, self.sizes, axis=1)
            count = tf.shape(rows)[0]
            transformed_weight0 = (tf.reshape(weight0, [count, input_dim, hidden_width]) - center[None, :, None] * bias0[:, None, :]) / scale[None, :, None]
            transformed_weight2 = tf.reshape(weight2, [count, hidden_width, output_dim]) * output_scale[None, None, :]
            return tf.concat([tf.reshape(transformed_weight0, [count, self.sizes[0]]), bias0, weight1, bias1,
                              tf.reshape(transformed_weight2, [count, self.sizes[4]]), bias2 * output_scale], axis=1)

        self.to_raw, self.from_raw, self.pullback = to_raw, from_raw, pullback
