"""Training-only current/successor coverage and derivative least squares.

This additive implementation leaves ``generic_derivative_warm_start.py`` intact:
its source bytes are bound into retained evidence. The fixed hidden features and
raw input derivative convention match that implementation. This module adds
explicit group masses, row weights and derivative coordinate reductions.

For group g and row i, the empirical probability is
``p[g,i] = mass[g]/sum(mass) * row_weight[g,i]/sum(row_weight[g])``.
The objective averages squared value errors over outputs and rows. Each
derivative task also averages over outputs, and either averages or sums over
its input coordinates, before dividing by its declared task denominator.
Repeating every row of a group leaves its probability measure unchanged.

Jacobians are partial derivatives with respect to the supplied raw input vector
at fixed inputs, including fixed successor states. They are not derivatives
through the trajectory used to generate those states. Generating successors,
checking reference validity and qualifying equilibrium accuracy belong to the
caller; a successful linear fit is only an initializer, not a solver certificate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from .generic_policy_coordinates import AffinePolicyCoordinates, TanhCoordinateMap
from .generic_training_contracts import ImmutableJSONMapping, stable_hash


def _immutable_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers")
    array = np.asarray(array, dtype="<f8")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be representable in float64")
    return np.frombuffer(array.tobytes(), dtype="<f8").reshape(array.shape)


def _positive_scalar(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a positive finite scalar")
    result = float(array)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _probabilities(weights):
    scaled = weights / np.max(weights)
    probabilities = scaled / np.sum(scaled)
    if not np.all(probabilities > 0):
        raise ValueError("weight ratios exceed float64 precision")
    return probabilities


def _array_binding(array):
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


@dataclass(frozen=True, eq=False)
class CoverageGroup:
    """Owned training rows with explicit relative mass and fixed-input targets.

    ``inputs`` is [N,D], ``values`` is [N,O], ``raw_jacobians`` is [N,O,D],
    and ``row_weights`` is [N]. All arrays and nested metadata are immutable.
    ``role`` must be "training"; ``location`` is "current" or "successor".
    Relative row weights and ``mass`` must be strictly positive and finite.
    """

    group_id: str
    location: str
    role: str
    inputs: np.ndarray
    values: np.ndarray
    raw_jacobians: np.ndarray
    row_weights: np.ndarray
    mass: float
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("nonempty group_id required")
        if self.role != "training":
            raise ValueError("coverage initialization accepts training-only groups")
        if self.location not in ("current", "successor"):
            raise ValueError("location must be current or successor")
        for name in ("inputs", "values", "raw_jacobians", "row_weights"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), name))
        if self.inputs.ndim != 2 or min(self.inputs.shape) == 0:
            raise ValueError("inputs must have nonempty shape [N,D]")
        count, input_dim = self.inputs.shape
        if self.values.ndim != 2 or self.values.shape[0] != count or self.values.shape[1] == 0:
            raise ValueError("values must have shape [N,O] with O positive")
        if self.raw_jacobians.shape != (count, self.values.shape[1], input_dim):
            raise ValueError("raw_jacobians must have shape [N,O,D]")
        if self.row_weights.shape != (count,) or not np.all(self.row_weights > 0):
            raise ValueError("row_weights must be positive with shape [N]")
        object.__setattr__(self, "mass", _positive_scalar(self.mass, "mass"))
        object.__setattr__(self, "metadata", ImmutableJSONMapping(self.metadata))

    def binding_hash(self):
        return stable_hash({
            "schema": "generic_neural_solver.coverage_group.v1",
            "group_id": self.group_id, "location": self.location, "role": self.role,
            "mass": self.mass, "metadata": self.metadata,
            **{name: _array_binding(getattr(self, name)) for name in
               ("inputs", "values", "raw_jacobians", "row_weights")},
        })

    def to_dict(self):
        return {
            "schema": "generic_neural_solver.coverage_group.v1",
            "group_id": self.group_id, "location": self.location, "role": self.role,
            "mass": self.mass, "metadata": self.metadata.to_dict(),
            **{name: getattr(self, name).tolist() for name in
               ("inputs", "values", "raw_jacobians", "row_weights")},
            "binding_hash": self.binding_hash(),
        }

    @classmethod
    def from_dict(cls, value):
        payload = dict(value)
        if payload.pop("schema", None) != "generic_neural_solver.coverage_group.v1":
            raise ValueError("unknown coverage group schema")
        expected = payload.pop("binding_hash", None)
        group = cls(**payload)
        if group.binding_hash() != expected:
            raise ValueError("coverage group binding mismatch")
        return group


@dataclass(frozen=True)
class CoverageTrainingData:
    """A union with normalized masses, independent of each group's row count.

    A group may be current-only for legacy parity. When successors are supplied,
    their configured mass controls their influence rather than their number.
    Duplicate group IDs are rejected. Equal input coordinates in distinct
    declared groups retain their explicit masses; values are not silently
    deduplicated because targets and provenance may differ.
    """

    groups: tuple[CoverageGroup, ...]

    def __post_init__(self):
        groups = tuple(self.groups)
        if not groups or not all(isinstance(group, CoverageGroup) for group in groups):
            raise ValueError("nonempty CoverageGroup collection required")
        if len({group.group_id for group in groups}) != len(groups):
            raise ValueError("duplicate coverage group_id")
        shapes = {(group.inputs.shape[1], group.values.shape[1]) for group in groups}
        if len(shapes) != 1:
            raise ValueError("coverage groups must share input and output dimensions")
        object.__setattr__(self, "groups", groups)

    def assemble(self):
        """Return immutable inputs, values, raw Jacobians and row probabilities."""
        masses = _probabilities(np.array([group.mass for group in self.groups]))
        weights = np.concatenate([
            mass * _probabilities(group.row_weights)
            for mass, group in zip(masses, self.groups)
        ])
        if not np.all(weights > 0):
            raise ValueError("combined weight ratios exceed float64 precision")
        arrays = [np.concatenate([getattr(group, name) for group in self.groups])
                  for name in ("inputs", "values", "raw_jacobians")]
        return tuple(_immutable_array(array, "assembled data") for array in (*arrays, weights))

    def binding_hash(self):
        return stable_hash({"schema": "generic_neural_solver.coverage_training_data.v1",
                            "groups": [group.binding_hash() for group in self.groups]})

    def to_dict(self):
        return {"schema": "generic_neural_solver.coverage_training_data.v1",
                "groups": [group.to_dict() for group in self.groups],
                "binding_hash": self.binding_hash()}

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != "generic_neural_solver.coverage_training_data.v1":
            raise ValueError("unknown coverage training data schema")
        data = cls(tuple(CoverageGroup.from_dict(group) for group in value["groups"]))
        if data.binding_hash() != value.get("binding_hash"):
            raise ValueError("coverage training data binding mismatch")
        return data


def fit_coverage_coordinates(data, *, input_scale_floor=1e-6, output_scale_floor=1e-6):
    """Fit weighted population moments of the declared current/successor union.

    Floors are explicit positive absolute scalars or vectors, broadcast over
    input/output coordinates. They only define affine coordinates, not loss
    denominators. The returned profile binds the complete training snapshot.
    """
    inputs, values, _jacobians, weights = data.assemble()
    centers, scales = [], []
    for rows, floor in ((inputs, input_scale_floor), (values, output_scale_floor)):
        floor = _immutable_array(floor, "coordinate scale floor")
        if floor.ndim > 1 or (floor.ndim == 1 and floor.shape != (rows.shape[1],)):
            raise ValueError("coordinate scale floor must be scalar or match the dimension")
        if not np.all(floor > 0):
            raise ValueError("coordinate scale floors must be positive")
        center = np.sum(weights[:, None] * rows, axis=0)
        variance = np.sum(weights[:, None] * np.square(rows - center), axis=0)
        centers.append(tuple(center))
        scales.append(tuple(np.maximum(np.sqrt(variance), floor)))
    return AffinePolicyCoordinates(centers[0], scales[0], centers[1], scales[1], data.binding_hash())


def make_coverage_feature_parameters(profile, width, seed):
    """Return seeded fixed features and a zero output layer in coordinate space.

    Hidden weights have standard deviations sqrt(2/(input_dim + width)) and
    sqrt(1/width). Both hidden biases use Normal(0, 0.5) to break the forced odd
    symmetry of zero-bias tanh features about the input center. The bias scale
    is an explicit initialization candidate, not a universal accuracy guarantee.
    The initial raw prediction equals ``profile.output_center``. Randomness is
    local to the seed; the returned packed float64 parameters are immutable.
    """
    mapping = TanhCoordinateMap(profile, width)
    if type(seed) is not int or seed < 0:
        raise ValueError("feature seed must be a nonnegative integer")
    generator = np.random.default_rng(seed)
    input_dim = len(profile.input_center)
    parameters = np.concatenate((
        generator.normal(0.0, np.sqrt(2.0 / (input_dim + width)), mapping.sizes[0]),
        generator.normal(0.0, 0.5, mapping.sizes[1]),
        generator.normal(0.0, np.sqrt(1.0 / width), mapping.sizes[2]),
        generator.normal(0.0, 0.5, mapping.sizes[3]),
        np.zeros(sum(mapping.sizes[4:])),
    ))
    return _immutable_array(parameters, "feature parameters")


def _feature_rows(parameters, inputs, mapping):
    profile, width = mapping.profile, mapping.hidden_width
    input_dim = len(profile.input_center)
    weight0, bias0, weight1, bias1, _weight2, _bias2 = tf.split(parameters, mapping.sizes)
    first = tf.reshape(weight0, [input_dim, width])
    second = tf.reshape(weight1, [width, width])
    scale = tf.constant(profile.input_scale, tf.float64)
    hidden0 = tf.tanh(tf.matmul((inputs - profile.input_center) / scale, first) + bias0)
    hidden1 = tf.tanh(tf.matmul(hidden0, second) + bias1)
    derivative0 = ((1.0 - tf.square(hidden0))[:, :, None]
                   * tf.transpose(first)[None, :, :] / scale[None, None, :])
    derivative1 = (tf.einsum("nwd,wh->nhd", derivative0, second)
                   * (1.0 - tf.square(hidden1))[:, :, None])
    count = tf.shape(inputs)[0]
    features = tf.concat([hidden1, tf.ones([count, 1], tf.float64)], axis=1)
    derivatives = tf.concat([derivative1, tf.zeros([count, 1, input_dim], tf.float64)], axis=1)
    return features, derivatives


def make_coverage_predictor(profile, hidden_width):
    """Return one traced evaluator: (coordinate parameters, raw inputs) -> (y, dy/dx)."""
    mapping = TanhCoordinateMap(profile, hidden_width)
    input_dim, output_dim = len(profile.input_center), len(profile.output_center)

    @tf.function(input_signature=[tf.TensorSpec([mapping.parameter_dim], tf.float64),
                                  tf.TensorSpec([None, input_dim], tf.float64)], autograph=False)
    def predict(parameters, inputs):
        features, derivatives = _feature_rows(parameters, inputs, mapping)
        coefficients = tf.reshape(parameters[sum(mapping.sizes[:4]):], [hidden_width + 1, output_dim])
        raw_coefficients = coefficients * tf.constant(profile.output_scale, tf.float64)[None, :]
        values = tf.matmul(features, raw_coefficients) + profile.output_center
        jacobians = tf.einsum("nfd,fo->nod", derivatives, raw_coefficients)
        return values, jacobians

    return predict


@dataclass(frozen=True, eq=False)
class CoverageWarmStartResult:
    """Linear-fit diagnostics; ``valid`` does not assert equilibrium accuracy."""

    parameters: np.ndarray
    design: np.ndarray
    target: np.ndarray
    raw_coefficients: np.ndarray
    singular_values: np.ndarray
    task_losses: np.ndarray
    normal_residual: float
    rank: int
    valid: bool
    data_hash: str
    profile_hash: str

    def __post_init__(self):
        for name in ("parameters", "design", "target", "raw_coefficients", "singular_values", "task_losses"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), name))


def fit_coverage_warm_start(profile, hidden_width, parameters, data, denominators,
                            derivative_split, *, state_reduction="mean",
                            parameter_reduction="mean", rcond=1e-12):
    """Fit the last layer once using analytic features and a thin SVD.

    ``denominators`` contains positive value/state/parameter MSE denominators.
    Inputs before ``derivative_split`` are states; the rest are parameters.
    The reductions affect derivative coordinates only; outputs are averaged.
    Both "mean" matches the legacy SGU objective; both "sum" matches Rotemberg's
    local coordinate reduction. The profile must bind this training snapshot.
    """
    mapping = TanhCoordinateMap(profile, hidden_width)
    input_dim, output_dim = len(profile.input_center), len(profile.output_center)
    if type(derivative_split) is not int or not 0 < derivative_split < input_dim:
        raise ValueError("derivative split must be inside the input dimension")
    if state_reduction not in ("mean", "sum") or parameter_reduction not in ("mean", "sum"):
        raise ValueError("derivative coordinate reductions must be mean or sum")
    rcond = _positive_scalar(rcond, "rcond")
    if rcond >= 1:
        raise ValueError("rcond must be below one")
    parameters = _immutable_array(parameters, "parameters")
    denominators = _immutable_array(denominators, "denominators")
    if parameters.shape != (mapping.parameter_dim,):
        raise ValueError("parameters must match the TanhCoordinateMap dimension")
    if denominators.shape != (3,) or not np.all(denominators > 0):
        raise ValueError("three positive task denominators required")
    inputs, values, jacobians, weights = data.assemble()
    if inputs.shape[1] != input_dim or values.shape[1] != output_dim:
        raise ValueError("coordinate profile and coverage dimensions disagree")
    data_hash = data.binding_hash()
    if profile.training_binding != data_hash:
        raise ValueError("coordinate profile training binding mismatch")
    coordinate_counts = [1, derivative_split if state_reduction == "mean" else 1,
                          input_dim - derivative_split if parameter_reduction == "mean" else 1]
    with np.errstate(over="ignore", under="ignore"):
        divisor_values = denominators * output_dim * np.array(coordinate_counts)
    if not np.all(np.isfinite(divisor_values)) or not np.all(divisor_values > 0):
        raise ValueError("task denominator and coordinate counts exceed float64 range")
    divisors = tf.constant(divisor_values, tf.float64)

    @tf.function(input_signature=[
        tf.TensorSpec([mapping.parameter_dim], tf.float64),
        tf.TensorSpec([None, input_dim], tf.float64),
        tf.TensorSpec([None, output_dim], tf.float64),
        tf.TensorSpec([None, output_dim, input_dim], tf.float64),
        tf.TensorSpec([None], tf.float64),
    ], autograph=False)
    def fit(initial, rows, targets, raw_jacobians, row_probabilities):
        features, derivatives = _feature_rows(initial, rows, mapping)
        task_weights = tf.sqrt(row_probabilities[:, None] / divisors[None, :])
        tf.debugging.assert_all_finite(task_weights, "nonfinite task row weights")
        tf.debugging.assert_positive(task_weights, "task row weights underflowed")
        designs = [features * task_weights[:, :1]]
        target_blocks = [(targets - profile.output_center) * task_weights[:, :1]]
        for task_index, start, stop in ((1, 0, derivative_split), (2, derivative_split, input_dim)):
            row_scales = task_weights[:, task_index, None, None]
            designs.append(tf.reshape(tf.transpose(derivatives[:, :, start:stop], [0, 2, 1])
                                      * row_scales, [-1, hidden_width + 1]))
            target_blocks.append(tf.reshape(tf.transpose(raw_jacobians[:, :, start:stop], [0, 2, 1])
                                            * row_scales, [-1, output_dim]))
        design, target = tf.concat(designs, axis=0), tf.concat(target_blocks, axis=0)
        tf.debugging.assert_all_finite(design, "nonfinite weighted feature design")
        tf.debugging.assert_all_finite(target, "nonfinite weighted targets")
        singular, left, right = tf.linalg.svd(design, full_matrices=False)
        retained = singular > tf.constant(rcond, tf.float64) * tf.reduce_max(singular)
        inverse = tf.where(retained, tf.math.reciprocal(tf.where(retained, singular, 1.0)), 0.0)
        raw_coefficients = tf.matmul(right * inverse[None, :], tf.matmul(left, target, transpose_a=True))
        coefficients = raw_coefficients / tf.constant(profile.output_scale, tf.float64)[None, :]
        fitted = tf.concat([initial[:sum(mapping.sizes[:4])], tf.reshape(coefficients, [-1])], axis=0)
        residual = tf.matmul(design, raw_coefficients) - target
        normal = (tf.linalg.norm(tf.matmul(design, residual, transpose_a=True))
                  / (1.0 + tf.linalg.norm(design) * tf.linalg.norm(target)))
        losses = tf.stack([tf.reduce_sum(tf.square(tf.matmul(block, raw_coefficients) - desired))
                           for block, desired in zip(designs, target_blocks)])
        valid = (tf.reduce_all(tf.math.is_finite(fitted)) & tf.reduce_all(tf.math.is_finite(residual))
                 & tf.math.is_finite(normal) & (normal < tf.constant(1e-10, tf.float64)))
        return fitted, design, target, raw_coefficients, singular, losses, normal, tf.reduce_sum(tf.cast(retained, tf.int32)), valid

    result = [item.numpy() for item in fit(parameters, inputs, values, jacobians, weights)]
    return CoverageWarmStartResult(*result[:6], float(result[6]), int(result[7]), bool(result[8]),
                                   data_hash, profile.binding_hash())
