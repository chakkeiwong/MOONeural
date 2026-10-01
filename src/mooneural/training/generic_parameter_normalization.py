"""Host contracts for parameter chain rules and explicit metric normalization."""

import math
from dataclasses import dataclass

import numpy as np

from .generic_training_contracts import (
    ImmutableJSONMapping,
    MetricCoordinates,
    NormalizationMetadata,
    stable_hash,
)


def _array(value, name):
    array = np.asarray(value)
    if array.dtype != np.float64 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float64")
    return array


def _owned(value, name):
    array = _array(value, name).copy()
    array.flags.writeable = False
    return array


def _names(values, name):
    values = tuple(values)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain nonempty axis names")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be distinct")
    return values


@dataclass(frozen=True)
class ParameterTransformPoint:
    raw_names: tuple[str, ...]
    physical_names: tuple[str, ...]
    raw: np.ndarray
    physical: np.ndarray
    jacobian: np.ndarray
    recipe: ImmutableJSONMapping

    def __post_init__(self):
        object.__setattr__(self, "raw_names", _names(self.raw_names, "raw_names"))
        object.__setattr__(self, "physical_names", _names(self.physical_names, "physical_names"))
        for name in ("raw", "physical", "jacobian"):
            object.__setattr__(self, name, _owned(getattr(self, name), name))
        if (self.raw.shape != (len(self.raw_names),) or self.physical.shape != (len(self.physical_names),)
                or self.jacobian.shape != (len(self.physical_names), len(self.raw_names))):
            raise ValueError("parameter transform dimensions disagree")
        recipe = ImmutableJSONMapping(self.recipe)
        if not recipe:
            raise ValueError("parameter transform requires an explicit recipe binding")
        object.__setattr__(self, "recipe", recipe)

    def binding_hash(self):
        return stable_hash({"raw_names": self.raw_names, "physical_names": self.physical_names,
            "raw": self.raw.tolist(), "physical": self.physical.tolist(), "jacobian": self.jacobian.tolist(),
            "recipe": self.recipe})


def local_parameter_pullback(partial_theta, transform):
    if not isinstance(transform, ParameterTransformPoint):
        raise TypeError("an explicit parameter transform point is required")
    partial = _array(partial_theta, "partial_theta")
    if partial.ndim < 2 or partial.shape[-1] != len(transform.physical_names) or 0 in partial.shape:
        raise ValueError("local parameter partial must have axes [...,output,physical_parameter]")
    return _array(partial @ transform.jacobian, "raw_parameter_partial")


def composed_parameter_pullback(direct_theta, argument_jacobian, argument_theta, transform):
    direct = _array(direct_theta, "direct_theta")
    partial = _array(argument_jacobian, "argument_jacobian")
    motion = _array(argument_theta, "argument_theta")
    if (direct.ndim < 2 or partial.ndim != direct.ndim or motion.ndim != direct.ndim
            or partial.shape[:-2] != direct.shape[:-2] or motion.shape[:-2] != direct.shape[:-2]
            or partial.shape[-2] != direct.shape[-2] or partial.shape[-1] != motion.shape[-2]
            or motion.shape[-1] != direct.shape[-1] or 0 in partial.shape or 0 in motion.shape):
        raise ValueError("composed derivative argument and leading axes disagree")
    return local_parameter_pullback(direct + partial @ motion, transform)


def divide_jet(value, derivative, denominator, denominator_derivative):
    value, derivative = _array(value, "value"), _array(derivative, "derivative")
    denominator = _array(denominator, "denominator")
    denominator_derivative = _array(denominator_derivative, "denominator_derivative")
    if (denominator.shape != value.shape or derivative.shape != denominator_derivative.shape
            or derivative.ndim != value.ndim+1 or derivative.shape[:-1] != value.shape
            or derivative.shape[-1] == 0 or 0 in value.shape or np.any(denominator <= 0)):
        raise ValueError("quotient jet axes or positive denominator invalid")
    normalized = value/denominator
    partial = (derivative-normalized[..., None]*denominator_derivative)/denominator[..., None]
    return _array(normalized, "normalized_value"), _array(partial, "normalized_derivative")


@dataclass(frozen=True)
class ReductionAxes:
    names: tuple[str, ...]
    mean_axes: tuple[str, ...]
    sum_axes: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "names", _names(self.names, "tensor axis names"))
        object.__setattr__(self, "mean_axes", tuple(self.mean_axes))
        object.__setattr__(self, "sum_axes", tuple(self.sum_axes))
        contracted = self.mean_axes+self.sum_axes
        if len(contracted) != len(set(contracted)) or set(contracted) != set(self.names):
            raise ValueError("each residual axis must be declared exactly once as mean or sum")


@dataclass(frozen=True)
class RawMetricJet:
    value: float
    derivative: np.ndarray
    reduction: ImmutableJSONMapping

    def __post_init__(self):
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("raw squared metric must be finite and nonnegative")
        derivative = _owned(self.derivative, "raw_metric_derivative")
        if derivative.ndim != 1 or derivative.size == 0:
            raise ValueError("raw metric derivative needs a nonempty parameter axis")
        object.__setattr__(self, "derivative", derivative)
        object.__setattr__(self, "reduction", ImmutableJSONMapping(self.reduction))


def squared_error_jet(error, error_derivative, reduction):
    error, partial = _array(error, "error"), _array(error_derivative, "error_derivative")
    if not isinstance(reduction, ReductionAxes):
        raise TypeError("explicit named reduction axes required")
    if (error.ndim != len(reduction.names) or 0 in error.shape or partial.ndim != error.ndim+1
            or partial.shape[:-1] != error.shape or partial.shape[-1] == 0):
        raise ValueError("residual and derivative reduction axes disagree")
    mean_count = math.prod(error.shape[reduction.names.index(name)] for name in reduction.mean_axes)
    derivative = np.sum(2*error[..., None]*partial, axis=tuple(range(error.ndim)))/mean_count
    return RawMetricJet(float(np.sum(error*error)/mean_count), derivative,
        ImmutableJSONMapping({"axis_names": reduction.names, "shape": error.shape,
            "mean_axes": reduction.mean_axes, "sum_axes": reduction.sum_axes, "mean_count": mean_count}))


def inverse_mse_scale(denominator, denominator_derivative):
    denominator = _array(denominator, "MSE denominator")
    derivative = _array(denominator_derivative, "MSE denominator derivative")
    if denominator.shape != () or derivative.ndim != 1:
        raise ValueError("MSE denominator must be scalar with one parameter derivative axis")
    factor, partial = divide_jet(np.array(1., dtype=np.float64), np.zeros_like(derivative), denominator, derivative)
    return float(factor), partial


@dataclass(frozen=True)
class ScaledMetricJet:
    coordinates: MetricCoordinates
    derivative: np.ndarray

    def __post_init__(self):
        if not isinstance(self.coordinates, MetricCoordinates):
            raise TypeError("existing metric coordinate contract required")
        derivative = _owned(self.derivative, "scaled_metric_derivative")
        if derivative.ndim != 1 or derivative.size == 0:
            raise ValueError("scaled derivative needs a nonempty parameter axis")
        object.__setattr__(self, "derivative", derivative)


def apply_metric_scale(raw_metric, normalization, coordinate, factor_derivative):
    if type(raw_metric) is not RawMetricJet:
        raise TypeError("scale application starts from RawMetricJet exactly once")
    if not isinstance(normalization, NormalizationMetadata):
        raise TypeError("existing normalization registry required")
    if coordinate not in ("training", "selection", "terminal"):
        raise ValueError("select an explicit non-raw metric role")
    partial = _array(factor_derivative, "factor_derivative")
    if partial.shape != raw_metric.derivative.shape:
        raise ValueError("metric and role factor derivative axes disagree")
    scale = normalization.scale_for(coordinate)
    coordinates = MetricCoordinates(task_id=normalization.task_id, raw=raw_metric.value,
        training=raw_metric.value*normalization.training.factor,
        selection=raw_metric.value*normalization.selection.factor,
        terminal=raw_metric.value*normalization.terminal.factor, normalization=normalization,
        applied_scale_ids=(scale.scale_id,), coordinate=coordinate)
    return ScaledMetricJet(coordinates, scale.factor*raw_metric.derivative+raw_metric.value*partial)
