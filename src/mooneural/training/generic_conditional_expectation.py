"""Gaussian inputs and float64 conditional-mean loss reductions.

Innovations have independent standard-normal coordinates. Models apply their
own transitions, covariance factors and nonlinear transformations *before*
integration, and supply total pathwise parameter Jacobians. Quadrature accuracy
and interchange of differentiation and expectation need separate qualification.
No samples are rejected, weights repaired, or negative diagnostics clipped here.
"""

import hashlib
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import tensorflow as tf

from .generic_training_contracts import stable_hash

_SCHEMA = "generic_neural_solver.gaussian_expectation.v1"
_WEIGHT_TOLERANCE = 64 * np.finfo(np.float64).eps


def _integer(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class GaussianExpectationProfile:
    """An immutable integration recipe; seeds and roles belong to input batches.

    ``draws`` is the total count, including both halves of antithetic pairs.
    Tensor-product Gauss-Hermite uses ``order ** shock_dim`` points. Different
    roles have distinct random streams, but deterministic quadrature roles use
    identical nodes and cannot provide an independent Monte Carlo audit.
    """

    method: str
    shock_dim: int
    draws: int | None = None
    order: int | None = None

    def __post_init__(self):
        if self.method not in ("iid", "antithetic", "gauss_hermite"):
            raise ValueError("unknown Gaussian expectation method")
        _integer("shock_dim", self.shock_dim)
        if self.method == "gauss_hermite":
            _integer("order", self.order)
            if self.draws is not None:
                raise ValueError("Gauss-Hermite requires order, not draws")
        else:
            _integer("draws", self.draws)
            if self.order is not None:
                raise ValueError("Monte Carlo requires draws, not order")
            if self.method == "antithetic" and self.draws % 2:
                raise ValueError("antithetic draws must be even")

    @property
    def point_count(self):
        return self.order**self.shock_dim if self.method == "gauss_hermite" else self.draws

    def to_dict(self):
        return {
            "schema": _SCHEMA,
            "distribution": "standard_normal",
            "dtype": "float64",
            "method": self.method,
            "shock_dim": self.shock_dim,
            "draws": self.draws,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, value):
        payload = dict(value)
        if (
            payload.pop("schema", None) != _SCHEMA
            or payload.pop("distribution", None) != "standard_normal"
            or payload.pop("dtype", None) != "float64"
        ):
            raise ValueError("unknown Gaussian expectation schema, law or dtype")
        if set(payload) != {"method", "shock_dim", "draws", "order"}:
            raise ValueError("Gaussian expectation profile fields disagree")
        return cls(**payload)

    def binding_hash(self):
        return stable_hash(self.to_dict())


def _readonly_float64(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers")
    converted = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} cannot be represented in float64")
    return np.frombuffer(converted.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class GaussianExpectationInputs:
    """Prepared host inputs, with read-only arrays and reproducible stream identity.

    ``nodes`` has shape [rows, draws, shock_dim]; ``weights`` has shape [draws].
    Binding records describe preparation, not proof of downstream callable or
    integrand provenance. Separate stream identities do not prove independence
    if a consumer reuses evaluations or introduces shared auxiliary randomness.
    """

    profile: GaussianExpectationProfile
    nodes: np.ndarray
    weights: np.ndarray
    seed: int
    role: str
    stream: int = 0

    def __post_init__(self):
        if not isinstance(self.profile, GaussianExpectationProfile):
            raise TypeError("Gaussian expectation profile required")
        _integer("seed", self.seed, 0)
        _integer("stream", self.stream, 0)
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("role must be a nonempty string")
        nodes = _readonly_float64(self.nodes, "nodes")
        weights = _readonly_float64(self.weights, "weights")
        if (
            nodes.ndim != 3 or nodes.shape[0] == 0
            or nodes.shape[1:] != (self.profile.point_count, self.profile.shock_dim)
        ):
            raise ValueError("nodes must have shape [rows, profile points, shock_dim]")
        if weights.shape != (self.profile.point_count,):
            raise ValueError("weights must have shape [profile points]")
        if np.any(weights <= 0) or abs(float(weights.sum()) - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError("weights must be positive and normalized")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)

    @property
    def profile_fingerprint(self):
        return self.profile.binding_hash()

    @property
    def stream_key(self):
        return stable_hash({"rng": "numpy.PCG64", "seed": self.seed,
                            "role": self.role, "stream": self.stream})

    def binding_hash(self):
        return stable_hash({"profile": self.profile_fingerprint,
                            "stream": self.stream_key,
                            "nodes_shape": self.nodes.shape,
                            "nodes_sha256": hashlib.sha256(self.nodes.tobytes()).hexdigest(),
                            "weights_shape": self.weights.shape,
                            "weights_sha256": hashlib.sha256(self.weights.tobytes()).hexdigest()})

    def require_profile(self, profile):
        if not isinstance(profile, GaussianExpectationProfile):
            raise TypeError("Gaussian expectation profile required")
        if profile.binding_hash() != self.profile_fingerprint:
            raise ValueError("Gaussian expectation profile mismatch")


def prepare_gaussian_inputs(profile, *, rows, seed, role, stream=0, max_points):
    """Prepare inputs with an explicit per-row point budget and no global RNG.

    Antithetic nodes are [base, -base], with draws/2 independent base vectors.
    For Gaussian quadrature, physicists' Hermite nodes are multiplied by sqrt(2)
    and weights divided by sqrt(pi) in each independent shock coordinate.
    """

    if not isinstance(profile, GaussianExpectationProfile):
        raise TypeError("Gaussian expectation profile required")
    for name, value, minimum in (("rows", rows, 1), ("seed", seed, 0),
                                 ("stream", stream, 0), ("max_points", max_points, 1)):
        _integer(name, value, minimum)
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must be a nonempty string")
    count = profile.point_count
    if count > max_points:
        raise ValueError("Gaussian expectation exceeds max_points before allocation")
    if profile.method == "gauss_hermite":
        abscissas, one_weights = np.polynomial.hermite.hermgauss(profile.order)
        abscissas = abscissas * np.sqrt(2.0)
        one_weights = one_weights / np.sqrt(np.pi)
        indices = np.arange(count, dtype=np.int64)
        points = np.empty((count, profile.shock_dim), dtype=np.float64)
        weights = np.ones(count, dtype=np.float64)
        for coordinate in range(profile.shock_dim - 1, -1, -1):
            digit = indices % profile.order
            points[:, coordinate] = abscissas[digit]
            weights *= one_weights[digit]
            indices = indices // profile.order
        nodes = np.broadcast_to(points, (rows, count, profile.shock_dim))
    else:
        stream_key = stable_hash({"rng": "numpy.PCG64", "seed": seed,
                                  "role": role, "stream": stream})
        generator = np.random.Generator(np.random.PCG64(int(stream_key, 16)))
        independent_count = count // 2 if profile.method == "antithetic" else count
        nodes = generator.standard_normal((rows, independent_count, profile.shock_dim))
        if profile.method == "antithetic":
            nodes = np.concatenate((nodes, -nodes), axis=1)
        weights = np.full(count, 1.0 / count, dtype=np.float64)
    return GaussianExpectationInputs(profile, nodes, weights, seed, role, stream)


def antithetic_from_iid(base):
    """Return [base[:, :draws/2], -base[:, :draws/2]] without RNG or mutation.

    This host preparation operation retains float64 input values exactly. The
    caller binds original and effective arrays to its role/profile records. Half
    the retained draws are deliberately unused; the returned count is unchanged.
    """

    base = _readonly_float64(base, "base")
    if base.ndim != 3 or any(size == 0 for size in base.shape):
        raise ValueError("base must have nonempty shape [rows, draws, shocks]")
    if base.shape[1] % 2:
        raise ValueError("antithetic base draw count must be even")
    half = base[:, :base.shape[1] // 2, :]
    return _readonly_float64(np.concatenate((half, -half), axis=1), "antithetic nodes")


class ConditionalLossRows(NamedTuple):
    """Per-row raw and once-normalized losses, interleaved by equation."""

    raw: tf.Tensor
    normalized: tf.Tensor


class SignedAuditRows(NamedTuple):
    """Detached, possibly negative diagnostics; never acceptance or loss values."""

    raw: tf.Tensor
    normalized: tf.Tensor

    @property
    def diagnostic_only(self):
        return True


def _finite_nonempty(tensor, name):
    with tf.control_dependencies([
        tf.debugging.assert_positive(tf.shape(tensor), message=f"{name} axes must be nonempty"),
    ]):
        return tf.debugging.check_numerics(tensor, f"{name} must be finite")


def _checked_weights(weights, count):
    weights = _finite_nonempty(weights, "weights")
    with tf.control_dependencies([
        tf.debugging.assert_equal(tf.shape(weights)[0], count, message="weights/draws disagree"),
        tf.debugging.assert_positive(weights, message="weights must be positive"),
        tf.debugging.assert_less_equal(
            tf.abs(tf.reduce_sum(weights) - tf.constant(1.0, tf.float64)),
            tf.constant(_WEIGHT_TOLERANCE, tf.float64), message="weights must be normalized",
        ),
    ]):
        return tf.identity(weights)


def _weighted_mean(values, weights):
    count = tf.shape(values)[1]
    uniform = tf.reduce_all(tf.equal(weights, tf.ones_like(weights) / tf.cast(count, tf.float64)))
    weight_shape = [1, -1] + [1] * (values.shape.rank - 2)
    return tf.cond(
        uniform,
        lambda: tf.reduce_mean(values, axis=1),
        lambda: tf.reduce_sum(values * tf.reshape(weights, weight_shape), axis=1),
    )


@tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float64),
                              tf.TensorSpec([None], tf.float64)], autograph=False)
def _conditional_mean(integrand, weights):
    integrand = _finite_nonempty(integrand, "integrand")
    weights = _checked_weights(weights, tf.shape(integrand)[1])
    return tf.debugging.check_numerics(_weighted_mean(integrand, weights), "conditional mean overflow")


def _tensors_and_weights(integrand, weights):
    integrand = tf.ensure_shape(tf.convert_to_tensor(integrand, dtype=tf.float64), [None, None, None])
    if weights is None:
        count = tf.shape(integrand)[1]
        weights = tf.ones([count], tf.float64) / tf.cast(count, tf.float64)
    return integrand, tf.convert_to_tensor(weights, dtype=tf.float64)


def conditional_mean(integrand, weights=None):
    """Integrate [rows, draws, equations], preserving reduce_mean for uniform weights.

    No square or nonlinear outer transform is applied. Models must place any
    nonlinear integrand transformation before this call. Weights are one shared
    strictly positive normalized vector [draws]; per-row broadcasting is refused.
    """

    return _conditional_mean(*_tensors_and_weights(integrand, weights))


def _checked_loss_inputs(integrand, jacobian, projection, denominators, weights):
    integrand = _finite_nonempty(integrand, "integrand")
    jacobian = _finite_nonempty(jacobian, "jacobian")
    projection = _finite_nonempty(projection, "projection")
    denominators = _finite_nonempty(denominators, "denominators")
    weights = _checked_weights(weights, tf.shape(integrand)[1])
    with tf.control_dependencies([
        tf.debugging.assert_equal(tf.shape(jacobian)[:3], tf.shape(integrand),
                                  message="jacobian/integrand axes disagree"),
        tf.debugging.assert_equal(tf.shape(projection)[:2], tf.gather(tf.shape(integrand), [0, 2]),
                                  message="projection rows/equations disagree"),
        tf.debugging.assert_equal(tf.shape(projection)[3], tf.shape(jacobian)[3],
                                  message="projection/parameter axes disagree"),
        tf.debugging.assert_equal(tf.shape(denominators)[0], 2 * tf.shape(integrand)[2],
                                  message="denominators must be interleaved residual/derivative tasks"),
        tf.debugging.assert_positive(denominators, message="denominators must be positive"),
    ]):
        return tuple(tf.identity(value) for value in
                     (integrand, jacobian, projection, denominators, weights))


def _loss_means(integrand, jacobian, projection, weights):
    """Keep each equation's reduce_mean and projection operation order intact."""

    def equation_means(equation):
        mean = _weighted_mean(integrand[:, :, equation], weights)
        derivative = _weighted_mean(jacobian[:, :, equation, :], weights)
        projected = tf.einsum("bpd,bd->bp", projection[:, equation, :, :], derivative)
        return mean, projected

    mean, projected = tf.map_fn(
        equation_means, tf.range(tf.shape(integrand)[2]),
        fn_output_signature=(tf.TensorSpec([None], tf.float64),
                             tf.TensorSpec([None, None], tf.float64)),
        parallel_iterations=1,
    )
    return tf.transpose(mean, [1, 0]), tf.transpose(projected, [1, 0, 2])


def _interleave(residual, derivative_terms, derivative_mean):
    derivative = tf.cond(
        derivative_mean,
        lambda: tf.reduce_mean(derivative_terms, axis=2),
        lambda: tf.reduce_sum(derivative_terms, axis=2),
    )
    return tf.reshape(tf.stack((residual, derivative), axis=2), [tf.shape(residual)[0], -1])


_LOSS_SIGNATURE = [
    tf.TensorSpec([None, None, None], tf.float64),
    tf.TensorSpec([None, None, None, None], tf.float64),
    tf.TensorSpec([None, None, None, None], tf.float64),
    tf.TensorSpec([None], tf.float64),
    tf.TensorSpec([None], tf.float64),
    tf.TensorSpec([], tf.bool),
]


@tf.function(input_signature=_LOSS_SIGNATURE, autograph=False)
def _loss_rows(integrand, jacobian, projection, denominators, weights, derivative_mean):
    integrand, jacobian, projection, denominators, weights = _checked_loss_inputs(
        integrand, jacobian, projection, denominators, weights,
    )
    mean, projected = _loss_means(integrand, jacobian, projection, weights)
    raw = _interleave(tf.square(mean), tf.square(projected), derivative_mean)
    raw = tf.debugging.check_numerics(raw, "raw conditional loss overflow")
    normalized = tf.debugging.check_numerics(raw / denominators[None, :], "normalized loss overflow")
    return ConditionalLossRows(raw, normalized)


def _derivative_mean(reduction):
    if reduction not in ("mean", "sum"):
        raise ValueError("derivative_reduction must be mean or sum")
    return tf.constant(reduction == "mean")


def reduce_conditional_losses(integrand, jacobian, projection, denominators,
                              *, weights=None, derivative_reduction="mean"):
    """Square conditional means, project mean Jacobians, then normalize once.

    Shapes: integrand [rows, draws, equations], jacobian [rows, draws,
    equations, params], projection [rows, equations, projections, params],
    denominators [2 * equations]. Projection precedes mean/sum of its squared
    components. Task order is residual_0, derivative_0, residual_1, derivative_1,
    etc. Outputs retain the outer row axis: no row averaging is performed.

    For IID draws the positive square includes Var(integrand)/draws in its
    expectation. This function does not remove that penalty. Antithetic draws
    are dependent; the IID variance formula must not be applied to them.
    """

    integrand, weights = _tensors_and_weights(integrand, weights)
    return _loss_rows(integrand, tf.convert_to_tensor(jacobian, tf.float64),
                      tf.convert_to_tensor(projection, tf.float64),
                      tf.convert_to_tensor(denominators, tf.float64), weights,
                      _derivative_mean(derivative_reduction))


@tf.function(input_signature=_LOSS_SIGNATURE[:2] + _LOSS_SIGNATURE[:2]
             + _LOSS_SIGNATURE[2:5] + [_LOSS_SIGNATURE[4], _LOSS_SIGNATURE[5]], autograph=False)
def _signed_rows(left, left_jacobian, right, right_jacobian, projection,
                 denominators, left_weights, right_weights, derivative_mean):
    left, left_jacobian, projection, denominators, left_weights = _checked_loss_inputs(
        left, left_jacobian, projection, denominators, left_weights,
    )
    right, right_jacobian, projection, denominators, right_weights = _checked_loss_inputs(
        right, right_jacobian, projection, denominators, right_weights,
    )
    left_mean, left_projected = _loss_means(left, left_jacobian, projection, left_weights)
    right_mean, right_projected = _loss_means(right, right_jacobian, projection, right_weights)
    residual = left_mean * right_mean
    derivative = left_projected * right_projected
    raw = tf.debugging.check_numerics(_interleave(residual, derivative, derivative_mean),
                                     "signed audit overflow")
    normalized = tf.debugging.check_numerics(raw / denominators[None, :], "normalized audit overflow")
    return SignedAuditRows(tf.stop_gradient(raw), tf.stop_gradient(normalized))


def independent_signed_audit(left_integrand, left_jacobian, right_integrand, right_jacobian,
                             projection, denominators, *, left_inputs, right_inputs,
                             derivative_reduction="mean"):
    """Diagnostic product of separately estimated conditional means, without clipping.

    Inputs must describe distinct stochastic streams at identical conditioning
    rows under the same shock law. The estimated *complete residuals* and their
    projected derivatives must themselves be unbiased for the declared targets.
    Unbiased inner moments do not ensure this after a nonlinear recursion.
    Conditional independence gives E[left * right] = E[left] * E[right]; it
    cannot remove bias already present in either estimated residual or derivative.
    Rejection, correlated auxiliary randomness or reused evaluations also
    invalidate the unbiased squared-target interpretation.
    Pairing opposite halves of an antithetic batch is therefore refused. Each
    side may instead be its own independently generated antithetic batch.

    Distinct stream labels do not certify independence. Deterministic quadrature
    is refused here: separate labels cannot make fixed quadrature rules into
    independent unbiased residual estimators or certify their integration error.

    These products of means differ from legacy paired per-draw residual products.
    For identical marginal residual means, both estimators have the same
    expectation under independent sides.
    Both outputs are detached TensorFlow diagnostics, not optimization targets.
    """

    if not all(isinstance(value, GaussianExpectationInputs) for value in (left_inputs, right_inputs)):
        raise TypeError("audit requires prepared Gaussian expectation inputs")
    if any(value.profile.method == "gauss_hermite" for value in (left_inputs, right_inputs)):
        raise ValueError("deterministic quadrature is not an independent stochastic audit")
    if left_inputs.profile.shock_dim != right_inputs.profile.shock_dim:
        raise ValueError("audit shock dimensions disagree")
    if left_inputs.stream_key == right_inputs.stream_key:
        raise ValueError("audit requires independent streams")
    left, left_weights = _tensors_and_weights(left_integrand, left_inputs.weights)
    right, right_weights = _tensors_and_weights(right_integrand, right_inputs.weights)
    with tf.control_dependencies([
        tf.debugging.assert_equal(tf.shape(left)[:2], left_inputs.nodes.shape[:2],
                                  message="left audit inputs do not match integrand"),
        tf.debugging.assert_equal(tf.shape(right)[:2], right_inputs.nodes.shape[:2],
                                  message="right audit inputs do not match integrand"),
    ]):
        return _signed_rows(tf.identity(left), tf.convert_to_tensor(left_jacobian, tf.float64),
                            tf.identity(right), tf.convert_to_tensor(right_jacobian, tf.float64),
                            tf.convert_to_tensor(projection, tf.float64),
                            tf.convert_to_tensor(denominators, tf.float64), left_weights,
                            right_weights, _derivative_mean(derivative_reduction))
