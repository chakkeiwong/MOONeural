"""Explicit native source-order projection with an XLA packed-Adam child."""

from __future__ import annotations

from itertools import combinations

import tensorflow as tf

from .generic_execution_boundary import (
    DEFAULT_ADAM,
    _dimension,
    _make_projected_adam_function,
)
from .generic_xla_kernels import project_cone_impl
from .adam import packed_adam_impl


def native_update_binding():
    return {
        "schema": "generic_neural_solver.update_backend.v1",
        "profile": "tf_native_source_projection__xla_packed_adam.v1",
        "outer_jit_compile": False,
        "projector_jit_compile": False,
        "packed_adam_jit_compile": True,
        "projection": {
            "arithmetic": "source_gathered_subset_gram_pinv_rhs",
            "subset_order": "increasing_size_then_combinations",
            "dtype": "float64",
            "rcond": 1e-12,
            "feasibility_tolerance": 1e-10,
            "tie_rule": "first_minimum",
            "pre_and_post_adam": True,
        },
        "legacy_guard": {
            "proposal_veto": True,
            "residual": "legacy_jacobi_diagnostic",
            "qualification": "diagnostic_only_pending_source_failure_domain_coverage",
        },
    }


def _source_projection_impl(direction, protected_rows, constraint_count):
    if constraint_count == 0:
        return direction
    tolerance = tf.constant(1.0e-10, tf.float64)
    norms = tf.linalg.norm(protected_rows, axis=1, keepdims=True)
    normalized = protected_rows / norms
    candidates = [direction]
    for size in range(1, constraint_count + 1):
        for indices in combinations(range(constraint_count), size):
            selected = tf.gather(normalized, indices)
            gram = tf.linalg.matmul(selected, selected, transpose_b=True)
            rhs = -tf.linalg.matvec(selected, direction)
            multipliers = tf.linalg.matvec(
                tf.linalg.pinv(gram, rcond=tf.constant(1.0e-12, tf.float64)), rhs
            )
            candidates.append(
                direction + tf.linalg.matvec(selected, multipliers, transpose_a=True)
            )
    candidates.append(tf.zeros_like(direction))
    stacked = tf.stack(candidates, axis=0)
    directional = tf.linalg.matmul(stacked, normalized, transpose_b=True)
    feasible = tf.reduce_all(directional >= -tolerance, axis=1)
    distance = tf.reduce_sum(tf.square(stacked - direction[None, :]), axis=1)
    distance = tf.where(
        feasible,
        distance,
        tf.fill(tf.shape(distance), tf.constant(float("inf"), tf.float64)),
    )
    return stacked[tf.argmin(distance, output_type=tf.int32)]


def make_native_projection_function(parameter_dim, constraint_count):
    _dimension("parameter_dim", parameter_dim, 1)
    _dimension("constraint_count", constraint_count, 0, 6)
    signature = (
        tf.TensorSpec([parameter_dim], tf.float64, "direction"),
        tf.TensorSpec([constraint_count, parameter_dim], tf.float64, "protected_rows"),
        tf.TensorSpec([(1 << constraint_count) - 1, constraint_count], tf.bool, "subset_masks"),
        tf.TensorSpec([7, 4, 2], tf.int32, "schedule"),
        tf.TensorSpec([], tf.float64, "rcond"),
    )

    @tf.function(input_signature=signature, autograph=False, jit_compile=False)
    def project(direction, protected_rows, subset_masks, schedule, rcond):
        selected = _source_projection_impl(direction, protected_rows, constraint_count)
        normalized = protected_rows / tf.linalg.norm(protected_rows, axis=1, keepdims=True)
        submitted = tf.linalg.matvec(normalized, selected)
        legacy = project_cone_impl(direction, protected_rows, subset_masks, schedule, rcond)
        valid = (
            legacy[3]
            & tf.reduce_all(tf.math.is_finite(selected))
            & tf.reduce_all(tf.math.is_finite(submitted))
        )
        return selected, submitted, legacy[2], valid

    return project


def make_native_projected_adam_function(parameter_dim, constraint_count, config=DEFAULT_ADAM):
    projector = make_native_projection_function(parameter_dim, constraint_count)
    signature = [
        tf.TensorSpec([parameter_dim], tf.float64, "parameters"),
        tf.TensorSpec([parameter_dim], tf.float64, "first_moment"),
        tf.TensorSpec([parameter_dim], tf.float64, "second_moment"),
        tf.TensorSpec([], tf.int64, "iteration"),
        tf.TensorSpec([parameter_dim], tf.float64, "gradient"),
        *[
            tf.TensorSpec([], tf.float64, name)
            for name in ("learning_rate", "clip_norm", "beta1", "beta2", "epsilon")
        ],
    ]
    packed = tf.function(
        packed_adam_impl, input_signature=signature, autograph=False, jit_compile=True
    )
    return _make_projected_adam_function(
        parameter_dim, constraint_count, config,
        projector_impl=projector, packed_impl=packed, jit_compile=False,
    )
