"""Optional equality witnesses from public and supplied component gradients."""

import math

import tensorflow as tf


def component_descent_binding(relative_tolerance):
    return {"schema": "generic_neural_solver.component_descent.v1",
        "solver": "float64-thin-svd-equality-witness", "optimality_claim": False,
        "row_normalization": "stable-unit-norm", "active_target": -1., "protected_target": 0.,
        "rank_cutoff": "float64-epsilon-times-max-shape-times-largest-singular-value",
        "residual_check": "l2-residual-over-max-one-target-norm",
        "active_check": "strict-negative-scaled-unit-dot",
        "protected_check": "absolute-scaled-unit-dot-relative-to-source-norm",
        "relative_tolerance": float(relative_tolerance), "scale": "source-final-displacement-norm"}


def _stable_norm(vector):
    maximum = tf.reduce_max(tf.abs(vector))
    scaled = vector / tf.where(maximum > 0., maximum, 1.)
    return maximum * tf.linalg.norm(scaled)


def component_descent_impl(public_rows, active, component_rows, source_displacement, *, relative_tolerance=1e-12):
    """Construct A^+b and check achieved slopes; invalid is not infeasibility.

    Rows are already in their owners' raw/D coordinates. All supplied rows
    belong to active tasks. Consistent redundant equalities are allowed.
    """
    if not isinstance(relative_tolerance, (float, int)) or not math.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError("finite relative tolerance in [0, 1) required")
    public_rows, active, component_rows, source_displacement = (
        tf.convert_to_tensor(value) for value in (public_rows, active, component_rows, source_displacement))
    if active.dtype != tf.bool or any(value.dtype != tf.float64 for value in (
            public_rows, component_rows, source_displacement)):
        raise TypeError("float64 rows/displacement and Boolean public active mask required")
    checks = [tf.debugging.assert_rank(public_rows, 2), tf.debugging.assert_rank(component_rows, 2),
        tf.debugging.assert_rank(active, 1), tf.debugging.assert_rank(source_displacement, 1),
        tf.debugging.assert_positive(tf.size(source_displacement)), tf.debugging.assert_positive(tf.size(active)),
        tf.debugging.assert_equal(tf.shape(public_rows)[0], tf.size(active)),
        tf.debugging.assert_equal(tf.shape(public_rows)[1], tf.size(source_displacement)),
        tf.debugging.assert_equal(tf.shape(component_rows)[1], tf.size(source_displacement))]
    with tf.control_dependencies(checks):
        rows = tf.concat((public_rows, component_rows), axis=0)
        active_rows = tf.concat((active, tf.ones([tf.shape(component_rows)[0]], tf.bool)), axis=0)
        inputs_finite = tf.reduce_all(tf.math.is_finite(rows)) & tf.reduce_all(tf.math.is_finite(source_displacement))
        safe_rows = tf.where(tf.math.is_finite(rows), rows, 0.)
        maximum = tf.reduce_max(tf.abs(safe_rows), axis=1, keepdims=True)
        scaled_rows = safe_rows / tf.where(maximum > 0., maximum, 1.)
        norms = tf.linalg.norm(scaled_rows, axis=1, keepdims=True)
        unit_rows = scaled_rows / tf.where(norms > 0., norms, 1.)
    targets = -tf.cast(active_rows, tf.float64)
    singular, left, right = tf.linalg.svd(unit_rows, full_matrices=False)
    cutoff = (tf.constant(2.220446049250313e-16, tf.float64)
        * tf.cast(tf.reduce_max(tf.shape(unit_rows)), tf.float64) * tf.reduce_max(singular))
    retained = singular > cutoff
    coordinates = tf.linalg.matvec(left, targets, transpose_a=True)
    coefficients = tf.where(retained, coordinates / tf.where(retained, singular, 1.), 0.)
    witness = tf.linalg.matvec(right, coefficients)
    equality_dots = tf.linalg.matvec(unit_rows, witness)
    residual = _stable_norm(equality_dots - targets)
    relative_residual = residual / tf.maximum(tf.linalg.norm(targets), 1.)
    witness_norm = _stable_norm(witness)
    source_norm = _stable_norm(source_displacement)
    safe_witness_norm = tf.where(tf.math.is_finite(witness_norm) & (witness_norm > 0.), witness_norm, 1.)
    safe_source_norm = tf.where(tf.math.is_finite(source_norm) & (source_norm > 0.), source_norm, 1.)
    direction = (witness / safe_witness_norm) * safe_source_norm
    direction_norm = _stable_norm(direction)
    dots = tf.linalg.matvec(unit_rows, direction)
    tolerance = tf.constant(float(relative_tolerance), tf.float64)
    valid = (inputs_finite & tf.reduce_any(active) & tf.math.is_finite(witness_norm) & (witness_norm > 0.)
        & tf.math.is_finite(source_norm) & (source_norm > 0.) & tf.math.is_finite(direction_norm) & (direction_norm > 0.)
        & tf.reduce_all(tf.math.is_finite(direction)) & tf.reduce_all(tf.math.is_finite(dots))
        & tf.math.is_finite(relative_residual) & (relative_residual <= tolerance)
        & tf.reduce_all(tf.boolean_mask(dots, active_rows) < 0.)
        & tf.reduce_all(tf.abs(tf.boolean_mask(dots, ~active_rows)) <= tolerance * source_norm)
        & (tf.abs(direction_norm - source_norm) <= tolerance * source_norm))
    return {"direction": direction, "valid": valid, "rank": tf.reduce_sum(tf.cast(retained, tf.int32)),
        "singular_values": singular, "rank_cutoff": cutoff, "equality_dots": equality_dots,
        "equality_residual": residual, "relative_residual": relative_residual, "witness_norm": witness_norm,
        "source_norm": source_norm, "direction_norm": direction_norm, "dots": dots}


def make_component_descent_function(task_count, *, relative_tolerance=1e-12):
    """One non-pfor graph supporting supplied row counts without retracing."""
    if type(task_count) is not int or task_count < 1:
        raise ValueError("positive public task count required")
    if not isinstance(relative_tolerance, (float, int)) or not math.isfinite(relative_tolerance) or not 0 <= relative_tolerance < 1:
        raise ValueError("finite relative tolerance in [0, 1) required")
    return tf.function(lambda public_rows, active, component_rows, source_displacement: component_descent_impl(
        public_rows, active, component_rows, source_displacement, relative_tolerance=relative_tolerance),
        autograph=False, jit_compile=False, input_signature=[
            tf.TensorSpec([task_count, None], tf.float64), tf.TensorSpec([task_count], tf.bool),
            tf.TensorSpec([None, None], tf.float64), tf.TensorSpec([None], tf.float64)])
