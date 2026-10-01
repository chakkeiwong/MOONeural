"""Optional relative public-loss descent with a bounded feasible-seed QP."""

from __future__ import annotations

import math

import tensorflow as tf

from mooneural.training.generic_component_descent import _stable_norm


def relative_progress_impl(public_values, public_rows, descent_mask, component_values,
                           component_rows, owner_indices, source_displacement, *,
                           relative_tolerance=1e-10, tie_rtol=1e-11, max_iterations=200,
                           required_components=None, component_target_weights=None):
    """Refine a feasible equality seed; failure does not prove infeasibility."""
    if (not math.isfinite(relative_tolerance) or not 0. < relative_tolerance < 1.
            or not math.isfinite(tie_rtol) or not 0. <= tie_rtol < 1.
            or not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 0):
        raise ValueError("finite tolerances and a nonnegative integer iteration limit required")
    public_values, public_rows, descent_mask, component_values, component_rows, owner_indices, source_displacement = (
        tf.convert_to_tensor(value) for value in (public_values, public_rows, descent_mask,
            component_values, component_rows, owner_indices, source_displacement))
    if (any(value.dtype != tf.float64 for value in (public_values, public_rows, component_values,
            component_rows, source_displacement)) or descent_mask.dtype != tf.bool
            or owner_indices.dtype not in (tf.int32, tf.int64)):
        raise TypeError("float64 numerical arrays, Boolean mask and integer owners required")
    checks = [tf.debugging.assert_rank(public_values, 1), tf.debugging.assert_rank(public_rows, 2),
        tf.debugging.assert_rank(descent_mask, 1), tf.debugging.assert_rank(component_values, 1),
        tf.debugging.assert_rank(component_rows, 2), tf.debugging.assert_rank(owner_indices, 1),
        tf.debugging.assert_rank(source_displacement, 1), tf.debugging.assert_positive(tf.size(public_values)),
        tf.debugging.assert_positive(tf.size(source_displacement)),
        tf.debugging.assert_equal(tf.shape(public_rows)[0], tf.size(public_values)),
        tf.debugging.assert_equal(tf.size(descent_mask), tf.size(public_values)),
        tf.debugging.assert_equal(tf.shape(component_rows)[0], tf.size(component_values)),
        tf.debugging.assert_equal(tf.size(owner_indices), tf.size(component_values)),
        tf.debugging.assert_equal(tf.shape(public_rows)[1], tf.size(source_displacement)),
        tf.debugging.assert_equal(tf.shape(component_rows)[1], tf.size(source_displacement)),
        tf.debugging.assert_non_negative(owner_indices),
        tf.debugging.assert_less(owner_indices, tf.cast(tf.size(public_values), owner_indices.dtype))]
    if required_components is not None:
        required_components = tf.convert_to_tensor(required_components)
        if required_components.dtype != tf.bool:
            raise TypeError("Boolean required-component mask required")
        checks.extend((tf.debugging.assert_rank(required_components, 1),
            tf.debugging.assert_equal(tf.size(required_components), tf.size(component_values))))
    if component_target_weights is not None:
        component_target_weights = tf.convert_to_tensor(component_target_weights)
        if component_target_weights.dtype != tf.float64:
            raise TypeError("float64 component target weights required")
        checks.extend((tf.debugging.assert_rank(component_target_weights, 1),
            tf.debugging.assert_equal(tf.size(component_target_weights), tf.size(component_values)),
            tf.debugging.assert_all_finite(component_target_weights, "finite component target weights required"),
            tf.debugging.assert_positive(component_target_weights)))
    with tf.control_dependencies(checks):
        rows = tf.concat((public_rows, component_rows), axis=0)
        values_finite = tf.reduce_all(tf.math.is_finite(public_values)) & tf.reduce_all(tf.math.is_finite(component_values))
        inputs_finite = values_finite & tf.reduce_all(tf.math.is_finite(rows)) & tf.reduce_all(tf.math.is_finite(source_displacement))
        safe_rows = tf.where(tf.math.is_finite(rows), rows, 0.)
        maximum = tf.reduce_max(tf.abs(safe_rows), axis=1, keepdims=True)
        scaled = safe_rows / tf.where(maximum > 0., maximum, 1.)
        scaled_norms = tf.linalg.norm(scaled, axis=1, keepdims=True)
        unit_rows = scaled / tf.where(scaled_norms > 0., scaled_norms, 1.)
        row_maximum = tf.reshape(maximum, [-1])
        row_scaled_norms = tf.reshape(scaled_norms, [-1])
    tolerance = tf.constant(relative_tolerance, tf.float64)
    epsilon = tf.constant(2.220446049250313e-16, tf.float64)
    owner_values = tf.gather(public_values, owner_indices)
    tied = tf.abs(component_values - owner_values) <= tf.constant(tie_rtol, tf.float64) * tf.abs(owner_values)
    component_targets = tf.where(tied, owner_values, 0.)
    if required_components is not None:
        component_targets = tf.where(tied, owner_values, tf.where(required_components, component_values, 0.))
        inputs_finite &= tf.reduce_all(tf.boolean_mask(component_values, required_components) > 0.)
    required = tf.concat((tf.where(descent_mask, public_values, 0.), component_targets), axis=0)
    required = tf.where(tf.math.is_finite(required), required, 0.)
    relative_rows = required > 0.
    ratios = (required / tf.where(row_maximum > 0., row_maximum, 1.)) / tf.where(row_scaled_norms > 0., row_scaled_norms, 1.)
    targets_valid = tf.reduce_all(tf.boolean_mask(tf.math.is_finite(ratios) & (ratios > 0.), relative_rows))
    unweighted_ratios = ratios
    if component_target_weights is not None:
        target_weights = tf.concat((tf.ones(tf.shape(public_values), tf.float64), component_target_weights), axis=0)
        ratios *= target_weights
        targets_valid &= tf.reduce_all(tf.boolean_mask(tf.math.is_finite(ratios) & (ratios > 0.), relative_rows))
    targets = tf.where(tf.math.is_finite(ratios) & (ratios > 0.), ratios, 0.)
    seed_descent = tf.concat((descent_mask, tf.ones(tf.shape(component_values), tf.bool)), axis=0)
    seed_targets = -tf.cast(seed_descent, tf.float64)
    singular, left, basis = tf.linalg.svd(unit_rows, full_matrices=False)
    cutoff = epsilon * tf.cast(tf.reduce_max(tf.shape(unit_rows)), tf.float64) * tf.reduce_max(singular)
    retained = singular > cutoff
    coefficients = tf.where(retained,
        tf.linalg.matvec(left, seed_targets, transpose_a=True) / tf.where(retained, singular, 1.), 0.)
    reduced = left * tf.where(retained, singular, 0.)[None, :]
    seed_residual = _stable_norm(tf.linalg.matvec(reduced, coefficients) - seed_targets) / tf.maximum(_stable_norm(seed_targets), 1.)
    seed_norm = _stable_norm(coefficients)
    initial = coefficients / tf.where(seed_norm > 0., seed_norm, 1.)
    rates = -tf.linalg.matvec(reduced, initial) / tf.where(relative_rows, targets, 1.)
    scale = tf.reduce_min(tf.where(relative_rows, rates, tf.constant(float("inf"), tf.float64)))
    scale = tf.where(tf.math.is_finite(scale) & (scale > 0.), scale, 0.)
    bounds = -scale * targets
    source_norm = _stable_norm(tf.where(tf.math.is_finite(source_displacement), source_displacement, 0.))
    seed_valid = (inputs_finite & targets_valid & tf.reduce_any(descent_mask)
        & tf.reduce_all(tf.boolean_mask(public_values, descent_mask) > 0.)
        & tf.reduce_all(public_values >= 0.) & tf.reduce_all(component_values >= 0.)
        & tf.reduce_all(tf.gather(descent_mask, owner_indices))
        & tf.reduce_all(component_values <= owner_values + tf.constant(tie_rtol, tf.float64) * tf.abs(owner_values))
        & tf.math.is_finite(source_norm) & (source_norm > 0.) & tf.math.is_finite(seed_norm) & (seed_norm > 0.)
        & (seed_residual <= tolerance) & (scale > 0.)
        & tf.reduce_all(tf.boolean_mask(row_maximum, seed_descent) > 0.))
    active = tf.abs(tf.linalg.matvec(reduced, initial) - bounds) <= epsilon * 256.
    row_count = tf.shape(unit_rows)[0]

    def face(coordinates, active_mask):
        selected = tf.where(active_mask[:, None], reduced, 0.)
        face_singular, face_left, face_basis = tf.linalg.svd(selected, full_matrices=False)
        face_cutoff = tf.sqrt(epsilon * tf.cast(row_count, tf.float64)) * tf.reduce_max(face_singular)
        keep = face_singular > face_cutoff
        projected = tf.linalg.matvec(face_basis, coordinates, transpose_a=True)
        solved = tf.linalg.matvec(face_left, tf.where(keep, projected / tf.where(keep, face_singular, 1.), 0.))
        direction = -coordinates + tf.linalg.matvec(face_basis, tf.where(keep, projected, 0.))
        return direction, -solved

    def condition(iteration, coordinates, active_mask, converged):
        return seed_valid & ~converged & (iteration < max_iterations)

    def body(iteration, coordinates, active_mask, converged):
        search, multipliers = face(coordinates, active_mask)
        stationary = _stable_norm(search) <= epsilon * 256. * tf.maximum(_stable_norm(coordinates), 1.)
        smallest = tf.reduce_min(tf.where(active_mask, multipliers, tf.constant(float("inf"), tf.float64)))

        def at_face():
            remove = tf.argmin(tf.where(active_mask, multipliers, tf.constant(float("inf"), tf.float64)), output_type=tf.int32)
            finished = smallest >= -tolerance
            revised = active_mask & ~(tf.one_hot(remove, row_count, on_value=True, off_value=False) & ~finished)
            return coordinates, revised, finished

        def move():
            slopes = tf.linalg.matvec(reduced, search)
            slack = bounds - tf.linalg.matvec(reduced, coordinates)
            blocking = ~active_mask & (slopes > epsilon * 32. * tf.maximum(_stable_norm(search), 1e-30))
            ratios = tf.where(blocking, tf.maximum(slack, 0.) / tf.where(blocking, slopes, 1.),
                tf.constant(float("inf"), tf.float64))
            first = tf.argmin(ratios, output_type=tf.int32)
            fraction = tf.minimum(tf.reduce_min(ratios), 1.)
            revised = active_mask | (tf.one_hot(first, row_count, on_value=True, off_value=False) & (fraction < 1.))
            return coordinates + fraction * search, revised, tf.constant(False)

        updated, next_active, finished = tf.cond(stationary, at_face, move)
        return iteration + 1, updated, next_active, finished

    iterations, coordinates, active, converged = tf.while_loop(condition, body,
        (tf.constant(0), initial, active, tf.constant(False)), parallel_iterations=1)
    _search, multipliers = face(coordinates, active)
    primal_slack = tf.linalg.matvec(reduced, coordinates) - bounds
    stationarity = coordinates + tf.linalg.matvec(reduced, multipliers, transpose_a=True)
    coefficient_norm = _stable_norm(coordinates)
    parameter_direction = tf.linalg.matvec(basis, coordinates)
    direction = parameter_direction / tf.where(coefficient_norm > 0., coefficient_norm, 1.) * source_norm
    direction_norm = _stable_norm(direction)
    dots = tf.linalg.matvec(unit_rows, direction)
    achieved_rates = -dots / tf.where(relative_rows, targets, 1.)
    margin = tf.reduce_min(tf.where(relative_rows, achieved_rates, tf.constant(float("inf"), tf.float64)))
    margin = tf.where(tf.math.is_finite(margin), margin, 0.)
    expected_margin = source_norm * scale / tf.where(coefficient_norm > 0., coefficient_norm, 1.)
    constraint_violation = tf.reduce_max(dots + expected_margin * targets)
    valid = (seed_valid & converged & tf.math.is_finite(coefficient_norm) & (coefficient_norm > 0.)
        & tf.reduce_all(tf.math.is_finite(direction)) & tf.math.is_finite(direction_norm)
        & (tf.abs(direction_norm - source_norm) <= tolerance * source_norm)
        & (tf.reduce_max(primal_slack) <= tolerance)
        & (constraint_violation <= tolerance * source_norm)
        & tf.reduce_all(tf.boolean_mask(dots, relative_rows) < 0.)
        & (tf.reduce_min(multipliers) >= -tolerance)
        & (_stable_norm(stationarity) <= tolerance * tf.maximum(coefficient_norm, 1.))
        & (tf.reduce_max(tf.abs(multipliers * primal_slack)) <= tolerance))
    result = {"direction": direction, "valid": valid, "seed_valid": seed_valid, "targets_valid": targets_valid,
        "converged": converged,
        "iterations": iterations, "rank": tf.reduce_sum(tf.cast(retained, tf.int32)),
        "seed_relative_residual": seed_residual, "source_norm": source_norm, "direction_norm": direction_norm,
        "minimum_fractional_progress": margin, "expected_fractional_progress": expected_margin,
        "primal_violation": tf.reduce_max(primal_slack), "full_row_violation": constraint_violation,
        "stationarity_residual": _stable_norm(stationarity), "minimum_multiplier": tf.reduce_min(multipliers),
        "complementarity_residual": tf.reduce_max(tf.abs(multipliers * primal_slack)),
        "tied_components": tied, "relative_target_rows": relative_rows,
        "unit_direction_dots": dots, "coefficient_norm": coefficient_norm, "target_scale": scale}
    if component_target_weights is not None:
        actual_rates = -dots / tf.where(relative_rows & (unweighted_ratios > 0.), unweighted_ratios, 1.)
        actual_margin = tf.reduce_min(tf.where(relative_rows, actual_rates, tf.constant(float("inf"), tf.float64)))
        result.update(weighted_minimum_fractional_progress=margin,
            weighted_expected_fractional_progress=expected_margin,
            minimum_fractional_progress=tf.where(tf.math.is_finite(actual_margin), actual_margin, 0.),
            expected_fractional_progress=expected_margin * tf.reduce_min(
                tf.where(relative_rows, target_weights, tf.constant(float("inf"), tf.float64))))
    return result


def make_relative_progress_function(task_count, *, relative_tolerance=1e-10, tie_rtol=1e-11, max_iterations=200,
                                    explicit_component_targets=False, explicit_component_weights=False):
    """Create one stable graph for a declared public task count."""
    if type(explicit_component_targets) is not bool:
        raise TypeError("explicit_component_targets must be Boolean")
    if type(explicit_component_weights) is not bool:
        raise TypeError("explicit_component_weights must be Boolean")
    if explicit_component_weights:
        if not explicit_component_targets:
            raise ValueError("explicit component weights require explicit component targets")
        return tf.function(lambda public_values, public_rows, descent_mask, component_values, component_rows,
                owner_indices, source_displacement, required_components, component_target_weights:
                    relative_progress_impl(public_values, public_rows, descent_mask, component_values,
                        component_rows, owner_indices, source_displacement,
                        relative_tolerance=relative_tolerance, tie_rtol=tie_rtol, max_iterations=max_iterations,
                        required_components=required_components, component_target_weights=component_target_weights),
            input_signature=[tf.TensorSpec([task_count], tf.float64), tf.TensorSpec([task_count, None], tf.float64),
                tf.TensorSpec([task_count], tf.bool), tf.TensorSpec([None], tf.float64),
                tf.TensorSpec([None, None], tf.float64), tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.bool),
                tf.TensorSpec([None], tf.float64)], autograph=False)
    if explicit_component_targets:
        return tf.function(lambda public_values, public_rows, descent_mask, component_values, component_rows,
                owner_indices, source_displacement, required_components: relative_progress_impl(public_values,
                    public_rows, descent_mask, component_values, component_rows, owner_indices, source_displacement,
                    relative_tolerance=relative_tolerance, tie_rtol=tie_rtol, max_iterations=max_iterations,
                    required_components=required_components),
            input_signature=[tf.TensorSpec([task_count], tf.float64), tf.TensorSpec([task_count, None], tf.float64),
                tf.TensorSpec([task_count], tf.bool), tf.TensorSpec([None], tf.float64),
                tf.TensorSpec([None, None], tf.float64), tf.TensorSpec([None], tf.int32),
                tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.bool)], autograph=False)
    return tf.function(lambda public_values, public_rows, descent_mask, component_values, component_rows,
            owner_indices, source_displacement: relative_progress_impl(public_values, public_rows, descent_mask,
                component_values, component_rows, owner_indices, source_displacement,
                relative_tolerance=relative_tolerance, tie_rtol=tie_rtol, max_iterations=max_iterations),
        input_signature=[tf.TensorSpec([task_count], tf.float64), tf.TensorSpec([task_count, None], tf.float64),
            tf.TensorSpec([task_count], tf.bool), tf.TensorSpec([None], tf.float64),
            tf.TensorSpec([None, None], tf.float64), tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.float64)],
        autograph=False)
