"""Optional linearized displacement guard; changes no optimizer or default."""


def blend_displacements(proposal, reference, gradient_rows, active, *, margin_fraction, relative_tolerance=1e-12):
    """Choose the least reference mixing that satisfies declared linear bounds.

    Unit task rows remove arbitrary positive task scaling. Active rows target
    ``margin_fraction * min(row @ reference, 0)``; other rows target zero.
    Half the displacement-relative rounding allowance enters the interval;
    half is reserved for evaluating the rounded blend. Solve the intersection
    of those halfspaces with the segment joining the proposal and reference.
    An empty interval returns the unchanged proposal
    with valid=False; callers must not label it guarded or accept it on that
    basis. No finite-loss, unseen-cell or statistical guarantee is implied.

    This primitive neither updates moments nor selects a training policy. Use
    inside a fixed-signature adapter/update graph and bind its optional profile.
    """
    if not isinstance(margin_fraction, (float, int)) or not 0 <= margin_fraction <= 1:
        raise ValueError("explicit margin fraction in [0,1] required")
    if not isinstance(relative_tolerance, (float, int)) or not 0 <= relative_tolerance < 1:
        raise ValueError("relative rounding allowance in [0,1) required")
    import tensorflow as tf

    proposal, reference = tf.convert_to_tensor(proposal), tf.convert_to_tensor(reference)
    gradient_rows, active = tf.convert_to_tensor(gradient_rows), tf.convert_to_tensor(active)
    if any(value.dtype != tf.float64 for value in (proposal, reference, gradient_rows)) or active.dtype != tf.bool:
        raise TypeError("float64 vectors/rows and Boolean active mask required")
    checks = [tf.debugging.assert_rank(proposal, 1), tf.debugging.assert_rank(reference, 1),
        tf.debugging.assert_rank(gradient_rows, 2), tf.debugging.assert_rank(active, 1),
        tf.debugging.assert_equal(tf.shape(proposal), tf.shape(reference)),
        tf.debugging.assert_equal(tf.shape(gradient_rows)[1], tf.size(proposal)),
        tf.debugging.assert_equal(tf.shape(gradient_rows)[0], tf.size(active)),
        tf.debugging.assert_positive(tf.size(proposal)), tf.debugging.assert_positive(tf.size(active)),
        *[tf.debugging.assert_all_finite(value, "nonfinite displacement guard input")
            for value in (proposal, reference, gradient_rows)]]
    with tf.control_dependencies(checks):
        maximum = tf.reduce_max(tf.abs(gradient_rows), axis=1, keepdims=True)
        scaled = gradient_rows / tf.where(maximum > 0., maximum, 1.)
        norm = tf.linalg.norm(scaled, axis=1, keepdims=True)
        rows = scaled / tf.where(norm > 0., norm, 1.)
        proposed_slopes = tf.linalg.matvec(rows, proposal)
        reference_slopes = tf.linalg.matvec(rows, reference)
    limits = tf.where(active, tf.constant(float(margin_fraction), tf.float64) * tf.minimum(reference_slopes, 0.), 0.)

    def stable_norm(vector):
        magnitude = tf.reduce_max(tf.abs(vector))
        return magnitude * tf.linalg.norm(vector / tf.where(magnitude > 0., magnitude, 1.))

    slack = tf.constant(float(relative_tolerance), tf.float64) * tf.maximum(stable_norm(proposal), stable_norm(reference))
    interval_slack = .5 * slack
    difference = reference_slopes - proposed_slopes
    ratio = (limits + interval_slack - proposed_slopes) / tf.where(difference != 0., difference, 1.)
    lower = tf.maximum(tf.reduce_max(tf.where(difference < 0., ratio, 0.)), tf.constant(0., tf.float64))
    upper = tf.minimum(tf.reduce_min(tf.where(difference > 0., ratio, 1.)), tf.constant(1., tf.float64))
    constant_valid = tf.reduce_all(tf.where(difference == 0., proposed_slopes <= limits + interval_slack, True))
    candidate = (1. - lower) * proposal + lower * reference
    candidate_slopes = tf.linalg.matvec(rows, candidate)
    valid = (lower <= upper) & constant_valid & tf.reduce_all(candidate_slopes <= limits + slack)
    valid = valid & tf.reduce_all(tf.math.is_finite(candidate))
    valid = valid & tf.math.is_finite(slack) & tf.reduce_all(tf.math.is_finite(
        tf.stack((proposed_slopes, reference_slopes, difference, limits))))
    displacement = tf.where(valid, candidate, proposal)
    return {"displacement": displacement, "valid": valid, "fraction": tf.where(valid, lower, 0.),
        "lower": lower, "upper": upper, "slopes": tf.linalg.matvec(rows, displacement),
        "limits": limits, "slack": slack, "interval_slack": interval_slack,
        "proposal_slopes": proposed_slopes, "reference_slopes": reference_slopes}
