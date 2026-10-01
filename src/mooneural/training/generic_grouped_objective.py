"""Explicit raw-MSE aggregation before per-task training normalization."""


def reduce_grouped_objectives(raw_rows, probabilities, group_ids, group_count, *, aggregation):
    """Reduce float64 rows by probability or by the worst conditional group.

    ``raw_rows`` has shape [rows, tasks]. Positive row probabilities sum to
    one and integer group IDs assign every row to one nonempty group. The
    weighted mode preserves the original sum without renormalization. The
    maximum mode first computes conditional group means and then selects the
    maximum independently for each task, with equal subgradient shares at
    exact ties. This function applies neither D nor T and changes no default.

    Call inside a stable adapter graph with an explicit input signature.
    Group definition and aggregation choice belong in its objective binding.
    A maximum of training group means is not a bound on unseen populations,
    a Student-t upper bound, or a finite-displacement feasibility guarantee.
    """
    if aggregation not in ("weighted_mean", "max_group_mean"):
        raise ValueError("explicit weighted_mean or max_group_mean aggregation required")
    if type(group_count) is not int or group_count < 1:
        raise ValueError("positive integer group_count required")

    import tensorflow as tf

    raw_rows = tf.convert_to_tensor(raw_rows)
    probabilities = tf.convert_to_tensor(probabilities)
    group_ids = tf.convert_to_tensor(group_ids)
    if raw_rows.dtype != tf.float64 or probabilities.dtype != tf.float64:
        raise TypeError("raw rows and probabilities must be float64")
    if group_ids.dtype not in (tf.int32, tf.int64):
        raise TypeError("group IDs must be integers")
    checks = [
        tf.debugging.assert_rank(raw_rows, 2),
        tf.debugging.assert_rank(probabilities, 1),
        tf.debugging.assert_rank(group_ids, 1),
        tf.debugging.assert_equal(tf.shape(raw_rows)[0], tf.shape(probabilities)[0]),
        tf.debugging.assert_equal(tf.shape(raw_rows)[0], tf.shape(group_ids)[0]),
        tf.debugging.assert_positive(tf.shape(raw_rows)[1]),
        tf.debugging.assert_all_finite(raw_rows, "nonfinite raw task MSE"),
        tf.debugging.assert_non_negative(raw_rows),
        tf.debugging.assert_all_finite(probabilities, "nonfinite row probabilities"),
        tf.debugging.assert_positive(probabilities),
        tf.debugging.assert_near(tf.reduce_sum(probabilities), tf.constant(1., tf.float64),
            rtol=0., atol=1e-12),
        tf.debugging.assert_non_negative(group_ids),
        tf.debugging.assert_less(group_ids, tf.cast(group_count, group_ids.dtype)),
    ]
    with tf.control_dependencies(checks):
        weighted_rows = raw_rows * probabilities[:, None]
        group_mass = tf.math.unsorted_segment_sum(probabilities, group_ids, group_count)
    with tf.control_dependencies([tf.debugging.assert_positive(group_mass, message="empty objective group")]):
        if aggregation == "weighted_mean":
            return tf.reduce_sum(weighted_rows, axis=0)
        group_totals = tf.math.unsorted_segment_sum(weighted_rows, group_ids, group_count)
        return tf.reduce_max(group_totals / group_mass[:, None], axis=0)
