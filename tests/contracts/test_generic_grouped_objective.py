"""Numerical contract for explicit optional conditional-group loss reduction."""

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_grouped_objective import reduce_grouped_objectives


def test_weighted_mode_preserves_original_bits_and_stable_trace():
    @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float64)], autograph=False)
    def reduce(rows):
        return reduce_grouped_objectives(rows, tf.constant([.125, .375, .5], tf.float64),
            tf.constant([0, 0, 1]), 2, aggregation="weighted_mean")

    for multiplier in (1., 7.):
        rows = tf.constant([[1e-17, 9.], [5., 1e9], [3., 4.]], tf.float64) * multiplier
        expected = tf.reduce_sum(rows * tf.constant([.125, .375, .5], tf.float64)[:, None], axis=0)
        assert reduce(rows).numpy().tobytes() == expected.numpy().tobytes()
    assert reduce.experimental_get_tracing_count() == 1


def test_conditional_means_use_group_mass_and_task_specific_worst_group():
    rows = tf.constant([[1., 8.], [5., 4.], [3., 10.]], tf.float64)
    actual = reduce_grouped_objectives(rows, tf.constant([.125, .375, .5], tf.float64),
        tf.constant([0, 0, 1]), 2, aggregation="max_group_mean")
    np.testing.assert_array_equal(actual.numpy(), [4., 10.])


def test_weighted_decrease_can_hide_worse_conditional_group():
    probabilities = tf.constant([.75, .25], tf.float64)
    groups = tf.constant([0, 1])
    before = tf.constant([[2.], [4.]], tf.float64)
    after = tf.constant([[1.], [5.]], tf.float64)
    reductions = {mode: [float(reduce_grouped_objectives(rows, probabilities, groups, 2,
        aggregation=mode).numpy()[0]) for rows in (before, after)]
        for mode in ("weighted_mean", "max_group_mean")}
    assert reductions["weighted_mean"] == [2.5, 2.]
    assert reductions["max_group_mean"] == [4., 5.]


@pytest.mark.parametrize("parameters,expected", [(2., 4.), (.5, -3.), (1., 0.)])
def test_analytic_weight_gradient_and_equal_share_tie(parameters, expected):
    variable = tf.Variable(parameters, dtype=tf.float64)
    with tf.GradientTape() as tape:
        rows = tf.reshape(tf.stack((variable**2, (variable - 2.)**2)), (2, 1))
        loss = reduce_grouped_objectives(rows, tf.constant([.75, .25], tf.float64),
            tf.constant([0, 1]), 2, aggregation="max_group_mean")[0]
    assert float(tape.gradient(loss, variable)) == expected


@pytest.mark.parametrize("rows,probabilities,groups,count", [
    ([[1.], [2.]], [.5, .5], [0, 0], 2),
    ([[1.], [2.]], [.5, .5], [0, 2], 2),
    ([[1.], [2.]], [.5, .4], [0, 1], 2),
    ([[1.], [2.]], [1., 0.], [0, 1], 2),
    ([[float("nan")], [2.]], [.5, .5], [0, 1], 2),
    ([[-1.], [2.]], [.5, .5], [0, 1], 2),
    ([[1.], [2.]], [1.], [0, 1], 2),
])
def test_invalid_inputs_are_refused(rows, probabilities, groups, count):
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        reduce_grouped_objectives(tf.constant(rows, tf.float64), tf.constant(probabilities, tf.float64),
            tf.constant(groups), count, aggregation="max_group_mean")


def test_dtype_and_implicit_aggregation_are_refused():
    with pytest.raises(ValueError, match="explicit"):
        reduce_grouped_objectives([[1.]], [1.], [0], 1, aggregation="automatic")
    with pytest.raises(TypeError, match="float64"):
        reduce_grouped_objectives(tf.constant([[1.]], tf.float32), tf.constant([1.], tf.float64),
            tf.constant([0]), 1, aggregation="weighted_mean")
