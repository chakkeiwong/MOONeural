"""Synthetic tensor adapters for boundary qualification, not economic solvers."""

from __future__ import annotations

import tensorflow as tf

from .generic_training_contracts import (
    NormalizationMetadata,
    ScaleSpec,
    TaskDefinition,
    TaskRegistry,
)


class QuadraticTensorFixture:
    """Row-local least squares: reverse-mode gradients without jacobian or pfor."""

    def __init__(self, task_count=2, hard_check_count=1, training_factors=None):
        self.registry = TaskRegistry(
            tuple(
                TaskDefinition(f"synthetic-{index}", index)
                for index in range(task_count)
            )
        )
        self.hard_check_ids = tuple(
            f"synthetic-hard-{index}" for index in range(hard_check_count)
        )
        factors = (
            tuple(1.0 for _task in range(task_count))
            if training_factors is None
            else tuple(training_factors)
        )
        if len(factors) != task_count:
            raise ValueError("training factors must match the task registry")
        self.normalizations = tuple(
            NormalizationMetadata(
                task.task_id,
                ScaleSpec(f"raw-{index}", "raw", 1.0),
                ScaleSpec(f"training-{index}", "training", factors[index]),
                ScaleSpec(f"selection-{index}", "selection", 1.0),
                ScaleSpec(f"terminal-{index}", "terminal", 1.0),
            )
            for index, task in enumerate(self.registry.tasks)
        )

    def compute_task_tensors(
        self, parameters, features, targets, sample_weights, update_index
    ):
        independent_parameters = tf.broadcast_to(
            parameters[None, :], [len(self.registry.tasks), parameters.shape[0]]
        )
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(independent_parameters)
            prediction = tf.einsum("tbd,td->tb", features, independent_parameters)
            residuals = prediction - targets
            raw_values = tf.reduce_mean(sample_weights * tf.square(residuals), axis=1)
        rows = tape.gradient(raw_values, independent_parameters)
        connected = tf.fill([len(self.registry.tasks)], rows is not None)
        if rows is None:
            rows = tf.zeros_like(independent_parameters)
        hard_max = tf.fill(
            [len(self.hard_check_ids)], tf.reduce_max(tf.abs(parameters))
        )
        domain_valid = tf.reduce_all(sample_weights >= 0.0) & tf.reduce_all(
            tf.reduce_sum(sample_weights, axis=1) > 0.0
        )
        return raw_values, rows, hard_max, domain_valid, connected


def fixture_inputs(task_count, parameter_dim=8, batch_size=4):
    features = tf.reshape(
        tf.cast(tf.range(task_count * batch_size * parameter_dim), tf.float64),
        [task_count, batch_size, parameter_dim],
    )
    features = tf.sin(features + 1.0) + 0.25
    parameters = tf.linspace(
        tf.constant(-0.2, tf.float64), tf.constant(0.3, tf.float64), parameter_dim
    )
    targets = tf.reshape(
        tf.cos(tf.cast(tf.range(task_count * batch_size), tf.float64)),
        [task_count, batch_size],
    )
    weights = tf.ones([task_count, batch_size], tf.float64)
    return parameters, features, targets, weights, tf.constant(0, tf.int64)
