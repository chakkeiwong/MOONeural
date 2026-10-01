"""Public TensorFlow multiobjective result containers."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Tuple

import tensorflow as tf


class AggregationResult(NamedTuple):
    """Result returned by a TensorFlow multiobjective aggregation step."""

    combined_gradients: Tuple[tf.Tensor, ...]
    flat_gradient: tf.Tensor
    coefficients: tf.Tensor
    diagnostics: Mapping[str, tf.Tensor]
    state: Any


class FAMOState(NamedTuple):
    """Functional TensorFlow state for FAMO."""

    logits: tf.Tensor
    min_losses: tf.Tensor
    prev_losses: tf.Tensor
    adam_m: tf.Tensor
    adam_v: tf.Tensor
    step: tf.Tensor
    pending: tf.Tensor


class FAMOUpdateResult(NamedTuple):
    """Result returned by the FAMO same-batch after-step update."""

    state: FAMOState
    diagnostics: Mapping[str, tf.Tensor]


class GradNormState(NamedTuple):
    """Functional TensorFlow state for paper-audited GradNorm."""

    weights: tf.Tensor
    initial_losses: tf.Tensor
    step: tf.Tensor


class GradNormUpdateResult(NamedTuple):
    """Result returned by a GradNorm weight update."""

    state: GradNormState
    diagnostics: Mapping[str, tf.Tensor]
