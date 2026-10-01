"""Dtype coverage tests for TensorFlow multiobjective utilities."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest
import tensorflow as tf

from mooneural.multiobjective import (
    aggregate,
    create_famo_state,
    create_gradnorm_state,
    famo_after_step,
)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_mgda_preserves_runtime_dtype(dtype):
    variables = (tf.Variable(tf.zeros([2], dtype=dtype)),)
    grads = (
        (tf.constant([1.0, 0.0], dtype=dtype),),
        (tf.constant([0.0, 2.0], dtype=dtype),),
    )

    result = aggregate("mgda", grads, variables)

    assert result.flat_gradient.dtype == dtype
    assert result.coefficients.dtype == dtype
    assert result.combined_gradients[0].dtype == dtype
    assert bool(result.diagnostics["finite"].numpy())


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_imtl_preserves_runtime_dtype(dtype):
    variables = (tf.Variable(tf.zeros([2], dtype=dtype)),)
    grads = (
        (tf.constant([1.0, 0.0], dtype=dtype),),
        (tf.constant([0.0, 2.0], dtype=dtype),),
    )

    result = aggregate("imtl", grads, variables)

    assert result.flat_gradient.dtype == dtype
    assert result.coefficients.dtype == dtype
    assert result.combined_gradients[0].dtype == dtype
    assert bool(result.diagnostics["finite"].numpy())


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_famo_preserves_runtime_dtype(dtype):
    variables = (tf.Variable(tf.zeros([2], dtype=dtype)),)
    grads = (
        (tf.constant([1.0, 0.0], dtype=dtype),),
        (tf.constant([0.0, 1.0], dtype=dtype),),
    )
    state = create_famo_state(2, dtype=dtype)

    result = aggregate(
        "famo",
        grads,
        variables,
        losses=tf.constant([1.0, 2.0], dtype=dtype),
        state=state,
    )
    update = famo_after_step(
        tf.constant([0.8, 1.9], dtype=dtype),
        result.state,
    )

    assert result.flat_gradient.dtype == dtype
    assert result.coefficients.dtype == dtype
    assert result.combined_gradients[0].dtype == dtype
    assert result.state.logits.dtype == dtype
    assert update.state.logits.dtype == dtype
    assert update.diagnostics["famo_logit_update"].dtype == dtype
    assert bool(result.diagnostics["finite"].numpy())
    assert bool(update.diagnostics["finite"].numpy())


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_pcgrad_preserves_runtime_dtype(dtype):
    variables = (tf.Variable(tf.zeros([2], dtype=dtype)),)
    grads = (
        (tf.constant([1.0, 0.0], dtype=dtype),),
        (tf.constant([0.0, 1.0], dtype=dtype),),
    )

    result = aggregate(
        "pcgrad",
        grads,
        variables,
        seed=tf.constant([2, 3], tf.int32),
    )

    assert result.flat_gradient.dtype == dtype
    assert result.coefficients.dtype == dtype
    assert result.combined_gradients[0].dtype == dtype
    assert bool(result.diagnostics["finite"].numpy())


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_gradnorm_preserves_runtime_dtype(dtype):
    variables = (tf.Variable(tf.zeros([2], dtype=dtype)),)
    grads = (
        (tf.constant([1.0, 0.0], dtype=dtype),),
        (tf.constant([0.0, 1.0], dtype=dtype),),
    )
    losses = tf.constant([1.0, 2.0], dtype=dtype)
    state = create_gradnorm_state(2, dtype=dtype)

    result = aggregate(
        "gradnorm",
        grads,
        variables,
        losses=losses,
        state=state,
    )

    assert result.flat_gradient.dtype == dtype
    assert result.coefficients.dtype == dtype
    assert result.combined_gradients[0].dtype == dtype
    assert result.state.weights.dtype == dtype
    assert bool(result.diagnostics["finite"].numpy())


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_aligned_preserves_runtime_dtype(dtype):
    variables = (tf.Variable(tf.zeros([2], dtype=dtype)),)
    grads = (
        (tf.constant([2.0, 0.0], dtype=dtype),),
        (tf.constant([0.0, 1.0], dtype=dtype),),
    )

    result = aggregate("aligned", grads, variables)

    assert result.flat_gradient.dtype == dtype
    assert result.coefficients.dtype == dtype
    assert result.combined_gradients[0].dtype == dtype
    assert result.diagnostics["aligned_scale"].dtype == dtype
    assert bool(result.diagnostics["finite"].numpy())
