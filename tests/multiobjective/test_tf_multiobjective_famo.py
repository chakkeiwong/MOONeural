"""Model-independent TensorFlow FAMO tests."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest
import tensorflow as tf

from mooneural.multiobjective import aggregate, create_famo_state, famo_after_step
from mooneural.multiobjective.famo import famo_pre_step
from mooneural.multiobjective.famo import FAMO_REFERENCE_EPS


def _var(dim=2):
    return (tf.Variable(tf.zeros([dim], dtype=tf.float64)),)


def _rows(rows):
    return tuple((tf.constant(row, dtype=tf.float64),) for row in rows)


def test_famo_weighted_log_loss_coefficients():
    variables = _var()
    state = create_famo_state(3)
    losses = tf.constant([2.0, 4.0, 8.0], dtype=tf.float64)

    result = aggregate(
        "famo",
        _rows(([1.0, 0.0], [0.0, 2.0], [3.0, 0.0])),
        variables,
        losses=losses,
        state=state,
    )

    expected = 1.0 / (losses + tf.constant(FAMO_REFERENCE_EPS, tf.float64))
    expected = expected / tf.reduce_sum(expected)
    tf.debugging.assert_near(result.coefficients, expected, atol=1e-12)
    tf.debugging.assert_near(
        result.flat_gradient,
        expected[0] * [1.0, 0.0]
        + expected[1] * [0.0, 2.0]
        + expected[2] * [3.0, 0.0],
        atol=1e-12,
    )
    assert bool(result.state.pending.numpy())


def test_famo_missing_state_fails_closed():
    with pytest.raises(ValueError, match="requires an explicit FAMOState"):
        aggregate(
            "famo",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0, 2.0], dtype=tf.float64),
            state=None,
        )


def test_famo_direct_pre_step_rejects_non_matrix_flat_gradients():
    with pytest.raises(ValueError, match="shape \\[objectives, params\\]"):
        famo_pre_step(
            tf.constant([1.0, 2.0], dtype=tf.float64),
            tf.constant([1.0, 2.0], dtype=tf.float64),
            create_famo_state(2),
        )


def test_famo_pre_step_rejects_invalid_eps():
    with pytest.raises(ValueError, match="eps must be positive and finite"):
        aggregate(
            "famo",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0, 2.0], dtype=tf.float64),
            state=create_famo_state(2),
            eps=0.0,
        )


def test_famo_rejects_signed_losses():
    with pytest.raises(tf.errors.InvalidArgumentError, match="nonnegative"):
        aggregate(
            "famo",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0, -0.1], dtype=tf.float64),
            state=create_famo_state(2),
        )


def test_famo_missing_losses_fails_closed():
    with pytest.raises((TypeError, ValueError), match="Tensor|loss"):
        aggregate(
            "famo",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=None,
            state=create_famo_state(2),
        )


def test_famo_rejects_loss_count_mismatch():
    with pytest.raises(ValueError, match="one loss value"):
        aggregate(
            "famo",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0], dtype=tf.float64),
            state=create_famo_state(2),
        )


def test_famo_min_losses_above_current_losses_fail_closed():
    with pytest.raises(tf.errors.InvalidArgumentError, match="min_losses exceed current losses"):
        aggregate(
            "famo",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([0.5, 0.8], dtype=tf.float64),
            state=create_famo_state(
                2,
                min_losses=tf.constant([0.6, 0.8], dtype=tf.float64),
            ),
        )


def test_famo_pre_step_cannot_overwrite_pending_update():
    variables = _var()
    state = create_famo_state(2)
    result = aggregate(
        "famo",
        _rows(([1.0, 0.0], [0.0, 1.0])),
        variables,
        losses=tf.constant([1.0, 2.0], dtype=tf.float64),
        state=state,
    )

    with pytest.raises(tf.errors.InvalidArgumentError, match="pending"):
        aggregate(
            "famo",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            variables,
            losses=tf.constant([0.9, 1.8], dtype=tf.float64),
            state=result.state,
        )


def test_famo_after_step_updates_adam_state_and_clears_pending():
    variables = _var()
    state = create_famo_state(2)
    result = aggregate(
        "famo",
        _rows(([1.0, 0.0], [0.0, 1.0])),
        variables,
        losses=tf.constant([1.0, 2.0], dtype=tf.float64),
        state=state,
    )

    update = famo_after_step(
        tf.constant([0.8, 1.9], dtype=tf.float64),
        result.state,
        beta=0.025,
        weight_decay=0.0,
    )

    assert not bool(update.state.pending.numpy())
    assert int(update.state.step.numpy()) == 1
    assert float(tf.linalg.norm(update.state.adam_m).numpy()) > 0.0
    assert bool(update.diagnostics["finite"].numpy())


def test_famo_after_step_formula_matches_reference_first_adam_step():
    variables = _var()
    state = create_famo_state(
        2,
        min_losses=tf.constant([0.1, 0.0], dtype=tf.float64),
    )
    state = state._replace(
        logits=tf.constant([0.2, -0.1], dtype=tf.float64))
    before = tf.constant([0.7, 0.4], dtype=tf.float64)
    after = tf.constant([0.6, 0.36], dtype=tf.float64)
    result = aggregate(
        "famo",
        _rows(([1.0, 0.0], [0.0, 1.0])),
        variables,
        losses=before,
        state=state,
    )
    z = tf.nn.softmax(state.logits)
    delta = (
        tf.math.log(before - state.min_losses + FAMO_REFERENCE_EPS)
        - tf.math.log(after - state.min_losses + FAMO_REFERENCE_EPS)
    )
    expected_grad = z * (delta - tf.reduce_sum(z * delta))
    m = 0.1 * expected_grad
    v = 0.001 * tf.square(expected_grad)
    m_hat = m / (1.0 - 0.9)
    v_hat = v / (1.0 - 0.999)
    expected_update = 0.025 * m_hat / (
        tf.sqrt(v_hat) + FAMO_REFERENCE_EPS)

    update = famo_after_step(
        after,
        result.state,
        beta=0.025,
        weight_decay=0.0,
    )

    tf.debugging.assert_near(
        update.diagnostics["famo_logit_gradient"], expected_grad, atol=1e-12)
    tf.debugging.assert_near(
        update.diagnostics["famo_logit_update"], expected_update, atol=1e-6)
    tf.debugging.assert_near(
        update.state.logits, state.logits - expected_update, atol=1e-6)


def test_famo_after_step_requires_pending_pre_step():
    with pytest.raises(tf.errors.InvalidArgumentError, match="pending pre-step"):
        famo_after_step(
            tf.constant([0.8, 1.9], dtype=tf.float64),
            create_famo_state(2),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"beta": 0.0}, "beta must be positive and finite"),
        ({"weight_decay": -0.1}, "weight_decay must be nonnegative and finite"),
        ({"eps": 0.0}, "eps must be positive and finite"),
    ),
)
def test_famo_after_step_rejects_invalid_scalar_hyperparameters(kwargs, match):
    variables = _var()
    result = aggregate(
        "famo",
        _rows(([1.0, 0.0], [0.0, 1.0])),
        variables,
        losses=tf.constant([1.0, 2.0], dtype=tf.float64),
        state=create_famo_state(2),
    )

    with pytest.raises(ValueError, match=match):
        famo_after_step(
            tf.constant([0.8, 1.9], dtype=tf.float64),
            result.state,
            **kwargs,
        )


def test_famo_completed_cycle_allows_next_pre_step_and_preserves_finite_state():
    variables = _var()
    state = create_famo_state(2)

    result1 = aggregate(
        "famo",
        _rows(([1.0, 0.0], [0.0, 1.0])),
        variables,
        losses=tf.constant([1.0, 2.0], dtype=tf.float64),
        state=state,
    )
    update1 = famo_after_step(
        tf.constant([0.8, 1.9], dtype=tf.float64),
        result1.state,
    )
    result2 = aggregate(
        "famo",
        _rows(([1.0, 0.0], [0.0, 1.0])),
        variables,
        losses=tf.constant([0.7, 1.8], dtype=tf.float64),
        state=update1.state,
    )

    assert not bool(update1.state.pending.numpy())
    assert bool(result2.state.pending.numpy())
    assert int(update1.state.step.numpy()) == 1
    assert bool(update1.diagnostics["finite"].numpy())
    tf.debugging.assert_all_finite(update1.state.logits, "finite FAMO logits")
    tf.debugging.assert_near(
        result2.state.prev_losses,
        tf.constant([0.7, 1.8], dtype=tf.float64),
        atol=1e-12,
    )


def test_famo_tf_function_pre_and_after_step():
    variables = _var()
    grads = _rows(([1.0, 0.0], [0.0, 1.0]))
    state = create_famo_state(2)

    @tf.function
    def run(pre_state):
        result = aggregate(
            "famo",
            grads,
            variables,
            losses=tf.constant([1.0, 2.0], dtype=tf.float64),
            state=pre_state,
        )
        update = famo_after_step(
            tf.constant([0.8, 1.9], dtype=tf.float64), result.state)
        return result.flat_gradient, update.state.pending

    flat, pending = run(state)
    assert flat.shape == (2,)
    assert not bool(pending.numpy())
