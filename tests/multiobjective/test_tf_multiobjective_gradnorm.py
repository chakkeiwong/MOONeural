"""Model-independent TensorFlow GradNorm tests."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest
import tensorflow as tf

from mooneural.multiobjective import (
    aggregate,
    create_gradnorm_state,
    gradnorm_update,
)
from mooneural.multiobjective.gradnorm import gradnorm_flat
from mooneural.multiobjective.unsupported import METHOD_CONTRACTS


def _var(dim=2):
    return (tf.Variable(tf.zeros([dim], dtype=tf.float64)),)


def _rows(rows):
    return tuple((tf.constant(row, dtype=tf.float64),) for row in rows)


def test_initial_losses_are_captured_before_weighting():
    state = create_gradnorm_state(2)
    losses = tf.constant([1.0, 2.0], dtype=tf.float64)

    result = aggregate(
        "gradnorm",
        _rows(([1.0, 0.0], [0.0, 2.0])),
        _var(),
        losses=losses,
        state=state,
    )

    tf.debugging.assert_near(
        result.state.initial_losses, losses, atol=1e-12)
    tf.debugging.assert_near(result.coefficients, [1.0, 1.0], atol=1e-12)
    tf.debugging.assert_near(result.flat_gradient, [1.0, 2.0], atol=1e-12)


def test_gradnorm_missing_state_fails_closed():
    with pytest.raises(ValueError, match="requires an explicit GradNormState"):
        aggregate(
            "gradnorm",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0, 2.0], dtype=tf.float64),
            state=None,
        )


def test_gradnorm_metadata_declares_bounded_usable_contract():
    contract = METHOD_CONTRACTS["gradnorm"]

    assert contract["contract_version"] == "gradnorm_tf_paper_audited_v1"
    assert contract["reference_status"] == (
        "bounded_usable_selected_shared_gradients_not_official_code_parity"
    )


def test_gradnorm_flat_requires_rank_two_gradients():
    with pytest.raises(ValueError, match=r"shape \[objectives, params\]"):
        gradnorm_flat(
            tf.ones([2, 2, 1], dtype=tf.float64),
            tf.constant([1.0, 1.0], dtype=tf.float64),
            create_gradnorm_state(2),
        )


def test_gradnorm_update_requires_rank_two_gradients():
    with pytest.raises(ValueError, match=r"shape \[objectives, params\]"):
        gradnorm_update(
            tf.ones([2, 2, 1], dtype=tf.float64),
            tf.constant([1.0, 1.0], dtype=tf.float64),
            create_gradnorm_state(
                2, initial_losses=tf.constant([1.0, 1.0], dtype=tf.float64)),
        )


def test_gradnorm_negative_alpha_fails_closed():
    with pytest.raises(tf.errors.InvalidArgumentError, match="alpha"):
        aggregate(
            "gradnorm",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0, 1.0], dtype=tf.float64),
            state=create_gradnorm_state(
                2, initial_losses=tf.constant([1.0, 1.0], dtype=tf.float64)),
            gradnorm_alpha=-0.5,
        )


def test_gradnorm_update_nonpositive_learning_rate_fails_closed():
    with pytest.raises(tf.errors.InvalidArgumentError, match="learning_rate"):
        gradnorm_update(
            tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float64),
            tf.constant([1.0, 1.0], dtype=tf.float64),
            create_gradnorm_state(
                2, initial_losses=tf.constant([1.0, 1.0], dtype=tf.float64)),
            learning_rate=0.0,
        )


def test_gradnorm_update_increases_weight_for_slower_task():
    flat = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]], dtype=tf.float64)
    state = create_gradnorm_state(
        2, initial_losses=tf.constant([1.0, 1.0], dtype=tf.float64))

    update = gradnorm_update(
        flat,
        tf.constant([2.0, 0.5], dtype=tf.float64),
        state,
        learning_rate=0.1,
    )

    assert float(update.state.weights[0].numpy()) > 1.0
    assert float(update.state.weights[1].numpy()) < 1.0
    tf.debugging.assert_near(tf.reduce_sum(update.state.weights), 2.0, atol=1e-12)
    assert int(update.state.step.numpy()) == 1
    assert bool(update.diagnostics["finite"].numpy())


def test_gradnorm_relative_inverse_rate_and_target_formula():
    flat = tf.constant(
        [[3.0, 4.0], [0.0, 2.0]], dtype=tf.float64)
    state = create_gradnorm_state(
        2, initial_losses=tf.constant([2.0, 4.0], dtype=tf.float64))
    losses = tf.constant([1.0, 4.0], dtype=tf.float64)
    alpha = 1.5

    result = aggregate(
        "gradnorm",
        tuple((row,) for row in tf.unstack(flat)),
        _var(),
        losses=losses,
        state=state,
        gradnorm_alpha=alpha,
    )

    grad_norms = tf.constant([5.0, 2.0], dtype=tf.float64)
    loss_ratios = losses / state.initial_losses
    inverse_rates = loss_ratios / tf.reduce_mean(loss_ratios)
    weighted_norms = state.weights * grad_norms
    target = tf.reduce_mean(weighted_norms) * tf.pow(inverse_rates, alpha)

    tf.debugging.assert_near(
        result.diagnostics["gradnorm_objective_grad_norms"],
        grad_norms,
        atol=1e-12,
    )
    tf.debugging.assert_near(
        result.diagnostics["gradnorm_loss_ratios"],
        loss_ratios,
        atol=1e-12,
    )
    tf.debugging.assert_near(
        result.diagnostics["gradnorm_inverse_training_rates"],
        inverse_rates,
        atol=1e-12,
    )
    tf.debugging.assert_near(
        result.diagnostics["gradnorm_target_grad_norms"],
        target,
        atol=1e-12,
    )


def test_gradnorm_declares_selected_shared_gradient_contract():
    state = create_gradnorm_state(
        2, initial_losses=tf.constant([1.0, 1.0], dtype=tf.float64))

    result = aggregate(
        "gradnorm",
        _rows(([1.0, 0.0], [0.0, 1.0])),
        _var(),
        losses=tf.constant([1.0, 1.0], dtype=tf.float64),
        state=state,
    )

    assert (
        result.diagnostics["gradnorm_gradient_contract_status"].numpy()
        == b"caller_supplied_selected_shared_gradients")


def test_gradnorm_update_uses_separate_weight_update_from_gradient_combination():
    state = create_gradnorm_state(
        2, initial_losses=tf.constant([1.0, 1.0], dtype=tf.float64))
    grads = _rows(([1.0, 0.0], [0.0, 1.0]))
    losses = tf.constant([2.0, 0.5], dtype=tf.float64)

    result = aggregate(
        "gradnorm",
        grads,
        _var(),
        losses=losses,
        state=state,
    )
    update = gradnorm_update(
        tf.stack([g[0] for g in grads]), losses, result.state, learning_rate=0.1)

    tf.debugging.assert_near(result.coefficients, [1.0, 1.0], atol=1e-12)
    assert float(update.state.weights[0].numpy()) > float(result.state.weights[0].numpy())
    assert int(update.state.step.numpy()) == int(result.state.step.numpy()) + 1


def test_negative_losses_fail_closed():
    with pytest.raises(tf.errors.InvalidArgumentError, match="nonnegative"):
        aggregate(
            "gradnorm",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0, -0.1], dtype=tf.float64),
            state=create_gradnorm_state(2),
        )


def test_gradnorm_missing_losses_fails_closed():
    with pytest.raises((TypeError, ValueError), match="Tensor|loss"):
        aggregate(
            "gradnorm",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=None,
            state=create_gradnorm_state(2),
        )


def test_loss_count_mismatch_fails_closed():
    with pytest.raises(ValueError, match="one loss value"):
        aggregate(
            "gradnorm",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0], dtype=tf.float64),
            state=create_gradnorm_state(2),
        )


def test_gradnorm_nonpositive_initial_losses_fail_closed():
    with pytest.raises(tf.errors.InvalidArgumentError, match="initial losses must be positive"):
        aggregate(
            "gradnorm",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0, 2.0], dtype=tf.float64),
            state=create_gradnorm_state(
                2,
                initial_losses=tf.constant([0.0, 1.0], dtype=tf.float64),
            ),
        )


def test_weight_count_mismatch_fails_closed():
    state = create_gradnorm_state(3)

    with pytest.raises(tf.errors.InvalidArgumentError):
        aggregate(
            "gradnorm",
            _rows(([1.0, 0.0], [0.0, 1.0])),
            _var(),
            losses=tf.constant([1.0, 1.0], dtype=tf.float64),
            state=state,
        )


def test_zero_gradient_task_has_finite_diagnostics():
    state = create_gradnorm_state(
        2, initial_losses=tf.constant([1.0, 1.0], dtype=tf.float64))
    result = aggregate(
        "gradnorm",
        _rows(([1.0, 0.0], [0.0, 0.0])),
        _var(),
        losses=tf.constant([0.8, 0.9], dtype=tf.float64),
        state=state,
    )

    assert bool(result.diagnostics["finite"].numpy())
    tf.debugging.assert_near(
        result.diagnostics["gradnorm_objective_grad_norms"],
        [1.0, 0.0],
        atol=1e-12,
    )


def test_gradnorm_repeated_state_updates_preserve_initial_losses_and_renormalization():
    grads = _rows(([1.0, 0.0], [0.0, 1.0]))
    state = create_gradnorm_state(2)

    result1 = aggregate(
        "gradnorm",
        grads,
        _var(),
        losses=tf.constant([2.0, 0.5], dtype=tf.float64),
        state=state,
    )
    update1 = gradnorm_update(
        tf.stack([g[0] for g in grads]),
        tf.constant([2.0, 0.5], dtype=tf.float64),
        result1.state,
        learning_rate=0.1,
    )
    result2 = aggregate(
        "gradnorm",
        grads,
        _var(),
        losses=tf.constant([1.5, 0.6], dtype=tf.float64),
        state=update1.state,
    )
    update2 = gradnorm_update(
        tf.stack([g[0] for g in grads]),
        tf.constant([1.5, 0.6], dtype=tf.float64),
        result2.state,
        learning_rate=0.1,
    )

    tf.debugging.assert_near(
        result1.state.initial_losses,
        tf.constant([2.0, 0.5], dtype=tf.float64),
        atol=1e-12,
    )
    tf.debugging.assert_near(
        result2.state.initial_losses,
        tf.constant([2.0, 0.5], dtype=tf.float64),
        atol=1e-12,
    )
    tf.debugging.assert_near(
        tf.reduce_sum(update1.state.weights),
        2.0,
        atol=1e-12,
    )
    tf.debugging.assert_near(
        tf.reduce_sum(update2.state.weights),
        2.0,
        atol=1e-12,
    )
    assert int(update1.state.step.numpy()) == 1
    assert int(update2.state.step.numpy()) == 2
    assert bool(update1.diagnostics["finite"].numpy())
    assert bool(update2.diagnostics["finite"].numpy())


def test_gradnorm_update_loss_count_mismatch_fails_closed():
    state = create_gradnorm_state(
        2, initial_losses=tf.constant([1.0, 1.0], dtype=tf.float64))

    with pytest.raises(ValueError, match="one loss value"):
        gradnorm_update(
            tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float64),
            tf.constant([1.0], dtype=tf.float64),
            state,
        )


def test_gradnorm_tf_function_pre_and_update():
    variables = _var()
    grads = _rows(([1.0, 0.0], [0.0, 1.0]))
    state = create_gradnorm_state(
        2, initial_losses=tf.constant([1.0, 1.0], dtype=tf.float64))

    @tf.function
    def run(pre_state):
        result = aggregate(
            "gradnorm",
            grads,
            variables,
            losses=tf.constant([2.0, 0.5], dtype=tf.float64),
            state=pre_state,
        )
        update = gradnorm_update(
            tf.stack([g[0] for g in grads]),
            tf.constant([2.0, 0.5], dtype=tf.float64),
            result.state,
            learning_rate=0.1,
        )
        return result.flat_gradient, update.state.weights

    flat, weights = run(state)
    tf.debugging.assert_all_finite(flat, "finite GradNorm scalarization")
    assert float(weights[0].numpy()) > 1.0
