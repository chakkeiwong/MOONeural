"""Optional finite selection through the shared guard, without model training."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tests.contracts import test_generic_coupled_component_guard as coupled
from mooneural.training.generic_postproposal import (
    GuardedPostproposal,
    PostproposalRejected,
)

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "tests/support/archived_postproposal.py"
SPEC = importlib.util.spec_from_file_location("mooneural.training.archived_postproposal_default", ARCHIVE)
archived = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archived)


def batch_identity(batch, update):
    return {"batch": "fixed-manufactured", "update": update}


def empty_components(parameters, batch, update, context):
    return {"owner_indices": np.empty(0, np.int32), "cell_ids": [], "raw_values": np.empty(0),
        "normalized_values": np.empty(0), "normalized_gradients": np.empty((0, len(parameters))), "context": context}


def objective(parameters, batch, update):
    raw = np.array([(parameters[0] - .3) ** 2 + 1., .1])
    return raw, raw.copy()


def build(callback=objective, *, implementation=GuardedPostproposal, lookahead_steps=2, **options):
    return implementation(callback, batch_identity, [1., 1.], [1., 1.],
        binding={"test": "fixed-quadratic"}, component_rows=empty_components,
        component_binding={"test": "empty-components"}, component_direction="relative_public_progress",
        **({} if lookahead_steps is None else {"lookahead_steps": lookahead_steps}), **options)


def execute(guard):
    parameters = np.zeros(2)
    displacement = np.array([1., 0.])
    optimizer = (tf.constant(displacement), tf.constant([.2, .3]), tf.constant([.4, .5]),
        tf.constant(3, tf.int64), tf.constant(-displacement), tf.constant(displacement),
        tf.constant(displacement), tf.constant([0.]), tf.constant([0.]), tf.constant(0.), tf.constant(True))
    parent = np.array([1.09, .1])
    result, receipt = guard(parameters=parameters, batch=(), update_index=2, raw_losses=parent,
        normalized_losses=parent, gradient_rows=np.array([[-.6, 0.], [0., 1.]]), optimizer=optimizer,
        active_indices=(0,), constraint_indices=(1,), gradient_batch_binding=batch_identity((), 2))
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert result[index] is optimizer[index]
    return result, receipt


def test_smaller_feasible_fraction_selected_by_real_guard_preserves_slots():
    updated, receipt = execute(build())
    assert receipt["accepted_fraction"] == .25
    assert receipt["loss_calls"] == 5
    assert sum(trial["accepted"] for trial in receipt["candidates"]) == 1
    assert [trial["fraction"] for trial in receipt["candidates"]] == [1., .5, .25, .125]
    assert receipt["candidates"][1]["feasible"] and not receipt["candidates"][1]["accepted"]
    np.testing.assert_allclose(updated[0], [.25, 0.], atol=1e-12)
    assert receipt["finite_selection"]["selected_candidate_index"] == 2


@pytest.mark.parametrize("count,expected_calls", ((0, 3), (1, 4), (2, 5)))
def test_bounded_suffix_does_not_append_fractions(count, expected_calls):
    _, receipt = execute(build(lookahead_steps=count))
    assert receipt["loss_calls"] == expected_calls
    assert receipt["accepted_fraction"] == (.5 if count == 0 else .25)


def test_disabled_binding_and_outputs_match_archived_default():
    old = build(implementation=archived.GuardedPostproposal, lookahead_steps=None)
    new = build(lookahead_steps=0)
    assert new.binding == old.binding and new.binding_hash == old.binding_hash
    current, receipt = execute(new)
    previous, old_receipt = execute(old)
    assert receipt == old_receipt
    for actual, expected in zip(current, previous, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fault", ("nonfinite", "protected", "no-decrease"))
def test_failed_lookahead_consumes_entry_and_retains_first_feasible(fault):
    def losses(parameters, batch, update):
        raw, _ = objective(parameters, batch, update)
        if np.isclose(parameters[0], .25):
            raw[0 if fault != "protected" else 1] = {"nonfinite": np.nan, "protected": 1., "no-decrease": 2.}[fault]
        return raw, raw.copy()

    _, receipt = execute(build(losses, lookahead_steps=1))
    assert receipt["accepted_fraction"] == .5
    assert receipt["loss_calls"] == 4
    assert len(receipt["candidates"]) == 3
    assert sum(trial["accepted"] for trial in receipt["candidates"]) == 1


def test_contract_failure_after_feasibility_rejects_whole_transaction():
    def losses(parameters, batch, update):
        raw, normalized = objective(parameters, batch, update)
        if np.isclose(parameters[0], .25):
            normalized *= 2.
        return raw, normalized

    with pytest.raises(PostproposalRejected, match="raw/D") as failure:
        execute(build(losses))
    assert not failure.value.receipt["accepted"]
    assert not any(trial["accepted"] for trial in failure.value.receipt["candidates"])


@pytest.mark.parametrize("gain", (0., 1e-10))
def test_ties_and_insignificant_gains_keep_earlier_candidate(gain):
    def losses(parameters, batch, update):
        raw, _ = objective(parameters, batch, update)
        if np.isclose(parameters[0], .25):
            raw[0] = 1.04 - gain
        return raw, raw.copy()

    _, receipt = execute(build(losses, lookahead_steps=1))
    assert receipt["accepted_fraction"] == .5


def test_suffix_exhaustion_never_expands_list():
    _, receipt = execute(build(fractions=(1., .5)))
    assert receipt["accepted_fraction"] == .5 and receipt["loss_calls"] == 3


@pytest.mark.parametrize("value", (-1, 3, True, .5))
def test_malformed_lookahead_refused(value):
    with pytest.raises(ValueError, match="lookahead_steps"):
        build(lookahead_steps=value)


def test_mutated_option_refused():
    guard = build()
    guard.lookahead_steps = 0
    with pytest.raises(PostproposalRejected, match="mutated"):
        execute(guard)


def test_direction_changing_fallback_refused():
    with pytest.raises(ValueError, match="fixed relative"):
        build(finite_component_fallback=True)


def test_coupled_component_failures_still_control_optional_selection():
    baseline, provider = coupled.nonlinear_guard()
    guard = GuardedPostproposal(baseline.losses, baseline.batch_binding, baseline.denominators, baseline.limits,
        binding=baseline.binding["profile"], fractions=baseline.fractions, component_rows=provider,
        component_binding=provider.binding, component_direction="relative_public_progress",
        coupled_component_guard=True, component_loss_values=provider.loss_only,
        component_loss_binding=provider.binding, lookahead_steps=2)
    _, receipt = coupled.apply(guard)
    assert receipt["accepted_fraction"] == .25
    assert not receipt["candidates"][0]["component"]["strict_decrease"]
    assert receipt["loss_calls"] == 4 and receipt["component_guard"]["loss_calls"] == 3
    assert sum(trial["accepted"] for trial in receipt["candidates"]) == 1
