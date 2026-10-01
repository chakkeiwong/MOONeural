"""Finite guard geometry, shared transactions and recovery, without economics."""

import json
from collections.abc import Mapping
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_execution_boundary import (
    CompiledBoundary,
    MethodFallback,
)
from mooneural.training.generic_permanent_pass_executor import (
    PermanentPassExecutor,
    PermanentPassUpdateError,
)
from tests.support.fixtures import (
    fake_control,
    fixture,
    roles,
)
from mooneural.training.generic_postproposal import (
    GuardedPostproposal,
    PostproposalRejected,
)
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    canonical_json,
    stable_hash,
)


def batch_identity(batch, update_index):
    if isinstance(batch, Mapping):
        return dict(batch)
    return {"update": update_index, "batch": [np.asarray(value).tolist() for value in batch]}


def linear_losses(parameters, batch, update_index):
    raw = np.array([2. + parameters[0], .1 + parameters[1]])
    return raw, raw / np.array([2., 3.])


def make_guard(callback=linear_losses, **kwargs):
    return GuardedPostproposal(callback, batch_identity, [2., 3.], [1., 1.],
                              binding={"criterion": "synthetic-explicit-v1"}, **kwargs)


def proposal(parameters=(0., 0.), displacement=(1., 0.), descent=(1., 0.)):
    parameters, displacement, descent = (np.array(values, np.float64)
                                         for values in (parameters, displacement, descent))
    return (tf.constant(parameters + displacement), tf.constant([.3, .4], tf.float64),
            tf.constant([.5, .6], tf.float64), tf.constant(7, tf.int64), tf.constant(descent),
            tf.constant(displacement), tf.constant(displacement), tf.constant([0.], tf.float64),
            tf.constant([0.], tf.float64), tf.constant(0., tf.float64), tf.constant(True))


def call(guard, optimizer=None, raw=None, rows=None):
    initial_raw = np.array([2., .1]) if raw is None else np.array(raw, np.float64)
    return guard(parameters=np.zeros(2), batch=(), update_index=0, raw_losses=initial_raw,
                 normalized_losses=initial_raw / guard.denominators,
                 gradient_rows=np.diag([.5, 1. / 3.]) if rows is None else np.array(rows, np.float64),
                 optimizer=proposal() if optimizer is None else optimizer,
                 active_indices=(0,), constraint_indices=(1,),
                 gradient_batch_binding=guard._callbacks[1]((), 0))


def test_adam_sign_repaired_preserves_slots_and_truthful_displacement():
    original = proposal()
    updated, receipt = call(make_guard(), original)
    assert receipt["accepted"] and receipt["loss_calls"] == 2
    assert receipt["blend_fraction"] == pytest.approx(.55)
    assert np.asarray(updated[6])[0] < 0 < np.asarray(original[6])[0]
    np.testing.assert_array_equal(updated[0], updated[6])
    np.testing.assert_array_equal(updated[8], [0.])
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert updated[index] is original[index]


def test_maximum_switch_requires_finite_fraction():
    def losses(parameters, batch, update_index):
        coordinate = parameters[0]
        raw = np.array([max((coordinate - 1.) ** 2, 4. * coordinate**2 + .5), .1])
        return raw, raw / np.array([2., 3.])

    guard = make_guard(losses)
    updated, receipt = call(guard, proposal(displacement=(.5, 0.), descent=(-1., 0.)),
                            raw=[1., .1], rows=[[-1., 0.], [0., 0.]])
    assert receipt["blend_fraction"] == 0.
    assert receipt["accepted_fraction"] == .5
    assert receipt["loss_calls"] == 3
    assert [value["accepted"] for value in receipt["candidates"]] == [False, True]
    assert np.asarray(updated[0])[0] == .25


def test_protected_finite_increase_exhausts_bounded_candidates():
    def losses(parameters, batch, update_index):
        raw = np.array([2. + parameters[0], .1 + 1e8 * parameters[0] ** 2])
        return raw, raw / np.array([2., 3.])

    guard = make_guard(losses)
    with pytest.raises(PostproposalRejected, match="budget exhausted") as rejected:
        call(guard)
    assert rejected.value.receipt["loss_calls"] == 5
    assert len(rejected.value.receipt["candidates"]) == 4
    assert not rejected.value.receipt["accepted"]


def test_normalized_only_source_compares_parent_and_reconstructs_no_fake_raw():
    guard = make_guard()
    _updated, receipt = guard(parameters=np.zeros(2), batch=(), update_index=0, raw_losses=None,
        normalized_losses=np.array([1., .1 / 3.]), gradient_rows=np.diag([.5, 1. / 3.]),
        optimizer=proposal(), active_indices=(0,), constraint_indices=(1,),
        gradient_batch_binding=batch_identity((), 0))
    assert not receipt["source_raw_available"]


@pytest.mark.parametrize("failure", ["parent", "normalization", "batch", "profile", "callback", "fractions", "D"])
def test_mismatched_or_mutated_loss_contract_refused(failure):
    batch_state = {"version": 1}

    def losses(parameters, batch, update_index):
        raw, normalized = linear_losses(parameters, batch, update_index)
        if failure == "parent":
            return raw + 1., (raw + 1.) / np.array([2., 3.])
        if failure == "normalization":
            return raw, normalized / np.array([2., 3.])
        if failure == "batch":
            batch_state["version"] += 1
        return raw, normalized

    guard = GuardedPostproposal(losses, lambda batch, update: batch_state, [2., 3.], [1., 1.],
                               binding={"profile": {"name": "synthetic"}})
    if failure == "profile":
        guard.binding["profile"]["profile"]["name"] = "mutated"
    if failure == "callback":
        guard.losses = linear_losses
    if failure == "fractions":
        guard.fractions = (1.,)
    if failure == "D":
        guard.denominators = np.array([3., 3.])
    with pytest.raises(ValueError):
        call(guard)


def test_external_profile_mutation_cannot_change_captured_identity():
    binding = {"criterion": {"version": 1}}
    guard = GuardedPostproposal(linear_losses, batch_identity, [2., 3.], [1., 1.], binding=binding)
    binding["criterion"]["version"] = 2
    assert guard.binding["profile"]["criterion"]["version"] == 1
    call(guard)


def test_nonfinite_reference_norm_rejected_without_infinite_slack():
    with pytest.raises(PostproposalRejected, match="reference direction"):
        call(make_guard(), proposal(descent=(1.7e308, 1.7e308)))


def quadratic_batch_losses(parameters, batch, update_index):
    features, targets, weights = map(np.asarray, batch)
    residuals = np.einsum("tbd,d->tb", features, parameters) - targets
    raw = np.mean(weights * residuals**2, axis=1)
    return raw, raw


def test_shared_success_checkpoint_resume_and_absent_hook_exact_numerics():
    baseline, initial, batch = fixture(task_count=2)
    guard = GuardedPostproposal(quadratic_batch_losses, batch_identity, [1.] * 2, [.04] * 2,
                               binding={"fixture": "quadratic"})
    guarded = PermanentPassExecutor(baseline.adapter, baseline.spec, roles(), threshold=.04, postproposal=guard)
    control = fake_control(initial, baseline.adapter.registry.task_ids, [.2] * 2)
    parent = guarded.initialize(initial, control)
    reference, reference_event = baseline.step(baseline.initialize(initial, control), batch)
    state, event = guarded.step(parent, batch)
    assert event["postproposal"]["accepted_fraction"] == 1.
    assert event["postproposal"]["blend_fraction"] == 0.
    assert state.policy_state == reference.policy_state
    assert state.optimizer_state == reference.optimizer_state
    assert state.method_state == reference.method_state
    assert state.rng_state == reference.rng_state
    assert event["actual_constraint_dots"] == pytest.approx(reference_event["actual_constraint_dots"], abs=1e-15)
    restored = CheckpointState.from_dict(json.loads(canonical_json(state.to_dict())))
    resumed = PermanentPassExecutor(baseline.adapter, baseline.spec, roles(), threshold=.04, postproposal=guard)
    assert resumed.step(restored, batch)[0].to_dict() == guarded.step(state, batch)[0].to_dict()
    with pytest.raises(ValueError, match="binding mismatch"):
        baseline.step(state, batch)


def test_rejection_rolls_back_rates_rng_membership_and_adam(monkeypatch):
    executor, initial, batch = fixture()
    guard = make_guard()
    executor = PermanentPassExecutor(executor.adapter, executor.spec, roles(), threshold=.04, postproposal=guard)
    parent = executor.initialize(initial, fake_control(initial, executor.adapter.registry.task_ids, [.2] * 7))
    encoded = canonical_json(parent.to_dict())

    def reject(boundary, state, batch, update, **kwargs):
        boundary.methods.rates["cagrad"] *= .5
        boundary.methods.preferred = "pcgrad"
        boundary.methods.events.append({"simulated": "fallback"})
        raise PostproposalRejected("finite candidate budget exhausted", {"loss_calls": 5, "accepted": False})

    monkeypatch.setattr(CompiledBoundary, "step", reject)
    with pytest.raises(PermanentPassUpdateError) as failure:
        executor.step(parent, batch)
    assert failure.value.checkpoint is parent
    assert canonical_json(failure.value.checkpoint.to_dict()) == encoded
    assert failure.value.event["postproposal"]["loss_calls"] == 5
    boundary, = executor.boundaries.values()
    assert boundary.methods.rates == parent.method_state["rates"]
    assert boundary.methods.preferred == parent.method_state["preferred"]


def test_complete_state_calls_no_guard_or_numerical_boundary():
    existing, initial, batch = fixture()
    guard = make_guard()
    executor = PermanentPassExecutor(existing.adapter, existing.spec, roles(), threshold=.04, postproposal=guard)
    state = executor.initialize(initial, fake_control(initial, existing.adapter.registry.task_ids, [.04] * 7))
    stopped, event = executor.step(state, batch)
    assert stopped is state and event["update_calls"] == 0
    assert guard.blend.experimental_get_tracing_count() == 0
    assert not executor.boundaries


def runner_objective(parameters):
    differences = np.asarray(parameters)[None, :] - np.array([[0.], [1.]])
    raw = np.sum(differences**2, axis=1)
    return tuple(tf.constant(value, tf.float64) for value in (
        raw, raw / np.array([2., 5.]), 2. * differences / np.array([[2.], [5.]])))


def runner_losses(parameters, batch, update_index):
    return runner_objective(parameters)[:2]


def test_finite_runner_guarded_lifecycle_and_zero_call_completed_recovery(tmp_path, monkeypatch):
    source = finite._reference(__file__)
    guard = GuardedPostproposal(runner_losses, batch_identity, [2., 5.], [2., 5.], binding={"criterion": "quadratic"})
    args = {"task_ids": ("a", "b"), "denominators": [2., 5.], "threshold": 1.,
        "parameters": [3., 4.], "objective": runner_objective, "learning_rate": .01,
        "updates": 2, "boundary_every": 1, "postproposal": guard,
        "data_binding": {"objective_recipe": {"kind": "manufactured"}, "objective_source": source,
                         "postproposal_source": source},
        "source_evidence": {"source": source}}
    runner, _adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    result = runner.run()
    assert result.states[0].update_index == 2
    assert result.states[0].optimizer_state["iteration"] == 2
    transactions = [json.loads(path.read_text()) for path in sorted(
        (tmp_path / "runtime/checkpoints/arms/arm-0/checkpoints").glob("update-????????.json"))]
    assert [entry["event"]["postproposal"]["accepted"] for entry in transactions] == [True, True]
    assert runner.design.to_dict()["optimizer_binding"]["postproposal"]["profile"] == guard.binding
    rebuilt, adapter, provider = finite.build_runner(tmp_path / "runtime", **args)
    forbidden = Mock(side_effect=AssertionError("unexpected recovery numerical call"))
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    for name in ("evaluate", "evaluate_values", "compute_task_values_and_gradients"):
        monkeypatch.setattr(adapter, name, forbidden)
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    recovered = rebuilt.run()
    assert recovered.states[0].to_dict() == result.states[0].to_dict()
    assert recovered.evaluations == result.evaluations
    assert recovered.selection == result.selection
    forbidden.assert_not_called()


def test_nonfinite_candidate_is_counted_then_shorter_fraction_is_tried():
    def losses(parameters, batch, update_index):
        if parameters[0] < -.08:
            return np.full(2, np.nan), np.full(2, np.nan)
        return linear_losses(parameters, batch, update_index)

    _updated, receipt = call(make_guard(losses))
    assert receipt["loss_calls"] == 3
    assert receipt["accepted_fraction"] == .5
    assert receipt["candidates"][0]["numerical_failure"]


def test_all_nonfinite_candidates_exhaust_budget_with_truthful_count():
    def losses(parameters, batch, update_index):
        if np.any(parameters):
            return np.full(2, np.nan), np.full(2, np.nan)
        return linear_losses(parameters, batch, update_index)

    with pytest.raises(PostproposalRejected, match="budget exhausted") as failure:
        call(make_guard(losses))
    assert failure.value.receipt["loss_calls"] == 5
    assert all(row["numerical_failure"] for row in failure.value.receipt["candidates"])


def test_different_input_descriptor_rejected_even_with_equal_parent_values():
    losses = Mock(side_effect=linear_losses)
    guard = GuardedPostproposal(losses, lambda batch, update: {"rows": ["different"]},
                               [2., 3.], [1., 1.], binding={"criterion": "same-parent-value"})
    with pytest.raises(PostproposalRejected, match="gradient batch descriptor") as failure:
        guard(parameters=np.zeros(2), batch=(), update_index=0, raw_losses=np.array([2., .1]),
              normalized_losses=np.array([1., .1 / 3.]), gradient_rows=np.diag([.5, 1. / 3.]),
              optimizer=proposal(), active_indices=(0,), constraint_indices=(1,),
              gradient_batch_binding={"rows": ["actual-gradient-input"]})
    assert failure.value.receipt["loss_calls"] == 0
    losses.assert_not_called()


@pytest.mark.parametrize("failure_kind", ["callback", "tiny"])
def test_actual_boundary_contract_and_tiny_refusal_roll_back_every_state(monkeypatch, failure_kind):
    existing, initial, batch = fixture(task_count=2)

    def losses(parameters, batch, update):
        raw, normalized = quadratic_batch_losses(parameters, batch, update)
        return (raw + 1., normalized + 1.) if failure_kind == "callback" else (raw, normalized)

    guard = GuardedPostproposal(losses, batch_identity, [1.] * 2, [.04] * 2, binding={"fixture": "quadratic"})
    executor = PermanentPassExecutor(existing.adapter, existing.spec, roles(), threshold=.04, postproposal=guard)
    if failure_kind == "tiny":
        initial = replace(initial,
            optimizer_state={**initial.optimizer_state, "learning_rate": 1e-13},
            method_state={**initial.method_state, "rates": dict.fromkeys(initial.method_state["rates"], 1e-13)})
    parent = executor.initialize(initial, fake_control(initial, existing.adapter.registry.task_ids, [.2] * 2))
    saved = canonical_json(parent.to_dict())
    original_choose = MethodFallback.choose

    def changed_method_state(method, *arguments):
        result = original_choose(method, *arguments)
        method.rates["cagrad"] *= .5
        method.preferred = "pcgrad"
        method.events.append({"fixture": "attempted-method-state-change"})
        return result

    monkeypatch.setattr(MethodFallback, "choose", changed_method_state)
    expected = "source gradient receipt" if failure_kind == "callback" else "tiny projected update"
    with pytest.raises(PermanentPassUpdateError, match=expected) as failure:
        executor.step(parent, batch)
    assert failure.value.checkpoint is parent
    assert canonical_json(failure.value.checkpoint.to_dict()) == saved
    assert failure.value.event["transaction_rolled_back"]
    boundary, = executor.boundaries.values()
    assert boundary.methods.rates == parent.method_state["rates"]
    assert boundary.methods.preferred == parent.method_state["preferred"]
    if failure_kind == "callback":
        assert failure.value.event["postproposal"]["loss_calls"] == 1
        assert failure.value.event["postproposal"]["failure_kind"] == "contract_error"
    else:
        assert failure.value.event["postproposal"]["accepted"]


def supplied_component(parameters, batch, update_index, context):
    denominators = np.asarray(context["D"])
    return {"owner_indices": np.array([0]), "cell_ids": ["cell-rival"],
        "raw_values": np.array([1.5]), "normalized_values": np.array([1.5 / denominators[0]]),
        "normalized_gradients": np.array([[.25 / denominators[0], 0.]]), "context": context}


def component_guard(provider=supplied_component, callback=linear_losses, **options):
    return make_guard(callback, component_rows=provider,
                      component_binding={"source": "analytic-components-v1", "selection": "supplied"}, **options)


def test_disabled_component_binding_is_exact_legacy_binding():
    guard = make_guard(component_rows=None, component_binding=None)
    expected = {"schema": "generic_neural_solver.postproposal.v1",
        "profile": {"criterion": "synthetic-explicit-v1"}, "D": [2., 3.], "T": [1., 1.],
        "margin_fraction": .1, "fractions": [1., .5, .25, .125], "relative_tolerance": 1e-12,
        "roundoff_profile": "half-interval-half-evaluation-stable-norm-v1",
        "reference": "first-protected-descent-at-source-final-displacement-norm",
        "moments": "source-proposed-slots-and-single-clock-increment", "failure": "typed-rejection-no-commit",
        "maximum_loss_calls": 5, "protected_dot_absolute_tolerance": 1e-10,
        "loss_parity_rtol": 1e-11, "loss_parity_atol": 1e-18,
        "input_identity": "exact-gradient-batch-context-v1",
        "numerical_failure": "count-call-reject-fraction-continue",
        "contract_failure": "stop-with-receipt-complete-rollback"}
    assert guard.binding == expected and guard.binding_hash == stable_hash(expected)
    updated, receipt = call(guard)
    reference, baseline_receipt = call(make_guard())
    assert receipt == baseline_receipt
    for actual, previous in zip(updated, reference, strict=True):
        np.testing.assert_array_equal(actual, previous)


def test_valid_segment_and_finite_exhaustion_never_invoke_component_provider():
    calls = []

    def forbidden(*arguments):
        calls.append(arguments)
        raise AssertionError("component provider must remain lazy")

    guard = component_guard(forbidden)
    updated, receipt = call(guard)
    reference, _baseline_receipt = call(make_guard())
    for actual, previous in zip(updated, reference, strict=True):
        np.testing.assert_array_equal(actual, previous)
    assert not receipt["component_fallback"]["invoked"]
    assert guard.component_descent.experimental_get_tracing_count() == 0

    def finite_failure(parameters, batch, update):
        raw = np.array([2. + parameters[0], .1 + 1e8 * parameters[0] ** 2])
        return raw, raw / [2., 3.]

    with pytest.raises(PostproposalRejected, match="budget exhausted") as failure:
        call(component_guard(forbidden, finite_failure))
    assert failure.value.receipt["component_fallback"]["provider_calls"] == 0
    assert calls == []


@pytest.mark.parametrize("denominators", [[2., 3.], [7., 11.]])
def test_invalid_segment_uses_analytic_nonunit_D_component_once_preserving_slots(denominators):
    denominators = np.asarray(denominators)
    requests = []

    def losses(parameters, batch, update):
        coordinate = parameters[0]
        raw = np.array([max(2. + coordinate, 1.5 + .25 * coordinate + .25 * coordinate**2), .1 + parameters[1]])
        return raw, raw / denominators

    def provider(parameters, batch, update, context):
        assert not parameters.flags.writeable
        assert context["parameters_hash"] == stable_hash(parameters.tolist())
        assert list(context["active_indices"]) == [0]
        payload = supplied_component(parameters, batch, update, context)
        np.testing.assert_array_equal(payload["normalized_gradients"], [[.25 / denominators[0], 0.]])
        np.testing.assert_array_equal(payload["normalized_values"], [1.5 / denominators[0]])
        requests.append(payload)
        return payload

    guard = GuardedPostproposal(losses, batch_identity, denominators, [1., 1.], binding={"model": "max-quadratics"},
        component_rows=provider, component_binding={"source": "analytic-quadratic-derivative"})
    original = proposal(descent=(-1., 0.))
    updated, receipt = call(guard, original, rows=np.diag(1. / denominators))
    details = receipt["component_fallback"]
    assert len(requests) == details["provider_calls"] == details["witness_calls"] == 1
    assert details["valid"] and details["rank"] == 2
    assert details["normalized_gaps"] == pytest.approx([.5 / denominators[0]])
    assert receipt["loss_calls"] == 2 and receipt["accepted_fraction"] == 1.
    assert np.asarray(updated[0])[0] < 0.
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert updated[index] is original[index]


@pytest.mark.parametrize("fault", ["owner", "clock", "parent", "batch", "coordinate", "D", "profile", "partition",
    "value", "normalization", "shape", "nonfinite", "duplicate", "zero_gradient"])
def test_component_rows_refuse_stale_or_malformed_inputs_without_candidate_calls(fault):
    def provider(parameters, batch, update, context):
        payload = supplied_component(parameters, batch, update, context)
        context = json.loads(canonical_json(context))
        payload["context"] = context
        if fault == "owner":
            payload["owner_indices"] = np.array([1])
        elif fault == "clock":
            context["update_index"] += 1
        elif fault in ("parent", "batch", "profile"):
            context[{"parent": "parameters_hash", "batch": "batch_hash", "profile": "profile_hash"}[fault]] = "stale"
        elif fault == "coordinate":
            context["coordinate_mode"] = "raw_over_T"
        elif fault == "D":
            context["D"][0] *= 2.
        elif fault == "partition":
            context["active_indices"] = [1]
        elif fault == "value":
            payload["raw_values"][:] = 3.
            payload["normalized_values"][:] = 1.5
        elif fault == "normalization":
            payload["normalized_values"] /= 2.
        elif fault == "shape":
            payload["normalized_gradients"] = np.zeros((1, 3))
        elif fault == "nonfinite":
            payload["normalized_gradients"][0, 0] = np.nan
        elif fault == "duplicate":
            for name in ("owner_indices", "raw_values", "normalized_values", "normalized_gradients"):
                payload[name] = np.concatenate([payload[name], payload[name]], axis=0)
            payload["cell_ids"] *= 2
        else:
            payload["normalized_gradients"][:] = 0.
        return payload

    with pytest.raises(PostproposalRejected) as failure:
        call(component_guard(provider), proposal(descent=(-1., 0.)))
    receipt = failure.value.receipt
    assert receipt["loss_calls"] == 1 and receipt["candidates"] == []
    assert receipt["component_fallback"]["provider_calls"] == 1
    assert not receipt["accepted"]


def test_component_callback_and_profile_are_frozen_without_construction_calls():
    calls = []

    def provider(*arguments):
        calls.append(arguments)
        return supplied_component(*arguments)

    profile = {"source": {"revision": 1}}
    guard = make_guard(component_rows=provider, component_binding=profile)
    profile["source"]["revision"] = 2
    assert guard.binding["component_fallback"]["provider"]["source"]["revision"] == 1
    assert calls == [] and guard.component_descent.experimental_get_tracing_count() == 0
    guard.component_rows = supplied_component
    with pytest.raises(PostproposalRejected, match="mutated"):
        call(guard)
    assert calls == []
    with pytest.raises(ValueError, match="supplied together"):
        make_guard(component_rows=provider)


def test_third_cell_takeover_remains_subject_to_original_eleven_fractions():
    denominators = np.array([2., 3.])
    fractions = tuple(2. ** -index for index in range(11))
    calls = []

    def losses(parameters, batch, update):
        raw = np.array([max(2. + parameters[0], 1.99 + parameters[2], 1.8 - 4. * parameters[0]), .1 + parameters[1]])
        return raw, raw / denominators

    def provider(parameters, batch, update, context):
        calls.append(update)
        return {"owner_indices": np.array([0]), "cell_ids": ["rival"], "raw_values": np.array([1.99]),
            "normalized_values": np.array([1.99 / 2.]), "normalized_gradients": np.array([[0., 0., .5]]),
            "context": context}

    guard = component_guard(provider, losses, fractions=fractions)
    original = proposal(parameters=(0., 0., 0.), displacement=(1., 0., 0.), descent=(-1., 0., 0.))
    updated, receipt = guard(parameters=np.zeros(3), batch=(), update_index=0, raw_losses=np.array([2., .1]),
        normalized_losses=np.array([1., .1 / 3.]), gradient_rows=np.array([[.5, 0., 0.], [0., 1. / 3., 0.]]),
        optimizer=original, active_indices=(0,), constraint_indices=(1,), gradient_batch_binding=batch_identity((), 0))
    assert calls == [0] and receipt["accepted_fraction"] == 1. / 16.
    assert [trial["accepted"] for trial in receipt["candidates"]] == [False] * 4 + [True]
    assert receipt["loss_calls"] == 6 <= guard.binding["maximum_loss_calls"] == 12
    assert np.all(losses(np.asarray(updated[0]), (), 0)[0] < [2., 1.])


def test_component_witness_does_not_bypass_rounded_protected_dot_check():
    parameters = np.array([1e16, 0.])

    def losses(candidate, batch, update):
        displacement = candidate - parameters
        raw = np.array([2. + displacement[0] - displacement[1], .1 + .01 * displacement.sum()])
        return raw, raw / [2., 3.]

    def provider(candidate, batch, update, context):
        payload = supplied_component(candidate, batch, update, context)
        payload["normalized_gradients"] = np.array([[.5, -.5]])
        return payload

    guard = component_guard(provider, losses)
    with pytest.raises(PostproposalRejected, match="rounded displacement") as failure:
        guard(parameters=parameters, batch=(), update_index=0, raw_losses=np.array([2., .1]),
            normalized_losses=np.array([1., .1 / 3.]), gradient_rows=np.array([[.5, -.5], [.01 / 3., .01 / 3.]]),
            optimizer=proposal(parameters=parameters, descent=(-1., 0.)), active_indices=(0,), constraint_indices=(1,),
            gradient_batch_binding=batch_identity((), 0))
    assert failure.value.receipt["component_fallback"]["valid"]
    assert failure.value.receipt["candidates"][0]["accepted"]
    assert not failure.value.receipt["accepted"]


def test_component_contract_failure_rolls_back_shared_executor_state(monkeypatch):
    existing, initial, batch = fixture(task_count=2)

    def provider(*arguments):
        raise ValueError("stale supplied component")

    guard = GuardedPostproposal(quadratic_batch_losses, batch_identity, [1.] * 2, [.04] * 2,
        binding={"fixture": "quadratic"}, component_rows=provider, component_binding={"source": "failed-provider"})
    original_blend = guard.blend

    def invalid_segment(*arguments):
        result = original_blend(*arguments)
        return {**result, "valid": tf.constant(False)}

    monkeypatch.setattr(guard, "blend", invalid_segment)
    executor = PermanentPassExecutor(existing.adapter, existing.spec, roles(), threshold=.04, postproposal=guard)
    parent = executor.initialize(initial, fake_control(initial, existing.adapter.registry.task_ids, [.2] * 2))
    saved = canonical_json(parent.to_dict())
    with pytest.raises(PermanentPassUpdateError, match="stale supplied component") as failure:
        executor.step(parent, batch)
    assert failure.value.checkpoint is parent and canonical_json(parent.to_dict()) == saved
    assert failure.value.event["transaction_rolled_back"]
    assert failure.value.event["postproposal"]["component_fallback"]["provider_calls"] == 1


@pytest.mark.parametrize("failed_task", ["active", "protected"])
def test_witness_finite_failure_preserves_eleven_fraction_budget(failed_task):
    def losses(parameters, batch, update):
        raw = np.array([2. + parameters[0], .1 + parameters[1]])
        raw[0 if failed_task == "active" else 1] += 1e8 * parameters[0] ** 2
        return raw, raw / [2., 3.]

    guard = component_guard(callback=losses, fractions=tuple(2. ** -index for index in range(11)))
    with pytest.raises(PostproposalRejected, match="budget exhausted") as failure:
        call(guard, proposal(descent=(-1., 0.)))
    receipt = failure.value.receipt
    assert receipt["component_fallback"]["provider_calls"] == receipt["component_fallback"]["witness_calls"] == 1
    assert receipt["loss_calls"] == guard.binding["maximum_loss_calls"] == 12
    assert len(receipt["candidates"]) == 11 and not any(trial["accepted"] for trial in receipt["candidates"])


def test_nonfinite_original_segment_is_not_component_fallback_eligibility():
    def forbidden(*arguments):
        raise AssertionError("arithmetic failure is not an empty geometric segment")

    with pytest.raises(PostproposalRejected, match="nonfinite original segment") as failure:
        call(component_guard(forbidden), proposal(displacement=(1.7e308, 0.)))
    assert failure.value.receipt["component_fallback"]["provider_calls"] == 0
    assert failure.value.receipt["loss_calls"] == 1
