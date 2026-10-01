"""Trigger-only policy lineage with real builders and manufactured runtime leaves.

The parent runs qualification. These tests never evaluate the economic model,
component provider or numerical optimizer. Guard numerics have a separate suite.
"""

import copy
import json

import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_component_policy_refinement as refinements
from tests.contracts import test_generic_component_rate_continuation as rates
from tests.contracts import test_generic_partial_continuation as partial
from tests.contracts import test_generic_postproposal_policy_transition as policies
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_training_contracts import stable_hash

completed_parent = refinements.completed_parent
k3_parent40 = rates.k3_parent40
POLICY, SCHEDULE, RATE = rates.POLICY, rates.SCHEDULE, rates.RATE


def guard(count=60, **changes):
    return refinements.guard(count, finite_component_fallback=True, **changes)


def declaration(clock=50):
    return {"kind": "component_trigger_refinement", "effective_update": clock,
            "reason": "Enable finite-rejection fallback with the same inherited K3 provider"}


@pytest.fixture(scope="module")
def rate_parent50(k3_parent40, tmp_path_factory):
    runtime = tmp_path_factory.mktemp("finite-component-parent50") / "runtime"
    args = rates.rate_arguments(k3_parent40)
    before_calls = list(refinements.COMPONENT_CALLS), list(policies.COMPONENT_CALLS)
    runner, result, calls = refinements.run_host(runtime, args)
    state = result.states[0]
    rates.assert_lineage(runtime, runner, state)
    assert calls["updates"] == list(range(40, 50)) and calls["controls"] == [40, 50]
    assert state.update_index == state.optimizer_state["iteration"] == state.rng_state["next_seed_index"] == 50
    assert state.metadata["checkpoint_continuation"][RATE]["ratio"] == 16.
    assert state.metadata["checkpoint_continuation"][RATE]["effective_update"] == 40
    yield state, args, runtime, partial.completed_binding(runtime, state, runtime)
    assert (refinements.COMPONENT_CALLS, policies.COMPONENT_CALLS) == before_calls


def child_arguments(fixture):
    parent, original, _runtime, binding = fixture
    args = {**original, "parameters": parent.policy_state["values"], "updates": 10, "boundary_every": 10,
        "training_schedule": checkpoints.schedule(60), "postproposal": guard(), "stage_id": "finite-trigger50-to60",
        "continuation_checkpoint": parent, "continuation_binding": copy.deepcopy(binding),
        "data_binding": copy.deepcopy(original["data_binding"]),
        "source_evidence": copy.deepcopy(original["source_evidence"]), POLICY: declaration(),
        SCHEDULE: {"effective_update": 50, "reason": "Append fifty through fifty-nine without changing pooled inputs"}}
    args.pop(RATE)
    return args


def assert_trigger_only(old, new):
    expected = copy.deepcopy(old)
    expected["profile"]["profile"]["training_schedule"] = new["profile"]["profile"]["training_schedule"]
    expected["profile"]["maximum_loss_calls"] = 1 + 2 * len(old["profile"]["fractions"])
    expected["profile"]["component_fallback"].update(
        trigger="invalid-original-linear-segment-or-first-finite-rejection",
        finite_screen="component-fractions-before-original-shorter-fractions",
        finite_trigger_radius="original-valid-segment-norm")
    assert new == expected
    assert new["component_callback"] == old["component_callback"]
    for name in ("provider", "callback"):
        assert new["profile"]["component_fallback"][name] == old["profile"]["component_fallback"][name]


def test_trigger_at50_preserves_slots_rate_history_and_completed_recovery_then_inherits(rate_parent50, tmp_path, monkeypatch):
    parent, _original, parent_root, _binding = rate_parent50
    parent_files, parent_state = partial.hashes(parent_root), parent.to_dict()
    old_config = json.loads((parent_root / "configuration.json").read_text())
    before_calls = list(refinements.COMPONENT_CALLS), list(policies.COMPONENT_CALLS)
    args = child_arguments(rate_parent50)
    runtime = tmp_path / "finite60"
    runner, adapter, provider = finite.build_runner(runtime, **args)
    rates.assert_retained(runner.states[0], parent, 1.)
    assert adapter.policy_metadata == parent.policy_state["metadata"]
    configuration = rates.assert_lineage(runtime, runner, runner.states[0])
    history = configuration["checkpoint_continuation"]
    appended, triggered = history[SCHEDULE], history[POLICY]
    assert history[RATE] == old_config["checkpoint_continuation"][RATE]
    assert triggered["kind"] == "component_trigger_refinement"
    assert triggered["previous_transition"] == old_config["checkpoint_continuation"][POLICY]
    assert appended["previous_transition"] == old_config["checkpoint_continuation"][SCHEDULE]
    assert appended["previous_policy_transition_hash"] == stable_hash(triggered["previous_transition"])
    intermediate = copy.deepcopy(old_config["postproposal"])
    intermediate["profile"]["profile"]["training_schedule"] = list(checkpoints.schedule(60))
    assert appended["old_postproposal_hash"] == stable_hash(old_config["postproposal"])
    assert appended["new_postproposal_hash"] == triggered["old_postproposal_hash"] == stable_hash(intermediate)
    assert triggered["new_postproposal_hash"] == stable_hash(configuration["postproposal"])
    assert triggered["previous_schedule_transition_hash"] == stable_hash(appended)
    assert triggered["old_schedule_hash"] == triggered["new_schedule_hash"] == stable_hash(checkpoints.schedule(60))
    for receipt in (appended, triggered):
        assert receipt["effective_update"] == 50
        assert receipt["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
        assert receipt["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    assert triggered["old_component_fallback"] == old_config["postproposal"]["profile"]["component_fallback"]
    assert triggered["old_component_callback"] == triggered["component_callback"] == old_config["postproposal"]["component_callback"]
    assert_trigger_only(old_config["postproposal"], configuration["postproposal"])
    assert configuration["training"]["schedule"][:50] == old_config["training"]["schedule"]
    for name in ("callback", "values_callback"):
        assert configuration[name] == old_config[name]
    assert configuration["training"]["callback"] == old_config["training"]["callback"]
    assert configuration["data_binding"] == old_config["data_binding"]
    calls = checkpoints.host_runtime(runner, provider, monkeypatch)
    inherited_step = runner.executor.step
    controls_before = runner.executor.core.coordinator.read(runner.states[0]).controls

    def step_after_fresh_entry(state, batch, model_adapter):
        controls = runner.executor.core.coordinator.read(state).controls
        assert controls[:len(controls_before)] == controls_before
        assert controls[len(controls_before)].update_index == 50
        assert controls[len(controls_before)].stage_id == runner.stages[0].stage_id
        return inherited_step(state, batch, model_adapter)

    monkeypatch.setattr(runner.executor, "step", step_after_fresh_entry)
    result = runner.run()
    endpoint = result.states[0]
    assert calls["updates"] == list(range(50, 60)) and calls["controls"] == [50, 60]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 60
    assert endpoint.optimizer_state["learning_rate"] == parent.optimizer_state["learning_rate"]
    assert endpoint.method_state["rates"] == parent.method_state["rates"]
    assert endpoint.method_state["preferred"] == parent.method_state["preferred"]
    assert endpoint.optimizer_state["first_moment"] == [value + 10. for value in parent.optimizer_state["first_moment"]]
    assert endpoint.optimizer_state["second_moment"] == [value + 20. for value in parent.optimizer_state["second_moment"]]
    previous_history = list(parent.metadata["permanent_pass_rotation"]["history"])
    assert list(endpoint.metadata["permanent_pass_rotation"]["history"][:len(previous_history)]) == previous_history
    rates.assert_lineage(runtime, runner, endpoint)
    refinements.assert_recovery(runtime, args, result)

    retained_binding = partial.completed_binding(runtime, endpoint, runtime)
    retained_files = partial.hashes(runtime)
    later = {**args, "parameters": endpoint.policy_state["values"], "training_schedule": checkpoints.schedule(70),
        "postproposal": guard(70), "stage_id": "finite-trigger-inherited60-to70",
        "continuation_checkpoint": endpoint, "continuation_binding": retained_binding,
        SCHEDULE: {"effective_update": 60, "reason": "Append after trigger enablement without reapplying transitions"}}
    later.pop(POLICY)
    following_root = tmp_path / "ordinary70"
    following, _adapter, provider = finite.build_runner(following_root, **later)
    rates.assert_retained(following.states[0], endpoint, 1.)
    following_config = rates.assert_lineage(following_root, following, following.states[0])
    assert following_config["checkpoint_continuation"][POLICY] == triggered
    assert following_config["checkpoint_continuation"][RATE] == history[RATE]
    assert following_config["checkpoint_continuation"][SCHEDULE]["previous_transition"] == appended
    assert following_config["checkpoint_continuation"][SCHEDULE]["previous_policy_transition_hash"] == stable_hash(triggered)
    expected_guard = copy.deepcopy(configuration["postproposal"])
    expected_guard["profile"]["profile"]["training_schedule"] = list(checkpoints.schedule(70))
    assert following_config["postproposal"] == expected_guard
    calls = checkpoints.host_runtime(following, provider, monkeypatch)
    final = following.run()
    assert calls["updates"] == list(range(60, 70)) and calls["controls"] == [60, 70]
    assert final.states[0].optimizer_state["learning_rate"] == endpoint.optimizer_state["learning_rate"]
    assert final.states[0].method_state["rates"] == endpoint.method_state["rates"]
    rates.assert_lineage(following_root, following, final.states[0])
    refinements.assert_recovery(following_root, later, final)
    assert partial.hashes(runtime) == retained_files
    assert parent.to_dict() == parent_state and partial.hashes(parent_root) == parent_files
    assert (refinements.COMPONENT_CALLS, policies.COMPONENT_CALLS) == before_calls


def test_interruption_resumes_only_uncommitted_updates_without_retrigger_or_rate_scale(rate_parent50, tmp_path, monkeypatch):
    parent, _original, parent_root, _binding = rate_parent50
    parent_files = partial.hashes(parent_root)
    args = child_arguments(rate_parent50)
    _expected_runner, expected, _calls = refinements.run_host(tmp_path / "uninterrupted", args)
    runtime = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    rates.assert_retained(runner.states[0], parent, 1.)
    initial_config = rates.assert_lineage(runtime, runner, runner.states[0])
    calls = checkpoints.host_runtime(runner, provider, monkeypatch, fail_at=55)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    assert calls["updates"] == list(range(50, 55)) and calls["controls"] == [50]
    resumed, _adapter, provider = finite.build_runner(runtime, **args)
    calls = checkpoints.host_runtime(resumed, provider, monkeypatch)
    actual = resumed.run()
    assert calls["updates"] == list(range(55, 60)) and calls["controls"] == [60]
    assert checkpoints.numerical(actual.states[0]) == checkpoints.numerical(expected.states[0])
    configuration = rates.assert_lineage(runtime, resumed, actual.states[0])
    for name in (POLICY, SCHEDULE, RATE):
        assert configuration["checkpoint_continuation"][name] == initial_config["checkpoint_continuation"][name]
    refinements.assert_recovery(runtime, args, actual)
    assert partial.hashes(parent_root) == parent_files


def test_trigger_transition_refuses_provider_numerical_and_schedule_cochanges(rate_parent50, tmp_path):
    parent, _original, parent_root, _binding = rate_parent50
    parent_files, parent_state = partial.hashes(parent_root), parent.to_dict()
    before_calls = list(refinements.COMPONENT_CALLS), list(policies.COMPONENT_CALLS)
    faults = ("undeclared", "wrong-clock", "schedule-clock", "missing-schedule", "empty-reason", "missing-kind",
        "unknown-kind", "extra-key", "same-trigger", "provider-profile", "provider-callback", "fractions", "margin",
        "tolerance", "normalization", "limits", "prefix", "wrong-source", "new-rate", "undeclared-rate")
    for fault in faults:
        args = child_arguments(rate_parent50)
        if fault == "undeclared":
            args.pop(POLICY)
        elif fault == "wrong-clock":
            args[POLICY]["effective_update"] = 49
        elif fault == "schedule-clock":
            args[SCHEDULE]["effective_update"] = 49
        elif fault == "missing-schedule":
            args.pop(SCHEDULE)
        elif fault == "empty-reason":
            args[POLICY]["reason"] = " "
        elif fault == "missing-kind":
            args[POLICY].pop("kind")
        elif fault == "unknown-kind":
            args[POLICY]["kind"] = "replace-arbitrary-trigger"
        elif fault == "extra-key":
            args[POLICY]["reset_moments"] = False
        elif fault == "same-trigger":
            args["postproposal"] = refinements.guard(60)
        elif fault == "provider-profile":
            args["postproposal"] = guard(top_k=2)
        elif fault == "provider-callback":
            args["postproposal"] = guard(old=True)
            args["data_binding"]["postproposal_component_source"] = finite._reference(policies.__file__)
            args["source_evidence"]["replacement_component_fixture"] = finite._reference(policies.__file__)
        elif fault in ("fractions", "margin", "tolerance"):
            options = {"fractions": {"fractions": (1., .1)}, "margin": {"margin_fraction": .2},
                       "tolerance": {"relative_tolerance": 1e-10}}[fault]
            args["postproposal"] = guard(**options)
        elif fault in ("normalization", "limits"):
            current = args["postproposal"]
            altered = [2., *([1.] * 6)]
            args["postproposal"] = type(current)(partial.guard_losses, partial.batch_binding,
                altered if fault == "normalization" else [1.] * 7,
                altered if fault == "limits" else [1.] * 7,
                binding=current.binding["profile"], component_rows=refinements.component_rows,
                component_binding=refinements.component_profile(3), finite_component_fallback=True)
        elif fault == "prefix":
            changed = list(checkpoints.schedule(60))
            changed[49] = {**changed[49], "row_indices": [99]}
            args["training_schedule"] = tuple(changed)
            args["postproposal"].binding["profile"]["training_schedule"] = changed
            args["postproposal"].binding_hash = stable_hash(args["postproposal"].binding)
        elif fault == "wrong-source":
            args["data_binding"]["postproposal_component_source"] = finite._reference(policies.__file__)
        else:
            args["learning_rate"] *= 2.
            if fault == "new-rate":
                args[RATE] = {"effective_update": 50, "reason": "Forbidden concurrent new rate"}
        with pytest.raises((ValueError, TypeError)):
            finite.build_runner(tmp_path / fault, **args)
    assert parent.to_dict() == parent_state and partial.hashes(parent_root) == parent_files
    assert (refinements.COMPONENT_CALLS, policies.COMPONENT_CALLS) == before_calls


def test_provider_refinement_kind_cannot_enable_finite_trigger(rate_parent50, tmp_path):
    before_calls = list(refinements.COMPONENT_CALLS), list(policies.COMPONENT_CALLS)
    for replace_provider in (False, True):
        args = child_arguments(rate_parent50)
        args[POLICY] = refinements.declaration(50)
        if replace_provider:
            args["postproposal"] = guard(top_k=4)
        with pytest.raises(ValueError):
            finite.build_runner(tmp_path / ("provider-and-trigger" if replace_provider else "trigger-only"), **args)
    assert (refinements.COMPONENT_CALLS, policies.COMPONENT_CALLS) == before_calls
