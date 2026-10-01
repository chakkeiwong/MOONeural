"""Provider-only K3-to-K9 lineage after finite-trigger and rate enablement.

Real builders/checkpoints use manufactured update/control leaves. Component,
objective and optimizer numerics are forbidden; the parent runs qualification.
"""

import copy
import json

import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_component_policy_refinement as refinements
from tests.contracts import test_generic_component_rate_continuation as rates
from tests.contracts import test_generic_finite_component_policy_transition as triggers
from tests.contracts import test_generic_partial_continuation as partial
from tests.contracts import test_generic_postproposal_policy_transition as policies
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_training_contracts import stable_hash

completed_parent = triggers.completed_parent
k3_parent40 = triggers.k3_parent40
rate_parent50 = triggers.rate_parent50
POLICY, SCHEDULE, RATE = triggers.POLICY, triggers.SCHEDULE, triggers.RATE
COMPONENT_CALLS = []


def component_rows(parameters, batch, update_index, context):
    COMPONENT_CALLS.append(update_index)
    raise AssertionError("complete-cell transition or recovery invoked component numerics")


def guard(count=70, *, finite_fallback=True):
    return policies.enabled_guard(count, provider=component_rows,
        provider_binding=refinements.component_profile(9), finite_component_fallback=finite_fallback)


def component_calls():
    return list(COMPONENT_CALLS), list(refinements.COMPONENT_CALLS), list(policies.COMPONENT_CALLS)


@pytest.fixture(scope="module")
def trigger_parent60(rate_parent50, tmp_path_factory):
    runtime = tmp_path_factory.mktemp("complete-cell-parent60") / "runtime"
    args = triggers.child_arguments(rate_parent50)
    before_calls = component_calls()
    runner, result, calls = refinements.run_host(runtime, args)
    state = result.states[0]
    configuration = rates.assert_lineage(runtime, runner, state)
    assert calls["updates"] == list(range(50, 60)) and calls["controls"] == [50, 60]
    assert state.update_index == state.optimizer_state["iteration"] == state.rng_state["next_seed_index"] == 60
    history = configuration["checkpoint_continuation"]
    assert history[RATE]["ratio"] == 16. and history[RATE]["effective_update"] == 40
    assert history[POLICY]["kind"] == "component_trigger_refinement" and history[POLICY]["effective_update"] == 50
    assert configuration["postproposal"]["profile"]["component_fallback"]["provider"] == refinements.component_profile(3)
    yield state, args, runtime, partial.completed_binding(runtime, state, runtime)
    assert component_calls() == before_calls


def child_arguments(fixture):
    parent, original, _runtime, binding = fixture
    args = {**original, "parameters": parent.policy_state["values"], "updates": 10, "boundary_every": 10,
        "training_schedule": checkpoints.schedule(70), "postproposal": guard(), "stage_id": "complete-cells60-to70",
        "continuation_checkpoint": parent, "continuation_binding": copy.deepcopy(binding),
        "data_binding": copy.deepcopy(original["data_binding"]),
        "source_evidence": copy.deepcopy(original["source_evidence"]),
        POLICY: {"kind": "component_refinement", "effective_update": 60,
                 "reason": "Replace K3 with the explicit K9 provider while retaining the finite trigger"},
        SCHEDULE: {"effective_update": 60, "reason": "Append sixty through sixty-nine to the retained full prefix"}}
    args["data_binding"]["postproposal_component_source"] = finite._reference(__file__)
    args["source_evidence"]["complete_cell_fixture"] = finite._reference(__file__)
    assert RATE not in args
    return args


def assert_provider_only(old, new):
    expected = copy.deepcopy(old)
    expected["profile"]["profile"]["training_schedule"] = new["profile"]["profile"]["training_schedule"]
    expected["component_callback"] = new["component_callback"]
    old_component, new_component = old["profile"]["component_fallback"], new["profile"]["component_fallback"]
    for name in ("provider", "callback"):
        expected["profile"]["component_fallback"][name] = new_component[name]
    assert new == expected
    assert new_component["provider"] == refinements.component_profile(9)
    assert new_component["provider"]["max_component_rows"] == 27
    assert new["component_callback"]["source"] == finite._reference(__file__)
    assert new["component_callback"]["qualname"] == component_rows.__qualname__
    assert new["component_callback"] != old["component_callback"]
    assert new_component["trigger"] == old_component["trigger"] == "invalid-original-linear-segment-or-first-finite-rejection"
    assert new_component["finite_screen"] == old_component["finite_screen"] == "component-fractions-before-original-shorter-fractions"
    assert new_component["finite_trigger_radius"] == old_component["finite_trigger_radius"] == "original-valid-segment-norm"
    assert new["profile"]["maximum_loss_calls"] == old["profile"]["maximum_loss_calls"] == 1 + 2 * len(old["profile"]["fractions"])


def test_complete_cell_refinement_preserves_state_recovers_and_later_inherits(trigger_parent60, tmp_path, monkeypatch):
    parent, _original, parent_root, _binding = trigger_parent60
    parent_files, parent_state = partial.hashes(parent_root), parent.to_dict()
    old_config = json.loads((parent_root / "configuration.json").read_text())
    before_calls = component_calls()
    args = child_arguments(trigger_parent60)
    runtime = tmp_path / "complete70"
    runner, adapter, provider = finite.build_runner(runtime, **args)
    rates.assert_retained(runner.states[0], parent, 1.)
    assert adapter.policy_metadata == parent.policy_state["metadata"]
    configuration = rates.assert_lineage(runtime, runner, runner.states[0])
    history = configuration["checkpoint_continuation"]
    appended, refined = history[SCHEDULE], history[POLICY]
    assert history[RATE] == old_config["checkpoint_continuation"][RATE]
    assert refined["kind"] == "component_refinement"
    assert refined["previous_transition"] == old_config["checkpoint_continuation"][POLICY]
    assert refined["previous_transition"]["kind"] == "component_trigger_refinement"
    assert appended["previous_transition"] == old_config["checkpoint_continuation"][SCHEDULE]
    assert appended["previous_policy_transition_hash"] == stable_hash(refined["previous_transition"])
    intermediate = copy.deepcopy(old_config["postproposal"])
    intermediate["profile"]["profile"]["training_schedule"] = list(checkpoints.schedule(70))
    assert appended["old_postproposal_hash"] == stable_hash(old_config["postproposal"])
    assert appended["new_postproposal_hash"] == refined["old_postproposal_hash"] == stable_hash(intermediate)
    assert refined["new_postproposal_hash"] == stable_hash(configuration["postproposal"])
    assert refined["previous_schedule_transition_hash"] == stable_hash(appended)
    assert refined["old_schedule_hash"] == refined["new_schedule_hash"] == stable_hash(checkpoints.schedule(70))
    for receipt in (appended, refined):
        assert receipt["effective_update"] == 60
        assert receipt["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
        assert receipt["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    assert refined["old_component_fallback"] == old_config["postproposal"]["profile"]["component_fallback"]
    assert refined["old_component_callback"] == old_config["postproposal"]["component_callback"]
    assert_provider_only(old_config["postproposal"], configuration["postproposal"])
    assert configuration["training"]["schedule"][:60] == old_config["training"]["schedule"]
    for name in ("callback", "values_callback"):
        assert configuration[name] == old_config[name]
    assert configuration["training"]["callback"] == old_config["training"]["callback"]
    expected_data = {**old_config["data_binding"], "postproposal_component_source": finite._reference(__file__)}
    assert configuration["data_binding"] == expected_data
    calls = checkpoints.host_runtime(runner, provider, monkeypatch)
    inherited_step = runner.executor.step
    controls_before = runner.executor.core.coordinator.read(runner.states[0]).controls

    def step_after_fresh_entry(state, batch, model_adapter):
        controls = runner.executor.core.coordinator.read(state).controls
        assert controls[:len(controls_before)] == controls_before
        assert controls[len(controls_before)].update_index == 60
        assert controls[len(controls_before)].stage_id == runner.stages[0].stage_id
        return inherited_step(state, batch, model_adapter)

    monkeypatch.setattr(runner.executor, "step", step_after_fresh_entry)
    result = runner.run()
    endpoint = result.states[0]
    assert calls["updates"] == list(range(60, 70)) and calls["controls"] == [60, 70]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 70
    assert endpoint.optimizer_state["learning_rate"] == parent.optimizer_state["learning_rate"]
    assert endpoint.method_state["rates"] == parent.method_state["rates"]
    assert endpoint.method_state["preferred"] == parent.method_state["preferred"]
    assert endpoint.optimizer_state["first_moment"] == [value + 10. for value in parent.optimizer_state["first_moment"]]
    assert endpoint.optimizer_state["second_moment"] == [value + 20. for value in parent.optimizer_state["second_moment"]]
    previous_history = list(parent.metadata["permanent_pass_rotation"]["history"])
    assert list(endpoint.metadata["permanent_pass_rotation"]["history"][:len(previous_history)]) == previous_history
    rates.assert_lineage(runtime, runner, endpoint)
    refinements.assert_recovery(runtime, args, result)
    assert component_calls() == before_calls

    retained_binding = partial.completed_binding(runtime, endpoint, runtime)
    retained_files = partial.hashes(runtime)
    later = {**args, "parameters": endpoint.policy_state["values"], "training_schedule": checkpoints.schedule(80),
        "postproposal": guard(80), "stage_id": "complete-cells-inherited70-to80",
        "continuation_checkpoint": endpoint, "continuation_binding": retained_binding,
        SCHEDULE: {"effective_update": 70, "reason": "Append with the inherited complete-cell provider and trigger"}}
    later.pop(POLICY)
    following_root = tmp_path / "ordinary80"
    following, _adapter, provider = finite.build_runner(following_root, **later)
    rates.assert_retained(following.states[0], endpoint, 1.)
    following_config = rates.assert_lineage(following_root, following, following.states[0])
    assert following_config["checkpoint_continuation"][POLICY] == refined
    assert following_config["checkpoint_continuation"][RATE] == history[RATE]
    assert following_config["checkpoint_continuation"][SCHEDULE]["previous_transition"] == appended
    assert following_config["checkpoint_continuation"][SCHEDULE]["previous_policy_transition_hash"] == stable_hash(refined)
    expected_guard = copy.deepcopy(configuration["postproposal"])
    expected_guard["profile"]["profile"]["training_schedule"] = list(checkpoints.schedule(80))
    assert following_config["postproposal"] == expected_guard
    calls = checkpoints.host_runtime(following, provider, monkeypatch)
    final = following.run()
    assert calls["updates"] == list(range(70, 80)) and calls["controls"] == [70, 80]
    assert final.states[0].update_index == final.states[0].optimizer_state["iteration"] == final.states[0].rng_state["next_seed_index"] == 80
    assert final.states[0].optimizer_state["learning_rate"] == endpoint.optimizer_state["learning_rate"]
    assert final.states[0].method_state["rates"] == endpoint.method_state["rates"]
    rates.assert_lineage(following_root, following, final.states[0])
    refinements.assert_recovery(following_root, later, final)
    assert partial.hashes(runtime) == retained_files
    assert parent.to_dict() == parent_state and partial.hashes(parent_root) == parent_files
    assert component_calls() == before_calls


def test_complete_cell_interruption_and_completed_recovery_preserve_all_lineages(trigger_parent60, tmp_path, monkeypatch):
    parent, _original, parent_root, _binding = trigger_parent60
    parent_files = partial.hashes(parent_root)
    before_calls = component_calls()
    args = child_arguments(trigger_parent60)
    _expected_runner, expected, _calls = refinements.run_host(tmp_path / "uninterrupted", args)
    runtime = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    rates.assert_retained(runner.states[0], parent, 1.)
    initial_config = rates.assert_lineage(runtime, runner, runner.states[0])
    calls = checkpoints.host_runtime(runner, provider, monkeypatch, fail_at=65)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    assert calls["updates"] == list(range(60, 65)) and calls["controls"] == [60]
    resumed, _adapter, provider = finite.build_runner(runtime, **args)
    calls = checkpoints.host_runtime(resumed, provider, monkeypatch)
    actual = resumed.run()
    assert calls["updates"] == list(range(65, 70)) and calls["controls"] == [70]
    assert checkpoints.numerical(actual.states[0]) == checkpoints.numerical(expected.states[0])
    configuration = rates.assert_lineage(runtime, resumed, actual.states[0])
    for name in (POLICY, SCHEDULE, RATE):
        assert configuration["checkpoint_continuation"][name] == initial_config["checkpoint_continuation"][name]
    refinements.assert_recovery(runtime, args, actual)
    assert partial.hashes(parent_root) == parent_files and component_calls() == before_calls


def test_complete_cell_refinement_refuses_undeclared_provider_and_trigger_or_rate_cochanges(trigger_parent60, tmp_path):
    parent, original, parent_root, _binding = trigger_parent60
    parent_files, parent_state = partial.hashes(parent_root), parent.to_dict()
    before_calls = component_calls()
    faults = ("undeclared-provider", "wrong-clock", "wrong-kind", "stale-source", "unchanged-provider",
              "removed-trigger", "changed-loss-cap", "new-rate", "undeclared-rate", "rewritten-prefix")
    for fault in faults:
        args = child_arguments(trigger_parent60)
        if fault == "undeclared-provider":
            args.pop(POLICY)
        elif fault == "wrong-clock":
            args[POLICY]["effective_update"] = 59
        elif fault == "wrong-kind":
            args[POLICY]["kind"] = "component_trigger_refinement"
        elif fault == "stale-source":
            args["data_binding"]["postproposal_component_source"] = original["data_binding"]["postproposal_component_source"]
        elif fault == "unchanged-provider":
            args["postproposal"] = triggers.guard(70)
            args["data_binding"] = copy.deepcopy(original["data_binding"])
            args["source_evidence"] = copy.deepcopy(original["source_evidence"])
        elif fault == "removed-trigger":
            args["postproposal"] = guard(finite_fallback=False)
        elif fault == "changed-loss-cap":
            args["postproposal"].binding["maximum_loss_calls"] += 1
            args["postproposal"].binding_hash = stable_hash(args["postproposal"].binding)
        elif fault in ("new-rate", "undeclared-rate"):
            args["learning_rate"] *= 2.
            if fault == "new-rate":
                args[RATE] = {"effective_update": 60, "reason": "Forbidden new rate during provider refinement"}
        else:
            changed = list(checkpoints.schedule(70))
            changed[59] = {**changed[59], "row_indices": [99]}
            args["training_schedule"] = tuple(changed)
            args["postproposal"].binding["profile"]["training_schedule"] = changed
            args["postproposal"].binding_hash = stable_hash(args["postproposal"].binding)
        with pytest.raises((ValueError, TypeError)):
            finite.build_runner(tmp_path / fault, **args)
    assert parent.to_dict() == parent_state and partial.hashes(parent_root) == parent_files
    assert component_calls() == before_calls
