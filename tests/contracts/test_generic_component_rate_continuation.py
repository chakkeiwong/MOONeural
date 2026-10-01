"""Compose retained K3, rate and schedule lineages with manufactured updates.

The parent runs qualification. Recovery reconstructs runners in this process;
no economic model, component provider or numerical optimizer is evaluated.
"""

import copy
import json

import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_component_policy_refinement as refinements
from tests.contracts import test_generic_partial_continuation as partial
from tests.contracts import test_generic_postproposal_policy_transition as policies
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_training_contracts import canonical_json, stable_hash

completed_parent = refinements.completed_parent
POLICY, SCHEDULE = refinements.POLICY, refinements.SCHEDULE
RATE = "learning_rate_transition"


@pytest.fixture(scope="module")
def k3_parent40(completed_parent, tmp_path_factory):
    runtime = tmp_path_factory.mktemp("component-rate-parent40") / "runtime"
    args = refinements.child_arguments(completed_parent)
    before_calls = list(refinements.COMPONENT_CALLS), list(policies.COMPONENT_CALLS)
    _runner, result, calls = refinements.run_host(runtime, args)
    state = result.states[0]
    assert state.update_index == state.optimizer_state["iteration"] == state.rng_state["next_seed_index"] == 40
    assert calls["updates"] == list(range(30, 40))
    binding = partial.completed_binding(runtime, state, runtime)
    yield state, args, runtime, binding
    assert (refinements.COMPONENT_CALLS, policies.COMPONENT_CALLS) == before_calls


def rate_arguments(fixture):
    parent, original, _runtime, binding = fixture
    args = {**original, "parameters": parent.policy_state["values"], "updates": 10, "boundary_every": 10,
        "learning_rate": original["learning_rate"] * 16., "training_schedule": checkpoints.schedule(50),
        "postproposal": refinements.guard(50), "stage_id": "k3-rate16-at40",
        "continuation_checkpoint": parent, "continuation_binding": copy.deepcopy(binding),
        RATE: {"effective_update": 40, "reason": "Manufactured sixteenfold rate with the retained K3 provider"},
        SCHEDULE: {"effective_update": 40, "reason": "Append ten entries without changing the enabled provider"}}
    args.pop(POLICY)
    return args


def assert_retained(state, parent, ratio):
    expected = copy.deepcopy(checkpoints.numerical(parent))
    expected["optimizer_state"]["learning_rate"] *= ratio
    expected["method_state"]["rates"] = {
        method: rate * ratio for method, rate in expected["method_state"]["rates"].items()}
    assert checkpoints.numerical(state) == expected
    assert state.metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]


def assert_lineage(runtime, runner, state):
    configuration = refinements.assert_transition_copies(runtime, runner, state)
    receipt = configuration["checkpoint_continuation"][RATE]
    assert canonical_json(receipt) == canonical_json(runner.design.optimizer_binding[RATE])
    assert canonical_json(receipt) == canonical_json(state.metadata["checkpoint_continuation"][RATE])
    bound_state = state if "replication_design" in state.metadata else runner.design.bind_checkpoint(state)
    assert finite._rate_transition(bound_state, configuration, configuration["learning_rate"], None) == receipt
    assert not runner.design.optimizer_binding["fresh_moments"]
    return configuration


def test_k3_rate_and_schedule_complete_recover_then_extend_without_rescaling(k3_parent40, tmp_path, monkeypatch):
    parent, _original, parent_root, _binding = k3_parent40
    parent_files, parent_state = partial.hashes(parent_root), parent.to_dict()
    parent_config = json.loads((parent_root / "configuration.json").read_text())
    inherited_policy = parent_config["checkpoint_continuation"][POLICY]
    args = rate_arguments(k3_parent40)
    runtime = tmp_path / "rate50"
    runner, adapter, provider = finite.build_runner(runtime, **args)
    assert_retained(runner.states[0], parent, 16.)
    assert adapter.policy_metadata == parent.policy_state["metadata"]
    configuration = assert_lineage(runtime, runner, runner.states[0])
    history = configuration["checkpoint_continuation"]
    receipt = history[RATE]
    assert receipt["effective_update"] == 40 and receipt["ratio"] == 16.
    assert receipt["previous_transition"] is None
    assert receipt["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
    assert receipt["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    assert receipt["new_optimizer"]["adam"] == receipt["old_optimizer"]["adam"]
    assert history[POLICY] == inherited_policy
    assert history[SCHEDULE]["previous_transition"] == parent_config["checkpoint_continuation"][SCHEDULE]
    assert history[SCHEDULE]["previous_policy_transition_hash"] == stable_hash(inherited_policy)
    assert history[SCHEDULE]["parent_checkpoint_hash"] == receipt["parent_checkpoint_hash"]
    assert history[SCHEDULE]["parent_design_hash"] == receipt["parent_design_hash"]
    assert configuration["training"]["schedule"][:40] == parent_config["training"]["schedule"]
    expected_guard = copy.deepcopy(parent_config["postproposal"])
    expected_guard["profile"]["profile"]["training_schedule"] = list(checkpoints.schedule(50))
    assert configuration["postproposal"] == expected_guard
    for name in ("callback", "values_callback"):
        assert configuration[name] == parent_config[name]
    assert configuration["training"]["callback"] == parent_config["training"]["callback"]
    for name in ("objective_recipe", "postproposal_component_source"):
        assert configuration["data_binding"][name] == parent_config["data_binding"][name]
    calls = checkpoints.host_runtime(runner, provider, monkeypatch)
    host_step = runner.executor.step
    parent_controls = runner.executor.core.coordinator.read(runner.states[0]).controls

    def step_after_fresh_entry(state, batch, model_adapter):
        controls = runner.executor.core.coordinator.read(state).controls
        assert controls[:len(parent_controls)] == parent_controls
        entry = controls[len(parent_controls)]
        assert entry.update_index == 40 and entry.stage_id == runner.stages[0].stage_id
        return host_step(state, batch, model_adapter)

    monkeypatch.setattr(runner.executor, "step", step_after_fresh_entry)
    result = runner.run()
    endpoint = result.states[0]
    assert calls["updates"] == list(range(40, 50)) and calls["controls"] == [40, 50]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 50
    assert endpoint.optimizer_state["learning_rate"] == parent.optimizer_state["learning_rate"] * 16.
    assert endpoint.method_state["rates"] == {method: rate * 16. for method, rate in parent.method_state["rates"].items()}
    assert endpoint.optimizer_state["first_moment"] == [value + 10. for value in parent.optimizer_state["first_moment"]]
    assert endpoint.optimizer_state["second_moment"] == [value + 20. for value in parent.optimizer_state["second_moment"]]
    old_history = list(parent.metadata["permanent_pass_rotation"]["history"])
    assert list(endpoint.metadata["permanent_pass_rotation"]["history"][:len(old_history)]) == old_history
    assert_lineage(runtime, runner, endpoint)
    refinements.assert_recovery(runtime, args, result)

    binding = partial.completed_binding(runtime, endpoint, runtime)
    rate_files = partial.hashes(runtime)
    later = {**args, "parameters": endpoint.policy_state["values"], "training_schedule": checkpoints.schedule(60),
        "postproposal": refinements.guard(60), "stage_id": "k3-same-rate50-to60",
        "continuation_checkpoint": endpoint, "continuation_binding": binding,
        SCHEDULE: {"effective_update": 50, "reason": "Append at the inherited rate without scaling again"}}
    later.pop(RATE)
    following_root = tmp_path / "same-rate60"
    following, _adapter, provider = finite.build_runner(following_root, **later)
    assert_retained(following.states[0], endpoint, 1.)
    following_config = assert_lineage(following_root, following, following.states[0])
    assert following_config["checkpoint_continuation"][RATE] == receipt
    assert following_config["checkpoint_continuation"][POLICY] == inherited_policy
    assert following_config["checkpoint_continuation"][SCHEDULE]["previous_transition"] == history[SCHEDULE]
    calls = checkpoints.host_runtime(following, provider, monkeypatch)
    final = following.run()
    assert calls["updates"] == list(range(50, 60)) and calls["controls"] == [50, 60]
    assert final.states[0].update_index == final.states[0].optimizer_state["iteration"] == 60
    assert final.states[0].rng_state["next_seed_index"] == 60
    assert final.states[0].optimizer_state["learning_rate"] == endpoint.optimizer_state["learning_rate"]
    assert final.states[0].method_state["rates"] == endpoint.method_state["rates"]
    assert_lineage(following_root, following, final.states[0])
    refinements.assert_recovery(following_root, later, final)
    assert partial.hashes(runtime) == rate_files
    assert parent.to_dict() == parent_state and partial.hashes(parent_root) == parent_files


def test_k3_rate_schedule_interruption_and_completed_recovery_do_not_rescale(k3_parent40, tmp_path, monkeypatch):
    parent, _original, parent_root, _binding = k3_parent40
    parent_files = partial.hashes(parent_root)
    args = rate_arguments(k3_parent40)
    _expected_runner, expected, _calls = refinements.run_host(tmp_path / "uninterrupted", args)
    runtime = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    assert_retained(runner.states[0], parent, 16.)
    initial_config = assert_lineage(runtime, runner, runner.states[0])
    calls = checkpoints.host_runtime(runner, provider, monkeypatch, fail_at=45)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    assert calls["updates"] == list(range(40, 45)) and calls["controls"] == [40]
    resumed, _adapter, provider = finite.build_runner(runtime, **args)
    calls = checkpoints.host_runtime(resumed, provider, monkeypatch)
    actual = resumed.run()
    assert calls["updates"] == list(range(45, 50)) and calls["controls"] == [50]
    assert checkpoints.numerical(actual.states[0]) == checkpoints.numerical(expected.states[0])
    configuration = assert_lineage(runtime, resumed, actual.states[0])
    for name in (RATE, POLICY, SCHEDULE):
        assert configuration["checkpoint_continuation"][name] == initial_config["checkpoint_continuation"][name]
    refinements.assert_recovery(runtime, args, actual)
    assert partial.hashes(parent_root) == parent_files


@pytest.mark.parametrize("refusal_clock", [40, 45], ids=["zero-progress", "retained-prefix"])
def test_k3_rate_refusal_partial_child_inherits_without_second_scale(k3_parent40, tmp_path, refusal_clock):
    parent, _original, parent_root, _binding = k3_parent40
    parent_files = partial.hashes(parent_root)
    args = rate_arguments(k3_parent40)
    runtime = tmp_path / "refused"
    state, binding, calls = policies.refused_run(runtime, args, clock=refusal_clock)
    assert calls["updates"] == list(range(40, refusal_clock)) and calls["controls"] == [40]
    assert state.optimizer_state["iteration"] == state.rng_state["next_seed_index"] == refusal_clock
    assert state.optimizer_state["learning_rate"] == parent.optimizer_state["learning_rate"] * 16.
    assert state.method_state["rates"] == {method: rate * 16. for method, rate in parent.method_state["rates"].items()}
    refused_files, refused_state = partial.hashes(runtime), state.to_dict()
    inherited = state.metadata["checkpoint_continuation"]
    later = {**args, "parameters": state.policy_state["values"], "updates": 50 - refusal_clock,
        "boundary_every": 50 - refusal_clock, "stage_id": f"k3-rate-inherited-partial{refusal_clock}",
        "continuation_checkpoint": state, "continuation_binding": binding,
        "partial_parent_continuation": partial.declaration(refusal_clock)}
    later.pop(RATE)
    later.pop(SCHEDULE)
    resumed_root = tmp_path / "retained-partial50"
    runner, _adapter, provider = finite.build_runner(resumed_root, **later)
    assert_retained(runner.states[0], state, 1.)
    configuration = assert_lineage(resumed_root, runner, runner.states[0])
    for name in (RATE, POLICY, SCHEDULE):
        assert configuration["checkpoint_continuation"][name] == inherited[name]
    assert configuration["training"]["schedule"] == list(checkpoints.schedule(50))
    with pytest.MonkeyPatch.context() as patches:
        calls = checkpoints.host_runtime(runner, provider, patches)
        result = runner.run()
    assert calls["updates"] == list(range(refusal_clock, 50)) and calls["controls"] == [refusal_clock, 50]
    endpoint = result.states[0]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 50
    assert endpoint.optimizer_state["learning_rate"] == state.optimizer_state["learning_rate"]
    assert endpoint.method_state["rates"] == state.method_state["rates"]
    previous_history = list(state.metadata["permanent_pass_rotation"]["history"])
    history = list(endpoint.metadata["permanent_pass_rotation"]["history"])
    assert history[:len(previous_history)] == previous_history
    if refusal_clock == 40:
        entries = [point for point in history if point["update_index"] == 40 and point.get("stage_id") is not None]
        assert len(entries) == 2 and entries[0]["stage_id"] != entries[1]["stage_id"]
    assert_lineage(resumed_root, runner, endpoint)
    refinements.assert_recovery(resumed_root, later, result)
    assert partial.hashes(runtime) == refused_files and state.to_dict() == refused_state
    assert partial.hashes(parent_root) == parent_files
