"""One real guarded schedule extension with an explicit method transition."""

import json

from tests.contracts import test_generic_postproposal_continuation as fixtures
from mooneural.training import generic_finite_objective_runner as finite


def test_guarded_schedule_and_pcgrad_transition_preserve_state_and_recover(tmp_path, monkeypatch):
    original = fixtures.arguments(1)
    parent_root = tmp_path / "parent"
    parent_runner, _adapter, _provider = finite.build_runner(parent_root, **original)
    parent = parent_runner.run().states[0]
    assert parent.update_index == parent.optimizer_state["iteration"] == 1
    assert parent.method_state["preferred"] == "cagrad"
    assert any(parent.optimizer_state["first_moment"])
    assert any(parent.optimizer_state["second_moment"])
    parent_state = parent.to_dict()
    prior = parent_runner.executor.core.coordinator.read(parent)
    child = fixtures.continuation_arguments(original, parent, parent_root, 2, "guarded-pcgrad")
    declaration = {"effective_update": 1, "old_preferred": "cagrad",
                   "new_preferred": "pcgrad", "reason": "explicit manufactured method continuation"}
    child["method_policy_transition"] = declaration
    parent_hashes = fixtures.file_hashes(parent_root)
    runtime = tmp_path / "child"
    runner, adapter, _provider = finite.build_runner(runtime, **child)
    expected = fixtures.numerical(parent)
    expected["method_state"]["preferred"] = "pcgrad"
    assert fixtures.numerical(runner.states[0]) == expected
    assert runner.states[0].metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]
    assert adapter.policy_metadata == parent.policy_state["metadata"]
    assert runner.design.optimizer_binding["fresh_moments"] is False
    assert (runner.stages[0].start_update, runner.stages[0].stop_update) == (1, 2)

    result = runner.run()
    state = result.states[0]
    assert state.update_index == state.optimizer_state["iteration"] == state.rng_state["next_seed_index"] == 2
    assert state.method_state["preferred"] == "pcgrad"
    assert state.method_state["rates"] == parent.method_state["rates"]
    assert state.policy_state["values"] != parent.policy_state["values"]
    rotation = runner.executor.core.coordinator.read(state)
    assert rotation.controls[:len(prior.controls)] == prior.controls
    assert [point.update_index for point in rotation.controls] == [0, 1, 1, 2]
    assert rotation.controls[2].stage_id == "guarded-pcgrad"
    assert rotation.permanent == prior.permanent
    transactions = list((runtime / "checkpoints/arms/arm-0/checkpoints").glob("update-????????.json"))
    assert len(transactions) == 1
    event = json.loads(transactions[0].read_text())["event"]
    assert event["method"] == "pcgrad"
    assert event["optimizer_iteration"] == 2
    assert event["postproposal"]["accepted"] is True
    assert event["postproposal"]["update_index"] == 1
    assert event["postproposal"]["profile_hash"] == child["postproposal"].binding_hash
    fixtures.assert_lineage(runtime, runner, state, parent, None)
    config = json.loads((runtime / "configuration.json").read_text())
    lineage = config["checkpoint_continuation"]["method_policy_transition"]
    assert lineage == fixtures.clone(runner.design.optimizer_binding["method_policy_transition"])
    assert lineage == fixtures.clone(state.metadata["checkpoint_continuation"]["method_policy_transition"])
    assert all(lineage[key] == value for key, value in declaration.items())
    assert lineage["parent_checkpoint_hash"] == fixtures.stable_hash(parent.to_dict())
    assert lineage["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    assert lineage["method_rates"] == fixtures.clone(parent.method_state["rates"])
    assert lineage["saved_learning_rate"] == parent.optimizer_state["learning_rate"]
    assert lineage["previous_transition"] is None
    fixtures.assert_completed_recovery(runtime, child, result, monkeypatch)
    assert parent.to_dict() == parent_state
    assert fixtures.file_hashes(parent_root) == parent_hashes
