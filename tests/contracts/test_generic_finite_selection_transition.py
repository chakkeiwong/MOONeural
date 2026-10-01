"""Explicit selector transition, shared execution and retained-state recovery."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_component_policy_refinement as refinements
from tests.contracts import test_generic_component_rate_continuation as rates
from tests.contracts import test_generic_coupled_component_transition as coupled
from tests.contracts import test_generic_finite_objective_runner as objectives
from tests.contracts import test_generic_partial_continuation as partial
from tests.contracts import test_generic_postproposal as postproposal
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_permanent_pass_executor import PermanentPassExecutor
from mooneural.training.generic_postproposal import GuardedPostproposal
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    canonical_json,
    stable_hash,
)

POLICY = "postproposal_policy_transition"
SCHEDULE = "postproposal_schedule_transition"


def guard(count, lookahead=0, **options):
    return coupled.guard(count, lookahead_steps=lookahead, **options)


@pytest.fixture(scope="module")
def parent_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("finite-selection-parent")
    arguments = checkpoints.fixture_arguments(root, 2)
    arguments.update(postproposal=guard(2), boundary_every=1)
    arguments["data_binding"].update(postproposal_source=finite._reference(partial.__file__),
        postproposal_component_source=finite._reference(coupled.__file__))
    arguments["source_evidence"].update(partial_fixture=finite._reference(partial.__file__),
        component_fixture=finite._reference(coupled.__file__))
    runtime = root / "runtime"
    _, result, _ = refinements.run_host(runtime, arguments)
    return result.states[0], arguments, runtime


def child_arguments(parent_run):
    parent, arguments, runtime = parent_run
    return {**arguments, "parameters": parent.policy_state["values"], "updates": 2,
        "training_schedule": checkpoints.schedule(4), "postproposal": guard(4, 2),
        "stage_id": "selection-child", "continuation_checkpoint": parent,
        "continuation_binding": partial.completed_binding(runtime, parent, runtime),
        POLICY: {"kind": "finite_selection_refinement", "effective_update": 2,
            "reason": "Enable bounded finite selection"},
        SCHEDULE: {"effective_update": 2, "reason": "Append two original schedule entries"}}


def recover(filename):
    with open(filename) as stream:
        payload = json.load(stream)
    arguments = payload["arguments"]
    arguments.update(continuation_checkpoint=CheckpointState.from_dict(payload["parent"]),
        postproposal=guard(4, 2), objective=objectives.forbidden, training_objective=objectives.forbidden)
    arguments["training_schedule"] = tuple(arguments["training_schedule"])
    runner, adapter, provider = finite.build_runner(payload["runtime"], **arguments)
    with pytest.MonkeyPatch.context() as patches:
        partial.forbid_runtime(runner, adapter, provider, patches)
        result = runner.run()
    assert result.states[0].to_dict() == payload["expected"]
    print("zero numerical calls")


def test_retained_selection_schedule_later_schedule_and_fresh_recovery(parent_run, tmp_path):
    parent, _, old_runtime = parent_run
    args = child_arguments(parent_run)
    old_files = partial.hashes(old_runtime)
    runtime = tmp_path / "child"
    runner, _, _ = finite.build_runner(runtime, **args)
    rates.assert_retained(runner.states[0], parent, 1.)
    runner, result, calls = refinements.run_host(runtime, args)
    assert calls["updates"] == [2, 3]
    endpoint = result.states[0]
    configuration = refinements.assert_transition_copies(runtime, runner, endpoint)
    edge = configuration["checkpoint_continuation"][POLICY]
    assert edge["kind"] == "finite_selection_refinement"
    assert edge["new_binding"]["profile"]["finite_selection"]["lookahead_steps"] == 2
    assert edge["old_schedule_hash"] == edge["new_schedule_hash"]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 4
    refinements.assert_recovery(runtime, args, result)
    plain = {key: value for key, value in args.items()
        if key not in ("postproposal", "continuation_checkpoint", "objective", "training_objective")}
    recovery = tmp_path / "recover.json"
    recovery.write_text(json.dumps({"arguments": plain, "parent": parent.to_dict(),
        "expected": endpoint.to_dict(), "runtime": str(runtime)}))
    process = subprocess.run([sys.executable, "-c",
        "import sys,json; sys.path[:]=json.loads(sys.argv[2]); from tests.contracts.test_generic_finite_selection_transition import recover; recover(sys.argv[1])",
        str(recovery), json.dumps(sys.path)], env=os.environ.copy(), text=True, capture_output=True, timeout=35, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "zero numerical calls" in process.stdout
    later = {**args, "parameters": endpoint.policy_state["values"], "continuation_checkpoint": endpoint,
        "continuation_binding": partial.completed_binding(runtime, endpoint, runtime), "stage_id": "selection-later",
        "training_schedule": checkpoints.schedule(6), "postproposal": guard(6, 2),
        SCHEDULE: {"effective_update": 4, "reason": "Continue unchanged selector"}}
    later.pop(POLICY)
    later_runner, later_result, _ = refinements.run_host(tmp_path / "later", later)
    refinements.assert_transition_copies(tmp_path / "later", later_runner, later_result.states[0])
    current_files = partial.hashes(old_runtime)
    assert all(current_files.get(path) == digest for path, digest in old_files.items())


@pytest.mark.parametrize("fault", ("undeclared", "clock", "kind", "fractions", "margin", "rate", "T", "source"))
def test_selection_transition_refuses_unrelated_changes(parent_run, tmp_path, fault):
    args = child_arguments(parent_run)
    if fault == "undeclared":
        args.pop(POLICY)
    elif fault == "clock":
        args[POLICY]["effective_update"] = 1
    elif fault == "kind":
        args[POLICY]["kind"] = "component_refinement"
    elif fault in ("fractions", "margin"):
        args["postproposal"] = guard(4, 2, **({"fractions": (1., .25)} if fault == "fractions" else {"margin_fraction": .2}))
    elif fault == "rate":
        args["learning_rate"] *= 2.
    elif fault == "T":
        args["threshold"] *= 2.
    else:
        args["data_binding"] = {**args["data_binding"], "postproposal_component_source": finite._reference(partial.__file__)}
    with pytest.raises(ValueError):
        finite.build_runner(tmp_path / fault, **args)


def test_typed_valid_selection_lineage_corruption_refused(parent_run, tmp_path):
    args = child_arguments(parent_run)
    runner, result, _ = refinements.run_host(tmp_path / "child", args)
    configuration = refinements.assert_transition_copies(tmp_path / "child", runner, result.states[0])
    for field, replacement in (("minimum_gain", 0.), ("lookahead_steps", 3), ("criterion", "sum")):
        state = result.states[0].to_dict()
        state.pop("checkpoint_state_hash")
        changed = copy.deepcopy(configuration)
        edge = changed["checkpoint_continuation"][POLICY]
        edge["new_binding"]["profile"]["finite_selection"][field] = replacement
        edge["new_postproposal_hash"] = stable_hash(edge["new_binding"])
        changed["postproposal"] = copy.deepcopy(edge["new_binding"])
        state["metadata"]["checkpoint_continuation"][POLICY] = copy.deepcopy(edge)
        state["metadata"]["replication_design"]["optimizer_binding"][POLICY] = copy.deepcopy(edge)
        state["metadata"]["replication_design"]["optimizer_binding"]["postproposal"] = copy.deepcopy(changed["postproposal"])
        with pytest.raises(ValueError):
            finite._postproposal_lineage(CheckpointState(**{key: value for key, value in state.items() if key != "schema"}), changed)


def empty_components(parameters, batch, update, context):
    return {"owner_indices": np.empty(0, np.int32), "cell_ids": [], "raw_values": np.empty(0),
        "normalized_values": np.empty(0), "normalized_gradients": np.empty((0, len(parameters))), "context": context}


def test_actual_shared_executor_uses_optional_guard_and_resumes():
    baseline, initial, batch = postproposal.fixture(task_count=2)
    enabled = GuardedPostproposal(postproposal.quadratic_batch_losses, postproposal.batch_identity,
        [1., 1.], [.04, .04], binding={"fixture": "actual-shared-executor"}, component_rows=empty_components,
        component_binding={"fixture": "empty"}, component_direction="relative_public_progress", lookahead_steps=2)
    executor = PermanentPassExecutor(baseline.adapter, baseline.spec, postproposal.roles(), threshold=.04, postproposal=enabled)
    parent = executor.initialize(initial, postproposal.fake_control(initial, baseline.adapter.registry.task_ids, [.2, .2]))
    state, event = executor.step(parent, batch)
    assert event["postproposal"]["finite_selection"]["lookahead_steps"] == 2
    assert event["postproposal"]["loss_calls"] <= enabled.binding["maximum_loss_calls"]
    restored = CheckpointState.from_dict(json.loads(canonical_json(state.to_dict())))
    resumed = PermanentPassExecutor(baseline.adapter, baseline.spec, postproposal.roles(), threshold=.04, postproposal=enabled)
    assert resumed.step(restored, batch)[0].to_dict() == executor.step(state, batch)[0].to_dict()


def test_selection_profile_alone_supports_absent_training(parent_run):
    parent, _args, runtime = parent_run
    old = json.loads((runtime / "configuration.json").read_text())["postproposal"]
    new = copy.deepcopy(old)
    new["profile"]["finite_selection"] = guard(2, 2).binding["finite_selection"]
    declaration = {"kind": "finite_selection_refinement", "effective_update": parent.update_index, "reason": "Unscheduled"}
    record = finite._postproposal_policy_edge(parent, old, None, new, None, declaration, None, None)
    assert record["old_schedule_hash"] == stable_hash(None)
    finite._check_selection_record(record)


def test_selection_then_search_suffix_preserves_history(parent_run, tmp_path):
    from tests.contracts import test_generic_search_profile_transition as search
    args = child_arguments(parent_run)
    runtime = tmp_path / "selected"
    runner, result, _ = refinements.run_host(runtime, args)
    parent = result.states[0]
    old_config = refinements.assert_transition_copies(runtime, runner, parent)
    old_policy = old_config["checkpoint_continuation"][POLICY]
    binding = copy.deepcopy(old_config["postproposal"])
    binding["profile"]["fractions"].append(.0625)
    binding["profile"]["maximum_loss_calls"] += 1
    edge = finite._postproposal_search_profile_transition(parent, old_config, binding,
        {"effective_update": 4, "reason": "Declared smaller suffix after selector"})
    state = parent.to_dict()
    state.pop("checkpoint_state_hash")
    config = copy.deepcopy(old_config)
    config["postproposal"] = binding
    config["checkpoint_continuation"][search.NAME] = edge
    for target in (state["metadata"]["checkpoint_continuation"], state["metadata"]["replication_design"]["optimizer_binding"]):
        target[search.NAME] = copy.deepcopy(edge)
    state["metadata"]["replication_design"]["optimizer_binding"]["postproposal"] = binding
    bound = CheckpointState(**{key: value for key, value in state.items() if key != "schema"})
    assert finite._postproposal_lineage(bound, config)[1] == old_policy


def test_partial_selection_keeps_unconsumed_schedule(tmp_path):
    from tests.contracts import test_generic_search_profile_transition as search
    args = checkpoints.fixture_arguments(tmp_path, 4)
    args.update(postproposal=guard(4), boundary_every=1)
    args["data_binding"].update(postproposal_source=finite._reference(partial.__file__),
        postproposal_component_source=finite._reference(coupled.__file__))
    args["source_evidence"].update(partial_fixture=finite._reference(partial.__file__),
        component_fixture=finite._reference(coupled.__file__))
    runtime = tmp_path / "parent"
    runner, _, provider = finite.build_runner(runtime, **args)
    with pytest.MonkeyPatch.context() as patches:
        search.host_runtime(runner, provider, patches, reject_at=2)
        with pytest.raises(search.PermanentPassUpdateError) as rejected:
            runner.run()
    parent = rejected.value.checkpoint
    child = {**args, "parameters": parent.policy_state["values"], "updates": 2,
        "continuation_checkpoint": parent, "continuation_binding": search.saved_binding(runtime, parent),
        "partial_parent_continuation": partial.declaration(2), "postproposal": guard(4, 2),
        "stage_id": "partial-selection", POLICY: {"kind": "finite_selection_refinement",
            "effective_update": 2, "reason": "Change selector only on unconsumed schedule"}}
    child_runner, _, _ = finite.build_runner(tmp_path / "child", **child)
    rates.assert_retained(child_runner.states[0], parent, 1.)
    _, result, calls = refinements.run_host(tmp_path / "child", child)
    assert result.states[0].update_index == 4 and calls["updates"] == [2, 3]
