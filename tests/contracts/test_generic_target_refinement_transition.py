"""Weighted-policy continuation preserves source state and fresh recovery."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_component_policy_refinement as refinements
from tests.contracts import test_generic_component_rate_continuation as rates
from tests.contracts import test_generic_coupled_component_transition as coupled
from tests.contracts import test_generic_partial_continuation as partial
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_training_contracts import CheckpointState

POLICY = coupled.POLICY
SCHEDULE = coupled.SCHEDULE


@pytest.fixture(scope="module")
def coupled_parent(tmp_path_factory):
    relative = coupled.relative_parent.__wrapped__(tmp_path_factory)
    arguments = coupled.child_arguments(relative)
    runtime = tmp_path_factory.mktemp("target-parent") / "runtime"
    _runner, result, _calls = refinements.run_host(runtime, arguments)
    return result.states[0], arguments, runtime


def guard(count, **options):
    return coupled.guard(count, coupled=True, finite_target_refinement=True, **options)


def child_arguments(fixture):
    parent, original, runtime = fixture
    return {**original, "parameters": parent.policy_state["values"], "updates": 2,
        "training_schedule": checkpoints.schedule(6), "postproposal": guard(6),
        "stage_id": "target4-to6", "continuation_checkpoint": parent,
        "continuation_binding": partial.completed_binding(runtime, parent, runtime),
        "data_binding": copy.deepcopy(original["data_binding"]),
        POLICY: {"kind": "target_refinement", "effective_update": 4,
                 "reason": "Enable same-parent finite target redistribution"},
        SCHEDULE: {"effective_update": 4, "reason": "Append two unchanged schedule entries"}}


def recover(filename):
    payload = json.loads(Path(filename).read_text())
    arguments = payload["arguments"]
    arguments.update(objective=checkpoints.forbidden, training_objective=checkpoints.forbidden,
        continuation_checkpoint=CheckpointState.from_dict(payload["parent"]), postproposal=guard(6))
    arguments["training_schedule"] = tuple(arguments["training_schedule"])
    runner, adapter, provider = finite.build_runner(payload["runtime"], **arguments)
    with pytest.MonkeyPatch.context() as patches:
        partial.forbid_runtime(runner, adapter, provider, patches)
        result = runner.run()
    assert result.states[0].to_dict() == payload["expected"]
    print("exact target-refinement recovery; zero numerical calls")


def test_transition_inherits_slots_history_caps_and_zero_call_fresh_recovery(coupled_parent, tmp_path):
    parent, _original, parent_root = coupled_parent
    original_files = partial.hashes(parent_root)
    args = child_arguments(coupled_parent)
    runtime = tmp_path / "refined6"
    runner, _adapter, _provider = finite.build_runner(runtime, **args)
    rates.assert_retained(runner.states[0], parent, 1.)
    runner, result, calls = refinements.run_host(runtime, args)
    endpoint = result.states[0]
    assert calls["updates"] == [4, 5]
    configuration = refinements.assert_transition_copies(runtime, runner, endpoint)
    edge = configuration["checkpoint_continuation"][POLICY]
    assert edge["kind"] == "target_refinement"
    assert edge["new_maximum_loss_calls"] == 2 * edge["old_maximum_loss_calls"] - 1
    assert edge["component_callback"] == edge["old_component_callback"]
    assert edge["component_fallback"]["maximum_direction_solves"] == 2
    assert endpoint.optimizer_state["first_moment"] == [value + 2. for value in parent.optimizer_state["first_moment"]]
    assert endpoint.optimizer_state["second_moment"] == [value + 4. for value in parent.optimizer_state["second_moment"]]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 6
    refinements.assert_recovery(runtime, args, result)
    plain = {key: value for key, value in args.items()
             if key not in ("postproposal", "continuation_checkpoint", "objective", "training_objective")}
    recovery = tmp_path / "recover.json"
    recovery.write_text(json.dumps({"arguments": plain, "parent": parent.to_dict(), "expected": endpoint.to_dict(),
        "runtime": str(runtime)}))
    before = partial.hashes(runtime)
    child = subprocess.run([sys.executable, "-c",
        "import sys,json; sys.path[:] = json.loads(sys.argv[2]); from tests.contracts.test_generic_target_refinement_transition import recover; recover(sys.argv[1])",
        str(recovery), json.dumps(sys.path)], env=os.environ.copy(), text=True, capture_output=True, timeout=45, check=False)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "zero numerical calls" in child.stdout and partial.hashes(runtime) == before
    inherited = {**args, "parameters": endpoint.policy_state["values"], "continuation_checkpoint": endpoint,
        "continuation_binding": partial.completed_binding(runtime, endpoint, runtime),
        "stage_id": "target6-to8", "training_schedule": checkpoints.schedule(8),
        "postproposal": guard(8), SCHEDULE: {"effective_update": 6, "reason": "Retain target refinement"}}
    inherited.pop(POLICY)
    later_root = tmp_path / "refined8"
    later, final, later_calls = refinements.run_host(later_root, inherited)
    assert later_calls["updates"] == [6, 7]
    assert refinements.assert_transition_copies(later_root, later, final.states[0])["checkpoint_continuation"][POLICY] == edge
    refinements.assert_recovery(later_root, inherited, final)
    current_files = partial.hashes(parent_root)
    assert all(current_files.get(path) == digest for path, digest in original_files.items())


@pytest.mark.parametrize("fault", ["undeclared", "kind", "clock", "fraction", "margin", "rate", "schedule", "callback", "T", "D"])
def test_target_transition_refuses_unrelated_changes(coupled_parent, tmp_path, fault):
    args = child_arguments(coupled_parent)
    if fault == "undeclared":
        args.pop(POLICY)
    elif fault == "kind":
        args[POLICY]["kind"] = "component_refinement"
    elif fault == "clock":
        args[POLICY]["effective_update"] = 3
    elif fault in ("fraction", "margin"):
        args["postproposal"] = guard(6, **({"fractions": (1., .2)} if fault == "fraction" else {"margin_fraction": .2}))
    elif fault == "schedule":
        args["training_schedule"] = ({**args["training_schedule"][0], "row_indices": [2]}, *args["training_schedule"][1:])
    elif fault == "rate":
        args["learning_rate"] *= 2.
    elif fault == "T":
        args["threshold"] *= 2.
    elif fault == "D":
        args["denominators"] = [2. * value for value in args["denominators"]]
    else:
        args["data_binding"]["postproposal_component_loss_source"] = finite._reference(partial.__file__)
    with pytest.raises(ValueError):
        finite.build_runner(tmp_path / fault, **args)


def test_target_interruption_resumes_same_committed_numerical_state(coupled_parent, tmp_path):
    args = child_arguments(coupled_parent)
    _runner, expected, _calls = refinements.run_host(tmp_path / "uninterrupted", args)
    runtime = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    with pytest.MonkeyPatch.context() as patches:
        checkpoints.host_runtime(runner, provider, patches, fail_at=5)
        with pytest.raises(RuntimeError, match="manufactured interruption"):
            runner.run()
    resumed, result, calls = refinements.run_host(runtime, args)
    assert calls["updates"] == [5]
    assert checkpoints.numerical(result.states[0]) == checkpoints.numerical(expected.states[0])
    assert refinements.assert_transition_copies(runtime, resumed, result.states[0])["checkpoint_continuation"][POLICY]["kind"] == "target_refinement"
