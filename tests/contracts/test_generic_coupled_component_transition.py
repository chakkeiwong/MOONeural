"""Explicit coupled refinement, inherited state and zero-call process recovery."""

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
from tests.contracts import test_generic_partial_continuation as partial
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_training_contracts import CheckpointState

POLICY = "postproposal_policy_transition"
SCHEDULE = "postproposal_schedule_transition"


def components(*arguments):
    raise AssertionError("manufactured lifecycle must not call component gradients")


def component_values(*arguments):
    raise AssertionError("manufactured lifecycle must not call component values")


def guard(count, *, coupled=False, **changes):
    options = {"coupled_component_guard": True, "component_loss_values": component_values,
        "component_loss_binding": {"criterion": "manufactured maximum components"}} if coupled else {}
    return partial.guard(count, component_rows=components,
        component_binding={"criterion": "manufactured maximum components"},
        component_direction="relative_public_progress", extra_descent_indices=(1,), **options, **changes)


@pytest.fixture(scope="module")
def relative_parent(tmp_path_factory):
    root = tmp_path_factory.mktemp("coupled-parent")
    arguments = checkpoints.fixture_arguments(root, 2)
    arguments.update(postproposal=guard(2), boundary_every=1)
    arguments["data_binding"].update(postproposal_source=finite._reference(partial.__file__),
        postproposal_component_source=finite._reference(__file__))
    arguments["source_evidence"].update(partial_fixture=finite._reference(partial.__file__),
        component_fixture=finite._reference(__file__))
    runtime = root / "runtime"
    _runner, result, _calls = refinements.run_host(runtime, arguments)
    parent = result.states[0]
    return parent, arguments, runtime


def child_arguments(fixture):
    parent, original, runtime = fixture
    arguments = {**original, "parameters": parent.policy_state["values"], "updates": 2,
        "training_schedule": checkpoints.schedule(4), "postproposal": guard(4, coupled=True),
        "stage_id": "coupled2-to4", "continuation_checkpoint": parent,
        "continuation_binding": partial.completed_binding(runtime, parent, runtime),
        "data_binding": copy.deepcopy(original["data_binding"]),
        POLICY: {"kind": "coupled_component_refinement", "effective_update": 2,
                 "reason": "Enable required-cell targets and finite checks"},
        SCHEDULE: {"effective_update": 2, "reason": "Append two unchanged schedule entries"}}
    arguments["data_binding"]["postproposal_component_loss_source"] = finite._reference(__file__)
    return arguments


def recover(filename):
    payload = json.loads(Path(filename).read_text())
    arguments = payload["arguments"]
    arguments.update(objective=checkpoints.forbidden, training_objective=checkpoints.forbidden,
        continuation_checkpoint=CheckpointState.from_dict(payload["parent"]), postproposal=guard(4, coupled=True))
    arguments["training_schedule"] = tuple(arguments["training_schedule"])
    runner, adapter, provider = finite.build_runner(payload["runtime"], **arguments)
    with pytest.MonkeyPatch.context() as patches:
        partial.forbid_runtime(runner, adapter, provider, patches)
        result = runner.run()
    assert result.states[0].to_dict() == payload["expected"]
    print("exact coupled recovery; zero numerical calls")


def test_coupled_transition_recovers_and_inherits_without_state_reset(relative_parent, tmp_path):
    parent, _original, parent_root = relative_parent
    parent_files = partial.hashes(parent_root)
    args = child_arguments(relative_parent)
    runtime = tmp_path / "coupled4"
    runner, _adapter, _provider = finite.build_runner(runtime, **args)
    rates.assert_retained(runner.states[0], parent, 1.)
    runner, result, calls = refinements.run_host(runtime, args)
    endpoint = result.states[0]
    assert calls["updates"] == [2, 3]
    configuration = refinements.assert_transition_copies(runtime, runner, endpoint)
    edge = configuration["checkpoint_continuation"][POLICY]
    assert edge["kind"] == "coupled_component_refinement"
    assert edge["component_callback"]["loss_only"]["source"] == finite._reference(__file__)
    assert endpoint.optimizer_state["first_moment"] == [value + 2. for value in parent.optimizer_state["first_moment"]]
    assert endpoint.optimizer_state["second_moment"] == [value + 4. for value in parent.optimizer_state["second_moment"]]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 4
    refinements.assert_recovery(runtime, args, result)
    plain = {key: value for key, value in args.items()
             if key not in ("postproposal", "continuation_checkpoint", "objective", "training_objective")}
    recovery = tmp_path / "recover.json"
    recovery.write_text(json.dumps({"arguments": plain, "parent": parent.to_dict(), "expected": endpoint.to_dict(),
                                   "runtime": str(runtime)}))
    before = partial.hashes(runtime)
    child = subprocess.run([sys.executable, "-c",
        "import sys,json; sys.path[:] = json.loads(sys.argv[2]); from tests.contracts.test_generic_coupled_component_transition import recover; recover(sys.argv[1])",
        str(recovery), json.dumps(sys.path)], env=os.environ.copy(), text=True, capture_output=True, timeout=40, check=False)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "zero numerical calls" in child.stdout and partial.hashes(runtime) == before
    inherited = {**args, "parameters": endpoint.policy_state["values"], "continuation_checkpoint": endpoint,
        "continuation_binding": partial.completed_binding(runtime, endpoint, runtime),
        "stage_id": "coupled4-to6", "training_schedule": checkpoints.schedule(6),
        "postproposal": guard(6, coupled=True), SCHEDULE: {"effective_update": 4, "reason": "Retain coupled mode"}}
    inherited.pop(POLICY)
    later_root = tmp_path / "coupled6"
    later, final, later_calls = refinements.run_host(later_root, inherited)
    assert later_calls["updates"] == [4, 5]
    assert refinements.assert_transition_copies(later_root, later, final.states[0])["checkpoint_continuation"][POLICY] == edge
    refinements.assert_recovery(later_root, inherited, final)
    current_parent_files = partial.hashes(parent_root)
    assert all(current_parent_files.get(path) == digest for path, digest in parent_files.items())


def test_coupled_transition_refuses_unrelated_changes(relative_parent, tmp_path):
    for fault in ("undeclared", "kind", "clock", "fraction", "margin", "rate", "schedule", "loss-source", "T"):
        args = child_arguments(relative_parent)
        if fault == "undeclared":
            args.pop(POLICY)
        elif fault == "kind":
            args[POLICY]["kind"] = "component_refinement"
        elif fault == "clock":
            args[POLICY]["effective_update"] = 1
        elif fault in ("fraction", "margin"):
            args["postproposal"] = guard(4, coupled=True,
                **({"fractions": (1., .2)} if fault == "fraction" else {"margin_fraction": .2}))
        elif fault == "schedule":
            args["training_schedule"] = ({**args["training_schedule"][0], "row_indices": [2]},
                                          *args["training_schedule"][1:])
        elif fault == "rate":
            args["learning_rate"] *= 2.
        elif fault == "T":
            args["threshold"] *= 2.
        else:
            args["data_binding"]["postproposal_component_loss_source"] = finite._reference(partial.__file__)
        with pytest.raises(ValueError):
            finite.build_runner(tmp_path / fault, **args)


def test_coupled_interruption_resumes_same_state(relative_parent, tmp_path):
    args = child_arguments(relative_parent)
    _runner, expected, _calls = refinements.run_host(tmp_path / "uninterrupted", args)
    runtime = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    with pytest.MonkeyPatch.context() as patches:
        checkpoints.host_runtime(runner, provider, patches, fail_at=3)
        with pytest.raises(RuntimeError, match="manufactured interruption"):
            runner.run()
    resumed, result, calls = refinements.run_host(runtime, args)
    assert calls["updates"] == [3]
    assert checkpoints.numerical(result.states[0]) == checkpoints.numerical(expected.states[0])
    configuration = refinements.assert_transition_copies(runtime, resumed, result.states[0])
    assert configuration["checkpoint_continuation"][POLICY]["kind"] == "coupled_component_refinement"
