"""Optional direction transition preserves state, lineage and process recovery."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_complete_cell_policy_transition as complete
from tests.contracts import test_generic_component_policy_refinement as refinements
from tests.contracts import test_generic_component_rate_continuation as rates
from tests.contracts import test_generic_partial_continuation as partial
from tests.contracts import test_generic_protected_descent_transition as protected
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_training_contracts import CheckpointState

completed_parent = complete.completed_parent
k3_parent40 = complete.k3_parent40
rate_parent50 = complete.rate_parent50
trigger_parent60 = complete.trigger_parent60
POLICY, SCHEDULE, RATE = complete.POLICY, complete.SCHEDULE, complete.RATE


@pytest.fixture(scope="module", params=[False, True], ids=["active-only", "extra-descent"])
def equality_parent70(trigger_parent60, tmp_path_factory, request):
    runtime = tmp_path_factory.mktemp("relative-parent70") / "runtime"
    args = (protected if request.param else complete).child_arguments(trigger_parent60)
    _runner, result, _calls = refinements.run_host(runtime, args)
    state = result.states[0]
    return state, args, runtime, partial.completed_binding(runtime, state, runtime)


def guard(count=80, *, extras=True, **changes):
    if extras:
        return protected.guard(count, component_direction="relative_public_progress", **changes)
    return complete.policies.enabled_guard(count, provider=complete.component_rows,
        provider_binding=refinements.component_profile(9), finite_component_fallback=True,
        component_direction="relative_public_progress", **changes)


def child_arguments(fixture):
    parent, original, _runtime, binding = fixture
    return {**original, "parameters": parent.policy_state["values"], "updates": 10, "boundary_every": 10,
        "training_schedule": checkpoints.schedule(80),
        "postproposal": guard(extras=bool(original["postproposal"].extra_descent_indices)),
        "stage_id": "relative-progress70-to80", "continuation_checkpoint": parent,
        "continuation_binding": copy.deepcopy(binding), "data_binding": copy.deepcopy(original["data_binding"]),
        "source_evidence": copy.deepcopy(original["source_evidence"]),
        POLICY: {"kind": "relative_progress_refinement", "effective_update": 70,
                 "reason": "Change only the declared public-progress direction"},
        SCHEDULE: {"effective_update": 70, "reason": "Append the next ten unchanged schedule entries"}}


def recover_from_file(filename):
    payload = json.loads(Path(filename).read_text())
    args = payload["arguments"]
    args.update(objective=checkpoints.forbidden, training_objective=checkpoints.forbidden,
        continuation_checkpoint=CheckpointState.from_dict(payload["parent"]),
        postproposal=guard(extras=payload["extras"]))
    args["training_schedule"] = tuple(args["training_schedule"])
    runner, adapter, provider = finite.build_runner(payload["runtime"], **args)
    with pytest.MonkeyPatch.context() as patches:
        partial.forbid_runtime(runner, adapter, provider, patches)
        result = runner.run()
    assert result.states[0].to_dict() == payload["expected"]
    print("fresh-process completed recovery: exact state, zero numerical calls")


def test_refinement_preserves_history_recovers_and_later_inherits(equality_parent70, tmp_path):
    parent, original, parent_root, _binding = equality_parent70
    parent_files, parent_state = partial.hashes(parent_root), parent.to_dict()
    args = child_arguments(equality_parent70)
    runtime = tmp_path / "relative80"
    runner, _adapter, _provider = finite.build_runner(runtime, **args)
    rates.assert_retained(runner.states[0], parent, 1.)
    runner, result, calls = refinements.run_host(runtime, args)
    endpoint = result.states[0]
    assert calls["updates"] == list(range(70, 80)) and calls["controls"] == [70, 80]
    configuration = rates.assert_lineage(runtime, runner, endpoint)
    old = json.loads((parent_root / "configuration.json").read_text())
    refined = configuration["checkpoint_continuation"][POLICY]
    assert refined["kind"] == "relative_progress_refinement"
    assert refined["previous_transition"] == old["checkpoint_continuation"][POLICY]
    assert refined["old_component_fallback"] == old["postproposal"]["profile"]["component_fallback"]
    assert refined["component_fallback"]["provider"] == refined["old_component_fallback"]["provider"]
    assert refined["component_fallback"]["component_direction"] == "relative_public_progress"
    assert configuration["checkpoint_continuation"][RATE] == old["checkpoint_continuation"][RATE]
    assert configuration["training"]["schedule"][:70] == old["training"]["schedule"]
    assert endpoint.optimizer_state["first_moment"] == [value + 10. for value in parent.optimizer_state["first_moment"]]
    assert endpoint.optimizer_state["second_moment"] == [value + 20. for value in parent.optimizer_state["second_moment"]]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 80
    refinements.assert_recovery(runtime, args, result)
    plain = {key: value for key, value in args.items()
             if key not in ("postproposal", "continuation_checkpoint", "objective", "training_objective")}
    recovery = tmp_path / "recover.json"
    recovery.write_text(json.dumps({"arguments": plain, "parent": parent.to_dict(), "expected": endpoint.to_dict(),
        "runtime": str(runtime), "extras": bool(original["postproposal"].extra_descent_indices)}))
    before = partial.hashes(runtime)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(Path(finite.__file__).parents[2]), str(Path(__file__).parent)))}
    recovered = subprocess.run([sys.executable, "-c",
        "import sys,json; sys.path[:] = json.loads(sys.argv[2]); from tests.contracts.test_generic_relative_progress_transition import recover_from_file; recover_from_file(sys.argv[1])",
        str(recovery), json.dumps(sys.path)], env=environment, text=True, capture_output=True, timeout=60, check=False)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert "zero numerical calls" in recovered.stdout and partial.hashes(runtime) == before
    inherited = {**args, "parameters": endpoint.policy_state["values"], "continuation_checkpoint": endpoint,
        "continuation_binding": partial.completed_binding(runtime, endpoint, runtime),
        "stage_id": "relative-progress80-to90", "training_schedule": checkpoints.schedule(90),
        "postproposal": guard(90, extras=bool(original["postproposal"].extra_descent_indices)),
        SCHEDULE: {"effective_update": 80, "reason": "Retain the relative direction"}}
    inherited.pop(POLICY)
    later_root = tmp_path / "relative90"
    later, final, later_calls = refinements.run_host(later_root, inherited)
    assert later_calls["updates"] == list(range(80, 90))
    assert rates.assert_lineage(later_root, later, final.states[0])["checkpoint_continuation"][POLICY] == refined
    refinements.assert_recovery(later_root, inherited, final)
    assert parent.to_dict() == parent_state and partial.hashes(parent_root) == parent_files
    inherited[POLICY] = {**args[POLICY], "effective_update": 80}
    with pytest.raises(ValueError, match="original equality"):
        finite.build_runner(tmp_path / "repeated", **inherited)


def test_interruption_recovers_the_same_relative_state(equality_parent70, tmp_path, monkeypatch):
    args = child_arguments(equality_parent70)
    _runner, expected, _calls = refinements.run_host(tmp_path / "uninterrupted", args)
    runtime = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    calls = checkpoints.host_runtime(runner, provider, monkeypatch, fail_at=75)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    assert calls["updates"] == list(range(70, 75))
    resumed, _adapter, provider = finite.build_runner(runtime, **args)
    calls = checkpoints.host_runtime(resumed, provider, monkeypatch)
    result = resumed.run()
    assert calls["updates"] == list(range(75, 80))
    assert checkpoints.numerical(result.states[0]) == checkpoints.numerical(expected.states[0])
    rates.assert_lineage(runtime, resumed, result.states[0])
    refinements.assert_recovery(runtime, args, result)


def test_refinement_cannot_hide_unrelated_changes(equality_parent70, tmp_path):
    extras = bool(equality_parent70[1]["postproposal"].extra_descent_indices)
    for fault in ("undeclared", "wrong-kind", "wrong-clock", "fraction", "margin", "rate", "schedule"):
        args = child_arguments(equality_parent70)
        if fault == "undeclared":
            args.pop(POLICY)
        elif fault == "wrong-kind":
            args[POLICY]["kind"] = "component_refinement"
        elif fault == "wrong-clock":
            args[POLICY]["effective_update"] = 69
        elif fault in ("fraction", "margin"):
            args["postproposal"] = guard(extras=extras, **({"fractions": (1., .2)} if fault == "fraction" else {"margin_fraction": .2}))
        elif fault == "schedule":
            args["training_schedule"] = ({"update_index": 0, "row_indices": [2], "population_rows": 3},
                                          *args["training_schedule"][1:])
        else:
            args["learning_rate"] *= 2.
        with pytest.raises(ValueError):
            finite.build_runner(tmp_path / fault, **args)


def test_refinement_only_permits_identical_provider_source_relocation(equality_parent70, tmp_path):
    args = child_arguments(equality_parent70)
    runner, result, _calls = refinements.run_host(tmp_path / "relative", args)
    receipt = rates.assert_lineage(tmp_path / "relative", runner, result.states[0])["checkpoint_continuation"][POLICY]
    old, callback = receipt["old_component_fallback"], receipt["old_component_callback"]
    new, next_callback = copy.deepcopy(receipt["component_fallback"]), copy.deepcopy(receipt["component_callback"])
    next_callback["source"]["path"] = "/relocated/identical-provider.py"
    finite._check_component_refinement(old, callback, new, next_callback, relative_progress_refinement=True)
    for fault in ("provider", "extra", "solver", "source-code"):
        candidate, bound_callback = copy.deepcopy(new), copy.deepcopy(next_callback)
        if fault == "provider":
            candidate["provider"]["top_k"] = 1
        elif fault == "extra":
            candidate["extra_descent_indices"] = [5]
        elif fault == "solver":
            candidate["max_iterations"] = 1000
        else:
            bound_callback["source"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="only change"):
            finite._check_component_refinement(old, callback, candidate, bound_callback, relative_progress_refinement=True)


@pytest.mark.parametrize("extras", [False, True])
def test_archived_profile_reconstructs_the_actual_direction(extras):
    enabled = guard(extras=extras)
    archived = finite._archived_postproposal(enabled, {"profile": copy.deepcopy(enabled.binding)})
    assert archived.binding == enabled.binding
    assert archived.component_direction == "relative_public_progress"
    assert archived.component_descent.experimental_get_tracing_count() == 0
