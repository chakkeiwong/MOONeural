"""Explicit extra-descent enablement retains clocks, rate history and recovery."""

import copy
import json

import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_complete_cell_policy_transition as complete
from tests.contracts import test_generic_component_policy_refinement as refinements
from tests.contracts import test_generic_component_rate_continuation as rates
from tests.contracts import test_generic_partial_continuation as partial
from tests.contracts import test_generic_postproposal_policy_transition as policies
from mooneural.training import generic_finite_objective_runner as finite

completed_parent = complete.completed_parent
k3_parent40 = complete.k3_parent40
rate_parent50 = complete.rate_parent50
trigger_parent60 = complete.trigger_parent60
POLICY, SCHEDULE, RATE = complete.POLICY, complete.SCHEDULE, complete.RATE


def component_rows(parameters, batch, update_index, context):
    raise AssertionError("host transition or completed recovery invoked component numerics")


def guard(count=70, **changes):
    return policies.enabled_guard(count, provider=component_rows,
        provider_binding=refinements.component_profile(9), finite_component_fallback=True,
        extra_descent_indices=(1,), **changes)


def child_arguments(fixture):
    args = complete.child_arguments(fixture)
    args["postproposal"] = guard()
    args["stage_id"] = "extra-descent60-to70"
    args[POLICY] = {"kind": "protected_descent_refinement", "effective_update": 60,
                    "reason": "Enable explicit extra descent preference without membership changes"}
    args["data_binding"]["postproposal_component_source"] = finite._reference(__file__)
    args["source_evidence"]["extra_descent_fixture"] = finite._reference(__file__)
    return args


def test_explicit_preference_retains_state_recovers_and_later_inherits(trigger_parent60, tmp_path, monkeypatch):
    parent, _original, parent_root, _binding = trigger_parent60
    parent_files, parent_state = partial.hashes(parent_root), parent.to_dict()
    args = child_arguments(trigger_parent60)
    runtime = tmp_path / "extra70"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    rates.assert_retained(runner.states[0], parent, 1.)
    configuration = rates.assert_lineage(runtime, runner, runner.states[0])
    old = json.loads((parent_root / "configuration.json").read_text())
    refined = configuration["checkpoint_continuation"][POLICY]
    assert refined["kind"] == "protected_descent_refinement"
    assert refined["previous_transition"] == old["checkpoint_continuation"][POLICY]
    assert refined["old_maximum_loss_calls"] == 2 * refined["new_maximum_loss_calls"] - 1
    assert refined["component_fallback"]["extra_descent_indices"] == [1]
    assert configuration["checkpoint_continuation"][RATE] == old["checkpoint_continuation"][RATE]
    assert configuration["training"]["schedule"][:60] == old["training"]["schedule"]
    calls = checkpoints.host_runtime(runner, provider, monkeypatch)
    result = runner.run()
    endpoint = result.states[0]
    assert calls["updates"] == list(range(60, 70)) and calls["controls"] == [60, 70]
    assert endpoint.optimizer_state["learning_rate"] == parent.optimizer_state["learning_rate"]
    assert endpoint.method_state["rates"] == parent.method_state["rates"]
    assert endpoint.optimizer_state["first_moment"] == [value + 10. for value in parent.optimizer_state["first_moment"]]
    assert endpoint.optimizer_state["second_moment"] == [value + 20. for value in parent.optimizer_state["second_moment"]]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 70
    refinements.assert_recovery(runtime, args, result)
    rates.assert_lineage(runtime, runner, endpoint)
    inherited = {**args, "parameters": endpoint.policy_state["values"],
        "continuation_checkpoint": endpoint, "continuation_binding": partial.completed_binding(runtime, endpoint, runtime),
        "stage_id": "extra-descent70-to80", "training_schedule": checkpoints.schedule(80),
        "postproposal": guard(80), SCHEDULE: {"effective_update": 70, "reason": "Retain extra preference"}}
    inherited.pop(POLICY)
    later_root = tmp_path / "extra80"
    later, final, later_calls = refinements.run_host(later_root, inherited)
    assert later_calls["updates"] == list(range(70, 80))
    later_config = rates.assert_lineage(later_root, later, final.states[0])
    assert later_config["checkpoint_continuation"][POLICY] == refined
    assert later_config["checkpoint_continuation"][RATE] == old["checkpoint_continuation"][RATE]
    refinements.assert_recovery(later_root, inherited, final)
    assert parent.to_dict() == parent_state and partial.hashes(parent_root) == parent_files


def test_interruption_preserves_enabled_preference_and_all_lineages(trigger_parent60, tmp_path, monkeypatch):
    args = child_arguments(trigger_parent60)
    _runner, expected, _calls = refinements.run_host(tmp_path / "uninterrupted", args)
    runtime = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    calls = checkpoints.host_runtime(runner, provider, monkeypatch, fail_at=65)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    assert calls["updates"] == list(range(60, 65))
    resumed, _adapter, provider = finite.build_runner(runtime, **args)
    calls = checkpoints.host_runtime(resumed, provider, monkeypatch)
    result = resumed.run()
    assert calls["updates"] == list(range(65, 70))
    assert checkpoints.numerical(result.states[0]) == checkpoints.numerical(expected.states[0])
    rates.assert_lineage(runtime, resumed, result.states[0])
    refinements.assert_recovery(runtime, args, result)


def test_preference_cannot_be_enabled_silently_or_hide_other_numerical_changes(trigger_parent60, tmp_path):
    for fault in ("undeclared", "wrong-kind", "wrong-clock", "fraction", "margin", "rate"):
        args = child_arguments(trigger_parent60)
        if fault == "undeclared":
            args.pop(POLICY)
        elif fault == "wrong-kind":
            args[POLICY]["kind"] = "component_refinement"
        elif fault == "wrong-clock":
            args[POLICY]["effective_update"] = 59
        elif fault == "fraction":
            args["postproposal"] = guard(fractions=(1., .2))
        elif fault == "margin":
            args["postproposal"] = guard(margin_fraction=.2)
        else:
            args["learning_rate"] *= 2.
        with pytest.raises(ValueError):
            finite.build_runner(tmp_path / fault, **args)


def test_archived_enabled_profile_reconstruction_retains_preference():
    enabled = guard()
    binding = {"profile": json.loads(json.dumps(enabled.binding))}
    archived = finite._archived_postproposal(enabled, copy.deepcopy(binding))
    assert archived.binding == enabled.binding
    assert archived.extra_descent_indices == (1,)
