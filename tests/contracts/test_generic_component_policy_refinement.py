"""Retained policy metadata and shared lifecycle with manufactured numerical leaves.

The parent runs qualification. Recovery reconstructs runners in this process;
these fixtures do not claim fresh-process or economic-model recovery.
"""

import copy
import json
from dataclasses import replace

import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_partial_continuation as partial
from tests.contracts import test_generic_postproposal_continuation as schedules
from tests.contracts import test_generic_postproposal_policy_transition as policies
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_training_contracts import (
    PolicyView,
    canonical_json,
    stable_hash,
)

POLICY, SCHEDULE = policies.POLICY, policies.SCHEDULE
COMPONENT_CALLS = []


def component_rows(parameters, batch, update_index, context):
    COMPONENT_CALLS.append(update_index)
    raise AssertionError("refinement construction or recovery called component numerics")


def component_profile(top_k):
    return {"criterion": "manufactured maximum components", "coordinate_mode": "raw_over_D",
            "top_k": top_k, "max_component_rows": 3 * top_k}


def guard(count=40, *, top_k=3, old=False, **changes):
    return policies.enabled_guard(count, provider=policies.component_rows if old else component_rows,
        provider_binding=component_profile(top_k), **changes)


def declaration(clock=30):
    return {"kind": "component_refinement", "effective_update": clock,
            "reason": "Replace the explicitly bound optional component selection"}


def run_host(runtime, args, *, entry_pass=False):
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    with pytest.MonkeyPatch.context() as patches:
        calls = checkpoints.host_runtime(runner, provider, patches, entry_pass=entry_pass)
        result = runner.run()
    return runner, result, calls


@pytest.fixture(scope="module", params=[False, True], ids=["enabled-origin", "schedule20-enable21"])
def completed_parent(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("component-refinement")
    args = checkpoints.fixture_arguments(root, 20 if request.param else 30)
    args.update(postproposal=partial.guard(20) if request.param else guard(30, top_k=2, old=True),
        boundary_every=10, stage_id="refinement-parent-origin")
    args["data_binding"]["postproposal_source"] = finite._reference(partial.__file__)
    args["source_evidence"]["partial_fixture"] = finite._reference(partial.__file__)
    if request.param:
        runtime = root / "completed20"
        _runner, result, _calls = run_host(runtime, args)
        state = result.states[0]
        args = {**args, "parameters": state.policy_state["values"], "updates": 10,
            "training_schedule": checkpoints.schedule(30), "postproposal": partial.guard(30),
            "stage_id": "append-before-enabling", "continuation_checkpoint": state,
            "continuation_binding": partial.completed_binding(runtime, state, runtime),
            SCHEDULE: {"effective_update": 20, "reason": "Append ten original entries"}}
        runtime = root / "refused21"
        state, binding, _calls = policies.refused_run(runtime, args, clock=21)
        args = policies.child_arguments((state, args, runtime, binding))
        args["postproposal"] = guard(30, top_k=2, old=True)
    else:
        args["data_binding"]["postproposal_component_source"] = finite._reference(policies.__file__)
        args["source_evidence"]["component_fixture"] = finite._reference(policies.__file__)
    runtime = root / "completed30"
    _runner, result, _calls = run_host(runtime, args)
    state = result.states[0]
    return state, args, runtime, partial.completed_binding(runtime, state, runtime)


def child_arguments(completed_parent):
    state, original, _runtime, binding = completed_parent
    args = {**original, "parameters": state.policy_state["values"], "updates": 10, "boundary_every": 10,
        "training_schedule": checkpoints.schedule(40), "postproposal": guard(), "stage_id": "refine-and-append30",
        "continuation_checkpoint": state, "continuation_binding": copy.deepcopy(binding),
        "data_binding": copy.deepcopy(original["data_binding"]),
        "source_evidence": copy.deepcopy(original["source_evidence"]), POLICY: declaration(),
        SCHEDULE: {"effective_update": 30, "reason": "Append ten declared entries after completion"}}
    args.pop("partial_parent_continuation", None)
    args["data_binding"]["postproposal_component_source"] = finite._reference(__file__)
    args["source_evidence"]["refined_component_fixture"] = finite._reference(__file__)
    return args


def assert_recovery(runtime, args, expected):
    calls = list(COMPONENT_CALLS)
    policies.assert_recovery(runtime, args, expected)
    assert COMPONENT_CALLS == calls


def assert_transition_copies(runtime, runner, state):
    configuration = json.loads((runtime / "configuration.json").read_text())
    for name in (POLICY, SCHEDULE):
        assert canonical_json(configuration["checkpoint_continuation"][name]) == canonical_json(
            runner.design.optimizer_binding[name]) == canonical_json(state.metadata["checkpoint_continuation"][name])
    bound_state = state if "replication_design" in state.metadata else runner.design.bind_checkpoint(state)
    finite._postproposal_lineage(bound_state, configuration)
    return configuration


def test_completed_refinement_extension_preserves_state_and_recovers_then_extends(completed_parent, tmp_path, monkeypatch):
    parent, _original, parent_root, _binding = completed_parent
    old_files, old_state = partial.hashes(parent_root), parent.to_dict()
    old_config = json.loads((parent_root / "configuration.json").read_text())
    args = child_arguments(completed_parent)
    runtime = tmp_path / "refined40"
    before_calls = list(COMPONENT_CALLS), list(policies.COMPONENT_CALLS)
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    assert checkpoints.numerical(runner.states[0]) == checkpoints.numerical(parent)
    assert runner.states[0].metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]
    assert (COMPONENT_CALLS, policies.COMPONENT_CALLS) == before_calls
    assert not runner.design.optimizer_binding["fresh_moments"]
    configuration = assert_transition_copies(runtime, runner, runner.states[0])
    history = configuration["checkpoint_continuation"]
    appended, refined = history[SCHEDULE], history[POLICY]
    intermediate = schedules.clone(old_config["postproposal"])
    intermediate["profile"]["profile"]["training_schedule"] = list(checkpoints.schedule(40))
    assert appended["old_postproposal_hash"] == stable_hash(old_config["postproposal"])
    assert appended["new_postproposal_hash"] == refined["old_postproposal_hash"] == stable_hash(intermediate)
    assert refined["new_postproposal_hash"] == stable_hash(configuration["postproposal"])
    assert refined["previous_schedule_transition_hash"] == stable_hash(appended)
    assert refined["old_schedule_hash"] == refined["new_schedule_hash"] == stable_hash(checkpoints.schedule(40))
    previous = old_config.get("checkpoint_continuation", {})
    assert appended["previous_transition"] == previous.get(SCHEDULE)
    assert refined["previous_transition"] == previous.get(POLICY)
    assert appended.get("previous_policy_transition_hash") == (
        None if previous.get(POLICY) is None else stable_hash(previous[POLICY]))
    for receipt in (appended, refined):
        assert receipt["effective_update"] == 30
        assert receipt["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
        assert receipt["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    assert refined["old_component_fallback"] == old_config["postproposal"]["profile"]["component_fallback"]
    assert refined["old_component_callback"] == old_config["postproposal"]["component_callback"]
    assert configuration["data_binding"]["postproposal_component_source"] == finite._reference(__file__)
    assert old_config["data_binding"]["postproposal_component_source"] == finite._reference(policies.__file__)
    assert configuration["data_binding"]["objective_recipe"] == old_config["data_binding"]["objective_recipe"]
    assert configuration["callback"] == old_config["callback"]
    assert configuration["training"]["callback"] == old_config["training"]["callback"]
    for name in ("loss_callback", "batch_binding_callback"):
        assert configuration["postproposal"][name] == old_config["postproposal"][name]
    calls = checkpoints.host_runtime(runner, provider, monkeypatch, entry_pass=True)
    result = runner.run()
    endpoint = result.states[0]
    assert calls["updates"] == list(range(30, 40)) and calls["controls"] == [30, 40]
    assert endpoint.update_index == endpoint.optimizer_state["iteration"] == endpoint.rng_state["next_seed_index"] == 40
    assert endpoint.optimizer_state["first_moment"] == [value + 10. for value in parent.optimizer_state["first_moment"]]
    assert endpoint.optimizer_state["second_moment"] == [value + 20. for value in parent.optimizer_state["second_moment"]]
    old_history = list(parent.metadata["permanent_pass_rotation"]["history"])
    assert list(endpoint.metadata["permanent_pass_rotation"]["history"][:len(old_history)]) == old_history
    assert_recovery(runtime, args, result)
    later = {**args, "parameters": endpoint.policy_state["values"], "training_schedule": checkpoints.schedule(50),
        "postproposal": guard(50), "stage_id": "schedule-only40-to50", "continuation_checkpoint": endpoint,
        "continuation_binding": partial.completed_binding(runtime, endpoint, runtime),
        SCHEDULE: {"effective_update": 40, "reason": "Append after refined completion"}}
    later.pop(POLICY)
    following_root = tmp_path / "extended50"
    following, final, _calls = run_host(following_root, later)
    following_config = assert_transition_copies(following_root, following, final.states[0])
    assert following_config["checkpoint_continuation"][POLICY] == refined
    assert following_config["checkpoint_continuation"][SCHEDULE]["previous_policy_transition_hash"] == stable_hash(refined)
    assert_recovery(following_root, later, final)
    assert parent.to_dict() == old_state and partial.hashes(parent_root) == old_files
    assert (COMPONENT_CALLS, policies.COMPONENT_CALLS) == before_calls


def test_same_clock_refusal_ordinary_partial_and_policy_only_refinement(completed_parent, tmp_path):
    args = child_arguments(completed_parent)
    runtime = tmp_path / "refused30"
    state, binding, calls = policies.refused_run(runtime, args, clock=30)
    assert calls["updates"] == [] and calls["controls"] == [30]
    old_files = partial.hashes(runtime)
    later = {**args, "parameters": state.policy_state["values"], "stage_id": "ordinary-refined-partial30",
        "continuation_checkpoint": state, "continuation_binding": binding,
        "partial_parent_continuation": partial.declaration(30)}
    later.pop(POLICY)
    later.pop(SCHEDULE)
    ordinary_root = tmp_path / "ordinary40"
    runner, result, calls = run_host(ordinary_root, later)
    assert calls["updates"] == list(range(30, 40)) and calls["controls"] == [30, 40]
    assert_transition_copies(ordinary_root, runner, result.states[0])
    assert runner.checkpoint_continuation[POLICY] == state.metadata["checkpoint_continuation"][POLICY]
    assert_recovery(ordinary_root, later, result)
    refined_args = {**later, "stage_id": "policy-only-at-partial30", "postproposal": guard(top_k=4), POLICY: declaration()}
    refined_root = tmp_path / "partial-refined40"
    refined, _adapter, _provider = finite.build_runner(refined_root, **refined_args)
    assert checkpoints.numerical(refined.states[0]) == checkpoints.numerical(state)
    assert refined.states[0].metadata["permanent_pass_rotation"] == state.metadata["permanent_pass_rotation"]
    configured = assert_transition_copies(refined_root, refined, refined.states[0])
    assert configured["training"]["schedule"] == list(checkpoints.schedule(40))
    assert configured["checkpoint_continuation"][SCHEDULE] == state.metadata["checkpoint_continuation"][SCHEDULE]
    for fault in ("unused-entry", "combined-partial-append"):
        invalid = {**refined_args, "stage_id": fault, "postproposal": guard(top_k=4)}
        if fault == "unused-entry":
            changed = list(checkpoints.schedule(40))
            changed[39] = {**changed[39], "row_indices": [2]}
            invalid["training_schedule"] = tuple(changed)
            invalid["postproposal"].binding["profile"]["training_schedule"] = changed
            invalid["postproposal"].binding_hash = stable_hash(invalid["postproposal"].binding)
        else:
            invalid.update(updates=20, training_schedule=checkpoints.schedule(50), postproposal=guard(50, top_k=4))
            invalid[SCHEDULE] = {"effective_update": 30, "reason": "Unsupported combined partial append"}
        with pytest.raises(ValueError):
            finite.build_runner(tmp_path / fault, **invalid)
    assert partial.hashes(runtime) == old_files


def test_refinement_rejects_undeclared_cochanges_and_wrong_sources(completed_parent, tmp_path):
    faults = ("missing-policy", "missing-kind", "unknown-kind", "wrong-clock", "different-schedule-clock", "empty-reason",
        "missing-schedule", "no-change", "removed", "prefix", "guard-schedule", "fractions", "margin", "tolerance",
        "missing-source", "wrong-source", "missing-evidence", "recipe", "other-data", "old-loss",
        "rate-transition", "method-transition", "estimator-transition")
    before = list(COMPONENT_CALLS), list(policies.COMPONENT_CALLS)
    for fault in faults:
        args = child_arguments(completed_parent)
        if fault in ("missing-policy", "missing-schedule"):
            args.pop(POLICY if fault == "missing-policy" else SCHEDULE)
        elif fault == "missing-kind":
            args[POLICY].pop("kind")
        elif fault in ("unknown-kind", "wrong-clock", "empty-reason"):
            key, value = {"unknown-kind": ("kind", "replace-anything"), "wrong-clock": ("effective_update", 29),
                          "empty-reason": ("reason", " ")}[fault]
            args[POLICY][key] = value
        elif fault == "different-schedule-clock":
            args[SCHEDULE]["effective_update"] = 29
        elif fault == "no-change":
            args["postproposal"] = guard(40, top_k=2, old=True)
            args["data_binding"]["postproposal_component_source"] = finite._reference(policies.__file__)
        elif fault == "removed":
            args["postproposal"] = partial.guard(40)
            args["data_binding"].pop("postproposal_component_source")
        elif fault in ("prefix", "guard-schedule"):
            changed = list(checkpoints.schedule(40))
            changed[29] = {**changed[29], "row_indices": [0]}
            args["training_schedule"] = tuple(changed)
            if fault == "prefix":
                args["postproposal"].binding["profile"]["training_schedule"] = changed
                args["postproposal"].binding_hash = stable_hash(args["postproposal"].binding)
        elif fault in ("fractions", "margin", "tolerance"):
            kwargs = {"fractions": {"fractions": (1., .1)}, "margin": {"margin_fraction": .2},
                      "tolerance": {"relative_tolerance": 1e-10}}[fault]
            args["postproposal"] = guard(**kwargs)
        elif fault == "missing-source":
            args["data_binding"].pop("postproposal_component_source")
        elif fault == "wrong-source":
            args["data_binding"]["postproposal_component_source"] = finite._reference(policies.__file__)
        elif fault == "missing-evidence":
            args["source_evidence"].pop("refined_component_fixture")
        elif fault == "recipe":
            args["data_binding"]["objective_recipe"]["changed"] = True
        elif fault == "other-data":
            args["data_binding"]["unrelated"] = "changed"
        elif fault == "old-loss":
            current = args["postproposal"]
            current.losses = component_rows
            current._callbacks = current.losses, current.batch_binding
            args["data_binding"]["postproposal_source"] = finite._reference(__file__)
            args["data_binding"]["postproposal_binding_source"] = finite._reference(partial.__file__)
        else:
            key = {"rate-transition": "learning_rate_transition", "method-transition": "method_policy_transition",
                   "estimator-transition": "estimator_transition"}[fault]
            args[key] = {"effective_update": 30, "reason": "Forbidden concurrent transition"}
        with pytest.raises((ValueError, TypeError)):
            finite.build_runner(tmp_path / fault, **args)
    assert (COMPONENT_CALLS, policies.COMPONENT_CALLS) == before


def test_repeated_entries_preserve_history_and_require_fresh_bound_controls(completed_parent, tmp_path):
    args = child_arguments(completed_parent)
    runtime = tmp_path / "first-entry-refused"
    parent, binding, _calls = policies.refused_run(runtime, args, clock=30)
    child = {**args, "parameters": parent.policy_state["values"], "stage_id": "fresh-repeat-entry",
        "continuation_checkpoint": parent, "continuation_binding": binding,
        "partial_parent_continuation": partial.declaration(30)}
    child.pop(POLICY)
    child.pop(SCHEDULE)
    runner, _adapter, provider = finite.build_runner(tmp_path / "next-entry", **child)
    initial, stage = runner.states[0], runner.stages[0]
    coordinator = runner.executor.core.coordinator
    original = coordinator.read(initial)
    policy = PolicyView.from_dict(initial.policy_state)
    with pytest.MonkeyPatch.context() as patches:
        checkpoints.host_runtime(runner, provider, patches)
        control = provider.control(0, policy, stage, stage.from_round)
    with pytest.raises(ValueError, match="fresh stage-entry control"):
        coordinator.require_stage_entry(initial)
    for invalid, stage_id in (
            (original.controls[-1].evaluation, stage.stage_id),
            (control, "unrelated-stage"),
            (replace(control, request=replace(control.request,
                seeds=tuple(seed + 1 for seed in control.request.seeds))), stage.stage_id)):
        with pytest.raises(ValueError):
            coordinator.stage_entry(initial, invalid, stage_id=stage_id)
    with pytest.raises(ValueError):
        coordinator.stage_entry(replace(initial, update_index=31), control, stage_id=stage.stage_id)
    with pytest.raises(ValueError, match="distinct stage"):
        original.stage_entry(control, policy, provider.roles, stage_id=original.controls[-1].stage_id)
    updated, _entry = coordinator.stage_entry(initial, control, stage_id=stage.stage_id)
    actual = coordinator.require_stage_entry(updated)
    assert actual.controls[:-1] == original.controls
    assert actual.controls[-2].stage_id != actual.controls[-1].stage_id == stage.stage_id
    assert actual.controls[-2].update_index == actual.controls[-1].update_index == 30
    assert actual.controls[-1].evaluation == control
    assert checkpoints.numerical(updated) == checkpoints.numerical(initial)
    assert set(actual.permanent) >= set(original.permanent)
    assert PermanentPassState.from_dict(actual.to_dict()) == actual
    with pytest.raises(ValueError, match="distinct stage"):
        coordinator.stage_entry(updated, control, stage_id=stage.stage_id)
    assert coordinator.read(initial) == original


def helper_binding(count=40, *, top_k=3):
    binding = policies.helper_binding(guard(count, top_k=top_k))
    binding["component_callback"]["source"] = finite._reference(__file__)
    return binding


def refined_history():
    old = policies.helper_history()
    parent = policies.helper_parent(old, 30)
    binding, training = helper_binding(), {"schedule": list(checkpoints.schedule(40))}
    schedule, policy = finite._postproposal_transitions(parent, old, binding, training, declaration(),
        {"effective_update": 30, "reason": "Append with refinement"})
    return {"postproposal": binding, "training": training, "checkpoint_continuation": {SCHEDULE: schedule, POLICY: policy}}


def test_same_clock_refinements_follow_hashes_and_preserve_all_prior_edges():
    configuration = refined_history()
    parent = policies.helper_parent(configuration, 30)
    binding = helper_binding(top_k=4)
    schedule, policy = finite._postproposal_transitions(parent, configuration, binding,
        configuration["training"], declaration(), None, partial={"old_schedule_length": 40})
    updated = {**configuration, "postproposal": binding, "checkpoint_continuation": {SCHEDULE: schedule, POLICY: policy}}
    assert policy["previous_transition"] == configuration["checkpoint_continuation"][POLICY]
    assert schedule == configuration["checkpoint_continuation"][SCHEDULE]
    assert finite._postproposal_lineage(policies.helper_parent(updated, 30), updated) == (schedule, policy)


def test_reject_disconnected_intermediate_histories_and_component_records():
    for fault in ("missing-schedule", "missing-policy", "intermediate", "cross-schedule", "cross-policy", "old-component",
                  "old-callback", "new-component", "new-callback", "schedule-hash", "future-clock", "kind", "prior-policy"):
        configuration = refined_history()
        history = configuration["checkpoint_continuation"]
        policy, schedule = history[POLICY], history[SCHEDULE]
        if fault == "missing-schedule":
            history[SCHEDULE] = None
        elif fault == "missing-policy":
            history[POLICY] = None
        elif fault in ("intermediate", "cross-schedule", "schedule-hash"):
            key = {"intermediate": "old_postproposal_hash", "cross-schedule": "previous_schedule_transition_hash",
                   "schedule-hash": "old_schedule_hash"}[fault]
            policy[key] = "0" * 64
        elif fault == "cross-policy":
            schedule["previous_policy_transition_hash"] = "0" * 64
            policy["previous_schedule_transition_hash"] = stable_hash(schedule)
        elif fault in ("old-component", "new-component"):
            key = "old_component_fallback" if fault == "old-component" else "component_fallback"
            policy[key]["provider"]["criterion"] = "altered"
        elif fault in ("old-callback", "new-callback"):
            key = "old_component_callback" if fault == "old-callback" else "component_callback"
            policy[key]["qualname"] = "altered"
        elif fault == "future-clock":
            policy["effective_update"] = 31
        elif fault == "kind":
            policy["kind"] = "undeclared-kind"
        else:
            policy["previous_transition"] = None
        with pytest.raises(ValueError, match="postproposal lineage"):
            finite._postproposal_lineage(policies.helper_parent(configuration, 30), configuration)


def test_refinement_keeps_guard_criteria_and_data_independent_of_provider_metadata():
    configuration = policies.helper_history()
    parent = policies.helper_parent(configuration, 30)
    for keys, value in ((["profile", "D"], [2.] * 7), (["profile", "T"], [2.] * 7),
            (["profile", "component_fallback", "trigger"], "always"),
            (["profile", "component_fallback", "scale"], "unit"),
            (["profile", "profile", "criterion"], "changed"),
            (["loss_callback", "qualname"], "changed"), (["batch_binding_callback", "qualname"], "changed")):
        binding = helper_binding()
        schedules.set_nested(binding, keys, value)
        with pytest.raises(ValueError, match="postproposal"):
            finite._postproposal_transitions(parent, configuration, binding, {"schedule": list(checkpoints.schedule(40))},
                declaration(), {"effective_update": 30, "reason": "Append only the declared schedule"})


def test_archived_enabled_guard_is_exact_immutable_metadata_and_cannot_run(completed_parent):
    _state, _args, runtime, _binding = completed_parent
    old = json.loads((runtime / "configuration.json").read_text())["postproposal"]
    before = schedules.clone(old)
    archived = finite._archived_postproposal(guard(), old, refinement=True)
    assert canonical_json(archived.declaration) == canonical_json(old)
    assert canonical_json(archived.binding) == canonical_json(old["profile"])
    assert archived.binding_hash == stable_hash(old["profile"])
    archived.validate_binding()
    with pytest.raises(RuntimeError, match="cannot execute"):
        archived(parameters=None)
    with pytest.raises(TypeError):
        archived.binding["component_fallback"]["provider"]["top_k"] = 3
    with pytest.raises(ValueError, match="archived component provider"):
        finite._archived_postproposal(guard(), old)
    archived.binding = guard().binding
    with pytest.raises(ValueError, match="archived postproposal metadata"):
        archived.validate_binding()
    assert old == before


def test_inconsistent_refinement_metadata_copies_fail_before_import():
    for location in ("configuration", "design", "checkpoint"):
        configuration = refined_history()
        parent = policies.helper_parent(configuration, 30)
        metadata = schedules.clone(parent.metadata)
        if location == "configuration":
            configuration["checkpoint_continuation"][POLICY]["reason"] = "different"
        elif location == "design":
            metadata["replication_design"]["optimizer_binding"][POLICY]["reason"] = "different"
        else:
            metadata["checkpoint_continuation"][POLICY]["reason"] = "different"
        with pytest.raises(ValueError, match="postproposal lineage"):
            finite._postproposal_lineage(replace(parent, metadata=metadata), configuration)
