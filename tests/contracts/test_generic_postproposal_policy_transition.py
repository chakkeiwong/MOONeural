"""Guard-policy provenance and shared transactions with manufactured numerical leaves.

The parent runs qualification. These tests reuse the existing host lifecycle;
component/model callbacks are forbidden. Kernel correctness has its own suite.
"""

import copy
import json
from pathlib import Path
from dataclasses import replace

import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_partial_continuation as partial
from tests.contracts import test_generic_postproposal_continuation as schedules
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_permanent_pass_executor import PermanentPassUpdateError
from mooneural.training.generic_training_contracts import canonical_json, stable_hash

POLICY = "postproposal_policy_transition"
SCHEDULE = "postproposal_schedule_transition"
COMPONENT_CALLS = []


def component_rows(parameters, batch, update_index, context):
    COMPONENT_CALLS.append(update_index)
    raise AssertionError("host transition or recovery called component numerics")


def replacement_rows(parameters, batch, update_index, context):
    raise AssertionError("replacement provider must be rejected before execution")


def declaration(clock=21):
    return {"effective_update": clock, "reason": "Enable the qualified optional component provider"}


def enabled_guard(count=30, *, provider=component_rows, provider_binding=None, **changes):
    return partial.guard(count, component_rows=provider,
        component_binding={"criterion": "manufactured maximum components", "coordinate_mode": "raw_over_D"}
        if provider_binding is None else provider_binding, **changes)


def saved_partial_binding(runtime, state):
    reference = finite._write_json(runtime.parent / (runtime.name + "-partial.json"),
        {"training_completed": False, "final_committed_update": state.update_index, "status": "FAILED_OR_PARTIAL"})
    store = runtime / "checkpoints"
    checkpoint = store / "arms/arm-0/checkpoints" / f"update-{state.update_index:08d}.json"
    if not checkpoint.is_file():
        checkpoint = store / "arms/arm-0/initial/state.json"
    rotation = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"])
    return {"parent_result": reference, "parent_checkpoint": finite._reference(checkpoint),
        "parent_commit_marker": finite._reference(checkpoint.with_suffix(".complete.json")),
        "parent_configuration": finite._reference(runtime / "configuration.json"),
        "parent_refusal": finite._reference(store / "rejections/arm-0" / f"{stable_hash(state.to_dict())}.json"),
        "parent_control_archive": [finite._reference(store / "control_evidence" /
            f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in rotation.controls]}


def refused_run(runtime, args, clock=21):
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    with pytest.MonkeyPatch.context() as patches:
        calls = checkpoints.host_runtime(runner, provider, patches)
        inherited_step = runner.executor.step

        def refuse(state, batch, adapter):
            if state.update_index != clock:
                return inherited_step(state, batch, adapter)
            partition = runner.executor.core.coordinator.read(state).partition
            event = {"event": "update", "committed": False, "transaction_rolled_back": True,
                "policy_before": state.policy_fingerprint, "before": partition, "after": partition,
                "postproposal": {"accepted": False, "update_index": clock,
                    "profile_hash": args["postproposal"].binding_hash, "loss_calls": 1,
                    "candidates": []}}
            raise PermanentPassUpdateError("manufactured finite guard refusal", state, event)

        patches.setattr(runner.executor, "step", refuse)
        with pytest.raises(PermanentPassUpdateError, match="manufactured finite guard refusal") as failed:
            runner.run()
    state = failed.value.checkpoint
    return state, saved_partial_binding(runtime, state), calls


@pytest.fixture(scope="module", params=[False, True], ids=["no-schedule-lineage", "schedule20-lineage"])
def retained_parent(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("guard-policy-parent")
    count = 20 if request.param else 30
    args = checkpoints.fixture_arguments(root, count)
    args.update(postproposal=partial.guard(count), boundary_every=10, stage_id="policy-parent-origin")
    args["data_binding"]["postproposal_source"] = finite._reference(partial.__file__)
    args["source_evidence"]["partial_fixture"] = finite._reference(partial.__file__)
    if request.param:
        runtime = root / "completed20"
        runner, _adapter, provider = finite.build_runner(runtime, **args)
        with pytest.MonkeyPatch.context() as patches:
            checkpoints.host_runtime(runner, provider, patches)
            state = runner.run().states[0]
        args = {**args, "parameters": state.policy_state["values"], "updates": 10,
            "training_schedule": checkpoints.schedule(30), "postproposal": partial.guard(30),
            "stage_id": "policy-parent20-to30", "continuation_checkpoint": state,
            "continuation_binding": partial.completed_binding(root, state, runtime),
            SCHEDULE: {"effective_update": 20, "reason": "Original full schedule extension"}}
    runtime = root / "refused21"
    state, binding, _calls = refused_run(runtime, args)
    return state, args, runtime, binding


def child_arguments(retained_parent):
    parent, original, _runtime, binding = retained_parent
    args = {**original, "parameters": parent.policy_state["values"], "updates": 9, "boundary_every": 9,
        "training_schedule": checkpoints.schedule(30), "postproposal": enabled_guard(),
        "data_binding": copy.deepcopy(original["data_binding"]),
        "source_evidence": copy.deepcopy(original["source_evidence"]), "stage_id": "enable-at21",
        "continuation_checkpoint": parent, "continuation_binding": copy.deepcopy(binding),
        "partial_parent_continuation": partial.declaration(21), POLICY: declaration()}
    args.pop(SCHEDULE, None)
    args["data_binding"]["postproposal_component_source"] = finite._reference(__file__)
    args["source_evidence"]["component_fixture"] = finite._reference(__file__)
    return args


def assert_recovery(runtime, args, expected):
    before = partial.hashes(runtime)
    calls = list(COMPONENT_CALLS)
    runner, adapter, provider = finite.build_runner(runtime, **args)
    with pytest.MonkeyPatch.context() as patches:
        partial.forbid_runtime(runner, adapter, provider, patches)
        actual = runner.run()
    assert actual.states[0].to_dict() == expected.states[0].to_dict()
    assert actual.selection == expected.selection and actual.evaluations == expected.evaluations
    assert partial.hashes(runtime) == before and COMPONENT_CALLS == calls


def assert_policy_copies(runtime, runner, state):
    configuration = json.loads((runtime / "configuration.json").read_text())
    receipt = configuration["checkpoint_continuation"][POLICY]
    assert canonical_json(receipt) == canonical_json(runner.design.optimizer_binding[POLICY])
    assert canonical_json(receipt) == canonical_json(state.metadata["checkpoint_continuation"][POLICY])
    return configuration, receipt


def test_enable_complete_ordinary_extension_and_zero_call_recovery(retained_parent, tmp_path, monkeypatch):
    parent, original, parent_root, _binding = retained_parent
    old_files = partial.hashes(parent_root)
    old_state = parent.to_dict()
    args = child_arguments(retained_parent)
    runtime = tmp_path / "enabled"
    before_calls = list(COMPONENT_CALLS)
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    assert checkpoints.numerical(runner.states[0]) == checkpoints.numerical(parent)
    assert runner.states[0].metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]
    assert not runner.design.optimizer_binding["fresh_moments"] and COMPONENT_CALLS == before_calls
    configuration, policy = assert_policy_copies(runtime, runner, runner.states[0])
    assert configuration["training"]["schedule"] == list(checkpoints.schedule(30))
    assert policy["old_schedule_hash"] == policy["new_schedule_hash"] == stable_hash(checkpoints.schedule(30))
    assert policy["effective_update"] == 21 and policy["previous_transition"] is None
    assert policy["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
    assert policy["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    old_config = json.loads((parent_root / "configuration.json").read_text())
    assert configuration["data_binding"]["objective_recipe"] == old_config["data_binding"]["objective_recipe"]
    assert configuration["callback"] == old_config["callback"]
    assert configuration["training"]["callback"] == old_config["training"]["callback"]
    for name in ("loss_callback", "batch_binding_callback"):
        assert configuration["postproposal"][name] == old_config["postproposal"][name]
    assert configuration["checkpoint_continuation"].get(SCHEDULE) == old_config.get("checkpoint_continuation", {}).get(SCHEDULE)
    assert configuration["postproposal"]["component_callback"]["source"] == finite._reference(__file__)
    component_source = finite._reference(Path(finite.__file__).parent / "generic_component_descent.py")
    assert any(str((finite.SOURCE_ROOT / path).resolve()) == component_source["path"] and digest == component_source["sha256"]
               for path, digest in configuration["sources"].items())
    calls = checkpoints.host_runtime(runner, provider, monkeypatch, entry_pass=True)
    step = runner.executor.step

    def entry_before_update(state, batch, adapter):
        assert (runtime / "checkpoints/arms/arm-0/initial/state.complete.json").is_file()
        runner.executor.core.coordinator.require_stage_entry(state)
        return step(state, batch, adapter)

    monkeypatch.setattr(runner.executor, "step", entry_before_update)
    result = runner.run()
    assert calls["updates"] == list(range(21, 30)) and calls["controls"] == [21, 30]
    rotation = runner.executor.core.coordinator.read(result.states[0])
    prior = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"])
    assert rotation.controls[:len(prior.controls)] == prior.controls
    assert rotation.controls[len(prior.controls)].update_index == 21
    assert set(rotation.permanent) >= {*prior.permanent, "task.0"}
    assert_policy_copies(runtime, runner, result.states[0])
    assert_recovery(runtime, args, result)
    later = {**args, "parameters": result.states[0].policy_state["values"], "updates": 1, "boundary_every": 1,
        "training_schedule": checkpoints.schedule(31), "postproposal": enabled_guard(31),
        "stage_id": "ordinary-after-policy", "continuation_checkpoint": result.states[0],
        "continuation_binding": partial.completed_binding(tmp_path, result.states[0], runtime),
        SCHEDULE: {"effective_update": 30, "reason": "Append after the enabled completed stage"}}
    later.pop(POLICY)
    later.pop("partial_parent_continuation")
    extended = tmp_path / "ordinary"
    following, _adapter, provider = finite.build_runner(extended, **later)
    assert checkpoints.numerical(following.states[0]) == checkpoints.numerical(result.states[0])
    child_config, inherited = assert_policy_copies(extended, following, following.states[0])
    assert inherited == policy
    new_schedule = child_config["checkpoint_continuation"][SCHEDULE]
    assert new_schedule["old_postproposal_hash"] == policy["new_postproposal_hash"]
    assert new_schedule["previous_transition"] == configuration["checkpoint_continuation"].get(SCHEDULE)
    assert new_schedule["new_postproposal_hash"] != policy["new_postproposal_hash"]
    checkpoints.host_runtime(following, provider, monkeypatch)
    completed = following.run()
    assert completed.states[0].update_index == 31
    assert_recovery(extended, later, completed)
    assert COMPONENT_CALLS == before_calls and parent.to_dict() == old_state
    assert partial.hashes(parent_root) == old_files
    assert original["data_binding"].get("postproposal_component_source") is None


def test_enabled_child_refuses_again_at21_then_inherits_without_reenable(retained_parent, tmp_path, monkeypatch):
    args = child_arguments(retained_parent)
    runtime = tmp_path / "refused-enabled"
    parent, binding, calls = refused_run(runtime, args)
    assert calls["updates"] == [] and calls["controls"] == [21]
    assert binding["parent_checkpoint"]["path"].endswith("initial/state.json")
    original_files = partial.hashes(runtime)
    later = {**args, "stage_id": "repeat-partial-at21", "parameters": parent.policy_state["values"],
        "continuation_checkpoint": parent, "continuation_binding": binding}
    later.pop(POLICY)
    output = tmp_path / "repeat-child"
    runner, _adapter, provider = finite.build_runner(output, **later)
    assert checkpoints.numerical(runner.states[0]) == checkpoints.numerical(parent)
    assert runner.checkpoint_continuation[POLICY] == parent.metadata["checkpoint_continuation"][POLICY]
    calls = checkpoints.host_runtime(runner, provider, monkeypatch)
    result = runner.run()
    assert calls["updates"] == list(range(21, 30)) and calls["controls"] == [21, 30]
    history = runner.executor.core.coordinator.read(result.states[0]).controls
    at21 = [point for point in history if point.update_index == 21]
    assert len(at21) == 2 and at21[0].stage_id != at21[1].stage_id
    assert_recovery(output, later, result)
    assert partial.hashes(runtime) == original_files


@pytest.mark.parametrize("fault", ["missing", "wrong-clock", "empty-reason", "extra-key", "same-policy",
    "fractions", "margin", "tolerance", "normalization", "limits", "unused-row", "old-loss", "old-batch",
    "missing-source", "wrong-source", "missing-evidence", "recipe", "other-data", "schedule-transition",
    "unused-source", "rate-transition", "method-transition", "estimator-transition"])
def test_enable_refusals_before_callbacks(retained_parent, tmp_path, fault):
    args = child_arguments(retained_parent)
    if fault == "missing":
        args.pop(POLICY)
    elif fault in ("wrong-clock", "empty-reason", "extra-key"):
        key, value = {"wrong-clock": ("effective_update", 20), "empty-reason": ("reason", " "),
                      "extra-key": ("enabled", True)}[fault]
        args[POLICY][key] = value
    elif fault == "same-policy":
        args["postproposal"] = partial.guard(30)
        args["data_binding"].pop("postproposal_component_source")
    elif fault in ("fractions", "margin", "tolerance"):
        changes = {"fractions": {"fractions": (1., .1)}, "margin": {"margin_fraction": .2},
                   "tolerance": {"relative_tolerance": 1e-10}}[fault]
        args["postproposal"] = enabled_guard(**changes)
    elif fault in ("normalization", "limits"):
        guard = args["postproposal"]
        altered = [2., *([1.] * 6)]
        args["postproposal"] = type(guard)(partial.guard_losses, partial.batch_binding,
            altered if fault == "normalization" else [1.] * 7,
            altered if fault == "limits" else [1.] * 7,
            binding=guard.binding["profile"], component_rows=component_rows,
            component_binding=guard.binding["component_fallback"]["provider"])
    elif fault == "unused-row":
        args["training_schedule"] = copy.deepcopy(args["training_schedule"])
        args["training_schedule"][29]["row_indices"] = [99]
    elif fault in ("old-loss", "old-batch"):
        guard = args["postproposal"]
        if fault == "old-loss":
            guard.losses = replacement_rows
        else:
            guard.batch_binding = replacement_rows
        guard._callbacks = guard.losses, guard.batch_binding
        args["data_binding"]["postproposal_source" if fault == "old-loss" else "postproposal_binding_source"] = finite._reference(__file__)
    elif fault == "missing-source":
        args["data_binding"].pop("postproposal_component_source")
    elif fault == "wrong-source":
        args["data_binding"]["postproposal_component_source"] = finite._reference(partial.__file__)
    elif fault == "missing-evidence":
        args["source_evidence"].pop("component_fixture")
    elif fault == "recipe":
        args["data_binding"]["objective_recipe"]["component"] = True
    elif fault == "other-data":
        args["data_binding"]["profile"] = "different finite data"
    elif fault == "unused-source":
        args["postproposal"] = partial.guard(30)
    elif fault in ("rate-transition", "method-transition", "estimator-transition"):
        key = {"rate-transition": "learning_rate_transition", "method-transition": "method_policy_transition",
               "estimator-transition": "estimator_transition"}[fault]
        args[key] = {"effective_update": 21, "reason": "Forbidden simultaneous state transition"}
    else:
        args[SCHEDULE] = {"effective_update": 21, "reason": "Forbidden simultaneous extension"}
    before = list(COMPONENT_CALLS)
    with pytest.raises((ValueError, TypeError)):
        finite.build_runner(tmp_path / fault, **args)
    assert COMPONENT_CALLS == before


def helper_binding(guard):
    source = finite._reference(__file__)
    binding = {"profile": schedules.clone(guard.binding),
        "loss_callback": {"source": finite._reference(partial.__file__), "qualname": partial.guard_losses.__qualname__},
        "batch_binding_callback": {"source": finite._reference(partial.__file__), "qualname": partial.batch_binding.__qualname__}}
    if guard.component_rows is not None:
        binding["component_callback"] = {"source": source, "qualname": guard.component_rows.__qualname__}
    return binding


def helper_parent(config, clock=21):
    parent = schedules.helper_parent(config, clock)
    metadata = schedules.clone(parent.metadata)
    for name in (POLICY, SCHEDULE):
        previous = schedules.clone(config.get("checkpoint_continuation", {}).get(name))
        metadata["checkpoint_continuation"][name] = previous
        metadata["replication_design"]["optimizer_binding"][name] = previous
    return replace(parent, metadata=metadata)


def helper_history():
    original = {"postproposal": helper_binding(partial.guard(20)), "training": {"schedule": list(checkpoints.schedule(20))}}
    old = {"postproposal": helper_binding(partial.guard(30)), "training": {"schedule": list(checkpoints.schedule(30))}}
    schedule = finite._postproposal_schedule_transition(helper_parent(original, 20), original, old["postproposal"],
        old["training"], {"effective_update": 20, "reason": "Extend20 to30"})
    old["checkpoint_continuation"] = {SCHEDULE: schedule}
    binding = helper_binding(enabled_guard())
    policy = finite._postproposal_policy_transition(helper_parent(old), old, binding, old["training"], declaration())
    return {**old, "postproposal": binding, "checkpoint_continuation": {SCHEDULE: schedule, POLICY: policy}}


def test_hash_connected_same_clock_extension_and_legacy_disabled_bindings():
    config = helper_history()
    parent = helper_parent(config)
    updated = helper_binding(enabled_guard(31))
    training = {"schedule": list(checkpoints.schedule(31))}
    extended = finite._postproposal_schedule_transition(parent, config, updated, training,
        {"effective_update": 21, "reason": "Separate partial stage at the same clock"}, partial={"old_schedule_length": 30})
    final = {"postproposal": updated, "training": training,
        "checkpoint_continuation": {SCHEDULE: extended, POLICY: config["checkpoint_continuation"][POLICY]}}
    actual = finite._postproposal_lineage(helper_parent(final), final)
    assert actual == (extended, config["checkpoint_continuation"][POLICY])
    disabled = partial.guard(30)
    assert "component_fallback" not in disabled.binding
    archived = finite._archived_postproposal(enabled_guard(), helper_binding(disabled))
    assert archived.component_rows is None and archived.binding == disabled.binding
    enabled = finite._archived_postproposal(enabled_guard(31), config["postproposal"])
    assert enabled.component_rows is component_rows and enabled.binding == enabled_guard().binding


@pytest.mark.parametrize("fault", ["config-copy", "design-copy", "state-copy", "guard-copy", "missing-schedule", "missing-policy",
    "broken-link", "schedule-hash", "future-clock", "provider", "callback", "historical-link"])
def test_composed_lineage_refuses_inconsistent_copies_and_broken_links(fault):
    config = helper_history()
    parent = helper_parent(config)
    metadata = schedules.clone(parent.metadata)
    if fault in ("config-copy", "design-copy", "state-copy", "guard-copy"):
        if fault == "config-copy":
            config["checkpoint_continuation"][POLICY]["reason"] = "changed"
        elif fault == "design-copy":
            metadata["replication_design"]["optimizer_binding"][POLICY]["reason"] = "changed"
        elif fault == "state-copy":
            metadata["checkpoint_continuation"][POLICY]["reason"] = "changed"
        else:
            metadata["replication_design"]["optimizer_binding"]["postproposal"]["profile"]["margin_fraction"] = .3
        parent = replace(parent, metadata=metadata)
    else:
        history = config["checkpoint_continuation"]
        if fault == "missing-schedule":
            history[SCHEDULE] = None
        elif fault == "missing-policy":
            updated = helper_binding(enabled_guard(31))
            training = {"schedule": list(checkpoints.schedule(31))}
            history[SCHEDULE] = finite._postproposal_schedule_transition(parent, config, updated, training,
                {"effective_update": 21, "reason": "Later partial extension"}, partial={"old_schedule_length": 30})
            config.update(postproposal=updated, training=training)
            history[POLICY] = None
        elif fault == "broken-link":
            history[SCHEDULE]["new_postproposal_hash"] = "0" * 64
        elif fault == "schedule-hash":
            history[POLICY]["new_schedule_hash"] = "0" * 64
        elif fault == "future-clock":
            history[POLICY]["effective_update"] = 22
        elif fault == "provider":
            history[POLICY]["component_fallback"]["provider"]["criterion"] = "different"
        elif fault == "callback":
            history[POLICY]["component_callback"]["qualname"] = "different"
        else:
            extra = schedules.clone(history[SCHEDULE])
            extra["new_postproposal_hash"] = "1" * 64
            history[SCHEDULE]["previous_transition"] = extra
        parent = helper_parent(config)
    with pytest.raises(ValueError, match="postproposal lineage"):
        finite._postproposal_lineage(parent, config)


@pytest.mark.parametrize("fault", ["remove", "replace", "provider-binding", "reenable"])
def test_enabled_policy_cannot_be_removed_or_replaced(fault):
    config = helper_history()
    binding = helper_binding(partial.guard(30) if fault == "remove" else enabled_guard(
        provider=replacement_rows if fault == "replace" else component_rows,
        provider_binding={"criterion": "different"} if fault == "provider-binding" else None))
    parent = helper_parent(config)
    with pytest.raises(ValueError, match="postproposal"):
        if fault == "reenable":
            finite._postproposal_policy_transition(parent, config, binding, config["training"], declaration())
        else:
            finite._postproposal_schedule_transition(parent, config, binding, config["training"], None)


def test_component_declaration_references_are_checked(retained_parent, tmp_path):
    args = child_arguments(retained_parent)
    args["postproposal"] = enabled_guard(provider_binding={"source": {**finite._reference(__file__), "sha256": "0" * 64}})
    with pytest.raises(ValueError, match="bound file changed"):
        finite.build_runner(tmp_path / "bad-provider-artifact", **args)
