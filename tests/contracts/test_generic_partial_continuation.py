"""Real continuation/store paths with manufactured numerical leaves; parent runs qualification."""

import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace

import pytest
from tests.contracts.test_generic_checkpoint_continuation import (
    fixture_arguments,
    host_runtime,
    numerical,
    schedule,
)
from tests.contracts.test_generic_finite_objective_runner import forbidden

from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_permanent_pass_executor import PermanentPassUpdateError
from mooneural.training.generic_postproposal import GuardedPostproposal
from mooneural.training.generic_replication_design import ReplicationDesignBinding
from mooneural.training.generic_replication_runner import AtomicCheckpointStore
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    PolicyView,
    canonical_json,
    stable_hash,
)


def guard_losses(*arguments):
    return forbidden(*arguments)


def batch_binding(batch, update_index):
    assert batch["metadata"]["update_index"] == update_index
    return batch


def guard(count, **changes):
    return GuardedPostproposal(
        guard_losses, batch_binding, [1.] * 7, [1.] * 7,
        binding={"criterion": "manufactured-partial-parent", "training_schedule": list(schedule(count))},
        **changes,
    )


def declaration(clock=18):
    return {"effective_update": clock, "termination": "postproposal_rejected",
            "reason": "Retain the committed prefix after a manufactured finite rejection"}


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@contextmanager
def changed_file(path, contents):
    original = path.read_bytes() if path.exists() else None
    try:
        if contents is None:
            path.unlink()
        else:
            path.write_bytes(contents)
        yield
    finally:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)


def binding_for(root, checkpoint, result):
    store = root / "checkpoints"
    path = store / "arms/arm-0/checkpoints" / f"update-{checkpoint.update_index:08d}.json"
    rotation = PermanentPassState.from_dict(checkpoint.metadata["permanent_pass_rotation"])
    return {
        "parent_result": finite._reference(result), "parent_checkpoint": finite._reference(path),
        "parent_commit_marker": finite._reference(path.with_suffix(".complete.json")),
        "parent_configuration": finite._reference(root / "configuration.json"),
        "parent_refusal": finite._reference(store / "rejections/arm-0" / f"{stable_hash(checkpoint.to_dict())}.json"),
        "parent_control_archive": [finite._reference(store / "control_evidence" /
            f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in rotation.controls],
    }


def make_parent(root, *, boundary_every=10):
    args = fixture_arguments(root, 20)
    source = finite._reference(__file__)
    args.update(boundary_every=boundary_every, postproposal=guard(20))
    args["data_binding"]["postproposal_source"] = source
    args["source_evidence"]["partial_fixture"] = source
    runtime = root / "runtime"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    with pytest.MonkeyPatch.context() as patches:
        calls = host_runtime(runner, provider, patches)
        inherited_step = runner.executor.step

        def step(state, batch, adapter):
            if state.update_index != 18:
                return inherited_step(state, batch, adapter)
            partition = runner.executor.core.coordinator.read(state).partition
            event = {"event": "update", "committed": False, "transaction_rolled_back": True,
                "policy_before": state.policy_fingerprint, "before": partition, "after": partition,
                "postproposal": {"accepted": False, "update_index": 18,
                    "profile_hash": args["postproposal"].binding_hash, "loss_calls": 2,
                    "candidates": [{"fraction": 1., "accepted": False}]}}
            raise PermanentPassUpdateError("manufactured finite candidate refusal", state, event)

        patches.setattr(runner.executor, "step", step)
        with pytest.raises(PermanentPassUpdateError) as failed:
            runner.run()
        parent = failed.value.checkpoint
    assert calls["updates"] == list(range(18))
    assert parent.update_index == parent.optimizer_state["iteration"] == parent.rng_state["next_seed_index"] == 18
    result = root / "partial-result.json"
    finite._write_json(result, {"training_completed": False, "final_committed_update": 18,
                              "status": "FAILED_OR_PARTIAL"})
    return parent, args, runtime, binding_for(runtime, parent, result)


@pytest.fixture(scope="module")
def partial_parent(tmp_path_factory):
    return make_parent(tmp_path_factory.mktemp("generic-partial-parent"))


def child_arguments(fixture, count=20, *, stage_id="partial-child"):
    parent, original, _runtime, binding = fixture
    args = {**original, "parameters": parent.policy_state["values"], "updates": count - 18,
        "boundary_every": 1, "training_schedule": schedule(count), "postproposal": guard(count),
        "stage_id": stage_id, "continuation_checkpoint": parent,
        "continuation_binding": copy.deepcopy(binding), "partial_parent_continuation": declaration()}
    if count > 20:
        args["postproposal_schedule_transition"] = {"effective_update": 18, "reason": "Append after the full old prefix"}
    return args


def completed_binding(root, state, runtime):
    reference = finite._write_json(root / "completed-parent.json",
        {"training_completed": True, "final_state": state.to_dict()})
    rotation = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"])
    return {"parent_result": reference, "parent_checkpoint": reference,
        "parent_configuration": finite._reference(runtime / "configuration.json"),
        "parent_control_archive": [finite._reference(runtime / "checkpoints/control_evidence" /
            f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in rotation.controls]}


def forbid_runtime(runner, adapter, provider, patches):
    for name in ("control", "validation", "certification"):
        patches.setattr(provider, name, forbidden)
    for name in ("objective", "objective_values", "training_objective"):
        patches.setattr(adapter, name, forbidden)
    patches.setattr(runner, "batch_factory", forbidden)
    patches.setattr(runner.executor, "step", forbidden)


@pytest.mark.parametrize(("count", "change_method"), [(20, False), (20, True), (22, True)])
def test_partial_entry_state_prefix_recovery_and_ordinary_child(partial_parent, tmp_path, monkeypatch,
                                                               count, change_method):
    parent, _original, parent_root, _binding = partial_parent
    old_files = hashes(parent_root)
    args = child_arguments(partial_parent, count)
    expected = numerical(parent)
    if change_method:
        args["method_policy_transition"] = {"effective_update": 18, "old_preferred": "cagrad",
            "new_preferred": "pcgrad", "reason": "Manufactured explicit preference transition"}
        expected["method_state"]["preferred"] = "pcgrad"
    runtime = tmp_path / "child"
    runner, adapter, provider = finite.build_runner(runtime, **args)
    assert numerical(runner.states[0]) == expected
    assert runner.states[0].metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]
    assert not runner.design.optimizer_binding["fresh_moments"]
    receipt = runner.checkpoint_continuation["partial_parent_continuation"]
    assert receipt["old_schedule_length"] == receipt["declared_stop_update"] == 20
    assert receipt["last_control_update"] == 10
    assert receipt["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
    config = json.loads((runtime / "configuration.json").read_text())
    assert config["training"]["schedule"][:20] == list(schedule(20))
    assert config["checkpoint_continuation"]["partial_parent_continuation"] == receipt
    if count > 20:
        transition = config["checkpoint_continuation"]["postproposal_schedule_transition"]
        assert transition["old_schedule_hash"] == stable_hash(schedule(20))
    with pytest.raises(ValueError, match="fresh stage-entry control"):
        runner.executor.core.coordinator.require_stage_entry(runner.states[0])
    calls = host_runtime(runner, provider, monkeypatch, entry_pass=True)
    inherited_step = runner.executor.step

    def ordered_step(state, batch, supplied_adapter):
        assert (runtime / "checkpoints/arms/arm-0/initial/state.complete.json").is_file()
        runner.executor.core.coordinator.require_stage_entry(state)
        return inherited_step(state, batch, supplied_adapter)

    monkeypatch.setattr(runner.executor, "step", ordered_step)
    result = runner.run()
    assert calls["updates"] == list(range(18, count))
    assert calls["controls"] == list(range(18, count + 1))
    prior = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"])
    rotation = runner.executor.core.coordinator.read(result.states[0])
    assert rotation.controls[:len(prior.controls)] == prior.controls
    entry = rotation.controls[len(prior.controls)]
    assert entry.update_index == 18
    assert entry.evaluation.request.metadata["scheduled_role_scope"]["stage_id"] == args["stage_id"]
    assert len([point for point in rotation.controls if point.update_index == 18]) == 1
    assert set(rotation.permanent) >= {*prior.permanent, "task.0"}
    assert result.states[0].method_state["rates"] == parent.method_state["rates"]
    before = hashes(runtime)
    rebuilt, adapter, provider = finite.build_runner(runtime, **args)
    forbid_runtime(rebuilt, adapter, provider, monkeypatch)
    recovered = rebuilt.run()
    assert recovered.states[0].to_dict() == result.states[0].to_dict()
    assert recovered.selection == result.selection and recovered.evaluations == result.evaluations
    assert hashes(runtime) == before
    later = {**args, "parameters": result.states[0].policy_state["values"], "updates": 1,
        "stage_id": "ordinary-grandchild", "training_schedule": schedule(count + 1),
        "postproposal": guard(count + 1), "continuation_checkpoint": result.states[0],
        "continuation_binding": completed_binding(tmp_path, result.states[0], runtime),
        "postproposal_schedule_transition": {"effective_update": count, "reason": "Ordinary full-prefix extension"}}
    later.pop("partial_parent_continuation")
    later.pop("method_policy_transition", None)
    ordinary, _adapter, provider = finite.build_runner(tmp_path / "ordinary", **later)
    assert numerical(ordinary.states[0]) == numerical(result.states[0])
    assert "partial_parent_continuation" not in ordinary.checkpoint_continuation
    if change_method:
        assert ordinary.checkpoint_continuation["method_policy_transition"] == runner.checkpoint_continuation["method_policy_transition"]
    next_calls = host_runtime(ordinary, provider, monkeypatch)
    assert ordinary.run().states[0].update_index == count + 1
    assert next_calls["updates"] == [count]
    assert hashes(parent_root) == old_files


@pytest.mark.parametrize("fault", ["default", "clock", "termination", "future18", "future19", "past",
    "short_horizon", "data", "guard", "slots", "rng", "preference", "method_old", "method_clock",
    "fake_complete", "budget_only"])
def test_partial_refusals_before_numerical_calls(partial_parent, tmp_path, fault):
    parent, _original, parent_root, _binding = partial_parent
    before = hashes(parent_root)
    args = child_arguments(partial_parent)
    if fault == "default":
        args.pop("partial_parent_continuation")
    elif fault in ("clock", "termination"):
        args["partial_parent_continuation"]["effective_update" if fault == "clock" else "termination"] = (
            17 if fault == "clock" else "cpu_limit")
    elif fault in ("future18", "future19", "past"):
        index = {"future18": 18, "future19": 19, "past": 0}[fault]
        changed = list(schedule(20))
        changed[index] = {**changed[index], "row_indices": [99]}
        args["training_schedule"] = tuple(changed)
    elif fault == "short_horizon":
        args.update(updates=1, training_schedule=schedule(19), postproposal=guard(19))
    elif fault == "data":
        args["data_binding"] = {**args["data_binding"], "fixed_data": [99.]}
    elif fault == "guard":
        args["postproposal"] = guard(20, margin_fraction=.2)
    elif fault == "slots":
        args["continuation_checkpoint"] = replace(parent, optimizer_state={**parent.optimizer_state,
            "first_moment": [0.] * len(parent.optimizer_state["first_moment"])})
    elif fault == "rng":
        args["continuation_checkpoint"] = replace(parent, rng_state={**parent.rng_state, "next_seed_index": 17})
    elif fault == "preference":
        args["continuation_checkpoint"] = replace(parent, method_state={**parent.method_state, "preferred": "mgda"})
    elif fault in ("method_old", "method_clock"):
        args["method_policy_transition"] = {"effective_update": 17 if fault == "method_clock" else 18,
            "old_preferred": "pcgrad" if fault == "method_old" else "cagrad", "new_preferred": "mgda",
            "reason": "Invalid manufactured method transition"}
    elif fault == "fake_complete":
        args["continuation_binding"]["parent_result"] = finite._write_json(tmp_path / "fake-result.json",
            {"training_completed": True, "final_state": parent.to_dict()})
    else:
        args["continuation_binding"]["parent_refusal"] = finite._write_json(tmp_path / "budget-result.json",
            {"status": "CPU_LIMIT", "update_index": 18})
    with pytest.raises((ValueError, KeyError)):
        finite.build_runner(tmp_path / "rejected", **args)
    assert hashes(parent_root) == before


@pytest.mark.parametrize("fault", ["marker", "object", "control", "intent", "refusal_clock", "refusal_contract"])
def test_store_evidence_refusals_preserve_original_bytes(partial_parent, tmp_path, fault):
    parent, _args, runtime, binding = partial_parent
    args = child_arguments(partial_parent)
    before = hashes(runtime)
    if fault == "marker":
        path, content = finite._checked(binding["parent_commit_marker"]), None
    elif fault == "object":
        payload = json.loads(finite._checked(binding["parent_checkpoint"]).read_text())
        object_hash = payload["state"]["metadata"]["permanent_pass_rotation"]["history"]["object_hash"]
        path, content = runtime / "checkpoints/objects" / f"{object_hash}.json", b"[]"
    elif fault == "control":
        path, content = finite._checked(binding["parent_control_archive"][0]), b"{}"
    elif fault == "intent":
        path = runtime / "checkpoints/arms/arm-0/checkpoints/.update-00000019.json.publication.json"
        content = b"{}"
    else:
        path = finite._checked(binding["parent_refusal"])
        receipt = json.loads(path.read_text())
        if fault == "refusal_clock":
            receipt["update_index"] = 17
        else:
            receipt["event"]["postproposal"]["failure_kind"] = "contract_error"
        receipt["receipt_hash"] = stable_hash({key: value for key, value in receipt.items() if key != "receipt_hash"})
        content = (json.dumps(receipt) + "\n").encode()
    with changed_file(path, content):
        if fault.startswith("refusal_"):
            args["continuation_binding"]["parent_refusal"] = finite._reference(path)
        with pytest.raises((ValueError, OSError)):
            finite.build_runner(tmp_path / "invalid", **args)
    assert hashes(runtime) == before
    assert parent.update_index == 18


def test_newer_same_clock_boundary_is_not_the_bound_update(tmp_path):
    fixture = make_parent(tmp_path / "same-clock", boundary_every=2)
    with pytest.raises(ValueError, match="stale"):
        finite.build_runner(tmp_path / "child", **child_arguments(fixture))


def test_parent_advance_after_construction_refuses_before_control(tmp_path, monkeypatch):
    fixture = make_parent(tmp_path / "advance")
    parent, original, parent_root, _binding = fixture
    child, adapter, provider = finite.build_runner(tmp_path / "child", **child_arguments(fixture))
    old, old_adapter, old_provider = finite.build_runner(parent_root, **original)
    host_runtime(old, old_provider, monkeypatch)
    batch = old.batch_factory(0, PolicyView.from_dict(parent.policy_state), old.stages[0], 18)
    advanced, event = old.executor.step(parent, batch, old_adapter)
    old.store.commit_update(0, parent, advanced, event)
    forbid_runtime(child, adapter, provider, monkeypatch)
    with pytest.raises(ValueError, match="stale"):
        child.run()


def test_interrupted_entry_and_update_recover_without_repetition(partial_parent, tmp_path, monkeypatch):
    args = child_arguments(partial_parent)
    runtime = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    host_runtime(runner, provider, monkeypatch)
    with monkeypatch.context() as patches:
        patches.setattr(provider, "control", forbidden)
        with pytest.raises(AssertionError, match="host construction/recovery"):
            runner.run()
    assert not (runtime / "checkpoints/arms/arm-0/checkpoints/update-00000019.complete.json").exists()
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    first = host_runtime(runner, provider, monkeypatch, fail_at=19)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    assert first["controls"] == [18, 19] and first["updates"] == [18]
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    remaining = host_runtime(runner, provider, monkeypatch)
    assert runner.run().states[0].update_index == 20
    assert remaining["controls"] == [20] and remaining["updates"] == [19]


def test_original_refusal_remains_effective(partial_parent, monkeypatch):
    _parent, original, runtime, _binding = partial_parent
    runner, adapter, provider = finite.build_runner(runtime, **original)
    before = hashes(runtime)
    forbid_runtime(runner, adapter, provider, monkeypatch)
    with pytest.raises(PermanentPassUpdateError, match="manufactured finite candidate refusal"):
        runner.run()
    assert hashes(runtime) == before
