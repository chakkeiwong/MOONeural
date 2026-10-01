"""Host-only checkpoint continuation fixtures; no objective/optimizer kernels.

Preflight: compare the same split schedule, including its fresh entry control,
across uninterrupted and interrupted child runs. Check clocks, retained slots,
method/RNG state, monotone membership, original controls and parent immutability.
Bad bindings veto construction. These checks support engineering only, never
scientific admission or equality to a run without the extra entry control.
Shared test/lint allowance:80 wall seconds including failures; GPUs hidden.
"""

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from tests.contracts.test_generic_finite_objective_runner import (
    arguments,
    forbidden,
    quadratic,
    quadratic_values,
    scheduled_quadratic,
)

from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    ControlEvaluation,
    PolicyView,
    ValidationEvaluation,
    stable_hash,
)


def schedule(count):
    return tuple({"update_index": index, "row_indices": [index % 3], "population_rows": 3}
                 for index in range(count))


def fixture_arguments(root, count):
    args = arguments(root, task_ids=tuple(f"task.{index}" for index in range(7)),
                     denominators=[1.] * 7, updates=count, boundary_every=1,
                     lifecycle_endpoint="validation-only", training_schedule=schedule(count), training_objective=forbidden)
    args["data_binding"]["training_objective_source"] = args["data_binding"]["objective_source"]
    return args


def host_runtime(runner, provider, monkeypatch, *, fail_at=None, entry_pass=False):
    calls = {"updates": [], "controls": [], "validation": [], "partitions": []}
    stage = runner.stages[0]
    coordinator = runner.executor.core.coordinator

    def metrics():
        return {task: .1 if index == 6 or entry_pass and index == 0 else 2.
                for index, task in enumerate(runner.design.task_ids)}

    def control(arm, policy, requested_stage, number):
        assert stage == requested_stage
        update = stage.start_update + (number - stage.from_round) * stage.updates_per_round
        calls["controls"].append(update)
        request = provider.roles.request_for("control", policy, stage_id=stage.stage_id, arm_id=arm,
                                             round_number=number, update_index=update)
        return ControlEvaluation(metrics(), raw_records={"fixture": "host control", "clock": update}, request=request)

    def validation(arm, policy, requested_stage, *, update_index):
        assert stage == requested_stage
        calls["validation"].append(update_index)
        request = provider.roles.request_for("validation", policy, stage_id=stage.stage_id, arm_id=arm,
                                             round_number=None, update_index=update_index)
        return ValidationEvaluation(metrics(), dict.fromkeys(runner.design.task_ids, 1),
            {"fixture": "host validation", "scheduled_role_registry_hash": provider.roles.binding_hash(),
             "role_bank_manifest_hash": request.metadata["role_bank_manifest_hash"]}, request=request)

    def step(state, batch, adapter):
        if state.update_index == fail_at:
            raise RuntimeError("manufactured interruption")
        assert batch.metadata["training_schedule_entry"]["update_index"] == state.update_index
        calls["updates"].append(state.update_index)
        calls["partitions"].append(coordinator.read(state).partition)
        policy = PolicyView.from_dict(state.policy_state)
        first = [value + 1. for value in state.optimizer_state["first_moment"]]
        second = [value + 2. for value in state.optimizer_state["second_moment"]]
        policy = replace(policy, values=tuple(value + slot * .01 for value, slot in zip(policy.values, first, strict=True)))
        next_state = replace(state, update_index=state.update_index + 1, policy_state=policy.to_dict(), policy_fingerprint=None,
            optimizer_state={**state.optimizer_state, "first_moment": first, "second_moment": second,
                             "iteration": state.optimizer_state["iteration"] + 1},
            method_state={**state.method_state, "event_count": state.method_state.get("event_count", 0) + 1,
                          "last_event": {"update_index": state.update_index}},
            rng_state={**state.rng_state, "next_seed_index": state.rng_state["next_seed_index"] + 1})
        return coordinator.commit(state, next_state), {"event": "update"}

    monkeypatch.setattr(provider, "control", control)
    monkeypatch.setattr(provider, "validation", validation)
    monkeypatch.setattr(provider, "certification", forbidden)
    monkeypatch.setattr(runner.executor, "step", step)
    return calls


@pytest.fixture
def parent_run(tmp_path, monkeypatch):
    args = fixture_arguments(tmp_path, 5)
    runtime = tmp_path / "parent"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    host_runtime(runner, provider, monkeypatch)
    result = runner.run()
    state = result.states[0]
    result_ref = finite._write_json(tmp_path / "parent-result.json", {"training_completed": True, "final_state": state.to_dict()})
    controls = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"]).controls
    binding = {"parent_result": result_ref, "parent_checkpoint": result_ref,
               "parent_configuration": finite._reference(runtime / "configuration.json"),
               "parent_control_archive": [finite._reference(runtime / "checkpoints/control_evidence" /
                   f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in controls]}
    child = {**args, "parameters": state.policy_state["values"], "updates": 15, "training_schedule": schedule(20),
             "stage_id": "continuation-5-to-20", "continuation_checkpoint": state, "continuation_binding": binding}
    return state, child, runtime


def numerical(state):
    return {key: state.to_dict()[key] for key in ("update_index", "policy_state", "optimizer_state", "method_state", "rng_state")}


def test_continuation_preserves_state_history_clock_and_zero_call_recovery(parent_run, tmp_path, monkeypatch):
    parent, args, parent_root = parent_run
    before = {str(path): path.read_bytes() for path in parent_root.rglob("*") if path.is_file()}
    output = tmp_path / "child"
    runner, adapter, provider = finite.build_runner(output, **args)
    assert numerical(runner.states[0]) == numerical(parent)
    assert runner.design.optimizer_binding["fresh_moments"] is False
    assert adapter.policy_metadata == parent.policy_state["metadata"]
    assert adapter.metadata.input_fingerprint != adapter.policy_metadata["finite_objective_input_fingerprint"]
    stage = runner.stages[0]
    assert (stage.start_update, stage.from_round, stage.first_round, stage.last_round, stage.stop_update) == (5, 5, 6, 20, 20)
    calls = host_runtime(runner, provider, monkeypatch)
    result = runner.run()
    assert calls["updates"] == list(range(5, 20))
    assert calls["controls"] == list(range(5, 21)) and calls["validation"] == [20]
    assert all("task.6" in partition["constraints"] for partition in calls["partitions"])
    prior = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"])
    rotation = runner.executor.core.coordinator.read(result.states[0])
    assert rotation.controls[:6] == prior.controls and len(rotation.controls) == 22
    assert rotation.controls[6].stage_id == stage.stage_id and rotation.controls[6].update_index == 5
    assert calls["partitions"][0] == prior.partition
    assert rotation.permanent == prior.permanent
    assert result.states[0].optimizer_state["iteration"] == result.states[0].method_state["event_count"] == 20
    assert result.states[0].rng_state["next_seed_index"] == 20
    rebuilt, rebuilt_adapter, rebuilt_provider = finite.build_runner(output, **args)
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(rebuilt_provider, name, forbidden)
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    monkeypatch.setattr(rebuilt, "batch_factory", forbidden)
    monkeypatch.setattr(rebuilt_adapter, "evaluate", forbidden)
    recovered = rebuilt.run()
    assert recovered.states[0].to_dict() == result.states[0].to_dict()
    assert recovered.selection == result.selection and recovered.evaluations == result.evaluations
    assert before == {str(path): path.read_bytes() for path in parent_root.rglob("*") if path.is_file()}


def test_interrupted_child_matches_same_split_schedule(parent_run, tmp_path, monkeypatch):
    _parent, args, _root = parent_run
    uninterrupted, _adapter, provider = finite.build_runner(tmp_path / "uninterrupted", **args)
    host_runtime(uninterrupted, provider, monkeypatch)
    expected = uninterrupted.run()
    output = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(output, **args)
    calls = host_runtime(runner, provider, monkeypatch, fail_at=7)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    assert calls["controls"] == [5, 6, 7]
    resumed, _adapter, provider = finite.build_runner(output, **args)
    calls = host_runtime(resumed, provider, monkeypatch)
    actual = resumed.run()
    assert calls["updates"] == list(range(7, 20)) and calls["controls"] == list(range(8, 21))
    assert numerical(actual.states[0]) == numerical(expected.states[0])
    assert resumed.executor.core.coordinator.read(actual.states[0]).permanent == ("task.6",)


def test_entry_control_can_grow_membership_without_reset(parent_run, tmp_path, monkeypatch):
    parent, args, _root = parent_run
    runner, _adapter, provider = finite.build_runner(tmp_path / "growth", **args)
    host_runtime(runner, provider, monkeypatch, entry_pass=True)
    states, _events = runner._initialize(runner.stages[0])
    assert numerical(states[0]) == numerical(parent)
    rotation = runner.executor.core.coordinator.read(states[0])
    assert rotation.permanent == ("task.0", "task.6")
    assert set(rotation.partition["constraints"]) >= {"task.0", "task.6"}
    assert len(rotation.controls) == 7 and rotation.update_index == 5


@pytest.mark.parametrize("fault", ["prefix", "clock", "scale", "policy", "source", "archive", "history", "data", "stage"])
def test_invalid_parent_or_continuation_refused(parent_run, tmp_path, fault):
    parent, original, root = parent_run
    args = dict(original)
    if fault == "prefix":
        args["training_schedule"] = ({**schedule(20)[0], "row_indices": [99]}, *schedule(20)[1:])
    elif fault == "clock":
        args["continuation_checkpoint"] = replace(parent, update_index=4)
    elif fault == "scale":
        args["denominators"] = [2.] * 7
    elif fault == "policy":
        args["parameters"] = [0., 0.]
    elif fault == "source":
        (root / "source/training/generic_replication_runner.py").write_text("changed snapshot")
    elif fault == "archive":
        reference = args["continuation_binding"]["parent_control_archive"][0]
        finite._checked(reference).write_text("changed full control")
    elif fault == "history":
        payload = parent.to_dict()
        payload.pop("checkpoint_state_hash")
        payload["metadata"]["permanent_pass_rotation"]["history"][0]["control"]["task_upper_mse"]["task.0"] = .1
        with pytest.raises(ValueError):
            args["continuation_checkpoint"] = CheckpointState.from_dict(payload)
            finite.build_runner(tmp_path / fault, **args)
        return
    elif fault == "data":
        args["data_binding"] = {**args["data_binding"], "fixed_data": [8., 9.]}
    else:
        args["stage_id"] = finite.STAGE_ID
    with pytest.raises((ValueError, KeyError)):
        finite.build_runner(tmp_path / fault, **args)


def test_cached_child_still_verifies_imported_controls(parent_run, tmp_path, monkeypatch):
    _parent, args, _root = parent_run
    output = tmp_path / "child"
    runner, _adapter, provider = finite.build_runner(output, **args)
    host_runtime(runner, provider, monkeypatch)
    runner.run()
    first = args["continuation_binding"]["parent_control_archive"][0]
    (output / "checkpoints/control_evidence" / f'{first["sha256"]}.json').write_text("bad copy")
    with pytest.raises(ValueError, match="hash mismatch"):
        finite.build_runner(output, **args)


def test_history_requests_cannot_be_rebound_to_child_registry(parent_run, tmp_path):
    _parent, args, _root = parent_run
    runner, _adapter, _provider = finite.build_runner(tmp_path / "child", **args)
    checkpoint = runner.states[0]
    payload = json.loads(finite.canonical_json(checkpoint.metadata))
    payload["continuation_role_manifests"] = {}
    with pytest.raises(ValueError, match="historical registries"):
        runner.executor.core.coordinator.read(replace(checkpoint, metadata=payload))
    assert stable_hash(checkpoint.to_dict()) == runner.design.initial_state_hashes["0"]


@pytest.fixture
def role_root():
    directory = finite.ROOT / ".pytest_cache"
    directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="checkpoint-roles-", dir=directory) as temporary:
        yield Path(temporary)


def test_injected_child_uses_origin_policy_with_new_banks(role_root, monkeypatch):
    from tests.contracts.test_generic_live_role_lifecycle import host_updates, injected_arguments

    tmp_path = role_root
    args, _factory = injected_arguments(tmp_path / "parent-inputs")
    root = tmp_path / "parent"
    runner, _adapter, provider = finite.build_runner(root, **args)
    host_updates(runner, monkeypatch)
    parent = runner.run().states[0]
    result_ref = finite._write_json(tmp_path / "parent-result.json", {"training_completed": True, "final_state": parent.to_dict()})
    controls = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"]).controls
    child, _factory = injected_arguments(tmp_path / "child-inputs", updates=2)
    child.update(parameters=parent.policy_state["values"], stage_id="injected-continuation", continuation_checkpoint=parent,
        continuation_binding={"parent_result": result_ref, "parent_checkpoint": result_ref,
            "parent_configuration": finite._reference(root / "configuration.json"),
            "parent_control_archive": [finite._reference(root / "checkpoints/control_evidence" /
                f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in controls]})
    continued, adapter, provider = finite.build_runner(tmp_path / "child", **child)
    assert numerical(continued.states[0]) == numerical(parent)
    assert provider.bridge.provider_calls == 0
    assert adapter.policy_metadata == parent.policy_state["metadata"]
    calls = host_updates(continued, monkeypatch)
    result = continued.run()
    assert calls == [1, 2] and result.states[0].update_index == 3
    assert provider.bridge.provider_calls == 8
    rebuilt, _adapter, provider = finite.build_runner(tmp_path / "child", **child)
    provider.bridge.row_provider.forbidden = True
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    recovered = rebuilt.run()
    assert recovered.states[0].to_dict() == result.states[0].to_dict()
    assert provider.bridge.provider_calls == provider.bridge.estimator_calls == 0


def test_saved_fallback_rates_are_retained(parent_run, tmp_path):
    parent, original, _root = parent_run
    rates = {method: rate / 4. for method, rate in parent.method_state["rates"].items()}
    parent = replace(parent, optimizer_state={**parent.optimizer_state, "learning_rate": .0025},
                     method_state={**parent.method_state, "rates": rates})
    reference = finite._write_json(tmp_path / "fallback-result.json", {"training_completed": True, "final_state": parent.to_dict()})
    binding = {**original["continuation_binding"], "parent_result": reference, "parent_checkpoint": reference}
    args = {**original, "continuation_checkpoint": parent, "continuation_binding": binding}
    runner, _adapter, _provider = finite.build_runner(tmp_path / "fallback-child", **args)
    assert numerical(runner.states[0]) == numerical(parent)
    assert runner.states[0].optimizer_state["learning_rate"] == .0025
    assert runner.states[0].method_state["rates"] == rates
    assert runner.design.optimizer_binding["rate"] == .01


@pytest.mark.skipif(os.environ.get("GENERIC_CHECKPOINT_CONTINUATION_NUMERICAL") != "1",
                    reason="bounded tiny CAGrad/Adam fixture opt-in required")
def test_actual_cagrad_adam_same_split_schedule_recovery(tmp_path, monkeypatch):
    args = arguments(tmp_path, objective=quadratic, objective_values=quadratic_values,
                     updates=1, boundary_every=1, lifecycle_endpoint="validation-only",
                     training_objective=scheduled_quadratic, training_schedule=schedule(1))
    args["data_binding"]["training_objective_source"] = args["data_binding"]["objective_source"]
    root = tmp_path / "actual-parent"
    runner, _adapter, _provider = finite.build_runner(root, **args)
    parent = runner.run().states[0]
    result_ref = finite._write_json(tmp_path / "actual-parent-result.json", {"training_completed": True, "final_state": parent.to_dict()})
    controls = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"]).controls
    child = {**args, "updates": 2, "training_schedule": schedule(3), "parameters": parent.policy_state["values"],
        "stage_id": "actual-continuation", "continuation_checkpoint": parent,
        "continuation_binding": {"parent_result": result_ref, "parent_checkpoint": result_ref,
            "parent_configuration": finite._reference(root / "configuration.json"),
            "parent_control_archive": [finite._reference(root / "checkpoints/control_evidence" /
                f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in controls]}}
    complete, _adapter, _provider = finite.build_runner(tmp_path / "complete", **child)
    expected = complete.run()
    interrupted, _adapter, _provider = finite.build_runner(tmp_path / "interrupted", **child)
    original_step = interrupted.executor.step

    def interrupt(state, batch, adapter):
        if state.update_index == 2:
            raise RuntimeError("actual split interruption")
        return original_step(state, batch, adapter)

    monkeypatch.setattr(interrupted.executor, "step", interrupt)
    with pytest.raises(RuntimeError, match="actual split interruption"):
        interrupted.run()
    continued, _adapter, _provider = finite.build_runner(tmp_path / "interrupted", **child)
    actual = continued.run()
    assert numerical(actual.states[0]) == numerical(expected.states[0])
    assert actual.states[0].optimizer_state["iteration"] == actual.states[0].update_index == 3
    assert actual.states[0].method_state["preferred"] == "cagrad"
    for completed_runner, result in ((complete, expected), (continued, actual)):
        rotation = completed_runner.executor.core.coordinator.read(result.states[0])
        assert [point.update_index for point in rotation.controls] == [0, 1, 1, 2, 3]
        assert rotation.controls[2].stage_id == "actual-continuation"
    recovered, adapter, provider = finite.build_runner(tmp_path / "interrupted", **child)
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    monkeypatch.setattr(adapter, "compute_task_values_and_gradients", forbidden)
    monkeypatch.setattr(recovered.executor, "step", forbidden)
    assert recovered.run().states[0].to_dict() == actual.states[0].to_dict()
