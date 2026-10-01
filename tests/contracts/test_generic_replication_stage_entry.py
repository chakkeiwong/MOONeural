"""Source stage-entry promotion and same-update recovery contracts."""

import json
from dataclasses import replace

import pytest
from tests.contracts.test_generic_permanent_pass import (
    POLICY,
    REGISTRY,
    ROLES,
    checkpoint,
    control,
    initial,
)
from tests.contracts.test_generic_replication_runner import (
    FakeCoordinator,
    FakeEvaluationProvider,
    FakeExecutor,
    design_for,
    runner,
)
from tests.contracts.test_generic_replication_runner import (
    checkpoint as runner_checkpoint,
)

from mooneural.training.generic_permanent_pass import (
    POLICY_SCHEMA,
    STAGE_POLICY_SCHEMA,
    PermanentPassState,
)
from mooneural.training.generic_replication_policy import (
    ReplicationPermanentPassCoordinator,
    ReplicationRoleManifest,
)
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    ReplicationRunnerError,
    ReplicationStage,
)
from mooneural.training.generic_role_banks import RoleBank, RoleBankManifest
from mooneural.training.generic_scheduled_role_banks import (
    ScheduledRoleBankRegistry,
    ScheduledRoleBinding,
)
from mooneural.training.generic_training_contracts import stable_hash
from tests.support.working_set import (
    initialize_permanent_pass_state,
    promote_permanent_passes,
)


@pytest.mark.parametrize("update_index", (33000, 39000, 99000))
@pytest.mark.parametrize("unresolved_value", (0.039, 0.041))
def test_stage_entry_preserves_source_membership_and_promotes_once(update_index, unresolved_value):
    before = initial([0.03] * 6 + [0.1], update_index - 1)
    before = before.committed_update().boundary(control([0.2] * 7), POLICY, ROLES)
    legacy = before.to_dict()
    assert legacy["schema"] == POLICY_SCHEMA
    assert PermanentPassState.from_dict(legacy).to_dict() == legacy
    values = [0.2] * 6 + [unresolved_value]
    entered = before.stage_entry(control(values), POLICY, ROLES, stage_id=f"stage-{update_index}")
    source, _event = initialize_permanent_pass_state(dict(zip(REGISTRY.task_ids, [0.03] * 6 + [0.1], strict=True)))
    promoted, _event = promote_permanent_passes(source, dict(zip(REGISTRY.task_ids, values, strict=True)))
    assert entered.permanent == promoted.permanent
    assert entered.update_index == update_index
    assert entered.history()[:-1] == before.history()
    assert entered.history()[-1]["regressed_permanent_diagnostic"] == list(REGISTRY.task_ids[:6])
    assert entered.history()[-1]["previous_control_sha256"] == stable_hash(before.history()[-1])
    assert entered.to_dict()["schema"] == STAGE_POLICY_SCHEMA
    assert PermanentPassState.from_dict(entered.to_dict()).to_dict() == entered.to_dict()
    assert entered.complete == (unresolved_value <= 0.04)
    if entered.complete:
        with pytest.raises(ValueError, match="no optimizer update"):
            entered.committed_update()
    with pytest.raises(ValueError):
        entered.stage_entry(control(values), POLICY, ROLES, stage_id=f"stage-{update_index}")
    with pytest.raises(ValueError, match="new updates"):
        entered.boundary(control(values), POLICY, ROLES)


def test_stage_entry_rejects_uncommitted_boundary_and_reused_identity():
    before = initial([0.2] * 7, 33000)
    with pytest.raises(ValueError, match="completed boundary"):
        before.committed_update().stage_entry(control([0.2] * 7), POLICY, ROLES, stage_id="stage-1")
    entered = before.stage_entry(control([0.2] * 7), POLICY, ROLES, stage_id="stage-1")
    bounded = entered.committed_update().boundary(control([0.2] * 7), POLICY, ROLES)
    with pytest.raises(ValueError, match="distinct stage"):
        bounded.stage_entry(control([0.2] * 7), POLICY, ROLES, stage_id="stage-1")
    with pytest.raises(ValueError):
        before.stage_entry(control([0.2] * 7), POLICY, ROLES, stage_id=None)


@pytest.mark.parametrize("update_index", (33000, 39000, 99000))
@pytest.mark.parametrize("point", ("objects", "payload", "marker"))
def test_stage_entry_interruption_restores_exact_parent_and_selection_cutoff(
    tmp_path, monkeypatch, update_index, point
):
    coordinator = ReplicationPermanentPassCoordinator(REGISTRY, ReplicationRoleManifest(ROLES), threshold=0.04)
    parent = coordinator.initialize(checkpoint(update_index - 1), control([0.03] * 6 + [0.1]))
    committed = coordinator.commit(parent, replace(parent, update_index=update_index))
    bounded, _event = coordinator.boundary(committed, control([0.2] * 7))
    entered, _event = coordinator.stage_entry(bounded, control([0.2] * 6 + [0.039]), stage_id=f"stage-{update_index}")
    store = AtomicCheckpointStore(tmp_path, parent.attempt_id)
    store.initialize(4, parent)
    store.commit_update(4, parent, committed, {"event": "update"})
    store.commit_boundary(4, bounded, {"event": "boundary"})
    event = {
        "event": "stage_entry", "stage_id": f"stage-{update_index}",
        "update_index": update_index, "parent_state_sha256": stable_hash(bounded.to_dict()),
        "stage_ordinal": 1, "arm_id": 4, "round": update_index // 300, "boundary": 0,
        "control_sha256": stable_hash(control([0.2] * 6 + [0.039]).to_dict()),
    }
    original_write = store._write_publication_json

    def interrupted_write(path, payload):
        original_write(path, payload)
        if (
            (point == "objects" and path.parent.name == "objects")
            or (point == "payload" and path.name == f"stage_entry-{update_index:08d}.json")
            or (point == "marker" and path.name == f"stage_entry-{update_index:08d}.complete.json")
        ):
            raise RuntimeError("entry publication interruption")

    monkeypatch.setattr(store, "_write_publication_json", interrupted_write)
    with pytest.raises(RuntimeError, match="entry publication interruption"):
        store.commit_stage_entry(4, bounded, entered, event)
    recovered_store = AtomicCheckpointStore(tmp_path, parent.attempt_id)
    recovered, events = recovered_store.recover(4)
    if point == "objects":
        assert recovered.to_dict() == bounded.to_dict()
        recovered_store.commit_stage_entry(4, bounded, entered, event)
        recovered, events = recovered_store.recover(4)
    assert recovered.to_dict() == entered.to_dict()
    assert [item["event"] for item in events] == ["initialization", "update", "boundary", "stage_entry"]
    assert coordinator.read(recovered).complete
    cutoff, _events = recovered_store.recover(4, through_update_index=update_index, include_stage_entry_at_cutoff=False)
    assert cutoff.to_dict() == bounded.to_dict()
    assert replace(entered, metadata=bounded.metadata).to_dict() == bounded.to_dict()
    with pytest.raises(FileExistsError):
        recovered_store.commit_stage_entry(4, bounded, entered, event)
    with pytest.raises(ReplicationRunnerError, match="parent state"):
        recovered_store.commit_stage_entry(4, bounded, entered, {**event, "parent_state_sha256": "0" * 64})


def test_runner_commits_completing_entry_before_stop_and_does_not_replay(tmp_path):
    class CompletingProvider(FakeEvaluationProvider):
        def control(self, arm_id, policy, stage, round_number):
            result = super().control(arm_id, policy, stage, round_number)
            if stage.stage_id == "continuation":
                result = replace(result, task_upper_mse={task: 0.03 for task in result.task_upper_mse})
            return result

    executor = FakeExecutor()
    current, store, batches = runner(tmp_path, executor=executor, provider=CompletingProvider())
    result = current.run()
    assert executor.calls == 20
    assert result.states[4].update_index == 4
    assert all(stage == "population" for _arm, stage, _update in batches)
    assert result.events[-1]["event"] == "stage_entry"
    resumed, _store, resumed_batches = runner(tmp_path, executor=executor, store=store)
    restored = resumed.run()
    assert restored.selection == result.selection
    assert restored.states[4].to_dict() == result.states[4].to_dict()
    assert executor.calls == 20
    assert not resumed_batches


def test_selection_provider_receives_actual_early_completion_update(tmp_path):
    class EarlyCoordinator(FakeCoordinator):
        def boundary(self, state, control):
            bounded, event = super().boundary(state, control)
            return replace(bounded, metadata={**bounded.metadata, "fake_complete": True}), event

    class RecordingProvider(FakeEvaluationProvider):
        def validation(self, arm_id, policy, stage, *, update_index):
            actual_updates.append((arm_id, update_index))
            return super().validation(arm_id, policy, stage, update_index=update_index)

    actual_updates = []
    executor = FakeExecutor()
    executor.core.coordinator = EarlyCoordinator()
    current, _store, _batches = runner(tmp_path, executor=executor, provider=RecordingProvider())
    result = current.run()
    assert actual_updates == [(arm_id, 2) for arm_id in range(5)]
    assert all(state.update_index == 2 for state in result.states.values())
    assert all(event["event"] != "stage_entry" for event in result.events)


def test_runner_refuses_continuation_history_without_entry(tmp_path):
    current, store, _batches = runner(tmp_path)
    current.run()
    directory = store.root / "arms/arm-4/stage_entries"
    for path in directory.iterdir():
        path.unlink()
    resumed, _store, _batches = runner(tmp_path, store=store)
    with pytest.raises(ReplicationRunnerError, match="lack their stage-entry"):
        resumed.run()


@pytest.mark.parametrize("wrong", ("membership", "method"))
def test_runner_refuses_wrong_inherited_stage_state_before_control(tmp_path, wrong):
    parents = {arm: runner_checkpoint(arm) for arm in range(5)}
    design = design_for(parents)
    requirements = {
        "permanent_tasks": ["task-a"] if wrong == "membership" else [],
        "preferred_method": "wrong" if wrong == "method" else "fake",
    }
    design = replace(design, permanent_pass_binding={
        **design.permanent_pass_binding,
        "stage_entries": {"continuation": requirements},
    })
    current, store, batches = runner(tmp_path, design=design, initial_states=parents)
    with pytest.raises(ReplicationRunnerError, match="inherited stage-entry"):
        current.run()
    assert current.executor.calls == 20
    assert current.evaluation_provider.control_calls == 15
    assert all(stage == "population" for _arm, stage, _index in batches)
    assert not list((store.root / "arms/arm-4/stage_entries").glob("*.complete.json"))


def test_stage_entry_state_schema_cannot_be_relabelled_as_legacy():
    state = initial([0.2] * 7, 33000).stage_entry(control([0.2] * 7), POLICY, ROLES, stage_id="new")
    payload = json.loads(json.dumps(state.to_dict()))
    payload["schema"] = POLICY_SCHEMA
    payload["state_hash"] = stable_hash({key: value for key, value in payload.items() if key != "state_hash"})
    with pytest.raises(ValueError, match="inconsistent"):
        PermanentPassState.from_dict(payload)


def scheduled_control_roles():
    bindings = []
    for index, (stage_id, round_number, update_index) in enumerate((
        ("previous", 0, 32999), ("previous", 1, 33000), ("next", 1, 33000),
    )):
        bank = RoleBank(
            "control", f"synthetic-bank-{index}", 22 + index,
            "synthetic-target", "v1", "v1", "v1",
            {"global": stable_hash([index, "global"])}, (f"synthetic-sample-{index}",),
        )
        manifest = RoleBankManifest(REGISTRY.task_ids, (bank,), {"control": 1})
        bindings.append(ScheduledRoleBinding(
            stage_id, 4, "control", round_number, update_index, update_index, manifest,
        ))
    return ScheduledRoleBankRegistry(tuple(bindings))


def test_coordinator_keeps_each_historical_control_bound_to_its_own_stage():
    roles = scheduled_control_roles()
    coordinator = ReplicationPermanentPassCoordinator(REGISTRY, roles, threshold=0.04)

    def bound_control(stage_id, round_number, update_index):
        return replace(control([0.03] * 6 + [0.1]), request=roles.request_for(
            "control", POLICY, stage_id=stage_id, arm_id=4,
            round_number=round_number, update_index=update_index,
        ))

    start = coordinator.initialize(checkpoint(32999), bound_control("previous", 0, 32999))
    updated = coordinator.commit(start, replace(start, update_index=33000))
    with pytest.raises(ValueError, match="update_index"):
        coordinator.boundary(updated, bound_control("previous", 0, 32999))
    bounded, _event = coordinator.boundary(updated, bound_control("previous", 1, 33000))
    next_control = bound_control("next", 1, 33000)
    with pytest.raises(ValueError, match="scope|coordinates"):
        coordinator.stage_entry(bounded, next_control, stage_id="previous")
    entered, _event = coordinator.stage_entry(bounded, next_control, stage_id="next")
    restored = coordinator.read(entered)
    assert [point.evaluation.request.seeds for point in restored.controls] == [(22,), (23,), (24,)]
    assert [point.evaluation.request.metadata["scheduled_role_scope"]["stage_id"] for point in restored.controls] == ["previous", "previous", "next"]
    assert entered.metadata["role_manifest"] == start.metadata["role_manifest"]


@pytest.mark.parametrize("wrong", ("arm", "stage", "round", "update"))
def test_runner_rejects_valid_control_for_wrong_current_coordinate(tmp_path, wrong):
    roles = scheduled_control_roles()
    current, _store, _batches = runner(tmp_path)
    current.executor.core.coordinator.roles = roles
    request = roles.request_for("control", POLICY, stage_id="next", arm_id=4, round_number=1, update_index=33000)
    evaluation = replace(control([0.2] * 7), request=request)
    stage = ReplicationStage("wrong" if wrong == "stage" else "next", 1, 2, 2, 33000)
    with pytest.raises(ReplicationRunnerError, match="binding mismatch"):
        current._validate_control_coordinates(
            evaluation, 0 if wrong == "arm" else 4,
            checkpoint(33001 if wrong == "update" else 33000),
            stage, 2 if wrong == "round" else 1,
        )
