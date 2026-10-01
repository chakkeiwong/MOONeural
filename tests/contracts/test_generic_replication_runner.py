"""Crash-safe staged replication runner contract tests."""

import hashlib
import json
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mooneural.artifacts import atomic_write_json
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_replication_design import ReplicationDesignBinding
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    GenericReplicationRunner,
    ReplicationRunnerError,
    ReplicationStage,
)
from mooneural.training.generic_training_contracts import (
    CertificationResult,
    CheckpointState,
    ControlEvaluation,
    EvaluationRequest,
    PolicyView,
    TaskDefinition,
    TaskRegistry,
    ValidationEvaluation,
    stable_hash,
)

TASK_IDS = ("task-a", "task-b")


def checkpoint(arm_id=0, update_index=0):
    policy = PolicyView((float(arm_id),), f"policy-{arm_id}")
    return CheckpointState(
        "runner-test",
        update_index,
        policy.to_dict(),
        {"iteration": update_index},
        {"preferred": "fake", "rates": {"fake": 1.0}},
        {"seed": arm_id},
        metadata={"arm_id": arm_id},
    )


class FakeCoordinator:
    def __init__(self, role_hash=None):
        self.roles = BoundRoles(role_hash)

    def read(self, state):
        policy = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"])
        return SimpleNamespace(
            complete=state.metadata.get("fake_complete", False) or policy.complete,
            permanent=policy.permanent,
            partition={"active": list(TASK_IDS), "constraints": []},
        )

    def boundary(self, state, control):
        boundaries = list(state.metadata.get("boundaries", ()))
        boundaries.append(control.request.metadata["round"])
        current = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"])
        policy = current.boundary(control, PolicyView.from_dict(state.policy_state), self.roles)
        return replace(
            state,
            metadata={**state.metadata, "boundaries": boundaries, "permanent_pass_rotation": policy.to_dict()},
        ), {"entered_permanent": []}

    def stage_entry(self, state, control, *, stage_id):
        current = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"])
        policy = current.stage_entry(control, PolicyView.from_dict(state.policy_state), self.roles, stage_id=stage_id)
        return replace(state, metadata={**state.metadata, "permanent_pass_rotation": policy.to_dict()}), {}


class BoundRoles:
    def __init__(self, role_hash):
        self.role_hash = role_hash

    def binding_hash(self):
        return self.role_hash

    def validate_request(self, request, policy):
        if request.role not in {"control", "validation", "certification"}:
            raise ValueError("unexpected fake role")
        if request.policy_fingerprint != policy.fingerprint():
            raise ValueError("stale fake request")
        if tuple(request.task_ids) != TASK_IDS:
            raise ValueError("fake task order mismatch")


class FakeExecutor:
    def __init__(self, fail_on_call=None, role_hash=None):
        self.core = SimpleNamespace(coordinator=FakeCoordinator(role_hash))
        self.metadata = SimpleNamespace(registry=SimpleNamespace(task_ids=TASK_IDS))
        self.calls = 0
        self.fail_on_call = fail_on_call

    def initialize(self, state, control):
        registry = TaskRegistry(tuple(TaskDefinition(task, index) for index, task in enumerate(TASK_IDS)))
        policy = PermanentPassState.initialize(
            registry, control, PolicyView.from_dict(state.policy_state), self.core.coordinator.roles,
            threshold=0.04, update_index=state.update_index,
        )
        return replace(state, metadata={**state.metadata, "initialized": True, "permanent_pass_rotation": policy.to_dict()})

    def step(self, state, _batch, _adapter):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("synthetic interruption")
        policy = PolicyView.from_dict(state.policy_state)
        next_policy = replace(policy, values=(policy.values[0] + 0.5,))
        rotation = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"]).committed_update()
        return (
            replace(
                state,
                update_index=state.update_index + 1,
                policy_state=next_policy.to_dict(),
                optimizer_state={"iteration": state.update_index + 1},
                policy_fingerprint=None,
                metadata={**state.metadata, "permanent_pass_rotation": rotation.to_dict()},
            ),
            {"event": "update", "arm_id": state.metadata["arm_id"]},
        )


class FakeEvaluationProvider:
    def __init__(self, selected=4, fail_on_control=None, role_hash=None):
        self.selected = selected
        self.control_calls = 0
        self.fail_on_control = fail_on_control
        self.role_hash = role_hash

    def control(self, arm_id, policy, stage, round_number):
        self.control_calls += 1
        if self.fail_on_control == self.control_calls:
            raise RuntimeError("synthetic boundary interruption")
        return ControlEvaluation(
            OrderedDict((task, 0.2) for task in TASK_IDS),
            raw_records={"arm": arm_id, "round": round_number, "stage": stage.stage_id},
            request=EvaluationRequest(
                "control", "synthetic-control", seeds=(22,),
                task_ids=TASK_IDS, policy_fingerprint=policy.fingerprint(),
                metadata={"round": round_number},
            ),
        )

    def validation(self, arm_id, policy, stage, *, update_index):
        value = 0.01 if arm_id == self.selected else 0.02
        request = None
        provenance = {"stage": stage.stage_id}
        if self.role_hash is not None:
            request = EvaluationRequest(
                role="validation",
                target_id="fake-validation-target",
                seeds=(1000 + arm_id,),
                sample_count=1,
                estimator_version="fake-v1",
                target_version="fake-v1",
                scale_version="fake-v1",
                policy_fingerprint=policy.fingerprint(),
                task_ids=TASK_IDS,
                anchor_ids=(f"validation-{arm_id}",),
            )
            provenance["role_bank_manifest_hash"] = self.role_hash
        return ValidationEvaluation(
            OrderedDict((task, value) for task in TASK_IDS),
            {task: 2 for task in TASK_IDS},
            provenance=provenance,
            request=request,
        )

    def certification(self, arm_id, policy, stage, *, update_index):
        validation = FakeEvaluationProvider.validation(self, arm_id, policy, stage, update_index=update_index)
        request = None if validation.request is None else replace(
            validation.request, role="certification", seeds=(2000 + arm_id,),
            anchor_ids=(f"certification-{arm_id}",),
        )
        return CertificationResult(
            {"retained-miss": False}, {"retained-miss": {"upper_normalized_mse": 0.0537}},
            {"synthetic_integrity": True}, validation.provenance, request=request,
        )


def stages():
    return (
        ReplicationStage("population", 0, 1, 2, 0, updates_per_round=2, population_size=5, select_after=True),
        ReplicationStage("continuation", 2, 3, 3, 4, updates_per_round=2, population_size=1),
    )


@pytest.mark.parametrize("rejected_arm", [0, 3])
def test_optional_rejection_is_durable_and_recovery_skips_numerics(tmp_path, monkeypatch, rejected_arm):
    from mooneural.training.generic_permanent_pass_executor import (
        PermanentPassUpdateError,
    )

    class RejectingExecutor(FakeExecutor):
        def step(self, state, batch, adapter):
            if state.metadata["arm_id"] != rejected_arm:
                return super().step(state, batch, adapter)
            self.calls += 1
            raise PermanentPassUpdateError("finite candidate budget exhausted", state,
                {"event": "update", "committed": False, "transaction_rolled_back": True,
                 "postproposal": {"accepted": False, "loss_calls": 5,
                                  "candidates": [{"fraction": .125, "accepted": False}]}})

    executor = RejectingExecutor()
    original, _executor, _provider = runner(tmp_path, executor=executor)
    with pytest.raises(PermanentPassUpdateError) as rejected:
        original.run()
    assert executor.calls == rejected_arm + 1
    paths = list(original.store.root.glob("rejections/arm-*/*.json"))
    assert len(paths) == 1
    receipt = json.loads(paths[0].read_text())
    assert receipt["event"] == rejected.value.event
    assert receipt["parent_state_hash"] == stable_hash(rejected.value.checkpoint.to_dict())
    parent, _events = original.store.recover(rejected_arm)
    assert parent.to_dict() == rejected.value.checkpoint.to_dict()
    assert parent.update_index == 0
    rebuilt, _executor, _provider = runner(tmp_path, executor=executor)

    def forbidden(*arguments):
        raise AssertionError("recorded rejection recovery repeated numerical work")

    monkeypatch.setattr(rebuilt, "batch_factory", forbidden)
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    with pytest.raises(PermanentPassUpdateError, match="finite candidate budget exhausted") as replay:
        rebuilt.run()
    assert replay.value.checkpoint.to_dict() == parent.to_dict()
    assert replay.value.event == rejected.value.event
    assert json.loads(paths[0].read_text()) == receipt


def test_optional_rejection_receipt_corruption_refuses(tmp_path):
    from mooneural.training.generic_permanent_pass_executor import (
        PermanentPassUpdateError,
    )

    instance, _executor, _provider = runner(tmp_path)
    state = instance.states[0]
    error = PermanentPassUpdateError("failed", state, {"postproposal": {"loss_calls": 5}})
    path = instance.store.record_update_rejection(0, state, error)
    payload = json.loads(path.read_text())
    payload["event"]["postproposal"]["loss_calls"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ReplicationRunnerError, match="invalid optional rejection"):
        instance.store.read_update_rejection(0, state)


def test_optional_rejection_cannot_archive_a_changed_checkpoint(tmp_path):
    from mooneural.training.generic_permanent_pass_executor import (
        PermanentPassUpdateError,
    )

    instance, _executor, _provider = runner(tmp_path)
    state = instance.states[0]
    changed = replace(state, rng_state={"different": 1})
    with pytest.raises(ReplicationRunnerError, match="original checkpoint"):
        instance.store.record_update_rejection(0, state, PermanentPassUpdateError("bad rollback", changed, {}))


def design_for(initial):
    stage_bindings = tuple(
        {
            "stage_id": stage.stage_id,
            "from_round": stage.from_round,
            "first_round": stage.first_round,
            "last_round": stage.last_round,
            "start_update": stage.start_update,
            "updates_per_round": stage.updates_per_round,
            "population_size": stage.population_size,
            "select_after": stage.select_after,
        }
        for stage in stages()
    )
    return ReplicationDesignBinding(
        design_id="runner-design-v1",
        master_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        generic_code_hashes={"runner.py": "3" * 64},
        environment_binding={"python": "3.11", "runtime": {"threads": 1}},
        backend_binding={"backend": "python_fake", "dtype": "float64"},
        role_manifest_sha256="4" * 64,
        target_hashes={"validation": "5" * 64},
        initial_state_hashes={
            str(arm_id): stable_hash(state.to_dict())
            for arm_id, state in initial.items()
        },
        task_ids=TASK_IDS,
        threshold=0.04,
        replica_ids=tuple(sorted(initial)),
        expected_selected_replica=4,
        selection_key=("absolute_threshold_ratio", "replica"),
        stages=stage_bindings,
        optimizer_binding={"name": "fake-adam", "clip_norm": 10.0},
        permanent_pass_binding={"threshold": 0.04, "active_count": "min(3, unresolved_count)"},
        seed_registry_sha256="6" * 64,
    )


def runner(
    tmp_path,
    *,
    executor=None,
    provider=None,
    store=None,
    selected=4,
    threshold=0.04,
    design=None,
    initial_states=None,
):
    role_hash = None if design is None else design.role_manifest_sha256
    executor = executor or FakeExecutor(role_hash=role_hash)
    provider = provider or FakeEvaluationProvider(selected, role_hash=role_hash)
    store = store or AtomicCheckpointStore(tmp_path / "run", "runner-test", design=design)
    initial = initial_states or {arm_id: checkpoint(arm_id) for arm_id in range(5)}
    adapters = {arm_id: object() for arm_id in range(5)}
    batches = []

    def batch_factory(arm_id, _policy, stage, update_index):
        batches.append((arm_id, stage.stage_id, update_index))
        return object()

    return (
        GenericReplicationRunner(
            executor,
            adapters,
            initial,
            provider,
            batch_factory,
            store,
            stages(),
            expected_selected_replica=4,
            threshold=threshold,
            design=design,
        ),
        store,
        batches,
    )


def test_store_recovers_update_and_boundary_pair(tmp_path):
    store = AtomicCheckpointStore(tmp_path / "run", "runner-test")
    initial = checkpoint()
    store.initialize(0, initial)
    updated = replace(initial, update_index=1)
    store.commit_update(0, initial, updated, {"event": "update"})
    store.commit_boundary(0, updated, {"event": "boundary", "round": 1})

    recovered, events = store.recover(0)

    assert recovered.to_dict() == updated.to_dict()
    assert [event["event"] for event in events] == ["initialization", "update", "boundary"]


@pytest.mark.parametrize("orphan", ("checkpoint", "marker"))
def test_store_refuses_orphaned_update_pair(tmp_path, orphan):
    store = AtomicCheckpointStore(tmp_path / "run", "runner-test")
    initial = checkpoint()
    store.initialize(0, initial)
    updated = replace(initial, update_index=1)
    store.commit_update(0, initial, updated, {"event": "update"})
    if orphan == "checkpoint":
        (store.root / "arms/arm-0/checkpoints/update-00000001.complete.json").unlink()
        (store.root / "arms/arm-0/checkpoints/.update-00000001.json.publication.json").unlink()
    else:
        (store.root / "arms/arm-0/checkpoints/update-00000001.json").unlink()

    with pytest.raises(ReplicationRunnerError, match="orphaned"):
        store.recover(0)


def test_store_selection_is_immutable_and_recoverable(tmp_path):
    store = AtomicCheckpointStore(tmp_path / "run", "runner-test")
    selection = {"selected_replica": 4, "key": ["absolute_threshold_ratio", "replica"]}
    store.commit_selection(selection)

    assert store.read_selection() == selection
    with pytest.raises(FileExistsError):
        store.commit_selection(selection)


@pytest.mark.parametrize("kind", ("initial", "update", "boundary", "selection", "stage_entry"))
@pytest.mark.parametrize("point", ("payload", "marker"))
def test_runner_resumes_interrupted_publication(tmp_path, monkeypatch, kind, point):
    initial = {arm_id: checkpoint(arm_id) for arm_id in range(5)}
    design = design_for(initial)
    current, store, prior_batches = runner(tmp_path, initial_states=initial, design=design)
    original_write = store._write_publication_json
    filenames = {
        "initial": "state.json",
        "update": "update-00000001.json",
        "boundary": "boundary-00000002.json",
        "selection": "selection.json",
        "stage_entry": "stage_entry-00000004.json",
    }
    target_name = filenames[kind]
    if point == "marker":
        target_name = target_name.removesuffix(".json") + ".complete.json"

    def interrupted_write(path, payload):
        if path.name == target_name:
            raise RuntimeError("injected publication interruption")
        original_write(path, payload)

    monkeypatch.setattr(store, "_write_publication_json", interrupted_write)
    with pytest.raises(RuntimeError, match="publication interruption"):
        current.run()
    monkeypatch.setattr(store, "_write_publication_json", original_write)
    resumed, _same_store, next_batches = runner(
        tmp_path, store=store, initial_states=initial, design=design
    )
    result = resumed.run()
    baseline, _baseline_store, _ = runner(
        tmp_path / "baseline", initial_states=initial, design=design
    )
    expected = baseline.run()
    assert {arm: state.to_dict() for arm, state in result.states.items()} == {
        arm: state.to_dict() for arm, state in expected.states.items()
    }
    assert result.selection == expected.selection
    assert len(result.selection["records"]) == 5
    repeated = set(prior_batches).intersection(next_batches)
    assert repeated == ({(0, stages()[0].stage_id, 0)} if kind == "update" and point == "payload" else set())


def test_interrupted_publication_refuses_changed_payload(tmp_path, monkeypatch):
    store = AtomicCheckpointStore(tmp_path / "run", "runner-test")
    original_write = store._write_publication_json

    def interrupted_write(path, payload):
        if path.name == "state.complete.json":
            raise RuntimeError("injected publication interruption")
        original_write(path, payload)

    monkeypatch.setattr(store, "_write_publication_json", interrupted_write)
    with pytest.raises(RuntimeError, match="publication interruption"):
        store.initialize(0, checkpoint())
    monkeypatch.setattr(store, "_write_publication_json", original_write)
    path = store.root / "arms/arm-0/initial/state.json"
    payload = json.loads(path.read_text())
    payload["event"]["update_index"] = 50
    atomic_write_json(path, payload)
    with pytest.raises(ReplicationRunnerError, match="publication payload changed"):
        store.has_initial(0)
    assert not path.with_suffix(".complete.json").exists()


def test_prepared_publication_refuses_changed_retry(tmp_path, monkeypatch):
    store = AtomicCheckpointStore(tmp_path / "run", "runner-test")
    original_write = store._write_publication_json

    def interrupted_write(path, payload):
        if path.name == "state.json":
            raise RuntimeError("injected publication interruption")
        original_write(path, payload)

    monkeypatch.setattr(store, "_write_publication_json", interrupted_write)
    with pytest.raises(RuntimeError, match="publication interruption"):
        store.initialize(0, checkpoint())
    monkeypatch.setattr(store, "_write_publication_json", original_write)
    assert not store.has_initial(0)
    with pytest.raises(ReplicationRunnerError, match="publication changed"):
        store.initialize(0, checkpoint(update_index=1))
    store.initialize(0, checkpoint())
    assert store.recover(0)[0].to_dict() == checkpoint().to_dict()


def test_runner_selects_five_candidates_and_transfers_only_replica_four(tmp_path):
    current, store, _batches = runner(tmp_path)

    result = current.run()

    assert result.selection["selected_replica"] == 4
    assert store.read_selection()["selected_replica"] == 4
    assert {arm_id: state.update_index for arm_id, state in result.states.items()} == {
        0: 4, 1: 4, 2: 4, 3: 4, 4: 6
    }
    assert all(
        state.metadata["boundaries"] == [1, 2]
        for arm_id, state in result.states.items()
        if arm_id != 4
    )
    assert result.states[4].metadata["boundaries"] == [1, 2, 3]


def test_runner_resumes_after_interruption_without_reinitializing(tmp_path):
    interrupted, store, _batches = runner(tmp_path, executor=FakeExecutor(fail_on_call=4))

    with pytest.raises(RuntimeError, match="interruption"):
        interrupted.run()
    assert store.recover(0)[0].update_index == 1
    assert store.recover(3)[0].update_index == 0

    resumed, _same_store, _ = runner(tmp_path, store=store)
    result = resumed.run()

    assert result.selection["selected_replica"] == 4
    assert result.states[4].update_index == 6


def test_runner_does_not_duplicate_boundary_after_interruption(tmp_path):
    interrupted, store, _batches = runner(
        tmp_path, provider=FakeEvaluationProvider(fail_on_control=7)
    )

    with pytest.raises(RuntimeError, match="boundary interruption"):
        interrupted.run()
    assert store.recover(0)[0].metadata["boundaries"] == [1]

    resumed, _same_store, _ = runner(tmp_path, store=store)
    result = resumed.run()

    assert result.states[0].metadata["boundaries"] == [1, 2]
    assert result.states[4].metadata["boundaries"] == [1, 2, 3]


def test_runner_refuses_different_selected_replica(tmp_path):
    current, store, _batches = runner(tmp_path, provider=FakeEvaluationProvider(selected=0))

    with pytest.raises(ReplicationRunnerError, match="different replica"):
        current.run()
    rejected = store.read_selection()
    assert rejected["selected_replica"] == 0
    assert rejected["candidate_union_ids"] == list(range(5))
    assert len(rejected["records"]) == 5


def test_runner_stops_completed_selected_replica_before_next_stage_and_on_resume(tmp_path):
    class EarlyStopCoordinator(FakeCoordinator):
        def boundary(self, state, control):
            bounded, event = super().boundary(state, control)
            if state.metadata["arm_id"] == 4 and control.request.metadata["round"] == 1:
                bounded = replace(bounded, metadata={**bounded.metadata, "fake_complete": True})
            return bounded, event

    executor = FakeExecutor()
    executor.core.coordinator = EarlyStopCoordinator()
    current, store, _batches = runner(tmp_path, executor=executor)
    result = current.run()
    assert result.states[4].update_index == 2
    calls = executor.calls
    resumed, _store, _batches = runner(tmp_path, executor=executor, store=store)
    resumed_result = resumed.run()
    assert resumed_result.states[4].to_dict() == result.states[4].to_dict()
    assert executor.calls == calls


def test_selection_rejects_missing_role_manifest_provenance(tmp_path):
    class UnboundValidationProvider(FakeEvaluationProvider):
        def validation(self, arm_id, policy, stage, *, update_index):
            validated = super().validation(arm_id, policy, stage, update_index=update_index)
            return replace(validated, provenance={})

    initial = {arm_id: checkpoint(arm_id) for arm_id in range(5)}
    design = design_for(initial)
    current, _store, _batches = runner(
        tmp_path, design=design, initial_states=initial,
        provider=UnboundValidationProvider(role_hash=design.role_manifest_sha256),
    )
    with pytest.raises(ReplicationRunnerError, match="provenance"):
        current.run()


def test_design_binds_all_transactions_and_selection(tmp_path):
    initial = {arm_id: checkpoint(arm_id) for arm_id in range(5)}
    design = design_for(initial)
    current, store, _batches = runner(tmp_path, design=design, initial_states=initial)

    result = current.run()

    assert result.selection["replication_design_sha256"] == design.binding_hash()
    for arm_id in range(5):
        initial_marker = json.loads(
            (store.root / f"arms/arm-{arm_id}/initial/state.complete.json").read_text()
        )
        assert initial_marker["replication_design_sha256"] == design.binding_hash()
        assert initial_marker["complete"] is True
    update_marker = json.loads(
        (store.root / "arms/arm-4/checkpoints/update-00000001.complete.json").read_text()
    )
    boundary_marker = json.loads(
        (store.root / "arms/arm-4/boundaries/boundary-00000002.complete.json").read_text()
    )
    assert update_marker["replication_design_sha256"] == design.binding_hash()
    assert boundary_marker["replication_design_sha256"] == design.binding_hash()


def test_design_refuses_changed_design_or_initial_state_on_resume(tmp_path):
    initial = {arm_id: checkpoint(arm_id) for arm_id in range(5)}
    design = design_for(initial)
    current, store, _batches = runner(tmp_path, design=design, initial_states=initial)
    current.run()

    changed_design = replace(design, design_id="runner-design-changed")
    with pytest.raises(ValueError, match="checkpoint store"):
        runner(tmp_path, store=store, design=changed_design, initial_states=initial)

    changed_initial = dict(initial)
    changed_initial[0] = replace(initial[0], update_index=1)
    with pytest.raises(ValueError, match="initial state identity"):
        runner(tmp_path, store=store, design=design, initial_states=changed_initial)


def test_design_refuses_incomplete_persisted_candidate_union(tmp_path):
    initial = {arm_id: checkpoint(arm_id) for arm_id in range(5)}
    design = design_for(initial)
    current, store, _batches = runner(tmp_path, design=design, initial_states=initial)
    current.run()

    selection_path = store.root / "selection.json"
    marker_path = store.root / "selection.complete.json"
    payload = json.loads(selection_path.read_text())
    payload["selection"]["records"] = payload["selection"]["records"][:-1]
    payload["selection_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "selection_hash"}
    )
    selection_path.write_text(json.dumps(payload))
    marker = json.loads(marker_path.read_text())
    marker["selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    marker["selection_hash"] = payload["selection_hash"]
    marker_path.write_text(json.dumps(marker))

    resumed, _same_store, _ = runner(tmp_path, store=store, design=design, initial_states=initial)
    with pytest.raises(ReplicationRunnerError, match="candidate union"):
        resumed.run()


@pytest.mark.parametrize("threshold", (0, -0.01, float("nan"), float("inf"), True))
def test_runner_refuses_invalid_threshold(tmp_path, threshold):
    with pytest.raises(ValueError, match="finite and positive"):
        runner(tmp_path, threshold=threshold)


def test_runner_refuses_noncontiguous_stage_definition(tmp_path):
    with pytest.raises(ValueError, match="not contiguous"):
        GenericReplicationRunner(
            FakeExecutor(),
            {arm_id: object() for arm_id in range(5)},
            {arm_id: checkpoint(arm_id) for arm_id in range(5)},
            FakeEvaluationProvider(),
            lambda *_args: object(),
            AtomicCheckpointStore(tmp_path / "run", "runner-test"),
            (stages()[0], replace(stages()[1], start_update=5)),
        )


def test_store_refuses_orphaned_boundary(tmp_path):
    store = AtomicCheckpointStore(tmp_path / "run", "runner-test")
    initial = checkpoint()
    store.initialize(0, initial)
    atomic_write_json(
        store.root / "arms/arm-0/boundaries/boundary-00000001.json", {"orphan": True}
    )
    with pytest.raises(ReplicationRunnerError, match="orphaned"):
        store.recover(0)
