"""Compact control storage, synthetic membership and publication recovery."""

import json
import os
from dataclasses import replace
from unittest.mock import Mock

import pytest
from tests.contracts.test_generic_replication_runner import FakeEvaluationProvider, runner

from mooneural.training.generic_control_evidence import (
    CONTROL_REFERENCE_SCHEMA,
    ControlEvidenceArchive,
    ControlEvidenceError,
    full_control_hash,
)
from mooneural.training.generic_permanent_pass import (
    ControlPoint,
    PermanentPassCoordinator,
    PermanentPassState,
)
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    ReplicationRunnerError,
)
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    ControlEvaluation,
    EvaluationRequest,
    PolicyView,
    RoleBinding,
    RoleManifest,
    TaskDefinition,
    TaskRegistry,
    canonical_json,
    stable_hash,
)

TASKS = ("first", "second", "third")
REGISTRY = TaskRegistry(tuple(TaskDefinition(task, index) for index, task in enumerate(TASKS)))
POLICY = PolicyView((0.5, -0.2), "synthetic-storage-only")
ROLES = RoleManifest((RoleBinding("control", 22, "synthetic-control", "v1"),))


def control(values=(0.03, 0.2, 0.2), identity=0):
    return ControlEvaluation(
        dict(zip(TASKS, values, strict=True)),
        raw_records={"samples": [0.125] * 2048, "identity": identity, "synthetic_only": True},
        source_cell_results={"cell": {"upper": list(values)}},
        request=EvaluationRequest("control", "synthetic-control", seeds=(22,), task_ids=TASKS,
                                  policy_fingerprint=POLICY.fingerprint(), metadata={"identity": identity}),
    )


def state_fixture(store, *, compact=True):
    state = CheckpointState("control-test", 0, POLICY.to_dict(), {"iteration": 0},
                            {"preferred": "synthetic"}, {"next_seed_index": 0})
    coordinator = PermanentPassCoordinator(REGISTRY, ROLES, threshold=0.04)
    original = control()
    initial = coordinator.initialize(state, store.archive_control(original) if compact else original)
    return initial, coordinator, original


def advance(state, coordinator):
    return coordinator.commit(state, replace(state, update_index=state.update_index + 1))


def test_full_control_roundtrip_immutable_hash_and_compact_metadata(tmp_path):
    archive = ControlEvidenceArchive(tmp_path)
    original = control()
    before = original.to_dict()
    compact = archive.archive(original)
    assert compact.task_upper_mse == original.task_upper_mse
    assert compact.source_cell_results == original.source_cell_results
    assert compact.request.to_dict() == original.request.to_dict()
    assert compact.raw_records == {"schema": CONTROL_REFERENCE_SCHEMA,
                                   "full_control_sha256": stable_hash(before)}
    path = tmp_path / f"{full_control_hash(compact)}.json"
    assert path.read_text() == canonical_json(before)
    signature = path.stat()
    assert archive.archive(original).to_dict() == compact.to_dict()
    assert archive.archive(compact).to_dict() == compact.to_dict()
    assert path.stat() == signature
    expanded = ControlEvidenceArchive(tmp_path).rehydrate(compact)
    assert expanded.to_dict() == before == original.to_dict()
    expanded.raw_records["samples"][0] = 1.0
    assert archive.rehydrate(compact).to_dict() == before
    assert len(canonical_json(compact.to_dict())) < len(canonical_json(before)) / 5


@pytest.mark.parametrize("field", ("upper", "request", "cells", "hash", "fields", "schema"))
@pytest.mark.parametrize("cold", (False, True))
def test_reference_rejects_changed_control_binding(tmp_path, field, cold):
    archive = ControlEvidenceArchive(tmp_path)
    compact = archive.archive(control())
    if field == "upper":
        compact = replace(compact, task_upper_mse=dict.fromkeys(TASKS, 0.01))
    elif field == "request":
        compact = replace(compact, request=replace(compact.request, metadata={"identity": "wrong"}))
    elif field == "cells":
        compact = replace(compact, source_cell_results={"wrong": True})
    elif field == "hash":
        compact = replace(compact, raw_records={**compact.raw_records, "full_control_sha256": "../wrong"})
    elif field == "schema":
        compact = replace(compact, raw_records={**compact.raw_records,
                                               "schema": CONTROL_REFERENCE_SCHEMA.replace("v1", "v2")})
    else:
        compact = replace(compact, raw_records={**compact.raw_records, "extra": True})
    reader = ControlEvidenceArchive(tmp_path) if cold else archive
    with pytest.raises(ControlEvidenceError):
        reader.verify(compact)


@pytest.mark.parametrize("damage", ("missing", "changed", "same-size-restored-mtime", "replaced"))
def test_verified_cache_detects_file_mutation(tmp_path, damage):
    archive = ControlEvidenceArchive(tmp_path)
    compact = archive.archive(control())
    path = tmp_path / f"{full_control_hash(compact)}.json"
    before = path.stat()
    content = path.read_text()
    if damage == "missing":
        path.unlink()
    elif damage == "changed":
        path.write_text("{}")
    elif damage == "same-size-restored-mtime":
        path.write_text(content.replace("0.125", "0.225", 1))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        path.rename(path.with_suffix(".original"))
        path.write_text(content)
    with pytest.raises(ControlEvidenceError, match="missing|changed|hash mismatch"):
        archive.verify(compact)
    with pytest.raises(ControlEvidenceError, match="missing|changed|hash mismatch"):
        archive.rehydrate(compact)


@pytest.mark.parametrize("last_value", (0.039, 0.041))
def test_compact_history_preserves_membership_union_and_full_rehydration(tmp_path, last_value):
    archive = ControlEvidenceArchive(tmp_path)
    originals = (control(), control((0.2, 0.03, 0.2), 1), control((0.2, 0.2, last_value), 2))
    histories = []
    for compact in (False, True):
        controls = tuple(archive.archive(item) for item in originals) if compact else originals
        policy = PermanentPassState.initialize(REGISTRY, controls[0], POLICY, ROLES, threshold=0.04, update_index=0)
        policy = policy.committed_update().boundary(controls[1], POLICY, ROLES)
        policy = policy.stage_entry(controls[2], POLICY, ROLES, stage_id="continuation")
        histories.append(policy)
    expanded, compact = histories
    assert compact.permanent == expanded.permanent
    assert compact.partition == expanded.partition
    assert compact.complete == expanded.complete == (last_value <= 0.04)
    for before, after in zip(expanded.history(), compact.history(), strict=True):
        for field in ("before", "after", "entered_permanent", "regressed_permanent_diagnostic"):
            assert before[field] == after[field]
    restored = replace(compact, controls=tuple(ControlPoint(
        point.update_index, canonical_json(archive.rehydrate(point.evaluation).to_dict()), point.stage_id,
    ) for point in compact.controls))
    assert restored.to_dict() == expanded.to_dict()


def test_cold_recovery_verifies_each_distinct_control_once_and_keeps_history_compact(tmp_path):
    store = AtomicCheckpointStore(tmp_path, "control-test")
    initial, coordinator, original = state_fixture(store)
    store.initialize(0, initial)
    current = advance(initial, coordinator)
    store.commit_update(0, initial, current, {"event": "update"})
    second = control((0.2, 0.03, 0.2), 1)
    bounded, _event = coordinator.boundary(current, store.archive_control(second))
    store.commit_boundary(0, bounded, {"event": "boundary"})
    current = advance(bounded, coordinator)
    store.commit_update(0, bounded, current, {"event": "update"})
    reader = AtomicCheckpointStore(tmp_path, "control-test")
    reader.control_evidence._read = Mock(wraps=reader.control_evidence._read)
    recovered, _events = reader.recover(0)
    assert reader.control_evidence._read.call_count == 2
    assert recovered.to_dict() == current.to_dict()
    reader.recover(0)
    assert reader.control_evidence._read.call_count == 4
    points = coordinator.read(recovered).controls
    assert reader.rehydrate_control(points[0].evaluation).to_dict() == original.to_dict()
    assert reader.rehydrate_control(points[1].evaluation).to_dict() == second.to_dict()
    assert all(set(entry["control"]["raw_records"]) == {"schema", "full_control_sha256"}
               for entry in recovered.metadata["permanent_pass_rotation"]["history"])
    assert all(isinstance(value[1], str) for value in reader.control_evidence._verified.values())


@pytest.mark.parametrize("cold", (False, True))
@pytest.mark.parametrize("damage", ("missing", "corrupt", "same-size-restored-mtime"))
def test_completed_checkpoint_dependency_is_checked_on_every_recovery(tmp_path, cold, damage):
    store = AtomicCheckpointStore(tmp_path, "control-test")
    initial, _coordinator, _original = state_fixture(store)
    store.initialize(0, initial)
    assert store.recover(0)[0].to_dict() == initial.to_dict()
    path = next((tmp_path / "control_evidence").glob("*.json"))
    if damage == "missing":
        path.unlink()
    elif damage == "same-size-restored-mtime":
        before = path.stat()
        path.write_text(path.read_text().replace("0.125", "0.225", 1))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        path.write_text("{}")
    reader = AtomicCheckpointStore(tmp_path, "control-test") if cold else store
    with pytest.raises(ReplicationRunnerError, match="control evidence"):
        reader.recover(0)


def test_runner_archives_all_capture_locations_and_preserves_stage_entry_full_hash(tmp_path):
    class TrackedProvider(FakeEvaluationProvider):
        def __init__(self):
            super().__init__()
            self.originals = {}

        def control(self, arm_id, policy, stage, round_number):
            value = super().control(arm_id, policy, stage, round_number)
            self.originals[stable_hash(value.to_dict())] = value.to_dict()
            return value

    provider = TrackedProvider()
    current, store, _batches = runner(tmp_path, provider=provider)
    result = current.run()
    history = PermanentPassState.from_dict(result.states[4].metadata["permanent_pass_rotation"])
    assert [entry["event"] for entry in history.history()] == [
        "initialization", "boundary", "boundary", "stage_entry", "boundary",
    ]
    for state in result.states.values():
        for entry in state.metadata["permanent_pass_rotation"]["history"]:
            compact = ControlEvaluation.from_dict(entry["control"])
            assert store.rehydrate_control(compact).to_dict() == provider.originals[full_control_hash(compact)]
    stage_event = next(event for event in result.events if event["event"] == "stage_entry")
    assert stage_event["control_sha256"] == full_control_hash(history.controls[3].evaluation)
    assert len(list((store.root / "control_evidence").glob("*.json"))) == len(provider.originals)


@pytest.mark.parametrize("defect", ("negative", "stale", "order", "unbound"))
def test_original_control_is_validated_before_archiving(tmp_path, defect):
    current, store, _batches = runner(tmp_path)
    original = current.evaluation_provider.control

    def invalid(*args):
        value = original(*args)
        if defect == "negative":
            return replace(value, task_upper_mse=dict.fromkeys(value.task_upper_mse, -1.0))
        if defect == "unbound":
            return replace(value, request=None)
        request = replace(value.request, **({"policy_fingerprint": "0" * 64} if defect == "stale" else {
            "task_ids": tuple(reversed(value.request.task_ids)),
        }))
        return replace(value, request=request)

    current.evaluation_provider.control = invalid
    with pytest.raises(ValueError):
        current.run()
    assert not list(store.root.rglob("*.json"))
    assert current.executor.calls == 0


@pytest.mark.parametrize("point", ("before", "after"))
def test_interrupted_full_object_publication_has_exact_retry(tmp_path, monkeypatch, point):
    archive = ControlEvidenceArchive(tmp_path)
    original_write = archive._write

    def interrupted(path, content):
        if point == "after":
            original_write(path, content)
        raise InterruptedError("synthetic evidence publication interruption")

    monkeypatch.setattr(archive, "_write", interrupted)
    with pytest.raises(InterruptedError):
        archive.archive(control())
    retained = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
    resumed = ControlEvidenceArchive(tmp_path)
    assert resumed.rehydrate(resumed.archive(control())).to_dict() == control().to_dict()
    assert all(path.read_bytes() == content for path, content in retained.items())


@pytest.mark.parametrize("kind", ("initial", "update", "boundary", "stage_entry"))
@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_pending_marker_refuses_missing_or_corrupt_full_control(tmp_path, monkeypatch, kind, damage):
    current, store, _batches = runner(tmp_path)
    writer = store._write_publication_json
    pending = []

    def interrupted(path, payload):
        if path.name.endswith(".complete.json") and payload.get("kind") == kind:
            pending.append(path)
            raise InterruptedError("synthetic before marker")
        writer(path, payload)

    monkeypatch.setattr(store, "_write_publication_json", interrupted)
    with pytest.raises(InterruptedError):
        current.run()
    payload_path = pending[0].with_name(pending[0].name.replace(".complete.json", ".json"))
    payload = json.loads(payload_path.read_text())
    pending_state = store._decode_state(payload["state"])
    digest = pending_state.metadata["permanent_pass_rotation"]["history"][0]["control"]["raw_records"]["full_control_sha256"]
    broken = store.root / "control_evidence" / f"{digest}.json"
    if damage == "missing":
        broken.unlink()
    else:
        broken.write_text("{}")
    reopened = AtomicCheckpointStore(store.root, store.attempt_id)
    with pytest.raises(ReplicationRunnerError, match="control evidence"):
        reopened._resume_publications(pending[0].parent)
    assert not pending[0].exists()


@pytest.mark.parametrize("schema", (AtomicCheckpointStore.LEGACY_SCHEMA, AtomicCheckpointStore.OBJECT_SCHEMA))
def test_inline_v1_v2_remain_exact_and_new_v3_transactions_do_not_rewrite_them(tmp_path, schema):
    class HistoricalStore(AtomicCheckpointStore):
        SCHEMA = schema

        def _encode_state(self, state):
            return state.to_dict() if schema == self.LEGACY_SCHEMA else super()._encode_state(state)

    old = HistoricalStore(tmp_path, "control-test")
    initial, coordinator, _control = state_fixture(old, compact=False)
    old.initialize(0, initial)
    preserved = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    current = AtomicCheckpointStore(tmp_path, "control-test")
    assert current.recover(0)[0].to_dict() == initial.to_dict()
    updated = advance(initial, coordinator)
    current.commit_update(0, initial, updated, {"event": "update"})
    assert current.recover(0)[0].to_dict() == updated.to_dict()
    assert all(path.read_bytes() == content for path, content in preserved.items())
    payload = json.loads((tmp_path / "arms/arm-0/checkpoints/update-00000001.json").read_text())
    assert payload["schema"] == AtomicCheckpointStore.SCHEMA
