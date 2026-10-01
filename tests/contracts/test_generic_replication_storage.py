"""Exact semantic round trips for deduplicated replication metadata."""

import json
from dataclasses import replace

import pytest

from tests.support.fixtures import fake_control, fixture
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    ReplicationRunnerError,
)


def initialized_state():
    executor, checkpoint, _batch = fixture(update_index=0)
    control = fake_control(checkpoint, executor.adapter.registry.task_ids, [0.2] * 7)
    control = replace(control, raw_records={"synthetic_samples": list(range(2048))})
    return executor.coordinator.initialize(checkpoint, control), executor.coordinator


def next_state(state, coordinator):
    return coordinator.commit(state, replace(state, update_index=state.update_index + 1))


def test_metadata_objects_preserve_full_state_and_deduplicate_history(tmp_path):
    initial, coordinator = initialized_state()
    original = initial.to_dict()
    store = AtomicCheckpointStore(tmp_path, initial.attempt_id)
    store.initialize(0, initial)
    count = len(list((tmp_path / "objects").glob("*.json")))
    assert count == 4
    updated = initial
    for _index in range(3):
        previous = updated
        updated = next_state(previous, coordinator)
        store.commit_update(0, previous, updated, {"event": "update"})
        assert len(list((tmp_path / "objects").glob("*.json"))) == count
        assert store.recover(0)[0].to_dict() == updated.to_dict()
    assert initial.to_dict() == original
    current_control = fake_control(updated, coordinator.registry.task_ids, [0.1] * 7)
    bounded, event = coordinator.boundary(updated, current_control)
    store.commit_boundary(0, bounded, {"event": "boundary", "details": event["entered_permanent"]})
    assert len(list((tmp_path / "objects").glob("*.json"))) == count + 2
    recovered = store.recover(0)[0]
    assert recovered.to_dict() == bounded.to_dict()
    assert coordinator.read(recovered).history() == coordinator.read(bounded).history()
    encoded = json.loads((tmp_path / "arms/arm-0/checkpoints/update-00000003.json").read_text())["state"]
    assert len(json.dumps(encoded)) < len(json.dumps(updated.to_dict())) / 2


@pytest.mark.parametrize("damage", ("missing", "changed"))
def test_missing_or_corrupt_metadata_objects_refuse_resume(tmp_path, damage):
    initial, _coordinator = initialized_state()
    store = AtomicCheckpointStore(tmp_path, initial.attempt_id)
    store.initialize(0, initial)
    payload = json.loads((tmp_path / "arms/arm-0/initial/state.json").read_text())
    object_hash = payload["state"]["metadata"]["role_manifest"]["object_hash"]
    path = tmp_path / "objects" / f"{object_hash}.json"
    if damage == "missing":
        path.unlink()
    else:
        path.write_text("{}\n")
    with pytest.raises(ReplicationRunnerError, match="checkpoint object"):
        store.recover(0)


def test_mixed_legacy_and_deduplicated_transactions_resume_exactly(tmp_path):
    class LegacyStore(AtomicCheckpointStore):
        SCHEMA = AtomicCheckpointStore.LEGACY_SCHEMA

        def _encode_state(self, state):
            return state.to_dict()

    initial, coordinator = initialized_state()
    legacy = LegacyStore(tmp_path, initial.attempt_id)
    legacy.initialize(0, initial)
    updated = next_state(initial, coordinator)
    legacy.commit_update(0, initial, updated, {"event": "update"})
    current = AtomicCheckpointStore(tmp_path, initial.attempt_id)
    assert current.recover(0)[0].to_dict() == updated.to_dict()
    committed = next_state(updated, coordinator)
    current.commit_update(0, updated, committed, {"event": "update"})
    recovered, events = current.recover(0)
    assert recovered.to_dict() == committed.to_dict()
    assert [event["event"] for event in events] == ["initialization", "update", "update"]


@pytest.mark.parametrize("point", ("object", "payload", "marker"))
def test_object_publication_interruption_preserves_exact_retry(tmp_path, monkeypatch, point):
    initial, _coordinator = initialized_state()
    store = AtomicCheckpointStore(tmp_path, initial.attempt_id)
    original_write = store._write_publication_json

    def interrupted_write(path, payload):
        original_write(path, payload)
        if (
            (point == "object" and path.parent.name == "objects")
            or (point == "payload" and path.name == "state.json")
            or (point == "marker" and path.name == "state.complete.json")
        ):
            raise RuntimeError("storage publication interruption")

    monkeypatch.setattr(store, "_write_publication_json", interrupted_write)
    with pytest.raises(RuntimeError, match="storage publication interruption"):
        store.initialize(0, initial)
    object_bytes = {path: path.read_bytes() for path in (tmp_path / "objects").glob("*.json")}
    monkeypatch.setattr(store, "_write_publication_json", original_write)
    if not store.has_initial(0):
        store.initialize(0, initial)
    assert store.recover(0)[0].to_dict() == initial.to_dict()
    assert all(path.read_bytes() == original for path, original in object_bytes.items())


@pytest.mark.parametrize("kind", ("initial", "update", "boundary"))
@pytest.mark.parametrize("dependency", ("history_index", "history_entry"))
@pytest.mark.parametrize("damage", ("missing", "changed"))
def test_pending_marker_requires_all_history_objects(
    tmp_path, monkeypatch, kind, dependency, damage
):
    initial, coordinator = initialized_state()
    store = AtomicCheckpointStore(tmp_path, initial.attempt_id)
    updated = next_state(initial, coordinator)
    if kind != "initial":
        store.initialize(0, initial)
    if kind == "boundary":
        store.commit_update(0, initial, updated, {"event": "update"})
        control = fake_control(updated, coordinator.registry.task_ids, [0.1] * 7)
        bounded, _event = coordinator.boundary(updated, control)
    original_write = store._write_publication_json

    def interrupted_write(path, payload):
        if path.name.endswith(".complete.json"):
            raise RuntimeError("before marker")
        original_write(path, payload)

    monkeypatch.setattr(store, "_write_publication_json", interrupted_write)
    with pytest.raises(RuntimeError, match="before marker"):
        if kind == "initial":
            store.initialize(0, initial)
        elif kind == "update":
            store.commit_update(0, initial, updated, {"event": "update"})
        else:
            store.commit_boundary(0, bounded, {"event": "boundary"})
    directory_name = {"initial": "initial", "update": "checkpoints", "boundary": "boundaries"}[kind]
    directory = tmp_path / "arms/arm-0" / directory_name
    stem = "state" if kind == "initial" else f"{kind}-00000001"
    payload = json.loads((directory / f"{stem}.json").read_text())
    reference = payload["state"]["metadata"]["permanent_pass_rotation"]["history"]
    object_path = tmp_path / "objects" / f"{reference['object_hash']}.json"
    if dependency == "history_entry":
        reference = json.loads(object_path.read_text())[-1]
        object_path = tmp_path / "objects" / f"{reference['object_hash']}.json"
    if damage == "missing":
        object_path.unlink()
    else:
        object_path.write_text("{}\n")
    recovered_store = AtomicCheckpointStore(tmp_path, initial.attempt_id)
    with pytest.raises(ReplicationRunnerError, match="checkpoint object"):
        recovered_store._resume_publications(directory)
    assert not (directory / f"{stem}.complete.json").exists()
    with pytest.raises(ReplicationRunnerError, match="checkpoint object"):
        recovered_store.recover(0)
