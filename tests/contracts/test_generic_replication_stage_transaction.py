"""Host regressions for source stage transactions and named selection metrics."""

import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from tests.contracts.test_generic_permanent_pass import REGISTRY, checkpoint, control

from mooneural.artifacts import atomic_write_json
from mooneural.training.generic_replication_design import ReplicationDesignBinding
from mooneural.training.generic_replication_policy import (
    ReplicationPermanentPassCoordinator,
)
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    GenericReplicationRunner,
    ReplicationRunnerError,
    ReplicationStage,
)
from mooneural.training.generic_role_banks import RoleBank, RoleBankManifest
from mooneural.training.generic_scheduled_role_banks import (
    ScheduledRoleBankRegistry,
    ScheduledRoleBinding,
)
from mooneural.training.generic_training_contracts import (
    CertificationResult,
    PolicyView,
    ValidationEvaluation,
    canonical_json,
    stable_hash,
)

SOURCE_STAGES = (
    ReplicationStage("round91-110", 90, 91, 110, 27000, population_size=5, select_after=True),
    ReplicationStage("round111-130", 110, 111, 130, 33000),
    ReplicationStage("round131-330", 130, 131, 330, 39000),
    ReplicationStage("round331-380", 330, 331, 380, 99000),
)
PUBLICATION_MODES = ("commit", "completed", "pending")
PREFERRED_METHOD = "weighted_normalized_sum"


def scoped_bank(index, stage, arm_id, role, round_number, min_update, max_update):
    bank = RoleBank(
        role, f"synthetic-stage-bank-{index}", 900000 + index,
        "synthetic-stage-target", "v1", "v1", "v1",
        {"global": stable_hash(["synthetic-global", index])},
        (f"synthetic-stage-sample-{index}",),
    )
    return ScheduledRoleBinding(
        stage.stage_id, arm_id, role, round_number, min_update, max_update,
        RoleBankManifest(REGISTRY.task_ids, (bank,), {role: 1}),
    )


def source_design(initial_states, roles):
    return ReplicationDesignBinding(
        design_id="source-coordinate-host-transaction-fixture",
        master_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        generic_code_hashes={"synthetic-fixture": "3" * 64},
        environment_binding={"runtime": "host-only"},
        backend_binding={"backend": "python_fake", "dtype": "float64"},
        role_manifest_sha256=roles.binding_hash(),
        target_hashes={"synthetic-target": "4" * 64},
        initial_state_hashes={
            str(arm_id): stable_hash(state.to_dict())
            for arm_id, state in initial_states.items()
        },
        task_ids=REGISTRY.task_ids,
        threshold=0.04,
        replica_ids=tuple(sorted(initial_states)),
        expected_selected_replica=4,
        selection_key=("absolute_threshold_ratio", "replica"),
        stages=tuple(asdict(stage) for stage in SOURCE_STAGES),
        optimizer_binding={"name": "synthetic-state-only"},
        permanent_pass_binding={
            "stage_entries": {
                stage.stage_id: {
                    "permanent_tasks": list(REGISTRY.task_ids[:6]),
                    "preferred_method": PREFERRED_METHOD,
                }
                for stage in SOURCE_STAGES[1:]
            },
        },
        seed_registry_sha256="5" * 64,
    )


def transaction_fixture(
    tmp_path, *, ordinal=1, unresolved_value=0.039, wrong_guard=None,
    stale_boundary_control=False,
):
    stage = SOURCE_STAGES[ordinal]
    previous_stage = SOURCE_STAGES[ordinal - 1]
    update_index = stage.start_update
    roles = ScheduledRoleBankRegistry((
        scoped_bank(0, previous_stage, 4, "control", stage.from_round - 1, update_index - 1, update_index - 1),
        scoped_bank(1, previous_stage, 4, "control", stage.from_round, update_index, update_index),
        scoped_bank(2, stage, 4, "control", stage.from_round, update_index, update_index),
    ))
    initial_states = {
        arm_id: replace(
            checkpoint(update_index - 1),
            method_state={"preferred": "pcgrad" if wrong_guard == "method" else PREFERRED_METHOD},
            rng_state={"synthetic_arm_id": arm_id, "next_update": update_index - 1},
        )
        for arm_id in range(5)
    }
    design = source_design(initial_states, roles)
    coordinator = ReplicationPermanentPassCoordinator(REGISTRY, roles, threshold=0.04)
    policy = PolicyView.from_dict(initial_states[4].policy_state)

    def evaluation(values, source_stage, round_number, coordinate):
        return replace(control(values, policy=policy), request=roles.request_for(
            "control", policy, stage_id=source_stage.stage_id, arm_id=4,
            round_number=round_number, update_index=coordinate,
        ))

    initial_values = [0.03] * (5 if wrong_guard == "membership" else 6)
    initial_values += [0.1] * (7 - len(initial_values))
    initial = coordinator.initialize(
        design.bind_checkpoint(initial_states[4]),
        evaluation(initial_values, previous_stage, stage.from_round - 1, update_index - 1),
    )
    updated = coordinator.commit(initial, replace(initial, update_index=update_index))
    parent, _event = coordinator.boundary(
        updated, evaluation([0.2] * 7, previous_stage, stage.from_round, update_index)
    )
    entry_control = evaluation([0.2] * 6 + [unresolved_value], stage, stage.from_round, update_index)
    entered, _event = coordinator.stage_entry(parent, entry_control, stage_id=stage.stage_id)
    if stale_boundary_control:
        prior_policy = coordinator.read(updated)
        entry_point = replace(
            coordinator.read(entered).controls[-1],
            update_index=prior_policy.controls[-1].update_index,
        )
        next_policy = replace(prior_policy, controls=(*prior_policy.controls, entry_point))
        parent = updated
        entered = replace(parent, metadata={
            **parent.metadata, "permanent_pass_rotation": next_policy.to_dict(),
        })
    store = AtomicCheckpointStore(tmp_path, initial.attempt_id, design=design)
    store.initialize(4, initial)
    store.commit_update(4, initial, updated, {"event": "update", "arm_id": 4, "update_index": update_index})
    store.commit_boundary(4, parent, {
        "event": "boundary", "arm_id": 4, "round": stage.from_round,
        "update_index": update_index, "policy": parent.policy_fingerprint,
    })
    event = {
        "event": "stage_entry", "stage_id": stage.stage_id, "stage_ordinal": ordinal,
        "arm_id": 4, "round": stage.from_round, "boundary": 0,
        "update_index": update_index, "parent_state_sha256": stable_hash(parent.to_dict()),
        "control_sha256": stable_hash(entry_control.to_dict()),
    }
    return SimpleNamespace(
        store=store, coordinator=coordinator, design=design, parent=parent,
        entered=entered, event=event, stage=stage,
    )


def entry_paths(fixture):
    directory = fixture.store.root / "arms/arm-4/stage_entries"
    stem = f"stage_entry-{fixture.parent.update_index:08d}"
    return directory / f"{stem}.json", directory / f"{stem}.complete.json"


def publish_historical_entry(fixture, state, event, *, pending):
    """Represent an inconsistent old writer without bypassing the reader."""
    store = fixture.store
    path, marker_path = entry_paths(fixture)
    payload = {
        "schema": store.SCHEMA, "kind": "stage_entry", "attempt_id": store.attempt_id,
        "arm_id": 4, "state": store._encode_state(state), "event": dict(event),
        "replication_design_sha256": fixture.design.binding_hash(),
    }
    payload["transaction_hash"] = stable_hash(payload)
    marker = {
        "schema": store.MARKER_SCHEMA, "kind": "stage_entry", "attempt_id": store.attempt_id,
        "arm_id": 4, "update_index": state.update_index,
        "state_path": str(path.relative_to(store.root)), "state_sha256": "",
        "transaction_hash": payload["transaction_hash"],
        "replication_design_sha256": fixture.design.binding_hash(), "complete": True,
    }
    intent = {
        "schema": store.PUBLICATION_SCHEMA, "attempt_id": store.attempt_id,
        "path": str(path.relative_to(store.root)), "payload_hash": stable_hash(payload),
        "marker": marker, "hash_field": "state_sha256",
    }
    atomic_write_json(path.with_name(f".{path.name}.publication.json"), intent)
    atomic_write_json(path, payload)
    if not pending:
        atomic_write_json(marker_path, {
            **marker, "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })


def assert_invalid_entry(fixture, state, event, mode, *, supplied_parent=None):
    _path, marker_path = entry_paths(fixture)
    if mode == "commit":
        with pytest.raises(ReplicationRunnerError):
            fixture.store.commit_stage_entry(
                4, fixture.parent if supplied_parent is None else supplied_parent, state, event
            )
        assert not marker_path.exists()
        return
    publish_historical_entry(fixture, state, event, pending=mode == "pending")
    restored = AtomicCheckpointStore(fixture.store.root, fixture.parent.attempt_id, design=fixture.design)
    if mode == "pending":
        with pytest.raises(ReplicationRunnerError):
            restored._resume_publications(marker_path.parent)
        assert not marker_path.exists()
    with pytest.raises(ReplicationRunnerError):
        restored.recover(4)


@pytest.mark.parametrize("mode", PUBLICATION_MODES)
@pytest.mark.parametrize("damage", ("no_op", "truncated_history", "unrelated_metadata"))
def test_stage_entry_requires_appended_control_and_exact_parent_metadata(tmp_path, mode, damage):
    fixture = transaction_fixture(tmp_path)
    entered = fixture.entered
    if damage == "no_op":
        entered = fixture.parent
    elif damage == "truncated_history":
        policy = fixture.coordinator.read(entered)
        shortened = replace(policy, controls=policy.controls[-2:])
        entered = replace(entered, metadata={
            **entered.metadata, "permanent_pass_rotation": shortened.to_dict(),
        })
    else:
        entered = replace(entered, metadata={**entered.metadata, "source_parent": {"round": -1}})
    assert_invalid_entry(fixture, entered, fixture.event, mode)


@pytest.mark.parametrize("mode", PUBLICATION_MODES)
@pytest.mark.parametrize("damage", ("wrong", "missing"))
@pytest.mark.parametrize("field", (
    "stage_id", "stage_ordinal", "arm_id", "round", "boundary", "update_index", "control_sha256",
))
def test_stage_entry_event_fields_match_the_frozen_stage_and_control(tmp_path, mode, damage, field):
    fixture = transaction_fixture(tmp_path)
    event = dict(fixture.event)
    if damage == "missing":
        event.pop(field)
    else:
        event[field] = {
            "stage_id": "round131-330", "stage_ordinal": 3, "arm_id": 3,
            "round": 999, "boundary": 1, "update_index": 33001,
            "control_sha256": "0" * 64,
        }[field]
    assert_invalid_entry(fixture, fixture.entered, event, mode)


@pytest.mark.parametrize("mode", PUBLICATION_MODES)
@pytest.mark.parametrize("guard", ("membership", "method"))
def test_stage_entry_checks_inherited_guards_against_pre_entry_parent(tmp_path, mode, guard):
    fixture = transaction_fixture(tmp_path, wrong_guard=guard)
    assert_invalid_entry(fixture, fixture.entered, fixture.event, mode)


@pytest.mark.parametrize("mode", PUBLICATION_MODES)
def test_stage_entry_uses_the_actual_committed_boundary_parent(tmp_path, mode):
    fixture = transaction_fixture(tmp_path)
    alternate_parent = replace(fixture.parent, metadata={
        **fixture.parent.metadata, "source_parent": {"round": 89},
    })
    alternate_entry = replace(fixture.entered, metadata={
        **fixture.entered.metadata, "source_parent": {"round": 89},
    })
    event = {**fixture.event, "parent_state_sha256": stable_hash(alternate_parent.to_dict())}
    assert_invalid_entry(fixture, alternate_entry, event, mode, supplied_parent=alternate_parent)


@pytest.mark.parametrize("mode", PUBLICATION_MODES)
def test_stage_entry_parent_requires_control_at_its_actual_boundary_update(tmp_path, mode):
    fixture = transaction_fixture(tmp_path, unresolved_value=0.041, stale_boundary_control=True)
    assert fixture.coordinator.read(fixture.parent).controls[-1].update_index < fixture.parent.update_index
    assert_invalid_entry(fixture, fixture.entered, fixture.event, mode)


@pytest.mark.parametrize("mode", PUBLICATION_MODES)
@pytest.mark.parametrize("ordinal", (1, 2, 3))
@pytest.mark.parametrize("unresolved_value", (0.039, 0.041))
def test_valid_source_coordinate_entry_preserves_history_state_and_cutoff(
    tmp_path, mode, ordinal, unresolved_value
):
    fixture = transaction_fixture(tmp_path, ordinal=ordinal, unresolved_value=unresolved_value)
    before = canonical_json(fixture.parent.to_dict())
    if mode == "commit":
        fixture.store.commit_stage_entry(4, fixture.parent, fixture.entered, fixture.event)
    else:
        publish_historical_entry(fixture, fixture.entered, fixture.event, pending=mode == "pending")
    restored = AtomicCheckpointStore(fixture.store.root, fixture.parent.attempt_id, design=fixture.design)
    recovered, events = restored.recover(4)
    assert recovered.to_dict() == fixture.entered.to_dict()
    assert events[-1] == fixture.event
    recovered_policy = fixture.coordinator.read(recovered)
    assert recovered_policy.history()[:-1] == fixture.coordinator.read(fixture.parent).history()
    assert recovered_policy.complete == (unresolved_value == 0.039)
    assert replace(recovered, metadata=fixture.parent.metadata).to_dict() == fixture.parent.to_dict()
    cutoff, cutoff_events = restored.recover(
        4, through_update_index=fixture.stage.start_update, include_stage_entry_at_cutoff=False,
    )
    assert canonical_json(cutoff.to_dict()) == before
    assert cutoff_events[-1]["event"] == "boundary"
    assert restored.recover(4)[0].to_dict() == recovered.to_dict()
    assert canonical_json(fixture.parent.to_dict()) == before


class HostSelectionExecutor:
    def __init__(self, roles):
        self.core = SimpleNamespace(
            coordinator=ReplicationPermanentPassCoordinator(REGISTRY, roles, threshold=0.04)
        )
        self.metadata = SimpleNamespace(registry=REGISTRY)

    def initialize(self, state, evaluation):
        return self.core.coordinator.initialize(state, evaluation)

    def step(self, state, batch, adapter):
        raise AssertionError("a completed synthetic candidate cannot consume an update")


class HostSelectionProvider:
    def __init__(self, roles):
        self.roles = roles
        self.calls = []

    def control(self, arm_id, policy, stage, round_number):
        self.calls.append(("control", arm_id))
        return replace(control([0.03] * 7, policy=policy), request=self.roles.request_for(
            "control", policy, stage_id=stage.stage_id, arm_id=arm_id,
            round_number=round_number, update_index=stage.start_update,
        ))

    def validation(self, arm_id, policy, stage, *, update_index):
        self.calls.append(("validation", arm_id))
        request = self.roles.request_for(
            "validation", policy, stage_id=stage.stage_id, arm_id=arm_id,
            round_number=None, update_index=update_index,
        )
        value = 0.01 if arm_id == 4 else 0.02
        return ValidationEvaluation(
            dict.fromkeys(REGISTRY.task_ids, value), dict.fromkeys(REGISTRY.task_ids, 2),
            provenance={
                "role_bank_manifest_hash": request.metadata["role_bank_manifest_hash"],
                "scheduled_role_registry_hash": self.roles.binding_hash(),
            },
            request=request,
        )

    def certification(self, arm_id, policy, stage, *, update_index):
        self.calls.append(("certification", arm_id))
        request = self.roles.request_for(
            "certification", policy, stage_id=stage.stage_id, arm_id=arm_id,
            round_number=None, update_index=update_index,
        )
        return CertificationResult(
            {"synthetic": False}, {"synthetic": {"upper_normalized_mse": 0.0537}},
            {"synthetic_integrity": True}, {
                "role_bank_manifest_hash": request.metadata["role_bank_manifest_hash"],
                "scheduled_role_registry_hash": self.roles.binding_hash(),
            }, request=request,
        )


def selection_runner(tmp_path):
    stage = SOURCE_STAGES[0]
    roles = ScheduledRoleBankRegistry(tuple(
        scoped_bank(
            arm_id * 3 + role_index, stage, arm_id, role,
            stage.from_round if role == "control" else None,
            stage.start_update, stage.start_update if role == "control" else stage.stop_update,
        )
        for arm_id in range(5)
        for role_index, role in enumerate(("control", "validation", "certification"))
    ))
    initial = {
        arm_id: replace(
            checkpoint(stage.start_update),
            policy_state=PolicyView((float(arm_id), 0.5), f"host-candidate-{arm_id}").to_dict(),
            policy_fingerprint=None,
            rng_state={"synthetic_arm_id": arm_id},
        )
        for arm_id in range(5)
    }
    design = source_design(initial, roles)
    provider = HostSelectionProvider(roles)
    store = AtomicCheckpointStore(tmp_path, "parent", design=design)

    def no_batch(*args):
        raise AssertionError("no batch is needed for already complete synthetic candidates")

    current = GenericReplicationRunner(
        HostSelectionExecutor(roles), {arm_id: object() for arm_id in initial},
        initial, provider, no_batch, store, SOURCE_STAGES, design=design,
    )
    return current, store, provider


def test_source_seven_task_selection_survives_json_without_reselection(tmp_path):
    current, store, provider = selection_runner(tmp_path)
    result = current.run()
    assert result.selection["candidate_union_ids"] == list(range(5))
    assert result.selection["selected_replica"] == 4
    assert len(provider.calls) == 11
    stored = store.read_selection()
    assert stored == result.selection
    assert tuple(stored["records"][0]["validation"]["task_mean_mse"]) != REGISTRY.task_ids
    resumed, _store, resumed_provider = selection_runner(tmp_path)
    restored = resumed.run()
    assert restored.selection == result.selection
    assert not resumed_provider.calls
    assert {
        arm_id: state.to_dict() for arm_id, state in restored.states.items()
    } == {arm_id: state.to_dict() for arm_id, state in result.states.items()}


@pytest.mark.parametrize("damage", ("metric_inventory", "request_order"))
def test_source_seven_task_selection_still_rejects_invalid_named_or_ordered_tasks(tmp_path, damage):
    current, _store, _provider = selection_runner(tmp_path)
    result = current.run()
    selection = json.loads(json.dumps(result.selection))
    validation = selection["records"][0]["validation"]
    if damage == "metric_inventory":
        validation["task_mean_mse"].pop(REGISTRY.task_ids[0])
        validation["sample_counts"].pop(REGISTRY.task_ids[0])
    else:
        validation["request"]["task_ids"].reverse()
    with pytest.raises(ReplicationRunnerError):
        current._validate_selection(result.states, selection, SOURCE_STAGES[0])
