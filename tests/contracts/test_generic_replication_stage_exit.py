"""Carson-authored host regressions for stage-exit receipts and recovery."""

import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest
from tests.contracts.test_generic_permanent_pass import REGISTRY
from tests.contracts.test_generic_replication_runner import (
    FakeEvaluationProvider,
    FakeExecutor,
    checkpoint,
    runner,
    stages,
)
from tests.contracts.test_generic_replication_stage_transaction import selection_runner

from mooneural.artifacts import atomic_write_json
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_replication_runner import (
    GenericReplicationRunner,
    ReplicationRunnerError,
    ReplicationStage,
)
from mooneural.training.generic_training_contracts import (
    CertificationResult,
    ValidationEvaluation,
    canonical_json,
    stable_hash,
)

EXIT_STAGES = (
    *stages(),
    ReplicationStage("continuation-2", 3, 4, 4, 6, updates_per_round=2),
    ReplicationStage("continuation-3", 4, 5, 5, 8, updates_per_round=2),
)
CERTIFICATION_PATHS = tuple(
    f"global.{cell}.{task}"
    for cell in ("default", "interior-policy", "interior-preference", "interior-structure")
    for task in REGISTRY.task_ids[:4]
) + tuple(f"local.{task.removeprefix('local.second_order.')}" for task in REGISTRY.task_ids[4:])
EXPECTED_SLOTS = (
    *((0, arm_id, "validation") for arm_id in range(5)),
    (0, 4, "certification"),
    *((ordinal, 4, role) for ordinal in range(1, 4) for role in ("validation", "certification")),
)


class TrackedProvider(FakeEvaluationProvider):
    def __init__(self, *, completion=None):
        super().__init__()
        self.calls = []
        self.results = {}
        self.completion = completion

    def control(self, arm_id, policy, stage, round_number):
        result = super().control(arm_id, policy, stage, round_number)
        if self.completion is not None and self.completion(arm_id, stage, round_number):
            result = replace(result, task_upper_mse=dict.fromkeys(result.task_upper_mse, 0.03))
        return result

    def validation(self, arm_id, policy, stage, *, update_index):
        key = (stage.stage_id, arm_id, "validation", update_index)
        self.calls.append(key)
        result = super().validation(arm_id, policy, stage, update_index=update_index)
        result = replace(result, provenance={
            **result.provenance,
            "raw_samples": {task: [value, value] for task, value in result.task_mean_mse.items()},
            "maximum_component_diagnostic": {"passed": False, "acceptance_role": "explanatory_nonvetoing"},
            "synthetic_only": True,
        })
        self.results[key] = result.to_dict()
        return result

    def certification(self, arm_id, policy, stage, *, update_index):
        key = (stage.stage_id, arm_id, "certification", update_index)
        self.calls.append(key)
        result = super().certification(arm_id, policy, stage, update_index=update_index)
        result = replace(
            result,
            conjuncts={name: index != 18 for index, name in enumerate(CERTIFICATION_PATHS)},
            upper_records={
                "metrics": {name: {"upper_normalized_mse": 0.0537 if index == 18 else 0.01,
                                   "samples": [0.01, 0.02], "passed": index != 18}
                            for index, name in enumerate(CERTIFICATION_PATHS)},
                "maximum_component_diagnostic": {"passed": False, "acceptance_role": "explanatory_nonvetoing"},
                "synthetic_only": True,
            },
            hard_vetoes={"synthetic_integrity": True, "production_integrity_evidence": False},
            estimator_metadata={**result.estimator_metadata, "diagnostic_only": True,
                                "terminal_scientific_certification": False,
                                "unqualified_integrity": ["training_history", "domain_support"]},
        )
        self.results[key] = result.to_dict()
        return result


def staged_runner(tmp_path, *, provider=None, executor=None):
    provider = provider or TrackedProvider()
    executor = executor or FakeExecutor()
    current, store, batches = runner(tmp_path, provider=provider, executor=executor)
    extended = GenericReplicationRunner(
        executor, current.adapters, {arm_id: checkpoint(arm_id) for arm_id in range(5)},
        provider, current.batch_factory, store, EXIT_STAGES,
    )
    return extended, store, provider, executor, batches


def receipt_path(store, slot):
    ordinal, arm_id, role = slot
    return store.root / "evaluations" / f"stage-{ordinal}" / f"arm-{arm_id}" / role / "result.json"


def slot_for(record):
    return record["stage_ordinal"], record["arm_id"], record["role"]


def store_hashes(store):
    return {
        str(path.relative_to(store.root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.root.rglob("*") if path.is_file()
    }


def forbid_call(*args, **kwargs):
    raise AssertionError("completed resume must not evaluate, update, select or enter again")


def test_full_stage_exit_order_raw_outcomes_and_completed_restart(tmp_path, monkeypatch):
    current, store, provider, executor, batches = staged_runner(tmp_path)
    publication = []
    original_evaluation = store.commit_evaluation
    original_selection = store.commit_selection
    original_entry = store.commit_stage_entry

    def commit_evaluation(record, validator):
        original_evaluation(record, validator)
        publication.append(("evaluation", *slot_for(record)))

    def commit_selection(selection):
        result = original_selection(selection)
        publication.append(("selection",))
        return result

    def commit_entry(arm_id, previous, state, event):
        result = original_entry(arm_id, previous, state, event)
        publication.append(("entry", event["stage_ordinal"]))
        return result

    monkeypatch.setattr(store, "commit_evaluation", commit_evaluation)
    monkeypatch.setattr(store, "commit_selection", commit_selection)
    monkeypatch.setattr(store, "commit_stage_entry", commit_entry)
    result = current.run()
    assert tuple(slot_for(record) for record in result.evaluations) == EXPECTED_SLOTS
    expected = [("evaluation", 0, arm_id, "validation") for arm_id in range(5)]
    expected.extend([("selection",), ("evaluation", 0, 4, "certification")])
    for ordinal in range(1, 4):
        expected.extend([("entry", ordinal), ("evaluation", ordinal, 4, "validation"),
                         ("evaluation", ordinal, 4, "certification")])
    assert publication == expected
    assert executor.calls == len(batches) == 26
    assert result.states[4].update_index == 10
    assert result.selection["candidate_union_ids"] == list(range(5))
    for record in result.evaluations:
        stage = EXIT_STAGES[record["stage_ordinal"]]
        key = (stage.stage_id, record["arm_id"], record["role"], record["update_index"])
        assert record["result"] == provider.results[key]
        assert record["checkpoint_reload_equal"] is True
        assert record["update_index"] == stage.stop_update
        assert record["round"] == stage.last_round
        if record["role"] == "validation":
            assert record["predecessor_fingerprint"] is None
        elif record["stage_ordinal"] == 0:
            assert record["predecessor_fingerprint"] == stable_hash(result.selection)
        else:
            preceding = next(receipt for receipt in result.evaluations
                             if slot_for(receipt) == (record["stage_ordinal"], 4, "validation"))
            assert record["predecessor_fingerprint"] == stable_hash(preceding)
        saved, _events = store.recover(record["arm_id"], through_update_index=stage.stop_update,
                                       include_stage_entry_at_cutoff=False)
        assert stable_hash(saved.to_dict()) == record["state_fingerprint"]
        if record["role"] == "certification":
            typed = CertificationResult.from_dict(record["result"])
            assert len(typed.conjuncts) == 19
            assert not typed.passed
            assert typed.conjuncts[CERTIFICATION_PATHS[-1]] is False
            assert typed.hard_vetoes["production_integrity_evidence"] is False
            assert typed.estimator_metadata["terminal_scientific_certification"] is False
        else:
            ValidationEvaluation.from_dict(record["result"])
    rotation = PermanentPassState.from_dict(result.states[4].metadata["permanent_pass_rotation"])
    assert len(rotation.controls) == 9
    assert not rotation.permanent
    before = store_hashes(store)
    resumed, _store, resumed_provider, resumed_executor, resumed_batches = staged_runner(tmp_path)
    for method in ("control", "validation", "certification"):
        monkeypatch.setattr(resumed_provider, method, forbid_call)
    monkeypatch.setattr(resumed_executor, "step", forbid_call)
    monkeypatch.setattr(resumed.store, "commit_selection", forbid_call)
    restored = resumed.run()
    assert restored.evaluations == result.evaluations
    assert restored.selection == result.selection
    assert {arm_id: state.to_dict() for arm_id, state in restored.states.items()} == {
        arm_id: state.to_dict() for arm_id, state in result.states.items()
    }
    assert not resumed_batches
    assert before == store_hashes(store)


@pytest.mark.parametrize("slot", ((0, 2, "validation"), (0, 4, "certification"),
                                  (1, 4, "validation"), (1, 4, "certification"),
                                  (3, 4, "certification")))
@pytest.mark.parametrize("point", ("before_payload", "before_marker", "after_marker"))
def test_receipt_interruption_reuses_committed_evaluations_and_updates(tmp_path, monkeypatch, slot, point):
    current, store, provider, executor, batches = staged_runner(tmp_path)
    target = receipt_path(store, slot)
    marker = target.with_suffix(".complete.json")
    original_write = store._write_publication_json

    def interrupted_write(path, payload):
        if (point == "before_payload" and path == target) or (point == "before_marker" and path == marker):
            raise RuntimeError("synthetic receipt publication interruption")
        original_write(path, payload)
        if point == "after_marker" and path == marker:
            raise RuntimeError("synthetic receipt publication interruption")

    monkeypatch.setattr(store, "_write_publication_json", interrupted_write)
    with pytest.raises(RuntimeError, match="synthetic receipt publication interruption"):
        current.run()
    assert target.is_file() == (point != "before_payload")
    resumed, _store, next_provider, next_executor, next_batches = staged_runner(tmp_path)
    result = resumed.run()
    expected = Counter((EXIT_STAGES[ordinal].stage_id, arm_id, role, EXIT_STAGES[ordinal].stop_update)
                       for ordinal, arm_id, role in EXPECTED_SLOTS)
    if point == "before_payload":
        expected[(EXIT_STAGES[slot[0]].stage_id, slot[1], slot[2], EXIT_STAGES[slot[0]].stop_update)] += 1
    assert Counter(provider.calls + next_provider.calls) == expected
    assert executor.calls + next_executor.calls == 26
    assert len(batches + next_batches) == len(set(batches + next_batches)) == 26
    assert tuple(slot_for(record) for record in result.evaluations) == EXPECTED_SLOTS
    assert result.states[4].update_index == 10


@pytest.mark.parametrize("point", ("before_marker", "after_marker"))
def test_selection_interruption_retains_five_validation_receipts(tmp_path, monkeypatch, point):
    current, store, provider, executor, _batches = staged_runner(tmp_path)
    marker = store.root / "selection.complete.json"
    original_write = store._write_publication_json

    def interrupted_write(path, payload):
        if point == "before_marker" and path == marker:
            raise RuntimeError("synthetic selection publication interruption")
        original_write(path, payload)
        if point == "after_marker" and path == marker:
            raise RuntimeError("synthetic selection publication interruption")

    monkeypatch.setattr(store, "_write_publication_json", interrupted_write)
    with pytest.raises(RuntimeError, match="synthetic selection publication interruption"):
        current.run()
    assert len(provider.calls) == 5
    resumed, _store, next_provider, next_executor, _batches = staged_runner(tmp_path)
    monkeypatch.setattr(resumed, "_select", forbid_call)
    result = resumed.run()
    assert len(next_provider.calls) == 7
    assert executor.calls + next_executor.calls == 26
    assert result.selection["selected_replica"] == 4


@pytest.mark.parametrize("completion,updates,receipts,final_update", (
    ("initial", 0, 6, 0),
    ("population_boundary", 10, 6, 2),
    ("one_candidate", 16, 6, 0),
    ("entry", 20, 8, 4),
    ("later_boundary", 22, 8, 6),
    ("second_entry", 22, 10, 6),
    ("terminal_entry", 24, 12, 8),
))
def test_early_completion_retains_actual_executed_stage_exit(tmp_path, monkeypatch, completion, updates, receipts, final_update):
    def completes(arm_id, stage, round_number):
        if completion == "one_candidate":
            return stage == EXIT_STAGES[0] and round_number == 0 and arm_id == 4
        coordinate = {
            "initial": (0, 0), "population_boundary": (0, 1), "entry": (1, 2),
            "later_boundary": (1, 3), "second_entry": (2, 3), "terminal_entry": (3, 4),
        }[completion]
        return (EXIT_STAGES.index(stage), round_number) == coordinate

    current, store, _provider, executor, _batches = staged_runner(tmp_path, provider=TrackedProvider(completion=completes))
    result = current.run()
    assert executor.calls == updates
    assert len(result.evaluations) == receipts
    assert result.states[4].update_index == final_update
    assert current.executor.core.coordinator.read(result.states[4]).complete
    for record in result.evaluations:
        stage = EXIT_STAGES[record["stage_ordinal"]]
        assert record["round"] == stage.from_round + (record["update_index"] - stage.start_update) // 2
    final_receipt = result.evaluations[-1]
    assert final_receipt["update_index"] == final_update
    assert final_receipt["state_fingerprint"] == stable_hash(result.states[4].to_dict())
    assert final_receipt["role"] == "certification"
    assert final_receipt["result"]["passed"] is False
    if "entry" in completion:
        ordinal = final_receipt["stage_ordinal"]
        previous_receipt = result.evaluations[-3]
        assert previous_receipt["update_index"] == final_receipt["update_index"]
        assert previous_receipt["policy_fingerprint"] == final_receipt["policy_fingerprint"]
        assert previous_receipt["state_fingerprint"] != final_receipt["state_fingerprint"]
        assert previous_receipt["stage_ordinal"] == ordinal - 1
    before = store_hashes(store)
    resumed, _store, next_provider, next_executor, _batches = staged_runner(tmp_path)
    for method in ("control", "validation", "certification"):
        monkeypatch.setattr(next_provider, method, forbid_call)
    monkeypatch.setattr(next_executor, "step", forbid_call)
    restored = resumed.run()
    assert restored.evaluations == result.evaluations
    assert restored.states[4].to_dict() == result.states[4].to_dict()
    assert before == store_hashes(store)


def test_seven_task_scheduled_results_use_source_initial_coordinates(tmp_path, monkeypatch):
    current, store, provider = selection_runner(tmp_path)
    result = current.run()
    assert len(result.evaluations) == 6
    assert len(provider.calls) == 11
    for record in result.evaluations:
        assert record["stage_id"] == "round91-110"
        assert record["update_index"] == 27000
        assert record["round"] == 90
        assert record["result"]["request"]["metadata"]["scheduled_role_scope"]["round_number"] is None
        assert record["result"]["request"]["task_ids"] == list(REGISTRY.task_ids)
    for candidate, record in zip(result.selection["records"], result.evaluations[:5], strict=True):
        assert canonical_json(candidate["validation"]) == canonical_json(record["result"])
    before = store_hashes(store)
    resumed, _store, next_provider = selection_runner(tmp_path)
    for method in ("control", "validation", "certification"):
        monkeypatch.setattr(next_provider, method, forbid_call)
    assert resumed.run().evaluations == result.evaluations
    assert before == store_hashes(store)


COORDINATE_DEFECTS = {
    "stage_id": "other-stage", "stage_ordinal": 1, "arm_id": 3,
    "role": "control", "update_index": 33000, "round": 110,
    "state_fingerprint": "0" * 64, "policy_fingerprint": "0" * 64,
    "checkpoint_reload_equal": False,
    "predecessor_fingerprint": "0" * 64,
}
BINDING_DEFECTS = ("missing_request", "stale_policy", "wrong_role", "task_order",
                   "scope_stage", "scope_arm", "scope_update", "scope_round",
                   "component_manifest", "scheduled_registry", "request_registry")


def damaged_record(record, defect):
    changed = json.loads(canonical_json(record))
    if defect in COORDINATE_DEFECTS:
        changed[defect] = COORDINATE_DEFECTS[defect]
        return changed
    result = changed["result"]
    request = result["request"]
    metadata_key = "provenance" if record["role"] == "validation" else "estimator_metadata"
    if defect == "missing_request":
        result["request"] = None
    elif defect == "stale_policy":
        request["policy_fingerprint"] = "0" * 64
    elif defect == "wrong_role":
        request["role"] = "control"
    elif defect == "task_order":
        request["task_ids"].reverse()
    elif defect.startswith("scope_"):
        key, value = {"scope_stage": ("stage_id", "round111-130"), "scope_arm": ("arm_id", 3),
                      "scope_update": ("update_index", 27001), "scope_round": ("round_number", 90)}[defect]
        request["metadata"]["scheduled_role_scope"][key] = value
    elif defect == "component_manifest":
        result[metadata_key]["role_bank_manifest_hash"] = "0" * 64
    elif defect == "scheduled_registry":
        result[metadata_key]["scheduled_role_registry_hash"] = "0" * 64
    elif defect == "request_registry":
        request["metadata"]["scheduled_role_registry_hash"] = "0" * 64
    else:
        raise AssertionError(defect)
    return changed


def serialize_misbound_producer_result(path, record, *, pending=False):
    """Model a producer saving a coherent result for the wrong scientific scope."""
    marker_path = path.with_suffix(".complete.json")
    intent_path = path.with_name(f".{path.name}.publication.json")
    payload = json.loads(path.read_text())
    payload["record"] = record
    payload["evaluation_hash"] = stable_hash({key: value for key, value in payload.items() if key != "evaluation_hash"})
    atomic_write_json(path, payload)
    marker = json.loads(marker_path.read_text())
    marker["evaluation_hash"] = payload["evaluation_hash"]
    marker["evaluation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_write_json(marker_path, marker)
    if pending:
        intent = json.loads(intent_path.read_text())
        intent["payload_hash"] = stable_hash(payload)
        intent["marker"] = {key: value for key, value in marker.items() if key != "evaluation_sha256"}
        atomic_write_json(intent_path, intent)
        marker_path.unlink()


@pytest.mark.parametrize("role", ("validation", "certification"))
@pytest.mark.parametrize("mode", ("commit", "completed", "pending"))
@pytest.mark.parametrize("defect", (*COORDINATE_DEFECTS, *BINDING_DEFECTS))
def test_exit_binding_refused_at_commit_and_completed_or_pending_recovery(tmp_path, role, mode, defect):
    current, store, _provider = selection_runner(tmp_path)
    result = current.run()
    record = next(record for record in result.evaluations if record["role"] == role and record["arm_id"] == 4)
    changed = damaged_record(record, defect)
    stage = current.stages[0]
    state = result.states[4]

    def validate(candidate):
        current._validate_evaluation_record(candidate, 4, state, stage, role)

    if mode == "commit":
        with pytest.raises(ReplicationRunnerError):
            store.commit_evaluation(changed, validate)
    else:
        path = receipt_path(store, (0, 4, role))
        serialize_misbound_producer_result(path, changed, pending=mode == "pending")
        with pytest.raises(ReplicationRunnerError):
            store.read_evaluation(0, 4, role, validate)
        if mode == "pending":
            assert not path.with_suffix(".complete.json").exists()


@pytest.mark.parametrize("role", ("validation", "certification"))
@pytest.mark.parametrize("damage", ("raw_payload", "marker_hash", "missing_payload", "missing_marker"))
def test_incomplete_or_changed_receipts_refused_without_reevaluation(tmp_path, monkeypatch, role, damage):
    current, store, _provider = selection_runner(tmp_path)
    current.run()
    path = receipt_path(store, (0, 4, role))
    marker_path = path.with_suffix(".complete.json")
    if damage == "raw_payload":
        payload = json.loads(path.read_text())
        result = payload["record"]["result"]
        result["provenance" if role == "validation" else "upper_records"]["lost_or_changed_raw_outcome"] = [0.5]
        atomic_write_json(path, payload)
    elif damage == "marker_hash":
        marker = json.loads(marker_path.read_text())
        marker["evaluation_sha256"] = "0" * 64
        atomic_write_json(marker_path, marker)
    elif damage == "missing_payload":
        path.unlink()
    else:
        marker_path.unlink()
        path.with_name(f".{path.name}.publication.json").unlink()
    resumed, _store, provider = selection_runner(tmp_path)
    for method in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, method, forbid_call)
    with pytest.raises(ReplicationRunnerError):
        resumed.run()


@pytest.mark.parametrize("slot", ((0, 1, "validation"), (0, 4, "certification"),
                                  (1, 4, "validation"), (1, 4, "certification"),
                                  (2, 4, "certification")))
def test_saved_selection_and_later_stages_require_earlier_receipts(tmp_path, monkeypatch, slot):
    current, store, _provider, _executor, _batches = staged_runner(tmp_path)
    current.run()
    path = receipt_path(store, slot)
    for part in (path, path.with_suffix(".complete.json"), path.with_name(f".{path.name}.publication.json")):
        part.unlink()
    resumed, _store, provider, executor, _batches = staged_runner(tmp_path)
    for method in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, method, forbid_call)
    monkeypatch.setattr(executor, "step", forbid_call)
    with pytest.raises(ReplicationRunnerError, match="missing"):
        resumed.run()


@pytest.mark.parametrize("certificate_state", ("completed", "pending_payload", "pending_intent"))
def test_missing_terminal_validation_is_not_recomputed_after_certification_started(tmp_path, monkeypatch, certificate_state):
    current, store, _provider, _executor, _batches = staged_runner(tmp_path)
    current.run()
    validation = receipt_path(store, (3, 4, "validation"))
    certification = receipt_path(store, (3, 4, "certification"))
    for part in (validation, validation.with_suffix(".complete.json"),
                 validation.with_name(f".{validation.name}.publication.json")):
        part.unlink()
    if certificate_state != "completed":
        certification.with_suffix(".complete.json").unlink()
    if certificate_state == "pending_intent":
        certification.unlink()
    resumed, _store, provider, executor, _batches = staged_runner(tmp_path)
    for method in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, method, forbid_call)
    monkeypatch.setattr(executor, "step", forbid_call)
    with pytest.raises(ReplicationRunnerError, match="missing"):
        resumed.run()


@pytest.mark.parametrize("part", ("payload", "marker", "intent"))
def test_surplus_receipt_files_are_not_silently_ignored(tmp_path, part):
    current, store, _provider = selection_runner(tmp_path)
    current.run()
    source = receipt_path(store, (0, 4, "certification"))
    extra = receipt_path(store, (1, 4, "certification"))
    def component(path):
        if part == "payload":
            return path
        if part == "marker":
            return path.with_suffix(".complete.json")
        return path.with_name(f".{path.name}.publication.json")

    source_part = component(source)
    extra_part = component(extra)
    atomic_write_json(extra_part, json.loads(source_part.read_text()))
    resumed, _store, _provider = selection_runner(tmp_path)
    with pytest.raises(ReplicationRunnerError):
        resumed.run()


def test_certification_provider_is_required(tmp_path):
    provider = TrackedProvider()
    provider.certification = None
    with pytest.raises(ReplicationRunnerError, match="certification provider"):
        staged_runner(tmp_path, provider=provider)
