"""Shared-runner selection and recovery without a historical winner assertion."""

import hashlib
import json
from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest
from tests.contracts.test_generic_replication_calibration import fixture_case
from tests.contracts.test_generic_replication_runner import (
    FakeEvaluationProvider,
    FakeExecutor,
    checkpoint,
    design_for,
)
from tests.contracts.test_generic_replication_training_protocol import arguments

from mooneural.training import generic_replication_training_protocol as protocol
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    GenericReplicationRunner,
    ReplicationRunnerError,
    ReplicationStage,
)
from mooneural.training.generic_training_contracts import stable_hash


class TrackedProvider(FakeEvaluationProvider):
    def __init__(self, selected):
        super().__init__(selected, role_hash="4" * 64)
        self.validations = []
        self.certifications = []

    def validation(self, arm_id, policy, stage, *, update_index):
        self.validations.append((arm_id, stage.stage_id))
        return super().validation(arm_id, policy, stage, update_index=update_index)

    def certification(self, arm_id, policy, stage, *, update_index):
        self.certifications.append((arm_id, stage.stage_id))
        return super().certification(arm_id, policy, stage, update_index=update_index)


def population_runner(directory, *, replicas=(2, 7, 11), selected=7, expected=None, caller_expected=None):
    initial = {arm_id: checkpoint(arm_id) for arm_id in replicas}
    stages = (
        ReplicationStage("population", 0, 1, 1, 0, updates_per_round=2,
                         population_size=len(replicas), select_after=True),
        ReplicationStage("continuation", 1, 2, 2, 2, updates_per_round=2, population_size=1),
    )
    historical_template = design_for({arm_id: checkpoint(arm_id) for arm_id in range(5)})
    design = replace(
        historical_template,
        replica_ids=replicas,
        initial_state_hashes={str(arm_id): stable_hash(state.to_dict()) for arm_id, state in initial.items()},
        expected_selected_replica=expected,
        stages=tuple(asdict(stage) for stage in stages),
    )
    return GenericReplicationRunner(
        FakeExecutor(role_hash=design.role_manifest_sha256),
        {arm_id: object() for arm_id in replicas}, initial, TrackedProvider(selected),
        Mock(return_value=object()), AtomicCheckpointStore(directory, "runner-test", design), stages,
        expected_selected_replica=caller_expected, design=design,
    )


@pytest.mark.parametrize("replicas", ((7,), (2, 7, 11), (1, 2, 3, 7, 11, 13)))
def test_actual_validation_winner_and_complete_population_survive_reload(tmp_path, replicas):
    current = population_runner(tmp_path, replicas=replicas)
    result = current.run()
    assert current.expected_selected_replica is None
    assert result.selection["selected_replica"] == 7
    assert result.selection["candidate_union"] == "all final worker states"
    assert result.selection["candidate_union_ids"] == list(replicas)
    assert [record["replica"] for record in result.selection["records"]] == list(replicas)
    assert result.states[7].update_index == 4
    assert all(state.update_index == 2 for arm_id, state in result.states.items() if arm_id != 7)
    assert current.evaluation_provider.certifications == [(7, "population"), (7, "continuation")]
    assert all(not record["result"]["passed"] for record in result.evaluations if record["role"] == "certification")

    resumed = population_runner(tmp_path, replicas=replicas, selected=replicas[0])
    restored = resumed.run()
    assert restored.selection == result.selection
    assert {arm_id: state.to_dict() for arm_id, state in restored.states.items()} == {
        arm_id: state.to_dict() for arm_id, state in result.states.items()
    }
    assert resumed.executor.calls == 0
    assert not resumed.evaluation_provider.validations
    assert not resumed.evaluation_provider.certifications
    assert resumed.evaluation_provider.control_calls == 0


def test_equal_validation_scores_use_replica_tie_break_not_terminal_evidence(tmp_path):
    current = population_runner(tmp_path, selected=-1)
    result = current.run()
    assert result.selection["selected_replica"] == 2
    assert current.evaluation_provider.certifications == [(2, "population"), (2, "continuation")]


def test_selection_commit_interruption_resumes_without_reselection(tmp_path, monkeypatch):
    uninterrupted = population_runner(tmp_path / "uninterrupted").run()
    interrupted = population_runner(tmp_path / "interrupted")
    commit_selection = interrupted.store.commit_selection

    def interrupt_after_commit(selection):
        commit_selection(selection)
        raise InterruptedError("durable selection interruption")

    monkeypatch.setattr(interrupted.store, "commit_selection", interrupt_after_commit)
    with pytest.raises(InterruptedError, match="durable selection"):
        interrupted.run()
    assert interrupted.store.read_selection()["selected_replica"] == 7
    resumed = population_runner(tmp_path / "interrupted", selected=2)
    restored = resumed.run()
    assert restored.selection == uninterrupted.selection
    assert resumed.evaluation_provider.validations == [(7, "continuation")]
    assert resumed.executor.calls == 2
    assert {arm_id: state.to_dict() for arm_id, state in restored.states.items()} == {
        arm_id: state.to_dict() for arm_id, state in uninterrupted.states.items()
    }


@pytest.mark.parametrize("defect,match", (
    ("missing", "union is incomplete"),
    ("duplicate", "union is not complete"),
    ("winner", "key is not reproducible"),
    ("score", "score provenance"),
    ("five", "candidate union is invalid"),
))
def test_persisted_population_selection_is_revalidated(tmp_path, defect, match):
    current = population_runner(tmp_path)
    current.run()
    path = current.store.root / "selection.json"
    marker_path = current.store.root / "selection.complete.json"
    payload = json.loads(path.read_text())
    selection = payload["selection"]
    if defect == "missing":
        selection["records"].pop()
    elif defect == "duplicate":
        selection["records"][1] = selection["records"][0]
    elif defect == "winner":
        selection["selected_replica"] = 2
    elif defect == "score":
        selection["records"][0]["absolute_threshold_ratio"] = 0.0
    else:
        selection["candidate_union"] = "five final worker states"
    payload["selection_hash"] = stable_hash({key: value for key, value in payload.items() if key != "selection_hash"})
    path.write_text(json.dumps(payload))
    marker = json.loads(marker_path.read_text())
    marker["selection_hash"] = payload["selection_hash"]
    marker["selection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ReplicationRunnerError, match=match):
        population_runner(tmp_path).run()


def test_historical_design_still_asserts_winner_without_caller_override(tmp_path):
    current = population_runner(tmp_path, expected=2, selected=7)
    assert current.expected_selected_replica == 2
    with pytest.raises(ReplicationRunnerError, match="different replica"):
        current.run()
    assert current.store.read_selection()["selected_replica"] == 7
    assert not current.evaluation_provider.certifications


def test_caller_cannot_override_frozen_selection_mode(tmp_path):
    with pytest.raises(ValueError, match="selected replica mismatch"):
        population_runner(tmp_path / "historical", expected=2, caller_expected=7)
    with pytest.raises(ValueError, match="selected replica mismatch"):
        population_runner(tmp_path / "population", caller_expected=7)


def test_new_mode_cannot_resume_historical_design_store(tmp_path):
    population_runner(tmp_path, expected=7).run()
    with pytest.raises(ReplicationRunnerError, match="design binding mismatch"):
        population_runner(tmp_path).run()


def test_historical_five_worker_record_remains_readable(tmp_path):
    current = population_runner(tmp_path, replicas=tuple(range(5)), expected=4, selected=4)
    result = current.run()
    assert result.selection["candidate_union"] == "five final worker states"
    assert result.selection["selected_replica"] == 4
    resumed = population_runner(tmp_path, replicas=tuple(range(5)), expected=4, selected=0)
    assert resumed.run().selection == result.selection
    assert not resumed.evaluation_provider.validations


@pytest.mark.parametrize("replicas", ((7,), (2, 7, 11)))
def test_legacy_nonfive_population_label_can_resume_without_rewriting(tmp_path, replicas, monkeypatch):
    current = population_runner(tmp_path, replicas=replicas, expected=7)
    monkeypatch.setattr(current, "_candidate_union_description", lambda: "five final worker states")
    result = current.run()
    path = current.store.root / "selection.json"
    marker_path = current.store.root / "selection.complete.json"
    legacy_bytes = path.read_bytes(), marker_path.read_bytes()

    resumed = population_runner(tmp_path, replicas=replicas, expected=7, selected=-1)
    restored = resumed.run()
    assert restored.selection == result.selection
    assert restored.selection["candidate_union_ids"] == list(replicas)
    assert (path.read_bytes(), marker_path.read_bytes()) == legacy_bytes
    assert not resumed.evaluation_provider.validations
    assert not resumed.evaluation_provider.certifications
    assert resumed.executor.calls == 0


def test_full_result_consumer_uses_actual_winner_when_design_has_no_assertion(tmp_path):
    case = fixture_case(tmp_path)
    receipt = case.write()
    options = arguments(tmp_path, "full", population_runner)
    options["design"] = population_runner(tmp_path / "unused-population").design
    options["manifest"] = {"source_calibration_binding": case.binding.to_dict()}
    options["source_receipt"] = receipt.reference()
    result = protocol.execute_stage(**options)
    assert result["selection"]["selected_replica"] == 7
    assert result["committed_updates"] == result["aggregate_budget"] == 8
    assert result["selected_budget"] == 4
    assert result["updates_per_arm"] == {"2": 2, "7": 4, "11": 2}
    assert protocol.execute_stage(**options) == result
