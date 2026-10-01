"""Meaningful repair routing and interrupted phase publication checks."""

import json

import pytest
from tests.contracts.test_generic_program import create_attempt, fixture_program, write_json

from mooneural.training import generic_program_boundary as boundary_module
from mooneural.training.generic_program import (
    DECISIONS_PATH,
    GRAPH_END,
    GRAPH_START,
    PLAN_PATH,
    PROGRAM_ID,
    STATE_PATH,
    Program,
    ProgramError,
    file_hash,
)
from mooneural.training.generic_program_boundary import (
    POSITION_END,
    POSITION_START,
    _completion_repair_inputs,
    advance,
    recover_completion_requalification,
    recover_completion_requalification_batch,
    recover_publication,
    repair_completion,
    requalify_completion,
    requalify_completion_batch,
    revise,
)


def cycle_program(root):
    fixture_program(root)
    plan_path = root / PLAN_PATH
    graph = json.loads(plan_path.read_text().split(GRAPH_START)[1].split(GRAPH_END)[0])
    graph["between_phase_cycle"] = "repair_refresh_continue.v1"
    graph["steps"][-1]["optional"] = True
    plan_path.write_text(
        GRAPH_START + json.dumps(graph) + GRAPH_END + "\n" +
        POSITION_START + "\ninitial position\n" + POSITION_END
    )
    state = json.loads((root / STATE_PATH).read_text())
    state["canonical_plan_sha256"] = file_hash(plan_path)
    write_json(root / STATE_PATH, state)
    ledger = json.loads((root / DECISIONS_PATH).read_text())
    ledger["decisions"][0]["master_sha256"] = file_hash(plan_path)
    write_json(root / DECISIONS_PATH, ledger)
    return Program(root)


def record(root, name="refresh", **changes):
    program = Program(root)
    value = {
        "schema": "generic_neural_solver.phase_boundary.v1",
        "activity_id": name, "previous_activity_id": program.state["latest_attempt_id"],
        "master_sha256": program.plan_hash, "step_id": None, "outcome": "REFRESH",
        "evidence": [{"path": "intake.json", "sha256": file_hash(root / "intake.json")}],
        "repairs_completed": [], "repair_actions": [], "external_blockers": [],
    }
    value.update(changes)
    path = root / name / "phase-boundary.json"
    write_json(path, value)
    return path


def test_missing_cycle_blocks_execution_and_refresh_restores_it(tmp_path):
    program = cycle_program(tmp_path)
    assert program.status()["phase_boundary_current"] is False
    with pytest.raises(ProgramError, match="repair/refresh required"):
        program.require_step("work", execution=True)
    boundary = record(tmp_path)
    result = advance(tmp_path, boundary)
    assert result["phase_boundary_current"] is True
    Program(tmp_path).require_step("work", execution=True)
    plan_before = (boundary.parent / "publication/before-master.md").read_text()
    plan_after = (tmp_path / PLAN_PATH).read_text()
    assert plan_before.split(POSITION_START)[0] == plan_after.split(POSITION_START)[0]
    assert plan_before.split(POSITION_END)[1] == plan_after.split(POSITION_END)[1]


@pytest.mark.parametrize("packet_kind", ("artifact", "experiment"))
@pytest.mark.parametrize("nested", (False, True))
def test_publication_refuses_sealed_packet_roots(tmp_path, packet_kind, nested):
    cycle_program(tmp_path)
    packet = tmp_path / "sealed-packet"
    packet.mkdir()
    if packet_kind == "artifact":
        write_json(packet / "sealed-result.json", {"status": "SEALED"})
    else:
        write_json(packet / "attempt-binding.json", {"attempt_id": "sealed"})
        (packet / "hashes.txt").write_text("sealed inventory\n")
    name = "sealed-packet/nested" if nested else "sealed-packet"
    boundary = record(tmp_path, name)
    before = {name: (tmp_path / name).read_bytes() for name in (PLAN_PATH, STATE_PATH, DECISIONS_PATH)}
    with pytest.raises(ProgramError, match="outside sealed protocol"):
        advance(tmp_path, boundary)
    assert not (boundary.parent / "publication").exists()
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())


def test_repair_work_takes_priority_without_waiving_external_blocker(tmp_path):
    cycle_program(tmp_path)
    advance(tmp_path, record(tmp_path))
    path = record(tmp_path, "partial", step_id="work", outcome="PARTIAL",
                  repair_actions=["Bind available sidecars"], external_blockers=["Historical array unavailable"])
    status = advance(tmp_path, path)
    assert status["next_step"] == "work"
    assert next(row for row in status["steps"] if row["step"] == "work")["state"] == "REPAIR_REQUIRED"
    Program(tmp_path).require_step("work")
    with pytest.raises(ProgramError, match="external evidence blocker"):
        Program(tmp_path).require_step("work", execution=True)


def test_external_block_does_not_schedule_optional_regression(tmp_path):
    cycle_program(tmp_path)
    advance(tmp_path, record(tmp_path))
    status = advance(tmp_path, record(tmp_path, "blocked", step_id="work", outcome="PARTIAL",
                                     external_blockers=["Exact source payload absent in searched scopes"]))
    assert status["next_step"] is None
    assert "unimplemented" in status["ready_steps"]
    assert next(row for row in status["steps"] if row["step"] == "training")["state"] == "BLOCKED"


@pytest.mark.parametrize("changes", (
    {"previous_activity_id": "stale"},
    {"master_sha256": "a" * 64},
    {"outcome": "COMPLETE", "step_id": "work", "repair_actions": ["Unfixed bug"]},
    {"outcome": "PARTIAL", "step_id": "work"},
    {"outcome": "REFRESH", "completion": {"kind": "protocol"}},
    {"outcome": "REFRESH", "repair_actions": ["unresolved arithmetic"]},
    {"outcome": "REFRESH", "external_blockers": ["missing source"]},
))
def test_bad_boundary_refuses_before_publication(tmp_path, changes):
    cycle_program(tmp_path)
    path = record(tmp_path, **changes)
    before = {name: (tmp_path / name).read_bytes() for name in (PLAN_PATH, STATE_PATH, DECISIONS_PATH)}
    with pytest.raises(ProgramError):
        advance(tmp_path, path)
    assert not (path.parent / "publication").exists()
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())


def test_interrupted_publication_recovers_without_repeating_result(tmp_path, monkeypatch):
    cycle_program(tmp_path)
    path = record(tmp_path)
    original_write = boundary_module.atomic_write_text

    def fail_before_live_state(target, content):
        if target == tmp_path / STATE_PATH:
            raise OSError("simulated interruption")
        return original_write(target, content)

    monkeypatch.setattr(boundary_module, "atomic_write_text", fail_before_live_state)
    with pytest.raises(OSError, match="simulated interruption"):
        advance(tmp_path, path)
    with pytest.raises(ProgramError, match="binding mismatch"):
        Program(tmp_path)
    monkeypatch.setattr(boundary_module, "atomic_write_text", original_write)
    journal = path.parent / "publication/publication.json"
    assert recover_publication(tmp_path, journal)["phase_boundary_current"]
    state_before = (tmp_path / STATE_PATH).read_bytes()
    assert advance(tmp_path, path)["phase_boundary_current"]
    assert (tmp_path / STATE_PATH).read_bytes() == state_before
    ledger = json.loads((tmp_path / DECISIONS_PATH).read_text())
    assert len(ledger["decisions"]) == 2


def test_recovery_preserves_intervening_user_edit(tmp_path):
    cycle_program(tmp_path)
    path = record(tmp_path)
    advance(tmp_path, path)
    state_path = tmp_path / STATE_PATH
    state_path.write_text(state_path.read_text() + "\n")
    changed = state_path.read_bytes()
    with pytest.raises(ProgramError, match="outside this publication"):
        recover_publication(tmp_path, path.parent / "publication/publication.json")
    assert state_path.read_bytes() == changed


def test_structural_master_revision_publishes_and_reuses_journal(tmp_path):
    cycle_program(tmp_path)
    revision_root = tmp_path / "revision"
    proposed_path = revision_root / "proposed-master.md"
    review_path = revision_root / "review.json"
    evidence_path = revision_root / "diagnostic.json"
    proposed_path.parent.mkdir(parents=True)
    proposed_path.write_text(
        (tmp_path / PLAN_PATH).read_text().replace("initial position", "revised position")
    )
    write_json(evidence_path, {"checked": True})
    write_json(review_path, {"decision": "PASS", "reviewer_id": "independent-reviewer"})
    manifest_path = revision_root / "manifest.json"
    write_json(manifest_path, {
        "schema": "generic_neural_solver.master_revision.v1",
        "activity_id": "master-revision",
        "previous_activity_id": "previous",
        "executor_id": "master-repair",
        "previous_master_sha256": Program(tmp_path).plan_hash,
        "proposed_plan": {"path": str(proposed_path.relative_to(tmp_path)),
                          "sha256": file_hash(proposed_path)},
        "review": {"path": str(review_path.relative_to(tmp_path)),
                   "sha256": file_hash(review_path)},
        "evidence": [{"path": str(evidence_path.relative_to(tmp_path)),
                      "sha256": file_hash(evidence_path)}],
    })

    result = revise(tmp_path, manifest_path)
    assert result["phase_boundary_current"] is True
    assert Program(tmp_path).plan_hash == result["master_sha256"]
    assert revise(tmp_path, manifest_path)["phase_boundary_current"] is True


def test_phase_evidence_cannot_bind_live_ledgers(tmp_path):
    cycle_program(tmp_path)
    path = record(tmp_path, evidence=[
        {"path": PLAN_PATH, "sha256": file_hash(tmp_path / PLAN_PATH)},
    ])
    with pytest.raises(ProgramError, match="mutable live ledgers"):
        advance(tmp_path, path)
    assert not (path.parent / "publication").exists()


def _completion_repair_fixture(tmp_path):
    governance = tmp_path / "src/mooneural/training/generic_program_boundary.py"
    stable = tmp_path / "stable-input.txt"
    governance.parent.mkdir(parents=True)
    governance.write_text("original governance source\n")
    stable.write_text("stable scientific input\n")
    governance_reference = {
        "path": "src/mooneural/training/generic_program_boundary.py",
        "sha256": file_hash(governance),
    }
    stable_reference = {"path": "stable-input.txt", "sha256": file_hash(stable)}
    report = {
        "program_id": PROGRAM_ID,
        "step_id": "work",
        "decision": "READY",
        "executor_id": "original-executor",
        "question": "Fixture completion metadata",
        "protected_claim": "Fixture only",
        "comparator": "Stored fixture",
        "primary_criterion": "Stored result",
        "vetoes": ["binding drift"],
        "nonclaims": ["No scientific claim"],
        "observations": {"value": 1.0},
        "inputs": [governance_reference, stable_reference],
    }
    write_json(tmp_path / "report.json", report)
    report_reference = {"path": "report.json", "sha256": file_hash(tmp_path / "report.json")}
    review = {
        "decision": "PASS",
        "auditor_id": "original-auditor",
        "report_sha256": report_reference["sha256"],
    }
    write_json(tmp_path / "review.json", review)
    review_reference = {"path": "review.json", "sha256": file_hash(tmp_path / "review.json")}
    record = {"kind": "audited_report", "report": report_reference, "review": review_reference}
    steps = [{
        "id": "work", "phase_id": "work", "requires": [], "lane": "Validate",
        "mode": "host", "evidence_kind": "audited_report", "pass_decision": "READY",
        "entrypoint": None, "closure": "Fixture completion metadata",
    }]
    fixture_program(tmp_path, steps=steps, records={"work": record})
    state_path = tmp_path / STATE_PATH
    state = json.loads(state_path.read_text())
    state["step_followups"] = {
        "work": {
            "outcome": "COMPLETE",
            "repair_actions": [],
            "external_blockers": [],
            "advisory_followups": [],
            "repairs_completed": [],
            "boundary": {"path": "old-boundary.json", "sha256": "a" * 64},
        }
    }
    write_json(state_path, state)
    old_plan_hash = file_hash(tmp_path / PLAN_PATH)
    repair_root = tmp_path / "repair"
    memo = repair_root / "repair.json"
    write_json(memo, {"reason": "Remove unrelated mutable governance input"})
    new_report = json.loads((tmp_path / "report.json").read_text())
    new_report["inputs"] = [stable_reference]
    new_report["metadata_repair"] = {
        "kind": "completion_metadata_only",
        "old_report": report_reference,
        "removed_inputs": [governance_reference],
        "numerical_rerun": False,
        "claim_unchanged": True,
    }
    new_report_path = repair_root / "report.json"
    write_json(new_report_path, new_report)
    new_report_reference = {
        "path": str(new_report_path.relative_to(tmp_path)),
        "sha256": file_hash(new_report_path),
    }
    new_review_path = repair_root / "review.json"
    write_json(new_review_path, {
        "decision": "PASS",
        "auditor_id": "metadata-auditor",
        "report_sha256": new_report_reference["sha256"],
    })
    new_review_reference = {
        "path": str(new_review_path.relative_to(tmp_path)),
        "sha256": file_hash(new_review_path),
    }
    manifest_path = repair_root / "manifest.json"
    write_json(manifest_path, {
        "schema": "generic_neural_solver.completion_metadata_repair.v1",
        "activity_id": "completion-metadata-repair",
        "previous_activity_id": "previous",
        "executor_id": "repair-executor",
        "step_id": "work",
        "previous_master_sha256": old_plan_hash,
        "old_completion": record,
        "replacement_report": new_report_reference,
        "replacement_review": new_review_reference,
        "removed_inputs": [governance_reference],
        "reason": "Remove unrelated mutable governance input",
        "evidence": [
            {"path": str(memo.relative_to(tmp_path)), "sha256": file_hash(memo)},
            report_reference,
            review_reference,
            new_report_reference,
            new_review_reference,
        ],
    })
    return manifest_path, governance, report_reference, new_report_reference


def test_completion_metadata_repair_preserves_result_after_governance_drift(tmp_path):
    manifest_path, governance, old_report, new_report = _completion_repair_fixture(tmp_path)
    governance.write_text("unrelated governance change\n")
    with pytest.raises(ProgramError, match="closure input changed"):
        Program(tmp_path)
    status = repair_completion(tmp_path, manifest_path)
    assert status["phase_boundary_current"] is True
    assert Program(tmp_path).complete == {"work"}
    assert json.loads((tmp_path / old_report["path"]).read_text())["inputs"][0]["path"] == "src/mooneural/training/generic_program_boundary.py"
    governance.write_text("another unrelated governance change\n")
    assert Program(tmp_path).complete == {"work"}
    assert repair_completion(tmp_path, manifest_path)["phase_boundary_current"] is True
    assert Program(tmp_path).records["work"]["report"] == new_report


def test_completion_metadata_repair_cannot_remove_scientific_input(tmp_path):
    _manifest_path, _governance, old_report_reference, _new_report_reference = _completion_repair_fixture(tmp_path)
    old_report = json.loads((tmp_path / old_report_reference["path"]).read_text())
    replacement = dict(old_report)
    replacement["inputs"] = [old_report["inputs"][0]]
    with pytest.raises(ProgramError, match="only program-governance inputs"):
        _completion_repair_inputs(
            tmp_path, old_report, replacement, [old_report["inputs"][1]]
        )


def test_completion_metadata_repair_recovers_after_live_write_interruption(tmp_path, monkeypatch):
    manifest_path, governance, _old_report, _new_report = _completion_repair_fixture(tmp_path)
    governance.write_text("unrelated governance change\n")
    original = boundary_module.atomic_write_text

    def interrupt_live_state(target, content):
        if target == tmp_path / STATE_PATH:
            raise OSError("interrupted completion repair")
        return original(target, content)

    monkeypatch.setattr(boundary_module, "atomic_write_text", interrupt_live_state)
    with pytest.raises(OSError, match="interrupted completion repair"):
        repair_completion(tmp_path, manifest_path)
    monkeypatch.setattr(boundary_module, "atomic_write_text", original)
    assert repair_completion(tmp_path, manifest_path)["phase_boundary_current"] is True
    assert Program(tmp_path).complete == {"work"}


def _completion_requalification_fixture(
    tmp_path, *, step_id="p4.component-audit",
    input_path="src/mooneural/training/rotemberg_input_contracts.py",
):
    stale_input = tmp_path / input_path
    validation = tmp_path / "validation.json"
    stale_input.parent.mkdir(parents=True)
    stale_input.write_text("old implementation\n")
    old_input = {
        "path": input_path,
        "sha256": file_hash(stale_input),
    }
    old_report = {
        "program_id": PROGRAM_ID,
        "step_id": step_id,
        "decision": "READY",
        "executor_id": "original-executor",
        "question": "Fixture implementation requalification",
        "protected_claim": "Fixture only",
        "comparator": "Stored fixture",
        "primary_criterion": "Stored result",
        "vetoes": ["binding drift"],
        "nonclaims": ["No scientific claim"],
        "observations": {"value": 1.0},
        "inputs": [old_input],
    }
    write_json(tmp_path / "old-report.json", old_report)
    old_report_reference = {
        "path": "old-report.json", "sha256": file_hash(tmp_path / "old-report.json")
    }
    old_review = {
        "decision": "PASS",
        "auditor_id": "original-auditor",
        "report_sha256": old_report_reference["sha256"],
    }
    write_json(tmp_path / "old-review.json", old_review)
    old_review_reference = {
        "path": "old-review.json", "sha256": file_hash(tmp_path / "old-review.json")
    }
    old_record = {
        "kind": "audited_report",
        "report": old_report_reference,
        "review": old_review_reference,
    }
    steps = [{
        "id": step_id, "phase_id": step_id, "requires": [], "lane": "Validate",
        "mode": "host", "evidence_kind": "audited_report", "pass_decision": "READY",
        "entrypoint": None, "closure": "Fixture implementation requalification",
    }]
    fixture_program(tmp_path, steps=steps, records={step_id: old_record})
    state_path = tmp_path / STATE_PATH
    state = json.loads(state_path.read_text())
    state["step_followups"] = {
        step_id: {
            "outcome": "COMPLETE",
            "repair_actions": [],
            "external_blockers": [],
            "advisory_followups": [],
            "repairs_completed": [],
            "boundary": {"path": "old-boundary.json", "sha256": "a" * 64},
        }
    }
    write_json(state_path, state)
    stale_input.write_text("current implementation\n")
    current_input = {
        "path": input_path,
        "sha256": file_hash(stale_input),
    }
    changed_inputs = [{
        "path": input_path,
        "old_sha256": old_input["sha256"],
        "new_sha256": current_input["sha256"],
    }]
    write_json(validation, {
        "schema": "generic_neural_solver.completion_requalification_validation.v1",
        "decision": "CURRENT_IMPLEMENTATION_INPUTS_REQUALIFICATION_PASSED",
        "step_id": step_id,
        "changed_inputs": changed_inputs,
        "claim_unchanged": True,
        "numerical_rerun": False,
        "model_execution": False,
        "training_updates": 0,
    })
    validation_reference = {
        "path": "validation.json", "sha256": file_hash(validation)
    }
    new_report = dict(old_report)
    new_report["inputs"] = [current_input]
    new_report["requalification"] = {
        "kind": "implementation_requalification",
        "old_completion": old_record,
        "changed_inputs": changed_inputs,
        "validation_evidence": [validation_reference],
        "claim_unchanged": True,
        "numerical_rerun": False,
    }
    replacement_root = tmp_path / "requalification"
    new_report_path = replacement_root / "report.json"
    write_json(new_report_path, new_report)
    new_report_reference = {
        "path": str(new_report_path.relative_to(tmp_path)),
        "sha256": file_hash(new_report_path),
    }
    new_review_path = replacement_root / "review.json"
    write_json(new_review_path, {
        "decision": "PASS",
        "auditor_id": "independent-requalification-auditor",
        "report_sha256": new_report_reference["sha256"],
        "requalification": {
            "claim_unchanged": True,
            "changed_inputs": changed_inputs,
            "validation_evidence": [validation_reference],
        },
    })
    new_review_reference = {
        "path": str(new_review_path.relative_to(tmp_path)),
        "sha256": file_hash(new_review_path),
    }
    manifest_path = replacement_root / "manifest.json"
    write_json(manifest_path, {
        "schema": "generic_neural_solver.completion_requalification.v1",
        "activity_id": "completion-requalification",
        "previous_activity_id": "previous",
        "executor_id": "requalification-executor",
        "step_id": step_id,
        "previous_master_sha256": file_hash(tmp_path / PLAN_PATH),
        "old_completion": old_record,
        "replacement_report": new_report_reference,
        "replacement_review": new_review_reference,
        "changed_inputs": changed_inputs,
        "validation_evidence": [validation_reference],
        "reason": "Requalify the completion against the current implementation input.",
        "evidence": [
            old_report_reference, old_review_reference,
            new_report_reference, new_review_reference, validation_reference,
        ],
    })
    return manifest_path, stale_input, old_report_reference, new_report_reference


def test_completion_requalification_restores_stale_implementation_binding(tmp_path):
    manifest_path, stale_input, old_report, new_report = _completion_requalification_fixture(tmp_path)
    with pytest.raises(ProgramError, match="closure input changed"):
        Program(tmp_path)
    status = requalify_completion(tmp_path, manifest_path)
    assert status["phase_boundary_current"] is True
    assert Program(tmp_path).complete == {"p4.component-audit"}
    assert json.loads((tmp_path / old_report["path"]).read_text())["inputs"][0]["sha256"] != file_hash(stale_input)
    assert Program(tmp_path).records["p4.component-audit"]["report"] == new_report
    assert requalify_completion(tmp_path, manifest_path)["phase_boundary_current"] is True


def test_source_completion_requalification_accepts_current_shared_policy(tmp_path):
    manifest_path, stale_input, old_report, new_report = _completion_requalification_fixture(
        tmp_path, step_id="p5r.source",
        input_path="src/mooneural/training/generic_permanent_pass.py",
    )
    original_bytes = (tmp_path / old_report["path"]).read_bytes()
    with pytest.raises(ProgramError, match="closure input changed"):
        Program(tmp_path)
    status = requalify_completion(tmp_path, manifest_path)
    assert status["phase_boundary_current"] is True
    assert Program(tmp_path).complete == {"p5r.source"}
    assert Program(tmp_path).records["p5r.source"]["report"] == new_report
    assert (tmp_path / old_report["path"]).read_bytes() == original_bytes
    replacement = json.loads((tmp_path / new_report["path"]).read_text())
    assert replacement["inputs"][0]["sha256"] == file_hash(stale_input)
    assert replacement["observations"] == {"value": 1.0}


def test_sgu_objective_requalification_preserves_historical_result(tmp_path):
    manifest_path, _stale_input, old_report, new_report = _completion_requalification_fixture(
        tmp_path, step_id="repair.sgu178-candidate-lifecycle",
        input_path="src/mooneural/training/sgu_equilibrium_objective.py")
    original = (tmp_path / old_report["path"]).read_bytes()
    assert requalify_completion(tmp_path, manifest_path)["phase_boundary_current"] is True
    assert Program(tmp_path).records["repair.sgu178-candidate-lifecycle"]["report"] == new_report
    assert (tmp_path / old_report["path"]).read_bytes() == original
    assert json.loads((tmp_path / new_report["path"]).read_text())["observations"] == {"value": 1.0}


@pytest.mark.parametrize("step_id,input_path", (
    ("p4.component-audit", "src/mooneural/training/generic_permanent_pass.py"),
    ("p4.adapter", "src/mooneural/training/generic_permanent_pass.py"),
    ("p5r.source", "src/mooneural/training/generic_replication_runner.py"),
    ("p5r.source", "src/mooneural/training/rotemberg_public_objective_worker.py"),
    ("repair.sgu178-candidate-lifecycle", "src/mooneural/training/sgu_xla_kernel.py"),
))
def test_shared_policy_requalification_does_not_expand_other_paths(tmp_path, step_id, input_path):
    manifest_path, _stale_input, _old_report, _new_report = _completion_requalification_fixture(
        tmp_path, step_id=step_id, input_path=input_path,
    )
    before = {path: (tmp_path / path).read_bytes() for path in (PLAN_PATH, STATE_PATH, DECISIONS_PATH)}
    with pytest.raises(ProgramError, match="input is not allowed for this step"):
        requalify_completion(tmp_path, manifest_path)
    assert before == {path: (tmp_path / path).read_bytes() for path in before}


@pytest.mark.parametrize("step_id,input_path", (
    ("p4.component-audit", "src/mooneural/training/rotemberg_input_contracts.py"),
    ("p5r.source", "src/mooneural/training/generic_permanent_pass.py"),
    ("repair.sgu178-candidate-lifecycle", "src/mooneural/training/sgu_equilibrium_objective.py"),
))
def test_completion_requalification_rejects_scientific_report_change(tmp_path, step_id, input_path):
    manifest_path, _stale_input, _old_report, _new_report = _completion_requalification_fixture(
        tmp_path, step_id=step_id, input_path=input_path,
    )
    report_path = tmp_path / "requalification/report.json"
    report = json.loads(report_path.read_text())
    report["observations"]["value"] = 2.0
    write_json(report_path, report)
    manifest = json.loads(manifest_path.read_text())
    new_report_reference = {
        "path": "requalification/report.json", "sha256": file_hash(report_path)
    }
    review_path = tmp_path / "requalification/review.json"
    review = json.loads(review_path.read_text())
    review["report_sha256"] = new_report_reference["sha256"]
    write_json(review_path, review)
    new_review_reference = {
        "path": "requalification/review.json", "sha256": file_hash(review_path)
    }
    manifest["replacement_report"] = new_report_reference
    manifest["replacement_review"] = new_review_reference
    manifest["evidence"][2:4] = [new_report_reference, new_review_reference]
    write_json(manifest_path, manifest)
    with pytest.raises(ProgramError, match="scientific report"):
        requalify_completion(tmp_path, manifest_path)


@pytest.mark.parametrize("auditor_id", ("original-executor", "requalification-executor"))
def test_source_policy_requalification_rejects_self_review(tmp_path, auditor_id):
    manifest_path, _stale_input, _old_report, _new_report = _completion_requalification_fixture(
        tmp_path, step_id="p5r.source",
        input_path="src/mooneural/training/generic_permanent_pass.py",
    )
    manifest = json.loads(manifest_path.read_text())
    review_path = tmp_path / manifest["replacement_review"]["path"]
    review = json.loads(review_path.read_text())
    review["auditor_id"] = auditor_id
    write_json(review_path, review)
    manifest["replacement_review"]["sha256"] = file_hash(review_path)
    manifest["evidence"][3] = manifest["replacement_review"]
    write_json(manifest_path, manifest)
    before = {path: (tmp_path / path).read_bytes() for path in (PLAN_PATH, STATE_PATH, DECISIONS_PATH)}
    with pytest.raises(ProgramError, match="missing or self-audited"):
        requalify_completion(tmp_path, manifest_path)
    assert before == {path: (tmp_path / path).read_bytes() for path in before}


def test_completion_requalification_recovers_after_live_write_interruption(tmp_path, monkeypatch):
    manifest_path, _stale_input, _old_report, _new_report = _completion_requalification_fixture(tmp_path)
    original = boundary_module.atomic_write_text

    def interrupt_live_state(target, content):
        if target == tmp_path / STATE_PATH:
            raise OSError("interrupted completion requalification")
        return original(target, content)

    monkeypatch.setattr(boundary_module, "atomic_write_text", interrupt_live_state)
    with pytest.raises(OSError, match="interrupted completion requalification"):
        requalify_completion(tmp_path, manifest_path)
    monkeypatch.setattr(boundary_module, "atomic_write_text", original)
    journal = manifest_path.parent / "publication/publication.json"
    assert recover_completion_requalification(tmp_path, journal)["phase_boundary_current"] is True
    assert Program(tmp_path).complete == {"p4.component-audit"}


def _completion_requalification_batch_fixture(tmp_path):
    changed_path = tmp_path / "src/mooneural/training/rotemberg_input_contracts.py"
    changed_path.parent.mkdir(parents=True)
    changed_path.write_text("old input contract\n")
    stable_path = tmp_path / "stable-input.txt"
    stable_path.write_text("stable scientific input\n")
    changed_old_reference = {
        "path": "src/mooneural/training/rotemberg_input_contracts.py",
        "sha256": file_hash(changed_path),
    }
    stable_reference = {
        "path": "stable-input.txt", "sha256": file_hash(stable_path)
    }
    steps = []
    records = {}
    replacements = []
    evidence = []
    for step_id in ("p4.component-audit", "p4.adapter"):
        report = {
            "program_id": PROGRAM_ID,
            "step_id": step_id,
            "decision": "READY",
            "executor_id": "original-executor",
            "observations": {"step": step_id},
            "inputs": [changed_old_reference, stable_reference],
        }
        report_path = tmp_path / (step_id + "-report.json")
        write_json(report_path, report)
        report_reference = {
            "path": str(report_path.relative_to(tmp_path)),
            "sha256": file_hash(report_path),
        }
        review_path = tmp_path / (step_id + "-review.json")
        write_json(review_path, {
            "decision": "PASS",
            "auditor_id": "original-auditor",
            "report_sha256": report_reference["sha256"],
        })
        review_reference = {
            "path": str(review_path.relative_to(tmp_path)),
            "sha256": file_hash(review_path),
        }
        old_record = {
            "kind": "audited_report",
            "report": report_reference,
            "review": review_reference,
        }
        records[step_id] = old_record
        steps.append({
            "id": step_id,
            "phase_id": step_id,
            "requires": [],
            "lane": "Validate",
            "mode": "host",
            "evidence_kind": "audited_report",
            "pass_decision": "READY",
            "entrypoint": None,
            "closure": "Batch fixture",
        })
        evidence.extend([report_reference, review_reference])
        replacements.append((step_id, old_record, report_reference, review_reference))
    fixture_program(tmp_path, steps=steps, records=records)
    state_path = tmp_path / STATE_PATH
    state = json.loads(state_path.read_text())
    state["step_followups"] = {
        step_id: {
            "outcome": "COMPLETE",
            "repair_actions": [],
            "external_blockers": [],
            "advisory_followups": [],
            "repairs_completed": [],
        }
        for step_id in records
    }
    write_json(state_path, state)
    plan_hash = file_hash(tmp_path / PLAN_PATH)
    changed_path.write_text("new input contract\n")
    changed_new_reference = {
        "path": changed_old_reference["path"],
        "sha256": file_hash(changed_path),
    }
    validation_references = []
    entries = []
    for step_id, old_record, old_report_reference, old_review_reference in replacements:
        changed_inputs = [{
            "path": changed_old_reference["path"],
            "old_sha256": changed_old_reference["sha256"],
            "new_sha256": changed_new_reference["sha256"],
        }]
        validation_path = tmp_path / (step_id + "-validation.json")
        write_json(validation_path, {
            "schema": "generic_neural_solver.completion_requalification_validation.v1",
            "decision": "CURRENT_IMPLEMENTATION_INPUTS_REQUALIFICATION_PASSED",
            "step_id": step_id,
            "changed_inputs": changed_inputs,
            "claim_unchanged": True,
            "numerical_rerun": False,
            "model_execution": False,
            "training_updates": 0,
        })
        validation_reference = {
            "path": str(validation_path.relative_to(tmp_path)),
            "sha256": file_hash(validation_path),
        }
        validation_references.append(validation_reference)
        new_report = json.loads((tmp_path / old_report_reference["path"]).read_text())
        new_report["inputs"] = [changed_new_reference, stable_reference]
        new_report["requalification"] = {
            "kind": "implementation_requalification",
            "old_completion": old_record,
            "changed_inputs": changed_inputs,
            "validation_evidence": [validation_reference],
            "claim_unchanged": True,
            "numerical_rerun": False,
        }
        new_report_path = tmp_path / (step_id + "-replacement-report.json")
        write_json(new_report_path, new_report)
        new_report_reference = {
            "path": str(new_report_path.relative_to(tmp_path)),
            "sha256": file_hash(new_report_path),
        }
        new_review_path = tmp_path / (step_id + "-replacement-review.json")
        write_json(new_review_path, {
            "decision": "PASS",
            "auditor_id": "independent-batch-auditor",
            "report_sha256": new_report_reference["sha256"],
            "requalification": {
                "claim_unchanged": True,
                "changed_inputs": changed_inputs,
                "validation_evidence": [validation_reference],
            },
        })
        new_review_reference = {
            "path": str(new_review_path.relative_to(tmp_path)),
            "sha256": file_hash(new_review_path),
        }
        evidence.extend([
            new_report_reference, new_review_reference, validation_reference,
        ])
        entries.append({
            "step_id": step_id,
            "old_completion": old_record,
            "replacement_report": new_report_reference,
            "replacement_review": new_review_reference,
            "changed_inputs": changed_inputs,
            "validation_evidence": [validation_reference],
            "reason": "Refresh the batch fixture input binding",
        })
    manifest_path = tmp_path / "batch-requalification/manifest.json"
    write_json(manifest_path, {
        "schema": "generic_neural_solver.completion_requalification_batch.v1",
        "activity_id": "completion-requalification-batch",
        "previous_activity_id": "previous",
        "executor_id": "batch-executor",
        "previous_master_sha256": plan_hash,
        "created_at": "2026-09-12T00:00:00+00:00",
        "reason": "Refresh stale batch fixture input bindings",
        "entries": entries,
        "evidence": evidence,
    })
    return manifest_path


def _add_boundary_refresh_to_batch_fixture(tmp_path, manifest_path):
    plan_path = tmp_path / PLAN_PATH
    old_master_sha256 = file_hash(plan_path)
    graph = json.loads(plan_path.read_text().split(GRAPH_START)[1].split(GRAPH_END)[0])
    graph["between_phase_cycle"] = "repair_refresh_continue.v1"
    graph["steps"].append({
        "id": "p5r.design",
        "phase_id": "p5r.design",
        "requires": [],
        "lane": "Validate",
        "mode": "host",
        "evidence_kind": "audited_report",
        "pass_decision": "REPLICATION_DESIGN_FROZEN",
        "entrypoint": None,
        "closure": "Boundary refresh fixture",
    })
    plan_path.write_text(GRAPH_START + json.dumps(graph) + GRAPH_END)
    new_master_sha256 = file_hash(plan_path)

    changed_path = tmp_path / "boundary-refresh/input.txt"
    changed_path.parent.mkdir(parents=True)
    changed_path.write_text("old boundary input\n")
    old_input_reference = {
        "path": str(changed_path.relative_to(tmp_path)),
        "sha256": file_hash(changed_path),
    }
    changed_path.write_text("current boundary input\n")
    new_input_reference = {
        "path": old_input_reference["path"],
        "sha256": file_hash(changed_path),
    }
    changed_inputs = [{
        "path": old_input_reference["path"],
        "old_sha256": old_input_reference["sha256"],
        "new_sha256": new_input_reference["sha256"],
    }]
    old_boundary_path = tmp_path / "boundary-refresh/old-boundary.json"
    write_json(old_boundary_path, {
        "schema": "generic_neural_solver.phase_boundary.v1",
        "activity_id": "old-boundary-activity",
        "previous_activity_id": "previous",
        "master_sha256": old_master_sha256,
        "step_id": "p5r.design",
        "outcome": "PARTIAL",
        "completion": None,
        "repairs_completed": [],
        "repair_actions": ["Open design work"],
        "external_blockers": [],
        "advisory_followups": ["Old boundary evidence"],
        "evidence": [old_input_reference],
    })
    old_boundary_reference = {
        "path": str(old_boundary_path.relative_to(tmp_path)),
        "sha256": file_hash(old_boundary_path),
    }
    validation_path = tmp_path / "boundary-refresh/validation.json"
    write_json(validation_path, {
        "schema": "generic_neural_solver.boundary_evidence_refresh_validation.v1",
        "decision": "BOUNDARY_EVIDENCE_REFRESH_PASSED",
        "step_id": "p5r.design",
        "old_boundary": old_boundary_reference,
        "old_master_sha256": old_master_sha256,
        "new_master_sha256": new_master_sha256,
        "changed_inputs": changed_inputs,
        "claim_unchanged": True,
        "numerical_rerun": False,
        "model_execution": False,
        "training_updates": 0,
    })
    validation_reference = {
        "path": str(validation_path.relative_to(tmp_path)),
        "sha256": file_hash(validation_path),
    }
    new_boundary_path = tmp_path / "boundary-refresh/new-boundary.json"
    write_json(new_boundary_path, {
        "schema": "generic_neural_solver.phase_boundary.v1",
        "activity_id": "boundary-refresh-batch",
        "previous_activity_id": "old-boundary-activity",
        "master_sha256": new_master_sha256,
        "previous_master_sha256": old_master_sha256,
        "step_id": "p5r.design",
        "outcome": "REPAIRED",
        "repair_kind": "BOUNDARY_EVIDENCE_REFRESH",
        "completion": None,
        "repairs_completed": ["Refresh the stale boundary input binding"],
        "repair_actions": ["Open design work"],
        "external_blockers": [],
        "advisory_followups": [
            "Old boundary evidence",
            "This refresh changes only evidence fingerprints; the p5r.design partial claim remains open.",
        ],
        "evidence": [new_input_reference, validation_reference],
        "boundary_refresh": {
            "kind": "boundary_evidence_refresh",
            "old_boundary": old_boundary_reference,
            "old_master_sha256": old_master_sha256,
            "new_master_sha256": new_master_sha256,
            "changed_inputs": changed_inputs,
            "validation_evidence": [validation_reference],
            "claim_unchanged": True,
            "numerical_rerun": False,
        },
    })
    new_boundary_reference = {
        "path": str(new_boundary_path.relative_to(tmp_path)),
        "sha256": file_hash(new_boundary_path),
    }

    state_path = tmp_path / STATE_PATH
    state = json.loads(state_path.read_text())
    state["canonical_plan_sha256"] = new_master_sha256
    state["step_followups"]["p5r.design"] = {
        "outcome": "PARTIAL",
        "repair_actions": ["Open design work"],
        "external_blockers": [],
        "advisory_followups": [],
        "repairs_completed": [],
        "boundary": old_boundary_reference,
    }
    write_json(state_path, state)
    ledger_path = tmp_path / DECISIONS_PATH
    ledger = json.loads(ledger_path.read_text())
    ledger["decisions"][0]["master_sha256"] = new_master_sha256
    write_json(ledger_path, ledger)

    manifest = json.loads(manifest_path.read_text())
    manifest["previous_master_sha256"] = new_master_sha256
    manifest["activity_id"] = "boundary-refresh-batch"
    manifest["executor_id"] = "boundary-refresh-executor"
    manifest["evidence"].extend([
        old_boundary_reference,
        validation_reference,
    ])
    manifest["boundary_refresh"] = {
        "step_id": "p5r.design",
        "old_boundary": old_boundary_reference,
        "replacement_boundary": new_boundary_reference,
        "old_master_sha256": old_master_sha256,
        "new_master_sha256": new_master_sha256,
        "changed_inputs": changed_inputs,
        "validation_evidence": [validation_reference],
        "reason": "Refresh the stale boundary input binding",
    }
    write_json(manifest_path, manifest)
    return manifest_path, new_boundary_reference


def test_completion_requalification_batch_updates_all_stale_bindings_atomically(tmp_path):
    manifest_path = _completion_requalification_batch_fixture(tmp_path)
    with pytest.raises(ProgramError, match="closure input changed"):
        Program(tmp_path)
    status = requalify_completion_batch(tmp_path, manifest_path)
    assert status["phase_boundary_current"] is True
    assert Program(tmp_path).complete == {"p4.component-audit", "p4.adapter"}
    state = json.loads((tmp_path / STATE_PATH).read_text())
    assert state["latest_activity_kind"] == "COMPLETION_REQUALIFICATION_BATCH"
    assert len(state["step_followups"]["p4.component-audit"]["completion_requalifications"]) == 1
    assert len(state["step_followups"]["p4.adapter"]["completion_requalifications"]) == 1
    assert requalify_completion_batch(tmp_path, manifest_path)["phase_boundary_current"] is True


def test_completion_requalification_batch_refreshes_stale_followup_boundary(tmp_path):
    manifest_path = _completion_requalification_batch_fixture(tmp_path)
    manifest_path, replacement_boundary = _add_boundary_refresh_to_batch_fixture(
        tmp_path, manifest_path
    )

    status = requalify_completion_batch(tmp_path, manifest_path)

    assert status["phase_boundary_current"] is True
    assert status["next_step"] == "p5r.design"
    state = json.loads((tmp_path / STATE_PATH).read_text())
    followup = state["step_followups"]["p5r.design"]
    assert followup["boundary"] == replacement_boundary
    assert followup["boundary_refreshes"][0]["replacement_boundary"] == replacement_boundary
    boundary = json.loads(
        (tmp_path / "batch-requalification/phase-boundary.json").read_text()
    )
    assert boundary["boundary_refresh"]["replacement_boundary"] == replacement_boundary
    assert boundary["master_sha256"] == file_hash(tmp_path / PLAN_PATH)


def test_completion_requalification_batch_recovers_after_live_write_interruption(tmp_path, monkeypatch):
    manifest_path = _completion_requalification_batch_fixture(tmp_path)
    original = boundary_module.atomic_write_text

    def interrupt_live_state(target, content):
        if target == tmp_path / STATE_PATH:
            raise OSError("interrupted completion requalification batch")
        return original(target, content)

    monkeypatch.setattr(boundary_module, "atomic_write_text", interrupt_live_state)
    with pytest.raises(OSError, match="interrupted completion requalification batch"):
        requalify_completion_batch(tmp_path, manifest_path)
    monkeypatch.setattr(boundary_module, "atomic_write_text", original)
    journal = manifest_path.parent / "publication/publication.json"
    assert recover_completion_requalification_batch(tmp_path, journal)["phase_boundary_current"]
    assert Program(tmp_path).complete == {"p4.component-audit", "p4.adapter"}


def test_completion_requalification_batch_rejects_unrelated_state_payload_change(tmp_path):
    manifest_path = _completion_requalification_batch_fixture(tmp_path)
    requalify_completion_batch(tmp_path, manifest_path)
    journal_path = manifest_path.parent / "publication/publication.json"
    journal = json.loads(journal_path.read_text())
    state_payload_path = manifest_path.parent / "publication/current-state.json"
    altered_state = json.loads(state_payload_path.read_text())
    altered_state["latest_attempt_id"] = "unrelated-payload-edit"
    write_json(state_payload_path, altered_state)
    write_json(tmp_path / STATE_PATH, altered_state)
    journal["files"][STATE_PATH]["after_sha256"] = file_hash(state_payload_path)
    write_json(journal_path, journal)
    with pytest.raises(ProgramError, match="candidate payload does not match"):
        recover_completion_requalification_batch(tmp_path, journal_path)


def test_completion_requalification_batch_requires_batch_schema(tmp_path):
    manifest_path = _completion_requalification_batch_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "generic_neural_solver.completion_requalification.v1"
    write_json(manifest_path, manifest)
    with pytest.raises(ProgramError, match="unknown completion requalification batch manifest"):
        requalify_completion_batch(tmp_path, manifest_path)


def test_completion_requalification_batch_persists_intent_before_boundary(tmp_path, monkeypatch):
    manifest_path = _completion_requalification_batch_fixture(tmp_path)
    original = boundary_module.atomic_write_json
    boundary_path = manifest_path.parent / "phase-boundary.json"

    def interrupt_boundary(target, content):
        if target == boundary_path:
            raise OSError("interrupted before batch boundary")
        return original(target, content)

    monkeypatch.setattr(boundary_module, "atomic_write_json", interrupt_boundary)
    with pytest.raises(OSError, match="interrupted before batch boundary"):
        requalify_completion_batch(tmp_path, manifest_path)
    assert (manifest_path.parent / "publication/intent.json").is_file()
    monkeypatch.setattr(boundary_module, "atomic_write_json", original)
    assert requalify_completion_batch(tmp_path, manifest_path)["phase_boundary_current"]


def test_changing_progress_without_refresh_refuses_new_execution(tmp_path):
    cycle_program(tmp_path)
    advance(tmp_path, record(tmp_path))
    state = json.loads((tmp_path / STATE_PATH).read_text())
    state["step_followups"] = {"work": {"repair_actions": ["new unreviewed issue"]}}
    write_json(tmp_path / STATE_PATH, state)
    with pytest.raises(ProgramError, match="repair/refresh required"):
        Program(tmp_path).require_step("work")




@pytest.mark.parametrize("interrupt_at", (
    "intent.json", "boundary-receipt.json", "master.md", "decision-ledger.json",
    "current-state.json", "publication.json", "before-master.md",
    "before-decision-ledger.json", "before-current-state.json",
    PLAN_PATH, DECISIONS_PATH, STATE_PATH,
))
def test_every_publication_write_is_restartable(tmp_path, monkeypatch, interrupt_at):
    cycle_program(tmp_path)
    path = record(tmp_path)
    original_text = boundary_module.atomic_write_text
    original_json = boundary_module.atomic_write_json
    target_path = (
        tmp_path / interrupt_at if "/" in interrupt_at
        else path.parent / "publication" / interrupt_at
    )

    def fail_write(target, content):
        if target == target_path:
            raise OSError("interrupted publication write")
        return original_text(target, content)

    def fail_json(target, content):
        if target == target_path:
            raise OSError("interrupted publication write")
        return original_json(target, content)

    monkeypatch.setattr(boundary_module, "atomic_write_text", fail_write)
    monkeypatch.setattr(boundary_module, "atomic_write_json", fail_json)
    with pytest.raises(OSError, match="interrupted publication write"):
        advance(tmp_path, path)
    monkeypatch.setattr(boundary_module, "atomic_write_text", original_text)
    monkeypatch.setattr(boundary_module, "atomic_write_json", original_json)
    assert advance(tmp_path, path)["phase_boundary_current"]
    state_before = (tmp_path / STATE_PATH).read_bytes()
    assert advance(tmp_path, path)["phase_boundary_current"]
    assert (tmp_path / STATE_PATH).read_bytes() == state_before
    assert len(json.loads((tmp_path / DECISIONS_PATH).read_text())["decisions"]) == 2


def test_edit_during_preparation_survives_and_refuses_restart(tmp_path, monkeypatch):
    cycle_program(tmp_path)
    path = record(tmp_path)
    original = boundary_module.atomic_write_text
    state_path = tmp_path / STATE_PATH

    def concurrent_edit(target, content):
        if target == path.parent / "publication/master.md":
            state = json.loads(state_path.read_text())
            state["user_note"] = "Preserve this concurrent change"
            write_json(state_path, state)
        return original(target, content)

    monkeypatch.setattr(boundary_module, "atomic_write_text", concurrent_edit)
    with pytest.raises(ProgramError, match="outside this publication"):
        advance(tmp_path, path)
    before_retry = state_path.read_bytes()
    monkeypatch.setattr(boundary_module, "atomic_write_text", original)
    with pytest.raises(ProgramError, match="outside this publication"):
        advance(tmp_path, path)
    assert state_path.read_bytes() == before_retry


@pytest.mark.parametrize("action", ("execute", "recover", "interrupted-recover"))
def test_diagnostic_payload_is_reverified_before_use(tmp_path, monkeypatch, action):
    cycle_program(tmp_path)
    diagnostic = tmp_path / "diagnostic.json"
    write_json(diagnostic, {"repairs_checked": True})
    path = record(tmp_path, evidence=[{"path": "diagnostic.json", "sha256": file_hash(diagnostic)}])
    if action == "interrupted-recover":
        original = boundary_module.atomic_write_text

        def fail_live(target, content):
            if target == tmp_path / PLAN_PATH:
                raise OSError("before first live write")
            return original(target, content)

        monkeypatch.setattr(boundary_module, "atomic_write_text", fail_live)
        with pytest.raises(OSError):
            advance(tmp_path, path)
        monkeypatch.setattr(boundary_module, "atomic_write_text", original)
    else:
        advance(tmp_path, path)
    write_json(diagnostic, {"repairs_checked": False})
    before = {relative: (tmp_path / relative).read_bytes() for relative in (PLAN_PATH, STATE_PATH, DECISIONS_PATH)}
    with pytest.raises(ProgramError, match="evidence.*changed"):
        if action == "execute":
            Program(tmp_path).require_step("work", execution=True)
        else:
            recover_publication(tmp_path, path.parent / "publication/publication.json")
    assert all((tmp_path / relative).read_bytes() == content for relative, content in before.items())


def test_repair_blocks_unchanged_rerun_until_evidenced_repaired_boundary(tmp_path):
    cycle_program(tmp_path)
    advance(tmp_path, record(tmp_path, "failed", step_id="work", outcome="FAILED",
                             repair_actions=["Fix incorrect normalization"]))
    Program(tmp_path).require_step("work")
    with pytest.raises(ProgramError, match="repair evidence required"):
        Program(tmp_path).require_step("work", execution=True)
    advance(tmp_path, record(tmp_path, "repaired", step_id="work", outcome="REPAIRED",
                             repairs_completed=["Normalization fixed and checked"],
                             advisory_followups=["Optional wording improvement"]))
    Program(tmp_path).require_step("work", execution=True)
    assert "work" not in Program(tmp_path).complete




def test_terminal_transition_persists_pending_before_first_live_publication(tmp_path):
    cycle_program(tmp_path)
    advance(tmp_path, record(tmp_path))
    controller = create_attempt(tmp_path, previous_activity_id="refresh")
    controller.preflight_pass()
    controller.start_running()
    controller.block("Terminal packet before first live writer")
    state = Program(tmp_path).state
    assert state["pending_phase_boundary"] == controller.attempt.attempt_id
    with pytest.raises(ProgramError, match="repair/refresh required"):
        create_attempt(tmp_path, previous_activity_id="refresh")
    with pytest.raises(ProgramError, match="own step disposition"):
        advance(tmp_path, record(tmp_path, "terminal-refresh"))
