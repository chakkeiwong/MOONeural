"""Program readiness, claim binding, stale-attempt and recovery regressions."""

import json
from pathlib import Path

import pytest

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
from mooneural.training.generic_protocol import ProtocolController

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_step(step_id, requires=(), **changes):
    step = {
        "id": step_id, "phase_id": step_id, "requires": list(requires),
        "lane": "Validate", "mode": "protocol", "evidence_kind": "protocol",
        "pass_decision": "CONFORMANCE_PASSED", "entrypoint": "runner.py",
        "closure": "Declared test claim",
    }
    step.update(changes)
    return step


def fixture_program(tmp_path, *, steps=None, records=None):
    report_path = tmp_path / "intake.json"
    write_json(report_path, {"result": "INTAKE_ONLY"})
    if steps is None:
        steps = [
            make_step("intake", lane="Explore", mode="host", evidence_kind="inspection",
                      report_fields={"result": "INTAKE_ONLY"}),
            make_step("work", ["intake"]),
            make_step("training", ["work"]),
            make_step("unimplemented", ["intake"], entrypoint=None),
        ]
    if records is None:
        records = {"intake": {
            "kind": "inspection", "report": {"path": "intake.json", "sha256": file_hash(report_path)}
        }}
    plan = tmp_path / PLAN_PATH
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(GRAPH_START + json.dumps({"schema": "generic_neural_solver.program.v1", "steps": steps}) + GRAPH_END)
    write_json(tmp_path / STATE_PATH, {
        "program_id": PROGRAM_ID, "canonical_plan": PLAN_PATH,
        "canonical_plan_sha256": file_hash(plan), "program_revision_decision": "revision",
        "program_steps": records, "latest_attempt_id": "previous",
    })
    write_json(tmp_path / DECISIONS_PATH, {
        "program_id": PROGRAM_ID,
        "decisions": [{"decision_id": "revision", "master_sha256": file_hash(plan)}],
    })
    (tmp_path / "runner.py").write_text('"""Unused test entrypoint."""\n')
    return Program(tmp_path)


def create_attempt(root, step="work", **binding_changes):
    binding = {"repo_root": str(root), "step": step, "previous_activity_id": "previous"}
    binding.update(binding_changes)
    controller = ProtocolController(
        program_id=PROGRAM_ID, phase_id=step, lane="Validate",
        plan_hash=file_hash(root / PLAN_PATH), executor_id="executor", auditor_id="auditor",
        artifact_base=root / "attempts", experiment_base=root / "experiments",
    )
    controller.create_attempt(
        question="Does program routing refuse drift?", protected_claim="Fixture only",
        comparator="Declared graph", primary_criterion="No execution after drift",
        vetoes=("binding drift",), nonclaims=("No numerical work",),
        stop_conditions=("Before numerical work",), source_hashes={"source": "a" * 64},
        input_hashes={"input": "b" * 64}, role_manifest_hash="c" * 64,
        evidence_contract={"program": binding}, baseline={"fixture": True},
        resources={"host_only": True},
    )
    return controller


def test_missing_historical_evidence_allows_only_independent_work(tmp_path):
    program = fixture_program(tmp_path)
    assert program.status()["next_step"] == "work"
    assert "training" not in program.status()["ready_steps"]
    program.require_step("work")
    with pytest.raises(ProgramError, match="missing prerequisites"):
        program.require_step("training")


@pytest.mark.parametrize("step", ("unknown", "training", "intake", "unimplemented"))
def test_blocked_attempt_refuses_before_allocating_artifacts(tmp_path, step):
    fixture_program(tmp_path)
    with pytest.raises(ProgramError):
        create_attempt(tmp_path, step)
    assert not (tmp_path / "attempts").exists()
    assert not (tmp_path / "experiments").exists()


def test_stale_master_refuses_start_but_packet_remains_loadable(tmp_path):
    fixture_program(tmp_path)
    controller = create_attempt(tmp_path)
    controller.preflight_pass()
    plan = tmp_path / PLAN_PATH
    plan.write_text(plan.read_text() + "\nchanged\n")
    with pytest.raises(ProgramError, match="binding mismatch"):
        controller.start_running()
    recovered = ProtocolController.load(controller.artifact_root)
    assert recovered.attempt.status == "PREFLIGHT_PASSED"




@pytest.mark.parametrize("target", ("master", "decision", "evidence", "false-pass"))
def test_drift_and_false_inspection_closure_refuse(tmp_path, target):
    fixture_program(tmp_path)
    if target == "master":
        (tmp_path / PLAN_PATH).write_text("different master")
    elif target == "decision":
        write_json(tmp_path / DECISIONS_PATH, {"program_id": PROGRAM_ID, "decisions": []})
    else:
        write_json(tmp_path / "intake.json", {"result": "WRONG_CLAIM"})
        if target == "false-pass":
            state = json.loads((tmp_path / STATE_PATH).read_text())
            state["program_steps"]["intake"]["report"]["sha256"] = file_hash(tmp_path / "intake.json")
            write_json(tmp_path / STATE_PATH, state)
    with pytest.raises(ProgramError):
        Program(tmp_path)


@pytest.mark.parametrize("problem", ("cycle", "duplicate", "unknown-prerequisite", "unaudited-claim"))
def test_malformed_graph_is_not_a_new_execution_path(tmp_path, problem):
    steps = [make_step("work", ["work"] if problem == "cycle" else [])]
    if problem == "duplicate":
        steps.append(dict(steps[0]))
    elif problem == "unknown-prerequisite":
        steps[0]["requires"] = ["missing"]
    elif problem == "unaudited-claim":
        steps[0]["evidence_kind"] = "inspection"
    with pytest.raises(ProgramError):
        fixture_program(tmp_path, steps=steps, records={})


@pytest.mark.parametrize("problem", ("self-audit", "component-pass", "wrong-step", "wrong-report", "input-drift", None))
def test_reviewed_report_binds_claim_auditor_and_actual_inputs(tmp_path, problem):
    payload = tmp_path / "inputs.txt"
    payload.write_text("frozen inputs")
    report = {
        "program_id": PROGRAM_ID, "step_id": "work", "decision": "READY",
        "executor_id": "executor", "inputs": [{"path": "inputs.txt", "sha256": file_hash(payload)}],
    }
    if problem == "component-pass":
        report["decision"] = "COMPONENT_TESTS_PASSED"
    elif problem == "wrong-step":
        report["step_id"] = "other-step"
    write_json(tmp_path / "report.json", report)
    review = {
        "auditor_id": "executor" if problem == "self-audit" else "independent",
        "decision": "PASS", "report_sha256": file_hash(tmp_path / "report.json"),
    }
    if problem == "wrong-report":
        review["report_sha256"] = "a" * 64
    if problem == "input-drift":
        payload.write_text("changed")
    write_json(tmp_path / "review.json", review)
    record = {"kind": "audited_report", **{
        name: {"path": name + ".json", "sha256": file_hash(tmp_path / (name + ".json"))}
        for name in ("report", "review")
    }}
    steps = [make_step("work", mode="host", evidence_kind="audited_report", pass_decision="READY")]
    if problem:
        with pytest.raises(ProgramError):
            fixture_program(tmp_path, steps=steps, records={"work": record})
    else:
        assert fixture_program(tmp_path, steps=steps, records={"work": record}).complete == {"work"}
