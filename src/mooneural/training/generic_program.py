"""Read the controlling program and check evidence before starting an attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mooneural.artifacts import atomic_write_json

PROGRAM_ID = "mooneural-training"
PLAN_PATH = "docs/plans/mooneural-training-program.md"
STATE_PATH = "docs/experiments/generic-neural-solver/current-state.json"
DECISIONS_PATH = "docs/experiments/generic-neural-solver/decision-ledger.json"
GRAPH_START = "<!-- BEGIN EXECUTABLE PROGRAM -->\n```json\n"
GRAPH_END = "\n```\n<!-- END EXECUTABLE PROGRAM -->"
LANES = {"Explore": 0, "Validate": 1, "Admit": 2}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress_hash(state: dict) -> str:
    payload = {
        key: state.get(key, {}) for key in (
            "program_steps", "step_followups"
        )
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ProgramError(ValueError):
    """The requested action is inconsistent with the current program evidence."""


class Program:
    def __init__(self, repo_root: Path, *, snapshot=None):
        self.root = repo_root.resolve()
        self.snapshot = snapshot if snapshot is not None else {
            relative: (self.root / relative).read_bytes().decode()
            for relative in (PLAN_PATH, DECISIONS_PATH, STATE_PATH)
        }
        self.state = json.loads(self.snapshot[STATE_PATH])
        plan = self.snapshot[PLAN_PATH]
        self.plan_hash = hashlib.sha256(plan.encode()).hexdigest()
        if (
            self.state.get("program_id") != PROGRAM_ID
            or self.state.get("canonical_plan") != PLAN_PATH
            or self.state.get("canonical_plan_sha256") != self.plan_hash
        ):
            raise ProgramError("master/current-state binding mismatch; repair before execution")
        decisions = json.loads(self.snapshot[DECISIONS_PATH])
        if decisions.get("program_id") != PROGRAM_ID or not any(
            decision.get("decision_id") == self.state.get("program_revision_decision")
            and decision.get("master_sha256") == self.plan_hash
            for decision in decisions.get("decisions", [])
        ):
            raise ProgramError("master revision is not recorded in the decision ledger")
        if plan.count(GRAPH_START) != 1 or plan.count(GRAPH_END) != 1:
            raise ProgramError("master must contain exactly one executable program")
        graph = json.loads(plan.split(GRAPH_START)[1].split(GRAPH_END)[0])
        if graph.get("schema") != "generic_neural_solver.program.v1":
            raise ProgramError("unknown program schema")
        self.steps = {step["id"]: step for step in graph["steps"]}
        self.boundary_required = graph.get("between_phase_cycle") == "repair_refresh_continue.v1"
        if len(self.steps) != len(graph["steps"]):
            raise ProgramError("duplicate program step")
        for step in self.steps.values():
            if step["lane"] not in LANES or step["mode"] not in {
                "host", "implementation", "protocol"
            }:
                raise ProgramError("unknown lane or execution mode")
            if step["evidence_kind"] not in {"inspection", "audited_report", "protocol"}:
                raise ProgramError("unknown closure evidence kind")
            if step["evidence_kind"] == "inspection" and step["lane"] != "Explore":
                raise ProgramError("inspection cannot close a Validate or Admit step")
            if set(step["requires"]) - self.steps.keys():
                raise ProgramError("unknown prerequisite")
        self._validate_graph()
        self.records = self.state.get("program_steps", {})
        if set(self.records) - self.steps.keys():
            raise ProgramError("ledger contains unknown program step")
        self.complete = set()
        for step_id, record in self.records.items():
            self._verify_completion(step_id, record)
            self.complete.add(step_id)
        for step_id in self.complete:
            if set(self.steps[step_id]["requires"]) - self.complete:
                raise ProgramError(f"{step_id}: completion precedes prerequisite closure")
        self.followups = self.state.get("step_followups", {})
        if set(self.followups) - self.steps.keys():
            raise ProgramError("follow-up record names an unknown step")

    def boundary_current(self, *, continuing_attempt_id=None):
        if not self.boundary_required:
            return True
        reference = self.state.get("last_phase_boundary")
        if not reference:
            return False
        receipt = self._evidence(reference)
        boundary = self._evidence(receipt["boundary"])
        self.verify_references(boundary["evidence"])
        for followup in self.followups.values():
            if followup.get("boundary"):
                previous_boundary = self._evidence(followup["boundary"])
                self.verify_references(previous_boundary["evidence"])
        pending, _step = self.pending_result()
        if pending and pending != continuing_attempt_id:
            return False
        return (
            receipt.get("schema") == "generic_neural_solver.boundary_receipt.v1"
            and receipt.get("master_sha256") == self.plan_hash
            and receipt.get("progress_sha256") == progress_hash(self.state)
        )

    def pending_result(self):
        pending = self.state.get("pending_phase_boundary")
        step = self.state.get("active_program_step")
        if pending:
            return pending, step
        relative = self.state.get("artifact_root")
        if relative and (self._path(relative) / "protocol/phase-state.json").is_file():
            from .generic_protocol import ProtocolController

            controller = ProtocolController.load(self._path(relative))
            attempt = controller.attempt
            if (
                attempt.program_id == PROGRAM_ID
                and attempt.attempt_id == self.state.get("latest_attempt_id")
                and attempt.status in {"AUDIT_FAILED", "BLOCKED", "CLOSED"}
            ):
                return attempt.attempt_id, attempt.evidence_contract.get("program", {}).get("step")
        return None, None

    def _validate_graph(self):
        visited = set()
        active = set()

        def visit(step_id):
            if step_id in active:
                raise ProgramError("cyclic program dependencies")
            if step_id in visited:
                return
            active.add(step_id)
            for prerequisite in self.steps[step_id]["requires"]:
                visit(prerequisite)
            active.remove(step_id)
            visited.add(step_id)

        for step_id in self.steps:
            visit(step_id)

    def _path(self, relative):
        path = self.root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ProgramError("program evidence paths must be repository relative")
        return path

    def _evidence(self, reference):
        path = self._path(reference["path"])
        if file_hash(path) != reference["sha256"]:
            raise ProgramError(f"evidence hash changed: {reference['path']}")
        return json.loads(path.read_text())

    def verify_references(self, references):
        for reference in references:
            if file_hash(self._path(reference["path"])) != reference["sha256"]:
                raise ProgramError(f"evidence hash changed: {reference['path']}")

    def _verify_completion(self, step_id, record):
        step = self.steps[step_id]
        if record.get("kind") != step["evidence_kind"]:
            raise ProgramError(f"{step_id}: wrong closure evidence kind")
        if record["kind"] == "protocol":
            from .generic_protocol import AuditCoordinator, ProtocolController

            controller = ProtocolController.load(self._path(record["artifact_root"]))
            attempt = controller.attempt
            if (
                attempt.program_id != PROGRAM_ID
                or attempt.phase_id != step["phase_id"]
                or attempt.status != "CLOSED"
                or attempt.decision not in ("PASS", "PROMOTE")
                or LANES[attempt.lane] < LANES[step["lane"]]
            ):
                raise ProgramError(f"{step_id}: prerequisite protocol is not closed/passed")
            if file_hash(controller.artifact_root / "protocol/phase-state.json") != record["state_sha256"]:
                raise ProgramError(f"{step_id}: protocol closure binding changed")
            AuditCoordinator.verify_packet(controller)
            if controller.validate_sealed_result()["result"]["status"] != step["pass_decision"]:
                raise ProgramError(f"{step_id}: packet result does not close this claim")
            return
        report = self._evidence(record["report"])
        if record["kind"] == "inspection":
            for field, expected in step["report_fields"].items():
                if report.get(field) != expected:
                    raise ProgramError(f"{step_id}: inspection result mismatch: {field}")
            return
        review = self._evidence(record["review"])
        if (
            report.get("program_id") != PROGRAM_ID
            or report.get("step_id") != step_id
            or report.get("decision") != step["pass_decision"]
            or not report.get("executor_id")
            or review.get("decision") != "PASS"
            or not review.get("auditor_id")
            or review["auditor_id"] == report["executor_id"]
            or review.get("report_sha256") != record["report"]["sha256"]
        ):
            raise ProgramError(f"{step_id}: independent closure review missing or mismatched")
        if not report.get("inputs"):
            raise ProgramError(f"{step_id}: closure report must bind its input files")
        for reference in report["inputs"]:
            if reference.get("path") in {PLAN_PATH, DECISIONS_PATH, STATE_PATH}:
                raise ProgramError(f"{step_id}: closure inputs cannot bind mutable live ledgers")
            if file_hash(self._path(reference["path"])) != reference["sha256"]:
                raise ProgramError(f"{step_id}: closure input changed")

    def status(self, *, check_boundary=True):
        rows = []
        for step_id, step in self.steps.items():
            missing = sorted(set(step["requires"]) - self.complete)
            followup = self.followups.get(step_id, {})
            blockers = followup.get("external_blockers", [])
            repairs = followup.get("repair_actions", [])
            state = (
                "COMPLETE" if step_id in self.complete else
                "BLOCKED" if missing else
                "REPAIR_REQUIRED" if repairs else
                "BLOCKED_EXTERNAL" if blockers else "READY"
            )
            rows.append({
                "step": step_id, "state": state, "requires": missing,
                "mode": step["mode"], "entrypoint": step["entrypoint"],
                "closure": step["closure"],
                "repair_actions": repairs, "external_blockers": blockers,
                "optional": step.get("optional", False),
            })
        eligible = [row for row in rows if row["state"] in ("READY", "REPAIR_REQUIRED")]
        next_rows = sorted(
            (row for row in eligible if not row["optional"]),
            key=lambda row: row["state"] != "REPAIR_REQUIRED",
        )
        return {
            "program_id": PROGRAM_ID, "master_sha256": self.plan_hash,
            "next_step": next_rows[0]["step"] if next_rows else None,
            "ready_steps": [row["step"] for row in eligible],
            "phase_boundary_current": self.boundary_current() if check_boundary else False,
            "steps": rows,
            "scope": "Dependency readiness; no numerical execution or scientific promotion.",
        }

    def require_step(self, step_id, *, plan_hash=None, phase_id=None, lane=None, execution=False,
                     continuing_attempt_id=None):
        if not self.boundary_current(continuing_attempt_id=continuing_attempt_id):
            raise ProgramError("between-phase repair/refresh required before further execution")
        if plan_hash is not None and plan_hash != self.plan_hash:
            raise ProgramError("attempt uses a superseded master revision")
        if step_id not in self.steps:
            raise ProgramError(f"unknown program step: {step_id}")
        step = self.steps[step_id]
        followup = self.followups.get(step_id, {})
        if followup.get("external_blockers") and (execution or not followup.get("repair_actions")):
            raise ProgramError(f"{step_id}: unresolved external evidence blocker; repair or inspect new evidence")
        if execution and followup.get("repair_actions"):
            raise ProgramError(f"{step_id}: repair evidence required before numerical execution")
        if step_id in self.complete:
            raise ProgramError("completed steps are inspected; declare a separate regression attempt")
        missing = sorted(set(step["requires"]) - self.complete)
        if missing:
            raise ProgramError(f"{step_id}: missing prerequisites: {', '.join(missing)}")
        if phase_id is not None and phase_id != step["phase_id"]:
            raise ProgramError("attempt phase does not match declared program step")
        if lane is not None and LANES[lane] < LANES[step["lane"]]:
            raise ProgramError("attempt lane is below the declared program lane")
        if execution and (
            step["mode"] != "protocol" or not step["entrypoint"]
            or not self._path(step["entrypoint"]).is_file()
        ):
            raise ProgramError("numerical entrypoint is not implemented/registered for this step")
        return step


def require_program_attempt(
    program_id, phase_id, lane, plan_hash, evidence_contract, attempt_id=None, attempt_status=None
):
    if program_id != PROGRAM_ID:
        return
    binding = evidence_contract.get("program")
    if (
        not isinstance(binding, dict) or not binding.get("repo_root")
        or not binding.get("step") or not binding.get("previous_activity_id")
    ):
        raise ProgramError("new program attempts require an explicit master step binding")
    program = Program(Path(binding["repo_root"]))
    program.require_step(
        binding["step"], plan_hash=plan_hash, phase_id=phase_id, lane=lane,
        execution=attempt_status not in {"AUDIT_FAILED", "BLOCKED", "CLOSED"},
        continuing_attempt_id=attempt_id,
    )
    if program.state.get("latest_attempt_id") not in {
        binding["previous_activity_id"], attempt_id
    }:
        raise ProgramError("live program activity advanced; stale attempt cannot run or publish")


def mark_pending_boundary(state, controller):
    binding = controller.attempt.evidence_contract.get("program", {})
    if binding:
        state["active_program_step"] = binding["step"]
    if controller.attempt.program_id == PROGRAM_ID and controller.attempt.status in {
        "AUDIT_FAILED", "BLOCKED", "CLOSED"
    }:
        state["pending_phase_boundary"] = controller.attempt.attempt_id


def persist_pending_boundary(controller):
    if controller.attempt.program_id != PROGRAM_ID:
        return
    binding = controller.attempt.evidence_contract.get("program", {})
    if not binding:
        return
    root = Path(binding["repo_root"]).resolve()
    state_path = root / STATE_PATH
    state = json.loads(state_path.read_text())
    allowed_owners = {binding["previous_activity_id"], controller.attempt.attempt_id}
    if state.get("latest_attempt_id") not in allowed_owners:
        raise ProgramError("live program activity advanced; terminal result is stale")
    existing = state.get("pending_phase_boundary")
    if existing not in (None, controller.attempt.attempt_id):
        raise ProgramError("another terminal result already requires disposition")
    binding = controller.attempt.evidence_contract.get("program", {})
    if binding:
        state["active_program_step"] = binding["step"]
    state["pending_phase_boundary"] = controller.attempt.attempt_id
    atomic_write_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "check"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--step")
    args = parser.parse_args()
    try:
        program = Program(args.repo_root)
        if args.action == "check":
            if not args.step:
                parser.error("check requires --step")
            program.require_step(args.step)
        print(json.dumps(program.status(), indent=2))
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps({"status": "REFUSED", "reason": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
