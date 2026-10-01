"""Publish a phase repair/refresh decision and recover interrupted publication."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mooneural.artifacts import atomic_write_json, atomic_write_text

from .generic_program import (
    DECISIONS_PATH,
    GRAPH_END,
    GRAPH_START,
    PLAN_PATH,
    PROGRAM_ID,
    STATE_PATH,
    Program,
    ProgramError,
    file_hash,
    progress_hash,
)

POSITION_START = "<!-- BEGIN CURRENT PROGRAM POSITION -->"
POSITION_END = "<!-- END CURRENT PROGRAM POSITION -->"
MASTER_REVISION_SCHEMA = "generic_neural_solver.master_revision.v1"
COMPLETION_REPAIR_SCHEMA = "generic_neural_solver.completion_metadata_repair.v1"
COMPLETION_REPAIR_PUBLICATION_SCHEMA = "generic_neural_solver.completion_metadata_publication.v1"
COMPLETION_REQUALIFICATION_SCHEMA = "generic_neural_solver.completion_requalification.v1"
COMPLETION_REQUALIFICATION_PUBLICATION_SCHEMA = (
    "generic_neural_solver.completion_requalification_publication.v1"
)
COMPLETION_REQUALIFICATION_BATCH_SCHEMA = (
    "generic_neural_solver.completion_requalification_batch.v1"
)
COMPLETION_REQUALIFICATION_BATCH_PUBLICATION_SCHEMA = (
    "generic_neural_solver.completion_requalification_batch_publication.v1"
)
BOUNDARY_REFRESH_VALIDATION_SCHEMA = (
    "generic_neural_solver.boundary_evidence_refresh_validation.v1"
)
MUTABLE_LIVE_PATHS = frozenset({PLAN_PATH, DECISIONS_PATH, STATE_PATH})
REPAIRABLE_METADATA_INPUTS = frozenset({
    "src/mooneural/training/generic_program.py",
    "src/mooneural/training/generic_program_boundary.py",
})
REQUALIFICATION_ALLOWED_INPUTS = {
    "repair.sgu178-candidate-lifecycle": frozenset({
        "src/mooneural/training/generic_finite_objective_runner.py",
        "src/mooneural/training/sgu_equilibrium_objective.py",
    }),
    "p4.component-audit": frozenset({
        "src/mooneural/training/generic_objective_normalization.py",
        "src/mooneural/training/rotemberg_input_contracts.py",
    }),
    "p4.adapter": frozenset({
        "src/mooneural/training/generic_training_contracts.py",
        "src/mooneural/training/rotemberg_generic_adapter.py",
        "src/mooneural/training/rotemberg_input_contracts.py",
        "tests/contracts/test_rotemberg_generic_adapter.py",
    }),
    "p5r.source": frozenset({
        "src/mooneural/training/generic_objective_normalization.py",
        "src/mooneural/training/generic_permanent_pass.py",
        "src/mooneural/training/generic_training_contracts.py",
        "src/mooneural/training/rotemberg_generic_adapter.py",
        "src/mooneural/training/rotemberg_input_contracts.py",
    }),
}


def _require_unsealed_publication_path(root, path):
    for directory in (path.parent, *path.parent.parents):
        if directory == root:
            break
        if (directory / "sealed-result.json").is_file() or (
            (directory / "hashes.txt").is_file()
            and (directory / "attempt-binding.json").is_file()
        ):
            raise ProgramError("governance publication must be outside sealed protocol artifact and experiment roots")


def _text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _reference(root, path):
    return {"path": str(path.relative_to(root)), "sha256": file_hash(path)}


def _verify_references(root, references):
    for reference in references:
        relative = Path(reference["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ProgramError("publication evidence paths must be repository relative")
        if file_hash(root / relative) != reference["sha256"]:
            raise ProgramError(f"publication evidence changed: {relative}")


def _verify_immutable_references(root, references):
    for reference in references:
        if reference.get("path") in MUTABLE_LIVE_PATHS:
            raise ProgramError("phase evidence cannot bind mutable live ledgers")
    _verify_references(root, references)


def _verify_live(root, files):
    for relative, entry in files.items():
        if file_hash(root / relative) not in {entry["before_sha256"], entry["after_sha256"]}:
            raise ProgramError("live files changed outside this publication; no overwrite permitted")


def _read_live_snapshot(root):
    return {
        relative: (root / relative).read_bytes().decode()
        for relative in (PLAN_PATH, DECISIONS_PATH, STATE_PATH)
    }


def _snapshot_files(snapshot):
    return {
        relative: {
            "before_sha256": _text_hash(content),
            "after_sha256": _text_hash(content),
        }
        for relative, content in snapshot.items()
    }


def _raw_program_binding(snapshot):
    state = json.loads(snapshot[STATE_PATH])
    plan = snapshot[PLAN_PATH]
    plan_hash = _text_hash(plan)
    if (
        state.get("program_id") != PROGRAM_ID
        or state.get("canonical_plan") != PLAN_PATH
        or state.get("canonical_plan_sha256") != plan_hash
    ):
        raise ProgramError("master/current-state binding mismatch; repair before execution")
    ledger = json.loads(snapshot[DECISIONS_PATH])
    if ledger.get("program_id") != state.get("program_id") or not any(
        decision.get("decision_id") == state.get("program_revision_decision")
        and decision.get("master_sha256") == plan_hash
        for decision in ledger.get("decisions", [])
    ):
        raise ProgramError("master revision is not recorded in the decision ledger")
    if plan.count(GRAPH_START) != 1 or plan.count(GRAPH_END) != 1:
        raise ProgramError("master must contain exactly one executable program")
    graph = json.loads(plan.split(GRAPH_START)[1].split(GRAPH_END)[0])
    if graph.get("schema") != "generic_neural_solver.program.v1":
        raise ProgramError("unknown program schema")
    steps = {step["id"]: step for step in graph["steps"]}
    if len(steps) != len(graph["steps"]):
        raise ProgramError("duplicate program step")
    return state, ledger, steps, plan_hash


def _completion_repair_inputs(root, old_report, new_report, removed_inputs):
    old_inputs = old_report.get("inputs")
    if not isinstance(old_inputs, list) or not old_inputs:
        raise ProgramError("old completion report must bind inputs")
    if not isinstance(removed_inputs, list) or not removed_inputs:
        raise ProgramError("completion metadata repair must remove a declared input")
    if any(not isinstance(reference, dict) for reference in removed_inputs):
        raise ProgramError("removed completion inputs must be references")
    if any(reference not in old_inputs for reference in removed_inputs):
        raise ProgramError("completion repair removes an input absent from the old report")
    if any(reference["path"] not in REPAIRABLE_METADATA_INPUTS for reference in removed_inputs):
        raise ProgramError("completion repair may remove only program-governance inputs")
    if len({reference["path"] for reference in removed_inputs}) != len(removed_inputs):
        raise ProgramError("completion repair cannot remove an input twice")
    expected_inputs = [reference for reference in old_inputs if reference not in removed_inputs]
    if new_report.get("inputs") != expected_inputs:
        raise ProgramError("replacement report must preserve all non-repaired inputs")
    _verify_immutable_references(root, new_report["inputs"])
    return expected_inputs


def _validate_completion_repair(root, manifest, snapshot):
    state, ledger, steps, plan_hash = _raw_program_binding(snapshot)
    required = (
        "activity_id", "previous_activity_id", "executor_id", "step_id",
        "previous_master_sha256", "old_completion", "replacement_report",
        "replacement_review", "removed_inputs", "evidence",
    )
    if any(not manifest.get(field) for field in required):
        raise ProgramError("completion metadata repair manifest is incomplete")
    step_id = manifest["step_id"]
    if step_id not in steps:
        raise ProgramError("completion metadata repair names an unknown step")
    if manifest["previous_activity_id"] != state.get("latest_attempt_id"):
        raise ProgramError("completion repair predecessor does not match live activity")
    if manifest["previous_master_sha256"] != plan_hash:
        raise ProgramError("completion repair is based on a superseded master")
    if state.get("pending_phase_boundary"):
        raise ProgramError("completion repair cannot discard a pending terminal result")
    old_record = state.get("program_steps", {}).get(step_id)
    if old_record != manifest["old_completion"]:
        raise ProgramError("completion repair old binding does not match live state")
    if old_record.get("kind") != "audited_report":
        raise ProgramError("completion metadata repair requires an audited report")
    old_report_reference = old_record.get("report")
    old_review_reference = old_record.get("review")
    replacement_report_reference = manifest["replacement_report"]
    replacement_review_reference = manifest["replacement_review"]
    _verify_references(root, [
        old_report_reference, old_review_reference,
        replacement_report_reference, replacement_review_reference,
    ])
    _verify_immutable_references(root, manifest["evidence"])
    old_report = json.loads((root / old_report_reference["path"]).read_text())
    old_review = json.loads((root / old_review_reference["path"]).read_text())
    new_report = json.loads((root / replacement_report_reference["path"]).read_text())
    new_review = json.loads((root / replacement_review_reference["path"]).read_text())
    if file_hash(root / old_report_reference["path"]) != old_record["report"]["sha256"]:
        raise ProgramError("old completion report changed")
    if file_hash(root / old_review_reference["path"]) != old_record["review"]["sha256"]:
        raise ProgramError("old completion review changed")
    metadata = new_report.get("metadata_repair")
    if (
        not isinstance(metadata, dict)
        or metadata.get("kind") != "completion_metadata_only"
        or metadata.get("old_report") != old_report_reference
        or metadata.get("removed_inputs") != manifest["removed_inputs"]
        or metadata.get("numerical_rerun") is not False
        or metadata.get("claim_unchanged") is not True
    ):
        raise ProgramError("replacement report lacks a bounded metadata-repair declaration")
    comparable_old = copy.deepcopy(old_report)
    comparable_new = copy.deepcopy(new_report)
    comparable_old.pop("inputs", None)
    comparable_new.pop("inputs", None)
    comparable_new.pop("metadata_repair", None)
    if comparable_old != comparable_new:
        raise ProgramError("completion metadata repair changed the scientific report")
    _completion_repair_inputs(root, old_report, new_report, manifest["removed_inputs"])
    new_record = {
        "kind": "audited_report",
        "report": replacement_report_reference,
        "review": replacement_review_reference,
    }
    if (
        new_report.get("program_id") != state.get("program_id")
        or new_report.get("step_id") != step_id
        or new_report.get("decision") != steps[step_id]["pass_decision"]
        or not new_report.get("executor_id")
        or new_review.get("decision") != "PASS"
        or not new_review.get("auditor_id")
        or new_review["auditor_id"] == new_report["executor_id"]
        or new_review["auditor_id"] == manifest["executor_id"]
        or new_review.get("report_sha256") != replacement_report_reference["sha256"]
    ):
        raise ProgramError("replacement completion review is missing or self-audited")
    return {
        "state": state,
        "ledger": ledger,
        "steps": steps,
        "plan_hash": plan_hash,
        "old_record": old_record,
        "new_record": new_record,
        "old_report": old_report,
        "old_review": old_review,
        "new_report": new_report,
        "new_review": new_review,
    }


def _completion_requalification_inputs(
    root, old_report, new_report, changed_inputs, allowed_inputs
):
    old_inputs = old_report.get("inputs")
    new_inputs = new_report.get("inputs")
    if not isinstance(old_inputs, list) or not old_inputs:
        raise ProgramError("old completion report must bind inputs")
    if not isinstance(new_inputs, list) or len(new_inputs) != len(old_inputs):
        raise ProgramError("requalification must preserve the complete input list")
    if not isinstance(changed_inputs, list) or not changed_inputs:
        raise ProgramError("completion requalification must identify changed inputs")
    changed_by_path = {}
    for change in changed_inputs:
        if not isinstance(change, dict):
            raise ProgramError("changed completion inputs must be objects")
        if set(change) != {"path", "old_sha256", "new_sha256"}:
            raise ProgramError("changed completion input fields are invalid")
        path = change["path"]
        if (
            not isinstance(path, str)
            or not path
            or path in MUTABLE_LIVE_PATHS
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ProgramError("changed completion input path is invalid")
        if path not in allowed_inputs:
            raise ProgramError(f"completion requalification input is not allowed for this step: {path}")
        if path in changed_by_path:
            raise ProgramError("completion requalification changes an input twice")
        changed_by_path[path] = change
    if len(changed_by_path) != len(changed_inputs):
        raise ProgramError("completion requalification changes an input twice")
    for old_reference, new_reference in zip(old_inputs, new_inputs):
        if not isinstance(old_reference, dict) or not isinstance(new_reference, dict):
            raise ProgramError("completion input references must be objects")
        if old_reference.get("path") != new_reference.get("path"):
            raise ProgramError("requalification cannot add, remove or reorder inputs")
        path = old_reference.get("path")
        change = changed_by_path.get(path)
        if change is None:
            _verify_immutable_references(root, [old_reference])
            if new_reference != old_reference:
                raise ProgramError("requalification changed an undeclared input")
            continue
        if (
            change["old_sha256"] != old_reference.get("sha256")
            or change["new_sha256"] != new_reference.get("sha256")
            or change["old_sha256"] == change["new_sha256"]
        ):
            raise ProgramError("changed completion input hashes do not match the reports")
        if file_hash(root / path) != change["new_sha256"]:
            raise ProgramError(f"requalified input is not current: {path}")
        if change["old_sha256"] == file_hash(root / path):
            raise ProgramError(f"requalified input did not change: {path}")
        reference_keys = set(old_reference) | set(new_reference)
        for key in reference_keys - {"path", "sha256", "bytes"}:
            if old_reference.get(key) != new_reference.get(key):
                raise ProgramError("requalification changed input reference metadata")
        if "bytes" in reference_keys and (
            new_reference.get("bytes") != (root / path).stat().st_size
        ):
            raise ProgramError("requalification changed input byte metadata incorrectly")
    changed_paths_from_reports = {
        old_reference.get("path")
        for old_reference, new_reference in zip(old_inputs, new_inputs)
        if old_reference != new_reference
    }
    if set(changed_by_path) != changed_paths_from_reports:
        raise ProgramError("changed completion input declaration is incomplete")
    _verify_immutable_references(root, new_inputs)
    return new_inputs


def _validate_requalification_evidence(root, references, step_id, changed_inputs):
    if not isinstance(references, list) or not references:
        raise ProgramError("completion requalification requires validation evidence")
    for reference in references:
        _verify_immutable_references(root, [reference])
        document = json.loads((root / reference["path"]).read_text())
        if (
            document.get("schema") != "generic_neural_solver.completion_requalification_validation.v1"
            or document.get("decision") != "CURRENT_IMPLEMENTATION_INPUTS_REQUALIFICATION_PASSED"
            or document.get("step_id") != step_id
            or document.get("changed_inputs") != changed_inputs
            or document.get("claim_unchanged") is not True
            or document.get("numerical_rerun") is not False
            or document.get("model_execution") is not False
            or document.get("training_updates") != 0
        ):
            raise ProgramError("validation evidence does not bind the requalification claim")
        _verify_document_references(root, document)


def _verify_document_references(root, value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            _verify_immutable_references(root, [value])
        for child in value.values():
            _verify_document_references(root, child)
    elif isinstance(value, list):
        for child in value:
            _verify_document_references(root, child)


def _validate_completion_requalification(root, manifest, snapshot):
    state, ledger, steps, plan_hash = _raw_program_binding(snapshot)
    required = (
        "activity_id", "previous_activity_id", "executor_id", "step_id",
        "previous_master_sha256", "old_completion", "replacement_report",
        "replacement_review", "changed_inputs", "validation_evidence", "reason",
        "evidence",
    )
    if any(field not in manifest for field in required):
        raise ProgramError("completion requalification manifest is incomplete")
    if (
        not all(isinstance(manifest[field], str) and manifest[field].strip() for field in (
            "activity_id", "previous_activity_id", "executor_id", "step_id",
            "previous_master_sha256", "reason",
        ))
        or not isinstance(manifest["evidence"], list)
        or not manifest["evidence"]
        or any(not isinstance(reference, dict) for reference in manifest["evidence"])
    ):
        raise ProgramError("completion requalification manifest fields are invalid")
    step_id = manifest["step_id"]
    if step_id not in steps:
        raise ProgramError("completion requalification names an unknown step")
    if manifest["previous_activity_id"] != state.get("latest_attempt_id"):
        raise ProgramError("completion requalification predecessor does not match live activity")
    if manifest["previous_master_sha256"] != plan_hash:
        raise ProgramError("completion requalification is based on a superseded master")
    if state.get("pending_phase_boundary"):
        raise ProgramError("completion requalification cannot discard a pending terminal result")
    old_record = state.get("program_steps", {}).get(step_id)
    if old_record != manifest["old_completion"]:
        raise ProgramError("completion requalification old binding does not match live state")
    if old_record.get("kind") != "audited_report":
        raise ProgramError("completion requalification requires an audited report")
    old_report_reference = old_record.get("report")
    old_review_reference = old_record.get("review")
    replacement_report_reference = manifest["replacement_report"]
    replacement_review_reference = manifest["replacement_review"]
    _verify_references(root, [
        old_report_reference, old_review_reference,
        replacement_report_reference, replacement_review_reference,
    ])
    validation_evidence = manifest["validation_evidence"]
    if (
        not isinstance(validation_evidence, list)
        or not validation_evidence
        or any(not isinstance(reference, dict) for reference in validation_evidence)
    ):
        raise ProgramError("completion requalification requires validation evidence")
    _verify_immutable_references(root, manifest["evidence"])
    old_report = json.loads((root / old_report_reference["path"]).read_text())
    old_review = json.loads((root / old_review_reference["path"]).read_text())
    new_report = json.loads((root / replacement_report_reference["path"]).read_text())
    new_review = json.loads((root / replacement_review_reference["path"]).read_text())
    if file_hash(root / old_report_reference["path"]) != old_record["report"]["sha256"]:
        raise ProgramError("old completion report changed")
    if file_hash(root / old_review_reference["path"]) != old_record["review"]["sha256"]:
        raise ProgramError("old completion review changed")
    if replacement_report_reference == old_report_reference:
        raise ProgramError("replacement report must be a new artifact")
    if replacement_review_reference == old_review_reference:
        raise ProgramError("replacement review must be a new artifact")
    metadata = new_report.get("requalification")
    expected_metadata = {
        "kind": "implementation_requalification",
        "old_completion": old_record,
        "changed_inputs": manifest["changed_inputs"],
        "validation_evidence": validation_evidence,
        "claim_unchanged": True,
        "numerical_rerun": False,
    }
    if metadata != expected_metadata:
        raise ProgramError("replacement report lacks a bounded requalification declaration")
    comparable_old = copy.deepcopy(old_report)
    comparable_new = copy.deepcopy(new_report)
    comparable_old.pop("inputs", None)
    comparable_new.pop("inputs", None)
    comparable_old.pop("requalification", None)
    comparable_new.pop("requalification", None)
    if comparable_old != comparable_new:
        raise ProgramError("completion requalification changed the scientific report")
    _completion_requalification_inputs(
        root,
        old_report,
        new_report,
        manifest["changed_inputs"],
        REQUALIFICATION_ALLOWED_INPUTS.get(step_id, frozenset()),
    )
    _validate_requalification_evidence(
        root, validation_evidence, step_id, manifest["changed_inputs"]
    )
    boundary_refresh = manifest.get("boundary_refresh")
    if boundary_refresh is not None:
        boundary_refresh = _validate_boundary_refresh(
            root, manifest, snapshot, boundary_refresh, steps, plan_hash
        )
    review_metadata = new_review.get("requalification")
    if review_metadata != {
        "claim_unchanged": True,
        "changed_inputs": manifest["changed_inputs"],
        "validation_evidence": validation_evidence,
    }:
        raise ProgramError("independent review does not bind the requalification scope")
    if (
        new_report.get("program_id") != state.get("program_id")
        or new_report.get("step_id") != step_id
        or new_report.get("decision") != steps[step_id]["pass_decision"]
        or not new_report.get("executor_id")
        or new_review.get("decision") != "PASS"
        or not new_review.get("auditor_id")
        or new_review["auditor_id"] == new_report["executor_id"]
        or new_review["auditor_id"] == manifest["executor_id"]
        or new_review.get("report_sha256") != replacement_report_reference["sha256"]
    ):
        raise ProgramError("replacement completion review is missing or self-audited")
    return {
        "state": state,
        "ledger": ledger,
        "steps": steps,
        "plan_hash": plan_hash,
        "old_record": old_record,
        "new_record": {
            "kind": "audited_report",
            "report": replacement_report_reference,
            "review": replacement_review_reference,
        },
        "old_report": old_report,
        "old_review": old_review,
        "new_report": new_report,
        "new_review": new_review,
        "boundary_refresh": boundary_refresh,
    }


def recover_completion_repair(repo_root: Path, journal_path: Path):
    root = repo_root.resolve()
    journal_path = journal_path.resolve()
    journal_path.relative_to(root)
    _require_unsealed_publication_path(root, journal_path)
    journal = json.loads(journal_path.read_text())
    if journal.get("schema") != COMPLETION_REPAIR_PUBLICATION_SCHEMA:
        raise ProgramError("unknown completion repair publication journal")
    manifest_reference = journal.get("manifest")
    _verify_references(root, [manifest_reference])
    manifest_path = root / manifest_reference["path"]
    if file_hash(manifest_path) != journal.get("manifest_sha256"):
        raise ProgramError("completion repair manifest changed")
    manifest = json.loads(manifest_path.read_text())
    boundary_reference = journal.get("boundary")
    _verify_references(root, [boundary_reference])
    boundary = json.loads((root / boundary_reference["path"]).read_text())
    if (
        boundary.get("schema") != "generic_neural_solver.phase_boundary.v1"
        or boundary.get("outcome") != "REPAIRED"
        or boundary.get("step_id") != manifest.get("step_id")
    ):
        raise ProgramError("completion repair boundary is malformed")
    _verify_immutable_references(root, boundary.get("evidence", []))
    for relative, entry in journal["files"].items():
        payload = journal_path.parent / entry["payload"]
        if file_hash(payload) != entry["after_sha256"]:
            raise ProgramError("completion repair publication payload changed")
    candidate_snapshot = {
        relative: (journal_path.parent / entry["payload"]).read_text()
        for relative, entry in journal["files"].items()
    }
    candidate = Program(root, snapshot=candidate_snapshot)
    if not candidate.boundary_current():
        raise ProgramError("completion repair publication has stale evidence")
    initial_files = journal["files"]
    _verify_live(root, initial_files)
    for relative, entry in initial_files.items():
        if file_hash(root / relative) != entry["after_sha256"]:
            _verify_live(root, initial_files)
            atomic_write_text(root / relative, (journal_path.parent / entry["payload"]).read_text())
    status = Program(root).status()
    if not status["phase_boundary_current"]:
        raise ProgramError("completion repair publication did not restore a current boundary")
    return status


def repair_completion(repo_root: Path, manifest_path: Path):
    """Publish a reviewed replacement for stale completion metadata."""
    root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest_path.relative_to(root)
    _require_unsealed_publication_path(root, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != COMPLETION_REPAIR_SCHEMA:
        raise ProgramError("unknown completion metadata repair manifest")
    publication = manifest_path.parent / "publication"
    journal_path = publication / "publication.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal.get("manifest_sha256") != file_hash(manifest_path):
            raise ProgramError("existing completion repair belongs to another manifest")
        return recover_completion_repair(root, journal_path)
    intent_path = publication / "intent.json"
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
        _verify_references(root, [intent["manifest"]])
        if intent["manifest"]["path"] != str(manifest_path.relative_to(root)):
            raise ProgramError("completion repair intent belongs to another manifest")
        snapshot = intent["initial_files"]
        initial_files = _snapshot_files(snapshot)
    else:
        snapshot = _read_live_snapshot(root)
        initial_files = _snapshot_files(snapshot)
    _verify_live(root, initial_files)
    validated = _validate_completion_repair(root, manifest, snapshot)
    activity_id = manifest["activity_id"]
    timestamp = manifest.get("created_at") or datetime.now(timezone.utc).isoformat()
    decision_id = activity_id + "-completion-metadata-repair"
    if any(entry.get("decision_id") == decision_id for entry in validated["ledger"]["decisions"]):
        raise ProgramError("duplicate completion repair decision")
    ledger = copy.deepcopy(validated["ledger"])
    decision_entry = {
        "decision_id": decision_id,
        "attempt_id": activity_id,
        "decision": "COMPLETION_METADATA_REPAIRED",
        "step_id": manifest["step_id"],
        "master_sha256": validated["plan_hash"],
        "previous_master_sha256": validated["plan_hash"],
        "master_change": "COMPLETION_BINDING_ONLY__SCIENTIFIC_RESULT_UNCHANGED",
        "manifest": _reference(root, manifest_path),
        "old_completion": validated["old_record"],
        "replacement_completion": validated["new_record"],
        "effect": "Restored live-program loadability without rerunning or reclassifying the Explore diagnostic.",
        "owner": manifest["executor_id"],
        "status": "ACTIVE",
    }
    ledger["decisions"].append(decision_entry)
    state = copy.deepcopy(validated["state"])
    followup = state.get("step_followups", {}).get(manifest["step_id"])
    if not isinstance(followup, dict):
        raise ProgramError("completion repair requires the step follow-up record")
    repair_record = {
        "activity_id": activity_id,
        "reason": manifest.get("reason", "Repair stale completion metadata"),
        "old_completion": validated["old_record"],
        "replacement_completion": validated["new_record"],
        "removed_inputs": manifest["removed_inputs"],
    }
    followup.setdefault("completion_metadata_repairs", []).append(repair_record)
    state["program_steps"][manifest["step_id"]] = validated["new_record"]
    state.update({
        "latest_attempt_id": activity_id,
        "latest_activity_kind": "COMPLETION_METADATA_REPAIR",
        "latest_phase": manifest["step_id"],
        "artifact_root": str(manifest_path.parent.relative_to(root)),
        "experiment_root": str(manifest_path.parent.relative_to(root)),
        "updated_at": timestamp,
        "program_revision_decision": decision_id,
    })
    status = Program(root, snapshot={
        PLAN_PATH: snapshot[PLAN_PATH],
        DECISIONS_PATH: json.dumps(ledger),
        STATE_PATH: json.dumps(state),
    }).status(check_boundary=False)
    state["next_permitted_action"] = (
        f"Continue {status['next_step']}; repair local findings, refresh at its result, and follow verified dependencies."
        if status["next_step"] else "No primary step is ready; resolve recorded external evidence blockers."
    )
    state["latest_state"] = "CONTINUE_READY_WORK" if status["next_step"] else "BLOCKED_ON_RECORDED_EVIDENCE"
    boundary_path = manifest_path.parent / "phase-boundary.json"
    boundary = {
        "schema": "generic_neural_solver.phase_boundary.v1",
        "activity_id": activity_id,
        "step_id": manifest["step_id"],
        "previous_activity_id": manifest["previous_activity_id"],
        "master_sha256": validated["plan_hash"],
        "outcome": "REPAIRED",
        "completion": None,
        "repairs_completed": [manifest.get("reason", "Repair stale completion metadata")],
        "repair_actions": [],
        "external_blockers": [],
        "advisory_followups": [
            "The historical completion packet remains immutable; only its live metadata binding was repaired."
        ],
        "evidence": [
            _reference(root, manifest_path),
            *manifest["evidence"],
            manifest["replacement_report"],
            manifest["replacement_review"],
        ],
    }
    atomic_write_json(boundary_path, boundary)
    boundary_reference = _reference(root, boundary_path)
    followup["boundary"] = boundary_reference
    decision_entry.update({
        "boundary": boundary_reference,
        "next_step": status["next_step"],
    })
    ledger["updated_at"] = timestamp
    state["last_phase_boundary"] = _reference(root, boundary_path)
    receipt_path = publication / "boundary-receipt.json"
    publication.mkdir(exist_ok=True)
    atomic_write_json(publication / "intent.json", {
        "schema": "generic_neural_solver.completion_repair_intent.v1",
        "manifest": _reference(root, manifest_path),
        "initial_files": snapshot,
    })
    atomic_write_json(receipt_path, {
        "schema": "generic_neural_solver.boundary_receipt.v1",
        "master_sha256": validated["plan_hash"],
        "progress_sha256": progress_hash(state),
        "boundary": boundary_reference,
        "next_step": status["next_step"],
    })
    state["last_phase_boundary"] = _reference(root, receipt_path)
    payloads = {
        PLAN_PATH: ("master.md", snapshot[PLAN_PATH]),
        DECISIONS_PATH: ("decision-ledger.json", json.dumps(ledger, indent=2) + "\n"),
        STATE_PATH: ("current-state.json", json.dumps(state, indent=2) + "\n"),
    }
    files = {}
    for relative, (filename, content) in payloads.items():
        atomic_write_text(publication / filename, content)
        atomic_write_text(publication / ("before-" + filename), snapshot[relative])
        files[relative] = {
            "payload": filename,
            "before_sha256": initial_files[relative]["before_sha256"],
            "after_sha256": _text_hash(content),
        }
    _verify_live(root, initial_files)
    atomic_write_json(journal_path, {
        "schema": COMPLETION_REPAIR_PUBLICATION_SCHEMA,
        "manifest": _reference(root, manifest_path),
        "manifest_sha256": file_hash(manifest_path),
        "boundary": boundary_reference,
        "files": files,
        "evidence": [
            boundary_reference,
            _reference(root, receipt_path),
            _reference(root, manifest_path),
        ],
    })
    return recover_completion_repair(root, journal_path)


def recover_completion_requalification(repo_root: Path, journal_path: Path):
    root = repo_root.resolve()
    journal_path = journal_path.resolve()
    journal_path.relative_to(root)
    _require_unsealed_publication_path(root, journal_path)
    journal = json.loads(journal_path.read_text())
    if journal.get("schema") != COMPLETION_REQUALIFICATION_PUBLICATION_SCHEMA:
        raise ProgramError("unknown completion requalification publication journal")
    if set(journal.get("files", {})) != {PLAN_PATH, DECISIONS_PATH, STATE_PATH}:
        raise ProgramError("requalification publication must target live program files")
    manifest_reference = journal.get("manifest")
    _verify_references(root, [manifest_reference])
    manifest_path = root / manifest_reference["path"]
    if file_hash(manifest_path) != journal.get("manifest_sha256"):
        raise ProgramError("completion requalification manifest changed")
    manifest = json.loads(manifest_path.read_text())
    intent_path = journal_path.parent / "intent.json"
    if not intent_path.is_file():
        raise ProgramError("completion requalification intent is missing")
    intent = json.loads(intent_path.read_text())
    if intent.get("schema") != "generic_neural_solver.completion_requalification_intent.v1":
        raise ProgramError("completion requalification intent is malformed")
    if intent.get("manifest") != manifest_reference:
        raise ProgramError("completion requalification intent belongs to another manifest")
    snapshot = intent.get("initial_files")
    if not isinstance(snapshot, dict) or set(snapshot) != {PLAN_PATH, DECISIONS_PATH, STATE_PATH}:
        raise ProgramError("completion requalification intent lacks the live snapshot")
    for relative, entry in journal["files"].items():
        if file_hash(root / relative) not in {
            entry["before_sha256"], entry["after_sha256"]
        }:
            raise ProgramError("completion requalification live files changed")
        if _text_hash(snapshot[relative]) != entry["before_sha256"]:
            raise ProgramError("completion requalification snapshot does not match journal")
    validated = _validate_completion_requalification(root, manifest, snapshot)
    boundary_reference = journal.get("boundary")
    _verify_references(root, [boundary_reference])
    boundary = json.loads((root / boundary_reference["path"]).read_text())
    if (
        boundary.get("schema") != "generic_neural_solver.phase_boundary.v1"
        or boundary.get("outcome") != "REPAIRED"
        or boundary.get("repair_kind") != "COMPLETION_REQUALIFICATION"
        or boundary.get("step_id") != manifest.get("step_id")
    ):
        raise ProgramError("completion requalification boundary is malformed")
    if boundary != _build_completion_requalification_boundary(
        root, manifest_path, manifest, validated
    ):
        raise ProgramError("completion requalification boundary is inconsistent")
    _verify_immutable_references(root, boundary.get("evidence", []))
    for relative, entry in journal["files"].items():
        payload = journal_path.parent / entry["payload"]
        if file_hash(payload) != entry["after_sha256"]:
            raise ProgramError("completion requalification publication payload changed")
    candidate_snapshot = {
        relative: (journal_path.parent / entry["payload"]).read_text()
        for relative, entry in journal["files"].items()
    }
    candidate = Program(root, snapshot=candidate_snapshot)
    expected_record = validated["new_record"]
    candidate_record = candidate.state.get("program_steps", {}).get(
        manifest["step_id"]
    )
    if (
        candidate_record != expected_record
        or candidate_record.get("report") != manifest["replacement_report"]
        or candidate_record.get("review") != manifest["replacement_review"]
    ):
        raise ProgramError(
            "completion requalification candidate payload does not contain the validated replacement"
        )
    if not candidate.boundary_current():
        raise ProgramError("completion requalification publication has stale evidence")
    _verify_live(root, journal["files"])
    for relative, entry in journal["files"].items():
        if file_hash(root / relative) != entry["after_sha256"]:
            _verify_live(root, journal["files"])
            atomic_write_text(root / relative, (journal_path.parent / entry["payload"]).read_text())
    status = Program(root).status()
    if not status["phase_boundary_current"]:
        raise ProgramError("completion requalification did not restore a current boundary receipt")
    return status


def requalify_completion(repo_root: Path, manifest_path: Path):
    """Publish an independently reviewed replacement for stale scientific inputs."""
    root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest_path.relative_to(root)
    _require_unsealed_publication_path(root, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != COMPLETION_REQUALIFICATION_SCHEMA:
        raise ProgramError("unknown completion requalification manifest")
    publication = manifest_path.parent / "publication"
    journal_path = publication / "publication.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal.get("manifest_sha256") != file_hash(manifest_path):
            raise ProgramError("existing requalification belongs to another manifest")
        return recover_completion_requalification(root, journal_path)
    intent_path = publication / "intent.json"
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
        _verify_references(root, [intent["manifest"]])
        if intent["manifest"]["path"] != str(manifest_path.relative_to(root)):
            raise ProgramError("requalification intent belongs to another manifest")
        snapshot = intent["initial_files"]
        initial_files = _snapshot_files(snapshot)
    else:
        snapshot = _read_live_snapshot(root)
        initial_files = _snapshot_files(snapshot)
    _verify_live(root, initial_files)
    validated = _validate_completion_requalification(root, manifest, snapshot)
    activity_id = manifest["activity_id"]
    timestamp = manifest.get("created_at") or datetime.now(timezone.utc).isoformat()
    decision_id = activity_id + "-completion-requalification"
    if any(entry.get("decision_id") == decision_id for entry in validated["ledger"]["decisions"]):
        raise ProgramError("duplicate completion requalification decision")
    ledger = copy.deepcopy(validated["ledger"])
    decision_entry = {
        "decision_id": decision_id,
        "attempt_id": activity_id,
        "decision": "COMPLETION_REQUALIFIED",
        "step_id": manifest["step_id"],
        "master_sha256": validated["plan_hash"],
        "previous_master_sha256": validated["plan_hash"],
        "master_change": "COMPLETION_BINDING_ONLY__CLAIM_REQUALIFIED_AGAINST_CURRENT_INPUTS",
        "manifest": _reference(root, manifest_path),
        "old_completion": validated["old_record"],
        "replacement_completion": validated["new_record"],
        "changed_inputs": manifest["changed_inputs"],
        "validation_evidence": manifest["validation_evidence"],
        "effect": "Restored live-program loadability after independent current-input requalification; the old packet remains immutable.",
        "owner": manifest["executor_id"],
        "status": "ACTIVE",
    }
    boundary_refresh = validated.get("boundary_refresh")
    if boundary_refresh is not None:
        decision_entry["boundary_refresh"] = boundary_refresh
    ledger["decisions"].append(decision_entry)
    state = copy.deepcopy(validated["state"])
    followup = state.get("step_followups", {}).get(manifest["step_id"])
    if not isinstance(followup, dict):
        raise ProgramError("completion requalification requires the step follow-up record")
    requalification_record = {
        "activity_id": activity_id,
        "reason": manifest["reason"],
        "old_completion": validated["old_record"],
        "replacement_completion": validated["new_record"],
        "changed_inputs": manifest["changed_inputs"],
        "validation_evidence": manifest["validation_evidence"],
    }
    followup.setdefault("completion_requalifications", []).append(requalification_record)
    state["program_steps"][manifest["step_id"]] = validated["new_record"]
    if boundary_refresh is not None:
        refresh_followup = state["step_followups"].get(boundary_refresh["step_id"])
        if not isinstance(refresh_followup, dict):
            raise ProgramError("boundary evidence refresh requires the step follow-up record")
        refresh_followup.setdefault("boundary_refreshes", []).append({
            "activity_id": activity_id,
            "transaction_kind": "BOUNDARY_EVIDENCE_REFRESH",
            "reason": boundary_refresh["reason"],
            "old_boundary": boundary_refresh["old_boundary"],
            "replacement_boundary": boundary_refresh["replacement_boundary"],
            "old_master_sha256": boundary_refresh["old_master_sha256"],
            "new_master_sha256": boundary_refresh["new_master_sha256"],
            "changed_inputs": boundary_refresh["changed_inputs"],
            "validation_evidence": boundary_refresh["validation_evidence"],
        })
        refresh_followup.setdefault("repairs_completed", []).append(
            boundary_refresh["reason"]
        )
        refresh_followup["boundary"] = boundary_refresh["replacement_boundary"]
    state.update({
        "latest_attempt_id": activity_id,
        "latest_activity_kind": "COMPLETION_REQUALIFICATION",
        "latest_phase": manifest["step_id"],
        "artifact_root": str(manifest_path.parent.relative_to(root)),
        "experiment_root": str(manifest_path.parent.relative_to(root)),
        "updated_at": timestamp,
        "program_revision_decision": decision_id,
    })
    candidate = Program(root, snapshot={
        PLAN_PATH: snapshot[PLAN_PATH],
        DECISIONS_PATH: json.dumps(ledger),
        STATE_PATH: json.dumps(state),
    })
    status = candidate.status(check_boundary=False)
    next_step = status["next_step"]
    state["next_permitted_action"] = (
        f"Continue {next_step}; repair local findings, refresh at its result, and follow verified dependencies."
        if next_step else "No primary step is ready; resolve recorded external evidence blockers."
    )
    state["latest_state"] = "CONTINUE_READY_WORK" if next_step else "BLOCKED_ON_RECORDED_EVIDENCE"
    boundary_path = manifest_path.parent / "phase-boundary.json"
    boundary = _build_completion_requalification_boundary(
        root, manifest_path, manifest, validated
    )
    atomic_write_json(boundary_path, boundary)
    boundary_reference = _reference(root, boundary_path)
    followup["boundary"] = boundary_reference
    decision_entry.update({
        "boundary": boundary_reference,
        "next_step": next_step,
    })
    ledger["updated_at"] = timestamp
    state["last_phase_boundary"] = _reference(root, boundary_path)
    receipt_path = publication / "boundary-receipt.json"
    publication.mkdir(exist_ok=True)
    atomic_write_json(publication / "intent.json", {
        "schema": "generic_neural_solver.completion_requalification_intent.v1",
        "manifest": _reference(root, manifest_path),
        "initial_files": snapshot,
    })
    atomic_write_json(receipt_path, {
        "schema": "generic_neural_solver.boundary_receipt.v1",
        "master_sha256": validated["plan_hash"],
        "progress_sha256": progress_hash(state),
        "boundary": boundary_reference,
        "next_step": next_step,
    })
    state["last_phase_boundary"] = _reference(root, receipt_path)
    payloads = {
        PLAN_PATH: ("master.md", snapshot[PLAN_PATH]),
        DECISIONS_PATH: ("decision-ledger.json", json.dumps(ledger, indent=2) + "\n"),
        STATE_PATH: ("current-state.json", json.dumps(state, indent=2) + "\n"),
    }
    files = {}
    for relative, (filename, content) in payloads.items():
        atomic_write_text(publication / filename, content)
        atomic_write_text(publication / ("before-" + filename), snapshot[relative])
        files[relative] = {
            "payload": filename,
            "before_sha256": initial_files[relative]["before_sha256"],
            "after_sha256": _text_hash(content),
        }
    _verify_live(root, initial_files)
    atomic_write_json(journal_path, {
        "schema": COMPLETION_REQUALIFICATION_PUBLICATION_SCHEMA,
        "manifest": _reference(root, manifest_path),
        "manifest_sha256": file_hash(manifest_path),
        "boundary": boundary_reference,
        "files": files,
        "evidence": [
            boundary_reference,
            _reference(root, receipt_path),
            _reference(root, manifest_path),
        ],
    })
    return recover_completion_requalification(root, journal_path)


def _build_completion_requalification_boundary(root, manifest_path, manifest, validated):
    evidence = [
        _reference(root, manifest_path),
        *manifest["evidence"],
        *manifest["validation_evidence"],
        validated["new_record"]["report"],
        validated["new_record"]["review"],
    ]
    boundary = {
        "schema": "generic_neural_solver.phase_boundary.v1",
        "activity_id": manifest["activity_id"],
        "step_id": manifest["step_id"],
        "previous_activity_id": manifest["previous_activity_id"],
        "master_sha256": validated["plan_hash"],
        "outcome": "REPAIRED",
        "repair_kind": "COMPLETION_REQUALIFICATION",
        "completion": None,
        "repairs_completed": [manifest["reason"]],
        "repair_actions": [],
        "external_blockers": [],
        "advisory_followups": [
            "The historical completion packet remains immutable; the live binding now points to a separately reviewed current-input requalification.",
            "No historical replay, endpoint equivalence or training authorization is added by this metadata/provenance repair.",
        ],
        "evidence": evidence,
    }
    boundary_refresh = validated.get("boundary_refresh")
    if boundary_refresh is not None:
        boundary["repairs_completed"].append(boundary_refresh["reason"])
        boundary["boundary_refresh"] = boundary_refresh
        boundary["evidence"].extend([
            boundary_refresh["old_boundary"],
            boundary_refresh["replacement_boundary"],
            *boundary_refresh["validation_evidence"],
        ])
    return boundary


def _batch_entry_manifest(manifest, entry):
    return {
        "activity_id": manifest["activity_id"],
        "previous_activity_id": manifest["previous_activity_id"],
        "executor_id": manifest["executor_id"],
        "step_id": entry["step_id"],
        "previous_master_sha256": manifest["previous_master_sha256"],
        "old_completion": entry["old_completion"],
        "replacement_report": entry["replacement_report"],
        "replacement_review": entry["replacement_review"],
        "changed_inputs": entry["changed_inputs"],
        "validation_evidence": entry["validation_evidence"],
        "reason": entry["reason"],
        "evidence": manifest["evidence"],
    }


def _validate_completion_requalification_batch(root, manifest, snapshot):
    if manifest.get("schema") != COMPLETION_REQUALIFICATION_BATCH_SCHEMA:
        raise ProgramError("unknown completion requalification batch manifest")
    state, ledger, steps, plan_hash = _raw_program_binding(snapshot)
    required = (
        "activity_id", "previous_activity_id", "executor_id",
        "previous_master_sha256", "created_at", "entries", "reason", "evidence",
    )
    if any(field not in manifest for field in required):
        raise ProgramError("completion requalification batch manifest is incomplete")
    if (
        not all(
            isinstance(manifest[field], str) and manifest[field].strip()
            for field in (
                "activity_id", "previous_activity_id", "executor_id",
                "previous_master_sha256", "created_at", "reason",
            )
        )
        or not isinstance(manifest["entries"], list)
        or len(manifest["entries"]) < 2
        or not isinstance(manifest["evidence"], list)
        or not manifest["evidence"]
        or any(not isinstance(reference, dict) for reference in manifest["evidence"])
    ):
        raise ProgramError("completion requalification batch fields are invalid")
    if manifest["previous_activity_id"] != state.get("latest_attempt_id"):
        raise ProgramError("completion requalification batch predecessor does not match live activity")
    if manifest["previous_master_sha256"] != plan_hash:
        raise ProgramError("completion requalification batch is based on a superseded master")
    if state.get("pending_phase_boundary"):
        raise ProgramError("completion requalification batch cannot discard a pending terminal result")
    _verify_immutable_references(root, manifest["evidence"])
    step_ids = [entry.get("step_id") for entry in manifest["entries"]]
    if any(not isinstance(step_id, str) or not step_id for step_id in step_ids):
        raise ProgramError("completion requalification batch step ids are invalid")
    if len(set(step_ids)) != len(step_ids):
        raise ProgramError("completion requalification batch repeats a step")
    validated_entries = []
    entry_required = (
        "step_id", "old_completion", "replacement_report", "replacement_review",
        "changed_inputs", "validation_evidence", "reason",
    )
    for entry in manifest["entries"]:
        if not isinstance(entry, dict) or any(field not in entry for field in entry_required):
            raise ProgramError("completion requalification batch entry is incomplete")
        if entry["step_id"] not in steps:
            raise ProgramError("completion requalification batch names an unknown step")
        validated_entries.append(
            _validate_completion_requalification(
                root, _batch_entry_manifest(manifest, entry), snapshot
            )
        )
    boundary_refresh = manifest.get("boundary_refresh")
    validated_boundary_refresh = None
    if boundary_refresh is not None:
        validated_boundary_refresh = _validate_boundary_refresh(
            root, manifest, snapshot, boundary_refresh, steps, plan_hash
        )
    return {
        "state": state,
        "ledger": ledger,
        "steps": steps,
        "plan_hash": plan_hash,
        "step_ids": step_ids,
        "entries": validated_entries,
        "boundary_refresh": validated_boundary_refresh,
    }


def _validate_boundary_refresh(root, manifest, snapshot, refresh, steps, plan_hash):
    required = (
        "step_id", "old_boundary", "replacement_boundary", "changed_inputs",
        "validation_evidence", "old_master_sha256", "new_master_sha256", "reason",
    )
    if not isinstance(refresh, dict) or any(field not in refresh for field in required):
        raise ProgramError("boundary evidence refresh is incomplete")
    step_id = refresh["step_id"]
    if step_id not in steps:
        raise ProgramError("boundary evidence refresh names an unknown step")
    if any(
        not isinstance(refresh[field], str) or not refresh[field].strip()
        for field in ("old_master_sha256", "new_master_sha256", "reason")
    ):
        raise ProgramError("boundary evidence refresh reason is invalid")
    state = json.loads(snapshot[STATE_PATH])
    old_boundary_reference = state.get("step_followups", {}).get(step_id, {}).get("boundary")
    if refresh["old_boundary"] != old_boundary_reference:
        raise ProgramError("boundary evidence refresh does not bind the live follow-up boundary")
    _verify_references(root, [refresh["old_boundary"], refresh["replacement_boundary"]])
    validation_evidence = refresh["validation_evidence"]
    if (
        not isinstance(validation_evidence, list)
        or not validation_evidence
        or any(not isinstance(reference, dict) for reference in validation_evidence)
    ):
        raise ProgramError("boundary evidence refresh requires validation evidence")
    _verify_immutable_references(root, validation_evidence)
    old_boundary = json.loads((root / refresh["old_boundary"]["path"]).read_text())
    new_boundary = json.loads((root / refresh["replacement_boundary"]["path"]).read_text())
    old_boundary_path = root / refresh["old_boundary"]["path"]
    new_boundary_path = root / refresh["replacement_boundary"]["path"]
    if old_boundary.get("master_sha256") != refresh["old_master_sha256"]:
        raise ProgramError("boundary evidence refresh old master binding is inconsistent")
    if refresh["new_master_sha256"] != plan_hash:
        raise ProgramError("boundary evidence refresh replacement master is not current")
    changed_inputs = refresh["changed_inputs"]
    if not isinstance(changed_inputs, list) or not changed_inputs:
        raise ProgramError("boundary evidence refresh must identify changed inputs")
    changed_by_path = {}
    for change in changed_inputs:
        if (
            not isinstance(change, dict)
            or set(change) != {"path", "old_sha256", "new_sha256"}
            or not isinstance(change["path"], str)
            or change["path"] in MUTABLE_LIVE_PATHS
            or Path(change["path"]).is_absolute()
            or ".." in Path(change["path"]).parts
            or change["path"] in changed_by_path
        ):
            raise ProgramError("boundary evidence refresh changed-input declaration is invalid")
        changed_by_path[change["path"]] = change
    old_evidence = old_boundary.get("evidence")
    if not isinstance(old_evidence, list) or not old_evidence:
        raise ProgramError("boundary evidence refresh requires an old evidence list")
    refreshed_evidence = []
    for old_reference in old_evidence:
        path = old_reference.get("path")
        change = changed_by_path.get(path)
        if change is None:
            refreshed_evidence.append(old_reference)
            continue
        if change["old_sha256"] != old_reference.get("sha256"):
            raise ProgramError("boundary evidence refresh old hash does not match evidence")
        new_reference = copy.deepcopy(old_reference)
        new_reference["sha256"] = change["new_sha256"]
        if "bytes" in new_reference:
            new_reference["bytes"] = (root / path).stat().st_size
        if file_hash(root / path) != change["new_sha256"]:
            raise ProgramError(f"boundary evidence refresh input is not current: {path}")
        refreshed_evidence.append(new_reference)
    if set(changed_by_path) != {
        reference.get("path") for reference in old_evidence
        if reference != refreshed_evidence[old_evidence.index(reference)]
    }:
        raise ProgramError("boundary evidence refresh changed-input declaration is incomplete")
    for reference in refreshed_evidence:
        _verify_immutable_references(root, [reference])
    for reference in validation_evidence:
        document = json.loads((root / reference["path"]).read_text())
        if (
            document.get("schema") != BOUNDARY_REFRESH_VALIDATION_SCHEMA
            or document.get("decision") != "BOUNDARY_EVIDENCE_REFRESH_PASSED"
            or document.get("step_id") != step_id
            or document.get("old_boundary") != refresh["old_boundary"]
            or document.get("changed_inputs") != changed_inputs
            or document.get("old_master_sha256") != refresh["old_master_sha256"]
            or document.get("new_master_sha256") != refresh["new_master_sha256"]
            or document.get("claim_unchanged") is not True
            or document.get("numerical_rerun") is not False
            or document.get("model_execution") is not False
            or document.get("training_updates") != 0
        ):
            raise ProgramError("boundary evidence refresh validation does not bind the claim")
        _verify_document_references(root, document)
    expected_refresh = {
        "kind": "boundary_evidence_refresh",
        "old_boundary": refresh["old_boundary"],
        "old_master_sha256": refresh["old_master_sha256"],
        "new_master_sha256": refresh["new_master_sha256"],
        "changed_inputs": changed_inputs,
        "validation_evidence": validation_evidence,
        "claim_unchanged": True,
        "numerical_rerun": False,
    }
    expected_new_boundary = copy.deepcopy(old_boundary)
    expected_new_boundary.update({
        "activity_id": manifest["activity_id"],
        "previous_activity_id": old_boundary.get("activity_id"),
        "master_sha256": plan_hash,
        "previous_master_sha256": refresh["old_master_sha256"],
        "outcome": "REPAIRED",
        "repair_kind": "BOUNDARY_EVIDENCE_REFRESH",
        "repairs_completed": [refresh["reason"]],
        "advisory_followups": [
            *old_boundary.get("advisory_followups", []),
            "This refresh changes only evidence fingerprints; the p5r.design partial claim remains open.",
        ],
        "evidence": [*refreshed_evidence, *validation_evidence],
        "boundary_refresh": expected_refresh,
    })
    if new_boundary != expected_new_boundary:
        raise ProgramError("replacement boundary changes more than its declared evidence refresh")
    if file_hash(old_boundary_path) != refresh["old_boundary"]["sha256"]:
        raise ProgramError("old boundary evidence changed")
    if file_hash(new_boundary_path) != refresh["replacement_boundary"]["sha256"]:
        raise ProgramError("replacement boundary evidence changed")
    return {
        "step_id": step_id,
        "old_boundary": refresh["old_boundary"],
        "replacement_boundary": refresh["replacement_boundary"],
        "old_master_sha256": refresh["old_master_sha256"],
        "new_master_sha256": refresh["new_master_sha256"],
        "changed_inputs": changed_inputs,
        "validation_evidence": validation_evidence,
        "reason": refresh["reason"],
    }


def _batch_entry_map(manifest):
    return {entry["step_id"]: entry for entry in manifest["entries"]}


def _build_batch_boundary(root, manifest_path, manifest, validated):
    evidence = [
        _reference(root, manifest_path),
        *manifest["evidence"],
    ]
    for entry in manifest["entries"]:
        evidence.extend([
            entry["replacement_report"],
            entry["replacement_review"],
            *entry["validation_evidence"],
        ])
    boundary = {
        "schema": "generic_neural_solver.phase_boundary.v1",
        "activity_id": manifest["activity_id"],
        "step_id": None,
        "step_ids": validated["step_ids"],
        "previous_activity_id": manifest["previous_activity_id"],
        "master_sha256": validated["plan_hash"],
        "outcome": "REPAIRED",
        "repair_kind": "COMPLETION_REQUALIFICATION_BATCH",
        "completion": None,
        "repairs_completed": [entry["reason"] for entry in manifest["entries"]],
        "repair_actions": [],
        "external_blockers": [],
        "advisory_followups": [
            "The historical completion packets remain immutable; live bindings now point to separately reviewed current-input requalifications.",
            "No historical replay, endpoint equivalence or training authorization is added by this metadata/provenance repair.",
        ],
        "evidence": evidence,
    }
    boundary_refresh = validated.get("boundary_refresh")
    if boundary_refresh is not None:
        evidence.extend([
            boundary_refresh["old_boundary"],
            boundary_refresh["replacement_boundary"],
            *boundary_refresh["validation_evidence"],
        ])
        boundary["repairs_completed"].append(boundary_refresh["reason"])
        boundary["boundary_refresh"] = boundary_refresh
    return boundary


def _build_batch_publication_payloads(
    root, manifest_path, manifest, snapshot, validated, boundary_reference,
    receipt_reference=None,
):
    timestamp = manifest["created_at"]
    entry_map = _batch_entry_map(manifest)
    ledger = copy.deepcopy(validated["ledger"])
    boundary_refresh = validated.get("boundary_refresh")
    old_completions = {
        step_id: entry["old_record"]
        for step_id, entry in zip(validated["step_ids"], validated["entries"])
    }
    replacement_completions = {
        step_id: entry["new_record"]
        for step_id, entry in zip(validated["step_ids"], validated["entries"])
    }
    decision_id = manifest["activity_id"] + "-completion-requalification-batch"
    decision_entry = {
        "decision_id": decision_id,
        "attempt_id": manifest["activity_id"],
        "decision": "COMPLETION_REQUALIFIED_BATCH",
        "step_ids": validated["step_ids"],
        "master_sha256": validated["plan_hash"],
        "previous_master_sha256": validated["plan_hash"],
        "master_change": "COMPLETION_BINDINGS_ONLY__CLAIMS_REQUALIFIED_AGAINST_CURRENT_INPUTS",
        "manifest": _reference(root, manifest_path),
        "old_completions": old_completions,
        "replacement_completions": replacement_completions,
        "changed_inputs": {
            entry["step_id"]: entry["changed_inputs"] for entry in manifest["entries"]
        },
        "validation_evidence": {
            entry["step_id"]: entry["validation_evidence"] for entry in manifest["entries"]
        },
        "effect": "Restored live-program loadability after one atomic current-input requalification batch; old packets remain immutable.",
        "owner": manifest["executor_id"],
        "status": "ACTIVE",
    }
    if boundary_refresh is not None:
        decision_entry["boundary_refresh"] = boundary_refresh
    ledger["decisions"].append(decision_entry)
    state = copy.deepcopy(validated["state"])
    for step_id, validated_entry in zip(validated["step_ids"], validated["entries"]):
        followup = state.get("step_followups", {}).get(step_id)
        if not isinstance(followup, dict):
            raise ProgramError("completion requalification batch requires every step follow-up record")
        entry = entry_map[step_id]
        followup.setdefault("completion_requalifications", []).append({
            "activity_id": manifest["activity_id"],
            "transaction_kind": "COMPLETION_REQUALIFICATION_BATCH",
            "reason": entry["reason"],
            "old_completion": validated_entry["old_record"],
            "replacement_completion": validated_entry["new_record"],
            "changed_inputs": entry["changed_inputs"],
            "validation_evidence": entry["validation_evidence"],
        })
        state["program_steps"][step_id] = validated_entry["new_record"]
    if boundary_refresh is not None:
        followup = state.get("step_followups", {}).get(boundary_refresh["step_id"])
        if not isinstance(followup, dict):
            raise ProgramError("boundary evidence refresh requires a step follow-up record")
        followup.setdefault("boundary_refreshes", []).append({
            "activity_id": manifest["activity_id"],
            "transaction_kind": "BOUNDARY_EVIDENCE_REFRESH",
            "reason": boundary_refresh["reason"],
            "old_boundary": boundary_refresh["old_boundary"],
            "replacement_boundary": boundary_refresh["replacement_boundary"],
            "old_master_sha256": boundary_refresh["old_master_sha256"],
            "new_master_sha256": boundary_refresh["new_master_sha256"],
            "changed_inputs": boundary_refresh["changed_inputs"],
            "validation_evidence": boundary_refresh["validation_evidence"],
        })
        followup.setdefault("repairs_completed", []).append(boundary_refresh["reason"])
        followup["boundary"] = boundary_refresh["replacement_boundary"]
    state.update({
        "latest_attempt_id": manifest["activity_id"],
        "latest_activity_kind": "COMPLETION_REQUALIFICATION_BATCH",
        "latest_phase": "COMPLETION_REQUALIFICATION_BATCH",
        "artifact_root": str(manifest_path.parent.relative_to(root)),
        "experiment_root": str(manifest_path.parent.relative_to(root)),
        "updated_at": timestamp,
        "program_revision_decision": decision_id,
    })
    candidate = Program(root, snapshot={
        PLAN_PATH: snapshot[PLAN_PATH],
        DECISIONS_PATH: json.dumps(ledger),
        STATE_PATH: json.dumps(state),
    })
    next_step = candidate.status(check_boundary=False)["next_step"]
    state["next_permitted_action"] = (
        f"Continue {next_step}; repair local findings, refresh at its result, and follow verified dependencies."
        if next_step else "No primary step is ready; resolve recorded external evidence blockers."
    )
    state["latest_state"] = "CONTINUE_READY_WORK" if next_step else "BLOCKED_ON_RECORDED_EVIDENCE"
    for step_id in validated["step_ids"]:
        state["step_followups"][step_id]["boundary"] = boundary_reference
    decision_entry.update({
        "boundary": boundary_reference,
        "next_step": next_step,
    })
    ledger["updated_at"] = timestamp
    state["last_phase_boundary"] = receipt_reference or boundary_reference
    return {"state": state, "ledger": ledger, "next_step": next_step}


def recover_completion_requalification_batch(repo_root: Path, journal_path: Path):
    root = repo_root.resolve()
    journal_path = journal_path.resolve()
    journal_path.relative_to(root)
    _require_unsealed_publication_path(root, journal_path)
    journal = json.loads(journal_path.read_text())
    if journal.get("schema") != COMPLETION_REQUALIFICATION_BATCH_PUBLICATION_SCHEMA:
        raise ProgramError("unknown completion requalification batch publication journal")
    if set(journal.get("files", {})) != {PLAN_PATH, DECISIONS_PATH, STATE_PATH}:
        raise ProgramError("requalification batch publication must target live program files")
    manifest_reference = journal.get("manifest")
    _verify_references(root, [manifest_reference])
    manifest_path = root / manifest_reference["path"]
    if file_hash(manifest_path) != journal.get("manifest_sha256"):
        raise ProgramError("completion requalification batch manifest changed")
    manifest = json.loads(manifest_path.read_text())
    intent_path = journal_path.parent / "intent.json"
    if not intent_path.is_file():
        raise ProgramError("completion requalification batch intent is missing")
    intent = json.loads(intent_path.read_text())
    if intent.get("schema") != "generic_neural_solver.completion_requalification_batch_intent.v1":
        raise ProgramError("completion requalification batch intent is malformed")
    if intent.get("manifest") != manifest_reference:
        raise ProgramError("completion requalification batch intent belongs to another manifest")
    snapshot = intent.get("initial_files")
    if not isinstance(snapshot, dict) or set(snapshot) != {PLAN_PATH, DECISIONS_PATH, STATE_PATH}:
        raise ProgramError("completion requalification batch intent lacks the live snapshot")
    for relative, entry in journal["files"].items():
        if file_hash(root / relative) not in {
            entry["before_sha256"], entry["after_sha256"]
        }:
            raise ProgramError("completion requalification batch live files changed")
        if _text_hash(snapshot[relative]) != entry["before_sha256"]:
            raise ProgramError("completion requalification batch snapshot does not match journal")
    validated = _validate_completion_requalification_batch(root, manifest, snapshot)
    boundary_reference = journal.get("boundary")
    _verify_references(root, [boundary_reference])
    boundary = json.loads((root / boundary_reference["path"]).read_text())
    if boundary != _build_batch_boundary(root, manifest_path, manifest, validated):
        raise ProgramError("completion requalification batch boundary is malformed")
    _verify_immutable_references(root, boundary.get("evidence", []))
    _verify_immutable_references(root, journal.get("evidence", []))
    for relative, entry in journal["files"].items():
        payload = journal_path.parent / entry["payload"]
        if file_hash(payload) != entry["after_sha256"]:
            raise ProgramError("completion requalification batch publication payload changed")
    candidate_snapshot = {
        relative: (journal_path.parent / entry["payload"]).read_text()
        for relative, entry in journal["files"].items()
    }
    candidate = Program(root, snapshot=candidate_snapshot)
    receipt_path = journal_path.parent / "boundary-receipt.json"
    receipt_reference = _reference(root, receipt_path)
    if receipt_reference not in journal.get("evidence", []):
        raise ProgramError("completion requalification batch receipt is not journaled")
    receipt = json.loads(receipt_path.read_text())
    expected = _build_batch_publication_payloads(
        root, manifest_path, manifest, snapshot, validated,
        boundary_reference, receipt_reference,
    )
    candidate_ledger = json.loads(candidate_snapshot[DECISIONS_PATH])
    if candidate_ledger != expected["ledger"] or candidate.state != expected["state"]:
        raise ProgramError(
            "completion requalification batch candidate payload does not match the validated transaction"
        )
    expected_receipt = {
        "schema": "generic_neural_solver.boundary_receipt.v1",
        "master_sha256": validated["plan_hash"],
        "progress_sha256": progress_hash(expected["state"]),
        "boundary": boundary_reference,
        "next_step": expected["next_step"],
    }
    if receipt != expected_receipt:
        raise ProgramError("completion requalification batch receipt is inconsistent")
    for step_id, validated_entry in zip(validated["step_ids"], validated["entries"]):
        expected_record = validated_entry["new_record"]
        candidate_record = candidate.state.get("program_steps", {}).get(step_id)
        manifest_entry = next(
            entry for entry in manifest["entries"] if entry["step_id"] == step_id
        )
        if (
            candidate_record != expected_record
            or candidate_record.get("report") != manifest_entry["replacement_report"]
            or candidate_record.get("review") != manifest_entry["replacement_review"]
        ):
            raise ProgramError(
                "completion requalification batch candidate payload does not contain a validated replacement"
            )
    if not candidate.boundary_current():
        raise ProgramError("completion requalification batch publication has stale evidence")
    _verify_live(root, journal["files"])
    for relative, entry in journal["files"].items():
        if file_hash(root / relative) != entry["after_sha256"]:
            _verify_live(root, journal["files"])
            atomic_write_text(root / relative, (journal_path.parent / entry["payload"]).read_text())
    status = Program(root).status()
    if not status["phase_boundary_current"]:
        raise ProgramError("completion requalification batch did not restore a current boundary receipt")
    return status


def requalify_completion_batch(repo_root: Path, manifest_path: Path):
    """Publish multiple independently reviewed replacements from one live snapshot."""
    root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest_path.relative_to(root)
    _require_unsealed_publication_path(root, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != COMPLETION_REQUALIFICATION_BATCH_SCHEMA:
        raise ProgramError("unknown completion requalification batch manifest")
    publication = manifest_path.parent / "publication"
    journal_path = publication / "publication.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal.get("manifest_sha256") != file_hash(manifest_path):
            raise ProgramError("existing requalification batch belongs to another manifest")
        return recover_completion_requalification_batch(root, journal_path)
    intent_path = publication / "intent.json"
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
        _verify_references(root, [intent["manifest"]])
        if intent["manifest"]["path"] != str(manifest_path.relative_to(root)):
            raise ProgramError("requalification batch intent belongs to another manifest")
        snapshot = intent["initial_files"]
        initial_files = _snapshot_files(snapshot)
    else:
        snapshot = _read_live_snapshot(root)
        initial_files = _snapshot_files(snapshot)
    _verify_live(root, initial_files)
    validated = _validate_completion_requalification_batch(root, manifest, snapshot)
    decision_id = manifest["activity_id"] + "-completion-requalification-batch"
    if any(entry.get("decision_id") == decision_id for entry in validated["ledger"]["decisions"]):
        raise ProgramError("duplicate completion requalification batch decision")
    boundary_path = manifest_path.parent / "phase-boundary.json"
    publication.mkdir(exist_ok=True)
    atomic_write_json(publication / "intent.json", {
        "schema": "generic_neural_solver.completion_requalification_batch_intent.v1",
        "manifest": _reference(root, manifest_path),
        "initial_files": snapshot,
    })
    boundary = _build_batch_boundary(root, manifest_path, manifest, validated)
    atomic_write_json(boundary_path, boundary)
    boundary_reference = _reference(root, boundary_path)
    prepared = _build_batch_publication_payloads(
        root, manifest_path, manifest, snapshot, validated, boundary_reference
    )
    receipt_path = publication / "boundary-receipt.json"
    atomic_write_json(receipt_path, {
        "schema": "generic_neural_solver.boundary_receipt.v1",
        "master_sha256": validated["plan_hash"],
        "progress_sha256": progress_hash(prepared["state"]),
        "boundary": boundary_reference,
        "next_step": prepared["next_step"],
    })
    receipt_reference = _reference(root, receipt_path)
    prepared = _build_batch_publication_payloads(
        root, manifest_path, manifest, snapshot, validated,
        boundary_reference, receipt_reference,
    )
    payloads = {
        PLAN_PATH: ("master.md", snapshot[PLAN_PATH]),
        DECISIONS_PATH: ("decision-ledger.json", json.dumps(prepared["ledger"], indent=2) + "\n"),
        STATE_PATH: ("current-state.json", json.dumps(prepared["state"], indent=2) + "\n"),
    }
    files = {}
    for relative, (filename, content) in payloads.items():
        atomic_write_text(publication / filename, content)
        atomic_write_text(publication / ("before-" + filename), snapshot[relative])
        files[relative] = {
            "payload": filename,
            "before_sha256": initial_files[relative]["before_sha256"],
            "after_sha256": _text_hash(content),
        }
    _verify_live(root, initial_files)
    atomic_write_json(journal_path, {
        "schema": COMPLETION_REQUALIFICATION_BATCH_PUBLICATION_SCHEMA,
        "manifest": _reference(root, manifest_path),
        "manifest_sha256": file_hash(manifest_path),
        "boundary": boundary_reference,
        "files": files,
        "evidence": [
            boundary_reference,
            _reference(root, receipt_path),
            _reference(root, manifest_path),
        ],
    })
    return recover_completion_requalification_batch(root, journal_path)


def recover_publication(repo_root: Path, journal_path: Path):
    root = repo_root.resolve()
    journal_path = journal_path.resolve()
    journal_path.relative_to(root)
    _require_unsealed_publication_path(root, journal_path)
    journal = json.loads(journal_path.read_text())
    if journal.get("schema") != "generic_neural_solver.publication.v1":
        raise ProgramError("unknown publication journal")
    if set(journal["files"]) != {PLAN_PATH, DECISIONS_PATH, STATE_PATH}:
        raise ProgramError("publication must target only master and the two live ledgers")
    _verify_references(root, journal["evidence"])
    boundary = json.loads((root / journal["evidence"][0]["path"]).read_text())
    _verify_immutable_references(root, boundary["evidence"])
    proposed = Program(root, snapshot={
        relative: (journal_path.parent / entry["payload"]).read_text()
        for relative, entry in journal["files"].items()
    })
    if not proposed.boundary_current():
        raise ProgramError("proposed publication has stale evidence")
    for relative, entry in journal["files"].items():
        payload = journal_path.parent / entry["payload"]
        if file_hash(payload) != entry["after_sha256"]:
            raise ProgramError("publication payload changed")
    _verify_live(root, journal["files"])
    for relative in (PLAN_PATH, DECISIONS_PATH, STATE_PATH):
        entry = journal["files"][relative]
        if file_hash(root / relative) != entry["after_sha256"]:
            _verify_live(root, journal["files"])
            atomic_write_text(root / relative, (journal_path.parent / entry["payload"]).read_text())
    status = Program(root).status()
    if not status["phase_boundary_current"]:
        raise ProgramError("publication did not restore a current boundary receipt")
    return status


def advance(repo_root: Path, boundary_path: Path):
    root = repo_root.resolve()
    boundary_path = boundary_path.resolve()
    boundary_path.relative_to(root)
    _require_unsealed_publication_path(root, boundary_path)
    publication = boundary_path.parent / "publication"
    journal_path = publication / "publication.json"
    intent_path = publication / "intent.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal["boundary_sha256"] != file_hash(boundary_path):
            raise ProgramError("published boundary input changed")
        return recover_publication(root, journal_path)
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
        _verify_references(root, [intent["boundary"]])
        if intent["boundary"]["path"] != str(boundary_path.relative_to(root)):
            raise ProgramError("publication intent belongs to another boundary")
        program = Program(root, snapshot=intent["initial_files"])
        timestamp = intent["created_at"]
    else:
        program = Program(root)
        timestamp = datetime.now(timezone.utc).isoformat()
    initial_files = {
        relative: {"before_sha256": _text_hash(content), "after_sha256": _text_hash(content)}
        for relative, content in program.snapshot.items()
    }
    _verify_live(root, initial_files)
    boundary = json.loads(boundary_path.read_text())
    if (
        boundary.get("schema") != "generic_neural_solver.phase_boundary.v1"
        or boundary.get("master_sha256") != program.plan_hash
        or boundary.get("previous_activity_id") != program.state["latest_attempt_id"]
        or not boundary.get("activity_id")
    ):
        raise ProgramError("stale or malformed phase boundary")
    outcome = boundary.get("outcome")
    step_id = boundary.get("step_id")
    if outcome not in {"COMPLETE", "PARTIAL", "FAILED", "INCONCLUSIVE", "REPAIRED", "REFRESH"}:
        raise ProgramError("unknown phase boundary outcome")
    if outcome == "REFRESH":
        if step_id is not None or boundary.get("completion"):
            raise ProgramError("refresh cannot close a phase")
        if boundary.get("repair_actions") or boundary.get("external_blockers"):
            raise ProgramError("refresh cannot discard unresolved repairs/blockers; name the affected step")
    elif step_id not in program.steps:
        raise ProgramError("phase boundary names an unknown step")
    if not boundary.get("evidence"):
        raise ProgramError("phase boundary requires concrete evidence")
    _verify_immutable_references(root, boundary["evidence"])
    repairs = boundary.get("repair_actions", [])
    blockers = boundary.get("external_blockers", [])
    advisory = boundary.get("advisory_followups", [])
    for values in (repairs, blockers, boundary.get("repairs_completed", []), advisory):
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ProgramError("repair/blocker descriptions must be nonempty strings")
    if outcome == "COMPLETE" and (repairs or blockers):
        raise ProgramError("unresolved repairs/blockers cannot close a phase")
    if outcome == "REPAIRED" and (
        repairs or blockers or not boundary.get("repairs_completed")
        or not program.followups.get(step_id)
    ):
        raise ProgramError("repaired readiness needs an existing finding and evidenced fixes, without blockers")
    if outcome in {"PARTIAL", "FAILED", "INCONCLUSIVE"} and not (repairs or blockers):
        raise ProgramError("unfinished work requires a concrete repair or external blocker")
    state = copy.deepcopy(program.state)
    pending, pending_step = program.pending_result()
    if pending:
        if step_id != pending_step:
            raise ProgramError("pending result needs its own step disposition before refresh")
        state.pop("pending_phase_boundary", None)
    if outcome == "COMPLETE":
        if step_id in program.complete:
            raise ProgramError("completed phase cannot be overwritten")
        if set(program.steps[step_id]["requires"]) - program.complete:
            raise ProgramError("phase closure is missing prerequisites")
        program._verify_completion(step_id, boundary["completion"])
        state["program_steps"][step_id] = boundary["completion"]
        program.complete.add(step_id)
    elif boundary.get("completion"):
        raise ProgramError("partial result cannot carry a completion")
    if step_id is not None:
        if step_id in program.complete and outcome != "COMPLETE":
            raise ProgramError("existing closure needs an explicit reviewed invalidation")
        state.setdefault("step_followups", {})[step_id] = {
            "outcome": outcome, "repair_actions": repairs,
            "external_blockers": blockers,
            "advisory_followups": advisory,
            "repairs_completed": boundary.get("repairs_completed", []),
            "boundary": _reference(root, boundary_path),
        }
    state.update({
        "latest_attempt_id": boundary["activity_id"],
        "latest_activity_kind": "PHASE_BOUNDARY_REPAIR_REFRESH",
        "latest_phase": step_id or "program-refresh",
        "artifact_root": str(boundary_path.parent.relative_to(root)),
        "experiment_root": str(boundary_path.parent.relative_to(root)),
        "updated_at": timestamp,
    })
    program.state = state
    program.followups = state.get("step_followups", {})
    status = program.status(check_boundary=False)
    next_step = status["next_step"]
    state["next_permitted_action"] = (
        f"Continue {next_step}; repair local findings, refresh at its result, and follow verified dependencies."
        if next_step else "No primary step is ready; resolve recorded external evidence blockers."
    )
    state["latest_state"] = "CONTINUE_READY_WORK" if next_step else "BLOCKED_ON_RECORDED_EVIDENCE"
    plan_before = program.snapshot[PLAN_PATH]
    if plan_before.count(POSITION_START) != 1 or plan_before.count(POSITION_END) != 1:
        raise ProgramError("master must contain exactly one generated current-position block")
    block = (
        f"{POSITION_START}\nLatest boundary: `{boundary['activity_id']}`; "
        f"step `{step_id or 'program-refresh'}`; result `{outcome}`.\n\n"
        f"Next primary work: `{next_step or 'NONE_READY'}`. "
        "Run the status command for complete dependencies and evidence.\n\n"
        f"Ready work: {', '.join('`' + value + '`' for value in status['ready_steps']) or 'none'}.\n"
        f"{POSITION_END}"
    )
    plan_after = plan_before.split(POSITION_START)[0] + block + plan_before.split(POSITION_END)[1]
    plan_hash = _text_hash(plan_after)
    state["canonical_plan_sha256"] = plan_hash
    decision_id = boundary["activity_id"] + "-repair-refresh"
    state["program_revision_decision"] = decision_id
    ledger = json.loads(program.snapshot[DECISIONS_PATH])
    if any(entry["decision_id"] == decision_id for entry in ledger["decisions"]):
        raise ProgramError("duplicate boundary decision; inspect its publication journal")
    ledger["decisions"].append({
        "decision_id": decision_id, "attempt_id": boundary["activity_id"],
        "decision": outcome, "step_id": step_id,
        "master_sha256": plan_hash, "previous_master_sha256": program.plan_hash,
        "master_change": "GENERATED_POSITION_ONLY__SCIENTIFIC_CONTRACT_UNCHANGED",
        "boundary": _reference(root, boundary_path), "next_step": next_step,
        "effect": "Repair and refresh completed; continue only verified ready work.",
    })
    ledger["updated_at"] = state["updated_at"]
    _verify_live(root, initial_files)
    publication.mkdir(exist_ok=True)
    if not intent_path.exists():
        atomic_write_json(intent_path, {
            "schema": "generic_neural_solver.publication_intent.v1",
            "boundary": _reference(root, boundary_path), "created_at": timestamp,
            "initial_files": program.snapshot,
        })
    receipt_path = publication / "boundary-receipt.json"
    atomic_write_json(receipt_path, {
        "schema": "generic_neural_solver.boundary_receipt.v1",
        "master_sha256": plan_hash, "progress_sha256": progress_hash(state),
        "boundary": _reference(root, boundary_path), "next_step": next_step,
    })
    state["last_phase_boundary"] = _reference(root, receipt_path)
    payloads = {
        PLAN_PATH: ("master.md", plan_after),
        DECISIONS_PATH: ("decision-ledger.json", json.dumps(ledger, indent=2) + "\n"),
        STATE_PATH: ("current-state.json", json.dumps(state, indent=2) + "\n"),
    }
    files = {}
    for relative, (filename, content) in payloads.items():
        atomic_write_text(publication / filename, content)
        atomic_write_text(publication / ("before-" + filename), program.snapshot[relative])
        files[relative] = {
            "payload": filename, "before_sha256": initial_files[relative]["before_sha256"],
            "after_sha256": _text_hash(content),
        }
    _verify_live(root, initial_files)
    _verify_immutable_references(root, boundary["evidence"])
    atomic_write_json(journal_path, {
        "schema": "generic_neural_solver.publication.v1",
        "boundary_sha256": file_hash(boundary_path), "files": files,
        "evidence": [_reference(root, boundary_path), _reference(root, receipt_path)],
    })
    return recover_publication(root, journal_path)


def revise(repo_root: Path, manifest_path: Path):
    """Publish a reviewed structural master revision transactionally."""
    root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest_path.relative_to(root)
    _require_unsealed_publication_path(root, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MASTER_REVISION_SCHEMA:
        raise ProgramError("unknown master revision manifest")

    publication = manifest_path.parent / "publication"
    journal_path = publication / "publication.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal.get("revision_manifest_sha256") != file_hash(manifest_path):
            raise ProgramError("existing revision publication belongs to another manifest")
        return recover_publication(root, journal_path)

    program = Program(root)
    activity_id = manifest.get("activity_id")
    previous_activity_id = manifest.get("previous_activity_id")
    executor_id = manifest.get("executor_id")
    if not all(isinstance(value, str) and value.strip() for value in (
        activity_id, previous_activity_id, executor_id,
    )):
        raise ProgramError("master revision requires activity, predecessor and executor identities")
    if previous_activity_id != program.state.get("latest_attempt_id"):
        raise ProgramError("master revision predecessor does not match live activity")
    if manifest.get("previous_master_sha256") != program.plan_hash:
        raise ProgramError("master revision is based on a superseded master")
    review_reference = manifest.get("review")
    if not isinstance(review_reference, dict):
        raise ProgramError("master revision requires a review reference")
    _verify_references(root, [review_reference])
    review = json.loads((root / review_reference["path"]).read_text())
    if (
        review.get("decision") != "PASS"
        or not isinstance(review.get("reviewer_id"), str)
        or not review["reviewer_id"].strip()
        or review["reviewer_id"] == executor_id
    ):
        raise ProgramError("master revision requires a distinct passing reviewer")
    evidence = manifest.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        raise ProgramError("master revision requires evidence")
    _verify_immutable_references(root, evidence)

    proposed_reference = manifest.get("proposed_plan")
    if not isinstance(proposed_reference, dict):
        raise ProgramError("master revision requires a proposed plan")
    proposed_path = root / proposed_reference["path"]
    if proposed_path == root / PLAN_PATH:
        raise ProgramError("proposed plan must be staged outside the live master path")
    if file_hash(proposed_path) != proposed_reference.get("sha256"):
        raise ProgramError("proposed master hash changed")
    proposed_text = proposed_path.read_text()
    if proposed_text.count(POSITION_START) != 1 or proposed_text.count(POSITION_END) != 1:
        raise ProgramError("proposed master must contain exactly one current-position block")

    timestamp = manifest.get("created_at") or datetime.now(timezone.utc).isoformat()
    staged_decision_id = f"{activity_id}-staged"
    staged_plan_hash = _text_hash(proposed_text)
    staged_state = copy.deepcopy(program.state)
    staged_state["canonical_plan_sha256"] = staged_plan_hash
    staged_state["program_revision_decision"] = staged_decision_id
    staged_ledger = json.loads(program.snapshot[DECISIONS_PATH])
    staged_ledger["decisions"].append({
        "decision_id": staged_decision_id,
        "attempt_id": activity_id,
        "decision": "MASTER_REVISION_STAGED",
        "master_sha256": staged_plan_hash,
        "previous_master_sha256": program.plan_hash,
    })
    candidate = Program(root, snapshot={
        PLAN_PATH: proposed_text,
        DECISIONS_PATH: json.dumps(staged_ledger),
        STATE_PATH: json.dumps(staged_state),
    })
    candidate_status = candidate.status(check_boundary=False)
    next_step = candidate_status["next_step"]
    position = (
        f"{POSITION_START}\nLatest boundary: `{activity_id}`; step `program-refresh`; result `REFRESH`.\n\n"
        f"Next primary work: `{next_step or 'NONE_READY'}`. Run the status command for complete dependencies and evidence.\n\n"
        f"Ready work: {', '.join('`' + value + '`' for value in candidate_status['ready_steps']) or 'none'}.\n"
        f"{POSITION_END}"
    )
    final_plan = (
        proposed_text.split(POSITION_START)[0]
        + position
        + proposed_text.split(POSITION_END)[1]
    )
    plan_hash = _text_hash(final_plan)
    decision_id = f"{activity_id}-master-revision"
    if any(entry.get("decision_id") == decision_id for entry in staged_ledger["decisions"]):
        raise ProgramError("duplicate master revision decision")

    state = copy.deepcopy(program.state)
    state.update({
        "canonical_plan_sha256": plan_hash,
        "program_revision_decision": decision_id,
        "latest_attempt_id": activity_id,
        "latest_activity_kind": "MASTER_REVISION",
        "latest_phase": "program-refresh",
        "artifact_root": str(manifest_path.parent.relative_to(root)),
        "experiment_root": str(manifest_path.parent.relative_to(root)),
        "updated_at": timestamp,
        "next_permitted_action": (
            f"Continue {next_step}; repair local findings, refresh at its result, and follow verified dependencies."
            if next_step else "No primary step is ready; resolve recorded external evidence blockers."
        ),
        "latest_state": "CONTINUE_READY_WORK" if next_step else "BLOCKED_ON_RECORDED_EVIDENCE",
    })
    ledger = json.loads(program.snapshot[DECISIONS_PATH])
    ledger["decisions"].append({
        "decision_id": decision_id,
        "attempt_id": activity_id,
        "decision": "MASTER_REVISED",
        "step_id": None,
        "master_sha256": plan_hash,
        "previous_master_sha256": program.plan_hash,
        "master_change": manifest.get("master_change", "STRUCTURAL_MASTER_REPAIR"),
        "review": review_reference,
        "evidence": evidence,
        "next_step": next_step,
        "effect": manifest.get("effect", "Structural master repair published; unchanged scientific gates remain binding."),
        "owner": executor_id,
        "status": "ACTIVE",
    })
    ledger["updated_at"] = timestamp

    publication.mkdir(exist_ok=True)
    boundary_path = manifest_path.parent / "phase-boundary.json"
    receipt_path = publication / "boundary-receipt.json"
    boundary = {
        "schema": "generic_neural_solver.phase_boundary.v1",
        "activity_id": activity_id,
        "step_id": None,
        "previous_activity_id": previous_activity_id,
        "master_sha256": plan_hash,
        "outcome": "REFRESH",
        "completion": None,
        "repairs_completed": [manifest.get("reason", "Reviewed structural master repair")],
        "repair_actions": [],
        "external_blockers": [],
        "advisory_followups": [
            "This revision adds reconstruction engineering evidence only; historical source, input, trajectory and endpoint gates remain unchanged."
        ],
        "evidence": [],
    }
    atomic_write_text(publication / "master.md", final_plan)
    final_master_reference = _reference(root, publication / "master.md")
    boundary["evidence"] = [*evidence, review_reference, final_master_reference]
    atomic_write_json(boundary_path, boundary)
    boundary_reference = _reference(root, boundary_path)
    atomic_write_json(receipt_path, {
        "schema": "generic_neural_solver.boundary_receipt.v1",
        "master_sha256": plan_hash,
        "progress_sha256": progress_hash(state),
        "boundary": boundary_reference,
        "next_step": next_step,
    })
    state["last_phase_boundary"] = _reference(root, receipt_path)
    payloads = {
        PLAN_PATH: ("master.md", final_plan),
        DECISIONS_PATH: ("decision-ledger.json", json.dumps(ledger, indent=2) + "\n"),
        STATE_PATH: ("current-state.json", json.dumps(state, indent=2) + "\n"),
    }
    initial_files = {
        relative: {"before_sha256": _text_hash(content), "after_sha256": _text_hash(content)}
        for relative, content in program.snapshot.items()
    }
    _verify_live(root, initial_files)
    files = {}
    for relative, (filename, content) in payloads.items():
        atomic_write_text(publication / filename, content)
        atomic_write_text(publication / ("before-" + filename), program.snapshot[relative])
        files[relative] = {
            "payload": filename,
            "before_sha256": initial_files[relative]["before_sha256"],
            "after_sha256": _text_hash(content),
        }
    _verify_live(root, initial_files)
    atomic_write_json(journal_path, {
        "schema": "generic_neural_solver.publication.v1",
        "revision_manifest_sha256": file_hash(manifest_path),
        "boundary_sha256": file_hash(boundary_path),
        "files": files,
        "evidence": [boundary_reference, _reference(root, receipt_path)],
    })
    return recover_publication(root, journal_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "advance", "recover-publication", "revise", "repair-completion",
            "recover-completion-repair", "requalify-completion",
            "recover-completion-requalification", "requalify-completion-batch",
            "recover-completion-requalification-batch",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "advance":
            result = advance(args.repo_root, args.record)
        elif args.action == "recover-publication":
            result = recover_publication(args.repo_root, args.record)
        elif args.action == "repair-completion":
            result = repair_completion(args.repo_root, args.record)
        elif args.action == "recover-completion-repair":
            result = recover_completion_repair(args.repo_root, args.record)
        elif args.action == "requalify-completion":
            result = requalify_completion(args.repo_root, args.record)
        elif args.action == "recover-completion-requalification":
            result = recover_completion_requalification(args.repo_root, args.record)
        elif args.action == "requalify-completion-batch":
            result = requalify_completion_batch(args.repo_root, args.record)
        elif args.action == "recover-completion-requalification-batch":
            result = recover_completion_requalification_batch(args.repo_root, args.record)
        else:
            result = revise(args.repo_root, args.record)
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps({"status": "REFUSED", "reason": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
