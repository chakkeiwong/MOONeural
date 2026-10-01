"""Bounded normalization-source requalification preserves completion claims."""

import json

import pytest
from tests.contracts.test_generic_program import write_json
from tests.contracts.test_generic_program_boundary import _completion_requalification_fixture

from mooneural.training.generic_program import (
    DECISIONS_PATH,
    PLAN_PATH,
    STATE_PATH,
    Program,
    ProgramError,
    file_hash,
)
from mooneural.training.generic_program_boundary import requalify_completion

NORMALIZATION_SOURCE = "src/mooneural/training/generic_objective_normalization.py"
NORMALIZATION_STEPS = ("p4.component-audit", "p5r.source")


def _live_bytes(root):
    return {path: (root / path).read_bytes() for path in (PLAN_PATH, STATE_PATH, DECISIONS_PATH)}


def _rebind_replacement(root, manifest_path, report):
    manifest = json.loads(manifest_path.read_text())
    report_path = root / manifest["replacement_report"]["path"]
    write_json(report_path, report)
    report_reference = {
        "path": manifest["replacement_report"]["path"],
        "sha256": file_hash(report_path),
    }
    review_path = root / manifest["replacement_review"]["path"]
    review = json.loads(review_path.read_text())
    review["report_sha256"] = report_reference["sha256"]
    write_json(review_path, review)
    review_reference = {
        "path": manifest["replacement_review"]["path"],
        "sha256": file_hash(review_path),
    }
    for field, replacement in (
        ("replacement_report", report_reference),
        ("replacement_review", review_reference),
    ):
        previous = manifest[field]
        manifest["evidence"] = [
            replacement if reference == previous else reference
            for reference in manifest["evidence"]
        ]
        manifest[field] = replacement
    write_json(manifest_path, manifest)


@pytest.mark.parametrize("step_id", NORMALIZATION_STEPS)
def test_normalization_requalification_preserves_claim_and_replays(tmp_path, step_id):
    manifest_path, source_path, old_reference, replacement_reference = (
        _completion_requalification_fixture(
            tmp_path, step_id=step_id, input_path=NORMALIZATION_SOURCE,
        )
    )
    old_path = tmp_path / old_reference["path"]
    old_bytes = old_path.read_bytes()
    with pytest.raises(ProgramError, match="closure input changed"):
        Program(tmp_path)

    status = requalify_completion(tmp_path, manifest_path)

    assert status["phase_boundary_current"] is True
    program = Program(tmp_path)
    assert program.complete == {step_id}
    assert program.records[step_id]["report"] == replacement_reference
    assert old_path.read_bytes() == old_bytes
    original = json.loads(old_bytes)
    replacement = json.loads((tmp_path / replacement_reference["path"]).read_text())
    assert replacement["inputs"][0]["sha256"] == file_hash(source_path)
    assert {
        key: value for key, value in original.items() if key != "inputs"
    } == {
        key: value for key, value in replacement.items()
        if key not in ("inputs", "requalification")
    }
    completed_bytes = _live_bytes(tmp_path)
    assert requalify_completion(tmp_path, manifest_path)["phase_boundary_current"] is True
    assert _live_bytes(tmp_path) == completed_bytes


@pytest.mark.parametrize("step_id,input_path", (
    ("p4.adapter", NORMALIZATION_SOURCE),
    ("p4.component-audit", "tests/contracts/test_generic_objective_normalization.py"),
    ("p5r.source", "tests/contracts/test_generic_objective_normalization.py"),
    ("p4.component-audit", "src/mooneural/training/ez_nk_live_objective.py"),
    ("p5r.source", "src/mooneural/training/ez_nk_live_objective.py"),
))
def test_normalization_permission_does_not_cover_other_steps_or_files(tmp_path, step_id, input_path):
    manifest_path, _source, _original, _replacement = _completion_requalification_fixture(
        tmp_path, step_id=step_id, input_path=input_path,
    )
    before = _live_bytes(tmp_path)
    with pytest.raises(ProgramError, match="input is not allowed for this step"):
        requalify_completion(tmp_path, manifest_path)
    assert _live_bytes(tmp_path) == before


@pytest.mark.parametrize("step_id", NORMALIZATION_STEPS)
@pytest.mark.parametrize("mutation", ("claim", "added_input", "stale_source"))
def test_normalization_requalification_refuses_unqualified_replacements(tmp_path, step_id, mutation):
    manifest_path, source_path, _original, replacement_reference = (
        _completion_requalification_fixture(
            tmp_path, step_id=step_id, input_path=NORMALIZATION_SOURCE,
        )
    )
    report = json.loads((tmp_path / replacement_reference["path"]).read_text())
    if mutation == "claim":
        report["observations"]["value"] = 2.0
        _rebind_replacement(tmp_path, manifest_path, report)
        expected_error = "changed the scientific report"
    elif mutation == "added_input":
        new_test = tmp_path / "tests/contracts/test_generic_local_envelope.py"
        new_test.parent.mkdir(parents=True)
        new_test.write_text("new regression evidence\n")
        report["inputs"].append({
            "path": str(new_test.relative_to(tmp_path)), "sha256": file_hash(new_test),
        })
        _rebind_replacement(tmp_path, manifest_path, report)
        expected_error = "preserve the complete input list"
    else:
        source_path.write_text("unqualified subsequent implementation\n")
        expected_error = "requalified input is not current"
    before = _live_bytes(tmp_path)
    with pytest.raises(ProgramError, match=expected_error):
        requalify_completion(tmp_path, manifest_path)
    assert _live_bytes(tmp_path) == before
