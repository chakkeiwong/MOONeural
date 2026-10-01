"""Synthetic receipt-contract fixtures, never actual provider qualification."""

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mooneural.training.generic_protocol import GateRecord
from mooneural.training.generic_replication_calibration import (
    BANK_EVIDENCE_SCHEMA,
    QUALIFICATION_CHECKS,
    QUALIFICATION_SCHEMA,
    SourceCalibrationBinding,
    SourceCalibrationGateError,
    load_source_calibration_receipt,
    require_source_calibration_pass,
    source_calibration_gate,
    write_source_calibration_receipt,
)
from mooneural.training.generic_replication_design import ReplicationDesignBinding
from mooneural.training.generic_replication_statistics import (
    ReplicationBankVectors,
    ReplicationStatisticsSpec,
    evaluate_source_calibration,
)
from mooneural.training.generic_training_contracts import canonical_json


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def reference(root, path):
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def document(root, name, payload):
    path = root / name
    path.write_text(canonical_json(payload) + "\n")
    return reference(root, path)


def fixture_case(root, disposition="PASS"):
    metrics = tuple(f"metric_{index:02d}" for index in range(19))
    accepted = [0.02] * 19
    accepted[16] = 0.05371185695178999
    values = np.asarray(accepted) + np.outer(np.arange(20) - 9.5, np.full(19, .0001))
    if disposition == "NOT_EQUIVALENT":
        values[:, 0] += .002
    elif disposition == "BOOLEAN_FAILURE":
        values[:, 0] += .025
    elif disposition == "ZERO_VARIANCE":
        values[:, 0] = accepted[0]
    elif disposition == "NONFINITE":
        values[0, 0] = np.nan
    target = document(root, "target.json", {metric: {"value": value, "passed": value <= .04}
                                           for metric, value in zip(metrics, accepted, strict=True)})
    endpoint = document(root, "endpoint.json", {"synthetic_fixture": True, "global_update": 114000, "policy": [1.5, -2.]})
    components = {name: document(root, f"{name}.json", {"synthetic_fixture": True, "component": name, "version": 3})
                  for name in ("provider", "estimator", "backend")}
    units, coordinates = [], []
    for bank_index in range(20):
        path = root / f"bank-{bank_index}.npz"
        contributions = np.tile(values[bank_index], (20, 1))
        np.savez(path, inputs=np.arange(20, dtype=np.float64) + bank_index * 20,
                 targets=np.asarray(accepted), measured_contributions=contributions)
        tensors = reference(root, path)
        bank_units = [{"anchor_id": f"synthetic-anchor-{bank_index}-{unit_index}", "seed": bank_index * 20 + unit_index,
                       "inputs": [tensors], "targets": [target], "evaluated_tensors": [tensors],
                       "contributions": {metric: float(value) if np.isfinite(value) else "NaN"
                                         for metric, value in zip(metrics, values[bank_index], strict=True)}}
                      for unit_index in range(20)]
        units.append(bank_units)
        coordinates.append(document(root, f"coordinates-{bank_index}.json", {
            "bank_id": f"calibration-{bank_index}", "metric_order": list(metrics),
            "units": [{name: unit[name] for name in ("anchor_id", "seed", "inputs", "targets")} for unit in bank_units],
        }))
    registry_fields = {
        "calibration_bank_ids": tuple(f"calibration-{index}" for index in range(20)),
        "comparison_bank_ids": tuple(f"paired-{index}" for index in range(20)),
        "excluded_bank_ids": ("training", "validation", "selection", "untouched-certification"),
        "calibration_coordinate_sha256s": tuple(item["sha256"] for item in coordinates),
        "comparison_coordinate_sha256s": tuple(digest(f"synthetic-paired-coordinate-{index}") for index in range(20)),
    }
    registry = document(root, "registry.json", {**registry_fields, "predeclared_replacements": "synthetic fixed registry fixture"})
    design = ReplicationDesignBinding(
        design_id="synthetic-calibration-contract", master_sha256=digest("synthetic-master"),
        source_manifest_sha256=digest("synthetic-source-manifest"), generic_code_hashes={"provider": components["provider"]["sha256"]},
        environment_binding={"synthetic_fixture": True}, backend_binding={"artifact": components["backend"]},
        role_manifest_sha256=registry["sha256"], target_hashes={"accepted": target["sha256"]},
        initial_state_hashes={"0": digest("synthetic-parent")}, task_ids=("task",), threshold=.04,
        replica_ids=(0,), expected_selected_replica=0, selection_key=("absolute_threshold_ratio", "replica"),
        stages=({"stage_id": "synthetic", "from_round": 0, "first_round": 1, "last_round": 1,
                 "start_update": 0, "updates_per_round": 1, "population_size": 1, "select_after": False},),
        optimizer_binding={"name": "fixture"}, permanent_pass_binding={"threshold": .04}, seed_registry_sha256=registry["sha256"],
    )
    spec = ReplicationStatisticsSpec(
        design_sha256=design.binding_hash(), source_endpoint_sha256=endpoint["sha256"], accepted_target_sha256=target["sha256"],
        bank_registry_sha256=registry["sha256"], metric_order=metrics,
        metric_paths=tuple((metric, "value") for metric in metrics), boolean_paths=tuple((metric, "passed") for metric in metrics),
        accepted_values=tuple(accepted), accepted_booleans=tuple(value <= .04 for value in accepted), **registry_fields,
    )
    binding = SourceCalibrationBinding(spec, 114000, *(components[name]["sha256"] for name in ("provider", "estimator", "backend")))
    banks = ReplicationBankVectors(spec.binding_hash(), spec.source_endpoint_sha256, metrics, spec.calibration_bank_ids,
                                   spec.calibration_coordinate_sha256s, values)
    evaluations = [document(root, f"evaluation-{index}.json", {
        "schema": BANK_EVIDENCE_SCHEMA, "binding_sha256": binding.binding_hash(), "bank_id": spec.calibration_bank_ids[index],
        "coordinate_sha256": spec.calibration_coordinate_sha256s[index], "coordinates": coordinates[index],
        "metric_order": list(metrics), "values": banks.to_dict()["values"][index], "units": units[index],
    }) for index in range(20)]
    report = {
        "schema": QUALIFICATION_SCHEMA, "status": "PASS", "evidence_kind": "actual_source_provider",
        "binding_sha256": binding.binding_hash(), "checks": dict.fromkeys(QUALIFICATION_CHECKS, True),
        "qualification_evidence": [document(root, "synthetic-review.json", {
            "synthetic_fixture": True, "scope": "exercise report verification; no actual model qualification", "checks": list(QUALIFICATION_CHECKS),
        })],
        "components": {**components, "design": document(root, "design.json", design.to_dict()),
                       "source_endpoint": endpoint, "accepted_target": target, "bank_registry": registry},
        "calibration_banks": evaluations,
        "costs": {"bank_construction_seconds": .1, "evaluation_seconds": .2,
                  "storage_bytes": sum(path.stat().st_size for path in root.iterdir())},
    }
    qualification = document(root, "qualification.json", report)
    return SimpleNamespace(root=root, binding=binding, banks=banks, qualification=qualification, report=report,
                           write=lambda: write_source_calibration_receipt("receipt.json", root=root, binding=binding,
                                                                          source_banks=banks, qualification=qualification))


def revised_qualification(case, report):
    return document(case.root, "revised-qualification.json", report)


def test_roundtrip_training_condition_and_protocol_gate_use_existing_statistics(tmp_path):
    case = fixture_case(tmp_path)
    receipt = case.write()
    original = evaluate_source_calibration(case.binding.spec, case.banks)
    assert receipt.status == "PASS"
    assert receipt.result.to_dict() == original.to_dict()
    np.testing.assert_allclose(receipt.result.to_dict()["sigma_point"], .0001 * math.sqrt(35), rtol=1e-14)
    loaded = load_source_calibration_receipt(receipt.reference(), root=tmp_path, binding=case.binding)
    assert loaded.to_dict() == receipt.to_dict()
    assert require_source_calibration_pass(receipt.reference(), root=tmp_path, binding=case.binding).to_dict() == original.to_dict()
    gate = source_calibration_gate(receipt.reference(), root=tmp_path, binding=case.binding, owner="synthetic-protocol-owner")
    assert GateRecord.from_dict(gate.to_dict()) == gate
    assert gate.effect == "run_block" and gate.status == "PASS"
    assert gate.closure_hash == receipt.reference()["sha256"]
    assert receipt.to_dict()["scope"] == "source_only_full_training_condition__not_design_or_paired_equivalence"


@pytest.mark.parametrize("disposition,status", [("NOT_EQUIVALENT", "NOT_EQUIVALENT"), ("BOOLEAN_FAILURE", "NOT_EQUIVALENT"),
                                               ("ZERO_VARIANCE", "INCONCLUSIVE"), ("NONFINITE", "INCONCLUSIVE")])
def test_negative_and_undefined_statistics_persist_and_block_full_training(tmp_path, disposition, status):
    case = fixture_case(tmp_path, disposition)
    receipt = case.write()
    loaded = load_source_calibration_receipt(receipt.reference(), root=tmp_path, binding=case.binding)
    assert loaded.status == status
    assert loaded.result.to_dict() == evaluate_source_calibration(case.binding.spec, case.banks).to_dict()
    with pytest.raises(SourceCalibrationGateError) as error:
        require_source_calibration_pass(receipt.reference(), root=tmp_path, binding=case.binding)
    assert error.value.status == status
    gate = source_calibration_gate(receipt.reference(), root=tmp_path, binding=case.binding, owner="synthetic-owner")
    assert gate.status == ("FAIL" if status == "NOT_EQUIVALENT" else "INCONCLUSIVE")


@pytest.mark.parametrize("field", ["source_global_update", "provider_sha256", "estimator_sha256", "backend_sha256",
                                   "design_sha256", "source_endpoint_sha256", "accepted_target_sha256", "bank_registry_sha256",
                                   "comparison_coordinate_sha256s"])
def test_receipt_requires_current_expected_identity(tmp_path, field):
    case = fixture_case(tmp_path)
    receipt = case.write()
    if field in ("provider_sha256", "estimator_sha256", "backend_sha256"):
        binding = replace(case.binding, **{field: digest("different")})
    elif field == "source_global_update":
        binding = replace(case.binding, source_global_update=99000)
    else:
        value = tuple(digest(f"new-paired-{index}") for index in range(20)) if field == "comparison_coordinate_sha256s" else digest("different")
        binding = replace(case.binding, spec=replace(case.binding.spec, **{field: value}))
    with pytest.raises(SourceCalibrationGateError, match="binding mismatch"):
        require_source_calibration_pass(receipt.reference(), root=tmp_path, binding=binding)


@pytest.mark.parametrize("field,value", [("status", "PENDING"), ("evidence_kind", "synthetic_host_fixture"),
                                        ("checks", {name: True for name in QUALIFICATION_CHECKS[:-1]}),
                                        ("qualification_evidence", []), ("components", {"provider": "native-provider"}),
                                        ("costs", {"bank_construction_seconds": 0, "evaluation_seconds": 0, "storage_bytes": True})])
def test_unqualified_or_name_only_provider_cannot_issue_receipt(tmp_path, field, value):
    case = fixture_case(tmp_path)
    report = copy.deepcopy(case.report)
    report[field] = value
    with pytest.raises(SourceCalibrationGateError):
        write_source_calibration_receipt("receipt.json", root=tmp_path, binding=case.binding, source_banks=case.banks,
                                         qualification=revised_qualification(case, report))
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("defect", ["missing_bank", "reordered_banks", "missing_unit", "missing_metric", "vector_change",
                                   "no_tensors", "coordinate_change", "reused_anchor", "float_seed", "boolean_seed"])
def test_complete_measured_bank_evidence_is_required(tmp_path, defect):
    case = fixture_case(tmp_path)
    report = copy.deepcopy(case.report)
    if defect == "missing_bank":
        report["calibration_banks"].pop()
    elif defect == "reordered_banks":
        report["calibration_banks"].reverse()
    else:
        evaluation = json.loads((tmp_path / report["calibration_banks"][0]["path"]).read_text())
        if defect == "missing_unit":
            evaluation["units"].pop()
        elif defect == "missing_metric":
            evaluation["units"][0]["contributions"].pop(case.binding.spec.metric_order[0])
        elif defect == "vector_change":
            evaluation["values"][0] += .001
        elif defect == "no_tensors":
            evaluation["units"][0]["evaluated_tensors"] = []
        elif defect == "coordinate_change":
            evaluation["coordinates"] = evaluation["units"][0]["inputs"][0]
        elif defect == "reused_anchor":
            evaluation["units"][1]["anchor_id"] = evaluation["units"][0]["anchor_id"]
        else:
            evaluation["units"][0]["seed"] = 0.0 if defect == "float_seed" else False
        report["calibration_banks"][0] = document(tmp_path, "changed-evaluation.json", evaluation)
    with pytest.raises((SourceCalibrationGateError, ValueError, UnicodeError)):
        write_source_calibration_receipt("receipt.json", root=tmp_path, binding=case.binding, source_banks=case.banks,
                                         qualification=revised_qualification(case, report))


@pytest.mark.parametrize("name", ["provider.json", "endpoint.json", "target.json", "registry.json", "design.json",
                                  "bank-0.npz", "coordinates-0.json", "evaluation-0.json", "synthetic-review.json"])
def test_changed_artifact_invalidates_previously_passing_receipt(tmp_path, name):
    case = fixture_case(tmp_path)
    receipt = case.write()
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(SourceCalibrationGateError, match="evidence hash changed"):
        require_source_calibration_pass(receipt.reference(), root=tmp_path, binding=case.binding)


def test_missing_tensor_artifact_refuses_training(tmp_path):
    case = fixture_case(tmp_path)
    receipt = case.write()
    (tmp_path / "bank-0.npz").unlink()
    with pytest.raises(FileNotFoundError):
        require_source_calibration_pass(receipt.reference(), root=tmp_path, binding=case.binding)


@pytest.mark.parametrize("field", ["status", "statistics", "vectors", "extra"])
def test_receipt_recomputation_rejects_tampering_even_with_new_file_digest(tmp_path, field):
    case = fixture_case(tmp_path)
    receipt = case.write()
    payload = receipt.to_dict()
    if field == "status":
        payload["status"] = "NOT_EQUIVALENT"
    elif field == "statistics":
        payload["source_calibration"]["sigma_gate"][0] *= 100
    elif field == "vectors":
        payload["source_calibration"]["source_banks"]["values"][0][0] += .001
    else:
        payload["full_training_allowed"] = True
    altered = document(tmp_path, "altered-receipt.json", payload)
    with pytest.raises(SourceCalibrationGateError):
        require_source_calibration_pass(altered, root=tmp_path, binding=case.binding)


def test_identical_retry_is_immutable_and_changed_qualification_requires_new_receipt(tmp_path):
    case = fixture_case(tmp_path)
    receipt = case.write()
    before = (tmp_path / "receipt.json").read_bytes()
    assert case.write().reference() == receipt.reference()
    with pytest.raises(SourceCalibrationGateError, match="overwrite"):
        write_source_calibration_receipt("receipt.json", root=tmp_path, binding=case.binding, source_banks=case.banks,
                                         qualification=revised_qualification(case, case.report))
    assert (tmp_path / "receipt.json").read_bytes() == before
    altered = receipt.to_dict()
    altered["source_calibration"]["U_source_current"][0] = 12
    assert receipt.to_dict()["source_calibration"]["U_source_current"][0] != 12


@pytest.mark.parametrize("update", [114000.0, True, -1])
def test_binding_update_is_a_strict_nonnegative_integer(tmp_path, update):
    case = fixture_case(tmp_path)
    with pytest.raises(ValueError, match="integer"):
        replace(case.binding, source_global_update=update)


def test_module_import_has_no_native_or_active_role_source_imports():
    code = (
        "import sys; import mooneural.training.generic_replication_calibration; "
        "assert 'tensorflow' not in sys.modules; "
        "assert not any(name.startswith('mooneural.training.rotemberg_') for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=30,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
