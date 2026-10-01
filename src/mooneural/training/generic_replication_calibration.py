"""Content-bound source calibration receipts for the existing protocol gate.

Native providers supply separately reviewed completeness evidence. This module
verifies its declared artifacts and unit inventory, reuses the existing source
statistics, and checks a receipt before full training. It never evaluates a
model, qualifies scientific references itself, or advances protocol state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .generic_protocol import GateRecord
from .generic_replication_design import ReplicationDesignBinding
from .generic_replication_statistics import (
    ReplicationBankVectors,
    ReplicationStatisticsSpec,
    SourceCalibrationResult,
    _digest,
    evaluate_source_calibration,
)
from .generic_training_contracts import canonical_json, stable_hash

RECEIPT_SCHEMA = "dsge_hmc.generic_source_calibration_receipt.v1"
QUALIFICATION_SCHEMA = "dsge_hmc.generic_source_calibration_qualification.v1"
BANK_EVIDENCE_SCHEMA = "dsge_hmc.generic_source_calibration_bank_evidence.v1"
QUALIFICATION_CHECKS = (
    "source_endpoint", "target_and_estimator", "backend", "real_provider_completeness",
    "bank_completeness", "seed_role_disjointness", "predeclared_replacements",
)
_REGISTRY_FIELDS = (
    "calibration_bank_ids", "comparison_bank_ids", "excluded_bank_ids",
    "calibration_coordinate_sha256s", "comparison_coordinate_sha256s",
)


class SourceCalibrationGateError(ValueError):
    """Only this bound full-training source condition is rejected."""

    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(f"{status}: {message}")


def _require(condition, message):
    if not condition:
        raise SourceCalibrationGateError("BLOCKED", message)


@dataclass(frozen=True)
class SourceCalibrationBinding:
    """Final expected identity; Phase 5R callers supply source update 114000.

    Provider, estimator and backend digests identify artifact bytes (a manifest
    can bind multiple components). The statistics spec binds the final design's
    logical hash, endpoint/target bytes and complete 20/20 registry identity.
    """

    spec: ReplicationStatisticsSpec
    source_global_update: int
    provider_sha256: str
    estimator_sha256: str
    backend_sha256: str

    def __post_init__(self):
        if not isinstance(self.spec, ReplicationStatisticsSpec):
            raise TypeError("a ReplicationStatisticsSpec is required")
        if type(self.source_global_update) is not int or self.source_global_update < 0:
            raise ValueError("source global update must be a nonnegative integer")
        for name in ("provider_sha256", "estimator_sha256", "backend_sha256"):
            _digest(name, getattr(self, name))

    def to_dict(self):
        return {"spec": self.spec.to_dict(), "spec_sha256": self.spec.binding_hash(),
                "source_global_update": self.source_global_update,
                **{name: getattr(self, name) for name in ("provider_sha256", "estimator_sha256", "backend_sha256")}}

    def binding_hash(self):
        return stable_hash(self.to_dict())


class _EvidenceReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = {}

    def path(self, relative):
        _require(isinstance(relative, str) and bool(relative), "evidence path is missing")
        path = self.root / relative
        _require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                 and path.resolve().is_relative_to(self.root), "evidence must stay inside its root")
        return path

    def verify(self, reference):
        _require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "path/hash evidence required")
        expected = _digest("evidence", reference["sha256"])
        path = self.path(reference["path"])
        if reference["path"] not in self.records:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.records[reference["path"]] = digest.hexdigest()
        _require(self.records[reference["path"]] == expected, f"evidence hash changed: {reference['path']}")
        return path

    def document(self, reference):
        return json.loads(self.verify(reference).read_text())

    def references(self, values):
        _require(isinstance(values, list) and bool(values), "nonempty artifact references required")
        for reference in values:
            self.verify(reference)


def _verify_qualification(reader, binding, qualification, banks):
    spec = binding.spec
    report = reader.document(qualification)
    _require(report.get("schema") == QUALIFICATION_SCHEMA and report.get("status") == "PASS"
             and report.get("evidence_kind") == "actual_source_provider", "actual provider qualification PASS required")
    _require(report.get("binding_sha256") == binding.binding_hash(), "qualification binding mismatch")
    checks = report.get("checks", {})
    _require(set(checks) == set(QUALIFICATION_CHECKS) and all(value is True for value in checks.values()),
             "source/provider/bank qualification checks are incomplete")
    reader.references(report.get("qualification_evidence"))
    components = report.get("components", {})
    _require(set(components) == {"design", "source_endpoint", "accepted_target", "bank_registry", "provider", "estimator", "backend"},
             "qualified component artifact inventory mismatch")
    for reference in components.values():
        reader.verify(reference)
    design = ReplicationDesignBinding.from_dict(reader.document(components["design"]))
    _require(design.binding_hash() == spec.design_sha256, "qualified final design mismatch")
    for name, digest in (
        ("source_endpoint", spec.source_endpoint_sha256), ("accepted_target", spec.accepted_target_sha256),
        ("bank_registry", spec.bank_registry_sha256), ("provider", binding.provider_sha256),
        ("estimator", binding.estimator_sha256), ("backend", binding.backend_sha256),
    ):
        _require(components[name]["sha256"] == digest, f"qualified {name} identity mismatch")
    registry = reader.document(components["bank_registry"])
    for name in _REGISTRY_FIELDS:
        _require(registry.get(name) == list(getattr(spec, name)), f"complete registry mismatch: {name}")
    target = reader.document(components["accepted_target"])
    for path, expected in (*zip(spec.metric_paths, spec.accepted_values, strict=True),
                           *zip(spec.boolean_paths, spec.accepted_booleans, strict=True)):
        value = target
        for key in path:
            value = value[key]
        _require(type(value) is bool if type(expected) is bool else type(value) in (int, float), "target coordinate type mismatch")
        _require(value == expected, "accepted target coordinate mismatch")
    costs = report.get("costs", {})
    _require(set(costs) == {"bank_construction_seconds", "evaluation_seconds", "storage_bytes"}, "measured source costs required")
    for name in ("bank_construction_seconds", "evaluation_seconds"):
        _require(type(costs[name]) in (int, float) and math.isfinite(costs[name]) and costs[name] >= 0, "invalid measured cost")
    _require(type(costs["storage_bytes"]) is int and costs["storage_bytes"] > 0, "invalid retained storage size")
    evaluations = report.get("calibration_banks")
    _require(isinstance(evaluations, list) and len(evaluations) == 20, "20 complete source calibration evaluations required")
    anchors, seeds = set(), set()
    for index, reference in enumerate(evaluations):
        evaluation = reader.document(reference)
        _require(evaluation.get("schema") == BANK_EVIDENCE_SCHEMA and evaluation.get("binding_sha256") == binding.binding_hash(),
                 "source bank evidence binding mismatch")
        _require(evaluation.get("bank_id") == spec.calibration_bank_ids[index]
                 and evaluation.get("coordinate_sha256") == spec.calibration_coordinate_sha256s[index]
                 and evaluation.get("metric_order") == list(spec.metric_order), "source bank coordinate/order mismatch")
        _require(canonical_json(evaluation.get("values")) == canonical_json(banks.to_dict()["values"][index]),
                 "source vectors differ from provider evidence")
        units = evaluation.get("units")
        _require(isinstance(units, list) and len(units) == 20, "20 per-metric anchor units required")
        coordinate_reference = evaluation.get("coordinates")
        coordinates = reader.document(coordinate_reference)
        _require(coordinate_reference["sha256"] == spec.calibration_coordinate_sha256s[index],
                 "calibration coordinate artifact mismatch")
        _require(coordinates == {
            "bank_id": evaluation["bank_id"], "metric_order": list(spec.metric_order),
            "units": [{name: unit[name] for name in ("anchor_id", "seed", "inputs", "targets")} for unit in units],
        }, "source evaluation units differ from frozen input/target coordinates")
        for unit in units:
            anchor, seed = unit.get("anchor_id"), unit.get("seed")
            _require(isinstance(anchor, str) and bool(anchor) and anchor not in anchors, "anchor ID missing or reused")
            _require(type(seed) is int and seed >= 0 and seed not in seeds, "anchor seed missing or reused")
            anchors.add(anchor)
            seeds.add(seed)
            contributions = unit.get("contributions", {})
            _require(set(contributions) == set(spec.metric_order), "complete per-unit metric contributions required")
            _require(all(type(value) in (int, float) or value in ("NaN", "Infinity", "-Infinity")
                         for value in contributions.values()), "numeric per-unit contributions required")
            for name in ("inputs", "targets", "evaluated_tensors"):
                reader.references(unit.get(name))
    return report


@dataclass(frozen=True)
class SourceCalibrationReceipt:
    result: SourceCalibrationResult
    _payload_json: str
    _reference_json: str

    @property
    def status(self):
        return self.to_dict()["status"]

    def to_dict(self):
        return json.loads(self._payload_json)

    def reference(self):
        return json.loads(self._reference_json)


def _payload(binding, qualification, result):
    payload = {
        "schema": RECEIPT_SCHEMA, "binding": binding.to_dict(), "binding_sha256": binding.binding_hash(),
        "qualification": qualification, "source_calibration": result.to_dict(),
        "statistics_source_sha256": hashlib.sha256(Path(evaluate_source_calibration.__code__.co_filename).read_bytes()).hexdigest(),
        "status": "PASS" if result.status == "SOURCE_PROFILE_PASS" else result.status,
        "scope": "source_only_full_training_condition__not_design_or_paired_equivalence",
    }
    return {**payload, "receipt_sha256": stable_hash(payload)}


def _receipt(path, root, payload, result):
    return SourceCalibrationReceipt(result, canonical_json(payload), canonical_json({
        "path": str(path.resolve().relative_to(Path(root).resolve())),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }))


def write_source_calibration_receipt(path, *, root, binding, source_banks, qualification):
    """Verify actual evidence, compute existing statistics, then publish once.

    ``qualification`` is a root-relative path/SHA256 reference to the adapter
    report. Source evaluations retain 20 complete anchor contributions/tensor
    references per bank; paired banks are registry-bound, not evaluated here.
    A byte-identical retry reuses the receipt; changed content never overwrites.
    """
    reader = _EvidenceReader(root)
    path = reader.path(str(path))
    _verify_qualification(reader, binding, qualification, source_banks)
    result = evaluate_source_calibration(binding.spec, source_banks)
    payload = _payload(binding, qualification, result)
    content = (canonical_json(payload) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-calibration-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == content, "refusing to overwrite a different calibration receipt")
    finally:
        temporary.unlink()
    return _receipt(path, root, payload, result)


def load_source_calibration_receipt(reference, *, root, binding):
    """Reload and reverify every declared artifact and exact statistical result."""
    reader = _EvidenceReader(root)
    payload = reader.document(reference)
    _require(payload.get("schema") == RECEIPT_SCHEMA and payload.get("binding") == binding.to_dict()
             and payload.get("binding_sha256") == binding.binding_hash(), "calibration receipt binding mismatch")
    values = dict(payload["source_calibration"]["source_banks"])
    values["values"] = np.asarray(values["values"], dtype=np.float64)
    banks = ReplicationBankVectors(**values)
    _verify_qualification(reader, binding, payload["qualification"], banks)
    result = evaluate_source_calibration(binding.spec, banks)
    _require(payload == _payload(binding, payload["qualification"], result), "calibration receipt or statistics changed")
    return _receipt(reader.path(reference["path"]), root, payload, result)


def require_source_calibration_pass(reference, *, root, binding):
    """Call immediately before full training/resume using its expected binding.

    This returns the original source result for later paired evaluation. It
    does not authorize training independently of the protocol/design/canaries.
    """
    receipt = load_source_calibration_receipt(reference, root=root, binding=binding)
    if receipt.status != "PASS":
        raise SourceCalibrationGateError(receipt.status, "full training requires matching source calibration PASS")
    return receipt.result


def source_calibration_gate(reference, *, root, binding, owner):
    """A verified snapshot for existing protocol manifests/ledgers; no writes."""
    receipt = load_source_calibration_receipt(reference, root=root, binding=binding)
    return GateRecord(
        gate_id="source_calibration_before_full_training", protected_object=f"full training {binding.binding_hash()}",
        lane="Validate", effect="run_block", predicate="matching qualified source-only calibration PASS",
        evidence_paths=(reference["path"], receipt.to_dict()["qualification"]["path"]),
        status={"PASS": "PASS", "NOT_EQUIVALENT": "FAIL", "INCONCLUSIVE": "INCONCLUSIVE"}[receipt.status],
        owner=owner, closure_hash=reference["sha256"],
    )


__all__ = [
    "BANK_EVIDENCE_SCHEMA", "QUALIFICATION_CHECKS", "QUALIFICATION_SCHEMA",
    "SourceCalibrationBinding", "SourceCalibrationGateError", "SourceCalibrationReceipt",
    "load_source_calibration_receipt", "require_source_calibration_pass", "source_calibration_gate",
    "write_source_calibration_receipt",
]
