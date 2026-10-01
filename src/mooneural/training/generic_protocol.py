"""Executable phase protocol for the generic neural-solver program.

This module owns the lifecycle and evidence boundary.  It intentionally does
not know anything about a model, objective equations, an optimizer, or a
particular numerical backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mooneural.artifacts import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    safe_load_json,
)

from .generic_program import persist_pending_boundary, require_program_attempt
from .generic_training_contracts import (
    CONTRACT_SCHEMA_VERSION,
    GOVERNANCE_LANES,
    CheckpointRecord,
    CheckpointState,
    ContractSchemaError,
    ResultRecord,
    RoleManifest,
    _bool_map,
    _nonempty,
    _schema,
    _sha256,
    _strict_bool,
    _strict_int,
    _unique_strings,
    _verify_hash,
    canonical_json,
    stable_hash,
)

PROTOCOL_SCHEMA_VERSION = "dsge_hmc.generic_neural_solver_protocol.v2"
STATES = (
    "PLANNED",
    "PREFLIGHT_PASSED",
    "RUNNING",
    "SEALED",
    "AUDIT_PENDING",
    "AUDIT_PASSED",
    "AUDIT_FAILED",
    "BLOCKED",
    "CLOSED",
)
TERMINAL_STATES = frozenset({"AUDIT_FAILED", "BLOCKED", "CLOSED"})
ALLOWED_TRANSITIONS = {
    "PLANNED": frozenset({"PREFLIGHT_PASSED"}),
    "PREFLIGHT_PASSED": frozenset({"RUNNING"}),
    "RUNNING": frozenset({"SEALED", "BLOCKED"}),
    "SEALED": frozenset({"AUDIT_PENDING", "BLOCKED"}),
    "AUDIT_PENDING": frozenset({"AUDIT_PASSED", "AUDIT_FAILED", "BLOCKED"}),
    "AUDIT_PASSED": frozenset({"CLOSED"}),
    "AUDIT_FAILED": frozenset(),
    "BLOCKED": frozenset(),
    "CLOSED": frozenset(),
}
GATE_EFFECTS = (
    "diagnostic",
    "candidate_rejection",
    "run_block",
    "promotion_block",
)
GATE_STATUSES = ("OPEN", "PASS", "FAIL", "BLOCKED", "INCONCLUSIVE")
AUDIT_DECISIONS = ("PASS", "FAIL", "BLOCKED", "INCONCLUSIVE")
CLOSE_DECISIONS = ("PASS", "PROMOTE", "FAIL", "BLOCKED", "INCONCLUSIVE")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strings(
    name: str, values: Sequence[str], *, required: bool = True
) -> tuple[str, ...]:
    normalized = _unique_strings(name, values)
    if required and not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique values")
    return normalized


def _hash_map(name: str, values: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = {
        _nonempty(f"{name} key", key): _sha256(f"{name} value", value)
        for key, value in values.items()
    }
    return normalized


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("protocol metadata must be a mapping")
    return dict(json.loads(canonical_json(value)))


def _absolute_path(name: str, value: str | Path) -> str:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return str(path.resolve())


def _protocol_payload(payload, suffix, hash_field):
    data = _schema(payload, f"{PROTOCOL_SCHEMA_VERSION}.{suffix}")
    if data.get("contract_schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractSchemaError("unknown contract schema version")
    _verify_hash(suffix, data, hash_field)
    envelopes = {
        "phase_state": {
            "status",
            "updated_at",
            "attempt_hash",
            "transition_count",
            "last_transition_hash",
        },
        "transition": {
            "sequence",
            "previous_transition_hash",
            "previous_state",
            "next_state",
            "event",
            "created_at",
            "metadata",
            "attempt",
        },
        "experiment_binding": set(),
        "completion_marker": {"complete", "checkpoint_record", "policy_fingerprint"},
        "sealed_result": {
            "status",
            "result",
            "checkpoint_records",
            "counters",
            "command_log",
            "manifests",
            "manifest_hashes",
            "experiment_hash_manifest",
            "sealed_at",
        },
    }
    if suffix in envelopes:
        expected = {
            "schema",
            "contract_schema_version",
            hash_field,
            "attempt_id",
            "plan_hash",
            "phase_id",
            "artifact_root",
            "experiment_root",
        } | envelopes[suffix]
        if set(data) != expected:
            raise ContractSchemaError(f"{suffix} fields do not match schema")
    return data


def _deserialize(cls, payload, suffix, hash_field):
    data = _protocol_payload(payload, suffix, hash_field)
    for name in ("schema", "contract_schema_version", hash_field):
        del data[name]
    if set(data) != {item.name for item in fields(cls)}:
        raise ContractSchemaError(f"{suffix} fields do not match schema")
    return cls(**data)


def _binding(attempt):
    return {
        name: getattr(attempt, name)
        for name in (
            "attempt_id",
            "plan_hash",
            "phase_id",
            "artifact_root",
            "experiment_root",
        )
    }


def _check_binding(payload, attempt):
    for name, expected in _binding(attempt).items():
        if payload.get(name) != expected:
            raise ValueError(f"packet binding mismatch: {name}")


@dataclass(frozen=True)
class ProtocolAttempt:
    """Versioned identity and state for one immutable phase attempt."""

    attempt_id: str
    program_id: str
    phase_id: str
    lane: str
    plan_hash: str
    executor_id: str
    auditor_id: str
    decision_owner: str
    question: str
    protected_claim: str
    comparator: str
    primary_criterion: str
    vetoes: tuple[str, ...]
    nonclaims: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    source_hashes: Mapping[str, str]
    input_hashes: Mapping[str, str]
    role_manifest_hash: str
    artifact_root: str
    experiment_root: str
    evidence_contract: Mapping[str, Any] = field(default_factory=dict)
    baseline: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    status: str = "PLANNED"
    created_at: str = field(default_factory=_utc_now)
    sealed_at: str | None = None
    decision: str | None = None
    permitted_next_action: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "attempt_id",
            "program_id",
            "phase_id",
            "executor_id",
            "decision_owner",
            "question",
            "protected_claim",
            "comparator",
            "primary_criterion",
            "artifact_root",
            "experiment_root",
            "created_at",
        ):
            value = _nonempty(name, getattr(self, name))
            if name in {"artifact_root", "experiment_root"}:
                value = _absolute_path(name, value)
            object.__setattr__(self, name, value)
        if any(
            Path(getattr(self, name)).name != self.attempt_id
            for name in ("artifact_root", "experiment_root")
        ):
            raise ValueError(
                "artifact and experiment roots must be bound to attempt_id"
            )
        if self.artifact_root == self.experiment_root:
            raise ValueError("artifact and experiment roots must be distinct")
        if self.lane not in GOVERNANCE_LANES:
            raise ValueError(f"unknown governance lane: {self.lane}")
        if self.status not in STATES:
            raise ValueError(f"unknown protocol state: {self.status}")
        object.__setattr__(self, "plan_hash", _sha256("plan_hash", self.plan_hash))
        object.__setattr__(
            self,
            "role_manifest_hash",
            _sha256("role_manifest_hash", self.role_manifest_hash),
        )
        object.__setattr__(self, "vetoes", _strings("vetoes", self.vetoes))
        object.__setattr__(self, "nonclaims", _strings("nonclaims", self.nonclaims))
        object.__setattr__(
            self, "stop_conditions", _strings("stop_conditions", self.stop_conditions)
        )
        object.__setattr__(
            self, "source_hashes", _hash_map("source_hashes", self.source_hashes)
        )
        object.__setattr__(
            self, "input_hashes", _hash_map("input_hashes", self.input_hashes)
        )
        object.__setattr__(self, "evidence_contract", _mapping(self.evidence_contract))
        object.__setattr__(self, "baseline", _mapping(self.baseline))
        object.__setattr__(self, "resources", _mapping(self.resources))
        if self.sealed_at is not None:
            object.__setattr__(
                self, "sealed_at", _nonempty("sealed_at", self.sealed_at)
            )
        if self.decision is not None:
            object.__setattr__(self, "decision", _nonempty("decision", self.decision))
        if self.permitted_next_action is not None:
            object.__setattr__(
                self,
                "permitted_next_action",
                _nonempty("permitted_next_action", self.permitted_next_action),
            )
        if self.lane in ("Validate", "Admit"):
            auditor = _nonempty("auditor_id", self.auditor_id)
            if auditor == self.executor_id:
                raise ValueError(
                    "Validate and Admit require distinct executor and auditor identities"
                )
            object.__setattr__(self, "auditor_id", auditor)
        elif not isinstance(self.auditor_id, str):
            raise TypeError("auditor_id must be a string")
        if self.status == "CLOSED" and (
            self.decision not in CLOSE_DECISIONS or self.permitted_next_action is None
        ):
            raise ValueError(
                "closed attempt requires a known decision and one next action"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": f"{PROTOCOL_SCHEMA_VERSION}.attempt",
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "program_id": self.program_id,
            "phase_id": self.phase_id,
            "lane": self.lane,
            "plan_hash": self.plan_hash,
            "executor_id": self.executor_id,
            "auditor_id": self.auditor_id,
            "decision_owner": self.decision_owner,
            "question": self.question,
            "protected_claim": self.protected_claim,
            "comparator": self.comparator,
            "primary_criterion": self.primary_criterion,
            "vetoes": list(self.vetoes),
            "nonclaims": list(self.nonclaims),
            "stop_conditions": list(self.stop_conditions),
            "source_hashes": dict(self.source_hashes),
            "input_hashes": dict(self.input_hashes),
            "role_manifest_hash": self.role_manifest_hash,
            "artifact_root": self.artifact_root,
            "experiment_root": self.experiment_root,
            "evidence_contract": dict(self.evidence_contract),
            "baseline": dict(self.baseline),
            "resources": dict(self.resources),
            "status": self.status,
            "created_at": self.created_at,
            "sealed_at": self.sealed_at,
            "decision": self.decision,
            "permitted_next_action": self.permitted_next_action,
        }
        payload["attempt_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProtocolAttempt:
        return _deserialize(cls, payload, "attempt", "attempt_hash")


@dataclass(frozen=True)
class GateRecord:
    """One machine-readable gate and its declared effect."""

    gate_id: str
    protected_object: str
    lane: str
    effect: str
    predicate: str
    evidence_paths: tuple[str, ...]
    status: str
    owner: str
    closure_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("gate_id", "protected_object", "predicate", "owner"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.lane not in GOVERNANCE_LANES:
            raise ValueError(f"unknown gate lane: {self.lane}")
        if self.effect not in GATE_EFFECTS:
            raise ValueError(f"unknown gate effect: {self.effect}")
        if self.status not in GATE_STATUSES:
            raise ValueError(f"unknown gate status: {self.status}")
        object.__setattr__(
            self, "evidence_paths", _strings("evidence_paths", self.evidence_paths)
        )
        if (
            self.status in {"PASS", "FAIL", "BLOCKED", "INCONCLUSIVE"}
            and self.closure_hash is None
        ):
            raise ValueError("closed gate statuses require a closure hash")
        if self.closure_hash is not None:
            object.__setattr__(
                self, "closure_hash", _sha256("closure_hash", self.closure_hash)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": f"{PROTOCOL_SCHEMA_VERSION}.gate_record",
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "gate_id": self.gate_id,
            "protected_object": self.protected_object,
            "lane": self.lane,
            "effect": self.effect,
            "predicate": self.predicate,
            "evidence_paths": list(self.evidence_paths),
            "status": self.status,
            "owner": self.owner,
            "closure_hash": self.closure_hash,
        }
        payload["gate_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateRecord:
        return _deserialize(cls, payload, "gate_record", "gate_hash")


class GateLedger:
    """Small explicit ledger for gate effects and closure predicates."""

    def __init__(self, records: Sequence[GateRecord] = ()) -> None:
        self._records: dict[str, GateRecord] = {}
        for record in records:
            self.add(record)

    @property
    def records(self) -> tuple[GateRecord, ...]:
        return tuple(self._records.values())

    def add(self, record: GateRecord) -> None:
        if not isinstance(record, GateRecord):
            raise TypeError("gate ledger requires GateRecord instances")
        if record.gate_id in self._records:
            raise ValueError(f"duplicate gate ID: {record.gate_id}")
        self._records[record.gate_id] = record

    def update(self, record: GateRecord) -> None:
        if not isinstance(record, GateRecord):
            raise TypeError("gate ledger requires GateRecord instances")
        if record.gate_id not in self._records:
            raise KeyError(record.gate_id)
        self._records[record.gate_id] = record

    def validate_close(self, decision: str, *, promotion_claim: bool = False) -> None:
        normalized = _nonempty("decision", decision)
        _strict_bool("promotion_claim", promotion_claim)
        if normalized not in CLOSE_DECISIONS:
            raise ValueError("unknown closure decision")
        if not self._records:
            raise ValueError("a GateLedger is required for phase closure")
        blocking = [
            record
            for record in self._records.values()
            if record.effect in {"run_block", "promotion_block"}
            and record.status != "PASS"
        ]
        if normalized in {"PASS", "PROMOTE"} and blocking:
            raise ValueError(
                "blocking gates are not passed: "
                + ", ".join(record.gate_id for record in blocking)
            )
        if promotion_claim and any(
            record.effect == "promotion_block" and record.status != "PASS"
            for record in self._records.values()
        ):
            raise ValueError("promotion claim requires all promotion gates to pass")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": f"{PROTOCOL_SCHEMA_VERSION}.gate_ledger",
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "records": [record.to_dict() for record in self.records],
        }
        payload["ledger_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateLedger:
        data = _protocol_payload(payload, "gate_ledger", "ledger_hash")
        if set(data) != {"schema", "contract_schema_version", "records", "ledger_hash"}:
            raise ContractSchemaError("gate ledger fields do not match schema")
        return cls(tuple(GateRecord.from_dict(item) for item in data["records"]))


@dataclass(frozen=True)
class AuditRecord:
    """Independent audit result for one sealed attempt packet."""

    attempt_id: str
    auditor_id: str
    checklist_version: str
    artifact_hashes_verified: bool
    independent_checks: Mapping[str, bool]
    findings: tuple[str, ...]
    veto_status: str
    uncertainty: Mapping[str, Any]
    decision: str
    required_repairs: tuple[str, ...] = ()
    re_audit_of: str | None = None
    audited_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _nonempty("attempt_id", self.attempt_id))
        object.__setattr__(self, "auditor_id", _nonempty("auditor_id", self.auditor_id))
        object.__setattr__(
            self,
            "checklist_version",
            _nonempty("checklist_version", self.checklist_version),
        )
        _strict_bool("artifact_hashes_verified", self.artifact_hashes_verified)
        checks = _bool_map("independent_checks", self.independent_checks)
        if not checks:
            raise ValueError("independent_checks must not be empty")
        object.__setattr__(self, "independent_checks", checks)
        object.__setattr__(
            self, "findings", _strings("findings", self.findings, required=False)
        )
        object.__setattr__(
            self,
            "required_repairs",
            _strings("required_repairs", self.required_repairs, required=False),
        )
        object.__setattr__(self, "uncertainty", _mapping(self.uncertainty))
        object.__setattr__(
            self, "veto_status", _nonempty("veto_status", self.veto_status)
        )
        decision = _nonempty("decision", self.decision)
        if decision not in AUDIT_DECISIONS:
            raise ValueError(f"unknown audit decision: {decision}")
        if self.veto_status not in AUDIT_DECISIONS:
            raise ValueError("unknown veto status")
        if decision == "PASS" and (
            not self.artifact_hashes_verified
            or not all(checks.values())
            or self.veto_status != "PASS"
            or self.required_repairs
        ):
            raise ValueError(
                "a passing audit requires verified hashes, all checks, no veto or repairs"
            )
        _nonempty("audited_at", self.audited_at)
        object.__setattr__(self, "decision", decision)
        if self.re_audit_of is not None:
            object.__setattr__(
                self, "re_audit_of", _nonempty("re_audit_of", self.re_audit_of)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": f"{PROTOCOL_SCHEMA_VERSION}.audit_record",
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "auditor_id": self.auditor_id,
            "checklist_version": self.checklist_version,
            "artifact_hashes_verified": self.artifact_hashes_verified,
            "independent_checks": dict(self.independent_checks),
            "findings": list(self.findings),
            "veto_status": self.veto_status,
            "uncertainty": dict(self.uncertainty),
            "decision": self.decision,
            "required_repairs": list(self.required_repairs),
            "re_audit_of": self.re_audit_of,
            "audited_at": self.audited_at,
        }
        payload["audit_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuditRecord:
        return _deserialize(cls, payload, "audit_record", "audit_hash")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_files(root: Path) -> tuple[Path, ...]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("artifact payload must not contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() == "hashes.txt":
            continue
        if relative.parts and relative.parts[0] == "protocol":
            continue
        files.append(path)
    return tuple(sorted(files, key=lambda item: str(item.relative_to(root))))


def write_hash_manifest(root: str | Path) -> Path:
    """Hash immutable payload files, excluding mutable protocol records."""

    artifact_root = Path(_absolute_path("artifact root", root))
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(artifact_root).as_posix()}"
        for path in _payload_files(artifact_root)
    ]
    if not lines:
        raise ValueError("cannot seal an artifact root without payload files")
    return atomic_write_text(artifact_root / "hashes.txt", "\n".join(lines) + "\n")


def verify_hash_manifest(root: str | Path) -> dict[str, Any]:
    """Verify the complete immutable payload set against ``hashes.txt``."""

    artifact_root = Path(_absolute_path("artifact root", root))
    manifest = artifact_root / "hashes.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if separator != "  " or not relative or relative in expected:
            raise ValueError("invalid or duplicate hash manifest line")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("hash manifest paths must be relative")
        candidate = (artifact_root / relative).resolve()
        if artifact_root not in candidate.parents:
            raise ValueError("hash manifest path escapes artifact root")
        if not candidate.is_file():
            raise ValueError(f"hashed artifact is missing: {relative}")
        expected[relative] = _sha256("artifact digest", digest)
    actual_files = {
        path.relative_to(artifact_root).as_posix(): _sha256_file(path)
        for path in _payload_files(artifact_root)
    }
    if set(expected) != set(actual_files):
        raise ValueError(
            "hash manifest file set mismatch: "
            f"missing={sorted(set(actual_files) - set(expected))} "
            f"extra={sorted(set(expected) - set(actual_files))}"
        )
    mismatches = sorted(
        relative
        for relative, digest in expected.items()
        if actual_files[relative] != digest
    )
    if mismatches:
        raise ValueError(f"artifact hash mismatch: {mismatches}")
    return {
        "verified": True,
        "file_count": len(actual_files),
        "files": sorted(actual_files),
    }


def _relative_file(root: Path, relative: str) -> Path:
    _nonempty("artifact path", relative)
    candidate = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or root not in candidate.resolve().parents
    ):
        raise ValueError("artifact path must stay within its root")
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"artifact missing or not a regular file: {relative}")
    return candidate


def validate_checkpoint(
    attempt: ProtocolAttempt, record: CheckpointRecord
) -> CheckpointState:
    """Verify a durable completion marker and its typed checkpoint binding."""
    record = CheckpointRecord.from_dict(record.to_dict())
    if record.attempt_id != attempt.attempt_id:
        raise ValueError("checkpoint attempt mismatch")
    root = Path(attempt.artifact_root)
    checkpoint = _relative_file(root, record.checkpoint_path)
    if _sha256_file(checkpoint) != record.checkpoint_hash:
        raise ValueError("checkpoint file hash mismatch")
    state = CheckpointState.from_dict(safe_load_json(checkpoint))
    if (
        state.attempt_id != attempt.attempt_id
        or state.update_index != record.update_index
    ):
        raise ValueError("checkpoint state binding mismatch")
    marker = _protocol_payload(
        safe_load_json(_relative_file(root, record.completion_marker)),
        "completion_marker",
        "marker_hash",
    )
    _check_binding(marker, attempt)
    if _strict_bool("complete", marker["complete"]) is not True:
        raise ValueError("checkpoint is incomplete")
    if (
        CheckpointRecord.from_dict(marker["checkpoint_record"]).to_dict()
        != record.to_dict()
    ):
        raise ValueError("completion marker checkpoint binding mismatch")
    if marker.get("policy_fingerprint") != state.policy_fingerprint:
        raise ValueError("completion marker fingerprint mismatch")
    return state


def _durable_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    canonical_json(payload)
    atomic_write_json(path, payload)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_checkpoint(
    attempt: ProtocolAttempt, state: CheckpointState
) -> CheckpointRecord:
    """Commit checkpoint first, then atomically publish its completion marker."""
    if attempt.status != "RUNNING" or state.attempt_id != attempt.attempt_id:
        raise ValueError("checkpoint requires a matching RUNNING attempt")
    persisted = ProtocolController.load(
        attempt.artifact_root, attempt.experiment_root
    ).attempt
    if persisted.to_dict() != attempt.to_dict():
        raise ValueError("stale checkpoint attempt; no writes permitted")
    state = CheckpointState.from_dict(state.to_dict())
    checkpoint_id = f"update-{state.update_index:08d}"
    root = Path(attempt.artifact_root)
    checkpoint_path = f"checkpoints/{checkpoint_id}.json"
    completion_marker = f"recovery/{checkpoint_id}.complete.json"
    if (root / checkpoint_path).exists() or (root / completion_marker).exists():
        raise FileExistsError("checkpoint boundaries cannot be overwritten")
    _durable_write_json(root / checkpoint_path, state.to_dict())
    record = CheckpointRecord(
        attempt.attempt_id,
        checkpoint_id,
        checkpoint_path,
        _sha256_file(root / checkpoint_path),
        state.update_index,
        completion_marker,
    )
    marker = {
        "schema": f"{PROTOCOL_SCHEMA_VERSION}.completion_marker",
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        **_binding(attempt),
        "complete": True,
        "checkpoint_record": record.to_dict(),
        "policy_fingerprint": state.policy_fingerprint,
    }
    marker["marker_hash"] = stable_hash(marker)
    _durable_write_json(root / completion_marker, marker)
    validate_checkpoint(attempt, record)
    return record


class ProtocolController:
    """Create and advance one generated, immutable phase attempt."""

    def __init__(
        self,
        *,
        program_id: str,
        phase_id: str,
        lane: str,
        plan_hash: str,
        executor_id: str,
        auditor_id: str,
        decision_owner: str = "user",
        artifact_base: str | Path,
        experiment_base: str | Path | None = None,
    ) -> None:
        if lane not in GOVERNANCE_LANES:
            raise ValueError(f"unknown governance lane: {lane}")
        self.program_id = _nonempty("program_id", program_id)
        self.phase_id = _nonempty("phase_id", phase_id)
        self.lane = lane
        self.plan_hash = _sha256("plan_hash", plan_hash)
        self.executor_id = _nonempty("executor_id", executor_id)
        if not isinstance(auditor_id, str):
            raise TypeError("auditor_id must be a string")
        self.auditor_id = auditor_id
        self.decision_owner = _nonempty("decision_owner", decision_owner)
        if lane in ("Validate", "Admit") and (
            not self.auditor_id or self.auditor_id == self.executor_id
        ):
            raise ValueError("Validate and Admit require a distinct auditor identity")
        self.artifact_base = Path(_absolute_path("artifact_base", artifact_base))
        self.experiment_base = (
            self.artifact_base.parent
            if experiment_base is None
            else Path(_absolute_path("experiment_base", experiment_base))
        )
        if self.artifact_base == self.experiment_base:
            raise ValueError("artifact and experiment bases must be distinct")
        self._attempt: ProtocolAttempt | None = None
        self._artifact_root: Path | None = None
        self._experiment_root: Path | None = None
        self._transition_count = 0
        self._transition_hash: str | None = None

    @property
    def attempt(self) -> ProtocolAttempt:
        if self._attempt is None:
            raise RuntimeError("the protocol attempt has not been created")
        return self._attempt

    @property
    def artifact_root(self) -> Path:
        if self._artifact_root is None:
            raise RuntimeError("the protocol attempt has not been created")
        return self._artifact_root

    @property
    def experiment_root(self) -> Path:
        if self._experiment_root is None:
            raise RuntimeError("the protocol attempt has not been created")
        return self._experiment_root

    def create_attempt(
        self,
        *,
        question: str,
        protected_claim: str,
        comparator: str,
        primary_criterion: str,
        vetoes: Sequence[str],
        nonclaims: Sequence[str],
        stop_conditions: Sequence[str],
        source_hashes: Mapping[str, str],
        input_hashes: Mapping[str, str],
        role_manifest_hash: str,
        evidence_contract: Mapping[str, Any],
        baseline: Mapping[str, Any],
        resources: Mapping[str, Any],
    ) -> ProtocolAttempt:
        if self._attempt is not None:
            raise RuntimeError("an attempt already exists for this controller")
        require_program_attempt(
            self.program_id, self.phase_id, self.lane, self.plan_hash, evidence_contract
        )
        self.artifact_base.mkdir(parents=True, exist_ok=True)
        self.experiment_base.mkdir(parents=True, exist_ok=True)
        for _ in range(10):
            attempt_id = (
                f"gns-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{secrets.token_hex(5)}"
            )
            artifact_root = self.artifact_base / attempt_id
            experiment_root = self.experiment_base / attempt_id
            if not artifact_root.exists() and not experiment_root.exists():
                break
        else:
            raise RuntimeError("could not allocate a unique attempt namespace")
        self._artifact_root = artifact_root
        self._experiment_root = experiment_root
        attempt = ProtocolAttempt(
            attempt_id=attempt_id,
            program_id=self.program_id,
            phase_id=self.phase_id,
            lane=self.lane,
            plan_hash=self.plan_hash,
            executor_id=self.executor_id,
            auditor_id=self.auditor_id,
            decision_owner=self.decision_owner,
            question=question,
            protected_claim=protected_claim,
            comparator=comparator,
            primary_criterion=primary_criterion,
            vetoes=vetoes,
            nonclaims=nonclaims,
            stop_conditions=stop_conditions,
            source_hashes=source_hashes,
            input_hashes=input_hashes,
            role_manifest_hash=role_manifest_hash,
            artifact_root=str(artifact_root),
            experiment_root=str(experiment_root),
            evidence_contract=evidence_contract,
            baseline=baseline,
            resources=resources,
        )
        self._attempt = attempt
        artifact_root.mkdir(parents=True)
        (artifact_root / "protocol").mkdir()
        (artifact_root / "checkpoints").mkdir()
        (artifact_root / "recovery").mkdir()
        experiment_root.mkdir(parents=True)
        self._write_experiment_stub()
        self._append_transition(None, "PLANNED", "attempt_created", {})
        self._write_protocol_records()
        return attempt

    def _write_experiment_stub(self) -> None:
        text = (
            f"# {self.phase_id}\n\n"
            f"- attempt_id: `{self.attempt.attempt_id}`\n"
            f"- program_id: `{self.attempt.program_id}`\n"
            f"- governance lane: `{self.attempt.lane}`\n"
            f"- plan hash: `{self.attempt.plan_hash}`\n"
            f"- status: `{self.attempt.status}`\n\n"
            "This packet is generated by the Phase 1 protocol controller.\n"
        )
        atomic_write_text(self.experiment_root / "experiment.md", text)
        binding = {
            "schema": f"{PROTOCOL_SCHEMA_VERSION}.experiment_binding",
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            **_binding(self.attempt),
        }
        binding["binding_hash"] = stable_hash(binding)
        atomic_write_json(self.experiment_root / "attempt-binding.json", binding)

    def _write_protocol_records(self) -> None:
        attempt_payload = self.attempt.to_dict()
        atomic_write_json(
            self.artifact_root / "protocol" / "attempt-manifest.json", attempt_payload
        )
        phase_state = {
            "schema": f"{PROTOCOL_SCHEMA_VERSION}.phase_state",
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            **_binding(self.attempt),
            "status": self.attempt.status,
            "updated_at": _utc_now(),
            "attempt_hash": attempt_payload["attempt_hash"],
            "transition_count": self._transition_count,
            "last_transition_hash": self._transition_hash,
        }
        phase_state["state_hash"] = stable_hash(phase_state)
        atomic_write_json(
            self.artifact_root / "protocol" / "phase-state.json", phase_state
        )

    def _append_transition(
        self,
        previous_state: str | None,
        next_state: str,
        event: str,
        metadata: Mapping[str, Any],
    ) -> None:
        payload = {
            "schema": f"{PROTOCOL_SCHEMA_VERSION}.transition",
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            **_binding(self.attempt),
            "sequence": self._transition_count,
            "previous_transition_hash": self._transition_hash,
            "previous_state": previous_state,
            "next_state": next_state,
            "event": _nonempty("event", event),
            "created_at": _utc_now(),
            "metadata": _mapping(metadata),
            "attempt": self.attempt.to_dict(),
        }
        payload["transition_hash"] = stable_hash(payload)
        append_jsonl(self.artifact_root / "protocol" / "transitions.jsonl", payload)
        self._transition_count += 1
        self._transition_hash = payload["transition_hash"]

    def _transition(
        self,
        next_state: str,
        event: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        sealed_at: str | None = None,
        decision: str | None = None,
        permitted_next_action: str | None = None,
    ) -> ProtocolAttempt:
        current = self.attempt.status
        if next_state not in STATES:
            raise ValueError(f"unknown protocol state: {next_state}")
        if next_state not in ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"illegal protocol transition: {current} -> {next_state}")
        self._require_fresh()
        if next_state in TERMINAL_STATES:
            persist_pending_boundary(self)
        updated = replace(
            self.attempt,
            status=next_state,
            sealed_at=sealed_at if sealed_at is not None else self.attempt.sealed_at,
            decision=decision if decision is not None else self.attempt.decision,
            permitted_next_action=(
                permitted_next_action
                if permitted_next_action is not None
                else self.attempt.permitted_next_action
            ),
        )
        self._attempt = updated
        self._append_transition(current, next_state, event, metadata)
        self._write_protocol_records()
        return updated

    def _require_fresh(self) -> None:
        persisted = self.load(self.artifact_root, self.experiment_root).attempt
        if persisted.to_dict() != self.attempt.to_dict():
            raise ValueError("stale protocol controller; no writes permitted")

    def preflight_pass(self) -> ProtocolAttempt:
        attempt = self.attempt
        if not attempt.evidence_contract:
            raise ValueError("preflight requires an evidence contract")
        if not attempt.baseline:
            raise ValueError("preflight requires a baseline")
        if not attempt.resources:
            raise ValueError("preflight requires declared resources")
        if attempt.lane in ("Validate", "Admit") and not attempt.auditor_id:
            raise ValueError("preflight requires an independent auditor")
        return self._transition("PREFLIGHT_PASSED", "preflight_passed")

    def start_running(self) -> ProtocolAttempt:
        self._require_fresh()
        self.require_current_program()
        attempt = self.attempt
        if not attempt.source_hashes or not attempt.input_hashes:
            raise ValueError("running requires source and input hashes")
        if not attempt.role_manifest_hash:
            raise ValueError("running requires a role manifest hash")
        for path in (
            self.artifact_root / "checkpoints",
            self.artifact_root / "recovery",
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self._transition("RUNNING", "execution_started")

    def require_current_program(self) -> None:
        attempt = self.attempt
        require_program_attempt(
            attempt.program_id, attempt.phase_id, attempt.lane,
            attempt.plan_hash, attempt.evidence_contract, attempt.attempt_id,
            attempt.status,
        )

    def seal(
        self,
        *,
        result: ResultRecord | Mapping[str, Any],
        checkpoint_records: Sequence[CheckpointRecord | Mapping[str, Any]],
        counters: Mapping[str, Any],
        command_log: Sequence[str],
        manifests: Mapping[str, Any],
    ) -> ProtocolAttempt:
        if self.attempt.status != "RUNNING":
            raise ValueError("only a running attempt can be sealed")
        self._require_fresh()
        if (self.artifact_root / "sealed-result.json").exists():
            raise FileExistsError("an existing sealed result cannot be overwritten")
        if not checkpoint_records:
            raise ValueError("sealing requires checkpoint records")
        if not counters or not command_log or not manifests:
            raise ValueError("sealing requires counters, command log, and manifests")
        result_value = ResultRecord.from_dict(
            result.to_dict() if isinstance(result, ResultRecord) else result
        )
        records = tuple(
            CheckpointRecord.from_dict(
                item.to_dict() if isinstance(item, CheckpointRecord) else item
            )
            for item in checkpoint_records
        )
        self._validate_result(result_value, records, counters)
        command_log = _strings("command_log", command_log)
        manifests = _mapping(manifests)
        manifest_hashes = {
            name: _sha256_file(_relative_file(self.artifact_root, path))
            for name, path in manifests.items()
        }
        self._validate_roles(manifests)
        write_hash_manifest(self.experiment_root)
        envelope = {
            "schema": f"{PROTOCOL_SCHEMA_VERSION}.sealed_result",
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            **_binding(self.attempt),
            "status": "SEALED",
            "result": result_value.to_dict(),
            "checkpoint_records": [item.to_dict() for item in records],
            "counters": _mapping(counters),
            "command_log": list(command_log),
            "manifests": _mapping(manifests),
            "manifest_hashes": manifest_hashes,
            "experiment_hash_manifest": _sha256_file(
                self.experiment_root / "hashes.txt"
            ),
            "sealed_at": _utc_now(),
        }
        envelope["sealed_result_hash"] = stable_hash(envelope)
        atomic_write_json(self.artifact_root / "sealed-result.json", envelope)
        sealed_at = envelope["sealed_at"]
        updated = self._transition(
            "SEALED",
            "boundary_sealed",
            metadata={"sealed_result_hash": envelope["sealed_result_hash"]},
            sealed_at=sealed_at,
        )
        write_hash_manifest(self.artifact_root)
        return updated

    def _validate_roles(self, manifests):
        from .generic_scheduled_role_banks import (
            SCHEDULED_ROLE_REGISTRY_SCHEMA,
            ScheduledRoleBankRegistry,
        )

        if "role_manifest" not in manifests:
            raise ValueError("sealed packet requires a role manifest")
        payload = safe_load_json(
            _relative_file(self.artifact_root, manifests["role_manifest"])
        )
        if isinstance(payload, Mapping) and payload.get("schema") == SCHEDULED_ROLE_REGISTRY_SCHEMA:
            roles = ScheduledRoleBankRegistry.from_dict(payload)
        else:
            roles = RoleManifest.from_dict(payload)
        if roles.binding_hash() != self.attempt.role_manifest_hash:
            raise ValueError("role manifest binding mismatch")

    def _validate_result(self, result, records, counters):
        if result.attempt_id != self.attempt.attempt_id or not records:
            raise ValueError("result attempt or checkpoint binding mismatch")
        indices = tuple(item.update_index for item in records)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("checkpoint indices must be unique and ordered")
        states = tuple(validate_checkpoint(self.attempt, item) for item in records)
        latest = states[-1]
        if (
            result.update_count != latest.update_index
            or result.policy_fingerprint != latest.policy_fingerprint
        ):
            raise ValueError("result update count or policy fingerprint mismatch")
        if result.checkpoint_hash != records[-1].checkpoint_hash:
            raise ValueError("result checkpoint hash mismatch")
        counters = _mapping(counters)
        if (
            _strict_int("update_count", counters.get("update_count"))
            != result.update_count
        ):
            raise ValueError("result counters mismatch")
        if not set(self.attempt.nonclaims).issubset(result.nonclaims):
            raise ValueError("result must retain attempt nonclaims")

    def validate_sealed_result(self) -> Mapping[str, Any]:
        envelope = _protocol_payload(
            safe_load_json(self.artifact_root / "sealed-result.json"),
            "sealed_result",
            "sealed_result_hash",
        )
        _check_binding(envelope, self.attempt)
        if (
            envelope.get("status") != "SEALED"
            or envelope.get("sealed_at") != self.attempt.sealed_at
        ):
            raise ValueError("sealed result status or time mismatch")
        result = ResultRecord.from_dict(envelope["result"])
        records = tuple(
            CheckpointRecord.from_dict(item) for item in envelope["checkpoint_records"]
        )
        self._validate_result(result, records, envelope["counters"])
        _strings("command_log", envelope["command_log"])
        manifests = _mapping(envelope["manifests"])
        hashes = {
            name: _sha256_file(_relative_file(self.artifact_root, path))
            for name, path in manifests.items()
        }
        if hashes != envelope["manifest_hashes"]:
            raise ValueError("sealed manifest hash mismatch")
        self._validate_roles(manifests)
        verify_hash_manifest(self.experiment_root)
        if (
            _sha256_file(self.experiment_root / "hashes.txt")
            != envelope["experiment_hash_manifest"]
        ):
            raise ValueError("experiment hash manifest mismatch")
        return envelope

    def refresh_hash_manifest(self) -> Path:
        raise ValueError("sealed payloads are immutable; repairs require a new attempt")

    def submit_for_audit(self) -> ProtocolAttempt:
        if self.attempt.status != "SEALED":
            raise ValueError("only a sealed attempt can enter audit pending")
        verify_hash_manifest(self.artifact_root)
        self.validate_sealed_result()
        return self._transition("AUDIT_PENDING", "submitted_for_audit")

    def record_audit(self, record: AuditRecord) -> ProtocolAttempt:
        if self.attempt.status != "AUDIT_PENDING":
            raise ValueError("audit records require AUDIT_PENDING")
        self._require_fresh()
        if (self.artifact_root / "protocol" / "audit-record.json").exists():
            raise FileExistsError("an existing audit record cannot be overwritten")
        record = AuditRecord.from_dict(record.to_dict())
        if record.attempt_id != self.attempt.attempt_id:
            raise ValueError("audit attempt ID mismatch")
        if (
            self.attempt.lane in ("Validate", "Admit")
            and record.auditor_id == self.attempt.executor_id
        ):
            raise ValueError("audit identity must differ from executor identity")
        expected_auditor = self.attempt.auditor_id or self.attempt.executor_id
        if record.auditor_id != expected_auditor:
            raise ValueError("audit must use the declared auditor identity")
        if record.decision == "PASS":
            AuditCoordinator.verify_packet(self)
        expected_state = {
            "PASS": "AUDIT_PASSED",
            "FAIL": "AUDIT_FAILED",
            "BLOCKED": "BLOCKED",
            "INCONCLUSIVE": "BLOCKED",
        }[record.decision]
        atomic_write_json(
            self.artifact_root / "protocol" / "audit-record.json", record.to_dict()
        )
        return self._transition(
            expected_state,
            "audit_recorded",
            metadata={"audit_hash": record.to_dict()["audit_hash"]},
        )

    def close(
        self,
        *,
        decision: str,
        permitted_next_action: str,
        gate_ledger: GateLedger,
        promotion_claim: bool = False,
    ) -> ProtocolAttempt:
        if self.attempt.status != "AUDIT_PASSED":
            raise ValueError("only an audited-passed attempt can close")
        self._require_fresh()
        if (self.artifact_root / "protocol" / "gate-ledger.json").exists():
            raise FileExistsError("an existing closure ledger cannot be overwritten")
        _strict_bool("promotion_claim", promotion_claim)
        if self.attempt.lane == "Explore" and (
            promotion_claim
            or decision == "PROMOTE"
            or any(item.effect == "promotion_block" for item in gate_ledger.records)
        ):
            raise ValueError("Explore self-review cannot close a promotion gate")
        gate_ledger.validate_close(decision, promotion_claim=promotion_claim)
        _nonempty("permitted_next_action", permitted_next_action)
        AuditCoordinator.verify_packet(self)
        atomic_write_json(
            self.artifact_root / "protocol" / "gate-ledger.json", gate_ledger.to_dict()
        )
        return self._transition(
            "CLOSED",
            "phase_closed",
            metadata={
                "gate_ledger_hash": gate_ledger.to_dict()["ledger_hash"],
                "promotion_claim": promotion_claim,
            },
            decision=_nonempty("decision", decision),
            permitted_next_action=_nonempty(
                "permitted_next_action", permitted_next_action
            ),
        )

    def block(self, reason: str) -> ProtocolAttempt:
        if self.attempt.status not in {"RUNNING", "SEALED", "AUDIT_PENDING"}:
            raise ValueError("only an active attempt can be blocked")
        return self._transition(
            "BLOCKED",
            "attempt_blocked",
            metadata={"reason": _nonempty("reason", reason)},
        )

    @classmethod
    def load(
        cls, artifact_root: str | Path, experiment_root: str | Path | None = None
    ) -> ProtocolController:
        root = Path(_absolute_path("artifact_root", artifact_root))
        payload = safe_load_json(root / "protocol" / "attempt-manifest.json")
        state = safe_load_json(root / "protocol" / "phase-state.json")
        if not isinstance(payload, Mapping) or not isinstance(state, Mapping):
            raise TypeError(
                "protocol packet is missing attempt manifest or phase state"
            )
        attempt = ProtocolAttempt.from_dict(payload)
        if str(root) != attempt.artifact_root:
            raise ValueError("loaded artifact root mismatch")
        if (
            experiment_root is not None
            and _absolute_path("experiment_root", experiment_root)
            != attempt.experiment_root
        ):
            raise ValueError("loaded experiment root mismatch")
        state = _protocol_payload(state, "phase_state", "state_hash")
        _check_binding(state, attempt)
        if (
            state.get("attempt_hash") != payload["attempt_hash"]
            or state.get("status") != attempt.status
        ):
            raise ValueError("phase state and attempt disagree")
        binding = _protocol_payload(
            safe_load_json(Path(attempt.experiment_root) / "attempt-binding.json"),
            "experiment_binding",
            "binding_hash",
        )
        _check_binding(binding, attempt)
        controller = cls(
            program_id=attempt.program_id,
            phase_id=attempt.phase_id,
            lane=attempt.lane,
            plan_hash=attempt.plan_hash,
            executor_id=attempt.executor_id,
            auditor_id=attempt.auditor_id,
            decision_owner=attempt.decision_owner,
            artifact_base=root.parent,
            experiment_base=Path(attempt.experiment_root).parent,
        )
        controller._attempt = attempt
        controller._artifact_root = root
        controller._experiment_root = Path(attempt.experiment_root)
        controller._validate_transitions(state)
        return controller

    def _validate_transitions(self, phase_state):
        previous = None
        previous_hash = None
        snapshots = []
        for sequence, line in enumerate(
            (self.artifact_root / "protocol" / "transitions.jsonl")
            .read_text()
            .splitlines()
        ):
            record = _protocol_payload(
                json.loads(line), "transition", "transition_hash"
            )
            _check_binding(record, self.attempt)
            if (
                _strict_int("sequence", record["sequence"]) != sequence
                or record["previous_transition_hash"] != previous_hash
            ):
                raise ValueError("broken transition hash chain")
            if record["previous_state"] != previous:
                raise ValueError("transition previous state mismatch")
            current = record["next_state"]
            if (previous is None and current != "PLANNED") or (
                previous is not None and current not in ALLOWED_TRANSITIONS[previous]
            ):
                raise ValueError("illegal transition in journal")
            snapshot = ProtocolAttempt.from_dict(record["attempt"])
            if snapshot.status != current:
                raise ValueError("transition snapshot state mismatch")
            frozen = replace(
                snapshot,
                status=self.attempt.status,
                sealed_at=self.attempt.sealed_at,
                decision=self.attempt.decision,
                permitted_next_action=self.attempt.permitted_next_action,
            )
            if frozen.to_dict() != self.attempt.to_dict():
                raise ValueError("attempt identity changed in journal")
            snapshots.append(record)
            previous, previous_hash = current, record["transition_hash"]
        if not snapshots or snapshots[-1]["attempt"] != self.attempt.to_dict():
            raise ValueError("transition journal does not reach current attempt")
        if (
            _strict_int("transition_count", phase_state["transition_count"])
            != len(snapshots)
            or phase_state["last_transition_hash"] != previous_hash
        ):
            raise ValueError("phase state transition chain mismatch")
        for record in snapshots:
            metadata = record["metadata"]
            if record["next_state"] == "SEALED":
                envelope = self.validate_sealed_result()
                if metadata.get("sealed_result_hash") != envelope["sealed_result_hash"]:
                    raise ValueError("sealed result journal hash mismatch")
            if (
                record["previous_state"] == "AUDIT_PENDING"
                and record["next_state"] != "BLOCKED"
                or record["event"] == "audit_recorded"
            ):
                audit = AuditRecord.from_dict(
                    safe_load_json(
                        self.artifact_root / "protocol" / "audit-record.json"
                    )
                )
                if audit.attempt_id != self.attempt.attempt_id or audit.auditor_id != (
                    self.attempt.auditor_id or self.attempt.executor_id
                ):
                    raise ValueError("audit identity binding mismatch")
                expected_state = {
                    "PASS": "AUDIT_PASSED",
                    "FAIL": "AUDIT_FAILED",
                    "BLOCKED": "BLOCKED",
                    "INCONCLUSIVE": "BLOCKED",
                }[audit.decision]
                if (
                    metadata.get("audit_hash") != audit.to_dict()["audit_hash"]
                    or record["next_state"] != expected_state
                ):
                    raise ValueError("audit journal binding mismatch")
            if record["next_state"] == "CLOSED":
                ledger = GateLedger.from_dict(
                    safe_load_json(self.artifact_root / "protocol" / "gate-ledger.json")
                )
                if metadata.get("gate_ledger_hash") != ledger.to_dict()["ledger_hash"]:
                    raise ValueError("gate ledger journal hash mismatch")
                ledger.validate_close(
                    self.attempt.decision, promotion_claim=metadata["promotion_claim"]
                )
                if self.attempt.lane == "Explore" and (
                    metadata["promotion_claim"]
                    or self.attempt.decision == "PROMOTE"
                    or any(item.effect == "promotion_block" for item in ledger.records)
                ):
                    raise ValueError(
                        "Explore self-review cannot close a promotion gate"
                    )
        self._transition_count = len(snapshots)
        self._transition_hash = previous_hash


class AuditCoordinator:
    """Run the bounded packet audit required by the protocol."""

    @staticmethod
    def verify_packet(controller: ProtocolController) -> dict[str, Any]:
        loaded = ProtocolController.load(
            controller.artifact_root, controller.experiment_root
        )
        if loaded.attempt.to_dict() != controller.attempt.to_dict():
            raise ValueError("stale controller at audit boundary")
        result = verify_hash_manifest(controller.artifact_root)
        controller.validate_sealed_result()
        return result

    def audit(
        self,
        controller: ProtocolController,
        *,
        auditor_id: str,
        independent_checks: Mapping[str, bool],
        checklist_version: str = "phase1-v1",
        findings: Sequence[str] = (),
        veto_status: str = "PASS",
        uncertainty: Mapping[str, Any] | None = None,
        decision: str = "PASS",
        required_repairs: Sequence[str] = (),
    ) -> AuditRecord:
        if controller.attempt.status != "AUDIT_PENDING":
            raise ValueError("audit requires AUDIT_PENDING")
        verified = False
        try:
            self.verify_packet(controller)
            verified = True
        except (OSError, TypeError, ValueError) as exc:
            findings = tuple(findings) + (f"artifact_verification_failed:{exc}",)
        record = AuditRecord(
            attempt_id=controller.attempt.attempt_id,
            auditor_id=auditor_id,
            checklist_version=checklist_version,
            artifact_hashes_verified=verified,
            independent_checks=independent_checks,
            findings=tuple(findings),
            veto_status=veto_status,
            uncertainty={} if uncertainty is None else uncertainty,
            decision=decision if verified else "FAIL",
            required_repairs=tuple(required_repairs),
        )
        controller.record_audit(record)
        return record


__all__ = [
    "ALLOWED_TRANSITIONS",
    "AUDIT_DECISIONS",
    "GATE_EFFECTS",
    "GATE_STATUSES",
    "PROTOCOL_SCHEMA_VERSION",
    "STATES",
    "TERMINAL_STATES",
    "AuditCoordinator",
    "AuditRecord",
    "GateLedger",
    "GateRecord",
    "ProtocolAttempt",
    "ProtocolController",
    "validate_checkpoint",
    "verify_hash_manifest",
    "write_checkpoint",
    "write_hash_manifest",
]
