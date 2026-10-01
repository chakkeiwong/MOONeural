"""Immutable role-bank contracts for model evaluation and certification."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .generic_training_contracts import (
    CertificationResult,
    ControlEvaluation,
    EvaluationRequest,
    PolicyView,
    ValidationEvaluation,
    canonical_json,
    stable_hash,
)

ROLE_BANK_SCHEMA = "dsge_hmc.generic_role_bank.v1"
ROLE_BANK_MANIFEST_SCHEMA = "dsge_hmc.generic_role_bank_manifest.v1"


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _digest(name: str, value: str) -> str:
    normalized = _nonempty(name, value).lower()
    if len(normalized) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(normalized, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    return normalized


def _strings(name: str, values: Sequence[str], *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_nonempty(name, value) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return normalized


def _freeze_mapping(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        normalized = __import__("json").loads(canonical_json(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-safe") from error
    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(value) for key, value in item.items()})
        if isinstance(item, list):
            return tuple(freeze(value) for value in item)
        return item

    return freeze(normalized)


@dataclass(frozen=True)
class RoleBank:
    """One independently seeded, fully identified evaluation bank."""

    role: str
    bank_id: str
    seed: int
    target_id: str
    target_version: str
    scale_version: str
    estimator_version: str
    input_hashes: Mapping[str, str]
    sample_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _nonempty("role", self.role))
        object.__setattr__(self, "bank_id", _nonempty("bank_id", self.bank_id))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("role-bank seed must be a non-negative integer")
        for name in ("target_id", "target_version", "scale_version", "estimator_version"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if not isinstance(self.input_hashes, Mapping) or not self.input_hashes:
            raise ValueError("role-bank input hashes must not be empty")
        normalized_hashes = {
            _nonempty("input hash key", key): _digest("input hash", value)
            for key, value in self.input_hashes.items()
        }
        object.__setattr__(self, "input_hashes", MappingProxyType(normalized_hashes))
        object.__setattr__(self, "sample_ids", _strings("sample ID", self.sample_ids))
        object.__setattr__(self, "metadata", _freeze_mapping("metadata", self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROLE_BANK_SCHEMA,
            "role": self.role,
            "bank_id": self.bank_id,
            "seed": self.seed,
            "target_id": self.target_id,
            "target_version": self.target_version,
            "scale_version": self.scale_version,
            "estimator_version": self.estimator_version,
            "input_hashes": dict(self.input_hashes),
            "sample_ids": list(self.sample_ids),
            "metadata": dict(self.metadata),
        }

    def binding_hash(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoleBank:
        if not isinstance(payload, Mapping) or payload.get("schema") != ROLE_BANK_SCHEMA:
            raise ValueError("unknown role-bank schema")
        data = dict(payload)
        data.pop("schema", None)
        expected = {
            "role", "bank_id", "seed", "target_id", "target_version",
            "scale_version", "estimator_version", "input_hashes", "sample_ids",
            "metadata",
        }
        if set(data) != expected:
            raise ValueError("role-bank fields do not match schema")
        return cls(**data)


@dataclass(frozen=True)
class RoleBankManifest:
    """Exact role-bank inventory used by control, validation, and certification."""

    task_ids: tuple[str, ...]
    banks: tuple[RoleBank, ...]
    expected_counts: Mapping[str, int]
    manifest_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_ids", _strings("task ID", self.task_ids))
        banks = tuple(self.banks)
        if not banks or not all(isinstance(bank, RoleBank) for bank in banks):
            raise ValueError("role-bank manifest must contain RoleBank entries")
        expected = {}
        for role, count in self.expected_counts.items():
            role_name = _nonempty("expected role", role)
            if type(count) is not int or count <= 0:
                raise ValueError("role-bank expected counts must be positive integers")
            expected[role_name] = count
        if not expected:
            raise ValueError("role-bank expected counts must not be empty")
        counts: dict[str, int] = {}
        bank_ids: set[str] = set()
        seeds: dict[int, str] = {}
        samples: dict[str, str] = {}
        for bank in banks:
            if bank.bank_id in bank_ids:
                raise ValueError("role-bank IDs must be unique")
            bank_ids.add(bank.bank_id)
            counts[bank.role] = counts.get(bank.role, 0) + 1
            if bank.seed in seeds:
                raise ValueError("role-bank seeds must be unique across roles")
            seeds[bank.seed] = bank.role
            for sample_id in bank.sample_ids:
                if sample_id in samples:
                    raise ValueError("role-bank samples must be unique across roles")
                samples[sample_id] = bank.role
        if set(counts) != set(expected) or counts != expected:
            raise ValueError("role-bank inventory does not match exact expected counts")
        for role in expected:
            role_banks = tuple(bank for bank in banks if bank.role == role)
            identity = tuple(
                (bank.target_id, bank.target_version, bank.scale_version, bank.estimator_version)
                for bank in role_banks
            )
            if len(set(identity)) != 1:
                raise ValueError("role-bank target, scale, or estimator identity drift")
        object.__setattr__(self, "banks", banks)
        object.__setattr__(self, "expected_counts", MappingProxyType(expected))
        object.__setattr__(self, "manifest_version", _nonempty("manifest version", self.manifest_version))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": ROLE_BANK_MANIFEST_SCHEMA,
            "manifest_version": self.manifest_version,
            "task_ids": list(self.task_ids),
            "expected_counts": dict(self.expected_counts),
            "banks": [bank.to_dict() for bank in self.banks],
        }
        payload["role_bank_manifest_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoleBankManifest:
        if not isinstance(payload, Mapping) or payload.get("schema") != ROLE_BANK_MANIFEST_SCHEMA:
            raise ValueError("unknown role-bank manifest schema")
        data = dict(payload)
        supplied = data.pop("role_bank_manifest_hash", None)
        if supplied != stable_hash(data):
            raise ValueError("role-bank manifest hash mismatch")
        data.pop("schema", None)
        expected = {"manifest_version", "task_ids", "expected_counts", "banks"}
        if set(data) != expected:
            raise ValueError("role-bank manifest fields do not match schema")
        return cls(
            task_ids=tuple(data["task_ids"]),
            banks=tuple(RoleBank.from_dict(item) for item in data["banks"]),
            expected_counts=data["expected_counts"],
            manifest_version=data["manifest_version"],
        )

    def binding_hash(self) -> str:
        return self.to_dict()["role_bank_manifest_hash"]

    def role_banks(self, role: str) -> tuple[RoleBank, ...]:
        role = _nonempty("role", role)
        if role not in self.expected_counts:
            raise ValueError("role is not registered in the role-bank manifest")
        result = tuple(bank for bank in self.banks if bank.role == role)
        if len(result) != self.expected_counts[role]:
            raise ValueError("role-bank count is not exact")
        return result

    def require_global_inputs(self, role: str) -> tuple[RoleBank, ...]:
        banks = self.role_banks(role)
        if any("global" not in bank.input_hashes for bank in banks):
            raise ValueError("exact global role-bank inputs are unavailable")
        return banks

    def request_for(self, role: str, policy: PolicyView) -> EvaluationRequest:
        if not isinstance(policy, PolicyView):
            raise TypeError("a PolicyView is required to bind a role-bank request")
        banks = self.require_global_inputs(role)
        first = banks[0]
        return EvaluationRequest(
            role=role,
            target_id=first.target_id,
            seeds=tuple(bank.seed for bank in banks),
            sample_count=len(banks),
            target_version=first.target_version,
            estimator_version=first.estimator_version,
            scale_version=first.scale_version,
            policy_fingerprint=policy.fingerprint(),
            task_ids=self.task_ids,
            anchor_ids=tuple(sample_id for bank in banks for sample_id in bank.sample_ids),
            metadata={
                "role_bank_manifest_hash": self.binding_hash(),
                "bank_ids": [bank.bank_id for bank in banks],
                "bank_hashes": [bank.binding_hash() for bank in banks],
                "input_hashes": {bank.bank_id: dict(bank.input_hashes) for bank in banks},
                "sample_ids_by_bank": {bank.bank_id: list(bank.sample_ids) for bank in banks},
            },
        )

    def validate_request_binding(self, request: EvaluationRequest) -> None:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("an EvaluationRequest is required")
        banks = self.require_global_inputs(request.role)
        expected = self.request_for(request.role, PolicyView((0.0,), "request-binding"))
        actual = request.to_dict()
        actual.pop("policy_fingerprint")
        expected_payload = expected.to_dict()
        expected_payload.pop("policy_fingerprint")
        if actual != expected_payload:
            raise ValueError("role-bank request binding mismatch")
        if len(banks) != request.sample_count:
            raise ValueError("role-bank request sample count mismatch")

    def validate_request(self, request: EvaluationRequest, policy: PolicyView) -> None:
        expected = self.request_for(request.role, policy)
        if request.to_dict() != expected.to_dict():
            raise ValueError("role-bank request binding mismatch")


class RoleBankEvaluators:
    """Build result objects whose role, bank, and policy bindings are verified."""

    def __init__(
        self,
        manifest: RoleBankManifest,
        *,
        control: Callable[[PolicyView, EvaluationRequest], ControlEvaluation],
        validation: Callable[[PolicyView, EvaluationRequest], ValidationEvaluation],
        certification: Callable[[PolicyView, EvaluationRequest], CertificationResult],
    ) -> None:
        if not isinstance(manifest, RoleBankManifest):
            raise TypeError("a RoleBankManifest is required")
        for callback in (control, validation, certification):
            if not callable(callback):
                raise TypeError("all role-bank evaluators must be callable")
        self.manifest = manifest
        self._callbacks = {
            "control": control,
            "validation": validation,
            "certification": certification,
        }

    @property
    def binding_hash(self) -> str:
        return self.manifest.binding_hash()

    def _invoke(self, role: str, policy: PolicyView, expected_type: type):
        request = self.manifest.request_for(role, policy)
        result = self._callbacks[role](policy, request)
        if not isinstance(result, expected_type):
            raise TypeError(f"{role} evaluator returned the wrong result type")
        if result.request is None or result.request.to_dict() != request.to_dict():
            raise ValueError(f"{role} evaluator returned an unbound request")
        if role == "control":
            if tuple(result.task_upper_mse) != self.manifest.task_ids:
                raise ValueError("control task order does not match role-bank manifest")
        elif role == "validation":
            if tuple(result.task_mean_mse) != self.manifest.task_ids:
                raise ValueError("validation task order does not match role-bank manifest")
            if any(count != request.sample_count for count in result.sample_counts.values()):
                raise ValueError("validation sample counts do not match role-bank count")
            if result.provenance.get("role_bank_manifest_hash") != self.binding_hash:
                raise ValueError("validation provenance is not role-bank bound")
        else:
            if tuple(result.estimator_metadata.get("task_ids", self.manifest.task_ids)) != self.manifest.task_ids:
                raise ValueError("certification task order does not match role-bank manifest")
            if result.estimator_metadata.get("role_bank_manifest_hash") != self.binding_hash:
                raise ValueError("certification provenance is not role-bank bound")
        return result

    def control_evaluation(self, policy: PolicyView) -> ControlEvaluation:
        return self._invoke("control", policy, ControlEvaluation)

    def validation_evaluation(self, policy: PolicyView) -> ValidationEvaluation:
        return self._invoke("validation", policy, ValidationEvaluation)

    def certification_result(self, policy: PolicyView) -> CertificationResult:
        return self._invoke("certification", policy, CertificationResult)


__all__ = [
    "ROLE_BANK_MANIFEST_SCHEMA",
    "ROLE_BANK_SCHEMA",
    "RoleBank",
    "RoleBankEvaluators",
    "RoleBankManifest",
]
