"""Immutable scoped dispatch to exact materialized role-bank manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from .generic_role_banks import RoleBankManifest
from .generic_training_contracts import (
    EvaluationRequest,
    ImmutableJSONMapping,
    PolicyView,
    canonical_json,
    stable_hash,
)

SCHEDULED_ROLE_BINDING_SCHEMA = "dsge_hmc.scheduled_role_binding.v1"
SCHEDULED_ROLE_REGISTRY_SCHEMA = "dsge_hmc.scheduled_role_bank_registry.v1"
ROLES = ("control", "validation", "certification")


def _nonnegative(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _scope(stage_id, arm_id, role, round_number):
    _nonempty("stage_id", stage_id)
    _nonnegative("arm_id", arm_id)
    if role not in ROLES:
        raise ValueError("unknown scheduled role")
    if role == "control":
        _nonnegative("control round_number", round_number)
    elif round_number is not None:
        raise ValueError("validation/certification round_number must be None")
    return stage_id, arm_id, role, round_number


def _unpack(payload, schema, hash_field, fields):
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ValueError("unknown scheduled role schema")
    if set(payload) != fields | {"schema", hash_field}:
        raise ValueError("scheduled role fields do not match schema")
    data = dict(payload)
    supplied = data.pop(hash_field)
    if supplied != stable_hash(data):
        raise ValueError("scheduled role hash mismatch")
    data.pop("schema")
    return data


@dataclass(frozen=True)
class ScheduledRoleBinding:
    """One stage/arm/role/round bank with inclusive allowed update bounds."""

    stage_id: str
    arm_id: int
    role: str
    round_number: int | None
    min_update: int
    max_update: int
    manifest: RoleBankManifest

    def __post_init__(self) -> None:
        _scope(self.stage_id, self.arm_id, self.role, self.round_number)
        _nonnegative("min_update", self.min_update)
        _nonnegative("max_update", self.max_update)
        if self.min_update > self.max_update:
            raise ValueError("scheduled update range is reversed")
        if self.role == "control" and self.min_update != self.max_update:
            raise ValueError("control requires one exact update")
        if not isinstance(self.manifest, RoleBankManifest):
            raise TypeError("a materialized RoleBankManifest is required")
        manifest = RoleBankManifest.from_dict(json.loads(canonical_json(self.manifest.to_dict())))
        if set(manifest.expected_counts) != {self.role}:
            raise ValueError("scheduled manifest must contain exactly its bound role")
        manifest.require_global_inputs(self.role)
        object.__setattr__(self, "manifest", manifest)

    @property
    def scope(self) -> tuple[str, int, str, int | None]:
        return self.stage_id, self.arm_id, self.role, self.round_number

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SCHEDULED_ROLE_BINDING_SCHEMA,
            "stage_id": self.stage_id,
            "arm_id": self.arm_id,
            "role": self.role,
            "round_number": self.round_number,
            "min_update": self.min_update,
            "max_update": self.max_update,
            "manifest": json.loads(canonical_json(self.manifest.to_dict())),
        }
        payload["scheduled_role_binding_hash"] = stable_hash(payload)
        return payload

    def binding_hash(self) -> str:
        return self.to_dict()["scheduled_role_binding_hash"]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScheduledRoleBinding:
        data = _unpack(
            payload, SCHEDULED_ROLE_BINDING_SCHEMA, "scheduled_role_binding_hash",
            {"stage_id", "arm_id", "role", "round_number", "min_update", "max_update", "manifest"},
        )
        data["manifest"] = RoleBankManifest.from_dict(data["manifest"])
        binding = cls(**data)
        if canonical_json(binding.to_dict()) != canonical_json(payload):
            raise ValueError("scheduled role binding is not canonical")
        return binding


@dataclass(frozen=True)
class ScheduledRoleBankRegistry:
    """Immutable role registry; bank generation and evaluation stay external."""

    bindings: tuple[ScheduledRoleBinding, ...]
    local_seed_reuse: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    registry_version: str = "v1"
    _by_scope: Mapping[tuple[str, int, str, int | None], ScheduledRoleBinding] = field(
        init=False, repr=False, compare=False,
    )
    _binding_hash: str = field(init=False, repr=False, compare=False)
    _payload_cache: ImmutableJSONMapping = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        if not bindings or not all(isinstance(binding, ScheduledRoleBinding) for binding in bindings):
            raise ValueError("registry requires ScheduledRoleBinding entries")
        _nonempty("registry_version", self.registry_version)
        if not isinstance(self.local_seed_reuse, Mapping):
            raise TypeError("local_seed_reuse must be a mapping")
        declared = {}
        for role, seeds in self.local_seed_reuse.items():
            if role not in ROLES or not isinstance(seeds, (tuple, list)):
                raise ValueError("local seed reuse requires a role and seed sequence")
            normalized = tuple(_nonnegative("reused local seed", seed) for seed in seeds)
            if not normalized or len(set(normalized)) != len(normalized):
                raise ValueError("reused local seeds must be nonempty and unique")
            declared[role] = normalized
        by_scope = {}
        manifest_hashes = set()
        seeds_seen = {}
        samples_seen = {}
        occurrences = {}
        control_global_scopes = {}
        for binding in bindings:
            if binding.scope in by_scope:
                raise ValueError("duplicate scheduled scope")
            if binding.manifest.task_ids != bindings[0].manifest.task_ids:
                raise ValueError("scheduled role task order mismatch")
            manifest_hash = binding.manifest.binding_hash()
            if manifest_hash in manifest_hashes:
                raise ValueError("scopes require distinct role-bank manifests")
            by_scope[binding.scope] = binding
            manifest_hashes.add(manifest_hash)
            for bank in binding.manifest.banks:
                previous_role = seeds_seen.setdefault(bank.seed, bank.role)
                if previous_role != bank.role:
                    raise ValueError("local seeds must not overlap different roles")
                occurrences.setdefault((bank.role, bank.seed), []).append(bank)
                if bank.role == "control":
                    previous_scope = control_global_scopes.setdefault(bank.input_hashes["global"], binding.scope)
                    if previous_scope != binding.scope:
                        raise ValueError("control scopes require fresh global input hashes")
                for sample_id in bank.sample_ids:
                    previous_role = samples_seen.setdefault(sample_id, bank.role)
                    if previous_role != bank.role:
                        raise ValueError("sample IDs must not overlap different roles")
        repeated = {}
        for (role, seed), banks in occurrences.items():
            if len(banks) < 2:
                continue
            repeated.setdefault(role, set()).add(seed)
            local_hashes = {bank.input_hashes.get("local") for bank in banks}
            if None in local_hashes or len(local_hashes) != 1:
                raise ValueError("reused local seeds require identical local input hashes")
        if {role: set(seeds) for role, seeds in declared.items()} != repeated:
            raise ValueError("local seed reuse must declare exactly the repeated seeds")
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "local_seed_reuse", MappingProxyType(declared))
        object.__setattr__(self, "_by_scope", MappingProxyType(by_scope))
        payload = self._payload()
        object.__setattr__(self, "_binding_hash", stable_hash(payload))
        object.__setattr__(self, "_payload_cache", ImmutableJSONMapping({
            **payload, "scheduled_role_registry_hash": self._binding_hash,
        }))

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.bindings[0].manifest.task_ids

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULED_ROLE_REGISTRY_SCHEMA,
            "registry_version": self.registry_version,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "local_seed_reuse": {role: list(seeds) for role, seeds in self.local_seed_reuse.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        return self._payload_cache.to_dict()

    def immutable_payload(self):
        return self._payload_cache

    def binding_hash(self) -> str:
        return self._binding_hash

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScheduledRoleBankRegistry:
        data = _unpack(
            payload, SCHEDULED_ROLE_REGISTRY_SCHEMA, "scheduled_role_registry_hash",
            {"registry_version", "bindings", "local_seed_reuse"},
        )
        registry = cls(
            bindings=tuple(ScheduledRoleBinding.from_dict(item) for item in data["bindings"]),
            local_seed_reuse=data["local_seed_reuse"],
            registry_version=data["registry_version"],
        )
        if canonical_json(registry.to_dict()) != canonical_json(payload):
            raise ValueError("scheduled role registry is not canonical")
        return registry

    def _binding(self, role, *, stage_id, arm_id, round_number, update_index):
        scope = _scope(stage_id, arm_id, role, round_number)
        _nonnegative("update_index", update_index)
        binding = self._by_scope.get(scope)
        if binding is None:
            raise ValueError("scheduled role scope is not registered")
        if not binding.min_update <= update_index <= binding.max_update:
            raise ValueError("update_index is outside the scheduled role range")
        return binding

    def request_for(
        self, role: str, policy: PolicyView, *, stage_id: str, arm_id: int,
        round_number: int | None, update_index: int,
    ) -> EvaluationRequest:
        binding = self._binding(
            role, stage_id=stage_id, arm_id=arm_id,
            round_number=round_number, update_index=update_index,
        )
        delegated = binding.manifest.request_for(role, policy)
        return replace(delegated, metadata={
            **delegated.metadata,
            "scheduled_role_registry_hash": self.binding_hash(),
            "scheduled_role_scope": {
                "stage_id": stage_id,
                "arm_id": arm_id,
                "role": role,
                "round_number": round_number,
                "update_index": update_index,
            },
        })

    def _request_coordinates(self, request: EvaluationRequest) -> dict[str, Any]:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("an EvaluationRequest is required")
        scope = request.metadata.get("scheduled_role_scope")
        if not isinstance(scope, Mapping) or set(scope) != {
            "stage_id", "arm_id", "role", "round_number", "update_index",
        }:
            raise ValueError("scheduled role scope metadata mismatch")
        if scope["role"] != request.role:
            raise ValueError("scheduled role scope metadata mismatch")
        if request.metadata.get("scheduled_role_registry_hash") != self.binding_hash():
            raise ValueError("scheduled role registry binding mismatch")
        return {key: value for key, value in scope.items() if key != "role"}

    def validate_request_binding(self, request: EvaluationRequest) -> None:
        coordinates = self._request_coordinates(request)
        expected = self.request_for(
            request.role, PolicyView((0.0,), "request-binding"), **coordinates,
        ).to_dict()
        actual = request.to_dict()
        actual.pop("policy_fingerprint")
        expected.pop("policy_fingerprint")
        if canonical_json(actual) != canonical_json(expected):
            raise ValueError("scheduled role request binding mismatch")

    def validate_request(self, request: EvaluationRequest, policy: PolicyView) -> None:
        if not isinstance(policy, PolicyView):
            raise TypeError("a PolicyView is required")
        self.validate_request_binding(request)
        if request.policy_fingerprint != policy.fingerprint():
            raise ValueError("scheduled role policy binding mismatch")

    def validate_coordinates(
        self, request: EvaluationRequest, *, stage_id: str, arm_id: int,
        round_number: int | None, update_index: int,
    ) -> None:
        self.validate_request_binding(request)
        self._binding(
            request.role, stage_id=stage_id, arm_id=arm_id,
            round_number=round_number, update_index=update_index,
        )
        expected = {
            "stage_id": stage_id, "arm_id": arm_id,
            "round_number": round_number, "update_index": update_index,
        }
        if canonical_json(self._request_coordinates(request)) != canonical_json(expected):
            raise ValueError("scheduled role caller coordinates mismatch")


__all__ = [
    "SCHEDULED_ROLE_BINDING_SCHEMA",
    "SCHEDULED_ROLE_REGISTRY_SCHEMA",
    "ScheduledRoleBankRegistry",
    "ScheduledRoleBinding",
]
