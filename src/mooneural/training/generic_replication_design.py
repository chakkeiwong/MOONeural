"""Frozen identity binding for an independent replication lifecycle."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from types import MappingProxyType
from typing import Any

from .generic_training_contracts import (
    CheckpointState,
    canonical_json,
    stable_hash,
)

DESIGN_SCHEMA = "dsge_hmc.generic_replication_design.v1"
POPULATION_DESIGN_SCHEMA = "dsge_hmc.generic_replication_design.v2"


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


def _hashes(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must not be empty")
    return MappingProxyType({
        _nonempty(f"{name} key", key): _digest(name, value)
        for key, value in values.items()
    })


def _freeze(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    try:
        normalized = json.loads(canonical_json(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-safe") from error

    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(value) for key, value in item.items()})
        if isinstance(item, list):
            return tuple(freeze(value) for value in item)
        return item

    return freeze(normalized)


def _stage_payload(stage: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "stage_id", "from_round", "first_round", "last_round", "start_update",
        "updates_per_round", "population_size", "select_after",
    }
    if not isinstance(stage, Mapping) or set(stage) != required:
        raise ValueError("replication stage binding fields do not match")
    result = dict(stage)
    if not isinstance(result["stage_id"], str) or not result["stage_id"]:
        raise ValueError("replication stage ID is invalid")
    for name in ("from_round", "first_round", "last_round", "start_update"):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError("replication stage integer binding is invalid")
    if result["first_round"] != result["from_round"] + 1:
        raise ValueError("replication stage rounds must be contiguous")
    if result["last_round"] < result["first_round"]:
        raise ValueError("replication stage round range is invalid")
    for name in ("updates_per_round", "population_size"):
        if type(result[name]) is not int or result[name] <= 0:
            raise ValueError("replication stage size binding is invalid")
    if type(result["select_after"]) is not bool:
        raise ValueError("replication stage selection binding is invalid")
    return result


@dataclass(frozen=True)
class ReplicationDesignBinding:
    """All identity fields required to create or resume a real runner."""

    design_id: str
    master_sha256: str
    source_manifest_sha256: str
    generic_code_hashes: Mapping[str, str]
    environment_binding: Mapping[str, Any]
    backend_binding: Mapping[str, Any]
    role_manifest_sha256: str
    target_hashes: Mapping[str, str]
    initial_state_hashes: Mapping[str, str]
    task_ids: tuple[str, ...]
    threshold: float
    replica_ids: tuple[int, ...]
    expected_selected_replica: int | None
    selection_key: tuple[str, ...]
    stages: tuple[Mapping[str, Any], ...]
    optimizer_binding: Mapping[str, Any]
    permanent_pass_binding: Mapping[str, Any]
    seed_registry_sha256: str
    design_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_id", _nonempty("design ID", self.design_id))
        for name in ("master_sha256", "source_manifest_sha256", "role_manifest_sha256", "seed_registry_sha256"):
            object.__setattr__(self, name, _digest(name, getattr(self, name)))
        object.__setattr__(self, "generic_code_hashes", _hashes("generic code hashes", self.generic_code_hashes))
        object.__setattr__(self, "target_hashes", _hashes("target hashes", self.target_hashes))
        object.__setattr__(self, "initial_state_hashes", _hashes("initial state hashes", self.initial_state_hashes))
        object.__setattr__(self, "environment_binding", _freeze("environment binding", self.environment_binding))
        object.__setattr__(self, "backend_binding", _freeze("backend binding", self.backend_binding))
        object.__setattr__(self, "optimizer_binding", _freeze("optimizer binding", self.optimizer_binding))
        object.__setattr__(self, "permanent_pass_binding", _freeze("permanent-pass binding", self.permanent_pass_binding))
        task_ids = tuple(self.task_ids)
        if not task_ids or any(not isinstance(task, str) or not task for task in task_ids):
            raise ValueError("replication task IDs must be non-empty strings")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("replication task IDs must be unique")
        object.__setattr__(self, "task_ids", task_ids)
        if type(self.threshold) not in (int, float) or not math.isfinite(float(self.threshold)) or self.threshold <= 0:
            raise ValueError("replication threshold must be finite and positive")
        replicas = tuple(self.replica_ids)
        if not replicas or replicas != tuple(sorted(replicas)) or len(set(replicas)) != len(replicas):
            raise ValueError("replication IDs must be sorted and unique")
        if any(type(replica) is not int or replica < 0 for replica in replicas):
            raise ValueError("replication IDs must be non-negative integers")
        if self.expected_selected_replica is not None:
            if type(self.expected_selected_replica) is not int:
                raise ValueError("expected selected replica must be an integer")
            if self.expected_selected_replica not in replicas:
                raise ValueError("expected selected replica is not registered")
        if set(self.initial_state_hashes) != {str(replica) for replica in replicas}:
            raise ValueError("initial state hashes must cover every replica")
        object.__setattr__(self, "replica_ids", replicas)
        key = tuple(self.selection_key)
        if not key or any(not isinstance(item, str) or not item for item in key):
            raise ValueError("selection key must contain non-empty strings")
        object.__setattr__(self, "selection_key", key)
        stages = tuple(_stage_payload(stage) for stage in self.stages)
        if not stages:
            raise ValueError("replication stages must not be empty")
        if len({stage["stage_id"] for stage in stages}) != len(stages):
            raise ValueError("replication stage IDs must be unique")
        if self.expected_selected_replica is None:
            if key != ("absolute_threshold_ratio", "replica"):
                raise ValueError("population selection key is unsupported")
            if stages[0]["population_size"] != len(replicas):
                raise ValueError("population stage size does not match replica inventory")
            if not stages[0]["select_after"] or any(stage["select_after"] for stage in stages[1:]):
                raise ValueError("only the population stage may select final states")
        for previous, current in pairwise(stages):
            previous_stop = previous["start_update"] + (
                previous["last_round"] - previous["first_round"] + 1
            ) * previous["updates_per_round"]
            if (
                current["from_round"] != previous["last_round"]
                or current["start_update"] != previous_stop
                or current["population_size"] != 1
            ):
                raise ValueError("replication continuation stages are not contiguous")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "design_version", _nonempty("design version", self.design_version))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": POPULATION_DESIGN_SCHEMA if self.expected_selected_replica is None else DESIGN_SCHEMA,
            "design_version": self.design_version,
            "design_id": self.design_id,
            "master_sha256": self.master_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "generic_code_hashes": dict(self.generic_code_hashes),
            "environment_binding": dict(self.environment_binding),
            "backend_binding": dict(self.backend_binding),
            "role_manifest_sha256": self.role_manifest_sha256,
            "target_hashes": dict(self.target_hashes),
            "initial_state_hashes": dict(self.initial_state_hashes),
            "task_ids": list(self.task_ids),
            "threshold": float(self.threshold),
            "replica_ids": list(self.replica_ids),
            "expected_selected_replica": self.expected_selected_replica,
            "selection_key": list(self.selection_key),
            "stages": [dict(stage) for stage in self.stages],
            "optimizer_binding": dict(self.optimizer_binding),
            "permanent_pass_binding": dict(self.permanent_pass_binding),
            "seed_registry_sha256": self.seed_registry_sha256,
        }
        payload["design_sha256"] = stable_hash(payload)
        return json.loads(canonical_json(payload))

    def binding_hash(self) -> str:
        return self.to_dict()["design_sha256"]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReplicationDesignBinding:
        if not isinstance(payload, Mapping) or payload.get("schema") not in {DESIGN_SCHEMA, POPULATION_DESIGN_SCHEMA}:
            raise ValueError("unknown replication design schema")
        data = dict(payload)
        supplied = data.pop("design_sha256", None)
        if supplied != stable_hash(data):
            raise ValueError("replication design hash mismatch")
        schema = data.pop("schema")
        expected = {
            "design_version", "design_id", "master_sha256", "source_manifest_sha256",
            "generic_code_hashes", "environment_binding", "backend_binding",
            "role_manifest_sha256", "target_hashes", "initial_state_hashes", "task_ids",
            "threshold", "replica_ids", "expected_selected_replica", "selection_key",
            "stages", "optimizer_binding", "permanent_pass_binding", "seed_registry_sha256",
        }
        if set(data) != expected:
            raise ValueError("replication design fields do not match schema")
        if (schema == POPULATION_DESIGN_SCHEMA) != (data["expected_selected_replica"] is None):
            raise ValueError("replication design schema and selection mode mismatch")
        return cls(
            design_version=data.pop("design_version"),
            stages=tuple(data.pop("stages")),
            **data,
        )

    def bind_checkpoint(self, checkpoint: CheckpointState) -> CheckpointState:
        if not isinstance(checkpoint, CheckpointState):
            raise TypeError("a CheckpointState is required")
        metadata = dict(checkpoint.metadata)
        existing_hash = metadata.get("replication_design_sha256")
        existing_design = metadata.get("replication_design")
        if existing_hash is not None and existing_hash != self.binding_hash():
            raise ValueError("checkpoint belongs to a different replication design")
        if existing_design is not None and existing_design != self.to_dict():
            raise ValueError("checkpoint replication design payload mismatch")
        initial_state_hash = stable_hash(checkpoint.to_dict())
        existing_initial_hash = metadata.get("replication_initial_state_sha256")
        if existing_initial_hash is not None and existing_initial_hash != initial_state_hash:
            raise ValueError("checkpoint initial-state identity mismatch")
        metadata["replication_design_sha256"] = self.binding_hash()
        metadata["replication_design"] = self.to_dict()
        metadata["replication_initial_state_sha256"] = initial_state_hash
        return replace(checkpoint, metadata=metadata)

    def validate_checkpoint(self, checkpoint: CheckpointState) -> None:
        if checkpoint.metadata.get("replication_design_sha256") != self.binding_hash():
            raise ValueError("checkpoint replication design hash mismatch")
        if checkpoint.metadata.get("replication_design") != self.to_dict():
            raise ValueError("checkpoint replication design payload mismatch")
        initial_hash = checkpoint.metadata.get("replication_initial_state_sha256")
        if not isinstance(initial_hash, str) or len(initial_hash) != 64:
            raise ValueError("checkpoint initial-state identity is missing")
        try:
            int(initial_hash, 16)
        except ValueError as error:
            raise ValueError("checkpoint initial-state identity is invalid") from error

    def validate_stage_plan(self, stages: Sequence[Any]) -> None:
        actual = tuple({
            "stage_id": stage.stage_id,
            "from_round": stage.from_round,
            "first_round": stage.first_round,
            "last_round": stage.last_round,
            "start_update": stage.start_update,
            "updates_per_round": stage.updates_per_round,
            "population_size": stage.population_size,
            "select_after": stage.select_after,
        } for stage in stages)
        if actual != self.stages:
            raise ValueError("replication stage plan does not match frozen design")

    def validate_initial_states(self, states: Mapping[int, CheckpointState]) -> None:
        if set(states) != set(self.replica_ids):
            raise ValueError("initial state replica IDs do not match frozen design")
        for replica in self.replica_ids:
            if not isinstance(states[replica], CheckpointState):
                raise TypeError("initial states must contain CheckpointState entries")
            observed = stable_hash(states[replica].to_dict())
            if observed != self.initial_state_hashes[str(replica)]:
                raise ValueError("initial state identity does not match frozen design")


__all__ = ["DESIGN_SCHEMA", "POPULATION_DESIGN_SCHEMA", "ReplicationDesignBinding"]
