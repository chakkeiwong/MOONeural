"""Generic staged replication orchestration and crash-safe state storage."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from mooneural.artifacts import atomic_write_json

from .generic_control_evidence import (
    CONTROL_REFERENCE_PREFIX,
    ControlEvidenceArchive,
    ControlEvidenceError,
    full_control_hash,
)
from .generic_model_task_executor import ModelTaskExecutor
from .generic_permanent_pass import PermanentPassState
from .generic_permanent_pass_executor import PermanentPassUpdateError
from .generic_replication_design import ReplicationDesignBinding
from .generic_scheduled_role_banks import (
    SCHEDULED_ROLE_REGISTRY_SCHEMA,
    ScheduledRoleBankRegistry,
)
from .generic_training_contracts import (
    CertificationResult,
    CheckpointState,
    ControlEvaluation,
    ImmutableJSONMapping,
    PolicyView,
    TaskBatchTensors,
    ValidationEvaluation,
    canonical_json,
    stable_hash,
)


def validation_score(validation, task_ids, threshold):
    """Preserve the runner's minimax arithmetic and validation error semantics."""
    if set(validation.task_mean_mse) != set(task_ids):
        raise ReplicationRunnerError("validation task inventory mismatch")
    if validation.request is not None and validation.request.task_ids != task_ids:
        raise ReplicationRunnerError("validation request task order mismatch")
    return max(validation.task_mean_mse[task] / threshold for task in task_ids)


def select_validation_record(records):
    return min(records, key=lambda record: (record["absolute_threshold_ratio"], record["replica"]))


class ReplicationRunnerError(ValueError):
    """The replication transaction cannot proceed from the supplied state."""


class ReplicationEvaluationProvider(Protocol):
    def control(
        self, arm_id: int, policy: PolicyView, stage: ReplicationStage, round_number: int
    ) -> ControlEvaluation: ...

    def validation(
        self, arm_id: int, policy: PolicyView, stage: ReplicationStage, *, update_index: int
    ) -> ValidationEvaluation: ...

    def certification(
        self, arm_id: int, policy: PolicyView, stage: ReplicationStage, *, update_index: int
    ) -> CertificationResult: ...


@dataclass(frozen=True)
class ReplicationStage:
    stage_id: str
    from_round: int
    first_round: int
    last_round: int
    start_update: int
    updates_per_round: int = 300
    population_size: int = 1
    select_after: bool = False

    def __post_init__(self):
        if not self.stage_id or type(self.from_round) is not int or type(self.first_round) is not int:
            raise ValueError("replication stage identity is invalid")
        if self.first_round != self.from_round + 1 or self.last_round < self.first_round:
            raise ValueError("replication stage rounds must be contiguous")
        if type(self.start_update) is not int or self.start_update < 0:
            raise ValueError("replication stage update is invalid")
        if type(self.updates_per_round) is not int or self.updates_per_round <= 0:
            raise ValueError("replication updates_per_round must be positive")
        if type(self.population_size) is not int or self.population_size <= 0:
            raise ValueError("replication population size must be positive")
        if type(self.select_after) is not bool:
            raise TypeError("select_after must be boolean")

    @property
    def round_count(self):
        return self.last_round - self.first_round + 1

    @property
    def total_updates(self):
        return self.round_count * self.updates_per_round

    @property
    def stop_update(self):
        return self.start_update + self.total_updates


@dataclass(frozen=True)
class StoredTransaction:
    arm_id: int
    update_index: int
    state: CheckpointState
    event: Mapping[str, Any]
    kind: str


class AtomicCheckpointStore:
    """Write state first and a matching marker second; never overwrite either."""

    SCHEMA = "generic_neural_solver.replication_transaction.v3"
    OBJECT_SCHEMA = "generic_neural_solver.replication_transaction.v2"
    LEGACY_SCHEMA = "generic_neural_solver.replication_transaction.v1"
    MARKER_SCHEMA = "generic_neural_solver.replication_marker.v1"
    SELECTION_SCHEMA = "generic_neural_solver.replication_selection.v1"
    SELECTION_MARKER_SCHEMA = "generic_neural_solver.replication_selection_marker.v1"
    EVALUATION_SCHEMA = "generic_neural_solver.replication_evaluation.v1"
    EVALUATION_MARKER_SCHEMA = "generic_neural_solver.replication_evaluation_marker.v1"
    PUBLICATION_SCHEMA = "generic_neural_solver.replication_publication.v1"

    def __init__(
        self,
        root: Path,
        attempt_id: str,
        design: ReplicationDesignBinding | None = None,
    ):
        self.root = Path(root)
        self.attempt_id = attempt_id
        if not attempt_id:
            raise ValueError("attempt ID is required")
        if design is not None and not isinstance(design, ReplicationDesignBinding):
            raise TypeError("replication design binding is invalid")
        self.design = design
        self.control_evidence = ControlEvidenceArchive(self.root / "control_evidence")
        self._object_cache = {}

    def archive_control(self, control):
        return self.control_evidence.archive(control)

    def rehydrate_control(self, control):
        return self.control_evidence.rehydrate(control)

    def record_update_rejection(self, arm_id, previous, error):
        if error.checkpoint.to_dict() != previous.to_dict():
            raise ReplicationRunnerError("optional rejection did not preserve its original checkpoint")
        payload = {"schema": "generic_neural_solver.optional_update_rejection.v1",
            "arm_id": arm_id, "attempt_id": self.attempt_id,
            "update_index": previous.update_index, "parent_state_hash": stable_hash(previous.to_dict()),
            "message": str(error), "event": error.event}
        path = self.root / "rejections" / f"arm-{arm_id}" / f"{payload['parent_state_hash']}.json"
        receipt = {**payload, "receipt_hash": stable_hash(payload)}
        if path.exists():
            if json.loads(path.read_text()) != receipt:
                raise ReplicationRunnerError("optional rejection receipt conflicts with preserved evidence")
        else:
            atomic_write_json(path, receipt)
        return path

    def read_update_rejection(self, arm_id, state):
        parent_hash = stable_hash(state.to_dict())
        path = self.root / "rejections" / f"arm-{arm_id}" / f"{parent_hash}.json"
        if not path.exists():
            return None
        receipt = json.loads(path.read_text())
        digest = receipt.pop("receipt_hash", None)
        if (stable_hash(receipt) != digest or receipt.get("parent_state_hash") != parent_hash
                or receipt.get("arm_id") != arm_id or receipt.get("attempt_id") != self.attempt_id
                or receipt.get("update_index") != state.update_index
                or receipt.get("schema") != "generic_neural_solver.optional_update_rejection.v1"):
            raise ReplicationRunnerError("invalid optional rejection receipt")
        return receipt

    def _verify_control_history(self, state):
        policy = state.metadata.get("permanent_pass_rotation")
        if policy is not None:
            try:
                for entry in policy["history"]:
                    schema = entry["control"]["raw_records"].get("schema")
                    if isinstance(schema, str) and schema.startswith(CONTROL_REFERENCE_PREFIX):
                        self.control_evidence.verify(ControlEvaluation.from_dict(entry["control"]), use_cache=True)
            except ControlEvidenceError as error:
                raise ReplicationRunnerError(str(error)) from error

    def _validate_design_state(self, state: CheckpointState) -> None:
        if self.design is not None:
            self.design.validate_checkpoint(state)

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _directory(self, arm_id: int, kind: str, *, create=True) -> Path:
        if type(arm_id) is not int or arm_id < 0:
            raise ValueError("arm ID must be non-negative")
        if kind not in ("initial", "checkpoints", "boundaries", "stage_entries"):
            raise ValueError("unknown transaction kind")
        path = self.root / "arms" / f"arm-{arm_id}" / kind
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _store_object(self, value):
        object_hash = stable_hash(value)
        path = self.root / "objects" / f"{object_hash}.json"
        reference = {"object_hash": object_hash}
        if path.exists():
            self._load_object(reference)
        else:
            self._write_publication_json(path, value.to_dict() if isinstance(value, ImmutableJSONMapping) else value)
            if isinstance(value, ImmutableJSONMapping):
                stat = path.stat()
                self._object_cache[object_hash] = ((stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns), value)
        return reference

    def _load_object(self, reference):
        if not isinstance(reference, Mapping) or set(reference) != {"object_hash"}:
            raise ReplicationRunnerError("invalid checkpoint object reference")
        object_hash = reference["object_hash"]
        if not isinstance(object_hash, str) or len(object_hash) != 64 or any(
            character not in "0123456789abcdef" for character in object_hash
        ):
            raise ReplicationRunnerError("invalid checkpoint object hash")
        path = self.root / "objects" / f"{object_hash}.json"
        if not path.is_file():
            raise ReplicationRunnerError("checkpoint object is missing")
        stat = path.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._object_cache.get(object_hash)
        if cached is not None and cached[0] == signature:
            return cached[1]
        value = json.loads(path.read_text())
        if stable_hash(value) != object_hash:
            raise ReplicationRunnerError("checkpoint object content changed")
        if isinstance(value, dict):
            value = ImmutableJSONMapping(value)
            self._object_cache[object_hash] = (signature, value)
        return value

    def _encode_state(self, state):
        self._verify_control_history(state)
        payload = state.to_dict(shared=True)
        metadata = payload["metadata"]
        for name in ("task_registry", "role_manifest"):
            if name in metadata:
                metadata[name] = self._store_object(metadata[name])
        policy = metadata.get("permanent_pass_rotation")
        if policy is not None:
            history = [self._store_object(entry) for entry in policy["history"]]
            policy["history"] = self._store_object(history)
        return payload

    def _decode_state(self, payload):
        metadata = payload["metadata"]
        for name in ("task_registry", "role_manifest"):
            if name in metadata:
                metadata[name] = self._load_object(metadata[name])
        policy = metadata.get("permanent_pass_rotation")
        if policy is not None:
            history = self._load_object(policy["history"])
            if not isinstance(history, list):
                raise ReplicationRunnerError("checkpoint history object must be a list")
            policy["history"] = [self._load_object(entry) for entry in history]
        return CheckpointState.from_dict(payload)

    def _write_pair(
        self,
        path: Path,
        payload: Mapping[str, Any],
        marker_path: Path,
        marker: Mapping[str, Any],
        hash_field: str,
    ):
        if path.exists() or marker_path.exists():
            raise FileExistsError("replication transaction cannot be overwritten")
        intent_path = path.with_name(f".{path.name}.publication.json")
        intent = {
            "schema": self.PUBLICATION_SCHEMA,
            "attempt_id": self.attempt_id,
            "path": str(path.relative_to(self.root)),
            "payload_hash": stable_hash(payload),
            "marker": dict(marker),
            "hash_field": hash_field,
        }
        if intent_path.exists():
            if json.loads(intent_path.read_text()) != intent:
                raise ReplicationRunnerError("prepared replication publication changed")
        else:
            self._write_publication_json(intent_path, intent)
        self._write_publication_json(path, payload)
        marker_payload = dict(marker)
        marker_payload[hash_field] = self._hash(path)
        self._write_publication_json(marker_path, marker_payload)

    @staticmethod
    def _write_publication_json(path, payload):
        atomic_write_json(path, payload)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _resume_publications(self, directory, evaluation_validator=None, *, read_only=False):
        for intent_path in sorted(directory.glob(".*.publication.json")):
            path = directory / intent_path.name[1:].removesuffix(".publication.json")
            marker_path = path.with_suffix(".complete.json")
            if marker_path.exists():
                if not path.is_file():
                    raise ReplicationRunnerError("orphaned publication marker without payload")
                continue
            if read_only:
                raise ReplicationRunnerError("partial parent has an unresolved publication intent")
            intent = json.loads(intent_path.read_text())
            if (
                intent.get("schema") != self.PUBLICATION_SCHEMA
                or intent.get("attempt_id") != self.attempt_id
                or intent.get("path") != str(path.relative_to(self.root))
                or intent.get("hash_field") not in {"state_sha256", "selection_sha256", "evaluation_sha256"}
                or intent.get("marker", {}).get("attempt_id") != self.attempt_id
            ):
                raise ReplicationRunnerError("replication publication binding mismatch")
            if not path.exists():
                continue
            if stable_hash(json.loads(path.read_text())) != intent.get("payload_hash"):
                raise ReplicationRunnerError("replication publication payload changed")
            marker = dict(intent["marker"])
            marker[intent["hash_field"]] = self._hash(path)
            if intent["hash_field"] == "state_sha256":
                self._read_marker(marker)
            elif intent["hash_field"] == "evaluation_sha256":
                if evaluation_validator is None:
                    raise ReplicationRunnerError("evaluation recovery requires its stage validator")
                evaluation_validator(self._read_evaluation_marker(marker))
            self._write_publication_json(marker_path, marker)

    @staticmethod
    def _raw_files(directory: Path, prefix: str) -> list[Path]:
        return sorted(
            path for path in directory.glob(f"{prefix}*.json")
            if not path.name.endswith(".complete.json")
        )

    def has_initial(self, arm_id: int) -> bool:
        directory = self._directory(arm_id, "initial")
        self._resume_publications(directory)
        return (directory / "state.complete.json").is_file()

    def initialize(self, arm_id: int, state: CheckpointState) -> StoredTransaction:
        if state.attempt_id != self.attempt_id:
            raise ReplicationRunnerError("initial state attempt mismatch")
        self._validate_design_state(state)
        payload = {
            "schema": self.SCHEMA,
            "kind": "initial",
            "attempt_id": self.attempt_id,
            "arm_id": arm_id,
            "state": self._encode_state(state),
            "event": {"event": "initialization", "update_index": state.update_index},
            "replication_design_sha256": None
            if self.design is None
            else self.design.binding_hash(),
        }
        payload["transaction_hash"] = stable_hash(payload)
        directory = self._directory(arm_id, "initial")
        path = directory / "state.json"
        marker_path = directory / "state.complete.json"
        marker = {
            "schema": self.MARKER_SCHEMA,
            "kind": "initial",
            "attempt_id": self.attempt_id,
            "arm_id": arm_id,
            "state_path": str(path.relative_to(self.root)),
            "state_sha256": "",
            "transaction_hash": payload["transaction_hash"],
            "replication_design_sha256": payload["replication_design_sha256"],
            "complete": True,
        }
        self._write_pair(path, payload, marker_path, marker, "state_sha256")
        return StoredTransaction(arm_id, state.update_index, state, payload["event"], "initial")

    def _commit(self, arm_id: int, state: CheckpointState, event: Mapping[str, Any], kind: str):
        if state.attempt_id != self.attempt_id:
            raise ReplicationRunnerError("transaction attempt mismatch")
        self._validate_design_state(state)
        update_index = state.update_index
        payload = {
            "schema": self.SCHEMA,
            "kind": kind,
            "attempt_id": self.attempt_id,
            "arm_id": arm_id,
            "state": self._encode_state(state),
            "event": dict(event),
            "replication_design_sha256": None
            if self.design is None
            else self.design.binding_hash(),
        }
        payload["transaction_hash"] = stable_hash(payload)
        directory = self._directory(arm_id, {
            "update": "checkpoints", "boundary": "boundaries", "stage_entry": "stage_entries",
        }[kind])
        stem = f"{kind}-{update_index:08d}"
        path = directory / f"{stem}.json"
        marker_path = directory / f"{stem}.complete.json"
        marker = {
            "schema": self.MARKER_SCHEMA,
            "kind": kind,
            "attempt_id": self.attempt_id,
            "arm_id": arm_id,
            "update_index": update_index,
            "state_path": str(path.relative_to(self.root)),
            "state_sha256": "",
            "transaction_hash": payload["transaction_hash"],
            "replication_design_sha256": payload["replication_design_sha256"],
            "complete": True,
        }
        self._write_pair(path, payload, marker_path, marker, "state_sha256")
        return StoredTransaction(arm_id, update_index, state, event, kind)

    def commit_update(self, arm_id: int, previous: CheckpointState, state: CheckpointState, event):
        if state.update_index != previous.update_index + 1:
            raise ReplicationRunnerError("update transaction must advance exactly once")
        return self._commit(arm_id, state, event, "update")

    def commit_boundary(self, arm_id: int, state: CheckpointState, event):
        return self._commit(arm_id, state, event, "boundary")

    def commit_stage_entry(self, arm_id, previous, state, event):
        parent = self._stage_entry_parent(arm_id, state.update_index)
        if previous.to_dict() != parent.to_dict():
            raise ReplicationRunnerError("stage-entry parent differs from the committed boundary")
        self._validate_stage_entry(arm_id, parent, state, event)
        return self._commit(arm_id, state, event, "stage_entry")

    def _stage_entry_parent(self, arm_id, update_index):
        path = self._directory(arm_id, "boundaries", create=False) / f"boundary-{update_index:08d}.complete.json"
        if not path.is_file():
            raise ReplicationRunnerError("stage entry requires a committed parent boundary")
        return self._read(path, arm_id=arm_id, kind="boundary").state

    def _validate_stage_entry(self, arm_id, previous, state, event):
        if replace(state, metadata=previous.metadata).to_dict() != previous.to_dict():
            raise ReplicationRunnerError("stage entry cannot change optimizer or update state")
        if set(event) != {
            "event", "stage_id", "stage_ordinal", "arm_id", "round", "boundary",
            "update_index", "parent_state_sha256", "control_sha256",
        } or event.get("event") != "stage_entry" or not isinstance(event.get("stage_id"), str) or not event["stage_id"]:
            raise ReplicationRunnerError("stage-entry event identity is required")
        if any(type(event[name]) is not int for name in (
            "stage_ordinal", "arm_id", "round", "boundary", "update_index",
        )) or (
            event["stage_ordinal"] < 1 or event["arm_id"] != arm_id
            or event["round"] < 0 or event["boundary"] != 0
            or event["update_index"] != state.update_index
        ):
            raise ReplicationRunnerError("stage-entry event coordinates are invalid")
        if event.get("parent_state_sha256") != stable_hash(previous.to_dict()):
            raise ReplicationRunnerError("stage-entry parent state binding mismatch")
        old_metadata, new_metadata = dict(previous.metadata), dict(state.metadata)
        try:
            prior_policy = PermanentPassState.from_dict(old_metadata.pop("permanent_pass_rotation"))
            next_policy = PermanentPassState.from_dict(new_metadata.pop("permanent_pass_rotation"))
        except (KeyError, TypeError, ValueError) as error:
            raise ReplicationRunnerError("stage entry requires valid permanent-pass histories") from error
        if canonical_json(old_metadata) != canonical_json(new_metadata):
            raise ReplicationRunnerError("stage entry cannot rewrite unrelated metadata")
        if (
            prior_policy.complete or prior_policy.update_index != state.update_index
            or prior_policy.controls[-1].update_index != state.update_index
            or next_policy.update_index != state.update_index
            or prior_policy.task_ids != next_policy.task_ids
            or prior_policy.threshold != next_policy.threshold
            or len(next_policy.controls) != len(prior_policy.controls) + 1
            or next_policy.controls[:-1] != prior_policy.controls
            or next_policy.controls[-1].stage_id != event["stage_id"]
        ):
            raise ReplicationRunnerError("stage entry must append exactly one bound control to prior history")
        control = next_policy.controls[-1].evaluation
        policy = PolicyView.from_dict(state.policy_state)
        if event["control_sha256"] != full_control_hash(control) or control.request.policy_fingerprint != policy.fingerprint():
            raise ReplicationRunnerError("stage-entry control identity mismatch")
        manifest = old_metadata.get("role_manifest", {})
        if manifest.get("schema") == SCHEDULED_ROLE_REGISTRY_SCHEMA:
            try:
                roles = ScheduledRoleBankRegistry.from_dict(manifest)
                roles.validate_request(control.request, policy)
                roles.validate_coordinates(
                    control.request, stage_id=event["stage_id"], arm_id=arm_id,
                    round_number=event["round"], update_index=state.update_index,
                )
            except (TypeError, ValueError) as error:
                raise ReplicationRunnerError("stage-entry role binding is invalid") from error
        if self.design is not None:
            ordinal = event["stage_ordinal"]
            if ordinal >= len(self.design.stages):
                raise ReplicationRunnerError("stage-entry ordinal is outside the frozen design")
            stage = self.design.stages[ordinal]
            if (event["stage_id"], event["round"], state.update_index) != (
                stage["stage_id"], stage["from_round"], stage["start_update"],
            ):
                raise ReplicationRunnerError("stage-entry coordinates differ from the frozen design")
            requirement = self.design.permanent_pass_binding.get("stage_entries", {}).get(stage["stage_id"])
            if requirement is not None and (
                tuple(requirement["permanent_tasks"]) != prior_policy.permanent
                or requirement["preferred_method"] != previous.method_state.get("preferred")
            ):
                raise ReplicationRunnerError("inherited stage-entry membership or method mismatch")

    def _read(self, marker_path: Path, *, arm_id: int | None = None, kind: str | None = None) -> StoredTransaction:
        marker = json.loads(marker_path.read_text())
        return self._read_marker(marker, arm_id=arm_id, kind=kind)

    def _read_marker(self, marker, *, arm_id=None, kind=None):
        if marker.get("schema") != self.MARKER_SCHEMA or marker.get("complete") is not True:
            raise ReplicationRunnerError("invalid replication completion marker")
        expected_design = None if self.design is None else self.design.binding_hash()
        if marker.get("replication_design_sha256") != expected_design:
            raise ReplicationRunnerError("replication transaction design binding mismatch")
        state_path = Path(marker.get("state_path", ""))
        if state_path.is_absolute() or ".." in state_path.parts:
            raise ReplicationRunnerError("replication state path escapes attempt root")
        path = self.root / state_path
        if not path.is_file() or self._hash(path) != marker["state_sha256"]:
            raise ReplicationRunnerError("replication marker/state hash mismatch")
        payload = json.loads(path.read_text())
        if payload.get("schema") not in {self.SCHEMA, self.OBJECT_SCHEMA, self.LEGACY_SCHEMA} or stable_hash({
            key: value for key, value in payload.items() if key != "transaction_hash"
        }) != payload.get("transaction_hash"):
            raise ReplicationRunnerError("replication transaction hash mismatch")
        if payload.get("replication_design_sha256") != expected_design:
            raise ReplicationRunnerError("replication transaction design binding mismatch")
        if (payload.get("attempt_id"), payload.get("arm_id"), payload.get("kind")) != (
            self.attempt_id, marker.get("arm_id"), marker.get("kind")
        ) or (arm_id is not None and marker.get("arm_id") != arm_id) or (
            kind is not None and marker.get("kind") != kind
        ):
            raise ReplicationRunnerError("replication transaction binding mismatch")
        state = (
            self._decode_state(payload["state"])
            if payload["schema"] != self.LEGACY_SCHEMA
            else CheckpointState.from_dict(payload["state"])
        )
        self._validate_design_state(state)
        self._verify_control_history(state)
        if marker.get("update_index", state.update_index) != state.update_index:
            raise ReplicationRunnerError("replication transaction update mismatch")
        if marker.get("transaction_hash") != payload["transaction_hash"]:
            raise ReplicationRunnerError("replication marker transaction mismatch")
        if payload["kind"] == "stage_entry":
            parent = self._stage_entry_parent(int(marker["arm_id"]), state.update_index)
            self._validate_stage_entry(int(marker["arm_id"]), parent, state, payload["event"])
        return StoredTransaction(
            int(marker["arm_id"]), state.update_index, state, payload["event"], payload["kind"]
        )

    def recover(
        self,
        arm_id: int,
        *,
        through_update_index: int | None = None,
        include_stage_entry_at_cutoff: bool = True,
    ) -> tuple[CheckpointState, tuple[Mapping[str, Any], ...]]:
        return self._read_committed(
            arm_id, through_update_index=through_update_index,
            include_stage_entry_at_cutoff=include_stage_entry_at_cutoff,
        )

    def read_latest_committed(self, arm_id):
        """Validate the complete committed chain without creating or finishing files."""
        return self._read_committed(arm_id, read_only=True)

    def _read_committed(
        self,
        arm_id,
        *,
        through_update_index=None,
        include_stage_entry_at_cutoff=True,
        read_only: bool = False,
    ) -> tuple[CheckpointState, tuple[Mapping[str, Any], ...]]:
        if through_update_index is not None and (
            type(through_update_index) is not int or through_update_index < 0
        ):
            raise ValueError("recovery update cutoff must be a non-negative integer")
        self.control_evidence.begin_verification()
        initial_directory = self._directory(arm_id, "initial", create=not read_only)
        self._resume_publications(initial_directory, read_only=read_only)
        initial_markers = sorted(initial_directory.glob("*.complete.json"))
        if initial_markers != [initial_directory / "state.complete.json"]:
            raise ReplicationRunnerError("exactly one complete initial transaction is required")
        current = self._read(initial_markers[0], arm_id=arm_id, kind="initial")
        events = [current.event]
        checkpoints = self._directory(arm_id, "checkpoints", create=not read_only)
        boundaries = self._directory(arm_id, "boundaries", create=not read_only)
        stage_entries = self._directory(arm_id, "stage_entries", create=not read_only)
        self._resume_publications(checkpoints, read_only=read_only)
        self._resume_publications(boundaries, read_only=read_only)
        self._resume_publications(stage_entries, read_only=read_only)
        checkpoint_files = self._raw_files(checkpoints, "update-")
        marker_files = sorted(checkpoints.glob("update-*.complete.json"))
        checkpoint_names = {path.name.removesuffix(".json") for path in checkpoint_files}
        marker_names = {path.name.removesuffix(".complete.json") for path in marker_files}
        if checkpoint_names != marker_names:
            raise ReplicationRunnerError("orphaned update checkpoint or marker")
        for marker_path in marker_files:
            transaction = self._read(marker_path, arm_id=arm_id, kind="update")
            if through_update_index is not None and transaction.update_index > through_update_index:
                break
            if transaction.kind != "update" or transaction.update_index != current.update_index + 1:
                raise ReplicationRunnerError("non-contiguous update recovery")
            current = transaction
            events.append(transaction.event)
            boundary_path = boundaries / f"boundary-{current.update_index:08d}.complete.json"
            if boundary_path.exists():
                boundary = self._read(boundary_path, arm_id=arm_id, kind="boundary")
                if boundary.kind != "boundary" or boundary.update_index != current.update_index:
                    raise ReplicationRunnerError("boundary recovery mismatch")
                current = boundary
                events.append(boundary.event)
            entry_path = stage_entries / f"stage_entry-{current.update_index:08d}.complete.json"
            if entry_path.exists() and (
                include_stage_entry_at_cutoff or current.update_index != through_update_index
            ):
                entry = self._read(entry_path, arm_id=arm_id, kind="stage_entry")
                if (
                    current.kind != "boundary"
                    or replace(entry.state, metadata=current.state.metadata).to_dict() != current.state.to_dict()
                    or entry.event.get("event") != "stage_entry"
                    or not entry.event.get("stage_id")
                    or entry.event.get("parent_state_sha256") != stable_hash(current.state.to_dict())
                ):
                    raise ReplicationRunnerError("stage-entry recovery mismatch")
                current = entry
                events.append(entry.event)
        boundary_files = self._raw_files(boundaries, "boundary-")
        boundary_markers = sorted(boundaries.glob("boundary-*.complete.json"))
        boundary_names = {path.name.removesuffix(".json") for path in boundary_files}
        boundary_marker_names = {
            path.name.removesuffix(".complete.json") for path in boundary_markers
        }
        update_stems = checkpoint_names
        if boundary_names != boundary_marker_names or any(
            f"update-{path.name.removesuffix('.complete.json').removeprefix('boundary-')}"
            not in update_stems
            for path in boundary_markers
        ):
            raise ReplicationRunnerError("orphaned boundary checkpoint or marker")
        entry_names = {
            path.name.removesuffix(".json") for path in self._raw_files(stage_entries, "stage_entry-")
        }
        entry_markers = sorted(stage_entries.glob("stage_entry-*.complete.json"))
        if entry_names != {
            path.name.removesuffix(".complete.json") for path in entry_markers
        } or any(
            f"boundary-{path.name.removesuffix('.complete.json').removeprefix('stage_entry-')}"
            not in boundary_names
            for path in entry_markers
        ):
            raise ReplicationRunnerError("orphaned stage-entry checkpoint or marker")
        return current.state, tuple(events)

    def commit_selection(self, selection: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.design is not None and selection.get("replication_design_sha256") != self.design.binding_hash():
            raise ReplicationRunnerError("replication selection design binding mismatch")
        payload = {
            "schema": self.SELECTION_SCHEMA,
            "attempt_id": self.attempt_id,
            "selection": dict(selection),
            "replication_design_sha256": None
            if self.design is None
            else self.design.binding_hash(),
        }
        payload["selection_hash"] = stable_hash(payload)
        path = self.root / "selection.json"
        marker_path = self.root / "selection.complete.json"
        marker = {
            "schema": self.SELECTION_MARKER_SCHEMA,
            "attempt_id": self.attempt_id,
            "selection_path": str(path.relative_to(self.root)),
            "selection_sha256": "",
            "selection_hash": payload["selection_hash"],
            "replication_design_sha256": payload["replication_design_sha256"],
            "complete": True,
        }
        self._write_pair(path, payload, marker_path, marker, "selection_sha256")
        return dict(selection)

    def read_selection(self) -> Mapping[str, Any] | None:
        self._resume_publications(self.root)
        path = self.root / "selection.json"
        marker_path = self.root / "selection.complete.json"
        if not path.exists() and not marker_path.exists():
            return None
        if not path.is_file() or not marker_path.is_file():
            raise ReplicationRunnerError("orphaned selection or marker")
        marker = json.loads(marker_path.read_text())
        expected_design = None if self.design is None else self.design.binding_hash()
        if (
            marker.get("schema") != self.SELECTION_MARKER_SCHEMA
            or marker.get("attempt_id") != self.attempt_id
            or marker.get("complete") is not True
            or marker.get("selection_path") != "selection.json"
            or self._hash(path) != marker.get("selection_sha256")
            or marker.get("replication_design_sha256") != expected_design
        ):
            raise ReplicationRunnerError("invalid replication selection marker")
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") != self.SELECTION_SCHEMA
            or payload.get("attempt_id") != self.attempt_id
            or stable_hash({key: value for key, value in payload.items() if key != "selection_hash"})
            != payload.get("selection_hash")
            or marker.get("selection_hash") != payload.get("selection_hash")
            or payload.get("replication_design_sha256") != expected_design
        ):
            raise ReplicationRunnerError("replication selection hash mismatch")
        selection = payload.get("selection")
        if not isinstance(selection, Mapping):
            raise ReplicationRunnerError("replication selection payload is invalid")
        return dict(selection)

    def _evaluation_path(self, stage_ordinal, arm_id, role):
        if any(type(value) is not int or value < 0 for value in (stage_ordinal, arm_id)):
            raise ReplicationRunnerError("evaluation stage and arm must be nonnegative integers")
        if role not in {"validation", "certification"}:
            raise ReplicationRunnerError("unknown stage-exit evaluation role")
        return self.root / "evaluations" / f"stage-{stage_ordinal}" / f"arm-{arm_id}" / role / "result.json"

    def commit_evaluation(self, record, validator):
        validator(record)
        path = self._evaluation_path(record["stage_ordinal"], record["arm_id"], record["role"])
        payload = {
            "schema": self.EVALUATION_SCHEMA,
            "attempt_id": self.attempt_id,
            "replication_design_sha256": None if self.design is None else self.design.binding_hash(),
            "record": dict(record),
        }
        payload["evaluation_hash"] = stable_hash(payload)
        marker = {
            "schema": self.EVALUATION_MARKER_SCHEMA,
            "attempt_id": self.attempt_id,
            "evaluation_path": str(path.relative_to(self.root)),
            "evaluation_hash": payload["evaluation_hash"],
            "replication_design_sha256": payload["replication_design_sha256"],
            "complete": True,
        }
        self._write_pair(path, payload, path.with_suffix(".complete.json"), marker, "evaluation_sha256")

    def _read_evaluation_marker(self, marker):
        expected_design = None if self.design is None else self.design.binding_hash()
        relative = marker.get("evaluation_path")
        if (
            marker.get("schema") != self.EVALUATION_MARKER_SCHEMA
            or marker.get("attempt_id") != self.attempt_id
            or marker.get("replication_design_sha256") != expected_design
            or marker.get("complete") is not True
            or not isinstance(relative, str)
        ):
            raise ReplicationRunnerError("invalid stage evaluation marker")
        path = self.root / relative
        if not path.is_file() or self._hash(path) != marker.get("evaluation_sha256"):
            raise ReplicationRunnerError("stage evaluation payload hash mismatch")
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") != self.EVALUATION_SCHEMA
            or payload.get("attempt_id") != self.attempt_id
            or payload.get("replication_design_sha256") != expected_design
            or stable_hash({key: value for key, value in payload.items() if key != "evaluation_hash"})
            != payload.get("evaluation_hash")
            or marker.get("evaluation_hash") != payload.get("evaluation_hash")
        ):
            raise ReplicationRunnerError("stage evaluation binding mismatch")
        record = payload.get("record")
        if not isinstance(record, Mapping) or path != self._evaluation_path(
            record.get("stage_ordinal"), record.get("arm_id"), record.get("role")
        ):
            raise ReplicationRunnerError("stage evaluation path mismatch")
        return record

    def read_evaluation(self, stage_ordinal, arm_id, role, validator):
        path = self._evaluation_path(stage_ordinal, arm_id, role)
        self._resume_publications(path.parent, evaluation_validator=validator)
        marker_path = path.with_suffix(".complete.json")
        if not path.exists() and not marker_path.exists():
            return None
        if not path.is_file() or not marker_path.is_file():
            raise ReplicationRunnerError("orphaned stage evaluation or marker")
        marker = json.loads(marker_path.read_text())
        if marker.get("evaluation_path") != str(path.relative_to(self.root)):
            raise ReplicationRunnerError("stage evaluation marker path mismatch")
        record = self._read_evaluation_marker(marker)
        validator(record)
        return record


@dataclass(frozen=True)
class ReplicationRunResult:
    states: Mapping[int, CheckpointState]
    selection: Mapping[str, Any] | None
    events: tuple[Mapping[str, Any], ...]
    evaluations: tuple[Mapping[str, Any], ...] = ()


class GenericReplicationRunner:
    """Run population and inherited-continuation stages through the shared core."""

    def __init__(
        self,
        executor: ModelTaskExecutor,
        adapters: Mapping[int, Any],
        initial_states: Mapping[int, CheckpointState],
        evaluation_provider: ReplicationEvaluationProvider,
        batch_factory: Callable[[int, PolicyView, ReplicationStage, int], TaskBatchTensors],
        store: AtomicCheckpointStore,
        stages: Sequence[ReplicationStage],
        *,
        expected_selected_replica: int | None = None,
        threshold: float = 0.04,
        design: ReplicationDesignBinding | None = None,
        lifecycle_endpoint: str = "certification",
        budget_cpu_seconds: Callable[[], float] | None = None,
    ):
        if not stages or stages[0].population_size != len(initial_states):
            raise ValueError("replication stages and initial population mismatch")
        if set(adapters) != set(initial_states):
            raise ValueError("replication adapter/state IDs mismatch")
        if len({id(adapter) for adapter in adapters.values()}) != len(adapters):
            raise ValueError("replication adapters must be distinct per replica")
        if len({stable_hash(state.to_dict()) for state in initial_states.values()}) != len(initial_states):
            raise ValueError("replication initial states must be distinct per replica")
        if stages[0].select_after is not True:
            raise ValueError("the population stage must select its final states")
        if any(stage.select_after for stage in stages[1:]):
            raise ValueError("only the population stage may perform selection")
        if expected_selected_replica is not None and (
            type(expected_selected_replica) is not int or expected_selected_replica not in initial_states
        ):
            raise ValueError("expected selected replica is not in the population")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("replication threshold must be finite and positive")
        if design is not None and not isinstance(design, ReplicationDesignBinding):
            raise TypeError("replication design binding is invalid")
        if isinstance(executor, ModelTaskExecutor) and design is None:
            raise ValueError("real ModelTaskExecutor replication requires a frozen design")
        if lifecycle_endpoint not in ("certification", "validation-only"):
            raise ReplicationRunnerError("unknown replication lifecycle endpoint")
        if lifecycle_endpoint == "validation-only" and (design is None or len(stages) != 1):
            raise ReplicationRunnerError("validation-only requires one stage and a frozen design")
        if design is not None and design.permanent_pass_binding.get("lifecycle_endpoint", "certification") != lifecycle_endpoint:
            raise ReplicationRunnerError("replication lifecycle endpoint differs from design")
        if lifecycle_endpoint == "certification" and not callable(getattr(evaluation_provider, "certification", None)):
            raise ReplicationRunnerError("stage-exit certification provider is required")
        if design is not None:
            if design.threshold != float(threshold):
                raise ValueError("replication design threshold mismatch")
            if expected_selected_replica is not None and design.expected_selected_replica != expected_selected_replica:
                raise ValueError("replication design selected replica mismatch")
            expected_selected_replica = design.expected_selected_replica
            if design.replica_ids != tuple(sorted(initial_states)):
                raise ValueError("replication design replica inventory mismatch")
            if tuple(executor.metadata.registry.task_ids) != design.task_ids:
                raise ValueError("replication design task inventory mismatch")
            design.validate_stage_plan(stages)
            design.validate_initial_states(initial_states)
            if store.design is None or store.design.binding_hash() != design.binding_hash():
                raise ValueError("checkpoint store is not bound to the replication design")
            coordinator = getattr(getattr(executor, "core", None), "coordinator", None)
            roles = getattr(coordinator, "roles", None)
            if roles is None or not callable(getattr(roles, "validate_request", None)):
                raise ValueError("replication design requires a role-bound coordinator")
            if isinstance(executor, ModelTaskExecutor) and not isinstance(roles, ScheduledRoleBankRegistry):
                raise ValueError("real replication requires scheduled materialized role banks")
            if callable(getattr(roles, "binding_hash", None)) and (
                roles.binding_hash() != design.role_manifest_sha256
            ):
                raise ValueError("replication design role manifest mismatch")
        for previous, current in pairwise(stages):
            if (
                current.from_round != previous.last_round
                or current.start_update != previous.stop_update
                or current.population_size != 1
            ):
                raise ValueError("replication continuation stages are not contiguous")
        self.executor = executor
        self.adapters = dict(adapters)
        self.states = dict(initial_states)
        self.evaluation_provider = evaluation_provider
        self.batch_factory = batch_factory
        self.store = store
        self.stages = tuple(stages)
        self.stage_entry_requirements = None if design is None else design.permanent_pass_binding.get("stage_entries")
        if isinstance(executor, ModelTaskExecutor) and self.stage_entry_requirements is None:
            raise ValueError("real replication requires frozen stage-entry preconditions")
        if self.stage_entry_requirements is not None:
            if not isinstance(self.stage_entry_requirements, Mapping) or set(self.stage_entry_requirements) != {
                stage.stage_id for stage in stages[1:]
            }:
                raise ValueError("stage-entry preconditions must cover every continuation stage")
            for requirement in self.stage_entry_requirements.values():
                if not isinstance(requirement, Mapping) or set(requirement) != {"permanent_tasks", "preferred_method"}:
                    raise ValueError("stage-entry precondition fields are invalid")
                permanent = tuple(requirement["permanent_tasks"])
                if permanent != tuple(task for task in executor.metadata.registry.task_ids if task in permanent):
                    raise ValueError("stage-entry permanent task inventory is invalid")
                if not isinstance(requirement["preferred_method"], str) or not requirement["preferred_method"]:
                    raise ValueError("stage-entry preferred method is required")
        self.expected_selected_replica = expected_selected_replica
        self.threshold = threshold
        self.design = design
        self.lifecycle_endpoint = lifecycle_endpoint
        self.budget_stop_policy = None if design is None else design.permanent_pass_binding.get("budget_stop")
        self.budget_cpu_seconds = budget_cpu_seconds
        self.budget_stop_record = None
        self._last_budget_cpu = None
        if self.budget_stop_policy is None:
            if budget_cpu_seconds is not None:
                raise ReplicationRunnerError("budget callback requires a frozen budget-stop policy")
        else:
            policy = self.budget_stop_policy
            if (len(stages) != 1 or len(initial_states) != 1 or not callable(budget_cpu_seconds)
                    or not isinstance(policy, Mapping)
                    or set(policy) != {"training_cpu_limit", "boundary_cpu_reserve"}
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                           for value in policy.values())
                    or policy["boundary_cpu_reserve"] >= policy["training_cpu_limit"]):
                raise ReplicationRunnerError("budget stop requires one stage/arm and a positive frozen CPU limit/reserve")
        self.checkpoint_continuation = None if design is None else design.permanent_pass_binding.get("checkpoint_continuation")
        if self.checkpoint_continuation is not None:
            if (len(initial_states) != 1 or len(stages) != 1
                    or self.checkpoint_continuation.get("start_update") != stages[0].start_update):
                raise ReplicationRunnerError("checkpoint continuation requires one clock-bound initial state and stage")
            for state in initial_states.values():
                if canonical_json(state.metadata.get("checkpoint_continuation")) != canonical_json(self.checkpoint_continuation):
                    raise ReplicationRunnerError("checkpoint continuation provenance differs from design")
                executor.core._validate_numerical_state(state)
                executor.core.coordinator.read(state)

    def _candidate_union_description(self):
        if self.expected_selected_replica is not None and self.stages[0].population_size == 5:
            return "five final worker states"
        return "all final worker states"

    def _initialize(self, stage: ReplicationStage) -> tuple[Mapping[int, CheckpointState], list[Mapping[str, Any]]]:
        states = {}
        events = []
        for arm_id in sorted(self.states):
            state = self.states[arm_id]
            if state.update_index != stage.start_update:
                raise ReplicationRunnerError("initial state does not match stage update")
            if self.design is not None:
                state = self.design.bind_checkpoint(state)
                self.states[arm_id] = state
            if self.store.has_initial(arm_id):
                recovered, _recovered_events = self.store.recover(
                    arm_id, through_update_index=stage.stop_update,
                    include_stage_entry_at_cutoff=False,
                )
                if recovered.attempt_id != state.attempt_id:
                    raise ReplicationRunnerError("recovered initial state attempt mismatch")
                if self.design is not None:
                    expected_initial_hash = self.design.initial_state_hashes[str(arm_id)]
                    if recovered.metadata.get("replication_initial_state_sha256") != expected_initial_hash:
                        raise ReplicationRunnerError("recovered initial state identity mismatch")
                states[arm_id] = recovered
                continue
            partial = (self.checkpoint_continuation or {}).get("partial_parent_continuation")
            if partial is not None:
                from .generic_finite_objective_runner import _partial_parent_state

                _partial_parent_state(self.checkpoint_continuation["parent"], partial)
            policy = PolicyView.from_dict(state.policy_state)
            control = self.evaluation_provider.control(arm_id, policy, stage, stage.from_round)
            self._validate_control_coordinates(control, arm_id, state, stage, stage.from_round)
            control = self.store.archive_control(control)
            if self.checkpoint_continuation is None:
                initialized = self.executor.initialize(state, control)
            else:
                initialized, _entry = self.executor.core.coordinator.stage_entry(state, control, stage_id=stage.stage_id)
            self.store.initialize(arm_id, initialized)
            states[arm_id] = initialized
            events.append({"event": "initialization", "arm_id": arm_id, "update_index": initialized.update_index})
        return states, events

    def _last_kind(self, arm_id: int, through_update_index: int | None = None) -> str:
        _state, events = self.store.recover(
            arm_id, through_update_index=through_update_index,
            include_stage_entry_at_cutoff=False,
        )
        if not events:
            raise ReplicationRunnerError("replication recovery has no transaction events")
        return str(events[-1].get("event"))

    def _stage_done(self, state: CheckpointState, stage: ReplicationStage, last_kind: str) -> bool:
        if self.budget_stop_record is not None:
            record = self.budget_stop_record
            try:
                checkpoint_kind = self._budget_checkpoint_kind(last_kind)
            except ReplicationRunnerError:
                checkpoint_kind = None
            if (record["stage_id"] != stage.stage_id or record["update_index"] != state.update_index
                    or record["state_fingerprint"] != stable_hash(state.to_dict())
                    or checkpoint_kind != record["checkpoint_kind"]):
                raise ReplicationRunnerError("budget-stopped checkpoint changed")
            return True
        if self.executor.core.coordinator.read(state).complete:
            return True
        return state.update_index == stage.stop_update and last_kind == "boundary"

    @staticmethod
    def _budget_checkpoint_kind(last_kind):
        try:
            return {
                "initial": "initial",
                "initialization": "initial",
                "stage_entry": "stage_entry",
                "boundary": "boundary",
            }[last_kind]
        except (KeyError, TypeError) as error:
            raise ReplicationRunnerError("budget-stop checkpoint kind is invalid") from error

    def _validate_budget_stop(self, record, arm_id, state, stage, last_kind=None):
        if self.budget_stop_policy is None:
            raise ReplicationRunnerError("budget-stop receipt has no frozen policy")
        try:
            kind = self._budget_checkpoint_kind(
                record.get("checkpoint_kind") if last_kind is None else last_kind
            )
        except ReplicationRunnerError as error:
            raise ReplicationRunnerError("invalid budget-stop receipt") from error
        expected = {
            "schema": "generic_neural_solver.replication_budget_stop.v1",
            "attempt_id": self.store.attempt_id, "replication_design_sha256": self.design.binding_hash(),
            "stage_id": stage.stage_id, "arm_id": arm_id, "planned_stop_update": stage.stop_update,
            "update_index": state.update_index, "state_fingerprint": stable_hash(state.to_dict()),
            "policy_fingerprint": state.policy_fingerprint, "checkpoint_kind": kind,
            "policy": dict(self.budget_stop_policy), "reason": "training_cpu_budget",
        }
        used = record.get("consumed_cpu_seconds")
        if (set(record) != {*expected, "consumed_cpu_seconds"}
                or any(canonical_json(record.get(key)) != canonical_json(value) for key, value in expected.items())
                or type(used) not in (int, float) or not math.isfinite(used) or used < 0
                or used < self.budget_stop_policy["training_cpu_limit"] - self.budget_stop_policy["boundary_cpu_reserve"]
                or not stage.start_update <= state.update_index < stage.stop_update
                or kind in ("initial", "stage_entry") and state.update_index != stage.start_update
                or (state.update_index - stage.start_update) % stage.updates_per_round
                or self.executor.core.coordinator.read(state).complete):
            raise ReplicationRunnerError("invalid budget-stop receipt")
        directory = {"initial": "initial", "stage_entry": "stage_entries", "boundary": "boundaries"}[kind]
        stem = {
            "initial": "state",
            "stage_entry": f"stage_entry-{state.update_index:08d}",
            "boundary": f"boundary-{state.update_index:08d}",
        }[kind]
        marker = self.store._directory(arm_id, directory, create=False) / f"{stem}.complete.json"
        if not marker.is_file() or self.store._read(marker, arm_id=arm_id, kind=kind).state.to_dict() != state.to_dict():
            raise ReplicationRunnerError("budget stop requires the exact committed control boundary")

    def _restore_budget_stop(self, states, stage):
        path = self.store.root / "budget-stop.json"
        if path.exists():
            if self.budget_stop_policy is None:
                raise ReplicationRunnerError("budget-stop receipt has no frozen policy")
            receipt = json.loads(path.read_text())
            digest = receipt.pop("receipt_hash", None)
            if stable_hash(receipt) != digest:
                raise ReplicationRunnerError("budget-stop receipt hash mismatch")
            arm_id, state = next(iter(states.items()))
            self._validate_budget_stop(receipt, arm_id, state, stage)
            self.budget_stop_record = ImmutableJSONMapping(receipt)
        elif self.budget_stop_policy is not None:
            arm_id, state = next(iter(states.items()))
            if not self._stage_done(state, stage, self._last_kind(arm_id, stage.stop_update)) and (
                    (self.store.root / "selection.json").exists()
                    or any((self.store.root / "evaluations").rglob("*.json"))):
                raise ReplicationRunnerError("early stage evaluation is missing its budget-stop receipt")

    def _maybe_budget_stop(self, arm_id, state, stage, last_kind):
        """Meter cumulative attempt CPU only at a committed control boundary."""
        if (self.budget_stop_policy is None
                or last_kind not in ("initial", "initialization", "stage_entry", "boundary")
                or self._stage_done(state, stage, last_kind)):
            return
        used = self.budget_cpu_seconds()
        if (type(used) not in (int, float) or not math.isfinite(used) or used < 0
                or self._last_budget_cpu is not None and used < self._last_budget_cpu):
            raise ReplicationRunnerError("budget CPU must be finite, nonnegative and cumulative across attempts")
        self._last_budget_cpu = used
        if used < self.budget_stop_policy["training_cpu_limit"] - self.budget_stop_policy["boundary_cpu_reserve"]:
            return
        checkpoint_kind = self._budget_checkpoint_kind(last_kind)
        record = {
            "schema": "generic_neural_solver.replication_budget_stop.v1",
            "attempt_id": self.store.attempt_id, "replication_design_sha256": self.design.binding_hash(),
            "stage_id": stage.stage_id, "arm_id": arm_id, "planned_stop_update": stage.stop_update,
            "update_index": state.update_index, "state_fingerprint": stable_hash(state.to_dict()),
            "policy_fingerprint": state.policy_fingerprint,
            "checkpoint_kind": checkpoint_kind,
            "policy": dict(self.budget_stop_policy), "reason": "training_cpu_budget", "consumed_cpu_seconds": used,
        }
        self._validate_budget_stop(record, arm_id, state, stage, last_kind)
        path = self.store.root / "budget-stop.json"
        if path.exists():
            raise ReplicationRunnerError("budget-stop receipt must be recovered before further work")
        self.store._write_publication_json(path, {**record, "receipt_hash": stable_hash(record)})
        self.budget_stop_record = ImmutableJSONMapping(record)

    def _validate_control_coordinates(self, control, arm_id, state, stage, round_number):
        roles = getattr(self.executor.core.coordinator, "roles", None)
        if not isinstance(control, ControlEvaluation) or control.request is None:
            raise ReplicationRunnerError("control evaluation is unbound")
        if isinstance(roles, ScheduledRoleBankRegistry):
            try:
                roles.validate_coordinates(
                    control.request, stage_id=stage.stage_id, arm_id=arm_id,
                    round_number=round_number, update_index=state.update_index,
                )
            except (TypeError, ValueError) as error:
                raise ReplicationRunnerError("control stage/replica/update binding mismatch") from error
        if any(value < 0 for value in control.task_upper_mse.values()) or (
            control.request.task_ids != self.executor.metadata.registry.task_ids
        ):
            raise ReplicationRunnerError("control task values or order are invalid")
        roles.validate_request(control.request, PolicyView.from_dict(state.policy_state))

    def _enter_stage(self, arm_id, state, stage):
        coordinator = self.executor.core.coordinator
        if state.update_index != stage.start_update:
            raise ReplicationRunnerError("stage entry requires the preceding final update")
        if self.stage_entry_requirements is not None:
            requirement = self.stage_entry_requirements[stage.stage_id]
            if (
                tuple(coordinator.read(state).permanent) != tuple(requirement["permanent_tasks"])
                or state.method_state.get("preferred") != requirement["preferred_method"]
            ):
                raise ReplicationRunnerError("inherited stage-entry membership or method mismatch")
        policy = PolicyView.from_dict(state.policy_state)
        control = self.evaluation_provider.control(arm_id, policy, stage, stage.from_round)
        self._validate_control_coordinates(control, arm_id, state, stage, stage.from_round)
        control = self.store.archive_control(control)
        updated, _policy_event = coordinator.stage_entry(state, control, stage_id=stage.stage_id)
        event = {
            "event": "stage_entry", "stage_id": stage.stage_id,
            "stage_ordinal": self.stages.index(stage), "arm_id": arm_id,
            "round": stage.from_round, "boundary": 0,
            "update_index": state.update_index,
            "parent_state_sha256": stable_hash(state.to_dict()),
            "control_sha256": full_control_hash(control),
        }
        self.store.commit_stage_entry(arm_id, state, updated, event)
        return updated, event

    def _run_stage(self, states, stage, arm_ids, *, through_update_index=None):
        if len(arm_ids) != stage.population_size:
            raise ReplicationRunnerError("stage population size does not match arm set")
        current = dict(states)
        last_kinds = {
            arm_id: self._last_kind(arm_id, stage.stop_update) for arm_id in arm_ids
        }
        for arm_id in arm_ids:
            state = current[arm_id]
            if state.update_index < stage.start_update or state.update_index > stage.stop_update:
                raise ReplicationRunnerError("stage state update is outside declared schedule")
            if (self.checkpoint_continuation or {}).get("partial_parent_continuation") is not None:
                self.executor.core.coordinator.require_stage_entry(state)
            rejection = self.store.read_update_rejection(arm_id, state)
            if rejection is not None:
                raise PermanentPassUpdateError(rejection["message"], state, rejection["event"])
        events = []
        for round_number in range(stage.first_round, stage.last_round + 1):
            round_end = stage.start_update + (round_number - stage.first_round + 1) * stage.updates_per_round
            if through_update_index is not None and round_end > through_update_index:
                break
            for _ in range(stage.updates_per_round):
                for arm_id in arm_ids:
                    state = current[arm_id]
                    if self._stage_done(state, stage, last_kinds[arm_id]) or state.update_index >= round_end:
                        continue
                    rejection = self.store.read_update_rejection(arm_id, state)
                    if rejection is not None:
                        raise PermanentPassUpdateError(rejection["message"], state, rejection["event"])
                    policy = PolicyView.from_dict(state.policy_state)
                    batch = self.batch_factory(arm_id, policy, stage, state.update_index)
                    try:
                        next_state, event = self.executor.step(state, batch, self.adapters[arm_id])
                    except PermanentPassUpdateError as error:
                        if error.event.get("transaction_rolled_back") and "postproposal" in error.event:
                            self.store.record_update_rejection(arm_id, state, error)
                        raise
                    self.store.commit_update(arm_id, state, next_state, {**event, "arm_id": arm_id})
                    current[arm_id] = next_state
                    last_kinds[arm_id] = "update"
                    events.append({"event": "update", "arm_id": arm_id, "update_index": next_state.update_index})
            for arm_id in arm_ids:
                state = current[arm_id]
                if self._stage_done(state, stage, last_kinds[arm_id]) or (
                    last_kinds[arm_id] == "boundary" and state.update_index == round_end
                ) or state.update_index != round_end:
                    continue
                policy = PolicyView.from_dict(state.policy_state)
                control = self.evaluation_provider.control(arm_id, policy, stage, round_number)
                self._validate_control_coordinates(control, arm_id, state, stage, round_number)
                control = self.store.archive_control(control)
                next_state, boundary_event = self.executor.core.coordinator.boundary(state, control)
                self.store.commit_boundary(arm_id, next_state, {
                    "event": "boundary", "arm_id": arm_id, "round": round_number,
                    "update_index": next_state.update_index, "policy": next_state.policy_fingerprint,
                })
                current[arm_id] = next_state
                last_kinds[arm_id] = "boundary"
                events.append({"event": "boundary", "arm_id": arm_id, "round": round_number,
                               "update_index": next_state.update_index, "policy_event": boundary_event})
                if through_update_index is None or next_state.update_index < through_update_index:
                    self._maybe_budget_stop(arm_id, next_state, stage, "boundary")
            if all(self._stage_done(current[arm_id], stage, last_kinds[arm_id]) for arm_id in arm_ids):
                break
        return current, events

    def _select(self, states, stage):
        records = []
        for arm_id in sorted(states):
            policy = PolicyView.from_dict(states[arm_id].policy_state)
            receipt = self._stage_evaluation(arm_id, states[arm_id], stage, "validation")
            validation = ValidationEvaluation.from_dict(receipt["result"])
            self._validate_validation_binding(arm_id, policy, validation, stage, states[arm_id])
            task_ids = self.executor.metadata.registry.task_ids
            ratio = validation_score(validation, task_ids, self.threshold)
            records.append({
                "replica": arm_id,
                "stage_id": stage.stage_id,
                "validation_update_index": states[arm_id].update_index,
                "state_fingerprint": stable_hash(states[arm_id].to_dict()),
                "policy_fingerprint": policy.fingerprint(),
                "absolute_threshold_ratio": ratio,
                "validation": validation.to_dict(),
            })
        selected = select_validation_record(records)
        return {
            "rule": "minimum fresh validation maximum normalized MSE threshold ratio, then replica",
            "key": ["absolute_threshold_ratio", "replica"],
            "records": records,
            "selected_replica": selected["replica"],
            "candidate_union": self._candidate_union_description(),
            "candidate_union_ids": sorted(states),
            **({
            "replication_design_sha256": self.design.binding_hash(),
            } if self.design is not None else {}),
        }

    def _validate_evaluation_binding(self, arm_id, state, stage, role, evaluation):
        policy = PolicyView.from_dict(state.policy_state)
        if role == "validation":
            self._validate_validation_binding(arm_id, policy, evaluation, stage, state)
            if set(evaluation.task_mean_mse) != set(self.executor.metadata.registry.task_ids):
                raise ReplicationRunnerError("stage validation task inventory mismatch")
            if evaluation.request is not None and evaluation.request.task_ids != self.executor.metadata.registry.task_ids:
                raise ReplicationRunnerError("stage validation task order mismatch")
            return
        if not isinstance(evaluation, CertificationResult) or not evaluation.upper_records:
            raise ReplicationRunnerError("stage certification must retain complete typed outcomes")
        if self.design is None:
            return
        request = evaluation.request
        if request is None or request.role != "certification":
            raise ReplicationRunnerError("stage certification role is unbound")
        if request.task_ids != self.executor.metadata.registry.task_ids:
            raise ReplicationRunnerError("stage certification task order mismatch")
        roles = self.executor.core.coordinator.roles
        try:
            roles.validate_request(request, policy)
            if isinstance(roles, ScheduledRoleBankRegistry):
                roles.validate_coordinates(
                    request, stage_id=stage.stage_id, arm_id=arm_id,
                    round_number=None, update_index=state.update_index,
                )
        except (TypeError, ValueError) as error:
            raise ReplicationRunnerError("stage certification role binding mismatch") from error
        expected_component = self.design.role_manifest_sha256
        if isinstance(roles, ScheduledRoleBankRegistry):
            expected_component = request.metadata["role_bank_manifest_hash"]
            if evaluation.estimator_metadata.get("scheduled_role_registry_hash") != self.design.role_manifest_sha256:
                raise ReplicationRunnerError("stage certification registry provenance mismatch")
        if evaluation.estimator_metadata.get("role_bank_manifest_hash") != expected_component:
            raise ReplicationRunnerError("stage certification component provenance mismatch")

    def _validate_evaluation_record(self, record, arm_id, state, stage, role):
        expected = {
            "stage_id": stage.stage_id, "stage_ordinal": self.stages.index(stage),
            "arm_id": arm_id, "role": role, "update_index": state.update_index,
            "round": stage.from_round + (state.update_index - stage.start_update) // stage.updates_per_round,
            "state_fingerprint": stable_hash(state.to_dict()),
            "policy_fingerprint": state.policy_fingerprint,
            "predecessor_fingerprint": self._evaluation_predecessor(arm_id, state, stage, role),
        }
        if set(record) != {*expected, "result", "checkpoint_reload_equal"} or any(
            canonical_json(record.get(key)) != canonical_json(value) for key, value in expected.items()
        ):
            raise ReplicationRunnerError("stage evaluation checkpoint coordinates mismatch")
        if state.update_index == stage.start_update:
            kind = "initial" if stage == self.stages[0] else "stage_entry"
        else:
            kind = "boundary"
        directory = {"initial": "initial", "stage_entry": "stage_entries", "boundary": "boundaries"}[kind]
        stem = "state" if kind == "initial" else f"{kind}-{state.update_index:08d}"
        marker_path = self.store._directory(arm_id, directory) / f"{stem}.complete.json"
        if not marker_path.is_file():
            raise ReplicationRunnerError("stage evaluation requires a committed exit checkpoint")
        committed = self.store._read(marker_path, arm_id=arm_id, kind=kind)
        if (
            record["checkpoint_reload_equal"] is not True
            or committed.state.to_dict() != state.to_dict()
            or not stage.start_update <= state.update_index <= stage.stop_update
            or (state.update_index - stage.start_update) % stage.updates_per_round
            or not self._stage_done(state, stage, kind)
        ):
            raise ReplicationRunnerError("stage evaluation requires the exact committed exit checkpoint")
        try:
            evaluation = (
                ValidationEvaluation.from_dict(record["result"]) if role == "validation"
                else CertificationResult.from_dict(record["result"])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ReplicationRunnerError("invalid retained stage evaluation result") from error
        self._validate_evaluation_binding(arm_id, state, stage, role, evaluation)

    def _evaluation_predecessor(self, arm_id, state, stage, role):
        if role == "validation":
            return None
        if stage.select_after:
            selection = self.store.read_selection()
            if selection is None or selection.get("selected_replica") != arm_id:
                raise ReplicationRunnerError("population certification requires its committed selection")
            return stable_hash(selection)
        validation = self.store.read_evaluation(
            self.stages.index(stage), arm_id, "validation",
            lambda record: self._validate_evaluation_record(record, arm_id, state, stage, "validation"),
        )
        if validation is None:
            raise ReplicationRunnerError("certification requires its committed stage validation")
        return stable_hash(validation)

    def _stage_evaluation(self, arm_id, state, stage, role, *, required_saved=False):
        def validate(record):
            self._validate_evaluation_record(record, arm_id, state, stage, role)

        record = self.store.read_evaluation(self.stages.index(stage), arm_id, role, validate)
        if record is not None:
            return record
        if required_saved:
            raise ReplicationRunnerError("committed selection is missing its stage validation receipt")
        if role == "validation":
            certification_path = self.store._evaluation_path(self.stages.index(stage), arm_id, "certification")
            if any(path.exists() for path in (
                certification_path, certification_path.with_suffix(".complete.json"),
                certification_path.with_name(".result.json.publication.json"),
            )):
                raise ReplicationRunnerError("stage certification is missing its preceding validation receipt")
        if any((self.store.root / "arms" / f"arm-{arm_id}" / "stage_entries" /
                f"stage_entry-{later.start_update:08d}.complete.json").is_file()
               for later in self.stages[self.stages.index(stage) + 1:]):
            raise ReplicationRunnerError("later stage is missing an earlier evaluation receipt")
        state_hash = stable_hash(state.to_dict())
        policy = PolicyView.from_dict(state.policy_state)
        evaluation = getattr(self.evaluation_provider, role)(
            arm_id, policy, stage, update_index=state.update_index,
        )
        if stable_hash(state.to_dict()) != state_hash:
            raise ReplicationRunnerError("stage evaluation mutated checkpoint state")
        self._validate_evaluation_binding(arm_id, state, stage, role, evaluation)
        record = {
            "stage_id": stage.stage_id, "stage_ordinal": self.stages.index(stage),
            "arm_id": arm_id, "role": role, "update_index": state.update_index,
            "round": stage.from_round + (state.update_index - stage.start_update) // stage.updates_per_round,
            "state_fingerprint": state_hash, "policy_fingerprint": state.policy_fingerprint,
            "predecessor_fingerprint": self._evaluation_predecessor(arm_id, state, stage, role),
            "checkpoint_reload_equal": True, "result": evaluation.to_dict(),
        }
        self.store.commit_evaluation(record, validate)
        return record

    def _validate_validation_binding(self, arm_id, policy, validation, stage, state):
        if not isinstance(validation, ValidationEvaluation):
            raise ReplicationRunnerError("validation provider returned the wrong result type")
        if self.design is None:
            return
        if validation.request is None or validation.request.role != "validation":
            raise ReplicationRunnerError("fresh selection validation role is unbound")
        if validation.request.policy_fingerprint != policy.fingerprint():
            raise ReplicationRunnerError("fresh selection validation policy is stale")
        coordinator = getattr(getattr(self.executor, "core", None), "coordinator", None)
        roles = getattr(coordinator, "roles", None)
        if roles is None or not callable(getattr(roles, "validate_request", None)):
            raise ReplicationRunnerError("fresh selection validation role binding is unavailable")
        try:
            roles.validate_request(validation.request, policy)
            if isinstance(roles, ScheduledRoleBankRegistry):
                roles.validate_coordinates(
                    validation.request, stage_id=stage.stage_id, arm_id=arm_id,
                    round_number=None, update_index=state.update_index,
                )
        except (TypeError, ValueError) as error:
            raise ReplicationRunnerError("fresh selection validation role binding is invalid") from error
        expected_component = self.design.role_manifest_sha256
        if isinstance(roles, ScheduledRoleBankRegistry):
            expected_component = validation.request.metadata["role_bank_manifest_hash"]
            if validation.provenance.get("scheduled_role_registry_hash") != self.design.role_manifest_sha256:
                raise ReplicationRunnerError("fresh selection validation registry provenance is mismatched")
        if validation.provenance.get("role_bank_manifest_hash") != expected_component:
            raise ReplicationRunnerError("fresh selection validation provenance is mismatched")
        if state.update_index == stage.start_update:
            kind = "initial" if stage == self.stages[0] else "stage_entry"
        else:
            kind = "boundary"
        if self.budget_stop_record is not None:
            kind = self.budget_stop_record["checkpoint_kind"]
        if not self._stage_done(state, stage, kind):
            raise ReplicationRunnerError("fresh selection validation is not at the declared boundary")

    def _validate_selection(self, states, selection, stage):
        if self.design is not None:
            if selection.get("replication_design_sha256") != self.design.binding_hash():
                raise ReplicationRunnerError("persisted selection design binding mismatch")
            if tuple(selection.get("key", ())) != self.design.selection_key:
                raise ReplicationRunnerError("persisted selection key differs from frozen design")
        expected_key = self.design.selection_key if self.design is not None else (
            "absolute_threshold_ratio", "replica"
        )
        if tuple(selection.get("key", ())) != expected_key:
            raise ReplicationRunnerError("persisted selection key is invalid")
        descriptions = {self._candidate_union_description()}
        if self.expected_selected_replica is not None:
            descriptions.add("five final worker states")
        if selection.get("candidate_union") not in descriptions:
            raise ReplicationRunnerError("persisted selection candidate union is invalid")
        records = selection.get("records")
        if not isinstance(records, list) or len(records) != len(states):
            raise ReplicationRunnerError("persisted selection candidate union is incomplete")
        record_ids = [record.get("replica") for record in records]
        if record_ids != sorted(states) or selection.get("candidate_union_ids") != sorted(states):
            raise ReplicationRunnerError("persisted selection candidate union is not complete")
        task_ids = self.executor.metadata.registry.task_ids
        recalculated = []
        for arm_id, record in zip(sorted(states), records, strict=True):
            policy = PolicyView.from_dict(states[arm_id].policy_state)
            if record.get("stage_id") != stage.stage_id:
                raise ReplicationRunnerError("persisted selection stage binding mismatch")
            if record.get("validation_update_index") != states[arm_id].update_index:
                raise ReplicationRunnerError("persisted selection validation freshness mismatch")
            if record.get("state_fingerprint") != stable_hash(states[arm_id].to_dict()):
                raise ReplicationRunnerError("persisted selection state provenance mismatch")
            if record.get("policy_fingerprint") != policy.fingerprint():
                raise ReplicationRunnerError("persisted selection policy provenance mismatch")
            try:
                validation = ValidationEvaluation.from_dict(record["validation"])
            except (KeyError, TypeError, ValueError) as error:
                raise ReplicationRunnerError("persisted selection validation provenance is invalid") from error
            if set(validation.task_mean_mse) != set(task_ids):
                raise ReplicationRunnerError("persisted selection validation task inventory mismatch")
            if validation.request is not None and validation.request.task_ids != task_ids:
                raise ReplicationRunnerError("persisted selection validation request task order mismatch")
            if self.design is not None:
                self._validate_validation_binding(arm_id, policy, validation, stage, states[arm_id])
            ratio = max(validation.task_mean_mse[task] / self.threshold for task in task_ids)
            if record.get("absolute_threshold_ratio") != ratio:
                raise ReplicationRunnerError("persisted selection score provenance mismatch")
            recalculated.append((ratio, arm_id))
        expected = min(recalculated)
        if expected[1] != selection.get("selected_replica"):
            raise ReplicationRunnerError("persisted selection key is not reproducible")
        if self.expected_selected_replica is not None and selection.get("selected_replica") != self.expected_selected_replica:
            raise ReplicationRunnerError("persisted source selector chose a different replica")

    def run(self, *, through_update_index: int | None = None) -> ReplicationRunResult:
        """Run the lifecycle, or pause one stage after a committed control boundary.

        A pause returns durable states and this invocation's events without final
        roles or selection. The initial states and full design remain available
        for resuming this runner or reconstructing it against the same store.
        Natural completion or an actual budget stop can end before the cutoff.
        """
        population_stage = self.stages[0]
        if through_update_index is not None:
            if len(self.stages) != 1:
                raise ReplicationRunnerError("boundary pause supports only a single stage")
            if (type(through_update_index) is not int
                    or not population_stage.start_update <= through_update_index <= population_stage.stop_update
                    or (through_update_index - population_stage.start_update) % population_stage.updates_per_round):
                raise ReplicationRunnerError("pause cutoff must be an aligned boundary in the declared stage")
        if (self.store.root / "budget-stop.json").exists() and (
                self.budget_stop_policy is None or any(not self.store.has_initial(arm_id) for arm_id in self.states)):
            raise ReplicationRunnerError("budget-stop receipt requires its frozen policy and initial checkpoint")
        initial_states = dict(self.states) if through_update_index is not None else None
        try:
            states, events = self._initialize(population_stage)
        finally:
            if initial_states is not None:
                self.states = initial_states
        if through_update_index is not None and any(state.update_index > through_update_index for state in states.values()):
            raise ReplicationRunnerError("pause cutoff is behind recovered state")
        evaluations = []
        self._restore_budget_stop(states, population_stage)
        if self.budget_stop_policy is not None:
            for arm_id, state in states.items():
                if through_update_index is None or state.update_index < through_update_index:
                    self._maybe_budget_stop(arm_id, state, population_stage, self._last_kind(arm_id, population_stage.stop_update))
        if not all(
            self._stage_done(
                states[arm_id],
                population_stage,
                self._last_kind(arm_id, population_stage.stop_update),
            )
            for arm_id in sorted(states)
        ):
            states, stage_events = self._run_stage(states, population_stage, sorted(states),
                                                  through_update_index=through_update_index)
        else:
            stage_events = []
        events.extend(stage_events)
        if through_update_index is not None:
            return ReplicationRunResult(states, None, tuple(events))
        selection = self.store.read_selection()
        for arm_id in sorted(states):
            evaluations.append(self._stage_evaluation(
                arm_id, states[arm_id], population_stage, "validation", required_saved=selection is not None,
            ))
        if selection is None:
            selection = self._select(states, population_stage)
            selection = self.store.commit_selection(selection)
        self._validate_selection(states, selection, population_stage)
        for validation_record, receipt in zip(selection["records"], evaluations, strict=True):
            if canonical_json(validation_record["validation"]) != canonical_json(receipt["result"]):
                raise ReplicationRunnerError("selection differs from its committed validation receipt")
        selected_replica = int(selection["selected_replica"])
        if self.lifecycle_endpoint == "certification":
            evaluations.append(self._stage_evaluation(
                selected_replica, states[selected_replica], population_stage, "certification",
            ))
        for stage in self.stages[1:]:
            selected_state = states[selected_replica]
            if self.executor.core.coordinator.read(selected_state).complete:
                break
            if selected_state.update_index < stage.start_update:
                raise ReplicationRunnerError("selected replica cannot enter continuation stage")
            selected_state, selected_events = self.store.recover(
                selected_replica, through_update_index=stage.stop_update,
                include_stage_entry_at_cutoff=False,
            )
            entry_events = [event for event in selected_events if (
                event.get("event") == "stage_entry" and event.get("stage_id") == stage.stage_id
            )]
            if not entry_events:
                if selected_state.update_index != stage.start_update:
                    raise ReplicationRunnerError("continuation updates lack their stage-entry control")
                selected_state, entry_event = self._enter_stage(selected_replica, selected_state, stage)
                selected_events = (*selected_events, entry_event)
                events.append(entry_event)
            elif len(entry_events) != 1 or entry_events[0].get("update_index") != stage.start_update:
                raise ReplicationRunnerError("stage-entry history does not match the frozen schedule")
            states[selected_replica] = selected_state
            selected_last_kind = str(selected_events[-1].get("event"))
            if not self._stage_done(selected_state, stage, selected_last_kind):
                states, stage_events = self._run_stage(states, stage, [selected_replica])
            else:
                stage_events = []
            events.extend(stage_events)
            for role in ("validation", "certification"):
                evaluations.append(self._stage_evaluation(selected_replica, states[selected_replica], stage, role))
        expected_payloads = {
            self.store._evaluation_path(record["stage_ordinal"], record["arm_id"], record["role"])
            for record in evaluations
        }
        expected_files = {
            path for payload in expected_payloads for path in (
                payload, payload.with_suffix(".complete.json"),
                payload.with_name(".result.json.publication.json"),
            )
        }
        actual_files = {path for path in (self.store.root / "evaluations").rglob("*.json") if path.is_file()}
        if actual_files != expected_files:
            raise ReplicationRunnerError("stage evaluation receipt inventory mismatch")
        self.states = dict(states)
        return ReplicationRunResult(states, selection, tuple(events), tuple(evaluations))


__all__ = [
    "AtomicCheckpointStore",
    "GenericReplicationRunner",
    "ReplicationEvaluationProvider",
    "ReplicationRunResult",
    "ReplicationRunnerError",
    "ReplicationStage",
    "StoredTransaction",
]
