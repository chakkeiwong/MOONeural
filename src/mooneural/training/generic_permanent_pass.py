"""Host-only, monotone permanent-pass policy; no equations or replay acceptance."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from itertools import combinations, islice

from .generic_training_contracts import (
    CheckpointState,
    ContractSchemaError,
    ControlEvaluation,
    PolicyView,
    RoleManifest,
    TaskRegistry,
    canonical_json,
    stable_hash,
    validate_task_partition,
)

POLICY_NAME = "permanent_pass_rotation"
POLICY_SCHEMA = "dsge_hmc.generic_permanent_pass.v1"
STAGE_POLICY_SCHEMA = "dsge_hmc.generic_permanent_pass.v2"


def _update_index(value):
    if type(value) is not int or value < 0:
        raise ValueError("global update must be a non-negative integer")
    return value


def _partition(task_ids, permanent, update_index):
    permanent = tuple(task for task in task_ids if task in permanent)
    unresolved = tuple(task for task in task_ids if task not in permanent)
    active_count = min(3, len(unresolved))
    active = ()
    if active_count:
        offset = update_index % math.comb(len(unresolved), active_count)
        active = next(islice(combinations(unresolved, active_count), offset, None))
    temporary = tuple(task for task in unresolved if task not in active)
    return {
        "update_index": update_index,
        "permanent": list(permanent),
        "unresolved": list(unresolved),
        "active": list(active),
        "temporary_constraints": list(temporary),
        "constraints": [task for task in task_ids if task not in active],
    }


@dataclass(frozen=True)
class ControlPoint:
    update_index: int
    evaluation_json: str
    stage_id: str | None = None

    def __post_init__(self):
        _update_index(self.update_index)
        if self.stage_id is not None and (
            not isinstance(self.stage_id, str) or not self.stage_id.strip()
        ):
            raise ValueError("stage-entry identity must be nonempty")
        control = ControlEvaluation.from_dict(json.loads(self.evaluation_json))
        if control.request is None or control.request.role != "control":
            raise ValueError(
                "permanent-pass membership requires bound control evidence"
            )
        if any(value < 0.0 for value in control.task_upper_mse.values()):
            raise ValueError("control upper normalized MSE must be non-negative")
        object.__setattr__(self, "evaluation_json", canonical_json(control.to_dict()))

    @classmethod
    def capture(cls, update_index, control, policy, roles, *, stage_id=None):
        if not isinstance(control, ControlEvaluation) or control.request is None:
            raise ValueError(
                "a bound ControlEvaluation is required, not validation or replay"
            )
        roles.validate_request(control.request, policy)
        return cls(update_index, canonical_json(control.to_dict()), stage_id)

    @property
    def evaluation(self):
        return ControlEvaluation.from_dict(json.loads(self.evaluation_json))


@dataclass(frozen=True)
class PermanentPassState:
    task_ids: tuple[str, ...]
    threshold: float
    update_index: int
    controls: tuple[ControlPoint, ...]

    def __post_init__(self):
        if not isinstance(self.task_ids, (tuple, list)) or not self.task_ids:
            raise ValueError("an ordered nonempty task inventory is required")
        tasks = tuple(self.task_ids)
        if any(not isinstance(task, str) or not task.strip() for task in tasks) or len(
            set(tasks)
        ) != len(tasks):
            raise ValueError("task IDs must be nonempty and unique")
        if (
            type(self.threshold) not in (int, float)
            or not math.isfinite(self.threshold)
            or self.threshold <= 0.0
        ):
            raise ValueError("permanent-pass threshold must be finite and positive")
        _update_index(self.update_index)
        if not isinstance(self.controls, (tuple, list)) or not self.controls:
            raise ValueError("initial control evidence cannot be omitted")
        controls = tuple(self.controls)
        previous_update = -1
        permanent = set()
        entered_stages = set()
        for index, point in enumerate(controls):
            if not isinstance(point, ControlPoint):
                raise TypeError("membership history requires ControlPoint entries")
            if point.stage_id is not None:
                if (
                    index == 0
                    or point.stage_id in entered_stages
                    or point.update_index != previous_update
                ):
                    raise ValueError("stage entry requires a distinct stage at the preceding boundary")
                entered_stages.add(point.stage_id)
            elif (
                point.update_index <= previous_update
                or point.update_index > self.update_index
            ):
                raise ValueError(
                    "control boundaries must be strictly increasing and not in the future"
                )
            if len(permanent) == len(tasks):
                raise ValueError("a completed policy cannot continue or reactivate")
            control = point.evaluation
            if control.request.task_ids != tasks or set(control.task_upper_mse) != set(
                tasks
            ):
                raise ValueError("control task order or inventory mismatch")
            permanent.update(
                task for task in tasks if control.task_upper_mse[task] <= self.threshold
            )
            previous_update = point.update_index
        if len(permanent) == len(tasks) and previous_update != self.update_index:
            raise ValueError("a completed policy cannot consume an update")
        object.__setattr__(self, "task_ids", tasks)
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "controls", controls)

    @classmethod
    def initialize(cls, registry, control, policy, roles, *, threshold, update_index):
        point = ControlPoint.capture(update_index, control, policy, roles)
        return cls(registry.task_ids, threshold, update_index, (point,))

    @property
    def permanent(self):
        passed = {
            task
            for point in self.controls
            for task, value in point.evaluation.task_upper_mse.items()
            if value <= self.threshold
        }
        return tuple(task for task in self.task_ids if task in passed)

    @property
    def partition(self):
        return _partition(self.task_ids, self.permanent, self.update_index)

    @property
    def complete(self):
        return len(self.permanent) == len(self.task_ids)

    def committed_update(self):
        if self.complete:
            raise ValueError(
                "no optimizer update is authorized after permanent-pass completion"
            )
        return replace(self, update_index=self.update_index + 1)

    def boundary(self, control, policy, roles):
        if self.complete or self.controls[-1].update_index == self.update_index:
            raise ValueError(
                "a control boundary requires new updates and an unresolved pool"
            )
        point = ControlPoint.capture(self.update_index, control, policy, roles)
        return replace(self, controls=(*self.controls, point))

    def stage_entry(self, control, policy, roles, *, stage_id):
        if self.complete or self.controls[-1].update_index != self.update_index:
            raise ValueError("stage entry requires an unresolved pool at a completed boundary")
        point = ControlPoint.capture(
            self.update_index, control, policy, roles, stage_id=stage_id
        )
        if point.stage_id is None:
            raise ValueError("stage-entry identity is required")
        return replace(self, controls=(*self.controls, point))

    def history(self):
        permanent = set()
        records = []
        for index, point in enumerate(self.controls):
            control = point.evaluation
            before = _partition(self.task_ids, permanent, point.update_index)
            entered = [
                task
                for task in self.task_ids
                if task not in permanent
                and control.task_upper_mse[task] <= self.threshold
            ]
            regressed = [
                task
                for task in self.task_ids
                if task in permanent and control.task_upper_mse[task] > self.threshold
            ]
            permanent.update(entered)
            records.append(
                {
                    "event": "stage_entry" if point.stage_id is not None else (
                        "initialization" if index == 0 else "boundary"
                    ),
                    **({
                        "stage_id": point.stage_id,
                        "previous_control_sha256": stable_hash(records[-1]),
                    } if point.stage_id is not None else {}),
                    "update_index": point.update_index,
                    "before": before,
                    "after": _partition(self.task_ids, permanent, point.update_index),
                    "entered_permanent": entered,
                    "regressed_permanent_diagnostic": regressed,
                    "control": control.to_dict(),
                }
            )
        return records

    def to_dict(self):
        payload = {
            "schema": STAGE_POLICY_SCHEMA if any(
                point.stage_id is not None for point in self.controls
            ) else POLICY_SCHEMA,
            "policy_name": POLICY_NAME,
            "task_ids": list(self.task_ids),
            "threshold": self.threshold,
            "update_index": self.update_index,
            "partition": self.partition,
            "history": self.history(),
        }
        return {**payload, "state_hash": stable_hash(payload)}

    @classmethod
    def from_dict(cls, payload):
        fields = {
            "schema",
            "policy_name",
            "task_ids",
            "threshold",
            "update_index",
            "partition",
            "history",
            "state_hash",
        }
        if (
            set(payload) != fields
            or payload.get("schema") not in {POLICY_SCHEMA, STAGE_POLICY_SCHEMA}
            or payload.get("policy_name") != POLICY_NAME
        ):
            raise ContractSchemaError(
                "unknown permanent-pass schema; no implicit policy migration"
            )
        content = {key: value for key, value in payload.items() if key != "state_hash"}
        if stable_hash(content) != payload["state_hash"]:
            raise ValueError("permanent-pass state hash mismatch")
        result = cls(
            payload["task_ids"],
            payload["threshold"],
            payload["update_index"],
            tuple(
                ControlPoint(
                    item["update_index"], canonical_json(item["control"]),
                    item.get("stage_id"),
                )
                for item in payload["history"]
            ),
        )
        if canonical_json(result.to_dict()) != canonical_json(payload):
            raise ValueError(
                "membership history or serialized partition is inconsistent; reactivation refused"
            )
        return result


class PermanentPassCoordinator:
    """Pure host checkpoint transactions; preserve all non-policy continuation state."""

    def __init__(
        self, registry: TaskRegistry, roles: RoleManifest, *, threshold: float
    ):
        self.registry = TaskRegistry.from_dict(registry.to_dict())
        self.roles = RoleManifest.from_dict(roles.to_dict())
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("invalid permanent-pass threshold")
        self.threshold = float(threshold)

    def _store(self, checkpoint, policy):
        metadata = dict(checkpoint.metadata)
        metadata[POLICY_NAME] = policy.to_dict()
        metadata["task_registry"] = self.registry.to_dict()
        metadata["role_manifest"] = self.roles.to_dict()
        return replace(checkpoint, update_index=policy.update_index, metadata=metadata)

    def initialize(self, checkpoint, control):
        if POLICY_NAME in checkpoint.metadata:
            raise ValueError("existing permanent membership cannot be reinitialized")
        state = PermanentPassState.initialize(
            self.registry,
            control,
            PolicyView.from_dict(checkpoint.policy_state),
            self.roles,
            threshold=self.threshold,
            update_index=checkpoint.update_index,
        )
        return self._store(checkpoint, state)

    def read(self, checkpoint):
        state = PermanentPassState.from_dict(checkpoint.metadata[POLICY_NAME])
        if (
            state.task_ids != self.registry.task_ids
            or state.threshold != self.threshold
            or state.update_index != checkpoint.update_index
        ):
            raise ValueError(
                "checkpoint policy/task/threshold/global-update binding mismatch"
            )
        if (
            checkpoint.metadata["task_registry"] != self.registry.to_dict()
            or checkpoint.metadata["role_manifest"] != self.roles.to_dict()
        ):
            raise ValueError("checkpoint registry or role manifest mismatch")
        binding = next(item for item in self.roles.bindings if item.role == "control")
        for point in state.controls:
            request = point.evaluation.request
            if request.seeds != (binding.seed,) or (
                request.target_id,
                request.target_version,
                request.scale_version,
                request.estimator_version,
            ) != (
                binding.target_id,
                binding.target_version,
                binding.scale_version,
                binding.estimator_version,
            ):
                raise ValueError(
                    "historical control role/target/scale binding mismatch"
                )
        partition = state.partition
        validate_task_partition(
            self.registry, partition["active"], partition["constraints"]
        )
        if state.controls[-1].update_index == checkpoint.update_index:
            self.roles.validate_request(
                state.controls[-1].evaluation.request,
                PolicyView.from_dict(checkpoint.policy_state),
            )
        return state

    def boundary(self, checkpoint, control):
        state = self.read(checkpoint)
        updated = state.boundary(
            control, PolicyView.from_dict(checkpoint.policy_state), self.roles
        )
        return self._store(checkpoint, updated), updated.history()[-1]

    def commit(self, checkpoint, proposed):
        state = self.read(checkpoint)
        if (
            proposed.attempt_id != checkpoint.attempt_id
            or proposed.update_index != checkpoint.update_index + 1
        ):
            raise ValueError(
                "an update must preserve the attempt and advance global update exactly once"
            )
        if proposed.metadata != checkpoint.metadata:
            raise ValueError(
                "an optimizer update cannot rewrite membership or checkpoint metadata"
            )
        return self._store(proposed, state.committed_update())

    def transfer(self, checkpoint, *, attempt_id):
        self.read(checkpoint)
        copied = CheckpointState.from_dict(
            json.loads(canonical_json(checkpoint.to_dict()))
        )
        return replace(copied, attempt_id=attempt_id)
