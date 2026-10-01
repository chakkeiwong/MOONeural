"""Replication-scoped role banks and permanent-pass coordination.

The historical generic policy remains unchanged. This module supplies the
multi-seed role binding needed by the independent replication route.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

from .generic_permanent_pass import PermanentPassState
from .generic_role_banks import RoleBankManifest
from .generic_scheduled_role_banks import ScheduledRoleBankRegistry
from .generic_training_contracts import (
    CheckpointState,
    EvaluationRequest,
    PolicyView,
    RoleBinding,
    RoleManifest,
    TaskRegistry,
    canonical_json,
    stable_hash,
    validate_task_partition,
)


class ReplicationRoleManifest:
    """Immutable role manifest allowing a role to own a complete seed bank."""

    def __init__(self, manifest: RoleManifest):
        if not isinstance(manifest, RoleManifest):
            raise TypeError("a RoleManifest is required")
        self._manifest = RoleManifest.from_dict(manifest.to_dict())

    @classmethod
    def from_seed_registries(cls, registries: dict[str, tuple[int, ...]]):
        bindings = []
        for role, seeds in registries.items():
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("role seed registries must be nonempty and unique")
            bindings.append(
                RoleBinding(
                    role,
                    seeds[0],
                    f"replication-{role}-target",
                    "replication-v1",
                    tuple(f"{role}-{seed}" for seed in seeds),
                    metadata={"seed_registry": list(seeds)},
                )
            )
        return cls(RoleManifest(tuple(bindings), "replication-v1"))

    @property
    def bindings(self):
        return self._manifest.bindings

    def to_dict(self):
        return self._manifest.to_dict()

    def binding_hash(self):
        return self._manifest.binding_hash()

    def _binding(self, request: EvaluationRequest) -> RoleBinding:
        binding = next((item for item in self.bindings if item.role == request.role), None)
        if binding is None:
            raise ValueError("replication role is not registered")
        allowed = tuple(binding.metadata.get("seed_registry", (binding.seed,)))
        if request.seeds != allowed:
            raise ValueError("replication role seed bank mismatch")
        if (request.target_id, request.target_version, request.scale_version,
                request.estimator_version) != (
                    binding.target_id,
                    binding.target_version,
                    binding.scale_version,
                    binding.estimator_version,
                ):
            raise ValueError("replication role target or scale mismatch")
        return binding

    def validate_request_binding(self, request: EvaluationRequest) -> None:
        self._binding(request)

    def validate_request(self, request: EvaluationRequest, policy: PolicyView) -> None:
        self._binding(request)
        if request.policy_fingerprint != policy.fingerprint():
            raise ValueError("replication role policy binding mismatch")


class ReplicationPermanentPassCoordinator:
    """Permanent-pass state transactions with replication seed-bank bindings."""

    @staticmethod
    def contract_identity():
        return {
            "module": "mooneural.training.generic_replication_policy",
            "name": "ReplicationPermanentPassCoordinator",
            "schema": "dsge_hmc.generic_replication_policy.v1",
        }

    def __init__(self, registry: TaskRegistry, roles: ReplicationRoleManifest, *, threshold: float):
        if not isinstance(roles, (ReplicationRoleManifest, RoleBankManifest, ScheduledRoleBankRegistry)):
            raise TypeError("replication role or role-bank manifest is required")
        self.registry = TaskRegistry.from_dict(registry.to_dict())
        self.roles = roles
        self._role_payload = roles.immutable_payload() if isinstance(roles, ScheduledRoleBankRegistry) else None
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("invalid permanent-pass threshold")
        self.threshold = float(threshold)
        self._historical_controls = ()
        self._historical_roles = {}
        self._partial_parent = None

    def _validate_control_coordinate(self, request, update_index, stage_id=None, *, roles=None):
        roles = self.roles if roles is None else roles
        if not isinstance(request, EvaluationRequest) or request.role != "control":
            raise ValueError("a bound control request is required")
        roles.validate_request_binding(request)
        if isinstance(roles, ScheduledRoleBankRegistry):
            scope = request.metadata["scheduled_role_scope"]
            roles.validate_coordinates(
                request, stage_id=scope["stage_id"] if stage_id is None else stage_id,
                arm_id=scope["arm_id"], round_number=scope["round_number"],
                update_index=update_index,
            )

    def import_history(self, checkpoint, *, partial_parent=None):
        """Bind an exact historical prefix to its original immutable registries."""
        state = PermanentPassState.from_dict(checkpoint.metadata["permanent_pass_rotation"])
        if (state.task_ids != self.registry.task_ids or state.threshold != self.threshold
                or state.update_index != checkpoint.update_index
                or checkpoint.metadata["task_registry"] != self.registry.to_dict()):
            raise ValueError("continuation history task/threshold/clock mismatch")
        if partial_parent is not None and (
                partial_parent.get("schema") != "generic_neural_solver.partial_parent_continuation.v1"
                or partial_parent.get("parent_checkpoint_hash") != stable_hash(checkpoint.to_dict())
                or partial_parent.get("effective_update") != checkpoint.update_index
                or partial_parent.get("history_count") != len(state.controls)
                or partial_parent.get("last_control_update") != state.controls[-1].update_index):
            raise ValueError("partial continuation history differs from verified parent")
        if state.complete or (partial_parent is None and state.controls[-1].update_index != checkpoint.update_index):
            raise ValueError("continuation requires an unresolved completed control boundary")
        manifests = dict(checkpoint.metadata.get("continuation_role_manifests", {}))
        original = ScheduledRoleBankRegistry.from_dict(checkpoint.metadata["role_manifest"])
        manifests[original.binding_hash()] = original.to_dict()
        roles = {digest: ScheduledRoleBankRegistry.from_dict(payload) for digest, payload in manifests.items()}
        if any(digest != registry.binding_hash() for digest, registry in roles.items()):
            raise ValueError("historical registry hash mismatch")
        for point in state.controls:
            registry = roles[point.evaluation.request.metadata["scheduled_role_registry_hash"]]
            self._validate_control_coordinate(point.evaluation.request, point.update_index, point.stage_id, roles=registry)
        if state.controls[-1].update_index == checkpoint.update_index:
            latest = state.controls[-1].evaluation.request
            roles[latest.metadata["scheduled_role_registry_hash"]].validate_request(
                latest, PolicyView.from_dict(checkpoint.policy_state))
        self._historical_controls, self._historical_roles = state.controls, roles
        self._partial_parent = partial_parent
        return replace(checkpoint, metadata={**checkpoint.metadata,
            "role_manifest": self._role_payload, "continuation_role_manifests": manifests})

    def _store(self, checkpoint, policy):
        metadata = dict(checkpoint.metadata)
        metadata["permanent_pass_rotation"] = policy.to_dict()
        metadata["task_registry"] = self.registry.to_dict()
        metadata["role_manifest"] = self._role_payload if self._role_payload is not None else self.roles.to_dict()
        return replace(checkpoint, update_index=policy.update_index, metadata=metadata)

    def initialize(self, checkpoint, control):
        if "permanent_pass_rotation" in checkpoint.metadata:
            raise ValueError("existing permanent membership cannot be reinitialized")
        self._validate_control_coordinate(control.request, checkpoint.update_index)
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
        state = PermanentPassState.from_dict(checkpoint.metadata["permanent_pass_rotation"])
        if state.task_ids != self.registry.task_ids or state.threshold != self.threshold:
            raise ValueError("replication policy/task/threshold binding mismatch")
        if state.update_index != checkpoint.update_index:
            raise ValueError("replication policy/global-update binding mismatch")
        declared_roles = self._role_payload if self._role_payload is not None else self.roles.to_dict()
        if (checkpoint.metadata["task_registry"] != self.registry.to_dict()
                or checkpoint.metadata["role_manifest"] != declared_roles):
            raise ValueError("replication registry or role manifest mismatch")
        prefix_length = len(self._historical_controls)
        if state.controls[:prefix_length] != self._historical_controls:
            raise ValueError("continuation historical control prefix changed")
        manifests = {digest: roles.to_dict() for digest, roles in self._historical_roles.items()}
        if canonical_json(checkpoint.metadata.get("continuation_role_manifests", {})) != canonical_json(manifests):
            raise ValueError("continuation historical registries changed")
        for index, point in enumerate(state.controls):
            roles = self.roles if index >= prefix_length else self._historical_roles[
                point.evaluation.request.metadata["scheduled_role_registry_hash"]]
            self._validate_control_coordinate(point.evaluation.request, point.update_index, point.stage_id, roles=roles)
        partition = state.partition
        validate_task_partition(self.registry, partition["active"], partition["constraints"])
        if state.controls[-1].update_index == checkpoint.update_index:
            roles.validate_request(state.controls[-1].evaluation.request, PolicyView.from_dict(checkpoint.policy_state))
        return state

    def require_stage_entry(self, checkpoint):
        state = self.read(checkpoint)
        if self._partial_parent is not None:
            count = len(self._historical_controls)
            if len(state.controls) <= count:
                raise ValueError("partial continuation requires its fresh stage-entry control before updates")
            point = state.controls[count]
            scope = point.evaluation.request.metadata["scheduled_role_scope"]
            if (point.update_index != self._partial_parent["effective_update"]
                    or scope["stage_id"] != self._partial_parent["entry_stage_id"]):
                raise ValueError("partial continuation entry control has wrong stage or clock")
        return state

    def boundary(self, checkpoint, control):
        state = self.read(checkpoint)
        self._validate_control_coordinate(control.request, checkpoint.update_index)
        updated = state.boundary(
            control, PolicyView.from_dict(checkpoint.policy_state), self.roles
        )
        return self._store(checkpoint, updated), updated.history()[-1]

    def stage_entry(self, checkpoint, control, *, stage_id):
        state = self.read(checkpoint)
        self._validate_control_coordinate(control.request, checkpoint.update_index, stage_id)
        if self._partial_parent is not None and len(state.controls) == len(self._historical_controls):
            if (stage_id != self._partial_parent["entry_stage_id"]
                    or checkpoint.update_index != self._partial_parent["effective_update"]):
                raise ValueError("partial continuation entry differs from verified stage/clock")
            if state.controls[-1].update_index < checkpoint.update_index:
                updated = state.boundary(control, PolicyView.from_dict(checkpoint.policy_state), self.roles)
            else:
                updated = state.stage_entry(control, PolicyView.from_dict(checkpoint.policy_state), self.roles, stage_id=stage_id)
        else:
            updated = state.stage_entry(
                control, PolicyView.from_dict(checkpoint.policy_state), self.roles,
                stage_id=stage_id,
            )
        return self._store(checkpoint, updated), updated.history()[-1]

    def commit(self, checkpoint, proposed):
        state = self.require_stage_entry(checkpoint)
        if (proposed.attempt_id != checkpoint.attempt_id
                or proposed.update_index != checkpoint.update_index + 1):
            raise ValueError("replication update must advance global update exactly once")
        if proposed.metadata != checkpoint.metadata:
            raise ValueError("replication optimizer update cannot rewrite policy metadata")
        return self._store(proposed, state.committed_update())

    def transfer(self, checkpoint, *, attempt_id):
        self.read(checkpoint)
        copied = CheckpointState.from_dict(json.loads(canonical_json(checkpoint.to_dict())))
        return replace(copied, attempt_id=attempt_id)


__all__ = ["ReplicationPermanentPassCoordinator", "ReplicationRoleManifest"]
