"""Replication role-bank policy tests; no model evaluation or training."""

from dataclasses import replace

import pytest

from mooneural.training.generic_permanent_pass_executor import PermanentPassExecutor
from tests.support.fixtures import fixture
from mooneural.training.generic_replication_policy import (
    ReplicationPermanentPassCoordinator,
    ReplicationRoleManifest,
)
from mooneural.training.generic_training_contracts import (
    ControlEvaluation,
    EvaluationRequest,
    PolicyView,
)


def control(checkpoint, seeds, task_ids):
    policy = PolicyView.from_dict(checkpoint.policy_state)
    return ControlEvaluation(
        {task: 0.2 for task in task_ids},
        request=EvaluationRequest(
            role="control",
            target_id="replication-control-target",
            seeds=seeds,
            sample_count=len(seeds),
            target_version="v1",
            estimator_version="v1",
            scale_version="replication-v1",
            policy_fingerprint=policy.fingerprint(),
            task_ids=task_ids,
        ),
    )


def test_replication_control_accepts_complete_seed_bank_and_round_trips():
    executor, checkpoint, _ = fixture()
    seeds = tuple(range(2026083200, 2026083220))
    roles = ReplicationRoleManifest.from_seed_registries({"control": seeds})
    coordinator = ReplicationPermanentPassCoordinator(
        executor.adapter.registry,
        roles,
        threshold=0.04,
    )
    initialized = coordinator.initialize(
        checkpoint, control(checkpoint, seeds, executor.adapter.registry.task_ids)
    )
    restored = coordinator.read(initialized)
    assert restored.controls[0].evaluation.request.seeds == seeds
    assert initialized.metadata["role_manifest"] == roles.to_dict()


def test_replication_control_rejects_partial_or_wrong_seed_bank():
    executor, checkpoint, _ = fixture()
    seeds = tuple(range(2026083200, 2026083220))
    roles = ReplicationRoleManifest.from_seed_registries({"control": seeds})
    coordinator = ReplicationPermanentPassCoordinator(executor.adapter.registry, roles, threshold=0.04)
    valid = control(checkpoint, seeds, executor.adapter.registry.task_ids)
    bad = replace(
        valid,
        request=replace(valid.request, seeds=seeds[:-1]),
    )
    with pytest.raises(ValueError, match="seed bank"):
        coordinator.initialize(checkpoint, bad)


def test_replication_coordinator_binds_to_the_shared_executor_without_updates():
    source_executor, checkpoint, _ = fixture()
    seeds = (2026083200, 2026083201)
    roles = ReplicationRoleManifest.from_seed_registries({"control": seeds})
    executor = PermanentPassExecutor(
        source_executor.adapter, source_executor.spec, roles, threshold=0.04,
        coordinator_factory=ReplicationPermanentPassCoordinator,
    )
    initialized = executor.initialize(
        checkpoint, control(checkpoint, seeds, executor.adapter.registry.task_ids)
    )
    assert executor.coordinator.read(initialized).update_index == checkpoint.update_index


def test_prior_control_remains_valid_between_updates_and_new_boundary_is_fresh():
    executor, checkpoint, _ = fixture()
    seeds = (2026083200, 2026083201)
    task_ids = executor.adapter.registry.task_ids
    roles = ReplicationRoleManifest.from_seed_registries({"control": seeds})
    coordinator = ReplicationPermanentPassCoordinator(executor.adapter.registry, roles, threshold=0.04)
    initialized = coordinator.initialize(checkpoint, control(checkpoint, seeds, task_ids))
    policy = PolicyView.from_dict(initialized.policy_state)
    changed_policy = replace(policy, values=tuple(value + 1e-5 for value in policy.values))
    proposed = replace(
        initialized, update_index=initialized.update_index + 1,
        policy_state=changed_policy.to_dict(), policy_fingerprint=changed_policy.fingerprint(),
    )
    committed = coordinator.commit(initialized, proposed)
    assert coordinator.read(committed).controls[-1].update_index == initialized.update_index
    with pytest.raises(ValueError, match="policy binding"):
        coordinator.boundary(committed, control(initialized, seeds, task_ids))
    bounded, _event = coordinator.boundary(committed, control(committed, seeds, task_ids))
    assert coordinator.read(bounded).controls[-1].update_index == committed.update_index
    altered = replace(bounded, policy_state=policy.to_dict(), policy_fingerprint=policy.fingerprint())
    with pytest.raises(ValueError, match="policy binding"):
        coordinator.read(altered)


@pytest.mark.parametrize("threshold", (float("nan"), float("inf")))
def test_replication_coordinator_rejects_nonfinite_threshold(threshold):
    executor, _checkpoint, _ = fixture()
    roles = ReplicationRoleManifest.from_seed_registries({"control": (20, 21)})
    with pytest.raises(ValueError, match="threshold"):
        ReplicationPermanentPassCoordinator(executor.adapter.registry, roles, threshold=threshold)
