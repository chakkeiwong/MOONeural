"""Source-bound host state-machine, checkpoint and role conformance."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from mooneural.training.generic_permanent_pass import (
    POLICY_NAME,
    PermanentPassCoordinator,
    PermanentPassState,
)
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    ControlEvaluation,
    EvaluationRequest,
    PolicyView,
    RoleBinding,
    RoleManifest,
    TaskDefinition,
    TaskRegistry,
    ValidationEvaluation,
    canonical_json,
    stable_hash,
)
from tests.support.task_order import SEVEN_TASK_ORDER
from tests.support.working_set import (
    initialize_permanent_pass_state,
    permanent_pass_rotation_indices,
    promote_permanent_passes,
)

REGISTRY = TaskRegistry(
    tuple(TaskDefinition(task, index) for index, task in enumerate(SEVEN_TASK_ORDER))
)
ROLES = RoleManifest((RoleBinding("control", 22, "fake-control", "v1"),))
POLICY = PolicyView((0.5, -0.2), "fake-only")


def control(values, policy=POLICY, tasks=SEVEN_TASK_ORDER):
    return ControlEvaluation(
        dict(zip(tasks, values, strict=True)),
        request=EvaluationRequest(
            "control",
            "fake-control",
            seeds=(22,),
            task_ids=tasks,
            policy_fingerprint=policy.fingerprint(),
        ),
    )


def initial(values, update_index=0, registry=REGISTRY):
    return PermanentPassState.initialize(
        registry,
        control(values, tasks=registry.task_ids),
        POLICY,
        ROLES,
        threshold=0.04,
        update_index=update_index,
    )


def checkpoint(update_index=27000):
    return CheckpointState(
        "parent",
        update_index,
        POLICY.to_dict(),
        {
            "iteration": 17,
            "first_moment": [0.1, -0.3],
            "second_moment": [0.02, 0.7],
            "learning_rate": 0.001,
            "other_slot": {"saved": [5.0]},
        },
        {
            "preferred": "pcgrad",
            "rates": {"pcgrad": 0.002},
            "state": None,
            "opaque": [3.0],
        },
        {"pcgrad_seed": 20260722, "next_seed_index": update_index, "other_rng": [1, 2]},
        metadata={"source_parent": {"round": 90}},
    )


def test_source_comparator_hash_is_frozen():
    path = Path(__file__).parents[2] / "tests/support/working_set.py"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "d2721cc09c452234a01dfb43cb9a5e8cf9da8d443f23f0e54fd2d39f94d340d0"
    )


@pytest.mark.parametrize("mask", range(128))
def test_every_source_membership_mask_and_complete_rotation_cycles(mask):
    values = [0.04 if mask & (1 << index) else 0.2 for index in range(7)]
    source, _record = initialize_permanent_pass_state(
        dict(zip(SEVEN_TASK_ORDER, values, strict=True))
    )
    state = initial(values)
    count = len(source.unresolved)
    period = math.comb(count, min(3, count))
    offsets = (0,) if count == 0 else (*range(2 * period), 27000, 99000, 114000)
    for offset in offsets:
        current = replace(state, update_index=offset)
        partition = current.partition
        active, temporary = permanent_pass_rotation_indices(source, offset)
        assert tuple(partition["permanent"]) == source.permanent
        assert tuple(partition["unresolved"]) == source.unresolved
        assert partition["active"] == [SEVEN_TASK_ORDER[index] for index in active]
        assert partition["temporary_constraints"] == [
            SEVEN_TASK_ORDER[index] for index in temporary
        ]
        assert partition["constraints"] == [
            task for task in SEVEN_TASK_ORDER if task not in partition["active"]
        ]
        assert len(partition["active"]) == min(3, count)
        assert (
            PermanentPassState.from_dict(
                json.loads(canonical_json(current.to_dict()))
            ).to_dict()
            == current.to_dict()
        )


def test_gate_is_inclusive_and_has_no_hysteresis():
    values = [
        0.04,
        math.nextafter(0.04, math.inf),
        math.nextafter(0.04, 0.0),
        0.032,
        0.039,
        0.0,
        1.0,
    ]
    state = initial(values, update_index=27000)
    assert state.permanent == tuple(
        SEVEN_TASK_ORDER[index] for index in (0, 2, 3, 4, 5)
    )
    assert state.partition["active"] == [SEVEN_TASK_ORDER[1], SEVEN_TASK_ORDER[6]]
    assert len(state.partition["constraints"]) == 5


def test_new_passes_leave_immediately_regression_never_reactivates():
    values = [0.03, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    state = initial(values, update_index=27000).committed_update()
    source, _record = initialize_permanent_pass_state(
        dict(zip(SEVEN_TASK_ORDER, values, strict=True))
    )
    for values in (
        [0.5, 0.04, 0.04, 0.04, 0.2, 0.2, 0.2],
        [0.9, 0.9, 0.9, 0.9, 0.03, 0.03, 0.2],
    ):
        state = state.boundary(control(values), POLICY, ROLES)
        source, transition = promote_permanent_passes(
            source, dict(zip(SEVEN_TASK_ORDER, values, strict=True))
        )
        assert state.permanent == source.permanent
        assert (
            state.history()[-1]["entered_permanent"] == transition["entered_permanent"]
        )
        assert set(state.partition["permanent"]) <= set(state.partition["constraints"])
        assert not set(state.partition["permanent"]) & set(state.partition["active"])
        state = state.committed_update()
    assert state.partition["active"] == [SEVEN_TASK_ORDER[-1]]
    assert len(state.partition["constraints"]) == 6
    assert state.history()[-1]["regressed_permanent_diagnostic"] == list(
        SEVEN_TASK_ORDER[:4]
    )


@pytest.mark.parametrize("count", (1, 2, 3, 4, 8, 11, 20))
def test_policy_is_generic_in_task_count_and_names(count):
    registry = TaskRegistry(
        tuple(
            TaskDefinition(f"other-equation-{index}", index) for index in range(count)
        )
    )
    state = initial([0.2] * count, update_index=299, registry=registry)
    assert len(state.partition["active"]) == min(3, count)
    assert len(state.partition["temporary_constraints"]) == max(count - 3, 0)
    assert (
        PermanentPassState.from_dict(
            json.loads(canonical_json(state.to_dict()))
        ).task_ids
        == registry.task_ids
    )


def test_empty_pool_and_invalid_boundary_timing_are_refused():
    complete = initial([0.04] * 7, update_index=27000)
    assert complete.complete
    assert complete.partition["active"] == []
    assert complete.partition["constraints"] == list(SEVEN_TASK_ORDER)
    with pytest.raises(ValueError, match="no optimizer"):
        complete.committed_update()
    with pytest.raises(ValueError, match="completed"):
        replace(complete, update_index=27001)
    state = initial([0.2] * 7)
    with pytest.raises(ValueError, match="requires new updates"):
        state.boundary(control([0.1] * 7), POLICY, ROLES)


@pytest.mark.parametrize("index", (-1, True, 2.0, "2"))
def test_invalid_global_update_refused(index):
    with pytest.raises(ValueError, match="global update"):
        initial([0.2] * 7, update_index=index)


@pytest.mark.parametrize("threshold", (0.0, -0.1, float("nan"), float("inf"), True))
def test_invalid_threshold_refused(threshold):
    with pytest.raises(ValueError, match="threshold"):
        replace(initial([0.2] * 7), threshold=threshold)


@pytest.mark.parametrize(
    "field,value",
    (
        ("seed", 99),
        ("scale_version", "wrong"),
        ("target_id", "wrong"),
        ("estimator_version", "wrong"),
        ("policy_fingerprint", "a" * 64),
        ("task_ids", tuple(reversed(SEVEN_TASK_ORDER))),
    ),
)
def test_control_role_target_scale_policy_and_order_are_bound(field, value):
    evaluation = control([0.2] * 7)
    changes = {field: value} if field != "seed" else {"seeds": (value,)}
    evaluation = replace(evaluation, request=replace(evaluation.request, **changes))
    with pytest.raises(ValueError):
        PermanentPassState.initialize(
            REGISTRY, evaluation, POLICY, ROLES, threshold=0.04, update_index=0
        )


def test_unbound_control_negative_mse_and_validation_cannot_promote():
    for invalid in (
        ControlEvaluation(dict.fromkeys(SEVEN_TASK_ORDER, 0.01)),
        control([-0.01] * 7),
        ValidationEvaluation(
            dict.fromkeys(SEVEN_TASK_ORDER, 0.01), dict.fromkeys(SEVEN_TASK_ORDER, 1)
        ),
    ):
        with pytest.raises(ValueError):
            PermanentPassState.initialize(
                REGISTRY, invalid, POLICY, ROLES, threshold=0.04, update_index=0
            )


@pytest.mark.parametrize(
    "damage",
    ("schema", "policy_name", "active", "permanent", "history", "extra", "hash"),
)
def test_serialized_state_rejects_reactivation_and_schema_drift(damage):
    payload = initial([0.03, 0.03, 0.2, 0.2, 0.2, 0.2, 0.2]).to_dict()
    if damage in ("schema", "policy_name"):
        payload[damage] = "paired_student_t_replay"
    elif damage in ("active", "permanent"):
        payload["partition"][damage] = list(SEVEN_TASK_ORDER[:3])
    elif damage == "history":
        payload["history"][0]["entered_permanent"] = []
    elif damage == "extra":
        payload["required_active_improvements"] = 2
    if damage != "hash":
        payload["state_hash"] = stable_hash(
            {key: value for key, value in payload.items() if key != "state_hash"}
        )
    else:
        payload["state_hash"] = "a" * 64
    with pytest.raises(ValueError):
        PermanentPassState.from_dict(payload)


def test_checkpoint_boundary_resume_and_transfer_preserve_all_state():
    coordinator = PermanentPassCoordinator(REGISTRY, ROLES, threshold=0.04)
    parent = coordinator.initialize(checkpoint(), control([0.03] * 5 + [0.2] * 2))
    proposed = replace(parent, update_index=parent.update_index + 1)
    updated = coordinator.commit(parent, proposed)
    updated, record = coordinator.boundary(updated, control([0.9] * 5 + [0.04, 0.2]))
    restored = CheckpointState.from_dict(json.loads(canonical_json(updated.to_dict())))
    assert coordinator.read(restored).to_dict() == coordinator.read(updated).to_dict()
    assert record["after"]["active"] == [SEVEN_TASK_ORDER[-1]]
    child = coordinator.transfer(restored, attempt_id="child")
    assert child.optimizer_state == parent.optimizer_state
    assert child.method_state == parent.method_state
    assert child.rng_state == parent.rng_state
    assert child.update_index == 27001
    assert child.policy_state == parent.policy_state
    assert child.metadata == restored.metadata
    child.optimizer_state["other_slot"]["saved"][0] = 999.0
    child.method_state["opaque"][0] = 999.0
    assert parent.optimizer_state["other_slot"]["saved"] == [5.0]
    assert parent.method_state["opaque"] == [3.0]
    with pytest.raises(ValueError, match="reinitialized"):
        coordinator.initialize(restored, control([0.2] * 7))


def test_checkpoint_mismatch_and_membership_rewrite_are_refused():
    coordinator = PermanentPassCoordinator(REGISTRY, ROLES, threshold=0.04)
    state = coordinator.initialize(checkpoint(), control([0.2] * 7))
    with pytest.raises(ValueError, match="binding mismatch"):
        coordinator.read(replace(state, update_index=state.update_index + 1))
    with pytest.raises(ValueError, match="exactly once"):
        coordinator.commit(state, state)
    with pytest.raises(ValueError, match="cannot rewrite"):
        coordinator.commit(
            state, replace(state, update_index=state.update_index + 1, metadata={})
        )
    with pytest.raises(ValueError, match="manifest mismatch"):
        coordinator.read(
            replace(state, metadata={**state.metadata, "role_manifest": {}})
        )
    assert POLICY_NAME in state.metadata
