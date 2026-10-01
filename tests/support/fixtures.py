"""Upstream synthetic factories, without the DSGE campaign CLI."""
from mooneural.training.generic_training_contracts import (CheckpointState, ControlEvaluation, EvaluationRequest, PolicyView, RoleBinding, RoleManifest)

def roles():
    return RoleManifest(
        (RoleBinding("control", 20260908, "fake-policy-control", "v1"),)
    )

def fake_control(checkpoint, task_ids, values):
    return ControlEvaluation(
        dict(zip(task_ids, values, strict=True)),
        raw_records={"scripted_policy_trace_only": True},
        request=EvaluationRequest(
            "control",
            "fake-policy-control",
            seeds=(20260908,),
            policy_fingerprint=checkpoint.policy_fingerprint,
            task_ids=task_ids,
        ),
    )

def fixture(
    attempt_id="phase3-fixture", *, task_count=7, update_index=27000, preferred="cagrad"
):
    import tensorflow as tf

    from mooneural.training.generic_execution_boundary import METHODS, ExecutionSpec
    from mooneural.training.generic_permanent_pass_executor import PermanentPassExecutor
    from mooneural.training.generic_tensor_fixture import QuadraticTensorFixture

    adapter = QuadraticTensorFixture(task_count)
    dimension = max(8, task_count)
    spec = ExecutionSpec(task_count, dimension, 2, 1)
    executor = PermanentPassExecutor(adapter, spec, roles(), threshold=0.04)
    policy = PolicyView(
        tuple(0.2 + index * 0.01 for index in range(dimension)), "fake-quadratic-only"
    )
    state = CheckpointState(
        attempt_id,
        update_index,
        policy.to_dict(),
        {
            "first_moment": [0.0] * dimension,
            "second_moment": [0.001] * dimension,
            "iteration": 17,
            "learning_rate": 0.001,
            "opaque_optimizer_metadata": {"keep": [13]},
        },
        {
            "preferred": preferred,
            "rates": {
                method: 0.001 + index * 0.0001 for index, method in enumerate(METHODS)
            },
            "gradnorm_reset_per_call": True,
            "state": None,
            "opaque_method_metadata": {"keep": [17]},
        },
        {
            "pcgrad_seed": 20260722,
            "next_seed_index": update_index,
            "other_rng": [3, 5, 7],
        },
        metadata={"synthetic_only": True, "source_parent_not_available": True},
    )
    batch = (
        tf.repeat(
            tf.eye(task_count, dimension, dtype=tf.float64)[:, None, :], 2, axis=1
        ),
        tf.ones([task_count, 2], tf.float64),
        tf.ones([task_count, 2], tf.float64),
    )
    return executor, state, batch
