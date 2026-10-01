"""One constrained six-task update; a mechanics example, not model training.

Run after installing MOONeural, with CUDA_VISIBLE_DEVICES=-1 and
TF_FORCE_GPU_ALLOW_GROWTH=true. Real applications provide their own adapter,
independent control measurements and scientifically justified task limits.
"""

import json
import os


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
        raise RuntimeError("This CPU example requires explicit GPU hiding and memory-growth environment")
    import numpy as np
    import tensorflow as tf

    from mooneural.training.generic_execution_boundary import METHODS, ExecutionSpec, numerical_batch_binding
    from mooneural.training.generic_permanent_pass_executor import PermanentPassExecutor
    from mooneural.training.generic_postproposal import GuardedPostproposal
    from mooneural.training.generic_tensor_fixture import QuadraticTensorFixture
    from mooneural.training.generic_training_contracts import (
        CheckpointState, ControlEvaluation, EvaluationRequest, PolicyView, RoleBinding, RoleManifest,
    )

    adapter = QuadraticTensorFixture(task_count=6)
    spec = ExecutionSpec(6, 8, 2, 1)
    role = RoleBinding("control", 101, "synthetic-demo", "v1")
    policy = PolicyView((.2,)*8, "quadratic-demo")
    parent = CheckpointState("demo", 0, policy.to_dict(),
        {"first_moment": [0.]*8, "second_moment": [0.]*8, "iteration": 0, "learning_rate": 1e-3},
        {"preferred": "mgda", "rates": dict.fromkeys(METHODS, 1e-3), "gradnorm_reset_per_call": True, "state": None},
        {"pcgrad_seed": 20260722, "next_seed_index": 0})
    batch = (tf.repeat(tf.eye(6, 8, dtype=tf.float64)[:, None, :], 2, 1),
             tf.zeros([6, 2], tf.float64), tf.ones([6, 2], tf.float64))

    def losses(parameters, values, update_index):
        features, targets, weights = values
        prediction = tf.einsum("tbd,d->tb", features, tf.constant(parameters, tf.float64))
        raw = tf.reduce_mean(weights*tf.square(prediction-targets), 1).numpy()
        return raw, raw  # D=1 for this synthetic example.

    guard = GuardedPostproposal(losses, numerical_batch_binding, [1.]*6, [.01]*6,
                               binding={"example": "quadratic-v1"})
    executor = PermanentPassExecutor(adapter, spec, RoleManifest((role,)), threshold=.01, postproposal=guard)
    control = ControlEvaluation(dict.fromkeys(adapter.registry.task_ids, .04), request=EvaluationRequest(
        "control", role.target_id, seeds=(role.seed,), task_ids=adapter.registry.task_ids,
        policy_fingerprint=parent.policy_fingerprint))
    parent = executor.initialize(parent, control)
    result, event = executor.step(parent, batch)
    assert event["committed"] and np.max(event["actual_constraint_dots"]) <= 1e-10
    print(json.dumps({"committed": True, "update_index": result.update_index,
                      "method": event["method"], "all_tasks_satisfied": False,
                      "scope": "synthetic one-step mechanics only"}))


if __name__ == "__main__":
    main()
