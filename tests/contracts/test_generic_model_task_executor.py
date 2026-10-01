"""Synthetic adapter integration; no economic training or endpoint claims."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_execution_boundary import METHODS
from mooneural.training.generic_model_task_executor import (
    ModelTaskEvaluation,
    ModelTaskExecutor,
    ModelTaskMetadata,
    make_normalized_task_evaluation,
)
from mooneural.training.generic_permanent_pass_executor import (
    PermanentPassExecutor,
    PermanentPassUpdateError,
)
from tests.support.fixtures import (
    fake_control,
    fixture,
    roles,
)
from mooneural.training.generic_tensor_fixture import QuadraticTensorFixture
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    PolicyView,
    TaskBatchTensors,
    TaskEvaluationTensors,
    stable_hash,
)


class LogicalQuadraticAdapter:
    def __init__(self, tensor_adapter, tensors, normalization_hash):
        self.tensor_adapter = tensor_adapter
        self.tensors = tensors
        self.normalization_hash = normalization_hash
        self.input_fingerprint = stable_hash({"inputs": "fixed-unit-fixture"})
        self.calls = 0

    def adapter_identity(self):
        payload = {
            "model": "logical-quadratic-unit-fixture",
            "normalization_hash": self.normalization_hash,
            "input_fingerprint": self.input_fingerprint,
            "state_mode": "functional",
        }
        return {**payload, "identity_hash": stable_hash(payload)}

    def compute_task_values_and_gradients(self, policy, batch, update_index):
        self.calls += 1
        raw_values, raw_rows, hard_max, valid, connected = self.tensor_adapter.compute_task_tensors(
            tf.constant(policy.values, tf.float64), *self.tensors,
            tf.constant(update_index, tf.int64),
        )
        assert bool(valid.numpy()) and bool(tf.reduce_all(connected).numpy())
        factors = tf.constant([item.training.factor for item in self.tensor_adapter.normalizations], tf.float64)
        tensors = TaskEvaluationTensors(
            batch.task_ids, tuple((raw_values * factors).numpy().tolist()),
            tuple(tuple(row) for row in (raw_rows * factors[:, None]).numpy().tolist()),
            tuple(hard_max.numpy().tolist()), batch.policy_dimension,
            backend="tensorflow", normalization_hash=self.normalization_hash,
            hard_check_ids=batch.hard_check_ids,
        )
        return ModelTaskEvaluation(
            tensors=tensors,
            evaluated_policy_fingerprint=policy.fingerprint(),
            evaluated_update_index=update_index,
            input_fingerprint=self.input_fingerprint,
            native_values=raw_values * factors,
            native_rows=raw_rows * factors[:, None],
            native_hard_max=hard_max,
        )


class MutableLogicalAdapter(LogicalQuadraticAdapter):
    def __init__(self, tensor_adapter, tensors, normalization_hash):
        super().__init__(tensor_adapter, tensors, normalization_hash)
        self.worker_policy = ()
        self.counter = 0
        self.failure = None

    def adapter_identity(self):
        payload = {**super().adapter_identity(), "state_mode": "snapshot_commit"}
        payload.pop("identity_hash")
        return {**payload, "identity_hash": stable_hash(payload)}

    def snapshot_state(self):
        return self.worker_policy, self.counter

    def restore_state(self, snapshot):
        self.worker_policy, self.counter = snapshot
        return True

    def compute_task_values_and_gradients(self, policy, batch, update_index):
        self.worker_policy = policy.values
        self.counter += 1
        if self.failure == "evaluate":
            raise RuntimeError("intentional evaluation failure")
        return super().compute_task_values_and_gradients(policy, batch, update_index)

    def commit_policy(self, policy):
        self.worker_policy = policy.values
        if self.failure == "commit":
            self.worker_policy = (float("nan"),)
            raise RuntimeError("intentional commit failure")
        return {
            "policy_fingerprint": policy.fingerprint(),
            "policy_dimension": len(policy.values),
        }

    def readback_policy(self):
        return PolicyView(tuple(self.worker_policy), "fixture-policy")


class SilentNoOpCommitAdapter(MutableLogicalAdapter):
    def commit_policy(self, policy):
        return {
            "policy_fingerprint": policy.fingerprint(),
            "policy_dimension": len(policy.values),
        }


class RestoreFailingMutableAdapter(MutableLogicalAdapter):
    def __init__(self, tensor_adapter, tensors, normalization_hash):
        super().__init__(tensor_adapter, tensors, normalization_hash)
        self.restore_failure = False
        self.unusable_reason = None

    def restore_state(self, snapshot):
        if self.restore_failure:
            raise RuntimeError("intentional restore failure")
        return super().restore_state(snapshot)

    def mark_unusable(self, reason):
        self.unusable_reason = reason


@pytest.fixture(scope="module")
def system():
    previous, initial, tensors = fixture()
    metadata_adapter = QuadraticTensorFixture(7, training_factors=(0.5, 2., 3., 4., 5., 6., 7.))
    dense = PermanentPassExecutor(metadata_adapter, previous.spec, roles(), threshold=0.04)
    metadata = ModelTaskMetadata(metadata_adapter.registry, metadata_adapter.normalizations,
                                 metadata_adapter.hard_check_ids, (1.0,),
                                 input_fingerprint=stable_hash({"inputs": "fixed-unit-fixture"}),
                                 input_binding={"inputs": "fixed-unit-fixture"})
    normalization_hash = metadata.normalization_hash
    batch = TaskBatchTensors(metadata.registry.task_ids, dense.spec.batch_size,
                            dense.spec.parameter_dim, backend="tensorflow",
                            metadata={"normalization_hash": normalization_hash,
                                      "input_fingerprint": metadata.input_fingerprint,
                                      "input_binding": metadata.input_binding,
                                      "inputs": "fixed-unit-fixture"},
                            hard_check_ids=metadata.hard_check_ids)
    logical = LogicalQuadraticAdapter(metadata_adapter, tensors, normalization_hash)
    executor = ModelTaskExecutor(metadata, dense.spec, roles(), threshold=0.04,
                                 model_binding=logical.adapter_identity())
    return dense, executor, logical, initial, tensors, batch


def initialize(system, preferred="cagrad", permanent=6):
    dense, executor, logical, initial, tensors, batch = system
    initial = replace(initial, method_state={**initial.method_state, "preferred": preferred})
    control = fake_control(initial, batch.task_ids, [0.04] * permanent + [0.2] * (7 - permanent))
    return (dense.initialize(initial, control), executor.initialize(initial, control),
            dense, executor, logical, tensors, batch)


def test_update_backend_model_executor_threads_pair_and_stops_without_evaluation(system):
    _dense, existing, logical, initial, _tensors, batch = system
    factory = Mock(side_effect=AssertionError("unexpected numerical factory call"))
    custom = ModelTaskExecutor(
        existing.metadata, existing.spec, roles(), threshold=0.04,
        model_binding=logical.adapter_identity(), update_factory=factory,
        update_binding={"profile": "sentinel"},
    )
    assert custom.core.update_factory is factory
    assert custom.core.binding != existing.core.binding
    state = custom.initialize(initial, fake_control(initial, batch.task_ids, [0.04] * 7))
    before_calls = logical.calls
    stopped, event = custom.step(state, batch, logical)
    assert stopped is state and event["update_calls"] == 0
    assert logical.calls == before_calls
    factory.assert_not_called()


def test_update_backend_model_executor_refuses_missing_binding(system):
    _dense, existing, logical, _initial, _tensors, _batch = system
    with pytest.raises(ValueError, match="custom update requires"):
        ModelTaskExecutor(existing.metadata, existing.spec, roles(), threshold=0.04,
                          model_binding=logical.adapter_identity(), update_factory=Mock())


@pytest.mark.parametrize("method", METHODS)
def test_logical_adapter_matches_existing_transaction_without_rescaling(system, method):
    dense_state, model_state, dense, executor, logical, tensors, batch = initialize(system, method)
    expected, expected_event = dense.step(dense_state, tensors)
    observed, event = executor.step(model_state, batch, logical)
    np.testing.assert_allclose(observed.policy_state["values"], expected.policy_state["values"], rtol=1e-12, atol=1e-14)
    for name in ("first_moment", "second_moment", "learning_rate", "iteration"):
        np.testing.assert_allclose(observed.optimizer_state[name], expected.optimizer_state[name], rtol=1e-12, atol=1e-14)
    assert observed.method_state == expected.method_state
    assert observed.rng_state == expected.rng_state
    assert event["method"] == expected_event["method"]
    assert event["before"] == expected_event["before"]
    assert observed.update_index == expected.update_index == 27001
    assert event["model_evaluation"]["normalization_hash"] == logical.normalization_hash
    assert event["model_evaluation"]["batch_fingerprint"] == stable_hash(batch.to_dict())
    assert event["model_evaluation"]["policy_fingerprint"] == model_state.policy_fingerprint


def test_resume_repeats_the_next_transaction_and_uses_one_evaluation_trace(system):
    _, initial, _, executor, logical, _, batch = initialize(system, permanent=0)
    current, _ = executor.step(initial, batch, logical)
    resumed = CheckpointState.from_dict(current.to_dict())
    first, first_event = executor.step(current, batch, logical)
    second, second_event = executor.step(resumed, batch, logical)
    assert first.to_dict() == second.to_dict()
    assert first_event == second_event
    boundaries = list(executor.core.boundaries.values())
    assert len({id(boundary.evaluate) for boundary in boundaries}) == 1
    assert boundaries[0].evaluate.experimental_get_tracing_count() == 1


def test_terminal_state_skips_the_model_and_all_kernels(system):
    _, initial, _, executor, logical, _, batch = initialize(system, permanent=7)
    calls = logical.calls
    result, event = executor.step(initial, batch, logical)
    assert result is initial and event["update_calls"] == 0
    assert logical.calls == calls


def test_wrong_normalization_is_refused_before_optimizer(system):
    _, initial, _, executor, logical, _, batch = initialize(system)
    batch = replace(batch, metadata={**batch.metadata, "normalization_hash": "0" * 64})
    before = initial.to_dict()
    with pytest.raises(ValueError, match="normalization binding"):
        executor.step(initial, batch, logical)
    assert initial.to_dict() == before


def test_wrong_model_is_refused_before_evaluation(system):
    _, initial, _, executor, logical, _, batch = initialize(system)
    wrong = LogicalQuadraticAdapter(logical.tensor_adapter, logical.tensors, logical.normalization_hash)
    wrong.adapter_identity = lambda: {"model": "different-model"}
    with pytest.raises(ValueError, match="source binding"):
        executor.step(initial, batch, wrong)
    assert wrong.calls == 0


@pytest.mark.parametrize("damage", ("old_policy", "old_update", "negative_mse", "hard_violation", "nonfinite_row"))
def test_compiled_transport_rejects_stale_or_invalid_inputs(system, damage):
    _, executor, _, _, _, _ = system
    evaluate = make_normalized_task_evaluation(executor.metadata, executor.spec)
    parameters = tf.ones([8], tf.float64)
    evaluated_parameters = tf.zeros([8], tf.float64) if damage == "old_policy" else parameters
    values = tf.fill([7], tf.constant(-1. if damage == "negative_mse" else 0.2, tf.float64))
    rows = tf.fill([7, 8], tf.constant(float("nan") if damage == "nonfinite_row" else 1., tf.float64))
    hard_max = tf.constant([2. if damage == "hard_violation" else 0.], tf.float64)
    result = evaluate(parameters, evaluated_parameters, values, rows, hard_max,
                      tf.constant(1 if damage == "old_update" else 2, tf.int64), tf.constant(2, tf.int64))
    assert result[0] is None and result[1] is None
    assert not bool(result[-1].numpy())


def test_hard_violation_rolls_back_transaction(system):
    _, initial, _, executor, logical, _, batch = initialize(system)
    bad_metadata = replace(executor.metadata, hard_limits=(0.,))
    guarded = ModelTaskExecutor(bad_metadata, executor.spec, roles(), threshold=0.04,
                               model_binding=logical.adapter_identity())
    _, parent, _ = fixture()
    state = guarded.initialize(parent, fake_control(parent, batch.task_ids, [0.04] * 6 + [0.2]))
    with pytest.raises(PermanentPassUpdateError, match="invalid task evaluation") as caught:
        guarded.step(state, batch, logical)
    failed = caught.value.checkpoint
    assert failed.policy_state == state.policy_state
    assert failed.optimizer_state == state.optimizer_state
    assert failed.update_index == initial.update_index
    assert caught.value.event["method_events"] == []


def test_custom_evaluation_requires_binding_and_cannot_resume_dense_checkpoint(system):
    dense_state, _, dense, executor, _, _, _ = initialize(system)
    with pytest.raises(ValueError, match="checkpoint binding"):
        PermanentPassExecutor(dense.adapter, dense.spec, roles(), threshold=0.04,
                              evaluation_factory=make_normalized_task_evaluation)
    with pytest.raises(ValueError, match="binding mismatch"):
        executor.core._validate_numerical_state(dense_state)


def test_stale_policy_or_update_proof_is_refused_before_core(system, monkeypatch):
    _, initial, _, executor, logical, _, batch = initialize(system)
    original = logical.compute_task_values_and_gradients

    def stale(policy, model_batch, update_index):
        evaluation = original(policy, model_batch, update_index)
        return replace(evaluation, evaluated_policy_fingerprint="0" * 64)

    logical.compute_task_values_and_gradients = stale
    monkeypatch.setattr(executor.core, "step", lambda *args: pytest.fail("core was called"))
    with pytest.raises(PermanentPassUpdateError, match="freshness"):
        executor.step(initial, batch, logical)


def test_canonical_binding_rejects_a_self_consistent_wrong_input_hash(system):
    _, initial, _, executor, logical, _, batch = initialize(system)
    wrong_hash = stable_hash({"inputs": "different-fixture"})
    wrong_batch = replace(batch, metadata={**batch.metadata, "input_fingerprint": wrong_hash})
    with pytest.raises(ValueError, match="input fingerprint"):
        executor.step(initial, wrong_batch, logical)


def test_typed_normalization_binding_cannot_be_self_consistently_wrong(system):
    _, _, _, executor, _, _, _ = initialize(system)
    binding = dict(executor.metadata.normalization_binding)
    normalizations = list(binding["normalizations"])
    normalizations[0] = {**normalizations[0], "task_id": "wrong-task"}
    binding["normalizations"] = normalizations
    with pytest.raises(ValueError, match="normalization binding does not match"):
        ModelTaskMetadata(
            executor.metadata.registry,
            executor.metadata.normalizations,
            executor.metadata.hard_check_ids,
            executor.metadata.hard_limits,
            normalization_binding=binding,
            normalization_hash=stable_hash(binding),
            input_fingerprint=executor.metadata.input_fingerprint,
            input_binding=executor.metadata.input_binding,
        )


@pytest.mark.parametrize("binding_name", ("normalization_binding", "input_binding"))
def test_mutated_nested_metadata_binding_is_refused(system, binding_name):
    _, initial, _, executor, logical, _, batch = initialize(system)
    if binding_name == "normalization_binding":
        target = executor.metadata.normalization_binding["normalizations"][0]
        original = target["task_id"]
        target["task_id"] = "mutated"
    else:
        target = executor.metadata.input_binding
        original = target["inputs"]
        target["inputs"] = "mutated"
    try:
        with pytest.raises(ValueError, match="binding.*mutated"):
            executor.step(initial, batch, logical)
    finally:
        if binding_name == "normalization_binding":
            target["task_id"] = original
        else:
            target["inputs"] = original


def test_mutable_adapter_restores_on_evaluation_and_core_failure(system):
    _, executor, logical, initial, _, batch = system
    mutable = MutableLogicalAdapter(logical.tensor_adapter, logical.tensors, logical.normalization_hash)
    executor = ModelTaskExecutor(
        executor.metadata,
        executor.spec,
        roles(),
        threshold=0.04,
        model_binding=mutable.adapter_identity(),
    )
    control = fake_control(initial, batch.task_ids, [0.04] * 6 + [0.2])
    state = executor.initialize(initial, control)
    mutable.worker_policy, mutable.counter = ("sentinel",), 7
    mutable.failure = "evaluate"
    with pytest.raises(PermanentPassUpdateError, match="intentional evaluation failure"):
        executor.step(state, batch, mutable)
    assert mutable.worker_policy == ("sentinel",) and mutable.counter == 7

    mutable.failure = None
    expected, _ = executor.step(state, batch, mutable)
    assert mutable.worker_policy == tuple(expected.policy_state["values"])

    _, _, _, failing_executor, _, _, _ = initialize(system)
    bad_metadata = replace(failing_executor.metadata, hard_limits=(0.0,))
    failing_executor = ModelTaskExecutor(
        bad_metadata,
        failing_executor.spec,
        roles(),
        threshold=0.04,
        model_binding=mutable.adapter_identity(),
    )
    mutable = MutableLogicalAdapter(logical.tensor_adapter, logical.tensors, logical.normalization_hash)
    mutable.worker_policy, mutable.counter = ("sentinel",), 11
    state = failing_executor.initialize(initial, control)
    with pytest.raises(PermanentPassUpdateError, match="invalid task evaluation"):
        failing_executor.step(state, batch, mutable)
    assert mutable.worker_policy == ("sentinel",) and mutable.counter == 11


def test_mutable_adapter_commit_failure_rolls_back(system):
    _, executor, logical, initial, _, batch = system
    mutable = MutableLogicalAdapter(logical.tensor_adapter, logical.tensors, logical.normalization_hash)
    executor = ModelTaskExecutor(
        executor.metadata,
        executor.spec,
        roles(),
        threshold=0.04,
        model_binding=mutable.adapter_identity(),
    )
    state = executor.initialize(initial, fake_control(initial, batch.task_ids, [0.04] * 6 + [0.2]))
    mutable.worker_policy, mutable.counter = ("sentinel",), 13
    mutable.failure = "commit"
    with pytest.raises(PermanentPassUpdateError, match="intentional commit failure"):
        executor.step(state, batch, mutable)
    assert mutable.worker_policy == ("sentinel",) and mutable.counter == 13


def test_silent_noop_commit_is_rejected_by_policy_readback(system):
    _, executor, logical, initial, _, batch = system
    mutable = SilentNoOpCommitAdapter(
        logical.tensor_adapter, logical.tensors, logical.normalization_hash
    )
    executor = ModelTaskExecutor(
        executor.metadata,
        executor.spec,
        roles(),
        threshold=0.04,
        model_binding=mutable.adapter_identity(),
    )
    state = executor.initialize(initial, fake_control(initial, batch.task_ids, [0.04] * 6 + [0.2]))
    mutable.worker_policy, mutable.counter = ("sentinel",), 17
    with pytest.raises(PermanentPassUpdateError, match="readback"):
        executor.step(state, batch, mutable)
    assert mutable.worker_policy == ("sentinel",) and mutable.counter == 17


def test_restore_failure_taints_mutable_adapter(system):
    _, executor, logical, initial, _, batch = system
    mutable = RestoreFailingMutableAdapter(
        logical.tensor_adapter, logical.tensors, logical.normalization_hash
    )
    executor = ModelTaskExecutor(
        executor.metadata,
        executor.spec,
        roles(),
        threshold=0.04,
        model_binding=mutable.adapter_identity(),
    )
    state = executor.initialize(initial, fake_control(initial, batch.task_ids, [0.04] * 6 + [0.2]))
    mutable.restore_failure = True
    mutable.failure = "evaluate"
    with pytest.raises(PermanentPassUpdateError, match="restore was not proven"):
        executor.step(state, batch, mutable)
    assert mutable.unusable_reason == "intentional restore failure"


def test_claimed_float32_native_output_is_refused(system):
    _, initial, _, executor, logical, _, batch = initialize(system)
    original = logical.compute_task_values_and_gradients

    def wrong_dtype(policy, model_batch, update_index):
        evaluation = original(policy, model_batch, update_index)
        return replace(evaluation, native_values=tf.cast(evaluation.native_values, tf.float32))

    logical.compute_task_values_and_gradients = wrong_dtype
    with pytest.raises(PermanentPassUpdateError, match="float32|native"):
        executor.step(initial, batch, logical)
