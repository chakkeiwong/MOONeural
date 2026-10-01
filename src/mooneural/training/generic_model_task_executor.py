"""Connect logical model adapters to the shared permanent-pass transaction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from .generic_execution_boundary import DEFAULT_ADAM
from .generic_permanent_pass_executor import (
    PermanentPassExecutor,
    PermanentPassUpdateError,
)
from .generic_training_contracts import PolicyView, TaskEvaluationTensors, stable_hash


@dataclass(frozen=True)
class ModelTaskMetadata:
    registry: object
    normalizations: tuple
    hard_check_ids: tuple[str, ...]
    hard_limits: tuple[float, ...]
    normalization_binding: Mapping[str, Any] | None = None
    normalization_hash: str | None = None
    input_fingerprint: str | None = None
    input_binding: Mapping[str, Any] | None = None

    def __post_init__(self):
        if tuple(item.task_id for item in self.normalizations) != self.registry.task_ids:
            raise ValueError("normalization registry order mismatch")
        if len(self.hard_check_ids) != len(set(self.hard_check_ids)):
            raise ValueError("hard check identifiers must be unique")
        if len(self.hard_limits) != len(self.hard_check_ids):
            raise ValueError("each hard check requires an explicit limit")
        if any(not math.isfinite(value) or value < 0 for value in self.hard_limits):
            raise ValueError("hard limits must be finite and nonnegative")
        declared_normalizations = [item.to_dict() for item in self.normalizations]
        binding = dict(self.normalization_binding or {})
        if "normalizations" in binding and binding["normalizations"] != declared_normalizations:
            raise ValueError("normalization binding does not match typed normalizations")
        binding["normalizations"] = declared_normalizations
        normalization_hash = stable_hash(binding)
        if self.normalization_hash is not None and self.normalization_hash != normalization_hash:
            raise ValueError("normalization hash does not match canonical binding")
        if self.input_binding is None:
            raise ValueError("a typed canonical input binding is required")
        input_binding = dict(self.input_binding)
        input_fingerprint = stable_hash(input_binding)
        if self.input_fingerprint is not None and self.input_fingerprint != input_fingerprint:
            raise ValueError("input fingerprint does not match canonical input binding")
        if not isinstance(normalization_hash, str) or len(normalization_hash) != 64:
            raise ValueError("a canonical normalization hash is required")
        if input_fingerprint is None or len(input_fingerprint) != 64:
            raise ValueError("a canonical input fingerprint is required")
        try:
            int(normalization_hash, 16)
            int(input_fingerprint, 16)
        except ValueError as error:
            raise ValueError("canonical bindings must be SHA-256 values") from error
        object.__setattr__(self, "normalization_hash", normalization_hash)
        object.__setattr__(self, "normalization_binding", binding)
        object.__setattr__(self, "input_fingerprint", input_fingerprint)
        object.__setattr__(self, "input_binding", input_binding)

    def validate_bindings(self):
        declared_normalizations = [item.to_dict() for item in self.normalizations]
        binding = dict(self.normalization_binding)
        if binding.get("normalizations") != declared_normalizations:
            raise ValueError("normalization binding was mutated")
        if stable_hash(binding) != self.normalization_hash:
            raise ValueError("normalization binding hash was mutated")
        if stable_hash(dict(self.input_binding)) != self.input_fingerprint:
            raise ValueError("input binding hash was mutated")


@dataclass(frozen=True)
class ModelTaskEvaluation:
    tensors: TaskEvaluationTensors
    evaluated_policy_fingerprint: str
    evaluated_update_index: int
    input_fingerprint: str
    native_values: Any
    native_rows: Any
    native_hard_max: Any

    def __post_init__(self):
        if not isinstance(self.tensors, TaskEvaluationTensors):
            raise TypeError("model evaluation must contain TaskEvaluationTensors")
        if len(self.evaluated_policy_fingerprint) != 64:
            raise ValueError("evaluated policy fingerprint must be SHA-256")
        if len(self.input_fingerprint) != 64:
            raise ValueError("evaluated input fingerprint must be SHA-256")
        try:
            int(self.evaluated_policy_fingerprint, 16)
            int(self.input_fingerprint, 16)
        except ValueError as error:
            raise ValueError("evaluation bindings must be SHA-256 values") from error
        if type(self.evaluated_update_index) is not int or self.evaluated_update_index < 0:
            raise ValueError("evaluated update index must be non-negative")
        for name, shape, expected in (
            ("native_values", (len(self.tensors.task_ids),), self.tensors.values),
            ("native_rows", (len(self.tensors.task_ids), self.tensors.policy_dimension), self.tensors.rows),
            ("native_hard_max", (len(self.tensors.hard_check_ids),), self.tensors.hard_max),
        ):
            value = getattr(self, name)
            if not tf.is_tensor(value) or value.dtype != tf.float64:
                raise ValueError("model evaluation must expose native TensorFlow float64 outputs")
            if tuple(value.shape) != shape:
                raise ValueError("native model output shape mismatch")
            value = tf.identity(value)
            if not bool(tf.reduce_all(value == tf.constant(expected, tf.float64)).numpy()):
                raise ValueError("native model output differs from retained task values")
            object.__setattr__(self, name, value)

    @property
    def actual_dtype(self):
        return self.native_values.dtype.name

    @property
    def actual_backend(self):
        return "tensorflow"

    @property
    def backend_verified(self):
        return True

    @property
    def task_ids(self):
        return self.tensors.task_ids

    @property
    def values(self):
        return self.tensors.values

    @property
    def rows(self):
        return self.tensors.rows

    @property
    def hard_max(self):
        return self.tensors.hard_max

    @property
    def hard_check_ids(self):
        return self.tensors.hard_check_ids


def make_normalized_task_evaluation(metadata, spec):
    """Validate normalized inputs once; raw-coordinate results are unavailable."""
    if (len(metadata.registry.tasks) != spec.task_count
            or len(metadata.hard_check_ids) != spec.hard_check_count):
        raise ValueError("model task registry does not match execution signature")
    limits = tf.constant(metadata.hard_limits, tf.float64)
    signature = [
        tf.TensorSpec([spec.parameter_dim], tf.float64, "parameters"),
        tf.TensorSpec([spec.parameter_dim], tf.float64, "evaluated_parameters"),
        tf.TensorSpec([spec.task_count], tf.float64, "normalized_values"),
        tf.TensorSpec([spec.task_count, spec.parameter_dim], tf.float64, "normalized_rows"),
        tf.TensorSpec([spec.hard_check_count], tf.float64, "hard_max"),
        tf.TensorSpec([], tf.int64, "evaluated_update"),
        tf.TensorSpec([], tf.int64, "update_index"),
    ]

    @tf.function(input_signature=signature, autograph=False, jit_compile=True)
    def evaluate(parameters, evaluated_parameters, values, rows, hard_max,
                 evaluated_update, update_index):
        with tf.device(spec.device):
            valid = tf.reduce_all(parameters == evaluated_parameters)
            valid = valid & (evaluated_update == update_index) & (update_index >= 0)
            valid = valid & tf.reduce_all(tf.math.is_finite(parameters))
            valid = valid & tf.reduce_all(tf.math.is_finite(values))
            valid = valid & tf.reduce_all(tf.math.is_finite(rows))
            valid = valid & tf.reduce_all(tf.math.is_finite(hard_max))
            valid = valid & tf.reduce_all(values >= 0)
            valid = valid & tf.reduce_all((hard_max >= 0) & (hard_max <= limits))
            return None, None, values, rows, hard_max, tf.ones([spec.task_count], tf.bool), valid

    return evaluate


class ModelTaskExecutor:
    """Host adapter evaluation followed by the existing compiled transaction.

    The logical adapter call is diagnostic until its model-specific graph and
    runtime are qualified. This wrapper does not claim to compile that call.
    """

    def __init__(self, metadata, spec, roles, *, threshold, model_binding,
                 adam=DEFAULT_ADAM, coordinator_factory=None,
                 update_factory=None, update_binding=None,
                 method_factory=None, method_binding=None, postproposal=None):
        if not model_binding:
            raise ValueError("explicit model/source binding is required")
        supplied_binding = dict(model_binding)
        identity_hash = supplied_binding.pop("identity_hash", None)
        if not isinstance(identity_hash, str):
            raise TypeError("canonical model identity hash is required")
        if identity_hash != stable_hash(supplied_binding):
            raise ValueError("canonical model identity hash does not match binding")
        for key, expected in (
            ("normalization_hash", metadata.normalization_hash),
            ("input_fingerprint", metadata.input_fingerprint),
        ):
            if key in supplied_binding and supplied_binding[key] != expected:
                raise ValueError(f"configured {key} conflicts with metadata")
        self.metadata = metadata
        self.spec = spec
        self.model_binding = {
            **supplied_binding,
            "normalization_hash": metadata.normalization_hash,
            "input_fingerprint": metadata.input_fingerprint,
            "identity_hash": identity_hash,
        }
        self.core = PermanentPassExecutor(
            metadata, spec, roles, threshold=threshold, adam=adam,
            evaluation_factory=make_normalized_task_evaluation,
            coordinator_factory=coordinator_factory,
            update_factory=update_factory,
            update_binding=update_binding,
            method_factory=method_factory,
            method_binding=method_binding,
            postproposal=postproposal,
            evaluation_binding={
                "schema": "generic_neural_solver.normalized_model_task_executor.v1",
                "coordinate": "training_normalized_by_model_adapter",
                "model": self.model_binding,
                "input_binding": self.metadata.input_binding,
                "hard_limits": list(metadata.hard_limits),
            },
        )

    def initialize(self, checkpoint, control):
        return self.core.initialize(checkpoint, control)

    def step(self, checkpoint, batch, model_adapter):
        self.metadata.validate_bindings()
        state = self.core.coordinator.read(checkpoint)
        self.core._validate_numerical_state(checkpoint)
        if state.complete:
            return self.core.step(checkpoint, ())
        identity = model_adapter.adapter_identity()
        if any(
            identity.get(key) != value for key, value in self.model_binding.items()
        ):
            raise ValueError("model adapter/source binding mismatch")
        if (batch.task_ids != self.metadata.registry.task_ids
                or batch.hard_check_ids != self.metadata.hard_check_ids
                or batch.policy_dimension != self.spec.parameter_dim
                or batch.sample_count != self.spec.batch_size
                or batch.backend != "tensorflow" or batch.dtype != "float64"):
            raise ValueError("model batch does not match executor signature")
        normalization_hash = batch.metadata.get("normalization_hash")
        input_fingerprint = batch.metadata.get("input_fingerprint")
        if batch.metadata.get("input_binding") != self.metadata.input_binding:
            raise ValueError("model batch input binding mismatch")
        if normalization_hash != self.metadata.normalization_hash:
            raise ValueError("model batch normalization binding mismatch")
        if input_fingerprint != self.metadata.input_fingerprint:
            raise ValueError("model batch input fingerprint mismatch")
        policy = PolicyView.from_dict(checkpoint.policy_state)
        snapshot = self._snapshot(model_adapter)
        try:
            evaluation = model_adapter.compute_task_values_and_gradients(
                policy, batch, checkpoint.update_index
            )
            if not isinstance(evaluation, ModelTaskEvaluation):
                raise TypeError("model adapter must return ModelTaskEvaluation")
            evaluation.tensors.validate_batch(batch, normalization_hash)
            if evaluation.evaluated_policy_fingerprint != policy.fingerprint():
                raise ValueError("model evaluation policy freshness binding mismatch")
            if evaluation.evaluated_update_index != checkpoint.update_index:
                raise ValueError("model evaluation update freshness binding mismatch")
            if evaluation.input_fingerprint != input_fingerprint:
                raise ValueError("model evaluation input binding mismatch")
            evaluation_tensors = evaluation.tensors
            committed, event = self.core.step(checkpoint, (
                tf.constant(policy.values, tf.float64),
                evaluation.native_values,
                evaluation.native_rows,
                evaluation.native_hard_max,
                tf.constant(checkpoint.update_index, tf.int64),
            ), **({"objective_context": batch.to_dict()} if self.core.postproposal is not None else {}))
            self._commit(model_adapter, committed)
        except Exception as error:
            try:
                self._restore(model_adapter, snapshot)
            except Exception as restore_error:
                self._taint(model_adapter, restore_error)
                raise PermanentPassUpdateError(
                    f"model task transaction failed and restore was not proven: {error}; "
                    f"restore error: {restore_error}",
                    checkpoint,
                    {"event": "model_task_restore_failed", "error": str(error),
                     "restore_error": str(restore_error)},
                ) from restore_error
            if isinstance(error, PermanentPassUpdateError):
                raise
            raise PermanentPassUpdateError(
                f"model task transaction failed: {error}",
                checkpoint,
                {"event": "model_task_transaction_failed", "error": str(error)},
            ) from error
        event["model_evaluation"] = {
            "policy_fingerprint": evaluation.evaluated_policy_fingerprint,
            "update_index": checkpoint.update_index,
            "batch_fingerprint": stable_hash(batch.to_dict()),
            "normalization_hash": normalization_hash,
            "input_fingerprint": input_fingerprint,
            "coordinate": evaluation_tensors.coordinate,
            "actual_dtype": evaluation.actual_dtype,
            "actual_backend": evaluation.actual_backend,
            "backend_verified": evaluation.backend_verified,
            "task_values": list(evaluation_tensors.values),
        }
        return committed, event

    @staticmethod
    def _snapshot(model_adapter: Any):
        snapshot_method = getattr(model_adapter, "snapshot_state", None)
        if callable(snapshot_method):
            snapshot = snapshot_method()
            if snapshot is None:
                raise ValueError("mutable model adapter returned an empty snapshot")
            return snapshot
        identity = model_adapter.adapter_identity()
        if identity.get("state_mode") == "functional":
            return None
        raise ValueError("model adapter must provide snapshot_state or functional state mode")

    @staticmethod
    def _restore(model_adapter: Any, snapshot):
        if snapshot is None:
            return
        restore_method = getattr(model_adapter, "restore_state", None)
        if not callable(restore_method):
            raise TypeError("model adapter snapshot cannot be restored")
        receipt = restore_method(snapshot)
        if receipt is not True:
            raise ValueError("model adapter did not prove state restoration")

    @staticmethod
    def _commit(model_adapter: Any, checkpoint):
        commit_method = getattr(model_adapter, "commit_policy", None)
        if callable(commit_method):
            policy = PolicyView.from_dict(checkpoint.policy_state)
            receipt = commit_method(policy)
            if receipt is False:
                raise ValueError("model adapter did not prove policy commit")
            readback = getattr(model_adapter, "readback_policy", None)
            if callable(readback):
                observed = readback()
                if (
                    not isinstance(observed, PolicyView)
                    or len(observed.values) != len(policy.values)
                    or observed.values != policy.values
                ):
                    raise ValueError("model adapter policy commit readback mismatch")
            elif not isinstance(receipt, Mapping) or receipt != {
                "policy_fingerprint": policy.fingerprint(),
                "policy_dimension": len(policy.values),
            }:
                raise ValueError("mutable model adapter must return a verified commit receipt")
        elif model_adapter.adapter_identity().get("state_mode") != "functional":
            raise ValueError("mutable model adapter must provide commit_policy")

    @staticmethod
    def _taint(model_adapter: Any, error: Exception):
        marker = getattr(model_adapter, "mark_unusable", None)
        if callable(marker):
            marker(str(error))


__all__ = [
    "ModelTaskEvaluation",
    "ModelTaskExecutor",
    "ModelTaskMetadata",
    "make_normalized_task_evaluation",
]
