"""Thin permanent-pass transactions over the qualified compiled numerical boundary."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import tensorflow as tf

from .generic_execution_boundary import DEFAULT_ADAM, METHODS, CompiledBoundary
from .generic_permanent_pass import PermanentPassCoordinator
from .generic_training_contracts import CheckpointState, PolicyView, stable_hash


class PermanentPassUpdateError(ValueError):
    """Failed update with unchanged policy/Adam/counter and source method-rate evidence."""

    def __init__(self, message, checkpoint, event):
        super().__init__(message)
        self.checkpoint = checkpoint
        self.event = event


class PermanentPassExecutor:
    def __init__(
        self, adapter, spec, roles, *, threshold, adam=DEFAULT_ADAM,
        evaluation_factory=None, evaluation_binding=None, coordinator_factory=None,
    ):
        if (evaluation_factory is None) != (evaluation_binding is None):
            raise ValueError("custom evaluation requires its checkpoint binding")
        if evaluation_factory is not None and not callable(evaluation_factory):
            raise TypeError("evaluation factory must be callable")
        if coordinator_factory is not None and not callable(coordinator_factory):
            raise TypeError("coordinator factory must be callable")
        self.adapter = adapter
        self.spec = spec
        self.adam = adam
        self.evaluation_factory = evaluation_factory
        self.coordinator = (
            coordinator_factory or PermanentPassCoordinator
        )(adapter.registry, roles, threshold=threshold)
        self.kernel_cache = {}
        self.boundaries = {}
        binding = {
            "spec": asdict(spec),
            "adam": asdict(adam),
            "normalizations": [item.to_dict() for item in adapter.normalizations],
            "hard_check_ids": list(adapter.hard_check_ids),
        }
        if coordinator_factory is None:
            binding["coordinator"] = {
                "module": "mooneural.training.generic_permanent_pass",
                "name": "PermanentPassCoordinator",
                "schema": "dsge_hmc.generic_permanent_pass.v1",
            }
        else:
            identity = getattr(coordinator_factory, "contract_identity", None)
            if not callable(identity):
                raise ValueError(
                    "custom coordinator factory must expose contract_identity"
                )
            coordinator_binding = identity()
            if not isinstance(coordinator_binding, dict):
                raise TypeError("coordinator contract identity must be a mapping")
            binding["coordinator"] = coordinator_binding
        if evaluation_binding is not None:
            binding["evaluation"] = evaluation_binding
        self.binding = stable_hash(binding)

        @tf.function(
            input_signature=[
                tf.TensorSpec([spec.parameter_dim], tf.float64, "committed_delta")
            ],
            autograph=False,
            jit_compile=True,
        )
        def commit_guard(committed_delta):
            with tf.device(spec.device):
                return tf.linalg.norm(committed_delta) > tf.constant(
                    1.0e-12, tf.float64
                )

        self.commit_guard = commit_guard

    def initialize(self, checkpoint, control):
        if "permanent_pass_executor_binding" in checkpoint.metadata:
            raise ValueError("executor state cannot be reinitialized")
        checkpoint = replace(
            checkpoint,
            metadata={
                **checkpoint.metadata,
                "permanent_pass_executor_binding": self.binding,
            },
        )
        self._validate_numerical_state(checkpoint)
        return self.coordinator.initialize(checkpoint, control)

    def _validate_numerical_state(self, checkpoint):
        if checkpoint.metadata.get("permanent_pass_executor_binding") != self.binding:
            raise ValueError("executor signature/normalization/Adam binding mismatch")
        policy = PolicyView.from_dict(checkpoint.policy_state)
        optimizer = checkpoint.optimizer_state
        method = checkpoint.method_state
        if len(policy.values) != self.spec.parameter_dim:
            raise ValueError("policy parameter dimension mismatch")
        if (
            not {"first_moment", "second_moment", "iteration", "learning_rate"}
            <= optimizer.keys()
        ):
            raise ValueError("complete explicit Adam state is required")
        if type(optimizer["iteration"]) is not int or optimizer["iteration"] < 0:
            raise ValueError(
                "invalid Adam iteration; it is distinct from global update"
            )
        if (
            len(optimizer["first_moment"]) != self.spec.parameter_dim
            or len(optimizer["second_moment"]) != self.spec.parameter_dim
        ):
            raise ValueError("Adam slot dimensions must match the policy")
        if (
            method is None
            or not {"rates", "preferred", "gradnorm_reset_per_call"} <= method.keys()
        ):
            raise ValueError("complete method-rate and reset semantics are required")
        if method["preferred"] not in METHODS or set(method["rates"]) != set(METHODS):
            raise ValueError("method inventory mismatch")
        rates = [optimizer["learning_rate"], *method["rates"].values()]
        if any(
            type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0.0
            for rate in rates
        ):
            raise ValueError("Adam and method rates must be finite and positive")
        if (
            method["gradnorm_reset_per_call"] is not True
            or method.get("state") is not None
        ):
            raise ValueError(
                "source requires reset-per-call GradNorm with no persistent learning state"
            )
        if (
            type(checkpoint.rng_state.get("pcgrad_seed")) is not int
            or type(checkpoint.rng_state.get("next_seed_index")) is not int
            or checkpoint.rng_state["pcgrad_seed"] != 20260722
            or checkpoint.rng_state["next_seed_index"] != checkpoint.update_index
        ):
            raise ValueError("PCGrad source seed/global-update binding mismatch")
        return policy

    def step(self, checkpoint: CheckpointState, batch):
        policy_state = self.coordinator.read(checkpoint)
        policy = self._validate_numerical_state(checkpoint)
        partition = policy_state.partition
        if policy_state.complete:
            return checkpoint, {
                "event": "stopped_complete",
                "partition": partition,
                "update_calls": 0,
            }
        tasks = policy_state.task_ids
        active = tuple(tasks.index(task) for task in partition["active"])
        constraints = tuple(tasks.index(task) for task in partition["constraints"])
        key = active, constraints
        if key not in self.boundaries:
            self.boundaries[key] = CompiledBoundary(
                self.adapter,
                self.spec,
                active,
                constraints,
                checkpoint.method_state["rates"],
                preferred=checkpoint.method_state["preferred"],
                adam=self.adam,
                kernel_cache=self.kernel_cache,
                evaluation_factory=self.evaluation_factory,
            )
        boundary = self.boundaries[key]
        boundary.methods.rates = dict(checkpoint.method_state["rates"])
        boundary.methods.preferred = checkpoint.method_state["preferred"]
        boundary.methods.events = []
        tensor_state = (
            tf.constant(policy.values, tf.float64),
            tf.constant(checkpoint.optimizer_state["first_moment"], tf.float64),
            tf.constant(checkpoint.optimizer_state["second_moment"], tf.float64),
            tf.constant(checkpoint.optimizer_state["iteration"], tf.int64),
        )
        event = {
            "event": "update",
            "before": partition,
            "policy_before": policy.fingerprint(),
        }
        try:
            tensors, result = boundary.step(
                tensor_state, batch, checkpoint.update_index
            )
            if not bool(self.commit_guard(result["optimizer"][6]).numpy()):
                raise ValueError(
                    "source permanent-pass policy refuses a zero or tiny projected update"
                )
        except Exception as error:
            failed = replace(
                checkpoint, method_state=self._method_state(checkpoint, boundary)
            )
            event.update(
                {
                    "committed": False,
                    "after": partition,
                    "method_events": boundary.methods.events,
                    "error": f"{type(error).__name__}:{error}",
                }
            )
            raise PermanentPassUpdateError(str(error), failed, event) from error
        next_policy = replace(policy, values=tuple(tensors[0].numpy().tolist()))
        optimizer = {
            **checkpoint.optimizer_state,
            "first_moment": tensors[1].numpy().tolist(),
            "second_moment": tensors[2].numpy().tolist(),
            "iteration": int(tensors[3].numpy()),
            "learning_rate": boundary.methods.rates[result["method"]],
        }
        proposed = replace(
            checkpoint,
            update_index=checkpoint.update_index + 1,
            policy_state=next_policy.to_dict(),
            policy_fingerprint=next_policy.fingerprint(),
            optimizer_state=optimizer,
            method_state=self._method_state(checkpoint, boundary),
            rng_state={
                **checkpoint.rng_state,
                "next_seed_index": checkpoint.update_index + 1,
            },
        )
        committed = self.coordinator.commit(checkpoint, proposed)
        event.update(
            {
                "committed": True,
                "after": self.coordinator.read(committed).partition,
                "policy_after": next_policy.fingerprint(),
                "method": result["method"],
                "preferred": boundary.methods.preferred,
                "rates": dict(boundary.methods.rates),
                "method_events": list(boundary.methods.events),
                "submitted_constraint_dots": result["optimizer"][7].numpy().tolist(),
                "actual_constraint_dots": result["optimizer"][8].numpy().tolist(),
                "optimizer_iteration": optimizer["iteration"],
            }
        )
        return committed, event

    @staticmethod
    def _method_state(checkpoint, boundary):
        return {
            **checkpoint.method_state,
            "rates": dict(boundary.methods.rates),
            "preferred": boundary.methods.preferred,
            "last_events": list(boundary.methods.events),
            "event_count": checkpoint.method_state.get("event_count", 0)
            + len(boundary.methods.events),
        }
