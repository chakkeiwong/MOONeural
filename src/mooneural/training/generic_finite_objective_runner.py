"""Run a bound finite training objective through the common permanent-pass engine.

The caller rebuilds its functional objective from the declared recipe and data.
By default roles measure the same fixed finite objective. Optional source-bound
role factories supply separate scheduled evidence. Neither path admits science.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from .generic_coverage_runner import (
    ROOT,
    SOURCE_ROOT,
    _check_sources,
    _checked,
    _copy,
    _evidence,
    _reference,
    _write_json,
)
from .generic_training_contracts import (
    CertificationResult,
    CheckpointState,
    ControlEvaluation,
    ImmutableJSONMapping,
    NormalizationMetadata,
    PolicyView,
    ScaleSpec,
    TaskBatchTensors,
    TaskDefinition,
    TaskEvaluationTensors,
    TaskRegistry,
    ValidationEvaluation,
    canonical_json,
    stable_hash,
)

ARM_ID = 0
STAGE_ID = "generic-finite-objective-diagnostic-v1"
MEASURE = {
    "population": "same fixed finite training objective in every scope",
    "estimator": "exact declared finite objective; no independence or Student-t claim",
    "held_out": False,
    "independent_roles": False,
    "scientific_admission": False,
    "production_integrity_evidence": False,
    "scope_ids": "evaluation job identifiers, not independent data identifiers",
    "seeds": "scope ordinals required by the schema; no sampling is performed",
    "selection": "single fixed arm; required engine receipt does not compare scientific candidates",
    "threshold_meaning": "diagnostic normalized finite-objective convergence only",
    "sample_count_semantics": "one complete finite objective, not one IID observation",
}
TRAINING_MEASURE = {
    "population": "caller-bound deterministic scheduled minibatch from the finite training population",
    "role_evaluation": "full finite population exact objective",
    "normalization": "same full-population denominators; raw/D once",
    "sampling": "caller declares the frozen schedule; runner performs no sampling",
    "parity": "minibatch losses are not compared with full-population losses",
    "independence_claim": False,
    "scientific_admission": False,
    "production_integrity_evidence": False,
}
INJECTED_TRAINING_MEASURE = {
    "population": "caller-bound finite training population",
    "role_evaluation": "separately bound scheduled provider; not the finite training objective",
    "normalization": "raw/D once",
    "independence_claim": False,
    "scientific_admission": False,
    "production_integrity_evidence": False,
}
SOURCE_MODULES = (
    "generic_finite_objective_runner", "generic_coverage_runner", "generic_certified_numerics",
    "generic_replication_runner", "generic_replication_design", "generic_replication_policy",
    "generic_permanent_pass", "generic_permanent_pass_executor", "generic_model_task_executor",
    "generic_execution_boundary", "generic_training_contracts", "generic_role_banks",
    "generic_scheduled_role_banks", "generic_control_evidence", "generic_xla_kernels",
    "moo", "adam",
)


def _positive(name, value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _array(name, value, shape=None):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty flat real vector")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float64")
    return np.frombuffer(array.tobytes(), dtype=np.float64)


def _callback_binding(objective, data_binding, source_references, source_key="objective_source"):
    if not callable(objective):
        raise TypeError("finite objective must be callable")
    if not isinstance(data_binding, Mapping) or not data_binding:
        raise ValueError("explicit finite-objective data_binding required")
    recipe = data_binding.get("objective_recipe")
    if not isinstance(recipe, Mapping) or not recipe:
        raise ValueError("data_binding requires a nonempty objective_recipe")
    reference = data_binding.get(source_key)
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"data_binding requires an {source_key} artifact reference")
    source = _reference(_checked(reference))
    if source not in source_references:
        raise ValueError(f"{source_key} must be included in source_evidence")
    implementation = getattr(objective, "python_function", objective)
    if not inspect.isfunction(implementation) and not inspect.ismethod(implementation):
        implementation = implementation.__call__
    filename = inspect.getsourcefile(implementation)
    if filename is None or Path(filename).resolve() != Path(source["path"]):
        raise ValueError(f"objective callback source does not match {source_key}")
    return {"module": implementation.__module__, "qualname": implementation.__qualname__, "source": source,
            "recipe_hash": stable_hash(recipe), "state_mode": "caller-declared functional fixed data",
            "provenance_scope": "implementation source and declared recipe; no claim of complete callable ancestry"}


def _sources():
    paths = [Path(__file__).parent / f"{name}.py" for name in SOURCE_MODULES]
    paths.append(SOURCE_ROOT / "artifacts.py")
    return {str(path.relative_to(SOURCE_ROOT)): _reference(path)["sha256"] for path in paths}


class FiniteObjectiveAdapter:
    """Functional host adapter; the supplied callback owns all numerical work."""

    def __init__(self, task_ids, denominators, parameter_dim, objective, callback_binding,
                 input_binding, sources, references, threshold, objective_values=None, values_binding=None,
                 training_objective=None, training_binding=None, measure=None, role_coordinate_contract=None):
        from .generic_model_task_executor import ModelTaskMetadata

        self.task_ids, self.denominators, self.policy_dimension = task_ids, denominators, parameter_dim
        self.objective = objective
        self.objective_values = objective_values
        self.callback_binding = ImmutableJSONMapping(callback_binding)
        self.values_binding = None if values_binding is None else ImmutableJSONMapping(values_binding)
        self.training_objective = training_objective
        self.training_binding = None if training_binding is None else ImmutableJSONMapping(training_binding)
        self._last_measurements = {}
        self.sources = ImmutableJSONMapping(sources)
        self.references = tuple(ImmutableJSONMapping(reference) for reference in references)
        self.threshold = threshold
        self.role_coordinate_contract = None if role_coordinate_contract is None else ImmutableJSONMapping(role_coordinate_contract)
        self.policy_metadata = None
        gate_denominators = denominators.tolist() if role_coordinate_contract is None else role_coordinate_contract["coordinates"]["control"]
        normalizations = tuple(NormalizationMetadata(task,
            ScaleSpec(f"{task}.raw", "raw"), ScaleSpec(f"{task}.training", "training", 1. / denominator),
            ScaleSpec(f"{task}.selection", "selection", 1. / gate),
            ScaleSpec(f"{task}.terminal", "terminal", 1. / gate))
            for task, denominator, gate in zip(task_ids, denominators.tolist(), gate_denominators, strict=True))
        normalization_binding = {"denominators": denominators.tolist(), "threshold": threshold,
                                 "measure": MEASURE if measure is None else measure, "normalization": "raw/D once"}
        if self.training_binding is not None:
            normalization_binding["training_measure"] = self.training_binding["measure"]
        if role_coordinate_contract is not None:
            normalization_binding["role_coordinate_contract"] = _coordinate_semantics(role_coordinate_contract)
        self.metadata = ModelTaskMetadata(TaskRegistry(tuple(TaskDefinition(task, index) for index, task in enumerate(task_ids))),
            normalizations, (), (), normalization_binding=normalization_binding, input_binding=input_binding)

    def adapter_identity(self):
        payload = {"schema": "generic_neural_solver.finite_objective_adapter.v1", "state_mode": "functional",
                   "sources": self.sources, "callback": self.callback_binding, "values_callback": self.values_binding,
                   "input_fingerprint": self.metadata.input_fingerprint, "normalization_hash": self.metadata.normalization_hash}
        if self.training_binding is not None:
            payload["training"] = self.training_binding
        return {**payload, "identity_hash": stable_hash(payload)}

    def require_policy(self, policy):
        if not isinstance(policy, PolicyView) or len(policy.values) != self.policy_dimension:
            raise ValueError("finite-objective policy dimension mismatch")
        expected = self.metadata.input_fingerprint if self.policy_metadata is None else self.policy_metadata["finite_objective_input_fingerprint"]
        if policy.metadata.get("finite_objective_input_fingerprint") != expected:
            raise ValueError("finite-objective policy/data binding mismatch")

    def prepare_batch(self, policy, update_index):
        self.require_policy(policy)
        self.metadata.validate_bindings()
        if type(update_index) is not int or update_index < 0:
            raise ValueError("finite-objective update index must be nonnegative")
        metadata = {"normalization_hash": self.metadata.normalization_hash, "input_fingerprint": self.metadata.input_fingerprint,
                    "input_binding": self.metadata.input_binding, "policy_fingerprint": policy.fingerprint(),
                    "update_index": update_index, "sample_count_semantics": "one complete finite objective, not IID rows"}
        if self.training_binding is not None:
            schedule = self.training_binding["schedule"]
            transition = self.training_binding.get("estimator_transition")
            if transition is not None and update_index < transition["effective_update"]:
                raise ValueError("training update is before the declared estimator transition")
            if update_index >= len(schedule):
                raise ValueError("finite-objective training schedule exhausted")
            entry = schedule[update_index]
            metadata.update({"training_schedule_hash": self.training_binding["schedule_hash"],
                "training_schedule_entry": entry, "training_schedule_entry_hash": stable_hash(entry),
                "training_measure": self.training_binding["measure"],
                "sample_count_semantics": "one scheduled minibatch objective, not an independent replicate"})
        return TaskBatchTensors(self.task_ids, 1, self.policy_dimension, backend="tensorflow", metadata=metadata)

    def _require_inputs(self, policy):
        self.require_policy(policy)
        self.metadata.validate_bindings()
        _check_sources(self.sources)
        for reference in self.references:
            _checked(reference)

    def _validate_outputs(self, result, *, gradients):
        import tensorflow as tf

        if not isinstance(result, (tuple, list)) or len(result) != (3 if gradients else 2):
            raise ValueError("finite objective must return raw, normalized and, when requested, gradient tensors")
        task_count = len(self.task_ids)
        shapes = ((task_count,), (task_count,)) + (((task_count, self.policy_dimension),) if gradients else ())
        for value, shape in zip(result, shapes, strict=True):
            if not tf.is_tensor(value) or value.dtype != tf.float64 or tuple(value.shape) != shape:
                raise ValueError("finite-objective output shape/dtype mismatch")
            if not np.isfinite(value.numpy()).all():
                raise ValueError("finite-objective output contains nonfinite values")
        raw, normalized = result[:2]
        if np.any(raw.numpy() < 0) or np.any(normalized.numpy() < 0):
            raise ValueError("finite-objective losses must be nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            expected = raw.numpy() / self.denominators
        if not np.isfinite(expected).all() or not np.allclose(normalized.numpy(), expected, rtol=2e-13, atol=0.):
            raise ValueError("finite-objective normalization must equal raw/D exactly once")
        return result

    def _record_measurement(self, policy, result, kind):
        fingerprint = policy.fingerprint()
        losses = tuple(value.numpy() for value in result[:2])
        other = self._last_measurements.get("values" if kind == "full" else "full")
        if (other is not None and other[0] == fingerprint
                and any(not np.allclose(value, previous, rtol=2e-13, atol=0.)
                        for value, previous in zip(losses, other[1], strict=True))):
            raise ValueError("finite-objective loss-only/full callback parity mismatch")
        self._last_measurements[kind] = (fingerprint, losses)

    def evaluate(self, policy):
        import tensorflow as tf

        self._require_inputs(policy)
        result = self._validate_outputs(self.objective(tf.constant(policy.values, tf.float64)), gradients=True)
        self._record_measurement(policy, result, "full")
        return result

    def evaluate_values(self, policy):
        import tensorflow as tf

        if self.objective_values is None:
            return self.evaluate(policy)[:2]
        self._require_inputs(policy)
        result = self._validate_outputs(self.objective_values(tf.constant(policy.values, tf.float64)), gradients=False)
        self._record_measurement(policy, result, "values")
        return result

    def compute_task_values_and_gradients(self, policy, batch, update_index):
        import tensorflow as tf

        from .generic_model_task_executor import ModelTaskEvaluation

        if batch.to_dict() != self.prepare_batch(policy, update_index).to_dict():
            raise ValueError("finite-objective prepared policy/update binding mismatch")
        if self.training_binding is None:
            _raw, normalized, gradients = self.evaluate(policy)
        else:
            self._require_inputs(policy)
            _raw, normalized, gradients = self._validate_outputs(
                self.training_objective(tf.constant(policy.values, tf.float64), update_index), gradients=True)
        tensors = TaskEvaluationTensors(self.task_ids, tuple(normalized.numpy().tolist()),
            tuple(tuple(row) for row in gradients.numpy().tolist()), policy_dimension=self.policy_dimension,
            backend="tensorflow", normalization_hash=self.metadata.normalization_hash)
        return ModelTaskEvaluation(tensors, policy.fingerprint(), update_index, self.metadata.input_fingerprint,
                                   normalized, gradients, tf.zeros([0], tf.float64))


class FiniteObjectiveProvider:
    """Actual finite-objective measurements at every scheduled diagnostic call."""

    def __init__(self, adapter, roles, stage, scope_references):
        if adapter.role_coordinate_contract is not None:
            raise ValueError("separate training/gate coordinates require a scheduled role provider")
        self.adapter, self.roles, self.stage = adapter, roles, stage
        self.scope_references = tuple(scope_references)

    def _measure(self, role, arm, policy, stage, round_number, update_index):
        if type(arm) is not int or arm != ARM_ID or stage != self.stage:
            raise ValueError("finite-objective request outside frozen design")
        self.adapter.require_policy(policy)
        for reference in self.scope_references:
            _checked(reference)
        request = self.roles.request_for(role, policy, stage_id=stage.stage_id, arm_id=arm,
                                        round_number=round_number, update_index=update_index)
        raw, normalized = self.adapter.evaluate_values(policy)
        metrics = dict(zip(self.adapter.task_ids, normalized.numpy().tolist(), strict=True))
        records = {"schema": "generic_neural_solver.finite_training_diagnostic.v1", "measure": MEASURE,
                   "role": role, "update_index": update_index, "policy_fingerprint": policy.fingerprint(),
                   "input_fingerprint": self.adapter.metadata.input_fingerprint,
                   "raw_losses": dict(zip(self.adapter.task_ids, raw.numpy().tolist(), strict=True)),
                   "normalized_losses": metrics, "denominators": self.adapter.denominators.tolist(),
                   "threshold": self.adapter.threshold, "data_status": "fixed-finite-training-diagnostic",
                   "upper_mse_semantics": "exact declared finite objective, not a population confidence bound",
                   "role_bank_manifest_hash": request.metadata["role_bank_manifest_hash"],
                   "scheduled_role_registry_hash": self.roles.binding_hash(), "scientific_admission": False,
                   "production_integrity_evidence": False}
        return request, metrics, records

    def control(self, arm, policy, stage, round_number):
        update = stage.start_update + (round_number - stage.from_round) * stage.updates_per_round
        request, metrics, records = self._measure("control", arm, policy, stage, round_number, update)
        return ControlEvaluation(metrics, raw_records=records, request=request)

    def validation(self, arm, policy, stage, *, update_index):
        request, metrics, records = self._measure("validation", arm, policy, stage, None, update_index)
        return ValidationEvaluation(metrics, dict.fromkeys(self.adapter.task_ids, 1), records, request=request)

    def certification(self, arm, policy, stage, *, update_index):
        request, metrics, records = self._measure("certification", arm, policy, stage, None, update_index)
        return CertificationResult({task: value <= self.adapter.threshold for task, value in metrics.items()}, records,
                                   {"finite": True, "production_integrity_evidence": False}, records, request=request)


class ScheduledRoleProvider:
    """Existing runner provider interface over a source-bound role bridge."""

    def __init__(self, adapter, stage, bridge):
        self.adapter, self.stage, self.bridge = adapter, stage, bridge
        self.roles = bridge.registry

    def _evaluate(self, role, arm, policy, stage, round_number, update_index):
        if type(arm) is not int or arm != ARM_ID or stage != self.stage:
            raise ValueError("scheduled provider request outside frozen design")
        self.adapter._require_inputs(policy)
        request = self.roles.request_for(role, policy, stage_id=stage.stage_id, arm_id=arm,
                                        round_number=round_number, update_index=update_index)
        return self.bridge.evaluate(policy, request)

    def control(self, arm, policy, stage, round_number):
        update = stage.start_update + (round_number - stage.from_round) * stage.updates_per_round
        return self._evaluate("control", arm, policy, stage, round_number, update)

    def validation(self, arm, policy, stage, *, update_index):
        return self._evaluate("validation", arm, policy, stage, None, update_index)

    def certification(self, arm, policy, stage, *, update_index):
        return self._evaluate("certification", arm, policy, stage, None, update_index)


def _role_declaration(factory, binding, task_ids, denominators, threshold):
    from .generic_independent_role_bridge import GroupedRoleSpec, LossCoordinates

    declaration, references = _evidence(binding)
    required = {"factory_source", "inputs", "coordinates", "measure", "training_row_ids"}
    if not required <= set(declaration):
        raise ValueError("role declaration is incomplete")
    _evidence(declaration["inputs"])
    coordinates = LossCoordinates(**declaration["coordinates"])
    mode = declaration.get("coordinate_mode")
    if mode is None:
        if "raw_role_limits" in declaration or "grouping" in declaration or coordinates.task_ids != task_ids or coordinates.threshold != threshold or any(
                tuple(getattr(coordinates, role)) != tuple(denominators)
                for role in ("training", "control", "validation", "certification")):
            raise ValueError("injected builder requires matching task order, threshold and equal role denominators")
    else:
        if (mode != "separate-training-gates-v1" or coordinates.task_ids != task_ids or threshold != 1.
                or coordinates.threshold != 1. or coordinates.training != tuple(denominators)
                or any(getattr(coordinates, role) != coordinates.control for role in ("validation", "certification"))):
            raise ValueError("opt-in coordinates require training D, common gate T and threshold one")
        limits = declaration.get("raw_role_limits")
        if not isinstance(limits, Mapping) or set(limits) != {"control", "validation", "certification"} or any(
                not np.array_equal(_array("raw role limits", limits[role], (len(task_ids),)), coordinates.control)
                for role in limits):
            raise ValueError("raw role limits must equal the declared gate denominators")
        if "grouping" in declaration and GroupedRoleSpec(**declaration["grouping"]).task_ids != task_ids:
            raise ValueError("grouped role task order mismatch")
    measure = declaration["measure"]
    if not isinstance(measure, Mapping) or any(measure.get(key) is not False
            for key in ("scientific_admission", "production_integrity_evidence")):
        raise ValueError("injected role measure must declare engineering evidence")
    rows = declaration["training_row_ids"]
    if not isinstance(rows, list) or not rows or any(not isinstance(row, str) or not row for row in rows) or len(set(rows)) != len(rows):
        raise ValueError("unique training row identities required")
    identity = _callback_binding(factory, {"objective_recipe": declaration, "objective_source": declaration["factory_source"]}, references)
    return ImmutableJSONMapping(declaration), references, identity, coordinates


def _coordinate_contract(declaration):
    if declaration is None or declaration.get("coordinate_mode") is None:
        return None
    return {name: declaration[name] for name in ("coordinate_mode", "coordinates", "raw_role_limits", "grouping")
            if name in declaration}


def _coordinate_semantics(contract):
    if contract is None:
        return None
    stable = dict(contract)
    if "grouping" in stable:
        stable["grouping"] = {key: item for key, item in stable["grouping"].items() if key != "bank_families"}
    return stable


def _check_continuation_coordinates(parent_training, contract):
    if canonical_json(_coordinate_semantics(parent_training.get("role_coordinate_contract"))) != canonical_json(_coordinate_semantics(contract)):
        raise ValueError("continuation role coordinates or grouping changed")


def _materialize_roles(factory, declaration, references, coordinates, adapter, stage, output, endpoint):
    from .generic_scheduled_role_banks import ScheduledRoleBankRegistry

    before = stable_hash(adapter.adapter_identity())
    for reference in references:
        _checked(reference)
    roles, provider, binding = factory(adapter, stage, output)
    adapter.metadata.validate_bindings()
    if stable_hash(adapter.adapter_identity()) != before:
        raise ValueError("role factory changed frozen training adapter")
    for reference in references:
        _checked(reference)
    if not isinstance(roles, ScheduledRoleBankRegistry) or roles.task_ids != adapter.task_ids:
        raise ValueError("role factory must return matching scheduled task registry")
    if not isinstance(getattr(provider, "roles", None), ScheduledRoleBankRegistry) or provider.roles.binding_hash() != roles.binding_hash():
        raise ValueError("role provider registry differs from materialization")
    if isinstance(provider, ScheduledRoleProvider) and provider.bridge.coordinates.binding_hash() != coordinates.binding_hash():
        raise ValueError("role bridge coordinates differ from declaration")
    if declaration.get("coordinate_mode") is not None:
        if not isinstance(provider, ScheduledRoleProvider):
            raise ValueError("separate training/gate coordinates require bound scheduled role provider")
        grouping = None if provider.bridge.grouping is None else provider.bridge.grouping.to_dict()
        if canonical_json(grouping) != canonical_json(declaration.get("grouping")):
            raise ValueError("role bridge grouping differs from declaration")
    bound, generated_references = _evidence(binding)
    if bound.get("adapter_input_fingerprint") != adapter.metadata.input_fingerprint or bound.get("registry_hash") != roles.binding_hash():
        raise ValueError("role materialization differs from training/registry identity")
    if "provider_source" not in bound:
        raise ValueError("role provider source required")
    methods = ("control", "validation", "certification") if endpoint == "certification" else ("control", "validation")
    for method in methods:
        _callback_binding(getattr(provider, method, None), {"objective_source": bound["provider_source"], "objective_recipe": bound}, generated_references)
    expected = {("control", stage.from_round + number, stage.start_update + number * stage.updates_per_round,
                 stage.start_update + number * stage.updates_per_round) for number in range(stage.round_count + 1)}
    expected.add(("validation", None, stage.start_update, stage.stop_update))
    if endpoint == "certification":
        expected.add(("certification", None, stage.start_update, stage.stop_update))
    scopes = {(entry.role, entry.round_number, entry.min_update, entry.max_update) for entry in roles.bindings}
    if scopes != expected or len(scopes) != len(roles.bindings) or any(
            entry.stage_id != stage.stage_id or entry.arm_id != ARM_ID for entry in roles.bindings):
        raise ValueError("injected role scopes differ from complete lifecycle schedule")
    training_rows = set(declaration["training_row_ids"])
    _inputs, input_references = _evidence(declaration["inputs"])
    input_hashes = {reference["sha256"] for reference in input_references}
    for entry in roles.bindings:
        for bank in entry.manifest.banks:
            if not set(bank.input_hashes.values()) <= input_hashes:
                raise ValueError("role bank inputs were not declared before factory materialization")
            if bank.scale_version != coordinates.binding_hash():
                raise ValueError("injected bank coordinate binding differs")
            rows = bank.metadata.get("row_ids")
            if not isinstance(rows, (tuple, list)) or not rows or training_rows.intersection(rows):
                raise ValueError("evaluation rows missing or overlap training rows")
    _registry, registry_references = _evidence({"registry": roles.to_dict()})
    return roles, provider, bound, [*generated_references, *registry_references]


def _schedule(output, stage, task_ids, binding_reference, input_fingerprint):
    from .generic_role_banks import RoleBank, RoleBankManifest
    from .generic_scheduled_role_banks import (
        ScheduledRoleBankRegistry,
        ScheduledRoleBinding,
    )

    scopes = [("control", stage.from_round + number, stage.start_update + number * stage.updates_per_round,
               stage.start_update + number * stage.updates_per_round)
              for number in range(stage.round_count + 1)]
    scopes += [("validation", None, stage.start_update, stage.stop_update),
               ("certification", None, stage.start_update, stage.stop_update)]
    bindings, references = [], []
    for ordinal, (role, number, minimum, maximum) in enumerate(scopes):
        scope_id = f"{role}-{number if number is not None else 'exit'}"
        reference = _write_json(output / "scopes" / f"{scope_id}.json", {
            "schema": "generic_neural_solver.finite_objective_job.v1", "scope": scope_id,
            "data_binding": binding_reference, "input_fingerprint": input_fingerprint, "measure": MEASURE})
        references.append(reference)
        bank = RoleBank(role, scope_id, ordinal, "bound-finite-training-objective", input_fingerprint,
            "caller-D-v1", "exact-finite-objective-diagnostic-v1", {"global": reference["sha256"], "local": binding_reference["sha256"]},
            (f"evaluation-job/{scope_id}",), metadata={"global_archive": reference, "data_binding": binding_reference,
                "measure": MEASURE, "global_hash_semantics": "evaluation job descriptor, not independent input samples"})
        bindings.append(ScheduledRoleBinding(stage.stage_id, ARM_ID, role, number, minimum, maximum,
                                            RoleBankManifest(task_ids, (bank,), {role: 1})))
    return ScheduledRoleBankRegistry(tuple(bindings), registry_version="finite-training-objective-diagnostic-v1"), references


def _partial_parent_state(declared, declaration):
    from .generic_permanent_pass import PermanentPassState
    from .generic_replication_design import ReplicationDesignBinding
    from .generic_replication_runner import AtomicCheckpointStore

    config_path = _checked(declared["parent_configuration"])
    config = json.loads(config_path.read_text())
    checkpoint_path = _checked(declared["parent_checkpoint"])
    payload = json.loads(checkpoint_path.read_text())
    design = ReplicationDesignBinding.from_dict(payload["state"]["metadata"]["replication_design"])
    store = AtomicCheckpointStore(config_path.parent / "checkpoints", payload["attempt_id"], design)
    marker_path = _checked(declared["parent_commit_marker"])
    if marker_path != checkpoint_path.with_suffix(".complete.json"):
        raise ValueError("partial parent marker does not name the bound checkpoint")
    stored = store._read(marker_path, arm_id=ARM_ID)
    latest, _events = store.read_latest_committed(ARM_ID)
    parent = stored.state
    parent_hash = stable_hash(parent.to_dict())
    if (latest.to_dict() != parent.to_dict() or parent.update_index != declaration["effective_update"]
            or declaration.get("parent_checkpoint_hash", parent_hash) != parent_hash):
        raise ValueError("partial parent is stale or differs from the declared clock/state")
    stage = config["stage"]
    stop = stage["start_update"] + (stage["last_round"] - stage["from_round"]) * stage["updates_per_round"]
    if not stage["start_update"] <= parent.update_index < stop:
        raise ValueError("partial parent must precede its original declared endpoint")
    if (design.replica_ids != (ARM_ID,) or design.stages[-1] != stage
            or design.target_hashes["configuration"] != declared["parent_configuration"]["sha256"]):
        raise ValueError("partial parent configuration/design mismatch")
    refusal_path = store.root / "rejections" / f"arm-{ARM_ID}" / f"{parent_hash}.json"
    if _checked(declared["parent_refusal"]) != refusal_path:
        raise ValueError("partial parent refusal does not belong to the bound store/state")
    refusal = store.read_update_rejection(ARM_ID, parent)
    event = refusal["event"]
    receipt = event.get("postproposal", {})
    rotation = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"])
    profile = config.get("postproposal")
    if (rotation.complete or profile is None or event.get("transaction_rolled_back") is not True
            or event.get("committed") is not False or receipt.get("accepted") is not False
            or receipt.get("failure_kind") == "contract_error"
            or receipt.get("update_index") != parent.update_index
            or receipt.get("profile_hash") != stable_hash(profile["profile"])
            or event.get("policy_before") != parent.policy_fingerprint
            or event.get("before") != rotation.partition or event.get("after") != rotation.partition):
        raise ValueError("partial parent requires a bound rolled-back finite postproposal rejection")
    result = json.loads(_checked(declared["parent_result"]).read_text())
    if result.get("training_completed") is True or any(
            key in result and result[key] != parent.update_index
            for key in ("final_committed_update", "final_update", "update_index")):
        raise ValueError("partial parent terminal result has inconsistent completion/clock")
    if "final_state" in result and result["final_state"] != parent.to_dict():
        raise ValueError("partial parent terminal result has a different state")
    for key in ("parent_state_hash", "parent_checkpoint_hash"):
        if key in result and result[key] != parent_hash:
            raise ValueError("partial parent terminal result has a different state hash")
    return parent


def _continuation_parent(checkpoint, binding, partial=None):
    from .generic_control_evidence import ControlEvidenceArchive, full_control_hash
    from .generic_permanent_pass import PermanentPassState
    from .generic_replication_design import ReplicationDesignBinding

    if not isinstance(checkpoint, CheckpointState):
        raise TypeError("continuation_checkpoint must be a typed CheckpointState")
    declared, references = _evidence(binding)
    expected = {"parent_result", "parent_checkpoint", "parent_configuration", "parent_control_archive"}
    if partial is not None:
        expected |= {"parent_commit_marker", "parent_refusal"}
    if set(declared) != expected:
        raise ValueError("continuation requires parent result/checkpoint/configuration/control references")
    parent = CheckpointState.from_dict(checkpoint.to_dict())
    if partial is None:
        result = json.loads(_checked(declared["parent_result"]).read_text())
        saved = json.loads(_checked(declared["parent_checkpoint"]).read_text())
        saved = saved.get("final_state", saved)
        if (result.get("training_completed") is not True or result.get("final_state") != parent.to_dict()
                or saved != parent.to_dict()):
            raise ValueError("continuation checkpoint differs from completed parent result")
    elif _partial_parent_state(declared, partial).to_dict() != parent.to_dict():
        raise ValueError("partial continuation checkpoint differs from its committed marker")
    config_path = _checked(declared["parent_configuration"])
    config = json.loads(config_path.read_text())
    design = ReplicationDesignBinding.from_dict(parent.metadata["replication_design"])
    design.validate_checkpoint(parent)
    if (design.replica_ids != (ARM_ID,) or design.target_hashes["configuration"] != declared["parent_configuration"]["sha256"]
            or design.stages[-1] != config["stage"]):
        raise ValueError("continuation parent configuration/design mismatch")
    stage = config["stage"]
    if partial is None and parent.update_index != stage["start_update"] + (stage["last_round"] - stage["from_round"]) * stage["updates_per_round"]:
        raise ValueError("continuation parent must complete its declared stage")
    training_ref = config.get("training_configuration", declared["parent_configuration"])
    training_config = json.loads(_checked(training_ref).read_text())
    data_ref = _reference(config_path.parent / "data-binding.json")
    if data_ref["sha256"] != design.target_hashes["data_binding"]:
        raise ValueError("continuation parent data snapshot differs from design")
    snapshots = [*config_path.parent.glob("evidence/*.snapshot"), *config_path.parent.glob("role-evidence/*.snapshot")]
    snapshot_hashes = {_reference(path)["sha256"] for path in snapshots}
    for relative, digest in config["sources"].items():
        # External callback identities may be absolute. Their retained evidence
        # snapshot is required; joining an absolute path would instead inspect
        # the live original and could incorrectly accept a missing snapshot.
        source = None if Path(relative).is_absolute() else config_path.parent / "source" / relative
        if not ((source is not None and source.is_file() and _reference(source)["sha256"] == digest)
                or digest in snapshot_hashes):
            raise ValueError("continuation parent source snapshot missing or changed")
    rotation = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"])
    archive_refs = declared["parent_control_archive"]
    if not isinstance(archive_refs, list) or len(archive_refs) != len(rotation.controls):
        raise ValueError("continuation requires one full control reference per history point")
    controls = []
    for point, reference in zip(rotation.controls, archive_refs, strict=True):
        path = _checked(reference)
        if reference["sha256"] != full_control_hash(point.evaluation) or path.stem != reference["sha256"]:
            raise ValueError("continuation historical control reference mismatch")
        controls.append(ControlEvidenceArchive(path.parent).rehydrate(point.evaluation))
    return parent, declared, references, config, training_config, training_ref, data_ref, controls


def _rate_transition(parent, parent_config, learning_rate, declaration):
    design = parent.metadata["replication_design"]
    optimizer_binding = design["optimizer_binding"]
    previous = parent_config.get("checkpoint_continuation", {}).get("learning_rate_transition")
    if (canonical_json(previous) != canonical_json(optimizer_binding.get("learning_rate_transition"))
            or canonical_json(previous) != canonical_json(
                parent.metadata.get("checkpoint_continuation", {}).get("learning_rate_transition"))):
        raise ValueError("continuation learning-rate lineage differs from parent design/state")
    if previous is not None and (previous["new_optimizer"]["base_learning_rate"] != parent_config["learning_rate"]
                                 or previous["effective_update"] > parent.update_index):
        raise ValueError("continuation learning-rate lineage has inconsistent rate/clock")
    if declaration is None:
        return previous
    if (not isinstance(declaration, Mapping) or set(declaration) != {"effective_update", "reason"}
            or type(declaration["effective_update"]) is not int
            or declaration["effective_update"] != parent.update_index
            or not isinstance(declaration["reason"], str) or not declaration["reason"].strip()):
        raise ValueError("invalid learning-rate transition at the parent clock")
    old_base = _positive("parent configured learning rate", parent_config["learning_rate"])
    if optimizer_binding["rate"] != old_base:
        raise ValueError("parent configured learning rate differs from optimizer binding")
    if learning_rate == old_base:
        raise ValueError("learning-rate transition must change the configured rate")
    ratio = _positive("learning-rate ratio", learning_rate / old_base)
    old = {"adam": optimizer_binding["adam"], "base_learning_rate": old_base,
           "saved_learning_rate": parent.optimizer_state["learning_rate"],
           "method_rates": dict(parent.method_state["rates"])}
    new = {**old, "base_learning_rate": learning_rate,
           "saved_learning_rate": old["saved_learning_rate"] * ratio,
           "method_rates": {method: rate * ratio for method, rate in old["method_rates"].items()}}
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        old_effective = np.asarray([old_base, old["saved_learning_rate"], *old["method_rates"].values()], dtype=np.float32)
        new_effective = np.asarray([learning_rate, new["saved_learning_rate"], *new["method_rates"].values()], dtype=np.float32)
    if (not np.isfinite(old_effective).all() or not np.isfinite(new_effective).all()
            or np.any(old_effective <= 0) or np.any(new_effective <= 0)):
        raise ValueError("learning rates must remain positive finite in the source float32 convention")
    if np.array_equal(old_effective[2:], new_effective[2:]):
        raise ValueError("learning-rate transition has no effective method-rate change")
    return json.loads(canonical_json({
        "schema": "generic_neural_solver.learning_rate_transition.v1", **declaration,
        "parent_checkpoint_hash": stable_hash(parent.to_dict()),
        "parent_design_hash": parent.metadata["replication_design_sha256"], "ratio": ratio,
        "old_optimizer": old, "new_optimizer": new,
        "numerical_binding": {name: design["backend_binding"][name] for name in ("method", "update")},
        "previous_transition": previous,
    }))


def _method_policy_transition(parent, parent_config, declaration, methods):
    design = parent.metadata["replication_design"]
    optimizer_binding = design["optimizer_binding"]
    previous = parent_config.get("checkpoint_continuation", {}).get("method_policy_transition")
    if (canonical_json(previous) != canonical_json(optimizer_binding.get("method_policy_transition"))
            or canonical_json(previous) != canonical_json(
                parent.metadata.get("checkpoint_continuation", {}).get("method_policy_transition"))):
        raise ValueError("continuation method-policy lineage differs from parent design/state")
    preferred = parent.method_state["preferred"]
    if preferred != optimizer_binding["preferred_method"]:
        raise ValueError("continuation parent has an undeclared preferred-method change")
    if previous is not None and (previous["new_preferred"] != preferred
                                 or previous["effective_update"] > parent.update_index
                                 or previous["methods"] != list(methods)):
        raise ValueError("continuation method-policy lineage has inconsistent preference/clock/methods")
    if declaration is None:
        return previous
    if (not isinstance(declaration, Mapping)
            or set(declaration) != {"effective_update", "old_preferred", "new_preferred", "reason"}
            or type(declaration["effective_update"]) is not int
            or declaration["effective_update"] != parent.update_index
            or declaration["old_preferred"] != preferred
            or not isinstance(declaration["new_preferred"], str)
            or declaration["new_preferred"] not in methods
            or declaration["new_preferred"] == preferred
            or not isinstance(declaration["reason"], str) or not declaration["reason"].strip()):
        raise ValueError("invalid method-policy transition at the parent clock/preference")
    return json.loads(canonical_json({
        "schema": "generic_neural_solver.method_policy_transition.v1", **declaration,
        "parent_checkpoint_hash": stable_hash(parent.to_dict()),
        "parent_design_hash": parent.metadata["replication_design_sha256"],
        "methods": list(methods), "method_rates": dict(parent.method_state["rates"]),
        "saved_learning_rate": parent.optimizer_state["learning_rate"],
        "numerical_binding": {name: design["backend_binding"][name] for name in ("method", "update")},
        "previous_transition": previous,
    }))


def _component_refinement(declaration):
    return isinstance(declaration, Mapping) and declaration.get("kind") in (
        "component_refinement", "component_trigger_refinement", "protected_descent_refinement",
        "relative_progress_refinement", "coupled_component_refinement", "target_refinement")


def _selection_refinement(declaration):
    return isinstance(declaration, Mapping) and declaration.get("kind") == "finite_selection_refinement"


def _check_selection_profiles(old_binding, binding):
    if not isinstance(old_binding, Mapping) or not isinstance(binding, Mapping):
        raise TypeError("selection refinement requires two bound guards")
    old_profile, new_profile = json.loads(canonical_json(old_binding)), json.loads(canonical_json(binding))
    selection = new_profile["profile"].pop("finite_selection", None)
    component = old_profile["profile"].get("component_fallback") or {}
    if ("finite_selection" in old_profile["profile"] or not isinstance(selection, Mapping)
            or type(selection.get("lookahead_steps")) is not int or selection["lookahead_steps"] not in (1, 2)
            or selection != {"profile": "bounded-feasible-suffix-v1", "lookahead_steps": selection["lookahead_steps"],
                "criterion": "minimum-active-relative-decrease", "minimum_gain": 1e-8,
                "ties": "earlier-candidate", "fractions": "existing-list-only"}
            or component.get("component_direction") != "relative_public_progress"
            or "finite_target_refinement" in component
            or component.get("trigger") == "invalid-original-linear-segment-or-first-finite-rejection"
            or canonical_json(old_profile) != canonical_json(new_profile)):
        raise ValueError("selection refinement must change only the supported finite selector")


def _check_selection_record(record):
    old_binding, binding = record.get("old_binding"), record.get("new_binding")
    _check_selection_profiles(old_binding, binding)
    if (record.get("old_postproposal_hash") != stable_hash(old_binding)
            or record.get("new_postproposal_hash") != stable_hash(binding)):
        raise ValueError("selection lineage hashes differ from its bound profiles")


def _check_component_refinement(old_component, old_callback, component, callback, *, trigger_refinement=False,
                                protected_descent_refinement=False, relative_progress_refinement=False,
                                coupled_component_refinement=False, target_refinement=False):
    if target_refinement:
        from .generic_postproposal import (
            coupled_component_binding,
            finite_target_binding,
        )

        if any(not isinstance(value, Mapping) or not value for value in (
                old_component, old_callback, component, callback)):
            raise ValueError("target refinement requires an enabled coupled parent")
        old_count = old_component.get("maximum_component_loss_calls")
        if (any(old_component.get(key) != value for key, value in coupled_component_binding().items())
                or old_component.get("component_direction") != "relative_public_progress"
                or any(key in old_component for key in finite_target_binding())
                or type(old_count) is not int or old_count < 1
                or not isinstance(old_callback.get("loss_only"), Mapping)):
            raise ValueError("target refinement requires the original coupled profile")
        expected = {**old_component, **finite_target_binding(), "maximum_component_loss_calls": 2 * old_count}
        if canonical_json(expected) != canonical_json(component) or canonical_json(old_callback) != canonical_json(callback):
            raise ValueError("target refinement may only add its weighted direction policy and derived caps")
        return
    if coupled_component_refinement:
        from .generic_postproposal import (
            coupled_component_binding,
            relative_progress_binding,
        )

        if any(not isinstance(value, Mapping) or not value for value in (
                old_component, old_callback, component, callback)):
            raise ValueError("coupled refinement requires an enabled relative-progress parent")
        if (any(old_component.get(key) != value for key, value in relative_progress_binding().items())
                or old_component.get("trigger") != "every-update-relative-public-progress"
                or "coupled_finite_guard" in old_component or "loss_only" in old_callback):
            raise ValueError("coupled refinement requires the original relative-progress profile")
        extras = {"component_loss_callback", "component_loss_provider", "maximum_component_loss_calls"}
        if (any(not isinstance(component.get(key), Mapping) or not component[key] for key in extras - {"maximum_component_loss_calls"})
                or type(component.get("maximum_component_loss_calls")) is not int
                or component["maximum_component_loss_calls"] < 1
                or not isinstance(callback.get("loss_only"), Mapping)):
            raise ValueError("coupled refinement requires a bound loss callback and finite call cap")
        expected = {**old_component, **coupled_component_binding(), **{key: component[key] for key in extras}}
        previous_callback, next_callback = (json.loads(canonical_json(value)) for value in (old_callback, callback))
        loss_callback = next_callback.pop("loss_only")
        if ({key: loss_callback.get(key) for key in ("module", "qualname")}
                != component["component_loss_callback"]):
            raise ValueError("coupled refinement loss callback identities differ")
        for value in (previous_callback, next_callback, loss_callback):
            source = value.get("source")
            if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
                raise ValueError("coupled refinement requires verified callback source provenance")
            source.pop("path")
        if canonical_json(expected) != canonical_json(component) or previous_callback != next_callback:
            raise ValueError("coupled refinement may only add required-cell targets and finite checks")
        return
    if relative_progress_refinement:
        from .generic_component_descent import component_descent_binding
        from .generic_postproposal import relative_progress_binding

        if any(not isinstance(value, Mapping) or not value for value in (
                old_component, old_callback, component, callback)):
            raise ValueError("relative progress refinement requires an enabled parent provider")
        old_solver = component_descent_binding(old_component.get("relative_tolerance", 1e-12))
        if (any(old_component.get(key) != value for key, value in old_solver.items())
                or old_component.get("component_direction") is not None
                or old_component.get("trigger") not in (
                    "invalid-original-linear-segment-only", "invalid-original-linear-segment-or-first-finite-rejection",
                    "every-update-with-extra-descent-preference")):
            raise ValueError("relative progress refinement requires the original equality direction")
        expected = {key: value for key, value in old_component.items() if key not in old_solver}
        expected.pop("finite_trigger_radius", None)
        expected.update(relative_progress_binding(), trigger="every-update-relative-public-progress",
            finite_screen="original-fractions-active-and-extra-public-decrease",
            direction_radius="source-final-displacement-norm")
        previous_callback, next_callback = (json.loads(canonical_json(value)) for value in (old_callback, callback))
        for value in (previous_callback, next_callback):
            source = value.get("source")
            if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
                raise ValueError("relative progress refinement requires verified provider source provenance")
            source.pop("path")
        if canonical_json(component) != canonical_json(expected) or previous_callback != next_callback:
            raise ValueError("relative progress refinement may only change the direction and source location")
        return
    if protected_descent_refinement:
        if any(not isinstance(value, Mapping) or not value for value in (
                old_component, old_callback, component, callback)):
            raise ValueError("protected descent refinement requires an enabled parent provider")
        excluded = {"provider", "callback", "trigger", "finite_screen", "finite_trigger_radius",
            "request_context", "extra_descent_indices", "extra_descent_radius"}
        extra = component.get("extra_descent_indices")
        if (old_component.get("extra_descent_indices") is not None
                or old_component.get("trigger") not in (
                    "invalid-original-linear-segment-only", "invalid-original-linear-segment-or-first-finite-rejection")
                or not isinstance(extra, (list, tuple)) or not extra
                or any(type(index) is not int or index < 0 for index in extra)
                or sorted(set(extra)) != list(extra)
                or component.get("trigger") != "every-update-with-extra-descent-preference"
                or component.get("request_context") != "parent-batch-update-D-active-and-extra-descent-indices-v1"
                or component.get("extra_descent_radius") != "source-final-displacement-norm"
                or component.get("finite_screen") != "original-fractions-active-and-extra-public-decrease"
                or "finite_trigger_radius" in component
                or any(not isinstance(component.get(key), Mapping) or not component[key]
                       for key in ("provider", "callback"))
                or canonical_json({key: value for key, value in old_component.items() if key not in excluded})
                    != canonical_json({key: value for key, value in component.items() if key not in excluded})):
            raise ValueError("protected descent refinement may only enable its declared preference and provider")
        return
    if trigger_refinement:
        if any(not isinstance(value, Mapping) or not value for value in (
                old_component, old_callback, component, callback)):
            raise ValueError("component trigger refinement requires an existing provider")
        excluded = {"trigger", "finite_screen", "finite_trigger_radius"}
        if (old_component.get("trigger") != "invalid-original-linear-segment-only"
                or old_component.get("finite_screen") != "original-fractions-on-witness-no-extra-loss-budget"
                or "finite_trigger_radius" in old_component
                or component.get("trigger") != "invalid-original-linear-segment-or-first-finite-rejection"
                or component.get("finite_screen") != "component-fractions-before-original-shorter-fractions"
                or component.get("finite_trigger_radius") != "original-valid-segment-norm"
                or canonical_json(old_callback) != canonical_json(callback)
                or canonical_json({key: value for key, value in old_component.items() if key not in excluded})
                    != canonical_json({key: value for key, value in component.items() if key not in excluded})):
            raise ValueError("component trigger refinement may only enable finite fallback on the unchanged provider")
        return
    if (any(not isinstance(value, Mapping) or not value for value in (
            old_component, old_callback, component, callback))
            or any(not isinstance(value.get(key), Mapping) or not value[key]
                for value in (old_component, component) for key in ("provider", "callback"))
            or canonical_json((old_component, old_callback)) == canonical_json((component, callback))
            or canonical_json({key: value for key, value in old_component.items() if key not in ("provider", "callback")})
                != canonical_json({key: value for key, value in component.items() if key not in ("provider", "callback")})):
        raise ValueError("postproposal component refinement requires only an explicit enabled-provider replacement")


def _postproposal_lineage(parent, parent_config):
    optimizer_binding = parent.metadata["replication_design"]["optimizer_binding"]
    old_binding = parent_config.get("postproposal")
    names = ("postproposal_schedule_transition", "postproposal_policy_transition",
             "postproposal_search_profile_transition")
    previous = [parent_config.get("checkpoint_continuation", {}).get(name) for name in names]
    if (canonical_json(old_binding) != canonical_json(optimizer_binding.get("postproposal"))
            or any(canonical_json(value) != canonical_json(optimizer_binding.get(name))
                or canonical_json(value) != canonical_json(parent.metadata.get("checkpoint_continuation", {}).get(name))
                for name, value in zip(names, previous, strict=True))):
        raise ValueError("continuation postproposal lineage differs from parent design/state")
    schedule_previous, policy_previous, _search_previous = previous
    if not any(value is not None for value in previous):
        return schedule_previous, policy_previous
    failure = "continuation postproposal lineage has inconsistent profile/schedule/clock"
    if old_binding is None:
        raise ValueError(failure)
    guard_hash = stable_hash(old_binding)
    training = parent_config.get("training")
    schedule_hash = stable_hash(None if training is None else training["schedule"])
    clock = parent.update_index
    heads = list(previous)
    component, callback = old_binding["profile"].get("component_fallback"), old_binding.get("component_callback")
    selection = old_binding["profile"].get("finite_selection")
    while any(head is not None for head in heads):
        matches = [index for index, head in enumerate(heads)
                   if isinstance(head, Mapping) and head.get("new_postproposal_hash") == guard_hash]
        if len(matches) != 1:
            raise ValueError(failure)
        index = matches[0]
        head = heads[index]
        if (head.get("schema") != f"generic_neural_solver.{names[index]}.v1"
                or type(head.get("effective_update")) is not int or not 0 <= head["effective_update"] <= clock
                or head.get("new_schedule_hash") != schedule_hash
                or not isinstance(head.get("old_postproposal_hash"), str)
                or head["old_postproposal_hash"] == guard_hash
                or not isinstance(head.get("old_schedule_hash"), str)
                or index == 1 and (head["old_schedule_hash"] != schedule_hash
                    or head.get("previous_schedule_transition_hash") != stable_hash(heads[0]))
                or index == 0 and head.get("previous_policy_transition_hash") != (
                    None if heads[1] is None else stable_hash(heads[1]))
                or index != 2 and head.get("previous_search_profile_transition_hash") != (
                    None if heads[2] is None else stable_hash(heads[2]))):
            raise ValueError(failure)
        if index == 2:
            _check_search_profile_record(head)
            if (head["old_schedule_hash"] != schedule_hash
                    or head.get("previous_schedule_transition_hash") != stable_hash(heads[0])
                    or head.get("previous_policy_transition_hash") != stable_hash(heads[1])
                    or canonical_json(head["new_binding"]["profile"].get("component_fallback")) != canonical_json(component)
                    or canonical_json(head["new_binding"].get("component_callback")) != canonical_json(callback)
                    or canonical_json(head["new_binding"]["profile"].get("finite_selection")) != canonical_json(selection)):
                raise ValueError(failure)
            component = head["old_binding"]["profile"].get("component_fallback")
            callback = head["old_binding"].get("component_callback")
        if index == 1:
            if (canonical_json(head.get("component_fallback")) != canonical_json(component)
                    or canonical_json(head.get("component_callback")) != canonical_json(callback)):
                raise ValueError(failure)
            if _selection_refinement(head):
                _check_selection_record(head)
                if (canonical_json(head["new_binding"]["profile"].get("finite_selection")) != canonical_json(selection)
                        or head["old_schedule_hash"] != head["new_schedule_hash"]):
                    raise ValueError(failure)
                selection = head["old_binding"]["profile"].get("finite_selection")
            elif _component_refinement(head):
                old_component, old_callback = head.get("old_component_fallback"), head.get("old_component_callback")
                try:
                    _check_component_refinement(old_component, old_callback, component, callback,
                        trigger_refinement=head.get("kind") == "component_trigger_refinement",
                        protected_descent_refinement=head.get("kind") == "protected_descent_refinement",
                        relative_progress_refinement=head.get("kind") == "relative_progress_refinement",
                        coupled_component_refinement=head.get("kind") == "coupled_component_refinement",
                        target_refinement=head.get("kind") == "target_refinement")
                    if head.get("kind") == "coupled_component_refinement" and (
                            head.get("old_maximum_loss_calls") != head.get("new_maximum_loss_calls")
                            or type(head.get("new_maximum_loss_calls")) is not int
                            or component["maximum_component_loss_calls"] != head["new_maximum_loss_calls"] - 1):
                        raise ValueError("coupled refinement loss cap differs")
                    if head.get("kind") in ("component_trigger_refinement", "target_refinement") and (
                            type(head.get("old_maximum_loss_calls")) is not int
                            or head["old_maximum_loss_calls"] < 2
                            or head.get("new_maximum_loss_calls") != 2 * head["old_maximum_loss_calls"] - 1):
                        raise ValueError("component trigger refinement loss cap differs")
                    if head.get("kind") == "target_refinement" and (
                            old_component["maximum_component_loss_calls"] != head["old_maximum_loss_calls"] - 1
                            or component["maximum_component_loss_calls"] != head["new_maximum_loss_calls"] - 1):
                        raise ValueError("target refinement component loss cap differs")
                    if head.get("kind") in ("protected_descent_refinement", "relative_progress_refinement"):
                        old_count, new_count = head.get("old_maximum_loss_calls"), head.get("new_maximum_loss_calls")
                        multiplier = 2 if old_component["trigger"] == "invalid-original-linear-segment-or-first-finite-rejection" else 1
                        if (type(new_count) is not int or new_count < 2 or type(old_count) is not int
                                or old_count != 1 + multiplier * (new_count - 1)):
                            raise ValueError("protected descent refinement loss cap differs")
                except ValueError as error:
                    raise ValueError(failure) from error
                component, callback = old_component, old_callback
            elif "kind" in head or head.get("previous_transition") is not None:
                raise ValueError(failure)
            else:
                component, callback = None, None
        guard_hash, schedule_hash, clock = head["old_postproposal_hash"], head["old_schedule_hash"], head["effective_update"]
        heads[index] = head.get("previous_transition")
    return schedule_previous, policy_previous


def _postproposal_policy_transition(parent, parent_config, binding, training, declaration):
    schedule_previous, previous = _postproposal_lineage(parent, parent_config)
    if declaration is None:
        return previous
    return _postproposal_policy_edge(parent, parent_config.get("postproposal"), parent_config.get("training"),
        binding, training, declaration, schedule_previous, previous)


def _postproposal_policy_edge(parent, old_binding, old_training, binding, training, declaration,
                             schedule_previous, previous):
    refinement = _component_refinement(declaration)
    selection = _selection_refinement(declaration)
    keys = {"effective_update", "reason", "kind"} if refinement or selection else {"effective_update", "reason"}
    if (not isinstance(declaration, Mapping) or set(declaration) != keys
            or type(declaration["effective_update"]) is not int or declaration["effective_update"] != parent.update_index
            or not isinstance(declaration["reason"], str) or not declaration["reason"].strip()):
        raise ValueError("invalid postproposal policy transition at the parent clock")
    if selection:
        _check_selection_profiles(old_binding, binding)
        if canonical_json(old_training) != canonical_json(training):
            raise ValueError("selection refinement changed its training binding")
        schedule_hash = stable_hash(None if training is None else training["schedule"])
        return json.loads(canonical_json({
            "schema": "generic_neural_solver.postproposal_policy_transition.v1", **declaration,
            "parent_checkpoint_hash": stable_hash(parent.to_dict()),
            "parent_design_hash": parent.metadata["replication_design_sha256"],
            "old_postproposal_hash": stable_hash(old_binding), "new_postproposal_hash": stable_hash(binding),
            "old_binding": old_binding, "new_binding": binding,
            "old_schedule_hash": schedule_hash, "new_schedule_hash": schedule_hash,
            "component_fallback": binding["profile"].get("component_fallback"),
            "component_callback": binding.get("component_callback"), "previous_transition": previous,
            "previous_schedule_transition_hash": stable_hash(schedule_previous),
        }))
    if old_binding is None or binding is None or old_training is None or training is None:
        raise ValueError("postproposal policy transition requires old/new guards and scheduled training")
    old_profile, new_profile = json.loads(canonical_json(old_binding)), json.loads(canonical_json(binding))
    component = new_profile["profile"].pop("component_fallback", None)
    callback = new_profile.pop("component_callback", None)
    refinement_record = {}
    if refinement:
        old_component = old_profile["profile"].pop("component_fallback", None)
        old_callback = old_profile.pop("component_callback", None)
        trigger_refinement = declaration["kind"] == "component_trigger_refinement"
        protected_descent_refinement = declaration["kind"] == "protected_descent_refinement"
        relative_progress_refinement = declaration["kind"] == "relative_progress_refinement"
        coupled_component_refinement = declaration["kind"] == "coupled_component_refinement"
        target_refinement = declaration["kind"] == "target_refinement"
        _check_component_refinement(old_component, old_callback, component, callback,
            trigger_refinement=trigger_refinement, protected_descent_refinement=protected_descent_refinement,
            relative_progress_refinement=relative_progress_refinement,
            coupled_component_refinement=coupled_component_refinement, target_refinement=target_refinement)
        refinement_record = {"old_component_fallback": old_component, "old_component_callback": old_callback}
        if trigger_refinement or target_refinement:
            old_count = old_profile["profile"].get("maximum_loss_calls")
            new_count = new_profile["profile"].get("maximum_loss_calls")
            fractions = old_profile["profile"].get("fractions", [])
            if (type(old_count) is not int or old_count != 1 + len(fractions)
                    or type(new_count) is not int or new_count != 1 + 2 * len(fractions)):
                raise ValueError("component trigger refinement requires its exact derived finite loss cap")
            refinement_record.update(old_maximum_loss_calls=old_count, new_maximum_loss_calls=new_count)
            new_profile["profile"]["maximum_loss_calls"] = old_count
            if target_refinement and (old_component["maximum_component_loss_calls"] != len(fractions)
                    or component["maximum_component_loss_calls"] != 2 * len(fractions)):
                raise ValueError("target refinement requires its exact derived component loss cap")
        elif protected_descent_refinement or relative_progress_refinement or coupled_component_refinement:
            old_count = old_profile["profile"].get("maximum_loss_calls")
            new_count = new_profile["profile"].get("maximum_loss_calls")
            fractions = old_profile["profile"].get("fractions", [])
            multiplier = 2 if old_component["trigger"] == "invalid-original-linear-segment-or-first-finite-rejection" else 1
            if (type(old_count) is not int or old_count != 1 + multiplier * len(fractions)
                    or type(new_count) is not int or new_count != 1 + len(fractions)):
                raise ValueError("protected descent refinement requires its exact derived loss cap")
            refinement_record.update(old_maximum_loss_calls=old_count, new_maximum_loss_calls=new_count)
            new_profile["profile"]["maximum_loss_calls"] = old_count
            if coupled_component_refinement and component["maximum_component_loss_calls"] != len(fractions):
                raise ValueError("coupled refinement requires its exact derived component loss cap")
    elif ("component_fallback" in old_profile["profile"] or "component_callback" in old_profile
            or not isinstance(component, Mapping) or not component or not isinstance(callback, Mapping) or not callback):
        raise ValueError("postproposal policy transition only enables a declared component provider")
    old_schedule = old_profile["profile"].get("profile", {}).get("training_schedule")
    new_schedule = new_profile["profile"].get("profile", {}).get("training_schedule")
    if (old_schedule != old_training["schedule"] or new_schedule != training["schedule"]
            or old_schedule != new_schedule or canonical_json(old_profile) != canonical_json(new_profile)):
        raise ValueError("postproposal policy transition changed schedule, callbacks, context or numerical profile")
    return json.loads(canonical_json({
        "schema": "generic_neural_solver.postproposal_policy_transition.v1", **declaration,
        "parent_checkpoint_hash": stable_hash(parent.to_dict()),
        "parent_design_hash": parent.metadata["replication_design_sha256"],
        "old_postproposal_hash": stable_hash(old_binding), "new_postproposal_hash": stable_hash(binding),
        "old_schedule_hash": stable_hash(old_schedule), "new_schedule_hash": stable_hash(new_schedule),
        "component_fallback": component, "component_callback": callback, "previous_transition": previous,
        "previous_schedule_transition_hash": stable_hash(schedule_previous),
        **refinement_record,
    }))


def _postproposal_schedule_transition(parent, parent_config, binding, training, declaration, *, partial=None):
    previous, policy_previous = _postproposal_lineage(parent, parent_config)
    old_binding = parent_config.get("postproposal")
    if declaration is None:
        if canonical_json(old_binding) != canonical_json(binding):
            raise ValueError("continuation postproposal profile changed without a verified transition")
        return previous
    return _postproposal_schedule_edge(parent, old_binding, parent_config.get("training"), binding, training,
        declaration, previous, policy_previous, partial=partial)


def _search_profile_lineage(parent, parent_config):
    _postproposal_lineage(parent, parent_config)
    return parent_config.get("checkpoint_continuation", {}).get("postproposal_search_profile_transition")


def _search_profile_components(binding):
    profile = json.loads(canonical_json(binding["profile"]))
    fractions = profile.pop("fractions", None)
    maximum_loss_calls = profile.pop("maximum_loss_calls", None)
    component = profile.get("component_fallback")
    maximum_component_loss_calls = None
    if isinstance(component, Mapping):
        component = dict(component)
        maximum_component_loss_calls = component.pop("maximum_component_loss_calls", None)
        profile["component_fallback"] = component
    return profile, fractions, maximum_loss_calls, maximum_component_loss_calls


def _check_search_profiles(old_binding, binding):
    if not isinstance(old_binding, Mapping) or not isinstance(binding, Mapping):
        raise TypeError("postproposal search-profile transition requires old and new guards")
    old_profile, old_fractions, old_count, old_component_count = _search_profile_components(old_binding)
    new_profile, new_fractions, new_count, new_component_count = _search_profile_components(binding)
    if (not isinstance(old_fractions, list) or not isinstance(new_fractions, list)
            or not old_fractions or not new_fractions
            or new_fractions[:len(old_fractions)] != old_fractions
            or len(new_fractions) <= len(old_fractions)
            or any(type(value) is not float or not np.isfinite(value) or value <= 0 or value > 1
                   for value in new_fractions)
            or any(first <= second for first, second in pairwise(new_fractions))):
        raise ValueError("search-profile transition must append a strictly decreasing positive fraction suffix")
    for profile, fractions, count, component_count in (
            (old_profile, old_fractions, old_count, old_component_count),
            (new_profile, new_fractions, new_count, new_component_count)):
        component = profile.get("component_fallback") or {}
        multiplier = 2 if (component.get("finite_target_refinement") is not None or component.get("trigger") ==
            "invalid-original-linear-segment-or-first-finite-rejection") else 1
        expected_component_count = multiplier * len(fractions) if component.get("coupled_finite_guard") is not None else None
        if type(count) is not int or count != 1 + multiplier * len(fractions) or component_count != expected_component_count:
            raise ValueError("postproposal loss cap is not derived from its fraction profile")
    if (canonical_json(old_profile) != canonical_json(new_profile)
            or canonical_json({key: value for key, value in old_binding.items() if key != "profile"})
            != canonical_json({key: value for key, value in binding.items() if key != "profile"})):
        raise ValueError("search-profile transition changed callbacks or guard semantics")


def _check_search_profile_record(record):
    old_binding, binding = record.get("old_binding"), record.get("new_binding")
    _check_search_profiles(old_binding, binding)
    if (record.get("old_postproposal_hash") != stable_hash(old_binding)
            or record.get("new_postproposal_hash") != stable_hash(binding)):
        raise ValueError("search-profile lineage hashes differ from its bound profiles")


def _postproposal_search_profile_transition(parent, parent_config, binding, declaration):
    previous = _search_profile_lineage(parent, parent_config)
    if declaration is None:
        return previous
    if (not isinstance(declaration, Mapping)
            or set(declaration) != {"effective_update", "reason"}
            or type(declaration["effective_update"]) is not int
            or declaration["effective_update"] != parent.update_index
            or not isinstance(declaration["reason"], str) or not declaration["reason"].strip()):
        raise ValueError("invalid postproposal search-profile transition at the parent clock")
    old_binding = parent_config.get("postproposal")
    _check_search_profiles(old_binding, binding)
    training = parent_config.get("training")
    schedule_hash = stable_hash(None if training is None else training["schedule"])
    old_continuation = parent_config.get("checkpoint_continuation", {})
    return json.loads(canonical_json({
        "schema": "generic_neural_solver.postproposal_search_profile_transition.v1",
        **declaration,
        "parent_checkpoint_hash": stable_hash(parent.to_dict()),
        "parent_design_hash": parent.metadata["replication_design_sha256"],
        "old_postproposal_hash": stable_hash(old_binding),
        "new_postproposal_hash": stable_hash(binding),
        "old_binding": old_binding, "new_binding": binding,
        "old_schedule_hash": schedule_hash, "new_schedule_hash": schedule_hash,
        "previous_schedule_transition_hash": stable_hash(old_continuation.get("postproposal_schedule_transition")),
        "previous_policy_transition_hash": stable_hash(old_continuation.get("postproposal_policy_transition")),
        "previous_transition": previous,
    }))


def _postproposal_schedule_edge(parent, old_binding, old_training, binding, training, declaration,
                               previous, policy_previous, *, partial=None):
    if (not isinstance(declaration, Mapping) or set(declaration) != {"effective_update", "reason"}
            or type(declaration["effective_update"]) is not int
            or declaration["effective_update"] != parent.update_index
            or not isinstance(declaration["reason"], str) or not declaration["reason"].strip()):
        raise ValueError("invalid postproposal schedule transition at the parent clock")
    if old_binding is None or binding is None or old_training is None or training is None:
        raise ValueError("postproposal schedule transition requires old/new guards and scheduled training")
    old_profile = json.loads(canonical_json(old_binding))
    new_profile = json.loads(canonical_json(binding))
    old_context = old_profile["profile"].get("profile", {})
    new_context = new_profile["profile"].get("profile", {})
    old_schedule = old_context.pop("training_schedule", None)
    new_schedule = new_context.pop("training_schedule", None)
    if (old_schedule != old_training["schedule"] or new_schedule != training["schedule"]
            or len(old_schedule) != (parent.update_index if partial is None else partial["old_schedule_length"])
            or len(new_schedule) <= len(old_schedule)
            or old_schedule != new_schedule[:len(old_schedule)]):
        raise ValueError("postproposal schedule transition must extend the exact generic schedule prefix")
    if canonical_json(old_profile) != canonical_json(new_profile):
        raise ValueError("postproposal schedule transition changed callbacks, context or numerical profile")
    return json.loads(canonical_json({
        "schema": "generic_neural_solver.postproposal_schedule_transition.v1", **declaration,
        "parent_checkpoint_hash": stable_hash(parent.to_dict()),
        "parent_design_hash": parent.metadata["replication_design_sha256"],
        "old_postproposal_hash": stable_hash(old_binding), "new_postproposal_hash": stable_hash(binding),
        "old_schedule_hash": stable_hash(old_schedule), "new_schedule_hash": stable_hash(new_schedule),
        "previous_transition": previous,
        **({"previous_policy_transition_hash": stable_hash(policy_previous)} if policy_previous is not None else {}),
    }))


def _postproposal_transitions(parent, parent_config, binding, training, policy_declaration, schedule_declaration,
                             *, partial=None, search_profile_declaration=None):
    if search_profile_declaration is not None and canonical_json(parent_config.get("training")) != canonical_json(training):
        raise ValueError("search-profile transition must preserve the entire training binding")
    search_previous = parent_config.get("checkpoint_continuation", {}).get("postproposal_search_profile_transition")

    def link_search(schedule_transition, policy_transition):
        if search_previous is not None:
            for declaration, record in ((schedule_declaration, schedule_transition), (policy_declaration, policy_transition)):
                if declaration is not None and record is not None:
                    record["previous_search_profile_transition_hash"] = stable_hash(search_previous)
        return schedule_transition, policy_transition

    if policy_declaration is not None and schedule_declaration is not None:
        if not (_component_refinement(policy_declaration) or _selection_refinement(policy_declaration)) or partial is not None:
            raise ValueError("only completed-parent component refinement may combine a new schedule transition")
        schedule_previous, policy_previous = _postproposal_lineage(parent, parent_config)
        old_binding, old_training = parent_config.get("postproposal"), parent_config.get("training")
        if old_binding is None or old_training is None or training is None:
            raise ValueError("postproposal refinement requires old/new guards and scheduled training")
        intermediate = json.loads(canonical_json(old_binding))
        intermediate["profile"]["profile"]["training_schedule"] = training["schedule"]
        schedule_transition = _postproposal_schedule_edge(parent, old_binding, old_training, intermediate, training,
            schedule_declaration, schedule_previous, policy_previous)
        link_search(schedule_transition, None)
        policy_transition = _postproposal_policy_edge(parent, intermediate, training, binding, training,
            policy_declaration, schedule_transition, policy_previous)
        return link_search(schedule_transition, policy_transition)
    policy_transition = _postproposal_policy_transition(parent, parent_config, binding, training, policy_declaration)
    schedule_binding = parent_config.get("postproposal") if (
        policy_declaration is not None or search_profile_declaration is not None) else binding
    schedule_transition = _postproposal_schedule_transition(parent, parent_config, schedule_binding, training,
        schedule_declaration, partial=partial)
    return link_search(schedule_transition, policy_transition)


class _ArchivedPostproposal:
    def __init__(self, binding):
        self.declaration = ImmutableJSONMapping(binding)
        self.binding = self.declaration["profile"]
        self.binding_hash = stable_hash(self.binding)
        self._declaration_hash = stable_hash(self.declaration)

    def validate_binding(self):
        if (stable_hash(self.declaration) != self._declaration_hash
                or stable_hash(self.binding) != self.binding_hash
                or canonical_json(self.binding) != canonical_json(self.declaration["profile"])):
            raise ValueError("archived postproposal metadata changed")

    def __call__(self, *args, **kwargs):
        raise RuntimeError("archived postproposal metadata cannot execute numerical work")


def _archived_postproposal(postproposal, binding, *, refinement=False, search_profile=False):
    from .generic_postproposal import GuardedPostproposal

    if type(postproposal) is not GuardedPostproposal:
        raise ValueError("guard transition requires the bound GuardedPostproposal implementation")
    if (getattr(postproposal, "residual_provider", None) is not None
            or binding["profile"].get("curvature_direction") is not None):
        raise ValueError("retained curvature profile transitions are not qualified")
    if search_profile:
        return _ArchivedPostproposal(binding)
    if refinement:
        if binding["profile"].get("component_fallback") is None or binding.get("component_callback") is None:
            raise ValueError("archived component refinement requires an enabled parent binding")
        return _ArchivedPostproposal(binding)
    profile = binding["profile"]
    component = profile.get("component_fallback")
    options = {}
    if profile.get("finite_selection") is not None:
        options["lookahead_steps"] = profile["finite_selection"]["lookahead_steps"]
    if component is not None:
        if canonical_json(component) != canonical_json(postproposal.binding.get("component_fallback")):
            raise ValueError("archived component provider differs from the retained provider")
        options.update({"component_rows": postproposal.component_rows, "component_binding": component["provider"],
                   "finite_component_fallback": postproposal.finite_component_fallback,
                   "extra_descent_indices": tuple(component.get("extra_descent_indices", ())),
                   "component_direction": component.get("component_direction", "equality")})
        if component.get("coupled_finite_guard") is not None:
            options.update(coupled_component_guard=True, component_loss_values=postproposal.component_loss_values,
                           component_loss_binding=component["component_loss_provider"])
        if component.get("finite_target_refinement") is not None:
            options["finite_target_refinement"] = True
    archived = GuardedPostproposal(postproposal.losses, postproposal.batch_binding, profile["D"], profile["T"],
        binding=profile["profile"], margin_fraction=profile["margin_fraction"],
        fractions=tuple(profile["fractions"]), relative_tolerance=profile["relative_tolerance"], **options)
    if canonical_json(archived.binding) != canonical_json(profile):
        raise ValueError("archived postproposal profile cannot be reconstructed unchanged")
    return archived


def build_runner(output, *, task_ids, denominators, threshold, parameters, objective, data_binding,
                 source_evidence, learning_rate, updates, boundary_every=1, objective_values=None,
                 training_objective=None, training_schedule=None, role_provider_factory=None,
                 role_binding=None, lifecycle_endpoint="certification", stage_id=STAGE_ID,
                 continuation_checkpoint=None, continuation_binding=None, estimator_transition=None,
                 learning_rate_transition=None, method_policy_transition=None, postproposal=None,
                 postproposal_schedule_transition=None, partial_parent_continuation=None,
                 postproposal_policy_transition=None, postproposal_search_profile_transition=None,
                 budget_stop=None, budget_cpu_seconds=None,
                 initial_preferred_method=None):
    """Return (runner, adapter, provider) without calling or serializing objective.

    Optional partial_parent_continuation admits a latest committed parent with
    a bound rolled-back postproposal refusal. Its effective_update, termination
    and reason preserve the entire old schedule, including unconsumed entries.
    It retains numerical state and requires a fresh child control before updates;
    completed-parent behavior and numerical guard profiles remain unchanged.

    The callback returns float64 raw[n], normalized[n], normalized gradients[n,P].
    ``data_binding`` contains ``objective_recipe`` and ``objective_source`` plus
    caller-declared fixed data. The implementation source must also occur in
    ``source_evidence``. Reconstruct with identical arguments and a freshly built
    callback to recover; serialized checkpoints do not contain executable code.
    The certified shared route supports one to nine tasks and exact fixed rounds.
    Optional ``objective_values`` returns only raw[n], normalized[n] for roles.
    It must compute the same losses; overlapping calls at one policy are checked
    without extra evaluations. Bind a wrapper in a different source file with
    ``data_binding.objective_values_source`` and include it in source evidence.
    Optional ``training_objective(parameters, update_index)`` and a tuple of
    nonempty ``training_schedule`` bindings select minibatch gradients only.
    Both must be supplied together; the schedule contains exactly ``updates``
    entries. Bind its callback with ``data_binding.training_objective_source``
    in source evidence. Schedule references are checked, captured and hashed.
    Roles still measure the full finite population; normalization uses the same
    denominators and no full-versus-minibatch parity comparison is performed.

    Optional initial_preferred_method selects a shared method for a fresh attempt.
    None retains the legacy CAGrad default. Non-default choices are bound before
    configuration and initial-state hashing, with fresh slots, RNG and membership.
    Any explicit choice is refused on continuation; use method_policy_transition.

    Continuation preserves numerical state and policy origin, binds new design
    provenance, and appends a fresh entry control at the parent's clock. Updates
    then counts additional work; supply the entire schedule through the new end.
    Parent references bind a completed result, typed checkpoint (or result with
    final_state), runtime configuration and ordered full control archive files.
    Keep objective recipe/data unchanged except the separately verified schedule.
    An explicit estimator_transition may replace only the scheduled training
    callback at the parent's clock. It supplies effective_update, a nonempty
    recipe and a reason; verified old/new callbacks and parent state are retained.
    Optional learning_rate_transition supplies effective_update and reason to
    scale saved Adam/method rates by learning_rate / parent configured rate.
    Slots, clocks, fallback-rate ratios and all other Adam settings are retained.
    Later unchanged continuation inherits its lineage without rescaling again.
    Simultaneous newly declared rate and estimator changes are unsupported.
    Optional method_policy_transition supplies effective_update, old_preferred,
    new_preferred and reason. It changes only the inherited method preference,
    preserves source fallback and every saved rate, and binds its lineage.
    It cannot be combined with a newly declared rate or estimator transition.
    Role factories use adapter.policy_metadata for the retained policy origin.
    Optional role_provider_factory(adapter, stage, output) materializes separate
    roles after the training identity is frozen. Its role_binding pins inputs,
    factory source, role coordinates and engineering measure beforehand. Distinct
    training D and gate T require explicit separate-training-gates-v1 metadata.
    The validation-only lifecycle binds a single-stage exit before certification.
    Optional budget_stop declares training_cpu_limit and boundary_cpu_reserve.
    Its source-bound budget_cpu_seconds callback returns cumulative training CPU
    including prior attempts. A committed control boundary can end training
    without changing membership or the planned horizon; final roles still run.
    Optional postproposal supplies a bound finite-loss guard after the shared
    optimizer. Its loss and batch-binding callbacks require postproposal_source
    (and postproposal_binding_source if different) in data_binding/source_evidence.
    A different guard profile cannot be silently continued from an old state.
    Explicit kind="finite_selection_refinement" enables the qualified bounded
    selector while retaining the entire other guard/input binding. Completed
    parents may compose this with an exact schedule-prefix extension; partial
    parents retain their original unconsumed schedule and marked refusal.
    Optional postproposal_schedule_transition supplies effective_update and reason
    to extend only the guard caller context's training_schedule. Old/new context
    schedules must equal the generic schedules with an unchanged prefix. Every
    other guard field and callback stays identical; all numerical state is kept.
    Optional postproposal_policy_transition supplies effective_update and reason
    to enable a previously absent component fallback on an unchanged schedule.
    Bind its callable with postproposal_component_source without changing the
    objective_recipe. All old guard settings, callbacks and numerical state
    stay identical. Its lineage composes with inherited schedule transitions.
    An explicit kind="component_refinement" instead replaces only an enabled
    provider and its source-bound component callback. It may accompany a
    schedule extension at a completed parent; the metadata edges compose
    schedule first, then policy. Partial refinement retains the whole schedule.
    An explicit kind="target_refinement" enables the optional finite target
    redistribution on a coupled parent, changing only its declared policy and
    derived loss caps. All callbacks, normalization and numerical state stay
    identical; recovery reconstructs the option without numerical calls.
    Optional postproposal_search_profile_transition appends a strictly
    decreasing fraction suffix to an existing guard at a verified continuation
    boundary. It changes only the derived finite loss-call caps; callbacks,
    guard semantics, normalization, objective data and numerical state remain
    identical. It can accompany an explicit rejected partial parent, including
    full-population training without a schedule, but no other numerical/profile
    transition. It never changes the accepted Rotemberg default profile.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("finite-objective runner requires CPU-only CUDA_VISIBLE_DEVICES=-1")
    from .generic_execution_boundary import METHODS

    if initial_preferred_method is not None:
        if not isinstance(initial_preferred_method, str) or initial_preferred_method not in METHODS:
            raise ValueError("initial_preferred_method must name a shared method")
        if continuation_checkpoint is not None or continuation_binding is not None:
            raise ValueError("initial_preferred_method is only supported for fresh attempts")
    preferred_method = "cagrad" if initial_preferred_method is None else initial_preferred_method
    if (budget_stop is None) != (budget_cpu_seconds is None):
        raise ValueError("budget-stop policy and CPU callback must be supplied together")
    if budget_stop is not None:
        if not isinstance(budget_stop, Mapping) or not callable(budget_cpu_seconds):
            raise ValueError("budget stop requires a policy mapping and CPU callback")
        budget_stop = ImmutableJSONMapping(budget_stop)
    if partial_parent_continuation is not None:
        declaration = partial_parent_continuation
        if (continuation_checkpoint is None or not isinstance(declaration, Mapping)
                or set(declaration) != {"effective_update", "termination", "reason"}
                or type(declaration["effective_update"]) is not int
                or declaration["termination"] != "postproposal_rejected"
                or not isinstance(declaration["reason"], str) or not declaration["reason"].strip()):
            raise ValueError("invalid explicit partial-parent continuation declaration")
        if estimator_transition is not None or learning_rate_transition is not None:
            raise ValueError("partial continuation does not combine new estimator or rate transitions")
    if estimator_transition is not None and continuation_checkpoint is None:
        raise ValueError("an estimator transition requires a verified continuation parent")
    if postproposal_schedule_transition is not None and continuation_checkpoint is None:
        raise ValueError("a postproposal schedule transition requires a verified continuation parent")
    if postproposal_policy_transition is not None:
        if continuation_checkpoint is None:
            raise ValueError("a postproposal policy transition requires a verified continuation parent")
        if (any(value is not None for value in (estimator_transition, learning_rate_transition, method_policy_transition))
                or postproposal_schedule_transition is not None and (
                    not (_component_refinement(postproposal_policy_transition)
                         or _selection_refinement(postproposal_policy_transition)) or partial_parent_continuation is not None)):
            raise ValueError("postproposal policy enablement does not combine new schedule, estimator, rate or method transitions")
    if postproposal_search_profile_transition is not None:
        if continuation_checkpoint is None:
            raise ValueError("a postproposal search-profile transition requires a verified continuation parent")
        if any(value is not None for value in (
                estimator_transition, learning_rate_transition, method_policy_transition,
                postproposal_schedule_transition, postproposal_policy_transition)):
            raise ValueError("postproposal search-profile transition cannot combine another state or guard transition")
    if learning_rate_transition is not None:
        if continuation_checkpoint is None:
            raise ValueError("a learning-rate transition requires a verified continuation parent")
        if estimator_transition is not None:
            raise ValueError("simultaneous new rate and estimator transitions are unsupported by this engineering API")
    if method_policy_transition is not None:
        if continuation_checkpoint is None:
            raise ValueError("a method-policy transition requires a verified continuation parent")
        if estimator_transition is not None or learning_rate_transition is not None:
            raise ValueError("simultaneous new method-policy and rate/estimator transitions are unsupported by this engineering API")
    if not isinstance(task_ids, (tuple, list)) or not 1 <= len(task_ids) <= 9:
        raise ValueError("finite-objective task_ids must contain one to nine tasks")
    task_ids = tuple(task_ids)
    if any(not isinstance(task, str) or not task.strip() for task in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("finite-objective task IDs must be nonempty and unique")
    parameters = _array("parameters", parameters)
    denominators = _array("denominators", denominators, (len(task_ids),))
    with np.errstate(over="ignore", divide="ignore"):
        if not np.all(denominators > 0) or not np.isfinite(1. / denominators).all():
            raise ValueError("denominators and inverse scales must be finite and positive")
    threshold, learning_rate = _positive("threshold", threshold), _positive("learning_rate", learning_rate)
    if lifecycle_endpoint not in ("certification", "validation-only"):
        raise ValueError("unknown finite runner lifecycle endpoint")
    if (role_provider_factory is None) != (role_binding is None):
        raise ValueError("role_provider_factory and role_binding must be supplied together")
    role_declaration, role_references, factory_binding, role_coordinates = None, [], None, None
    if role_provider_factory is not None:
        role_declaration, role_references, factory_binding, role_coordinates = _role_declaration(
            role_provider_factory, role_binding, task_ids, denominators, threshold)
    role_coordinate_contract = _coordinate_contract(role_declaration)
    training_measure = MEASURE if role_declaration is None else INJECTED_TRAINING_MEASURE
    if type(updates) is not int or updates < 1 or type(boundary_every) is not int or boundary_every < 1 or updates % boundary_every:
        raise ValueError("positive integer boundary_every must divide positive integer updates")
    if not isinstance(stage_id, str) or not stage_id.strip():
        raise ValueError("finite-objective stage_id must be nonempty")
    if (continuation_checkpoint is None) != (continuation_binding is None):
        raise ValueError("continuation checkpoint and binding must be supplied together")
    parent_info = None if continuation_checkpoint is None else _continuation_parent(
        continuation_checkpoint, continuation_binding, partial_parent_continuation)
    start_update = 0 if parent_info is None else parent_info[0].update_index
    stop_update = start_update + updates
    if (training_objective is None) != (training_schedule is None):
        raise ValueError("training_objective and training_schedule must be supplied together")
    if training_schedule is not None and (
            not isinstance(training_schedule, tuple) or len(training_schedule) != stop_update
            or any(not isinstance(entry, Mapping) or not entry for entry in training_schedule)):
        raise ValueError("training_schedule must be a tuple of nonempty bindings through start_update + updates")
    if training_schedule is not None and any(
            "update_index" in entry and (type(entry["update_index"]) is not int or entry["update_index"] != index)
            for index, entry in enumerate(training_schedule)):
        raise ValueError("training_schedule update_index must match its zero-based position")
    evidence, evidence_references = _evidence(source_evidence)
    callback_binding = _callback_binding(objective, data_binding, evidence_references)
    budget_callback = None
    if budget_stop is not None:
        implementation = getattr(budget_cpu_seconds, "python_function", budget_cpu_seconds)
        if not inspect.isfunction(implementation) and not inspect.ismethod(implementation):
            implementation = implementation.__call__
        filename = inspect.getsourcefile(implementation)
        if filename is None:
            raise ValueError("budget CPU callback requires a declared implementation source")
        budget_callback = _callback_binding(budget_cpu_seconds,
            {"objective_source": _reference(Path(filename)), "objective_recipe": budget_stop}, evidence_references)
        budget_callback["state_mode"] = "cumulative training CPU including prior attempts"
    values_binding = None
    if objective_values is not None:
        source_key = "objective_values_source" if "objective_values_source" in data_binding else "objective_source"
        values_binding = _callback_binding(objective_values, data_binding, evidence_references, source_key)
        values_binding["loss_contract"] = "same raw and normalized finite objective as full callback"
        values_binding["parity_check"] = "naturally overlapping policy evaluations only; no extra callback calls"
    training_binding, training_references = None, []
    if training_objective is not None:
        training_callback = _callback_binding(training_objective, data_binding, evidence_references, "training_objective_source")
        declaration, training_references = _evidence({
            "source": data_binding["training_objective_source"], "schedule": training_schedule})
        training_binding = {"schema": "generic_neural_solver.scheduled_training_objective.v1",
            "callback": training_callback, "schedule": declaration["schedule"],
            "schedule_hash": stable_hash(declaration["schedule"]), "measure": TRAINING_MEASURE}
        if role_declaration is not None:
            training_binding["measure"] = {**TRAINING_MEASURE, "role_evaluation": "separately bound scheduled provider"}
    bound_data, data_references = _evidence(data_binding)
    sources = _sources()
    postproposal_binding = None
    postproposal_references = []
    if "postproposal_residual_source" in data_binding and (
            postproposal is None or getattr(postproposal, "residual_provider", None) is None):
        raise ValueError("postproposal_residual_source requires an enabled residual provider")
    if "postproposal_component_source" in data_binding and (
            postproposal is None or getattr(postproposal, "component_rows", None) is None):
        raise ValueError("postproposal_component_source requires an enabled component provider")
    if "postproposal_component_loss_source" in data_binding and (
            postproposal is None or not getattr(postproposal, "coupled_component_guard", False)):
        raise ValueError("postproposal_component_loss_source requires an enabled coupled component guard")
    if postproposal is not None:
        postproposal.validate_binding()
        gate_denominators = denominators if role_coordinate_contract is None else np.asarray(
            role_coordinate_contract["coordinates"]["control"], np.float64)
        if (not np.array_equal(postproposal.denominators, denominators)
                or not np.array_equal(postproposal.limits, gate_denominators * threshold)):
            raise ValueError("postproposal D/T differs from the runner task coordinates")
        batch_source_key = "postproposal_binding_source" if "postproposal_binding_source" in data_binding else "postproposal_source"
        postproposal_binding = {
            "profile": json.loads(canonical_json(postproposal.binding)),
            "loss_callback": _callback_binding(postproposal.losses, data_binding, evidence_references, "postproposal_source"),
            "batch_binding_callback": _callback_binding(postproposal.batch_binding, data_binding, evidence_references, batch_source_key),
        }
        guard_modules = ["generic_postproposal", "generic_displacement_blend"]
        if getattr(postproposal, "residual_provider", None) is not None:
            residual_callback = _callback_binding(postproposal.residual_provider, data_binding, evidence_references,
                                                  "postproposal_residual_source")
            postproposal_binding["residual_callback"] = residual_callback
            _residual_evidence, residual_references = _evidence({"source": residual_callback["source"],
                "provider": postproposal_binding["profile"]["curvature_direction"]["provider"]})
            postproposal_references.extend(residual_references)
            guard_modules.extend(["generic_curvature_progress", "generic_relative_progress", "generic_residual_subspace"])
        if getattr(postproposal, "component_rows", None) is not None:
            component_callback = _callback_binding(postproposal.component_rows, data_binding, evidence_references,
                                                   "postproposal_component_source")
            postproposal_binding["component_callback"] = component_callback
            if getattr(postproposal, "coupled_component_guard", False):
                loss_callback = _callback_binding(postproposal.component_loss_values, data_binding, evidence_references,
                                                  "postproposal_component_loss_source")
                component_callback["loss_only"] = loss_callback
                postproposal_references.append(loss_callback["source"])
            _component_evidence, component_references = _evidence({"source": component_callback["source"],
                "provider": postproposal_binding["profile"]["component_fallback"]["provider"]})
            postproposal_references.extend(component_references)
            guard_modules.append("generic_component_descent")
            if getattr(postproposal, "component_direction", "equality") == "relative_public_progress":
                guard_modules.append("generic_relative_progress")
        for module in guard_modules:
            path = Path(__file__).parent / f"{module}.py"
            sources[str(path.relative_to(SOURCE_ROOT))] = _reference(path)["sha256"]
    from .generic_certified_numerics import (
        make_certified_method_function,
        make_certified_projected_adam_function,
        numerical_binding,
    )
    from .generic_execution_boundary import DEFAULT_ADAM, ExecutionSpec
    from .generic_model_task_executor import ModelTaskExecutor
    from .generic_replication_design import ReplicationDesignBinding
    from .generic_replication_policy import ReplicationPermanentPassCoordinator
    from .generic_replication_runner import (
        AtomicCheckpointStore,
        GenericReplicationRunner,
        ReplicationStage,
    )

    output = Path(output).resolve()
    from_round = 0 if parent_info is None else parent_info[3]["stage"]["last_round"]
    continuation = None
    rate_transition = None
    method_transition = None
    guard_transition = None
    guard_policy_transition = None
    guard_search_profile_transition = None
    partial_transition = None
    if parent_info is not None:
        parent, declared, parent_references, parent_config, parent_training, parent_training_ref, parent_data_ref, old_controls = parent_info
        old_training = parent_config.get("training")
        if partial_parent_continuation is not None:
            old_stage = parent_config["stage"]
            old_stop = old_stage["start_update"] + (old_stage["last_round"] - old_stage["from_round"]) * old_stage["updates_per_round"]
            if (stop_update < old_stop or (old_training is None) != (training_binding is None)
                    or old_training is not None and len(old_training["schedule"]) != old_stop):
                raise ValueError("partial continuation must preserve the entire originally declared schedule")
            partial_transition = {
                "schema": "generic_neural_solver.partial_parent_continuation.v1", **partial_parent_continuation,
                "parent_checkpoint_hash": stable_hash(parent.to_dict()),
                "parent_design_hash": parent.metadata["replication_design_sha256"],
                "parent_stage": old_stage, "declared_stop_update": old_stop,
                "old_schedule_length": None if old_training is None else old_stop,
                "old_schedule_hash": stable_hash(None if old_training is None else old_training["schedule"]),
                "history_count": len(old_controls),
                "last_control_update": parent.metadata["permanent_pass_rotation"]["history"][-1]["update_index"],
                "refusal_hash": declared["parent_refusal"]["sha256"], "entry_stage_id": stage_id,
            }
        guard_transition, guard_policy_transition = _postproposal_transitions(parent, parent_config,
            postproposal_binding, training_binding, postproposal_policy_transition, postproposal_schedule_transition,
            partial=partial_transition, search_profile_declaration=postproposal_search_profile_transition)
        guard_search_profile_transition = _postproposal_search_profile_transition(
            parent, parent_config, postproposal_binding, postproposal_search_profile_transition)
        transition = None
        if estimator_transition is not None:
            if (not isinstance(estimator_transition, Mapping)
                    or set(estimator_transition) != {"effective_update", "recipe", "reason"}
                    or type(estimator_transition["effective_update"]) is not int
                    or estimator_transition["effective_update"] != start_update
                    or not isinstance(estimator_transition["recipe"], Mapping) or not estimator_transition["recipe"]
                    or not isinstance(estimator_transition["reason"], str) or not estimator_transition["reason"].strip()
                    or old_training is None or training_binding is None):
                raise ValueError("invalid scheduled estimator transition at the parent clock")
            transition, transition_references = _evidence({
                "schema": "generic_neural_solver.estimator_transition.v1",
                **estimator_transition, "parent_checkpoint_hash": stable_hash(parent.to_dict()),
                "parent_training_binding_hash": stable_hash(old_training),
                "old_callback": old_training["callback"], "new_callback": training_binding["callback"],
            })
            training_references.extend(transition_references)
        _check_continuation_coordinates(parent_training, role_coordinate_contract)
        if output == _checked(declared["parent_configuration"]).parent:
            raise ValueError("continuation must use a new output directory")
        if stage_id in {point["control"]["request"]["metadata"]["scheduled_role_scope"]["stage_id"]
                        for point in parent.metadata["permanent_pass_rotation"]["history"]}:
            raise ValueError("continuation stage_id must be new")
        if (not np.array_equal(parameters, parent.policy_state["values"])
                or parent_config["task_ids"] != list(task_ids) or parent_config["denominators"] != denominators.tolist()
                or parent_config["threshold"] != threshold
                or learning_rate_transition is None and parent_config["learning_rate"] != learning_rate
                or parent_config["callback"] != callback_binding or parent_config["values_callback"] != values_binding):
            raise ValueError("continuation policy/task/scale/rate/objective mismatch")
        changing_keys = {"schedule"} if transition is None else {"schedule", "training_objective_source"}
        if postproposal_policy_transition is not None and not _selection_refinement(postproposal_policy_transition):
            source_key = "postproposal_component_source"
            if _component_refinement(postproposal_policy_transition):
                if (source_key not in parent_config["data_binding"]
                        or _reference(_checked(parent_config["data_binding"][source_key])) !=
                        guard_policy_transition["old_component_callback"]["source"]):
                    raise ValueError("postproposal refinement requires the verified old component source")
            elif source_key in parent_config["data_binding"]:
                raise ValueError("postproposal policy transition requires only the verified component source addition")
            if _reference(_checked(bound_data[source_key])) != guard_policy_transition["component_callback"]["source"]:
                raise ValueError("postproposal policy transition requires the verified new component source")
            changing_keys.add(source_key)
            if postproposal_policy_transition.get("kind") == "coupled_component_refinement":
                loss_key = "postproposal_component_loss_source"
                if (loss_key in parent_config["data_binding"] or _reference(_checked(bound_data[loss_key]))
                        != guard_policy_transition["component_callback"]["loss_only"]["source"]):
                    raise ValueError("coupled refinement requires only the verified loss callback source addition")
                changing_keys.add(loss_key)
        old_data = {key: value for key, value in parent_config["data_binding"].items() if key not in changing_keys}
        new_data = {key: value for key, value in bound_data.items() if key not in changing_keys}
        if canonical_json(old_data) != canonical_json(new_data):
            raise ValueError("continuation fixed objective data/recipe changed")
        if ((old_training is None) != (training_binding is None)
                or old_training is not None and (
                    old_training["schedule"] != training_binding["schedule"][
                        :(start_update if partial_transition is None else partial_transition["old_schedule_length"])]
                    or transition is None and old_training["callback"] != training_binding["callback"]
                    or old_training["measure"] != training_binding["measure"])):
            raise ValueError("continuation training schedule prefix or callback changed")
        if training_binding is not None:
            inherited_transition = transition or old_training.get("estimator_transition")
            if inherited_transition is not None:
                training_binding["estimator_transition"] = inherited_transition
        backend = parent.metadata["replication_design"]["backend_binding"]
        if (backend["method"] != numerical_binding() or backend["update"] != numerical_binding()
                or parent.metadata["replication_design"]["optimizer_binding"]["adam"] != asdict(DEFAULT_ADAM)):
            raise ValueError("continuation numerical method/Adam binding changed")
        rate_transition = _rate_transition(parent, parent_config, learning_rate, learning_rate_transition)
        method_transition = _method_policy_transition(parent, parent_config, method_policy_transition, METHODS)
        continuation = {"parent": declared, "parent_checkpoint_hash": stable_hash(parent.to_dict()),
                        "parent_design_hash": parent.metadata["replication_design_sha256"], "start_update": start_update,
                        "policy_origin_metadata": dict(parent.policy_state["metadata"]),
                        "history_count": len(old_controls), "initialization": "retained state plus fresh stage-entry control"}
        if transition is not None:
            continuation["estimator_transition"] = transition
        if rate_transition is not None:
            continuation["learning_rate_transition"] = rate_transition
        if method_transition is not None:
            continuation["method_policy_transition"] = method_transition
        if guard_transition is not None:
            continuation["postproposal_schedule_transition"] = guard_transition
        if guard_policy_transition is not None:
            continuation["postproposal_policy_transition"] = guard_policy_transition
        if guard_search_profile_transition is not None:
            continuation["postproposal_search_profile_transition"] = guard_search_profile_transition
        if partial_transition is not None:
            continuation["partial_parent_continuation"] = partial_transition
    stage = ReplicationStage(stage_id, from_round, from_round + 1, from_round + updates // boundary_every, start_update,
                             updates_per_round=boundary_every, population_size=1, select_after=True)
    configuration = {"schema": "generic_neural_solver.finite_objective_runner.v1", "task_ids": task_ids,
        "denominators": denominators.tolist(), "threshold": threshold, "parameters": parameters.tolist(),
        "callback": callback_binding, "values_callback": values_binding, "data_binding": bound_data, "source_evidence": evidence,
        "learning_rate": learning_rate, "stage": asdict(stage), "sources": sources, "measure": training_measure}
    if preferred_method != "cagrad":
        configuration["initial_preferred_method"] = preferred_method
    if budget_stop is not None:
        configuration["budget_stop"] = {"policy": budget_stop, "callback": budget_callback}
    if training_binding is not None:
        configuration["training"] = training_binding
    if postproposal_binding is not None:
        configuration["postproposal"] = postproposal_binding
    if role_coordinate_contract is not None:
        configuration["role_coordinate_contract"] = role_coordinate_contract
    if lifecycle_endpoint != "certification":
        configuration["lifecycle_endpoint"] = lifecycle_endpoint
    if continuation is not None:
        configuration["checkpoint_continuation"] = continuation
    config_name = "configuration.json" if role_declaration is None else "training-configuration.json"
    config_reference = _write_json(output / config_name, configuration)
    data_reference = _write_json(output / "data-binding.json", bound_data)
    for relative, digest in sources.items():
        _copy({"path": str(SOURCE_ROOT / relative), "sha256": digest}, output / "source" / relative)
    references = [*evidence_references, *data_references, *training_references, *postproposal_references]
    if continuation is not None:
        references.extend(parent_references)
    for index, reference in enumerate(references):
        _copy(reference, output / "evidence" / f"input-{index}.snapshot")
    input_binding = {"configuration": config_reference, "data_binding": data_reference,
                     "callback": callback_binding, "task_ids": list(task_ids), "measure": training_measure}
    if training_binding is not None:
        input_binding["training"] = training_binding
    adapter = FiniteObjectiveAdapter(task_ids, denominators, len(parameters), objective, callback_binding,
        input_binding, sources, [*references, config_reference, data_reference], threshold, objective_values, values_binding,
        training_objective, training_binding, None if role_declaration is None else training_measure, role_coordinate_contract)
    adapter.policy_metadata = ImmutableJSONMapping({"finite_objective_input_fingerprint": adapter.metadata.input_fingerprint}
        if continuation is None else parent.policy_state["metadata"])
    measure, role_manifest, seed_registry = MEASURE, None, None
    if role_declaration is None:
        roles, scope_references = _schedule(output, stage, task_ids, data_reference, adapter.metadata.input_fingerprint)
        provider = FiniteObjectiveProvider(adapter, roles, stage, scope_references)
    else:
        declaration_reference = _write_json(output / "role-declaration.json", role_declaration)
        roles, provider, materialization, generated_references = _materialize_roles(
            role_provider_factory, role_declaration, role_references, role_coordinates, adapter, stage, output / "roles", lifecycle_endpoint)
        role_manifest = {"declaration": declaration_reference, "factory": factory_binding, "materialization": materialization}
        role_reference = _write_json(output / "role-binding.json", role_manifest)
        for index, reference in enumerate([*role_references, *generated_references]):
            _copy(reference, output / "role-evidence" / f"input-{index}.snapshot")
            path = _checked(reference)
            if path.suffix == ".py":
                key = str(path.relative_to(SOURCE_ROOT)) if path.is_relative_to(SOURCE_ROOT) else str(path)
                sources[key] = reference["sha256"]
        measure = {"training": training_measure, "roles": role_declaration["measure"], "scientific_admission": False}
        seed_registry = {"roles": [{"stage": entry.stage_id, "arm": entry.arm_id, "role": entry.role, "round": entry.round_number,
            "banks": [{"bank_id": bank.bank_id, "seed": bank.seed} for bank in entry.manifest.banks]} for entry in roles.bindings],
            "training_schedule": None if training_binding is None else training_binding["schedule"],
            "optimizer": {"pcgrad_seed": 20260722, "next_seed_index": 0} if continuation is None else dict(parent.rng_state)}
        seed_reference = _write_json(output / "role-seeds.json", seed_registry)
        configuration = {**configuration, "training_configuration": config_reference, "role_binding": role_reference,
                         "role_manifest_hash": roles.binding_hash(), "seed_registry": seed_reference, "measure": measure}
        config_reference = _write_json(output / "configuration.json", configuration)
    executor = ModelTaskExecutor(adapter.metadata, ExecutionSpec(len(task_ids), adapter.policy_dimension, 1, 0), roles,
        threshold=threshold, model_binding=adapter.adapter_identity(), coordinator_factory=ReplicationPermanentPassCoordinator,
        update_factory=make_certified_projected_adam_function, update_binding=numerical_binding(),
        method_factory=make_certified_method_function, method_binding=numerical_binding(),
        postproposal=postproposal)
    policy = PolicyView(tuple(parameters.tolist()), "finite-objective-diagnostic-initialization",
                        {"finite_objective_input_fingerprint": adapter.metadata.input_fingerprint})
    initial = CheckpointState(stage_id, 0, policy.to_dict(),
        {"first_moment": [0.] * adapter.policy_dimension, "second_moment": [0.] * adapter.policy_dimension,
         "iteration": 0, "learning_rate": learning_rate},
        {"preferred": preferred_method, "rates": dict.fromkeys(METHODS, learning_rate), "gradnorm_reset_per_call": True, "state": None},
        {"pcgrad_seed": 20260722, "next_seed_index": 0}, metadata={"measure": measure, "diagnostic_only": True})
    if continuation is not None:
        old_input = {"configuration": parent_training_ref, "data_binding": parent_data_ref,
                     "callback": parent_training["callback"], "task_ids": list(task_ids), "measure": parent_training["measure"]}
        if old_training is not None:
            old_input["training"] = old_training
        old_adapter = FiniteObjectiveAdapter(task_ids, denominators, len(parameters), objective, parent_training["callback"],
            old_input, parent_training["sources"], (), threshold, objective_values, parent_training["values_callback"],
            training_objective, old_training, parent_training["measure"], parent_training.get("role_coordinate_contract"))
        old_executor = ModelTaskExecutor(old_adapter.metadata, executor.spec, executor.core.coordinator.roles,
            threshold=threshold, model_binding=old_adapter.adapter_identity(), coordinator_factory=ReplicationPermanentPassCoordinator,
            update_factory=make_certified_projected_adam_function, update_binding=numerical_binding(),
            method_factory=make_certified_method_function, method_binding=numerical_binding(),
            postproposal=postproposal if postproposal_schedule_transition is None and postproposal_policy_transition is None
                and postproposal_search_profile_transition is None else
                _archived_postproposal(postproposal, parent_config["postproposal"],
                    refinement=_component_refinement(postproposal_policy_transition)
                        or _selection_refinement(postproposal_policy_transition),
                    search_profile=postproposal_search_profile_transition is not None))
        old_executor.core._validate_numerical_state(parent)
        if old_adapter.metadata.normalization_hash != adapter.metadata.normalization_hash:
            raise ValueError("continuation normalization/measure changed")
        inherited = executor.core.coordinator.import_history(parent, partial_parent=partial_transition)
        metadata = {key: value for key, value in inherited.metadata.items() if key not in (
            "replication_design", "replication_design_sha256", "replication_initial_state_sha256")}
        metadata.update({"checkpoint_continuation": continuation, "permanent_pass_executor_binding": executor.core.binding,
                         "measure": measure})
        initial = replace(inherited, attempt_id=stage_id, metadata=metadata)
        if learning_rate_transition is not None:
            changed = rate_transition["new_optimizer"]
            initial = replace(initial,
                optimizer_state={**initial.optimizer_state, "learning_rate": changed["saved_learning_rate"]},
                method_state={**initial.method_state, "rates": changed["method_rates"]})
        if method_policy_transition is not None:
            initial = replace(initial,
                method_state={**initial.method_state, "preferred": method_transition["new_preferred"]})
        executor.core._validate_numerical_state(initial)
    governing_reference = evidence.get("master", evidence.get("plan", evidence_references[0]))
    _checked(governing_reference)
    target_hashes = {"data_binding": data_reference["sha256"], "configuration": config_reference["sha256"],
                     "callback": stable_hash(callback_binding), "evidence": stable_hash(evidence)}
    if training_binding is not None:
        target_hashes.update({"training_schedule": training_binding["schedule_hash"],
                              "training_objective": stable_hash(training_binding["callback"])})
    permanent_binding = {"stage_entries": {}, "diagnostic_threshold": threshold, "measure": measure}
    if budget_stop is not None:
        permanent_binding["budget_stop"] = budget_stop
    if len(task_ids) > 7:
        permanent_binding["capacity"] = {"task_count": len(task_ids), "max_constraints": len(task_ids) - 1,
            "supported_max_tasks": 9, "supported_max_constraints": 8, "projector": "certified_direct_svd"}
    if continuation is not None:
        permanent_binding["checkpoint_continuation"] = continuation
    if lifecycle_endpoint != "certification":
        permanent_binding["lifecycle_endpoint"] = lifecycle_endpoint
    if role_manifest is not None:
        target_hashes["role_binding"] = stable_hash(role_manifest)
        target_hashes["training_configuration"] = input_binding["configuration"]["sha256"]
        permanent_binding["role_coordinates"] = role_coordinates.to_dict()
    else:
        seed_registry = {"scope_ordinals": list(range(len(scope_references))), "sampler": None}
    design = ReplicationDesignBinding(stage_id, governing_reference["sha256"], stable_hash(sources), sources,
        environment_binding={"python": sys.version, "prefix": sys.prefix, "cpu_only": True,
            "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}},
        backend_binding={"backend": "tensorflow", "dtype": "float64", "device": "CPU", "method": numerical_binding(), "update": numerical_binding()},
        role_manifest_sha256=roles.binding_hash(), target_hashes=target_hashes,
        initial_state_hashes={str(ARM_ID): stable_hash(initial.to_dict())}, task_ids=task_ids, threshold=threshold, replica_ids=(ARM_ID,),
        expected_selected_replica=None, selection_key=("absolute_threshold_ratio", "replica"), stages=(asdict(stage),),
        optimizer_binding={"adam": asdict(DEFAULT_ADAM), "rate": learning_rate,
            "preferred_method": initial.method_state["preferred"], "fresh_moments": continuation is None,
            **({"postproposal": postproposal_binding} if postproposal_binding is not None else {}),
            **({"learning_rate_transition": rate_transition} if rate_transition is not None else {}),
            **({"method_policy_transition": method_transition} if method_transition is not None else {}),
            **({"postproposal_policy_transition": guard_policy_transition} if guard_policy_transition is not None else {}),
            **({"postproposal_search_profile_transition": guard_search_profile_transition}
               if guard_search_profile_transition is not None else {}),
            **({"postproposal_schedule_transition": guard_transition} if guard_transition is not None else {})},
        permanent_pass_binding=permanent_binding, seed_registry_sha256=stable_hash(seed_registry))
    _write_json(output / "design.json", design.to_dict())
    _write_json(output / "schedule.json", roles.to_dict())

    def prepare(arm, proposed_policy, requested_stage, update):
        if type(arm) is not int or arm != ARM_ID or requested_stage != stage or type(update) is not int or not start_update <= update < stop_update:
            raise ValueError("finite-objective batch request outside frozen update allowance")
        return adapter.prepare_batch(proposed_policy, update)

    store = AtomicCheckpointStore(output / "checkpoints", initial.attempt_id, design)
    if continuation is not None:
        for control in old_controls:
            store.archive_control(control)
    if store.has_initial(ARM_ID):
        store.recover(ARM_ID)
    runner = GenericReplicationRunner(executor, {ARM_ID: adapter}, {ARM_ID: initial}, provider, prepare, store,
                                      (stage,), threshold=threshold, design=design, lifecycle_endpoint=lifecycle_endpoint,
                                      budget_cpu_seconds=budget_cpu_seconds)
    return runner, adapter, provider
