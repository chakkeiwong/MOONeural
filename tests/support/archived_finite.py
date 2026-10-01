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
from pathlib import Path

import numpy as np

from .generic_coverage_runner import (
    ROOT,
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
    paths.append(ROOT / "src/mooneural/artifacts.py")
    return {str(path.relative_to(ROOT)): _reference(path)["sha256"] for path in paths}


class FiniteObjectiveAdapter:
    """Functional host adapter; the supplied callback owns all numerical work."""

    def __init__(self, task_ids, denominators, parameter_dim, objective, callback_binding,
                 input_binding, sources, references, threshold, objective_values=None, values_binding=None,
                 training_objective=None, training_binding=None, measure=None):
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
        self.policy_metadata = None
        normalizations = tuple(NormalizationMetadata(task,
            ScaleSpec(f"{task}.raw", "raw"), ScaleSpec(f"{task}.training", "training", 1. / denominator),
            ScaleSpec(f"{task}.selection", "selection", 1. / denominator),
            ScaleSpec(f"{task}.terminal", "terminal", 1. / denominator))
            for task, denominator in zip(task_ids, denominators.tolist(), strict=True))
        normalization_binding = {"denominators": denominators.tolist(), "threshold": threshold,
                                 "measure": MEASURE if measure is None else measure, "normalization": "raw/D once"}
        if self.training_binding is not None:
            normalization_binding["training_measure"] = self.training_binding["measure"]
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
    from .generic_independent_role_bridge import LossCoordinates

    declaration, references = _evidence(binding)
    required = {"factory_source", "inputs", "coordinates", "measure", "training_row_ids"}
    if not required <= set(declaration):
        raise ValueError("role declaration is incomplete")
    _evidence(declaration["inputs"])
    coordinates = LossCoordinates(**declaration["coordinates"])
    if coordinates.task_ids != task_ids or coordinates.threshold != threshold or any(
            tuple(getattr(coordinates, role)) != tuple(denominators)
            for role in ("training", "control", "validation", "certification")):
        raise ValueError("injected builder requires matching task order, threshold and equal role denominators")
    measure = declaration["measure"]
    if not isinstance(measure, Mapping) or any(measure.get(key) is not False
            for key in ("scientific_admission", "production_integrity_evidence")):
        raise ValueError("injected role measure must declare engineering evidence")
    rows = declaration["training_row_ids"]
    if not isinstance(rows, list) or not rows or any(not isinstance(row, str) or not row for row in rows) or len(set(rows)) != len(rows):
        raise ValueError("unique training row identities required")
    identity = _callback_binding(factory, {"objective_recipe": declaration, "objective_source": declaration["factory_source"]}, references)
    return ImmutableJSONMapping(declaration), references, identity, coordinates


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


def _continuation_parent(checkpoint, binding):
    from .generic_control_evidence import ControlEvidenceArchive, full_control_hash
    from .generic_permanent_pass import PermanentPassState
    from .generic_replication_design import ReplicationDesignBinding

    if not isinstance(checkpoint, CheckpointState):
        raise TypeError("continuation_checkpoint must be a typed CheckpointState")
    declared, references = _evidence(binding)
    if set(declared) != {"parent_result", "parent_checkpoint", "parent_configuration", "parent_control_archive"}:
        raise ValueError("continuation requires parent result/checkpoint/configuration/control references")
    parent = CheckpointState.from_dict(checkpoint.to_dict())
    result = json.loads(_checked(declared["parent_result"]).read_text())
    saved = json.loads(_checked(declared["parent_checkpoint"]).read_text())
    saved = saved.get("final_state", saved)
    if (result.get("training_completed") is not True or result.get("final_state") != parent.to_dict()
            or saved != parent.to_dict()):
        raise ValueError("continuation checkpoint differs from completed parent result")
    config_path = _checked(declared["parent_configuration"])
    config = json.loads(config_path.read_text())
    design = ReplicationDesignBinding.from_dict(parent.metadata["replication_design"])
    design.validate_checkpoint(parent)
    if (design.replica_ids != (ARM_ID,) or design.target_hashes["configuration"] != declared["parent_configuration"]["sha256"]
            or design.stages[-1] != config["stage"]):
        raise ValueError("continuation parent configuration/design mismatch")
    stage = config["stage"]
    if parent.update_index != stage["start_update"] + (stage["last_round"] - stage["from_round"]) * stage["updates_per_round"]:
        raise ValueError("continuation parent must complete its declared stage")
    training_ref = config.get("training_configuration", declared["parent_configuration"])
    training_config = json.loads(_checked(training_ref).read_text())
    data_ref = _reference(config_path.parent / "data-binding.json")
    if data_ref["sha256"] != design.target_hashes["data_binding"]:
        raise ValueError("continuation parent data snapshot differs from design")
    snapshots = [*config_path.parent.glob("evidence/*.snapshot"), *config_path.parent.glob("role-evidence/*.snapshot")]
    snapshot_hashes = {_reference(path)["sha256"] for path in snapshots}
    for relative, digest in config["sources"].items():
        source = config_path.parent / "source" / relative
        if not ((source.is_file() and _reference(source)["sha256"] == digest) or digest in snapshot_hashes):
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


def build_runner(output, *, task_ids, denominators, threshold, parameters, objective, data_binding,
                 source_evidence, learning_rate, updates, boundary_every=1, objective_values=None,
                 training_objective=None, training_schedule=None, role_provider_factory=None,
                 role_binding=None, lifecycle_endpoint="certification", stage_id=STAGE_ID,
                 continuation_checkpoint=None, continuation_binding=None):
    """Return (runner, adapter, provider) without calling or serializing objective.

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

    Continuation preserves numerical state and policy origin, binds new design
    provenance, and appends a fresh entry control at the parent's clock. Updates
    then counts additional work; supply the entire schedule through the new end.
    Parent references bind a completed result, typed checkpoint (or result with
    final_state), runtime configuration and ordered full control archive files.
    Keep objective recipe/data unchanged except the separately verified schedule.
    Role factories use adapter.policy_metadata for the retained policy origin.
    Optional role_provider_factory(adapter, stage, output) materializes separate
    roles after the training identity is frozen. Its role_binding pins inputs,
    factory source, equal role coordinates and engineering measure beforehand.
    The validation-only lifecycle binds a single-stage exit before certification.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("finite-objective runner requires CPU-only CUDA_VISIBLE_DEVICES=-1")
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
    training_measure = MEASURE if role_declaration is None else INJECTED_TRAINING_MEASURE
    if type(updates) is not int or updates < 1 or type(boundary_every) is not int or boundary_every < 1 or updates % boundary_every:
        raise ValueError("positive integer boundary_every must divide positive integer updates")
    if not isinstance(stage_id, str) or not stage_id.strip():
        raise ValueError("finite-objective stage_id must be nonempty")
    if (continuation_checkpoint is None) != (continuation_binding is None):
        raise ValueError("continuation checkpoint and binding must be supplied together")
    parent_info = None if continuation_checkpoint is None else _continuation_parent(continuation_checkpoint, continuation_binding)
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
    from .generic_certified_numerics import (
        make_certified_method_function,
        make_certified_projected_adam_function,
        numerical_binding,
    )
    from .generic_execution_boundary import DEFAULT_ADAM, METHODS, ExecutionSpec
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
    if parent_info is not None:
        parent, declared, parent_references, parent_config, parent_training, parent_training_ref, parent_data_ref, old_controls = parent_info
        if output == _checked(declared["parent_configuration"]).parent:
            raise ValueError("continuation must use a new output directory")
        if stage_id in {point["control"]["request"]["metadata"]["scheduled_role_scope"]["stage_id"]
                        for point in parent.metadata["permanent_pass_rotation"]["history"]}:
            raise ValueError("continuation stage_id must be new")
        if (not np.array_equal(parameters, parent.policy_state["values"])
                or parent_config["task_ids"] != list(task_ids) or parent_config["denominators"] != denominators.tolist()
                or parent_config["threshold"] != threshold or parent_config["learning_rate"] != learning_rate
                or parent_config["callback"] != callback_binding or parent_config["values_callback"] != values_binding):
            raise ValueError("continuation policy/task/scale/rate/objective mismatch")
        old_data = {key: value for key, value in parent_config["data_binding"].items() if key != "schedule"}
        new_data = {key: value for key, value in bound_data.items() if key != "schedule"}
        if canonical_json(old_data) != canonical_json(new_data):
            raise ValueError("continuation fixed objective data/recipe changed")
        old_training = parent_config.get("training")
        if ((old_training is None) != (training_binding is None)
                or old_training is not None and (
                    old_training["schedule"] != training_binding["schedule"][:start_update]
                    or old_training["callback"] != training_binding["callback"]
                    or old_training["measure"] != training_binding["measure"])):
            raise ValueError("continuation training schedule prefix or callback changed")
        backend = parent.metadata["replication_design"]["backend_binding"]
        if (backend["method"] != numerical_binding() or backend["update"] != numerical_binding()
                or parent.metadata["replication_design"]["optimizer_binding"]["adam"] != asdict(DEFAULT_ADAM)):
            raise ValueError("continuation numerical method/Adam binding changed")
        continuation = {"parent": declared, "parent_checkpoint_hash": stable_hash(parent.to_dict()),
                        "parent_design_hash": parent.metadata["replication_design_sha256"], "start_update": start_update,
                        "policy_origin_metadata": dict(parent.policy_state["metadata"]),
                        "history_count": len(old_controls), "initialization": "retained state plus fresh stage-entry control"}
    stage = ReplicationStage(stage_id, from_round, from_round + 1, from_round + updates // boundary_every, start_update,
                             updates_per_round=boundary_every, population_size=1, select_after=True)
    configuration = {"schema": "generic_neural_solver.finite_objective_runner.v1", "task_ids": task_ids,
        "denominators": denominators.tolist(), "threshold": threshold, "parameters": parameters.tolist(),
        "callback": callback_binding, "values_callback": values_binding, "data_binding": bound_data, "source_evidence": evidence,
        "learning_rate": learning_rate, "stage": asdict(stage), "sources": sources, "measure": training_measure}
    if training_binding is not None:
        configuration["training"] = training_binding
    if lifecycle_endpoint != "certification":
        configuration["lifecycle_endpoint"] = lifecycle_endpoint
    if continuation is not None:
        configuration["checkpoint_continuation"] = continuation
    config_name = "configuration.json" if role_declaration is None else "training-configuration.json"
    config_reference = _write_json(output / config_name, configuration)
    data_reference = _write_json(output / "data-binding.json", bound_data)
    for relative, digest in sources.items():
        _copy({"path": relative, "sha256": digest}, output / "source" / relative)
    references = [*evidence_references, *data_references, *training_references]
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
        training_objective, training_binding, None if role_declaration is None else training_measure)
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
                key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
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
        method_factory=make_certified_method_function, method_binding=numerical_binding())
    policy = PolicyView(tuple(parameters.tolist()), "finite-objective-diagnostic-initialization",
                        {"finite_objective_input_fingerprint": adapter.metadata.input_fingerprint})
    initial = CheckpointState(stage_id, 0, policy.to_dict(),
        {"first_moment": [0.] * adapter.policy_dimension, "second_moment": [0.] * adapter.policy_dimension,
         "iteration": 0, "learning_rate": learning_rate},
        {"preferred": "cagrad", "rates": dict.fromkeys(METHODS, learning_rate), "gradnorm_reset_per_call": True, "state": None},
        {"pcgrad_seed": 20260722, "next_seed_index": 0}, metadata={"measure": measure, "diagnostic_only": True})
    if continuation is not None:
        old_input = {"configuration": parent_training_ref, "data_binding": parent_data_ref,
                     "callback": parent_training["callback"], "task_ids": list(task_ids), "measure": parent_training["measure"]}
        if old_training is not None:
            old_input["training"] = old_training
        old_adapter = FiniteObjectiveAdapter(task_ids, denominators, len(parameters), objective, parent_training["callback"],
            old_input, parent_training["sources"], (), threshold, objective_values, parent_training["values_callback"],
            training_objective, old_training, parent_training["measure"])
        old_executor = ModelTaskExecutor(old_adapter.metadata, executor.spec, executor.core.coordinator.roles,
            threshold=threshold, model_binding=old_adapter.adapter_identity(), coordinator_factory=ReplicationPermanentPassCoordinator,
            update_factory=make_certified_projected_adam_function, update_binding=numerical_binding(),
            method_factory=make_certified_method_function, method_binding=numerical_binding())
        old_executor.core._validate_numerical_state(parent)
        if old_adapter.metadata.normalization_hash != adapter.metadata.normalization_hash:
            raise ValueError("continuation normalization/measure changed")
        inherited = executor.core.coordinator.import_history(parent)
        metadata = {key: value for key, value in inherited.metadata.items() if key not in (
            "replication_design", "replication_design_sha256", "replication_initial_state_sha256")}
        metadata.update({"checkpoint_continuation": continuation, "permanent_pass_executor_binding": executor.core.binding,
                         "measure": measure})
        initial = replace(inherited, attempt_id=stage_id, metadata=metadata)
        executor.core._validate_numerical_state(initial)
    governing_reference = evidence.get("master", evidence.get("plan", evidence_references[0]))
    _checked(governing_reference)
    target_hashes = {"data_binding": data_reference["sha256"], "configuration": config_reference["sha256"],
                     "callback": stable_hash(callback_binding), "evidence": stable_hash(evidence)}
    if training_binding is not None:
        target_hashes.update({"training_schedule": training_binding["schedule_hash"],
                              "training_objective": stable_hash(training_binding["callback"])})
    permanent_binding = {"stage_entries": {}, "diagnostic_threshold": threshold, "measure": measure}
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
            "preferred_method": initial.method_state["preferred"], "fresh_moments": continuation is None},
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
                                      (stage,), threshold=threshold, design=design, lifecycle_endpoint=lifecycle_endpoint)
    return runner, adapter, provider
