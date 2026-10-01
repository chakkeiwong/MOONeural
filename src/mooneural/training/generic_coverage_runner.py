"""Auxiliary supervised coverage training through the common execution engine.

All controls and exit receipts measure the same finite training population.
They are not held-out data, independent replicates, or scientific certification.
The 0.01 threshold is an auxiliary regression stopping rule, not a model limit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mooneural.artifacts import atomic_write_json

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

TASK_IDS = ("coverage.value", "coverage.state_derivative", "coverage.parameter_derivative")
THRESHOLD = 0.01
ARM_ID = 0
STAGE_ID = "generic-auxiliary-coverage-v1"
# Library source and client evidence need different anchors in an installed
# wheel. Relative evidence uses the caller's launch directory; generated
# artifact references are absolute. Source snapshots retain package-relative
# names, so installation paths never become snapshot output destinations.
ROOT = Path.cwd().resolve()
SOURCE_ROOT = Path(__file__).resolve().parents[1]
MEASURE = {
    "population": "same fixed training-only regression dataset in every scope",
    "estimator": "exact weighted finite-population loss; no Student-t or sampling uncertainty",
    "scope_ids": "evaluation job identifiers, not independent data identifiers",
    "seed_fields": "scope ordinals required by the schema; no sampling is performed",
    "held_out": False,
    "independent_roles": False,
    "scientific_admission": False,
    "production_integrity_evidence": False,
    "threshold": THRESHOLD,
    "threshold_meaning": "auxiliary normalized regression convergence, not scientific T",
    "selection": "one fixed arm; the engine requires an exit selection receipt",
}
SOURCE_MODULES = (
    "generic_coverage_runner", "generic_coverage_objective", "generic_coverage_initialization",
    "generic_policy_coordinates", "generic_certified_numerics", "generic_replication_runner",
    "generic_replication_design", "generic_replication_policy", "generic_permanent_pass",
    "generic_permanent_pass_executor", "generic_model_task_executor", "generic_execution_boundary",
    "generic_training_contracts", "generic_role_banks", "generic_scheduled_role_banks",
    "generic_control_evidence", "generic_xla_kernels", "moo", "adam",
)


def _plain(value):
    return json.loads(canonical_json(value))


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _checked(reference):
    path = (ROOT / reference["path"]).resolve()
    if _reference(path)["sha256"] != reference["sha256"]:
        raise ValueError(f"coverage bound file changed: {path}")
    return path


def _write_json(path, payload):
    if path.exists():
        if canonical_json(json.loads(path.read_text())) != canonical_json(payload):
            raise ValueError(f"coverage frozen artifact mismatch: {path}")
    else:
        atomic_write_json(path, _plain(payload))
    return _reference(path)


def _copy(reference, destination):
    content = _checked(reference).read_bytes()
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError(f"coverage frozen source snapshot mismatch: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def _evidence(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError("coverage source evidence must be a nonempty artifact mapping")
    result = _plain(value)
    references = []

    def visit(item):
        if isinstance(item, dict):
            if "path" in item or "sha256" in item:
                if set(item) != {"path", "sha256"}:
                    raise ValueError("coverage evidence references require path and sha256")
                references.append(_reference(_checked(item)))
            else:
                for child in item.values():
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(result)
    if not references:
        raise ValueError("coverage source evidence must include a checked artifact reference")
    return result, references


def _sources():
    paths = [Path(__file__).parent / f"{name}.py" for name in SOURCE_MODULES]
    paths.append(SOURCE_ROOT / "artifacts.py")
    return {str(path.relative_to(SOURCE_ROOT)): _reference(path)["sha256"] for path in paths}


def _check_sources(sources):
    for relative, digest in sources.items():
        _checked({"path": str(SOURCE_ROOT / relative), "sha256": digest})


def _validate_inputs(coordinates, width, data, parameters, denominators, split, learning_rate, updates):
    from .generic_coverage_initialization import CoverageTrainingData
    from .generic_policy_coordinates import AffinePolicyCoordinates

    if not isinstance(data, CoverageTrainingData) or not isinstance(coordinates, AffinePolicyCoordinates):
        raise TypeError("coverage data and affine coordinates required")
    input_dim, output_dim = data.groups[0].inputs.shape[1], data.groups[0].values.shape[1]
    if (len(coordinates.input_center), len(coordinates.output_center)) != (input_dim, output_dim):
        raise ValueError("coverage coordinate and dataset dimensions disagree")
    if coordinates.training_binding != data.binding_hash():
        raise ValueError("coverage coordinate profile training binding mismatch")
    if type(width) is not int or width <= 0:
        raise ValueError("coverage width must be a positive integer")
    if type(split) is not int or not 0 < split < input_dim:
        raise ValueError("coverage derivative split must be inside the input dimension")
    if type(updates) is not int or not 1 <= updates <= 50:
        raise ValueError("coverage update allowance must be an integer from 1 to 50")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("coverage learning rate must be finite and positive")
    parameter_dim = input_dim * width + width + width * width + width + width * output_dim + output_dim
    arrays = []
    for name, values, shape in (("parameters", parameters, (parameter_dim,)), ("denominators", denominators, (3,))):
        values = np.asarray(values)
        if values.shape != shape or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"coverage {name} must be finite real values with shape {shape}")
        values = values.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"coverage {name} cannot be represented in float64")
        arrays.append(values)
    if not np.all(arrays[1] > 0) or not np.isfinite(1. / arrays[1]).all():
        raise ValueError("coverage denominators and inverse scales must be finite and positive")
    return arrays


class CoverageAdapter:
    """Functional adapter for three fixed supervised regression objectives."""

    def __init__(self, coordinates, width, data, denominators, split, sources, input_binding):
        from .generic_coverage_objective import make_coverage_objective
        from .generic_model_task_executor import ModelTaskMetadata

        self.coordinates, self.width, self.data = coordinates, width, data
        self.denominators = np.frombuffer(np.asarray(denominators, dtype=np.float64).tobytes(), dtype=np.float64)
        self.split = split
        self.sources = ImmutableJSONMapping(sources)
        input_dim, output_dim = len(coordinates.input_center), len(coordinates.output_center)
        self.policy_dimension = input_dim * width + 2 * width + width**2 + width * output_dim + output_dim
        self.sample_count = sum(len(group.inputs) for group in data.groups)
        normalizations = tuple(NormalizationMetadata(task,
            ScaleSpec(f"{task}.raw", "raw"), ScaleSpec(f"{task}.training", "training", 1. / denominator),
            ScaleSpec(f"{task}.selection", "selection", 1. / denominator),
            ScaleSpec(f"{task}.terminal", "terminal", 1. / denominator))
            for task, denominator in zip(TASK_IDS, self.denominators.tolist(), strict=True))
        self.metadata = ModelTaskMetadata(TaskRegistry(tuple(TaskDefinition(task, index) for index, task in enumerate(TASK_IDS))),
            normalizations, (), (), normalization_binding={"denominators": self.denominators.tolist(),
                "derivative_reductions": {"state": "mean", "parameter": "mean", "outputs": "mean"},
                "auxiliary_threshold": THRESHOLD, "scientific_terminal_limits": None}, input_binding=input_binding)
        self._native = make_coverage_objective(coordinates, width, data, self.denominators, split)

    def adapter_identity(self):
        payload = {"schema": "generic_neural_solver.coverage_adapter.v1", "state_mode": "functional",
                   "sources": self.sources, "profile": self.coordinates.to_dict(), "width": self.width,
                   "data_hash": self.data.binding_hash(), "split": self.split,
                   "normalization_hash": self.metadata.normalization_hash, "input_fingerprint": self.metadata.input_fingerprint}
        return {**payload, "identity_hash": stable_hash(payload)}

    def require_policy(self, policy):
        if not isinstance(policy, PolicyView) or len(policy.values) != self.policy_dimension:
            raise ValueError("coverage policy dimension mismatch")
        if policy.metadata.get("coordinate_profile_hash") != self.coordinates.binding_hash():
            raise ValueError("coverage policy coordinate profile mismatch")
        if policy.metadata.get("coverage_input_fingerprint") != self.metadata.input_fingerprint:
            raise ValueError("coverage policy data/normalization binding mismatch")

    def prepare_batch(self, policy, update_index):
        self.require_policy(policy)
        self.metadata.validate_bindings()
        if type(update_index) is not int or update_index < 0:
            raise ValueError("coverage update must be a nonnegative integer")
        return TaskBatchTensors(TASK_IDS, self.sample_count, self.policy_dimension, backend="tensorflow",
            metadata={"normalization_hash": self.metadata.normalization_hash, "input_fingerprint": self.metadata.input_fingerprint,
                      "input_binding": self.metadata.input_binding, "policy_fingerprint": policy.fingerprint(),
                      "update_index": update_index})

    def evaluate(self, policy):
        import tensorflow as tf

        self.require_policy(policy)
        self.metadata.validate_bindings()
        _check_sources(self.sources)
        raw, normalized, gradients = self._native(tf.constant(policy.values, tf.float64))
        for values, shape in ((raw, (3,)), (normalized, (3,)), (gradients, (3, self.policy_dimension))):
            if not tf.is_tensor(values) or values.dtype != tf.float64 or tuple(values.shape) != shape:
                raise ValueError("coverage objective tensor schema mismatch")
            if not np.isfinite(values.numpy()).all():
                raise ValueError("coverage objective returned nonfinite losses or gradients")
        if np.any(raw.numpy() < 0) or np.any(normalized.numpy() < 0):
            raise ValueError("coverage objective losses must be nonnegative")
        np.testing.assert_allclose(normalized.numpy() * self.denominators, raw.numpy(), rtol=2e-13, atol=1e-14)
        return raw, normalized, gradients

    def compute_task_values_and_gradients(self, policy, batch, update_index):
        import tensorflow as tf

        from .generic_model_task_executor import ModelTaskEvaluation

        if batch.to_dict() != self.prepare_batch(policy, update_index).to_dict():
            raise ValueError("coverage prepared batch policy/update binding mismatch")
        _raw, normalized, gradients = self.evaluate(policy)
        tensors = TaskEvaluationTensors(TASK_IDS, tuple(normalized.numpy().tolist()),
            tuple(tuple(row) for row in gradients.numpy().tolist()), policy_dimension=self.policy_dimension,
            backend="tensorflow", normalization_hash=self.metadata.normalization_hash)
        return ModelTaskEvaluation(tensors, policy.fingerprint(), update_index, self.metadata.input_fingerprint,
                                   normalized, gradients, tf.zeros([0], tf.float64))


class CoverageDiagnosticProvider:
    """Measure exact finite training losses at the common engine's role calls."""

    def __init__(self, adapter, roles, stage, scope_files):
        self.adapter, self.roles, self.stage = adapter, roles, stage
        self.scope_files = tuple(scope_files)

    def _measure(self, role, arm, policy, stage, round_number, update_index):
        if arm != ARM_ID or stage != self.stage:
            raise ValueError("coverage diagnostic request outside frozen design")
        self.adapter.require_policy(policy)
        for reference in self.scope_files:
            _checked(reference)
        request = self.roles.request_for(role, policy, stage_id=stage.stage_id, arm_id=arm,
                                        round_number=round_number, update_index=update_index)
        raw, normalized, _gradients = self.adapter.evaluate(policy)
        metrics = dict(zip(TASK_IDS, normalized.numpy().tolist(), strict=True))
        records = {"schema": "generic_neural_solver.coverage_exact_training_loss.v1", "measure": MEASURE,
                   "data_hash": self.adapter.data.binding_hash(), "profile_hash": self.adapter.coordinates.binding_hash(),
                   "role": role, "update_index": update_index, "policy_fingerprint": policy.fingerprint(),
                   "raw_losses": dict(zip(TASK_IDS, raw.numpy().tolist(), strict=True)), "normalized_losses": metrics,
                   "denominators": self.adapter.denominators.tolist(), "training_rows": self.adapter.sample_count,
                   "groups": [{"group_id": group.group_id, "location": group.location,
                               "rows": len(group.inputs), "mass": group.mass, "data_hash": group.binding_hash()}
                              for group in self.adapter.data.groups],
                   "role_bank_manifest_hash": request.metadata["role_bank_manifest_hash"],
                   "scheduled_role_registry_hash": self.roles.binding_hash(),
                   "data_status": "training-only-deterministic", "scientific_admission": False,
                   "production_integrity_evidence": False, "upper_mse_semantics": "exact finite training objective"}
        return request, metrics, records

    def control(self, arm, policy, stage, round_number):
        update = stage.start_update + (round_number - stage.from_round) * stage.updates_per_round
        request, metrics, records = self._measure("control", arm, policy, stage, round_number, update)
        return ControlEvaluation(metrics, raw_records=records, request=request)

    def validation(self, arm, policy, stage, *, update_index):
        request, metrics, records = self._measure("validation", arm, policy, stage, None, update_index)
        return ValidationEvaluation(metrics, dict.fromkeys(TASK_IDS, 1), records, request=request)

    def certification(self, arm, policy, stage, *, update_index):
        request, metrics, records = self._measure("certification", arm, policy, stage, None, update_index)
        return CertificationResult({task: value <= THRESHOLD for task, value in metrics.items()}, records,
                                   {"finite": True, "production_integrity_evidence": False}, records, request=request)


def _schedule(output, stage, data_reference, input_fingerprint):
    from .generic_role_banks import RoleBank, RoleBankManifest
    from .generic_scheduled_role_banks import (
        ScheduledRoleBankRegistry,
        ScheduledRoleBinding,
    )

    scopes = [("control", number, number * stage.updates_per_round, number * stage.updates_per_round)
              for number in range(stage.round_count + 1)]
    scopes += [("validation", None, 0, stage.stop_update), ("certification", None, 0, stage.stop_update)]
    bindings, references = [], []
    for index, (role, round_number, minimum, maximum) in enumerate(scopes):
        scope_id = f"{role}-{round_number if round_number is not None else 'exit'}"
        record = {"schema": "generic_neural_solver.coverage_evaluation_job.v1", "scope": scope_id,
                  "data": data_reference, "input_fingerprint": input_fingerprint, "measure": MEASURE}
        reference = _write_json(output / "scopes" / f"{scope_id}.json", record)
        references.append(reference)
        bank = RoleBank(role, scope_id, index, "fixed-coverage-regression", input_fingerprint,
            "caller-D-exact-training-loss-v1", "deterministic-training-population-v1",
            {"global": reference["sha256"], "local": data_reference["sha256"]}, (f"evaluation-job/{scope_id}",),
            metadata={"global_archive": reference, "training_data": data_reference, "measure": MEASURE,
                      "global_hash_semantics": "scope descriptor hash, not independent sample bytes"})
        bindings.append(ScheduledRoleBinding(stage.stage_id, ARM_ID, role, round_number, minimum, maximum,
                                            RoleBankManifest(TASK_IDS, (bank,), {role: 1})))
    return ScheduledRoleBankRegistry(tuple(bindings), registry_version="auxiliary-coverage-training-only-v1"), references


def build_runner(output, coordinates, width, data, parameters, denominators, split, learning_rate, updates, source_evidence):
    """Return (runner, adapter, provider); construction performs no objective calls.

    Each call uses at most 50 updates and fresh Adam moments. A multiple of ten
    uses ten-update rounds. Other budgets use gcd(updates, 10) updates per round.
    ``source_evidence`` is one file reference or a mapping containing references.
    Rebuild the same output and arguments for checkpoint recovery, not extension.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("coverage runner requires CPU-only CUDA_VISIBLE_DEVICES=-1")
    parameters, denominators = _validate_inputs(coordinates, width, data, parameters, denominators, split, learning_rate, updates)
    evidence, references = _evidence(source_evidence)
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
    per_round = math.gcd(updates, 10)
    stage = ReplicationStage(STAGE_ID, 0, 1, updates // per_round, 0, updates_per_round=per_round, population_size=1, select_after=True)
    configuration = {"schema": "generic_neural_solver.coverage_runner.v1", "coordinates": coordinates.to_dict(),
                     "width": width, "data_hash": data.binding_hash(), "parameters": parameters.tolist(),
                     "denominators": denominators.tolist(), "split": split, "learning_rate": float(learning_rate),
                     "updates": updates, "stage": asdict(stage), "source_evidence": evidence,
                     "sources": sources, "measure": MEASURE}
    config_reference = _write_json(output / "configuration.json", configuration)
    data_reference = _write_json(output / "coverage-data.json", data.to_dict())
    for relative, digest in sources.items():
        _copy({"path": str(SOURCE_ROOT / relative), "sha256": digest}, output / "source" / relative)
    for index, reference in enumerate(references):
        _copy(reference, output / "evidence" / f"input-{index}.snapshot")
    input_binding = {"configuration": config_reference, "data": data_reference,
                     "data_hash": data.binding_hash(), "profile_hash": coordinates.binding_hash(),
                     "denominators": denominators.tolist(), "derivative_split": split, "measure": MEASURE}
    adapter = CoverageAdapter(coordinates, width, data, denominators, split, sources, input_binding)
    roles, scope_files = _schedule(output, stage, data_reference, adapter.metadata.input_fingerprint)
    provider = CoverageDiagnosticProvider(adapter, roles, stage, scope_files)
    executor = ModelTaskExecutor(adapter.metadata, ExecutionSpec(3, adapter.policy_dimension, adapter.sample_count, 0), roles,
        threshold=THRESHOLD, model_binding=adapter.adapter_identity(), coordinator_factory=ReplicationPermanentPassCoordinator,
        update_factory=make_certified_projected_adam_function, update_binding=numerical_binding(),
        method_factory=make_certified_method_function, method_binding=numerical_binding())
    policy = PolicyView(tuple(parameters.tolist()), "auxiliary-coverage-initialization",
                        {"coordinate_profile_hash": coordinates.binding_hash(), "coverage_input_fingerprint": adapter.metadata.input_fingerprint})
    initial = CheckpointState(STAGE_ID, 0, policy.to_dict(),
        {"first_moment": [0.] * adapter.policy_dimension, "second_moment": [0.] * adapter.policy_dimension,
         "iteration": 0, "learning_rate": float(learning_rate)},
        {"preferred": "cagrad", "rates": dict.fromkeys(METHODS, float(learning_rate)), "gradnorm_reset_per_call": True, "state": None},
        {"pcgrad_seed": 20260722, "next_seed_index": 0}, metadata={"measure": MEASURE, "auxiliary_training_only": True})
    governing_reference = evidence.get("master", evidence.get("plan", references[0]))
    _checked(governing_reference)
    design = ReplicationDesignBinding(STAGE_ID, governing_reference["sha256"], stable_hash(sources), sources,
        environment_binding={"python": sys.version, "prefix": sys.prefix, "cpu_only": True,
            "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}},
        backend_binding={"backend": "tensorflow", "dtype": "float64", "device": "CPU", "method": numerical_binding(), "update": numerical_binding()},
        role_manifest_sha256=roles.binding_hash(), target_hashes={"data": data_reference["sha256"], "configuration": config_reference["sha256"],
            "profile": coordinates.binding_hash(), "evidence": stable_hash(evidence)},
        initial_state_hashes={str(ARM_ID): stable_hash(initial.to_dict())}, task_ids=TASK_IDS, threshold=THRESHOLD, replica_ids=(ARM_ID,),
        expected_selected_replica=None, selection_key=("absolute_threshold_ratio", "replica"), stages=(asdict(stage),),
        optimizer_binding={"adam": asdict(DEFAULT_ADAM), "rate": float(learning_rate), "preferred_method": "cagrad", "fresh_moments": True},
        permanent_pass_binding={"stage_entries": {}, "auxiliary_threshold": THRESHOLD, "measure": MEASURE},
        seed_registry_sha256=stable_hash({"scope_ordinals": list(range(len(scope_files))), "sampler": None}))
    _write_json(output / "design.json", design.to_dict())
    _write_json(output / "schedule.json", roles.to_dict())

    def prepare(arm, proposed_policy, requested_stage, update):
        if arm != ARM_ID or requested_stage != stage or type(update) is not int or not 0 <= update < updates:
            raise ValueError("coverage batch request outside frozen update allowance")
        return adapter.prepare_batch(proposed_policy, update)

    store = AtomicCheckpointStore(output / "checkpoints", initial.attempt_id, design)
    if store.has_initial(ARM_ID):
        store.recover(ARM_ID)
    runner = GenericReplicationRunner(executor, {ARM_ID: adapter}, {ARM_ID: initial}, provider, prepare, store,
                                      (stage,), threshold=THRESHOLD, design=design)
    return runner, adapter, provider


def recover_completed(output):
    """Recover completed auxiliary receipts without evaluating or updating models."""
    from .generic_coverage_initialization import CoverageTrainingData
    from .generic_policy_coordinates import AffinePolicyCoordinates
    from .generic_replication_design import ReplicationDesignBinding

    output = Path(output).resolve()
    design = ReplicationDesignBinding.from_dict(json.loads((output / "design.json").read_text()))
    config = json.loads(_checked({"path": str(output / "configuration.json"),
                                "sha256": design.target_hashes["configuration"]}).read_text())
    data = CoverageTrainingData.from_dict(json.loads(_checked({"path": str(output / "coverage-data.json"),
                                                             "sha256": design.target_hashes["data"]}).read_text()))
    runner, adapter, provider = build_runner(output, AffinePolicyCoordinates.from_dict(config["coordinates"]),
        config["width"], data, config["parameters"], config["denominators"], config["split"],
        config["learning_rate"], config["updates"], config["source_evidence"])
    if not runner.store.has_initial(ARM_ID):
        raise ValueError("completed auxiliary checkpoint required for recovery")
    state, _events = runner.store.recover(ARM_ID)
    if not runner._stage_done(state, runner.stages[0], runner._last_kind(ARM_ID)):
        raise ValueError("auxiliary training transaction is incomplete")

    def forbidden(*_args, **_kwargs):
        raise ValueError("completed coverage recovery refuses model calls and updates")

    adapter.evaluate = adapter.compute_task_values_and_gradients = forbidden
    provider.control = provider.validation = provider.certification = forbidden
    runner.executor.step = runner.batch_factory = forbidden
    return runner.run()
