"""Host-side saved or callback loss rows to bound engineering role results."""

import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .generic_replication_training_protocol import publish, reference, verified_path
from .generic_scheduled_role_banks import ScheduledRoleBankRegistry
from .generic_statistical_quality import evaluate_upper_mean_mse
from .generic_training_contracts import (
    CertificationResult,
    CheckpointState,
    ControlEvaluation,
    ImmutableJSONMapping,
    PolicyView,
    ValidationEvaluation,
    canonical_json,
    stable_hash,
)

ROLES = ("control", "validation", "certification")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _identifiers(values, name):
    require(isinstance(values, (tuple, list)) and bool(values)
            and all(isinstance(value, str) and value for value in values), f"{name} needs nonempty identifiers")
    require(len(set(values)) == len(values), f"duplicate {name}")
    return tuple(values)


def _array(values, shape, name, *, positive=False):
    original = np.asarray(values)
    require(original.dtype.kind in "fiu", f"numeric {name} required")
    array = original.astype(np.float64)
    require(array.shape == shape and np.isfinite(array).all(), f"finite {name} with shape {shape} required")
    require(np.all(array > 0. if positive else array >= 0.), f"{'positive' if positive else 'nonnegative'} {name} required")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True)
class LossCoordinates:
    task_ids: tuple[str, ...]
    training: tuple[float, ...]
    control: tuple[float, ...]
    validation: tuple[float, ...]
    certification: tuple[float, ...]
    threshold: float
    minimum_banks: int = 20
    alpha: float = .05

    def __post_init__(self):
        object.__setattr__(self, "task_ids", _identifiers(self.task_ids, "task IDs"))
        for name in ("training", *ROLES):
            array = _array(getattr(self, name), (len(self.task_ids),), f"{name} denominators", positive=True)
            object.__setattr__(self, name, tuple(array.tolist()))
        require(type(self.threshold) in (float, int) and math.isfinite(self.threshold) and self.threshold > 0.,
                "positive finite common threshold required")
        require(all(math.isfinite(value * self.threshold) for role in ROLES for value in getattr(self, role)),
                "finite raw role limits required")
        require(type(self.minimum_banks) is int and self.minimum_banks >= 2, "at least two independent banks required")
        require(type(self.alpha) in (float, int) and math.isfinite(self.alpha) and 0. < self.alpha < 1., "invalid alpha")

    def to_dict(self):
        return {"task_ids": list(self.task_ids), **{name: list(getattr(self, name)) for name in ("training", *ROLES)},
                "threshold": self.threshold, "minimum_banks": self.minimum_banks, "alpha": self.alpha}

    def binding_hash(self):
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class BankLossRows:
    bank_id: str
    row_ids: tuple[str, ...]
    policy_fingerprint: str
    task_ids: tuple[str, ...]
    raw_mse: object
    empirical_addition_raw: object
    source_references: tuple[dict, ...]

    def __post_init__(self):
        require(isinstance(self.bank_id, str) and bool(self.bank_id), "bank ID required")
        require(isinstance(self.policy_fingerprint, str) and len(self.policy_fingerprint) == 64
                and all(char in "0123456789abcdef" for char in self.policy_fingerprint), "policy fingerprint required")
        object.__setattr__(self, "row_ids", _identifiers(self.row_ids, "inner row IDs"))
        object.__setattr__(self, "task_ids", _identifiers(self.task_ids, "task IDs"))
        shape = len(self.row_ids), len(self.task_ids)
        for name in ("raw_mse", "empirical_addition_raw"):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        require(isinstance(self.source_references, (tuple, list)) and bool(self.source_references), "source references required")
        references = tuple(ImmutableJSONMapping(dict(binding)) for binding in self.source_references)
        require(all(set(binding) == {"path", "sha256"} for binding in references), "path/hash source references required")
        object.__setattr__(self, "source_references", references)

    def binding_hash(self):
        return stable_hash({"bank_id": self.bank_id, "row_ids": list(self.row_ids), "policy": self.policy_fingerprint,
                            "task_ids": list(self.task_ids), "raw_mse": self.raw_mse.tolist(),
                            "empirical_addition_raw": self.empirical_addition_raw.tolist(),
                            "sources": [dict(binding) for binding in self.source_references]})

    def to_dict(self):
        return {"bank_id": self.bank_id, "row_ids": list(self.row_ids), "policy_fingerprint": self.policy_fingerprint,
                "task_ids": list(self.task_ids), "raw_mse": self.raw_mse.tolist(),
                "empirical_addition_raw": self.empirical_addition_raw.tolist(),
                "source_references": [dict(binding) for binding in self.source_references]}


@dataclass(frozen=True)
class GroupedRoleSpec:
    task_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    global_task_ids: tuple[str, ...]
    local_task_ids: tuple[str, ...]
    cell_probabilities: tuple[float, ...]
    validation_mode: str
    bank_families: Mapping
    global_alpha: float
    local_alpha: float
    global_minimum_banks: int
    local_minimum_banks: int
    uncertainty_semantics: str

    def __post_init__(self):
        for name in ("task_ids", "cell_ids", "global_task_ids", "local_task_ids"):
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
        require(not set(self.global_task_ids).intersection(self.local_task_ids)
                and set(self.global_task_ids + self.local_task_ids) == set(self.task_ids), "complete disjoint task partition required")
        for family in ("global", "local"):
            tasks = getattr(self, f"{family}_task_ids")
            require(tuple(task for task in self.task_ids if task in tasks) == tasks, "family task order mismatch")
            alpha, count = getattr(self, f"{family}_alpha"), getattr(self, f"{family}_minimum_banks")
            require(type(alpha) in (float, int) and math.isfinite(alpha) and 0. < alpha < 1., "invalid family alpha")
            require(type(count) is int and count >= 2, "at least two banks per family required")
        probabilities = _array(self.cell_probabilities, (len(self.cell_ids),), "cell probabilities", positive=True)
        require(math.isclose(float(np.sum(probabilities)), 1., rel_tol=0., abs_tol=1e-12), "cell probabilities must sum to one")
        object.__setattr__(self, "cell_probabilities", tuple(probabilities.tolist()))
        require(self.validation_mode in ("stratified", "whole-law"), "explicit validation sampling mode required")
        require(isinstance(self.bank_families, Mapping) and bool(self.bank_families)
                and all(isinstance(name, str) and name and family in ("global", "local")
                        for name, family in self.bank_families.items()), "bound family bank inventory required")
        object.__setattr__(self, "bank_families", ImmutableJSONMapping(dict(self.bank_families)))
        require(isinstance(self.uncertainty_semantics, str) and bool(self.uncertainty_semantics), "uncertainty semantics required")

    def to_dict(self):
        return {"task_ids": list(self.task_ids), "cell_ids": list(self.cell_ids),
                "global_task_ids": list(self.global_task_ids), "local_task_ids": list(self.local_task_ids),
                "cell_probabilities": list(self.cell_probabilities), "validation_mode": self.validation_mode,
                "bank_families": dict(self.bank_families), "global_alpha": self.global_alpha, "local_alpha": self.local_alpha,
                "global_minimum_banks": self.global_minimum_banks, "local_minimum_banks": self.local_minimum_banks,
                "uncertainty_semantics": self.uncertainty_semantics}

    def binding_hash(self):
        return stable_hash(self.to_dict())

    def conjunct(self, family, task, cell=None):
        return canonical_json([family, cell, task])


@dataclass(frozen=True)
class GroupedBankLossRows:
    bank_id: str
    row_ids: tuple[str, ...]
    policy_fingerprint: str
    task_ids: tuple[str, ...]
    raw_mse: object
    conservative_raw_mse: object
    source_references: tuple[dict, ...]
    family: str
    row_cell_ids: tuple[str, ...] = ()

    def __post_init__(self):
        rows = _identifiers(self.row_ids, "inner row IDs")
        tasks = _identifiers(self.task_ids, "task IDs")
        base = BankLossRows(self.bank_id, rows, self.policy_fingerprint, tasks, self.raw_mse,
                            np.zeros((len(rows), len(tasks))), self.source_references)
        for name in ("row_ids", "task_ids", "raw_mse", "source_references"):
            object.__setattr__(self, name, getattr(base, name))
        conservative = _array(self.conservative_raw_mse, base.raw_mse.shape, "conservative raw MSE")
        require(np.all(conservative >= base.raw_mse), "conservative raw MSE below central")
        object.__setattr__(self, "conservative_raw_mse", conservative)
        require(self.family in ("global", "local"), "grouped bank family required")
        require(isinstance(self.row_cell_ids, (list, tuple)) and
                (not self.row_cell_ids or len(self.row_cell_ids) == len(rows)
                 and all(isinstance(cell, str) and cell for cell in self.row_cell_ids)), "row cell identities mismatch")
        require(self.family != "local" or not self.row_cell_ids, "local rows cannot be replicated across cells")
        object.__setattr__(self, "row_cell_ids", tuple(self.row_cell_ids))

    def to_dict(self):
        return {"bank_id": self.bank_id, "row_ids": list(self.row_ids), "policy_fingerprint": self.policy_fingerprint,
                "task_ids": list(self.task_ids), "raw_mse": self.raw_mse.tolist(),
                "conservative_raw_mse": self.conservative_raw_mse.tolist(), "family": self.family,
                "row_cell_ids": list(self.row_cell_ids), "source_references": [dict(value) for value in self.source_references]}

    def binding_hash(self):
        return stable_hash(self.to_dict())


class IndependentBankRoleBridge:
    """Engineering evidence; exposed banks can enter diagnostic control only.

    Dynamic providers freeze inputs and source contracts before seeing policies.
    Receipts bind actual outputs, without certifying independence or truth error.
    """

    def __init__(self, root, registry, coordinates, records=(), *, evidence_scope, row_provider=None,
                 provider_binding=None, cache_directory=None, grouping=None):
        require(isinstance(registry, ScheduledRoleBankRegistry), "scheduled role registry required")
        require(isinstance(coordinates, LossCoordinates) and registry.task_ids == coordinates.task_ids, "ordered task binding mismatch")
        require(evidence_scope in ("manufactured", "exposed-diagnostic", "fresh-engineering", "fresh-finite-benchmark"), "engineering evidence scope required")
        self.root, self.registry, self.coordinates = Path(root).resolve(), registry, coordinates
        require(grouping is None or isinstance(grouping, GroupedRoleSpec) and grouping.task_ids == coordinates.task_ids
                and coordinates.threshold == 1., "grouped roles require matching tasks and threshold one")
        self.grouping = grouping
        self.evidence_scope = evidence_scope
        self.cache_directory = None if cache_directory is None else Path(cache_directory).resolve()
        require(self.cache_directory is None or self.cache_directory.is_relative_to(self.root), "cache must remain inside root")
        self.row_provider = row_provider
        self.provider_binding = None
        if row_provider is not None:
            require(callable(row_provider) and isinstance(provider_binding, Mapping), "bound row provider required")
            require(self.cache_directory is not None, "dynamic row provider requires durable cache")
            self.provider_binding = ImmutableJSONMapping(dict(provider_binding))
            self._verify_provider()
        else:
            require(provider_binding is None and evidence_scope not in ("fresh-engineering", "fresh-finite-benchmark"),
                    "fresh engineering requires bound row provider")
        self.records = {}
        for record in records:
            require(isinstance(record, BankLossRows) and record.task_ids == coordinates.task_ids if grouping is None else
                    isinstance(record, GroupedBankLossRows) and record.task_ids == getattr(grouping, f"{record.family}_task_ids"),
                    "saved bank task order mismatch")
            key = record.policy_fingerprint, record.bank_id
            require(key not in self.records, "duplicate saved policy/bank")
            self.records[key] = record
        require(not self.records if row_provider is not None else bool(self.records),
                "choose dynamic provider without saved records or saved bank records required")
        self._banks, owners = {}, {}
        for binding in registry.bindings:
            for bank in binding.manifest.banks:
                require(bank.sample_ids == (bank.bank_id,), "one bank observation ID required; inner rows are separate")
                require(bank.metadata.get("evidence_scope") == evidence_scope, "bank evidence scope mismatch")
                require(evidence_scope != "exposed-diagnostic" or bank.role == "control", "exposed data are control-only diagnostic")
                require(evidence_scope != "fresh-engineering" or bank.role != "certification", "fresh engineering terminal is not qualified")
                require(bank.scale_version == coordinates.binding_hash(), "bank scale coordinate binding mismatch")
                if grouping is not None:
                    require(bank.metadata.get("grouping_hash") == grouping.binding_hash()
                            and bank.metadata.get("family") == grouping.bank_families.get(bank.bank_id), "bank grouping binding mismatch")
                row_ids = _identifiers(tuple(bank.metadata.get("row_ids", ())), "registered inner row IDs")
                require(bank.metadata.get("row_count") == len(row_ids), "registered row count mismatch")
                if row_provider is not None:
                    self._bank_inputs(bank)
                previous = self._banks.setdefault(bank.bank_id, bank)
                require(previous.binding_hash() == bank.binding_hash(), "bank ID reused with different evidence")
                for row_id in row_ids:
                    owner = owners.setdefault(row_id, bank.bank_id)
                    require(owner == bank.bank_id, "inner rows overlap different banks/roles")
        if grouping is not None:
            require(set(grouping.bank_families) == set(self._banks), "grouping bank inventory mismatch")
        self.estimator_calls, self.cache_hits = 0, 0
        self.provider_calls, self.row_cache_hits = 0, 0

    def _verify_provider(self):
        binding = self.provider_binding
        require({"callback", "source_references", "policy_dimension", "policy_metadata", "coordinates_hash"} <= set(binding),
                "incomplete provider binding")
        require(type(binding["policy_dimension"]) is int and binding["policy_dimension"] > 0
                and isinstance(binding["policy_metadata"], Mapping), "provider policy compatibility required")
        require(binding["coordinates_hash"] == self.coordinates.binding_hash(), "provider coordinate binding mismatch")
        if self.grouping is not None:
            require(binding.get("grouping_hash") == self.grouping.binding_hash(), "provider grouping binding mismatch")
        sources = binding["source_references"]
        require(isinstance(sources, (tuple, list)) and bool(sources), "provider source references required")
        callback = binding["callback"]
        require(isinstance(callback, Mapping) and set(callback) == {"module", "qualname", "source"}, "callback binding required")
        require(callback["source"] in sources, "callback source must be declared")
        implementation = self.row_provider
        if not inspect.isfunction(implementation) and not inspect.ismethod(implementation):
            implementation = implementation.__call__
        filename = inspect.getsourcefile(implementation)
        require(filename is not None and Path(filename).resolve() == verified_path(self.root, dict(callback["source"])).resolve()
                and implementation.__module__ == callback["module"] and implementation.__qualname__ == callback["qualname"],
                "row provider callback source/identity mismatch")
        for source in sources:
            verified_path(self.root, dict(source))

    def _bank_inputs(self, bank):
        require("policy_row_hashes" not in bank.metadata, "dynamic registry cannot bind policy outputs")
        require(bank.metadata.get("provider_binding_hash") == stable_hash(self.provider_binding), "bank provider binding mismatch")
        references = bank.metadata.get("input_references")
        require(isinstance(references, Mapping) and set(references) == set(bank.input_hashes), "bank input references required")
        for name, binding in references.items():
            require(isinstance(binding, Mapping) and binding.get("sha256") == bank.input_hashes[name], "bank input hash mismatch")
            verified_path(self.root, dict(binding))
        return [dict(binding) for binding in references.values()]

    def _row_identity(self, policy, request, bank):
        identity = {"request": request.to_dict(), "policy_fingerprint": policy.fingerprint(), "bank": bank.to_dict(),
                "provider_binding": dict(self.provider_binding), "coordinates": self.coordinates.to_dict(),
                "evidence_scope": self.evidence_scope}
        if self.grouping is not None:
            identity["grouping"] = self.grouping.to_dict()
        return json.loads(canonical_json(identity))

    def _row_directory(self, policy, request, bank):
        return self.cache_directory / "dynamic-rows" / stable_hash(self._row_identity(policy, request, bank))

    def _validate_dynamic_record(self, record, policy, bank):
        require(isinstance(record, BankLossRows if self.grouping is None else GroupedBankLossRows), "row provider type mismatch")
        require(record.bank_id == bank.bank_id and record.policy_fingerprint == policy.fingerprint(), "output policy/bank mismatch")
        expected_tasks = self.coordinates.task_ids if self.grouping is None else getattr(self.grouping, f"{record.family}_task_ids")
        require(record.task_ids == expected_tasks and record.row_ids == tuple(bank.metadata["row_ids"]),
                "output task/row inventory mismatch")
        if self.grouping is not None:
            self._validate_group_record(record, bank)
        expected = self._bank_inputs(bank) + list(self.provider_binding["source_references"])
        available = {binding.canonical for binding in record.source_references}
        require(all(canonical_json(binding) in available for binding in expected), "output omits input/provider source references")
        for binding in record.source_references:
            verified_path(self.root, dict(binding))

    def _dynamic_inputs(self, policy, request):
        self._verify_provider()
        require(len(policy.values) == self.provider_binding["policy_dimension"]
                and all(key in policy.metadata and policy.metadata[key] == value
                        for key, value in self.provider_binding["policy_metadata"].items()), "incompatible provider policy")
        banks = [self._banks[bank_id] for bank_id in request.metadata["bank_ids"]]
        for bank in banks:
            self._bank_inputs(bank)
        records = []
        for bank in banks:
            self._bank_inputs(bank)
            identity = self._row_identity(policy, request, bank)
            directory = self._row_directory(policy, request, bank)
            intent, completed = directory / "intent.json", directory / "result.json"
            if completed.exists():
                receipt = json.loads(completed.read_text())
                digest = receipt.pop("receipt_hash")
                require(stable_hash(receipt) == digest and receipt["identity"] == identity, "changed dynamic row receipt")
                require(intent.exists() and json.loads(intent.read_text()) == identity, "changed dynamic row intent")
                record = (BankLossRows if self.grouping is None else GroupedBankLossRows)(**receipt["rows"])
                require(record.binding_hash() == receipt["rows_hash"], "changed dynamic row output")
                self._validate_dynamic_record(record, policy, bank)
                self.row_cache_hits += 1
            else:
                require(not intent.exists(), "row evaluation intent is incomplete; scoped repair required before retry")
                publish(intent, identity)
                self.provider_calls += 1
                record = self.row_provider(policy, request, bank, output_directory=directory / "provider")
                self._verify_provider()
                self._validate_dynamic_record(record, policy, bank)
                payload = {"identity": identity, "rows": record.to_dict(), "rows_hash": record.binding_hash()}
                publish(completed, {**payload, "receipt_hash": stable_hash(payload)})
            records.append(record)
        return records

    def _inputs(self, policy, request):
        self.registry.validate_request(request, policy)
        require(request.role in ROLES, "unsupported role")
        require(self.evidence_scope != "exposed-diagnostic" or request.role == "control", "exposed data are control-only diagnostic")
        bank_ids = request.metadata["bank_ids"]
        require(request.sample_count == len(bank_ids) and request.anchor_ids == tuple(bank_ids)
                and len(set(bank_ids)) == len(bank_ids), "bank observation count mismatch")
        if self.grouping is None:
            require(len(bank_ids) >= self.coordinates.minimum_banks, "incomplete independent bank inventory")
        else:
            for family in ("global", "local"):
                count = sum(self.grouping.bank_families[bank_id] == family for bank_id in bank_ids)
                require(count >= getattr(self.grouping, f"{family}_minimum_banks"), "incomplete independent family bank inventory")
        if self.row_provider is not None:
            return self._dynamic_inputs(policy, request)
        records, references = [], {}
        for bank_id in bank_ids:
            record = self.records.get((policy.fingerprint(), bank_id))
            require(record is not None, "missing saved policy/bank rows")
            bank = self._banks[bank_id]
            require(record.row_ids == tuple(bank.metadata["row_ids"]), "saved inner row inventory mismatch")
            if self.grouping is not None:
                self._validate_group_record(record, bank)
            require(bank.metadata.get("policy_row_hashes", {}).get(policy.fingerprint()) == record.binding_hash(),
                    "saved policy/loss-row binding mismatch")
            hashes = {binding["sha256"] for binding in record.source_references}
            require(set(bank.input_hashes.values()) <= hashes, "saved rows do not bind the registered bank inputs")
            for binding in record.source_references:
                previous = references.setdefault(binding["path"], dict(binding))
                require(previous == dict(binding), "conflicting saved source references")
            records.append(record)
        for binding in references.values():
            verified_path(self.root, binding)
        return records

    def evaluate(self, policy, request):
        records = self._inputs(policy, request)
        identity = {"request": request.to_dict(), "coordinates": self.coordinates.to_dict(),
                    "bank_rows": [record.binding_hash() for record in records], "evidence_scope": self.evidence_scope}
        if self.grouping is not None:
            identity["grouping"] = self.grouping.to_dict()
        if self.row_provider is not None:
            identity["provider_binding_hash"] = stable_hash(self.provider_binding)
            identity["row_receipts"] = [reference(self.root, self._row_directory(policy, request, self._banks[record.bank_id]) / "result.json")
                                        for record in records]
        cache = None if self.cache_directory is None else self.cache_directory / f"{stable_hash(identity)}.json"
        result_type = {"control": ControlEvaluation, "validation": ValidationEvaluation, "certification": CertificationResult}[request.role]
        if cache is not None and cache.exists():
            receipt = json.loads(cache.read_text())
            digest = receipt.pop("receipt_hash")
            require(stable_hash(receipt) == digest and receipt["identity"] == identity, "changed role cache")
            result = result_type.from_dict(receipt["result"])
            require(result.request.to_dict() == request.to_dict(), "cached role request changed")
            self.cache_hits += 1
            return result
        if self.grouping is not None:
            result = self._grouped_result(records, request, identity)
            if cache is not None:
                payload = {"identity": identity, "result": result.to_dict()}
                publish(cache, {**payload, "receipt_hash": stable_hash(payload)})
            return result
        scales = np.asarray(getattr(self.coordinates, request.role))
        bank_records = []
        for record in records:
            central = np.mean(record.raw_mse / scales, axis=0)
            conservative = np.mean((record.raw_mse + record.empirical_addition_raw) / scales, axis=0)
            bank_records.append({"bank_id": record.bank_id,
                "central_normalized_mse": dict(zip(self.coordinates.task_ids, central.tolist(), strict=True)),
                "conservative_normalized_mse": dict(zip(self.coordinates.task_ids, conservative.tolist(), strict=True))})
        self.estimator_calls += 1
        quality = evaluate_upper_mean_mse(bank_records, anchor_ids=[record.bank_id for record in records],
            task_order=self.coordinates.task_ids, minimum_anchors=self.coordinates.minimum_banks,
            alpha=self.coordinates.alpha, threshold_mse=self.coordinates.threshold)
        require(all(math.isfinite(value) for task in quality["tasks"].values()
                    for value in task.values() if isinstance(value, float)), "nonfinite bank summary")
        quality["threshold_definition"].update(
            mse_matches_training_objective_coordinate=tuple(scales) == self.coordinates.training,
            coordinate=f"raw_mse/{request.role}_denominator", raw_limits=(scales * self.coordinates.threshold).tolist())
        summary = {"quality": quality, "coordinates": self.coordinates.to_dict(), "bank_records": bank_records,
            "bank_count": len(records), "inner_row_count": sum(len(record.row_ids) for record in records),
            "role_bank_manifest_hash": request.metadata["role_bank_manifest_hash"],
            "scheduled_role_registry_hash": self.registry.binding_hash(), "task_ids": list(self.coordinates.task_ids),
            "evidence_scope": self.evidence_scope, "engineering_only": True, "scientific_admission": False,
            "empirical_addition_scope": "empirical allowances; not rigorous error-to-truth or quadrature bounds"}
        if self.row_provider is not None:
            summary.update(provider_binding_hash=identity["provider_binding_hash"], row_receipts=identity["row_receipts"])
        if request.role == "control":
            result = ControlEvaluation({task: quality["tasks"][task]["upper_normalized_mse"] for task in self.coordinates.task_ids},
                                       raw_records=summary, request=request)
        elif request.role == "validation":
            result = ValidationEvaluation({task: quality["tasks"][task]["mean_normalized_mse"] for task in self.coordinates.task_ids},
                                          {task: len(records) for task in self.coordinates.task_ids}, summary, request=request)
        else:
            result = CertificationResult({task: quality["tasks"][task]["passed"] for task in self.coordinates.task_ids},
                                         summary, {"saved_record_integrity": True, "scientific_qualification": False}, summary, request=request)
        if cache is not None:
            payload = {"identity": identity, "result": result.to_dict()}
            publish(cache, {**payload, "receipt_hash": stable_hash(payload)})
        return result

    def _validate_group_record(self, record, bank):
        grouping = self.grouping
        require(record.family == grouping.bank_families[bank.bank_id], "row family differs from registered bank")
        require(record.task_ids == getattr(grouping, f"{record.family}_task_ids"), "grouped row task order mismatch")
        require(record.row_cell_ids == tuple(bank.metadata.get("row_cell_ids", ())), "registered row cell inventory mismatch")
        cells_required = record.family == "global" and (bank.role != "validation" or grouping.validation_mode == "stratified")
        require(set(record.row_cell_ids) == set(grouping.cell_ids) if cells_required else not record.row_cell_ids,
                "complete declared cells or explicit whole-law/local rows required")

    def _grouped_result(self, records, request, identity):
        grouping = self.grouping
        qualities, bank_records, counts, inner_counts = {}, {}, {}, {}
        means, uppers, conjuncts = {}, {}, {}
        all_scales = dict(zip(self.coordinates.task_ids, getattr(self.coordinates, request.role), strict=True))
        for family in ("global", "local"):
            tasks = getattr(grouping, f"{family}_task_ids")
            members = [record for record in records if record.family == family]
            counts[family], inner_counts[family] = len(members), sum(len(record.row_ids) for record in members)
            by_cell = family == "global" and (request.role != "validation" or grouping.validation_mode == "stratified")
            keys = tuple(grouping.conjunct(family, task, cell) for cell in grouping.cell_ids for task in tasks) if by_cell else tasks
            scales = np.array([all_scales[task] for task in tasks])
            bank_records[family] = []
            validation_means = []
            for record in members:
                values = []
                for raw in (record.raw_mse, record.conservative_raw_mse):
                    if record.row_cell_ids:
                        cells = np.stack([np.mean(raw[np.array(record.row_cell_ids) == cell], axis=0)
                                          for cell in grouping.cell_ids])
                        reduced = cells.reshape(-1) if by_cell else np.sum(cells * np.array(grouping.cell_probabilities)[:, None], axis=0)
                    else:
                        reduced = np.mean(raw, axis=0)
                    values.append(reduced / (np.tile(scales, len(grouping.cell_ids)) if by_cell else scales))
                if request.role == "validation":
                    central = np.sum(values[0].reshape(len(grouping.cell_ids), len(tasks))
                                     * np.array(grouping.cell_probabilities)[:, None], axis=0) if by_cell else values[0]
                    validation_means.append(central)
                require(all(np.isfinite(value).all() for value in values), "nonfinite grouped bank means")
                bank_records[family].append({"bank_id": record.bank_id,
                    "central_normalized_mse": dict(zip(keys, values[0].tolist(), strict=True)),
                    "conservative_normalized_mse": dict(zip(keys, values[1].tolist(), strict=True))})
            self.estimator_calls += 1
            quality = evaluate_upper_mean_mse(bank_records[family], anchor_ids=[record.bank_id for record in members],
                task_order=keys, minimum_anchors=getattr(grouping, f"{family}_minimum_banks"),
                alpha=getattr(grouping, f"{family}_alpha"), threshold_mse=self.coordinates.threshold)
            require(all(math.isfinite(value) for task in quality["tasks"].values()
                        for value in task.values() if isinstance(value, float)), "nonfinite grouped bank summary")
            quality["threshold_definition"].update(
                mse_matches_training_objective_coordinate=all(all_scales[task] == self.coordinates.training[self.coordinates.task_ids.index(task)] for task in tasks),
                coordinate=f"raw_mse/{request.role}_denominator", raw_limits={task: all_scales[task] for task in tasks})
            quality.update(family=family, degrees_of_freedom=len(members) - 1, family_alpha=getattr(grouping, f"{family}_alpha"))
            qualities[family] = quality
            for task in tasks:
                task_keys = [grouping.conjunct(family, task, cell) for cell in grouping.cell_ids] if by_cell else [task]
                uppers[task] = max(quality["tasks"][key]["upper_normalized_mse"] for key in task_keys)
                if request.role == "validation":
                    means[task] = float(np.mean(validation_means, axis=0)[tasks.index(task)])
                for key in task_keys:
                    conjuncts[key if by_cell else grouping.conjunct(family, task)] = quality["tasks"][key]["passed"]
        summary = {"quality": qualities, "coordinates": self.coordinates.to_dict(), "grouping": grouping.to_dict(),
            "bank_records": bank_records, "bank_count": counts, "inner_row_count": inner_counts,
            "family_union_alpha_bound": min(1., grouping.global_alpha + grouping.local_alpha),
            "role_bank_manifest_hash": request.metadata["role_bank_manifest_hash"],
            "scheduled_role_registry_hash": self.registry.binding_hash(), "task_ids": list(self.coordinates.task_ids),
            "evidence_scope": self.evidence_scope, "engineering_only": True, "scientific_admission": False,
            "uncertainty_semantics": grouping.uncertainty_semantics}
        if request.role == "validation":
            summary["validation_reduction"] = "central whole-law mean within each bank, then equal bank means"
        if self.row_provider is not None:
            summary.update(provider_binding_hash=identity["provider_binding_hash"], row_receipts=identity["row_receipts"])
        if request.role == "control":
            return ControlEvaluation({task: uppers[task] for task in self.coordinates.task_ids}, raw_records=summary, request=request)
        if request.role == "validation":
            return ValidationEvaluation({task: means[task] for task in self.coordinates.task_ids},
                {task: counts["global" if task in grouping.global_task_ids else "local"] for task in self.coordinates.task_ids},
                summary, request=request)
        return CertificationResult(conjuncts, summary, {"saved_record_integrity": True, "scientific_qualification": False}, summary, request=request)

    def evaluate_campaign(self, envelope):
        """Adapt a verified scheduled result to the campaign's additional request binding."""
        require(self.evidence_scope in ("manufactured", "fresh-engineering", "fresh-finite-benchmark"),
                "actual exposed rows cannot enter campaign validation/terminal")
        payload = dict(envelope)
        digest = payload.pop("request_hash")
        require(stable_hash(payload) == digest, "campaign request changed")
        phase = envelope["phase"]
        require(phase in ("validation", "terminal"), "campaign evaluation phase required")
        require(self.evidence_scope != "fresh-engineering" or phase != "terminal", "fresh engineering terminal is not qualified")
        role = "validation" if phase == "validation" else "certification"
        declaration = json.loads(verified_path(self.root, envelope["role"]).read_text())
        require(declaration["role"] == phase and declaration["evidence_scope"] == "engineering", "campaign role declaration mismatch")
        require(declaration["scheduled_role_registry_hash"] == self.registry.binding_hash(), "campaign registry changed")
        checkpoint = CheckpointState.from_dict(json.loads(verified_path(self.root, envelope["checkpoint"]).read_text()))
        policy = PolicyView.from_dict(checkpoint.policy_state)
        require(policy.fingerprint() == envelope["policy_fingerprint"] and tuple(envelope["task_ids"]) == self.coordinates.task_ids,
                "campaign frozen policy/tasks mismatch")
        configuration = json.loads(verified_path(self.root, envelope["candidate"]["configuration"]).read_text())
        require(configuration["design"]["threshold"] == self.coordinates.threshold
                and tuple(configuration["design"]["task_ids"]) == self.coordinates.task_ids,
                "campaign selection threshold/task coordinates mismatch")
        if phase == "terminal":
            selection = json.loads(verified_path(self.root, envelope["selection"]).read_text())["selected"]
            require(selection["replica"] == envelope["candidate"]["replica"] and selection["checkpoint"] == envelope["checkpoint"]
                    and selection["configuration"] == envelope["candidate"]["configuration"], "terminal requires selected frozen candidate")
        request = self.registry.request_for(role, policy, stage_id=declaration["stage_id"],
            arm_id=envelope["candidate"]["replica"], round_number=None, update_index=checkpoint.update_index)
        require(request.anchor_ids == tuple(declaration["sample_ids"]), "campaign bank inventory mismatch")
        result = self.evaluate(policy, request)
        metadata = {"campaign_request_hash": digest, "scheduled_request": request.to_dict()}
        return replace(result, request=replace(request, metadata=metadata))


__all__ = ["BankLossRows", "GroupedBankLossRows", "GroupedRoleSpec", "IndependentBankRoleBridge", "LossCoordinates"]
