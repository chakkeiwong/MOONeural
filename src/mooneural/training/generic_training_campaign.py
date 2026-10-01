"""Engineering campaign accounting over the existing replication runner.

``run_campaign(root, manifest, runner_factory=..., validation_evaluator=...,
terminal_evaluator=...)`` uses repository-relative references and output paths.
The factory receives a detached candidate and its work directory. Evaluators
receive a detached frozen request and return ValidationEvaluation or
CertificationResult. An optional source-bound rejection_classifier(error,
candidate) returns a declared rejection reason or None for a harness failure.
An optional manifest.incumbent binds configuration/checkpoint references and a
policy_fingerprint under a distinct evaluation replica ID. It shares candidate
validation banks without entering the runner factory. Only a strictly lower
candidate minimax score nominates; ties retain the incumbent without terminal.
All evidence remains engineering-only; scientific qualification is not provided.
"""

import fcntl
import hashlib
import inspect
import json
import logging
import math
import traceback
from pathlib import Path

from .generic_replication_design import ReplicationDesignBinding
from .generic_replication_training_protocol import publish, reference, verified_path
from .generic_training_contracts import (
    CertificationResult,
    CheckpointState,
    ValidationEvaluation,
    canonical_json,
    stable_hash,
)

SCHEMA = "generic_training_campaign.engineering.v1"
ROLES = ("screening", "validation", "terminal")
INTERFACES = ("runner_factory", "validation_evaluator", "terminal_evaluator")
logger = logging.getLogger(__name__)


class CampaignError(ValueError):
    """Campaign evidence is missing or inconsistent."""


class CampaignRepairRequired(CampaignError):
    """A reserved evaluation may have run; repair its evidence before proceeding."""

    status = "REPAIR_REQUIRED"

    def __init__(self, phase, intent):
        super().__init__(f"{phase} intent has no receipt; ambiguous consumption, no automatic replay: {intent}")
        self.phase = phase
        self.intent = intent


class CandidateRejected(Exception):
    """An explicitly classified candidate failure, optionally with a checkpoint."""

    def __init__(self, reason, checkpoint=None):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("candidate rejection needs a reason")
        if checkpoint is not None and not isinstance(checkpoint, CheckpointState):
            raise TypeError("candidate rejection checkpoint must be typed")
        super().__init__(reason)
        self.checkpoint = checkpoint


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def _read(path):
    return json.loads(path.read_text())


def _receipt(path, payload=None):
    if payload is not None:
        payload = {**payload, "engineering_only": True}
        publish(path, {**payload, "receipt_hash": stable_hash(payload)})
    result = _read(path)
    digest = result.pop("receipt_hash")
    require(stable_hash(result) == digest, f"changed campaign receipt: {path}")
    require(result["engineering_only"] is True, "campaign receipts must remain engineering-only")
    return result


def _callable(root, function, binding):
    frozen = verified_path(root, binding["source"])
    source = inspect.getsourcefile(function)
    require(callable(function) and source is not None and function.__qualname__ == binding["qualname"],
            "callback identity mismatch")
    require(hashlib.sha256(Path(source).read_bytes()).hexdigest() == hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "callback source changed")


def _prepare(root, manifest, interfaces):
    require(set(manifest) - {"incumbent"} == {"schema", "campaign_id", "output", "task_ids", "threshold", "candidates", "roles", "interfaces"}
            and manifest["schema"] == SCHEMA, "invalid campaign manifest")
    require(isinstance(manifest["campaign_id"], str) and bool(manifest["campaign_id"]), "campaign ID required")
    tasks = manifest["task_ids"]
    require(isinstance(tasks, list) and bool(tasks) and all(isinstance(task, str) and task for task in tasks)
            and len(set(tasks)) == len(tasks), "unique task IDs required")
    threshold = manifest["threshold"]
    require(type(threshold) in (int, float) and math.isfinite(threshold) and threshold > 0., "invalid selection threshold")
    candidates = manifest["candidates"]
    require(isinstance(candidates, list) and bool(candidates), "declared candidate roster required")
    replicas = [candidate["replica"] for candidate in candidates]
    require(all(type(replica) is int and replica >= 0 for replica in replicas)
            and replicas == sorted(set(replicas)), "candidate IDs must be sorted and unique")
    for candidate in candidates:
        require(set(candidate) == {"replica", "configuration"}, "candidate ID/configuration required")
        configuration = _read(verified_path(root, candidate["configuration"]))
        design = ReplicationDesignBinding.from_dict(configuration["design"])
        require(design.replica_ids == (candidate["replica"],) and design.expected_selected_replica is None
                and design.task_ids == tuple(tasks) and design.threshold == threshold,
                "candidate design must bind one worker and the declared task/selection coordinates")
    if "incumbent" in manifest:
        incumbent = manifest["incumbent"]
        require(isinstance(incumbent, dict)
                and set(incumbent) == {"replica", "configuration", "checkpoint", "policy_fingerprint"},
                "incumbent ID/configuration/checkpoint/fingerprint required")
        require(type(incumbent["replica"]) is int and incumbent["replica"] >= 0
                and incumbent["replica"] not in replicas, "incumbent evaluation ID must differ from candidate IDs")
        configuration = _read(verified_path(root, incumbent["configuration"]))
        design = ReplicationDesignBinding.from_dict(configuration["design"])
        require(design.task_ids == tuple(tasks) and design.threshold == threshold,
                "incumbent task/selection coordinates differ")
        state = CheckpointState.from_dict(_read(verified_path(root, incumbent["checkpoint"])))
        design.validate_checkpoint(state)
        require(state.policy_fingerprint == incumbent["policy_fingerprint"], "incumbent checkpoint fingerprint differs")
    require(set(manifest["interfaces"]) == set(interfaces), "callback bindings required")
    for name in interfaces:
        _callable(root, interfaces[name], manifest["interfaces"][name])
    require(set(manifest["roles"]) == set(ROLES), "separate screening/validation/terminal roles required")
    seen, roles = set(), {}
    for role in ROLES:
        data = _read(verified_path(root, manifest["roles"][role]))
        samples = data["sample_ids"]
        require(data["role"] == role and data["evidence_scope"] == "engineering"
                and isinstance(samples, list) and bool(samples)
                and all(isinstance(sample, str) and sample for sample in samples)
                and len(set(samples)) == len(samples) and not seen.intersection(samples),
                "role inventories must be distinct engineering evidence")
        seen.update(samples)
        roles[role] = data
    relative = Path(manifest["output"])
    require(not relative.is_absolute() and ".." not in relative.parts and bool(relative.parts), "relative campaign output required")
    output = root / relative
    require(output.resolve().is_relative_to(root), "campaign output escaped root")
    output.mkdir(parents=True, exist_ok=True)
    publish(output / "manifest.json", {"manifest": manifest,
        "engineering_only": True, "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return output, roles


def _samples(payload):
    samples = set()
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name in ("sample_ids", "anchor_ids"):
                samples.update(value)
            else:
                samples.update(_samples(value))
    elif isinstance(payload, (tuple, list)):
        for value in payload:
            samples.update(_samples(value))
    return samples


def _check_screening(samples, role):
    require(samples <= set(role["sample_ids"]), "actual screening requests/registry exceed declared screening inventory")


def _failure(root, directory, identity, error, runner, classifier):
    failure = {"error_type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()}
    event = getattr(error, "event", None)
    if event is not None:
        try:
            failure["event"] = json.loads(canonical_json(event))
        except (TypeError, ValueError):
            failure["event_repr"] = repr(event)
    outcome = {**identity, "status": "HARNESS_FAILURE", **failure}
    try:
        reason = str(error) if isinstance(error, CandidateRejected) else (
            None if classifier is None else classifier(error, json.loads(canonical_json(identity["candidate"])))
        )
        require(reason is None or isinstance(reason, str) and bool(reason.strip()), "classifier must return a reason or None")
        state = getattr(error, "checkpoint", None)
        if isinstance(state, CheckpointState):
            publish(directory / "failure-checkpoint.json", state.to_dict())
            outcome["failure_checkpoint"] = reference(root, directory / "failure-checkpoint.json")
        if runner is not None and runner.store.has_initial(identity["candidate"]["replica"]):
            state, _ = runner.store.recover(identity["candidate"]["replica"])
        if isinstance(state, CheckpointState):
            publish(directory / "rejected-checkpoint.json", state.to_dict())
            outcome["checkpoint"] = reference(root, directory / "rejected-checkpoint.json")
            outcome["policy_fingerprint"] = state.policy_fingerprint
        if reason is not None:
            outcome.update(status="CANDIDATE_REJECTED", rejection_reason=reason)
    except Exception as classification_error:
        logger.exception("Campaign failure classification or checkpoint recovery failed")
        outcome["classification_or_recovery_error"] = {
            "error_type": type(classification_error).__name__, "reason": str(classification_error),
            "traceback": traceback.format_exc(),
        }
    return outcome


def _candidate(root, output, candidate, manifest_hash, factory, classifier, screening_role):
    from .generic_replication_runner import GenericReplicationRunner

    directory = output / f"candidate-{candidate['replica']}"
    directory.mkdir(exist_ok=True)
    path = directory / "outcome.json"
    identity = {"manifest_hash": manifest_hash, "candidate": candidate, "phase": "screening"}
    if path.exists():
        outcome = _receipt(path)
        require(all(outcome[key] == value for key, value in identity.items()), "candidate receipt binding mismatch")
        require(outcome["status"] in ("ELIGIBLE", "CANDIDATE_REJECTED", "HARNESS_FAILURE"), "invalid candidate outcome")
        if outcome["status"] == "ELIGIBLE":
            screening = _receipt(verified_path(root, outcome["screening"]))
            _check_screening(set(screening["observed_sample_ids"]), screening_role)
        for name in ("checkpoint", "failure_checkpoint"):
            if name not in outcome:
                continue
            state = CheckpointState.from_dict(_read(verified_path(root, outcome[name])))
            if name == "checkpoint":
                require(state.policy_fingerprint == outcome["policy_fingerprint"], "candidate checkpoint changed")
        return outcome
    runner, classify = None, False
    try:
        supplied = factory(json.loads(canonical_json(candidate)), directory / "runner")
        require(isinstance(supplied, GenericReplicationRunner), "factory must return the shared GenericReplicationRunner")
        runner = supplied
        configuration = _read(verified_path(root, candidate["configuration"]))
        require(runner.design is not None and runner.design.to_dict() == configuration["design"], "runner configuration differs")
        require(runner.store.root.resolve().is_relative_to(directory.resolve()), "candidate store must be inside its work directory")
        registry = runner.executor.core.coordinator.roles
        require(callable(getattr(registry, "to_dict", None)), "screening registry must expose its bound sample inventory")
        samples = _samples(json.loads(canonical_json(registry.to_dict())))
        require(bool(samples), "screening registry must expose a nonempty bound sample inventory")
        _check_screening(samples, screening_role)
        classify = True
        result = runner.run()
        classify = False
        verified_path(root, candidate["configuration"])
        replica = candidate["replica"]
        require(set(result.states) == {replica} and result.selection["selected_replica"] == replica, "runner returned another candidate")
        state = result.states[replica]
        samples.update(_samples(list(result.evaluations)))
        samples.update(_samples(state.to_dict().get("metadata", {}).get("permanent_pass_rotation", {})))
        _check_screening(samples, screening_role)
        publish(directory / "checkpoint.json", state.to_dict())
        _receipt(directory / "screening.json", {"evidence_scope": "diagnostic screening only",
            "selection": result.selection, "evaluations": list(result.evaluations), "events": list(result.events),
            "observed_sample_ids": sorted(samples),
            "runner_design_hash": runner.design.binding_hash(), "policy_fingerprint": state.policy_fingerprint})
        outcome = {**identity, "status": "ELIGIBLE", "checkpoint": reference(root, directory / "checkpoint.json"),
            "policy_fingerprint": state.policy_fingerprint, "screening": reference(root, directory / "screening.json")}
    except InterruptedError:
        raise
    except Exception as error:
        outcome = _failure(root, directory, identity, error, runner, classifier if classify else None)
        if outcome["status"] == "HARNESS_FAILURE":
            logger.exception("Candidate harness failure")
    return _receipt(path, outcome)


def _evaluation(root, output, manifest, roles, candidate, phase, evaluator, selection=None):
    path = output / (f"validation-{candidate['candidate']['replica']}.json" if phase == "validation" else "terminal.json")
    request = {"manifest_hash": stable_hash(manifest), "phase": phase, "candidate": candidate["candidate"],
        "checkpoint": candidate["checkpoint"], "policy_fingerprint": candidate["policy_fingerprint"],
        "task_ids": manifest["task_ids"], "role": manifest["roles"][phase], "selection": selection,
        "engineering_only": True}
    request["request_hash"] = stable_hash(request)
    intent = path.with_name(path.stem + "-intent.json")
    if path.exists():
        require(intent.exists(), "evaluation receipt lacks its intent")
        require(_receipt(intent) == request, "evaluation intent changed")
        receipt = _receipt(path)
        require(receipt["request"] == request, "evaluation receipt binding mismatch")
    else:
        if intent.exists():
            require(_receipt(intent) == request, "evaluation intent changed")
            raise CampaignRepairRequired(phase, reference(root, intent))
        _receipt(intent, request)
        evaluated = evaluator(json.loads(canonical_json(request)))
        expected_type = ValidationEvaluation if phase == "validation" else CertificationResult
        require(isinstance(evaluated, expected_type), "wrong evaluation type")
        receipt = {"request": request, "evaluation": evaluated.to_dict(), "evidence_scope": "engineering"}
        _check_evaluation(receipt, manifest, roles[phase], phase)
        _receipt(path, receipt)
    for binding in (candidate["candidate"]["configuration"], candidate["checkpoint"], manifest["roles"][phase], selection):
        if binding is not None:
            verified_path(root, binding)
    _check_evaluation(receipt, manifest, roles[phase], phase)
    return receipt, reference(root, path)


def _check_evaluation(receipt, manifest, role, phase):
    from .generic_replication_runner import validation_score

    result_type = ValidationEvaluation if phase == "validation" else CertificationResult
    evaluated = result_type.from_dict(receipt["evaluation"])
    request = evaluated.request
    require(request is not None and request.role == ("validation" if phase == "validation" else "certification")
            and request.policy_fingerprint == receipt["request"]["policy_fingerprint"]
            and request.task_ids == tuple(manifest["task_ids"])
            and request.anchor_ids == tuple(role["sample_ids"]) and request.sample_count == len(role["sample_ids"])
            and request.metadata.get("campaign_request_hash") == receipt["request"]["request_hash"],
            "evaluation must bind the frozen candidate and reserved campaign role")
    require(receipt["evidence_scope"] == "engineering", "scientific evidence is outside this wrapper slice")
    if phase == "validation":
        require(set(evaluated.task_mean_mse) == set(manifest["task_ids"])
                and all(value >= 0. for value in evaluated.task_mean_mse.values()), "nonnegative validation task means required")
        require(math.isfinite(validation_score(evaluated, tuple(manifest["task_ids"]), manifest["threshold"])),
                "finite validation score required")
    else:
        require(set(evaluated.conjuncts) == set(manifest["task_ids"]) and bool(evaluated.upper_records),
                "complete terminal conjuncts and evidence required")


def run_campaign(root, manifest, *, runner_factory, validation_evaluator, terminal_evaluator, rejection_classifier=None):
    """Run/recover a fixed engineering campaign; never promote scientific evidence."""
    from .generic_replication_runner import select_validation_record, validation_score

    root = Path(root).resolve()
    manifest = json.loads(canonical_json(manifest))
    interfaces = dict(zip(INTERFACES, (runner_factory, validation_evaluator, terminal_evaluator), strict=True))
    if rejection_classifier is not None:
        interfaces["rejection_classifier"] = rejection_classifier
    output, roles = _prepare(root, manifest, interfaces)
    with (output / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        outcomes = [_candidate(root, output, candidate, stable_hash(manifest), runner_factory, rejection_classifier, roles["screening"])
                    for candidate in manifest["candidates"]]
        union = {"manifest_hash": stable_hash(manifest), "candidates": outcomes}
        _receipt(output / "candidate-union.json", union)
        result = {"manifest_hash": stable_hash(manifest), "candidate_union": reference(root, output / "candidate-union.json"),
            "scientific_admission": False, "scientific_phase_closed": False, "evidence_scope": "engineering",
            "selection": None, "terminal": None}
        if any(outcome["status"] == "HARNESS_FAILURE" for outcome in outcomes):
            result.update(status="HARNESS_BLOCKED", reason="candidate harness failure retained; selection not performed")
        elif not any(outcome["status"] == "ELIGIBLE" for outcome in outcomes):
            result.update(status="ENGINEERING_COMPLETE", decision="ALL_CANDIDATES_REJECTED")
        else:
            records = []
            for candidate in outcomes:
                if candidate["status"] != "ELIGIBLE":
                    continue
                receipt, binding = _evaluation(root, output, manifest, roles, candidate, "validation", validation_evaluator)
                validation = ValidationEvaluation.from_dict(receipt["evaluation"])
                records.append({"replica": candidate["candidate"]["replica"], "policy_fingerprint": candidate["policy_fingerprint"],
                    "checkpoint": candidate["checkpoint"], "configuration": candidate["candidate"]["configuration"],
                    "absolute_threshold_ratio": validation_score(validation, tuple(manifest["task_ids"]), manifest["threshold"]),
                    "validation": binding})
            selected = select_validation_record(records)
            selection = {"manifest_hash": stable_hash(manifest), "candidate_union": result["candidate_union"],
                "rule": "minimum validation maximum normalized MSE threshold ratio, then replica",
                "records": records, "selected": selected}
            if "incumbent" in manifest:
                incumbent = manifest["incumbent"]
                envelope = {"candidate": {"replica": incumbent["replica"], "configuration": incumbent["configuration"]},
                    "checkpoint": incumbent["checkpoint"], "policy_fingerprint": incumbent["policy_fingerprint"]}
                receipt, binding = _evaluation(root, output, manifest, roles, envelope, "validation", validation_evaluator)
                validation = ValidationEvaluation.from_dict(receipt["evaluation"])
                record = {**incumbent, "absolute_threshold_ratio": validation_score(
                    validation, tuple(manifest["task_ids"]), manifest["threshold"]), "validation": binding}
                nominated = selected["absolute_threshold_ratio"] < record["absolute_threshold_ratio"]
                selection.update(incumbent=record, nominated=nominated,
                    rule="minimum candidate validation maximum normalized MSE threshold ratio, then replica; "
                         "strictly lower than incumbent, otherwise retain incumbent",
                    selected=selected if nominated else record)
            _receipt(output / "selection.json", selection)
            result["selection"] = reference(root, output / "selection.json")
            if "incumbent" in manifest and not selection["nominated"]:
                result.update(status="ENGINEERING_COMPLETE", decision="INCUMBENT_RETAINED")
                return _receipt(output / "result.json", result)
            winner = next(candidate for candidate in outcomes if candidate["candidate"]["replica"] == selected["replica"])
            receipt, result["terminal"] = _evaluation(root, output, manifest, roles, winner, "terminal",
                                                      terminal_evaluator, result["selection"])
            result.update(status="ENGINEERING_COMPLETE", decision="TERMINAL_REPORTED_PASS" if
                CertificationResult.from_dict(receipt["evaluation"]).passed else "TERMINAL_REPORTED_REJECTION")
        return _receipt(output / "result.json", result)
