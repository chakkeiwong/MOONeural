"""Engineering campaign accounting through the real runner and fake executor."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from tests.contracts.test_generic_population_selection import TrackedProvider
from tests.contracts.test_generic_replication_runner import (
    TASK_IDS,
    FakeExecutor,
    checkpoint,
    design_for,
)

from mooneural.training import generic_training_campaign as campaign
from mooneural.training.generic_permanent_pass_executor import PermanentPassUpdateError
from mooneural.training.generic_replication_design import ReplicationDesignBinding
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    GenericReplicationRunner,
    ReplicationStage,
)
from mooneural.training.generic_replication_training_protocol import (
    publish,
    reference,
    verified_path,
)
from mooneural.training.generic_training_contracts import (
    CertificationResult,
    CheckpointState,
    EvaluationRequest,
    ValidationEvaluation,
    stable_hash,
)


def read(path):
    return json.loads(path.read_text())


class CampaignFixture:
    def __init__(self, root):
        self.root = root
        self.factory_calls, self.validation_calls, self.terminal_calls = [], [], []
        self.runners = []
        self.reject, self.harness, self.interrupt, self.factory_reject = set(), set(), set(), set()
        self.initial_reject, self.corrupt_initial = set(), set()
        self.mutate_factory, self.classifier_failure = False, False
        self.screening_leak, self.registry_leak = None, False
        self.missing_registry, self.empty_registry = False, False
        self.bad_binding, self.interrupt_evaluation = None, None
        self.terminal_pass = False
        self.validation_values = {2: 0.02, 7: 0.01}
        self.manifest = {
            "schema": campaign.SCHEMA, "campaign_id": "fake-two-candidate", "output": "campaign",
            "task_ids": list(TASK_IDS), "threshold": 0.04, "candidates": [], "roles": {}, "interfaces": {},
        }
        stage = ReplicationStage("population", 0, 1, 1, 0, updates_per_round=2, population_size=1, select_after=True)
        template = design_for({replica: checkpoint(replica) for replica in range(5)})
        for replica in (2, 7):
            design = replace(template, replica_ids=(replica,), expected_selected_replica=None,
                             initial_state_hashes={str(replica): stable_hash(checkpoint(replica).to_dict())},
                             stages=(asdict(stage),))
            path = root / f"configuration-{replica}.json"
            publish(path, {"design": design.to_dict()})
            self.manifest["candidates"].append({"replica": replica, "configuration": reference(root, path)})
        for role in campaign.ROLES:
            samples = ([f"{kind}-{replica}" for replica in (2, 7) for kind in ("validation", "certification")]
                       if role == "screening" else [f"campaign-{role}"])
            path = root / f"{role}.json"
            publish(path, {"role": role, "evidence_scope": "engineering", "sample_ids": samples})
            self.manifest["roles"][role] = reference(root, path)
        source = root / "fixture-source.py"
        source.write_bytes(Path(__file__).read_bytes())
        for name in (*campaign.INTERFACES, "rejection_classifier"):
            self.manifest["interfaces"][name] = {"source": reference(root, source), "qualname": getattr(self, name).__qualname__}

    def runner_factory(self, candidate, directory):
        replica = candidate["replica"]
        self.factory_calls.append(replica)
        if replica in self.factory_reject:
            raise campaign.CandidateRejected("manufactured initial veto", checkpoint(replica))
        design = ReplicationDesignBinding.from_dict(read(verified_path(self.root, candidate["configuration"]))["design"])
        executor = FakeExecutor(role_hash=design.role_manifest_sha256)
        original_step = executor.step

        def step(state, batch, adapter):
            if state.update_index == 1:
                if replica in self.interrupt:
                    self.interrupt.remove(replica)
                    raise InterruptedError("manufactured process interruption")
                if replica in self.reject or replica in self.harness:
                    if replica in self.corrupt_initial:
                        (directory / f"arms/arm-{replica}/initial/state.complete.json").write_text("{}")
                    cause = "fixture domain violation" if replica in self.reject else "fixture harness fault"
                    raise PermanentPassUpdateError(cause, state, {"event": "update", "committed": False, "error": cause})
            return original_step(state, batch, adapter)

        executor.step = step
        if not self.missing_registry:
            samples = [] if self.empty_registry else [f"validation-{replica}", f"certification-{replica}"]
            executor.core.coordinator.roles.to_dict = lambda: {"banks": [{"sample_ids": samples}]}
        if self.registry_leak:
            executor.core.coordinator.roles.to_dict = lambda: {"banks": [{"sample_ids": ["campaign-terminal"]}]}
        provider = TrackedProvider(selected=2)
        if replica in self.initial_reject:
            def rejected_control(*args, **kwargs):
                raise ValueError("fixture initial domain violation")

            provider.control = rejected_control
        if self.screening_leak:
            certification = provider.certification

            def leaked_certification(*args, **kwargs):
                value = certification(*args, **kwargs)
                return replace(value, request=replace(value.request, anchor_ids=(self.screening_leak,)))

            provider.certification = leaked_certification
        runner = GenericReplicationRunner(
            executor, {replica: object()}, {replica: checkpoint(replica)}, provider, Mock(return_value=object()),
            AtomicCheckpointStore(directory, "runner-test", design),
            tuple(ReplicationStage(**stage) for stage in design.stages), design=design,
        )
        self.runners.append(runner)
        if self.mutate_factory:
            candidate["replica"] = 999
            candidate["configuration"]["sha256"] = "0" * 64
        return runner

    def rejection_classifier(self, error, candidate):
        if self.classifier_failure:
            raise RuntimeError("manufactured classifier fault")
        if type(error) is ValueError and str(error) == "fixture initial domain violation":
            return "declared initial domain failure"
        if (type(error) is PermanentPassUpdateError and str(error) == "fixture domain violation"
                and error.event == {"event": "update", "committed": False, "error": str(error)}):
            return f"declared domain failure for candidate {candidate['replica']}"
        return None

    def evaluation_request(self, request):
        role = read(verified_path(self.root, request["role"]))
        return EvaluationRequest(
            "validation" if request["phase"] == "validation" else "certification", "fixture-campaign-target",
            seeds=(31 if request["phase"] == "validation" else 41,),
            task_ids=tuple(request["task_ids"]), policy_fingerprint=request["policy_fingerprint"],
            anchor_ids=tuple(role["sample_ids"]), sample_count=len(role["sample_ids"]),
            metadata={"campaign_request_hash": request["request_hash"]},
        )

    def validation_evaluator(self, request):
        replica = request["candidate"]["replica"]
        self.validation_calls.append(replica)
        if self.interrupt_evaluation == "validation":
            raise InterruptedError("manufactured validation interruption")
        bound = self.evaluation_request(request)
        if self.bad_binding == "policy":
            bound = replace(bound, policy_fingerprint=stable_hash("different-policy"))
        elif self.bad_binding == "role":
            bound = replace(bound, anchor_ids=("campaign-terminal",))
        elif self.bad_binding == "count":
            bound = replace(bound, sample_count=2)
        elif self.bad_binding == "hash":
            bound = replace(bound, metadata={"campaign_request_hash": "0" * 64})
        return ValidationEvaluation({task: self.validation_values[replica] for task in TASK_IDS},
                                    {task: 1 for task in TASK_IDS}, request=bound)

    def terminal_evaluator(self, request):
        self.terminal_calls.append(request["candidate"]["replica"])
        assert read(verified_path(self.root, request["selection"]))["selected"]["checkpoint"] == request["checkpoint"]
        frozen = CheckpointState.from_dict(read(verified_path(self.root, request["checkpoint"])))
        assert frozen.policy_fingerprint == request["policy_fingerprint"]
        if self.interrupt_evaluation == "terminal":
            raise InterruptedError("manufactured terminal interruption")
        return CertificationResult(
            {task: self.terminal_pass for task in TASK_IDS},
            {task: {"upper_normalized_mse": 0.02 if self.terminal_pass else 0.08} for task in TASK_IDS},
            {"fixture_integrity": True}, request=self.evaluation_request(request),
        )

    def run(self):
        return campaign.run_campaign(self.root, self.manifest, **{
            name: getattr(self, name) for name in self.manifest["interfaces"]
        })

    def union(self):
        return read(self.root / "campaign/candidate-union.json")["candidates"]


@pytest.fixture
def case(tmp_path):
    return CampaignFixture(tmp_path)


@pytest.mark.parametrize("terminal_pass", (False, True))
def test_validation_only_selection_terminal_once_and_completed_recovery(case, terminal_pass):
    case.terminal_pass = terminal_pass
    result = case.run()
    selection = read(verified_path(case.root, result["selection"]))
    assert selection["selected"]["replica"] == 7
    assert [record["replica"] for record in selection["records"]] == [2, 7]
    assert case.validation_calls == [2, 7] and case.terminal_calls == [7]
    assert [runner.evaluation_provider.certifications for runner in case.runners] == [[(2, "population")], [(7, "population")]]
    for candidate in case.union():
        screening = read(verified_path(case.root, candidate["screening"]))
        assert screening["evidence_scope"] == "diagnostic screening only"
        assert all(not item["result"]["passed"] for item in screening["evaluations"] if item["role"] == "certification")
    assert result["status"] == "ENGINEERING_COMPLETE"
    assert result["decision"] == ("TERMINAL_REPORTED_PASS" if terminal_pass else "TERMINAL_REPORTED_REJECTION")
    assert result["engineering_only"] is True
    assert result["scientific_admission"] is result["scientific_phase_closed"] is False
    for path in (case.root / "campaign").glob("*.json"):
        assert read(path)["engineering_only"] is True
    case.validation_values = {2: 0.0, 7: 99.0}
    assert case.run() == result
    assert case.factory_calls == [2, 7] and case.validation_calls == [2, 7] and case.terminal_calls == [7]


def test_tie_break_uses_lowest_roster_id(case):
    case.validation_values = {2: 0.02, 7: 0.02}
    result = case.run()
    assert read(verified_path(case.root, result["selection"]))["selected"]["replica"] == 2
    assert case.terminal_calls == [2]


def test_factory_cannot_mutate_frozen_candidate_configuration(case):
    case.mutate_factory = True
    manifest_hash = stable_hash(case.manifest)
    result = case.run()
    assert stable_hash(case.manifest) == result["manifest_hash"] == manifest_hash
    assert read(case.root / "campaign/manifest.json")["manifest"] == case.manifest
    assert [candidate["candidate"]["replica"] for candidate in case.union()] == [2, 7]


@pytest.mark.parametrize("roster", ([], [2, 2], [7, 2]))
def test_invalid_roster_refuses_before_factory(case, roster):
    candidates = {candidate["replica"]: candidate for candidate in case.manifest["candidates"]}
    case.manifest["candidates"] = [candidates[replica] for replica in roster]
    with pytest.raises(campaign.CampaignError, match="roster|sorted and unique"):
        case.run()
    assert not case.factory_calls


@pytest.mark.parametrize("kind", ("configuration", "source", "role"))
def test_changed_input_binding_refuses_before_callbacks(case, kind):
    binding = (case.manifest["candidates"][0]["configuration"] if kind == "configuration" else
               case.manifest["interfaces"]["runner_factory"]["source"] if kind == "source" else
               case.manifest["roles"]["terminal"])
    (case.root / binding["path"]).write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        case.run()
    assert not case.factory_calls


def test_declared_role_overlap_refuses_before_factory(case):
    path = case.root / "terminal-overlap.json"
    publish(path, {"role": "terminal", "evidence_scope": "engineering", "sample_ids": ["validation-2"]})
    case.manifest["roles"]["terminal"] = reference(case.root, path)
    with pytest.raises(campaign.CampaignError, match="role inventories"):
        case.run()
    assert not case.factory_calls


@pytest.mark.parametrize("where", ("request", "registry"))
def test_actual_screening_terminal_reuse_is_detected(case, where):
    case.screening_leak = "campaign-terminal" if where == "request" else None
    case.registry_leak = where == "registry"
    result = case.run()
    assert result["status"] == "HARNESS_BLOCKED"
    assert all("screening inventory" in outcome["reason"] for outcome in case.union())
    assert not case.validation_calls and not case.terminal_calls
    if where == "registry":
        assert all(runner.executor.calls == 0 for runner in case.runners)


@pytest.mark.parametrize("missing", (False, True))
def test_missing_or_empty_registry_is_not_verified_role_separation(case, missing):
    case.missing_registry, case.empty_registry = missing, not missing
    result = case.run()
    assert result["status"] == "HARNESS_BLOCKED"
    assert all("bound sample inventory" in outcome["reason"] for outcome in case.union())
    assert all(runner.executor.calls == 0 for runner in case.runners)
    assert not case.validation_calls and not case.terminal_calls


def test_midrun_rejection_preserves_checkpoint_event_and_roster(case):
    case.reject = {2}
    result = case.run()
    rejected, eligible = case.union()
    assert rejected["status"] == "CANDIDATE_REJECTED" and eligible["status"] == "ELIGIBLE"
    assert rejected["event"]["committed"] is False and "PermanentPassUpdateError" in rejected["traceback"]
    state = CheckpointState.from_dict(read(verified_path(case.root, rejected["checkpoint"])))
    assert state.update_index == 1 and state.optimizer_state["iteration"] == 1
    assert state.to_dict() == case.runners[0].store.recover(2)[0].to_dict()
    assert verified_path(case.root, rejected["failure_checkpoint"]).is_file()
    assert case.validation_calls == [7] and case.terminal_calls == [7]
    assert case.run() == result and case.factory_calls == [2, 7]


def test_initial_control_rejection_needs_no_committed_checkpoint(case):
    case.initial_reject = {2, 7}
    result = case.run()
    assert result["decision"] == "ALL_CANDIDATES_REJECTED"
    assert all(outcome["status"] == "CANDIDATE_REJECTED" and "checkpoint" not in outcome for outcome in case.union())
    assert all(runner.executor.calls == 0 for runner in case.runners)
    assert case.run() == result and case.factory_calls == [2, 7]


def test_corrupt_initial_publication_stays_harness_failure(case):
    case.reject, case.corrupt_initial = {2}, {2}
    result = case.run()
    failure = case.union()[0]
    assert result["status"] == "HARNESS_BLOCKED" and failure["status"] == "HARNESS_FAILURE"
    assert "classification_or_recovery_error" in failure
    assert failure["event"]["error"] == "fixture domain violation"
    assert not case.validation_calls and not case.terminal_calls


@pytest.mark.parametrize("kind", ("harness", "classifier", "unclassified"))
def test_harness_failure_is_not_accuracy_rejection(case, kind):
    case.harness = {2} if kind == "harness" else set()
    case.reject = {2} if kind != "harness" else set()
    case.classifier_failure = kind == "classifier"
    if kind == "unclassified":
        del case.manifest["interfaces"]["rejection_classifier"]
    result = case.run()
    assert result["status"] == "HARNESS_BLOCKED" and result["selection"] is None
    failure = case.union()[0]
    assert failure["status"] == "HARNESS_FAILURE"
    assert failure["event"]["committed"] is False and "PermanentPassUpdateError" in failure["traceback"]
    assert not case.validation_calls and not case.terminal_calls
    assert case.run() == result


@pytest.mark.parametrize("factory", (False, True))
def test_all_rejected_campaign_finishes_negative_without_terminal(case, factory):
    if factory:
        case.factory_reject = {2, 7}
    else:
        case.reject = {2, 7}
    result = case.run()
    assert result["decision"] == "ALL_CANDIDATES_REJECTED" and result["selection"] is result["terminal"] is None
    assert len(case.union()) == 2 and not case.validation_calls and not case.terminal_calls


@pytest.mark.parametrize("field", ("checkpoint", "failure_checkpoint"))
def test_rejected_checkpoint_hash_is_verified_on_reload(case, field):
    case.reject = {2, 7}
    case.run()
    path = verified_path(case.root, case.union()[0][field])
    path.write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        case.run()
    assert case.factory_calls == [2, 7]


def test_runner_interruption_recovers_only_uncommitted_update(case):
    case.interrupt = {2}
    with pytest.raises(InterruptedError, match="process interruption"):
        case.run()
    assert not (case.root / "campaign/candidate-2/outcome.json").exists()
    assert case.runners[0].store.recover(2)[0].update_index == 1
    result = case.run()
    assert result["status"] == "ENGINEERING_COMPLETE"
    assert case.factory_calls == [2, 2, 7]
    assert [runner.executor.calls for runner in case.runners] == [1, 1, 2]


@pytest.mark.parametrize("bad_binding", ("policy", "role", "count", "hash"))
def test_evaluator_must_bind_exact_frozen_policy_and_reserved_role(case, bad_binding):
    case.bad_binding = bad_binding
    with pytest.raises(campaign.CampaignError, match="frozen candidate"):
        case.run()
    assert not case.terminal_calls
    with pytest.raises(campaign.CampaignRepairRequired, match="ambiguous consumption"):
        case.run()
    assert case.validation_calls == [2]


@pytest.mark.parametrize("phase", ("validation", "terminal"))
def test_interrupted_evaluation_requires_scoped_repair_without_repeat(case, phase):
    case.interrupt_evaluation = phase
    with pytest.raises(InterruptedError, match=phase):
        case.run()
    counts = len(case.validation_calls), len(case.terminal_calls)
    with pytest.raises(campaign.CampaignRepairRequired) as caught:
        case.run()
    assert caught.value.phase == phase and caught.value.status == "REPAIR_REQUIRED"
    assert verified_path(case.root, caught.value.intent).is_file()
    assert (len(case.validation_calls), len(case.terminal_calls)) == counts


@pytest.mark.parametrize("publication,after", (("selection.json", True), ("terminal.json", False), ("terminal.json", True)))
def test_publication_crash_keeps_terminal_at_most_once(case, monkeypatch, publication, after):
    original = campaign.publish

    def interrupt(path, payload):
        if path.name == publication:
            if after:
                original(path, payload)
            raise InterruptedError("manufactured publication interruption")
        return original(path, payload)

    monkeypatch.setattr(campaign, "publish", interrupt)
    with pytest.raises(InterruptedError, match="publication interruption"):
        case.run()
    monkeypatch.setattr(campaign, "publish", original)
    if publication == "terminal.json" and not after:
        with pytest.raises(campaign.CampaignRepairRequired, match="terminal intent"):
            case.run()
    else:
        assert case.run()["status"] == "ENGINEERING_COMPLETE"
    assert case.factory_calls == [2, 7] and case.validation_calls == [2, 7] and case.terminal_calls == [7]


@pytest.mark.parametrize("artifact", ("candidate-2/outcome.json", "validation-2.json", "terminal.json", "selection.json"))
def test_changed_receipts_refuse_recovery_without_new_evaluations(case, artifact):
    case.run()
    path = case.root / "campaign" / artifact
    payload = read(path)
    payload["changed"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="changed campaign receipt|refusing to overwrite"):
        case.run()
    assert case.factory_calls == [2, 7] and case.validation_calls == [2, 7] and case.terminal_calls == [7]


def test_missing_terminal_intent_refuses_completed_receipt(case):
    case.run()
    (case.root / "campaign/terminal-intent.json").unlink()
    with pytest.raises(campaign.CampaignError, match="lacks its intent"):
        case.run()
    assert case.terminal_calls == [7]
