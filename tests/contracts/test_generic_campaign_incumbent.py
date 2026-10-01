"""Frozen incumbent selection through the shared campaign and manufactured runner."""

import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from tests.contracts.test_generic_replication_runner import TASK_IDS, checkpoint
from tests.contracts.test_generic_training_campaign import CampaignFixture, read

from mooneural.training import generic_training_campaign as campaign
from mooneural.training.generic_replication_design import ReplicationDesignBinding
from mooneural.training.generic_replication_training_protocol import (
    publish,
    reference,
    verified_path,
)
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    PolicyView,
    stable_hash,
)


class IncumbentFixture(CampaignFixture):
    def __init__(self, root):
        super().__init__(root)
        self.forbidden = set()
        self.validation_requests = {}
        self.task_values = {}
        self.bad_incumbent_policy = False
        self.interrupt_incumbent = False
        configuration = read(root / "configuration-2.json")
        design = ReplicationDesignBinding.from_dict(configuration["design"])
        parent = design.bind_checkpoint(checkpoint(2, update_index=11))
        publish(root / "incumbent-configuration.json", configuration)
        publish(root / "incumbent-checkpoint.json", parent.to_dict())
        self.manifest["incumbent"] = {
            "replica": 99,
            "configuration": reference(root, root / "incumbent-configuration.json"),
            "checkpoint": reference(root, root / "incumbent-checkpoint.json"),
            "policy_fingerprint": parent.policy_fingerprint,
        }
        self.validation_values[99] = 0.03
        source = root / "incumbent-fixture-source.py"
        source.write_bytes(Path(__file__).read_bytes())
        for name in campaign.INTERFACES:
            self.manifest["interfaces"][name] = {
                "source": reference(root, source), "qualname": getattr(self, name).__qualname__,
            }

    def runner_factory(self, candidate, directory):
        assert ("factory", candidate["replica"]) not in self.forbidden
        assert candidate["replica"] != self.manifest["incumbent"]["replica"]
        return super().runner_factory(candidate, directory)

    def validation_evaluator(self, request):
        replica = request["candidate"]["replica"]
        assert ("validation", replica) not in self.forbidden
        self.validation_requests[replica] = request
        evaluated = super().validation_evaluator(request)
        if replica == 99 and self.interrupt_incumbent:
            raise InterruptedError("manufactured incumbent validation interruption")
        if replica == 99 and self.bad_incumbent_policy:
            evaluated = replace(evaluated, request=replace(
                evaluated.request, policy_fingerprint=stable_hash("another-policy")))
        if replica in self.task_values:
            evaluated = replace(evaluated, task_mean_mse=self.task_values[replica])
        return evaluated

    def terminal_evaluator(self, request):
        assert ("terminal", request["candidate"]["replica"]) not in self.forbidden
        selection = read(verified_path(self.root, request["selection"]))
        assert selection["nominated"] is True
        for record in (*selection["records"], selection["incumbent"]):
            assert verified_path(self.root, record["validation"]).is_file()
        return super().terminal_evaluator(request)

    def forbid_completed(self):
        self.forbidden.update(("factory", replica) for replica in self.factory_calls)
        self.forbidden.update(("validation", replica) for replica in self.validation_calls)
        self.forbidden.update(("terminal", replica) for replica in self.terminal_calls)


@pytest.fixture
def case(tmp_path):
    return IncumbentFixture(tmp_path)


@pytest.mark.parametrize("parent_mean,nominated", ((0.03, True), (0.01, False), (0.005, False)))
def test_strict_minimax_comparison_and_zero_call_recovery(case, parent_mean, nominated):
    case.validation_values[99] = parent_mean
    original = {name: verified_path(case.root, case.manifest["incumbent"][name]).read_bytes()
                for name in ("checkpoint", "configuration")}
    result = case.run()
    selection = read(verified_path(case.root, result["selection"]))
    assert selection["nominated"] is nominated
    assert selection["selected"]["replica"] == (7 if nominated else 99)
    assert [record["replica"] for record in selection["records"]] == [2, 7]
    assert case.factory_calls == [2, 7]
    assert case.validation_calls == [2, 7, 99]
    assert case.terminal_calls == ([7] if nominated else [])
    assert result["decision"] == ("TERMINAL_REPORTED_REJECTION" if nominated else "INCUMBENT_RETAINED")
    assert result["status"] == "ENGINEERING_COMPLETE"
    assert result["scientific_admission"] is result["scientific_phase_closed"] is False
    if not nominated:
        assert result["terminal"] is None
        assert not (case.root / "campaign/terminal-intent.json").exists()
    requests = list(case.validation_requests.values())
    assert all(request["role"] == case.manifest["roles"]["validation"] for request in requests)
    assert all(case.evaluation_request(request).anchor_ids == ("campaign-validation",) for request in requests)
    assert case.validation_requests[99]["checkpoint"] == case.manifest["incumbent"]["checkpoint"]
    parent = CheckpointState.from_dict(read(verified_path(case.root, case.manifest["incumbent"]["checkpoint"])))
    assert parent.update_index == 11 and parent.metadata["arm_id"] == 2
    for name, content in original.items():
        assert verified_path(case.root, case.manifest["incumbent"][name]).read_bytes() == content
    case.forbid_completed()
    assert case.run() == result
    assert case.factory_calls == [2, 7] and case.validation_calls == [2, 7, 99]


def test_minimax_does_not_select_by_mean_or_require_every_task_improve(case):
    case.task_values = {
        2: dict(zip(TASK_IDS, (0.009, 0.04), strict=True)),
        7: dict(zip(TASK_IDS, (0.03, 0.03), strict=True)),
        99: dict(zip(TASK_IDS, (0.005, 0.035), strict=True)),
    }
    case.terminal_pass = True
    result = case.run()
    selection = read(verified_path(case.root, result["selection"]))
    assert selection["selected"]["replica"] == 7
    assert selection["selected"]["absolute_threshold_ratio"] == 0.75
    assert selection["incumbent"]["absolute_threshold_ratio"] == pytest.approx(0.875)
    assert result["decision"] == "TERMINAL_REPORTED_PASS"


@pytest.mark.parametrize("nominated", (False, True))
def test_fresh_process_completed_recovery_has_no_callbacks(case, nominated):
    case.validation_values[99] = 0.03 if nominated else 0.005
    result = case.run()
    command = """
import json
import sys
sys.path[:] = json.loads(sys.argv[2])
from pathlib import Path
from tests.contracts.test_generic_campaign_incumbent import IncumbentFixture
case = IncumbentFixture(Path(sys.argv[1]))
case.forbidden = {(kind, replica) for kind in ("factory", "validation", "terminal") for replica in (2, 7, 99)}
result = case.run()
assert not case.factory_calls and not case.validation_calls and not case.terminal_calls
print(json.dumps(result))
"""
    completed = subprocess.run([sys.executable, "-c", command, str(case.root), json.dumps(sys.path)],
                               capture_output=True, text=True, check=False, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == result


def test_candidate_tie_still_uses_lowest_id_before_incumbent_comparison(case):
    case.validation_values.update({2: 0.01, 7: 0.01})
    result = case.run()
    assert read(verified_path(case.root, result["selection"]))["selected"]["replica"] == 2
    assert case.terminal_calls == [2]


@pytest.mark.parametrize("failure", ("rejected", "harness"))
def test_no_eligible_selection_means_no_incumbent_or_terminal_calls(case, failure):
    if failure == "rejected":
        case.reject = {2, 7}
    else:
        case.harness = {2}
    result = case.run()
    assert result["selection"] is result["terminal"] is None
    assert not case.validation_calls and not case.terminal_calls
    assert result.get("decision", result["status"]) == (
        "ALL_CANDIDATES_REJECTED" if failure == "rejected" else "HARNESS_BLOCKED")
    case.forbid_completed()
    assert case.run() == result


@pytest.mark.parametrize("replica", (2, 7, -1, True, "99"))
def test_invalid_incumbent_id_refuses_before_callbacks(case, replica):
    case.manifest["incumbent"]["replica"] = replica
    with pytest.raises(campaign.CampaignError, match="incumbent evaluation ID"):
        case.run()
    assert not case.factory_calls and not case.validation_calls


@pytest.mark.parametrize("change", ("missing", "extra", "null", "fingerprint"))
def test_invalid_incumbent_binding_refuses_before_callbacks(case, change):
    if change == "missing":
        del case.manifest["incumbent"]["checkpoint"]
    elif change == "extra":
        case.manifest["incumbent"]["extra"] = True
    elif change == "null":
        case.manifest["incumbent"] = None
    else:
        case.manifest["incumbent"]["policy_fingerprint"] = stable_hash("another-policy")
    with pytest.raises(campaign.CampaignError, match="incumbent"):
        case.run()
    assert not case.factory_calls and not case.validation_calls


@pytest.mark.parametrize("field", ("checkpoint", "configuration"))
@pytest.mark.parametrize("completed", (False, True))
def test_incumbent_reference_drift_refuses_even_on_completed_recovery(case, field, completed):
    if completed:
        case.run()
        case.forbid_completed()
    counts = len(case.factory_calls), len(case.validation_calls), len(case.terminal_calls)
    verified_path(case.root, case.manifest["incumbent"][field]).write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        case.run()
    assert (len(case.factory_calls), len(case.validation_calls), len(case.terminal_calls)) == counts


@pytest.mark.parametrize("change", ("tasks", "threshold", "design"))
def test_incumbent_configuration_must_describe_its_checkpoint_and_coordinates(case, change):
    configuration = read(verified_path(case.root, case.manifest["incumbent"]["configuration"]))
    design = ReplicationDesignBinding.from_dict(configuration["design"])
    design = (replace(design, task_ids=tuple(reversed(TASK_IDS))) if change == "tasks" else
              replace(design, threshold=0.02) if change == "threshold" else
              replace(design, design_id="different-design"))
    path = case.root / "different-incumbent-configuration.json"
    publish(path, {"design": design.to_dict()})
    case.manifest["incumbent"]["configuration"] = reference(case.root, path)
    with pytest.raises(ValueError, match="incumbent task/selection|checkpoint replication design"):
        case.run()
    assert not case.factory_calls and not case.validation_calls


@pytest.mark.parametrize("bad_policy", (False, True))
def test_incumbent_consumed_intent_without_receipt_is_never_replayed(case, bad_policy):
    case.bad_incumbent_policy = bad_policy
    case.interrupt_incumbent = not bad_policy
    with pytest.raises((campaign.CampaignError, InterruptedError)):
        case.run()
    assert not (case.root / "campaign/selection.json").exists()
    case.forbid_completed()
    with pytest.raises(campaign.CampaignRepairRequired, match="validation intent"):
        case.run()
    assert case.validation_calls == [2, 7, 99] and not case.terminal_calls


@pytest.mark.parametrize("publication", ("validation-2.json", "validation-99.json", "selection.json", "terminal.json"))
@pytest.mark.parametrize("nominated", (False, True))
def test_resume_after_durable_validation_and_selection_never_repeats_calls(case, monkeypatch, publication, nominated):
    if not nominated and publication == "terminal.json":
        publication = "result.json"
    case.validation_values[99] = 0.03 if nominated else 0.005
    original = campaign.publish

    def interrupt(path, payload):
        original(path, payload)
        if path.name == publication:
            raise InterruptedError("manufactured durable publication interruption")

    monkeypatch.setattr(campaign, "publish", interrupt)
    with pytest.raises(InterruptedError, match="durable publication interruption"):
        case.run()
    monkeypatch.setattr(campaign, "publish", original)
    case.forbid_completed()
    result = case.run()
    assert result["status"] == "ENGINEERING_COMPLETE"
    assert case.factory_calls == [2, 7] and case.validation_calls == [2, 7, 99]
    assert case.terminal_calls == ([7] if nominated else [])
    case.forbid_completed()
    assert case.run() == result


def test_changed_incumbent_validation_receipt_refuses_without_callbacks(case):
    case.run()
    case.forbid_completed()
    path = case.root / "campaign/validation-99.json"
    payload = read(path)
    payload["evaluation"]["task_mean_mse"][TASK_IDS[0]] = 0.0
    path.write_text(json.dumps(payload))
    with pytest.raises(campaign.CampaignError, match="changed campaign receipt"):
        case.run()


@pytest.mark.parametrize("nominated", (False, True))
def test_shared_bridge_dispatches_same_banks_to_candidate_and_historical_parent(case, nominated):
    from tests.contracts.test_generic_independent_role_bridge import (
        CampaignCallbacks,
        coordinates,
        fixture,
        provider,
    )

    from mooneural.training.generic_scheduled_role_banks import ScheduledRoleBankRegistry

    configuration = read(case.root / "configuration-2.json")
    design = replace(ReplicationDesignBinding.from_dict(configuration["design"]), replica_ids=(0,),
                     initial_state_hashes={"0": stable_hash(checkpoint(0).to_dict())})
    publish(case.root / "configuration-0.json", {"design": design.to_dict()})
    candidate = {"replica": 0, "configuration": reference(case.root, case.root / "configuration-0.json")}
    case.manifest["candidates"] = [candidate]
    parent = design.bind_checkpoint(checkpoint(0, update_index=178))
    publish(case.root / "parent-178.json", parent.to_dict())
    case.manifest["incumbent"] = {
        "replica": 1, "configuration": candidate["configuration"],
        "checkpoint": reference(case.root, case.root / "parent-178.json"),
        "policy_fingerprint": parent.policy_fingerprint,
    }
    screening = case.root / "screening-0.json"
    publish(screening, {"role": "screening", "evidence_scope": "engineering",
                        "sample_ids": ["validation-0", "certification-0"]})
    case.manifest["roles"]["screening"] = reference(case.root, screening)
    parent_policy = PolicyView.from_dict(parent.policy_state)
    candidate_policy = replace(parent_policy, values=(1.,))
    policies = (parent_policy, candidate_policy) if nominated else (candidate_policy, parent_policy)
    coords, registry, rows = fixture(case.root, coords=coordinates(TASK_IDS, threshold=.04),
                                    policies=policies, roles=("validation", "certification"), arms=(0,))
    bindings = [replace(binding, min_update=0, max_update=8) for binding in registry.bindings]
    validation = next(binding for binding in bindings if binding.role == "validation")
    bindings.append(replace(validation, arm_id=1, min_update=178, max_update=178,
                            manifest=replace(validation.manifest, manifest_version="validation-incumbent")))
    registry = ScheduledRoleBankRegistry(tuple(bindings), local_seed_reuse={
        "validation": tuple(bank.seed for bank in validation.manifest.banks),
    })
    callbacks = CampaignCallbacks(provider(case.root, (coords, registry, rows)))
    source = case.root / "bridge-fixture-source.py"
    source.write_bytes(Path(inspect.getsourcefile(CampaignCallbacks)).read_bytes())
    for phase, role in (("validation", "validation"), ("terminal", "certification")):
        banks = next(binding.manifest.banks for binding in registry.bindings if binding.role == role)
        path = case.root / f"bridge-{phase}.json"
        publish(path, {"role": phase, "evidence_scope": "engineering", "sample_ids": [bank.bank_id for bank in banks],
                       "scheduled_role_registry_hash": registry.binding_hash(), "stage_id": "campaign"})
        case.manifest["roles"][phase] = reference(case.root, path)
        callback = getattr(callbacks, phase)
        setattr(case, f"{phase}_evaluator", callback)
        case.manifest["interfaces"][f"{phase}_evaluator"] = {
            "source": reference(case.root, source), "qualname": callback.__qualname__,
        }
    result = case.run()
    selection = read(verified_path(case.root, result["selection"]))
    assert selection["nominated"] is nominated
    assert selection["selected"]["replica"] == (0 if nominated else 1)
    assert case.factory_calls == [0]
    assert callbacks.calls == [("validation", 0), ("validation", 1)] + ([("terminal", 0)] if nominated else [])
    evaluations = [read(verified_path(case.root, record["validation"]))
                   for record in (selection["records"][0], selection["incumbent"])]
    for evaluation, expected_update in zip(evaluations, (2, 178), strict=True):
        state = CheckpointState.from_dict(read(verified_path(case.root, evaluation["request"]["checkpoint"])))
        assert state.update_index == expected_update and state.metadata["arm_id"] == 0
        assert evaluation["evaluation"]["request"]["metadata"]["scheduled_request"]["policy_fingerprint"] == state.policy_fingerprint
    assert evaluations[0]["request"]["role"] == evaluations[1]["request"]["role"]
    assert evaluations[0]["evaluation"]["request"]["seeds"] == evaluations[1]["evaluation"]["request"]["seeds"]
    estimator_calls = callbacks.current.estimator_calls
    case.forbid_completed()
    assert case.run() == result
    assert callbacks.current.estimator_calls == estimator_calls == (3 if nominated else 2)
    assert len(callbacks.calls) == estimator_calls
