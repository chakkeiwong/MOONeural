"""Saved-row role plumbing, bank statistics and campaign handoff fixtures."""

import importlib.util
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mooneural.training import generic_independent_role_bridge as bridge_module
from mooneural.training.generic_independent_role_bridge import (
    BankLossRows,
    IndependentBankRoleBridge,
    LossCoordinates,
)
from mooneural.training.generic_replication_training_protocol import publish, reference
from mooneural.training.generic_role_banks import RoleBank, RoleBankManifest
from mooneural.training.generic_scheduled_role_banks import (
    ScheduledRoleBankRegistry,
    ScheduledRoleBinding,
)
from mooneural.training.generic_training_contracts import (
    ImmutableJSONMapping,
    PolicyView,
    stable_hash,
)

TASKS = ("value", "state", "parameter")
POLICY = PolicyView((1., 2.), "saved-policy")


def coordinates(tasks=TASKS, *, threshold=.25, minimum_banks=2):
    return LossCoordinates(tasks, (2.,) * len(tasks), (4.,) * len(tasks), (8.,) * len(tasks), (16.,) * len(tasks),
                           threshold, minimum_banks=minimum_banks)


def fixture(root, *, coords=None, policies=(POLICY,), roles=("control", "validation", "certification"), arms=(0,),
            evidence_scope="manufactured", inner_rows=3, banks=2):
    coords = coords or coordinates()
    records, bindings, repeated = [], [], {}
    for role_index, role in enumerate(roles):
        role_banks = []
        seeds = []
        for bank_index in range(banks):
            bank_id = f"{role}-bank-{bank_index}"
            seed = 100 * (role_index + 1) + bank_index
            seeds.append(seed)
            row_ids = tuple(f"{bank_id}-row-{index}" for index in range(inner_rows))
            source = root / f"{bank_id}.json"
            publish(source, {"bank_id": bank_id, "rows": list(row_ids), "manufactured": True})
            source_ref = reference(root, source)
            members = []
            for policy_index, policy in enumerate(policies):
                raw = np.full((inner_rows, len(coords.task_ids)), float(bank_index * 4 + 2) / (policy_index + 1))
                record = BankLossRows(bank_id, row_ids, policy.fingerprint(), coords.task_ids,
                                      raw, np.ones_like(raw), (source_ref,))
                records.append(record)
                members.append(record)
            role_banks.append(RoleBank(role, bank_id, seed, "fixture-target", "raw-loss-v1", coords.binding_hash(), "bank-mse-v1",
                {"global": source_ref["sha256"], "local": source_ref["sha256"]}, (bank_id,),
                {"row_ids": list(row_ids), "row_count": inner_rows, "evidence_scope": evidence_scope,
                 "policy_row_hashes": {record.policy_fingerprint: record.binding_hash() for record in members}}))
        for arm in arms:
            bindings.append(ScheduledRoleBinding("campaign", arm, role, 0 if role == "control" else None, 2, 2,
                RoleBankManifest(coords.task_ids, tuple(role_banks), {role: banks}, manifest_version=f"{role}-arm-{arm}")))
        if len(arms) > 1:
            repeated[role] = tuple(seeds)
    registry = ScheduledRoleBankRegistry(tuple(bindings), local_seed_reuse=repeated)
    return coords, registry, records


def provider(root, pieces, scope="manufactured"):
    coords, registry, records = pieces
    return IndependentBankRoleBridge(root, registry, coords, records, evidence_scope=scope, cache_directory=root / "cache")


def request(current, role="control", policy=POLICY, arm=0):
    return current.registry.request_for(role, policy, stage_id="campaign", arm_id=arm,
                                        round_number=0 if role == "control" else None, update_index=2)


def test_bank_observations_and_explicit_nonunit_coordinates(tmp_path):
    current = provider(tmp_path, fixture(tmp_path))
    control = current.evaluate(POLICY, request(current))
    validation = current.evaluate(POLICY, request(current, "validation"))
    terminal = current.evaluate(POLICY, request(current, "certification"))
    quality = control.raw_records["quality"]
    assert control.raw_records["bank_count"] == 2 and control.raw_records["inner_row_count"] == 6
    for task in TASKS:
        summary = quality["tasks"][task]
        assert summary["sample_count"] == 2
        assert summary["mean_normalized_mse"] == 1.
        assert summary["conservative_mean_normalized_mse"] == 1.25
        assert summary["standard_error_mse"] == pytest.approx(.5)
        assert validation.task_mean_mse[task] == .5
        assert validation.task_mean_mse[task] / current.coordinates.threshold == 4. / (8. * .25)
        assert validation.sample_counts[task] == 2
        assert terminal.upper_records["quality"]["tasks"][task]["mean_normalized_mse"] == .25
    assert quality["threshold_definition"]["raw_limits"] == [1.] * 3
    assert quality["threshold_definition"]["mse_matches_training_objective_coordinate"] is False
    assert terminal.passed is False and terminal.hard_vetoes["scientific_qualification"] is False
    assert current.coordinates.training == (2., 2., 2.)


def test_repeated_inner_rows_do_not_shrink_bank_standard_error(tmp_path):
    results = []
    for count in (1, 30):
        root = tmp_path / str(count)
        current = provider(root, fixture(root, inner_rows=count))
        results.append(current.evaluate(POLICY, request(current)).raw_records["quality"])
    assert results[0] == results[1]


def test_arbitrary_task_multiplicity_uses_shared_estimator_once(tmp_path):
    critical = []
    for count in (1, 5):
        root = tmp_path / str(count)
        tasks = tuple(f"task-{index}" for index in range(count))
        current = provider(root, fixture(root, coords=coordinates(tasks)))
        quality = current.evaluate(POLICY, request(current)).raw_records["quality"]
        assert quality["task_order"] == list(tasks)
        critical.append(quality["tasks"][tasks[0]]["critical_value"])
        assert current.estimator_calls == 1
    assert critical[1] > critical[0]


def test_immutable_rows_and_coordinate_roundtrip(tmp_path):
    pieces = fixture(tmp_path)
    coords, _registry, records = pieces
    assert LossCoordinates(**json.loads(json.dumps(coords.to_dict()))).binding_hash() == coords.binding_hash()
    with pytest.raises(ValueError):
        records[0].raw_mse[0, 0] = 99.
    with pytest.raises(ValueError):
        records[0].raw_mse.setflags(write=True)


@pytest.mark.parametrize("field,value", (("training", (0., 1., 1.)), ("control", (-1., 1., 1.)),
    ("validation", (float("nan"), 1., 1.)), ("certification", (1., 1.)), ("threshold", 0.),
    ("threshold", float("inf")), ("alpha", 1.), ("minimum_banks", True), ("task_ids", ("same", "same", "same"))))
def test_invalid_coordinates_refuse(field, value):
    with pytest.raises(ValueError):
        replace(coordinates(), **{field: value})


@pytest.mark.parametrize("field,value", (("raw_mse", np.full((3, 3), -1.)),
    ("raw_mse", np.full((3, 3), float("inf"))), ("raw_mse", np.ones((3, 2))),
    ("empirical_addition_raw", np.full((3, 3), -.1)), ("row_ids", ("same", "same", "same"))))
def test_invalid_saved_rows_refuse(tmp_path, field, value):
    with pytest.raises(ValueError):
        replace(fixture(tmp_path)[2][0], **{field: value})


@pytest.mark.parametrize("fault", ("missing", "duplicate", "task_order", "row_ids", "policy_hash"))
def test_complete_bound_policy_rows_required(tmp_path, fault):
    coords, registry, records = fixture(tmp_path)
    if fault == "missing":
        records.pop(0)
    elif fault == "duplicate":
        records.append(records[0])
    elif fault == "task_order":
        records[0] = replace(records[0], task_ids=tuple(reversed(TASKS)))
    elif fault == "row_ids":
        records[0] = replace(records[0], row_ids=("other1", "other2", "other3"))
    else:
        records[0] = replace(records[0], raw_mse=records[0].raw_mse + 1.)
    with pytest.raises(ValueError, match="missing|duplicate|order|inventory|policy/loss-row"):
        current = provider(tmp_path, (coords, registry, records))
        current.evaluate(POLICY, request(current))


def test_wrong_policy_or_schedule_cannot_reuse_rows(tmp_path):
    current = provider(tmp_path, fixture(tmp_path))
    bound = request(current)
    with pytest.raises(ValueError, match="policy binding"):
        current.evaluate(PolicyView((3.,), "different"), bound)
    with pytest.raises(ValueError, match="outside"):
        current.registry.request_for("control", POLICY, stage_id="campaign", arm_id=0, round_number=0, update_index=3)


def test_inner_row_overlap_is_not_hidden_by_distinct_bank_ids(tmp_path):
    coords, registry, records = fixture(tmp_path)
    entries = list(registry.bindings)
    banks = list(entries[1].manifest.banks)
    banks[0] = replace(banks[0], metadata={**banks[0].metadata, "row_ids": entries[0].manifest.banks[0].metadata["row_ids"]})
    entries[1] = replace(entries[1], manifest=replace(entries[1].manifest, banks=tuple(banks)))
    with pytest.raises(ValueError, match="overlap"):
        provider(tmp_path, (coords, ScheduledRoleBankRegistry(tuple(entries)), records))


def test_exposed_data_are_control_only_even_if_caller_requests_terminal(tmp_path):
    with pytest.raises(ValueError, match="control-only"):
        provider(tmp_path, fixture(tmp_path, evidence_scope="exposed-diagnostic"), "exposed-diagnostic")
    root = tmp_path / "control"
    current = provider(root, fixture(root, roles=("control",), evidence_scope="exposed-diagnostic"), "exposed-diagnostic")
    assert current.evaluate(POLICY, request(current)).raw_records["evidence_scope"] == "exposed-diagnostic"
    with pytest.raises(ValueError, match="cannot enter"):
        current.evaluate_campaign({})


def test_completed_cache_revalidates_inputs_without_estimator_call(tmp_path, monkeypatch):
    pieces = fixture(tmp_path)
    first = provider(tmp_path, pieces)
    expected = first.evaluate(POLICY, request(first))

    def forbidden(*args, **kwargs):
        raise AssertionError("estimator must not repeat")

    monkeypatch.setattr(bridge_module, "evaluate_upper_mean_mse", forbidden)
    recovered = provider(tmp_path, pieces)
    assert recovered.evaluate(POLICY, request(recovered)).to_dict() == expected.to_dict()
    assert recovered.estimator_calls == 0 and recovered.cache_hits == 1
    (tmp_path / "control-bank-0.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        recovered.evaluate(POLICY, request(recovered))


def test_corrupt_cache_refuses_without_estimator_call(tmp_path):
    current = provider(tmp_path, fixture(tmp_path))
    current.evaluate(POLICY, request(current))
    path = next((tmp_path / "cache").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["result"]["task_upper_mse"][TASKS[0]] = 0.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="changed role cache"):
        current.evaluate(POLICY, request(current))
    assert current.estimator_calls == 1


def test_bridge_import_without_tensorflow():
    completed = subprocess.run([sys.executable, "-c", ("import sys; "
        "from mooneural.training import generic_independent_role_bridge; assert 'tensorflow' not in sys.modules")],
        capture_output=True, text=True, timeout=15, check=False)
    assert completed.returncode == 0, completed.stderr


class CampaignCallbacks:
    def __init__(self, current):
        self.current, self.calls = current, []

    def validation(self, envelope):
        self.calls.append(("validation", envelope["candidate"]["replica"]))
        return self.current.evaluate_campaign(envelope)

    def terminal(self, envelope):
        self.calls.append(("terminal", envelope["candidate"]["replica"]))
        return self.current.evaluate_campaign(envelope)


@pytest.mark.parametrize("threshold", (.04, .25), ids=("matching", "unequal-threshold"))
def test_selected_only_campaign_terminal_and_no_call_completed_recovery(tmp_path, threshold):
    from tests.contracts.test_generic_replication_runner import TASK_IDS, checkpoint
    from tests.contracts.test_generic_training_campaign import CampaignFixture

    case = CampaignFixture(tmp_path)
    policies = tuple(replace(PolicyView.from_dict(checkpoint(replica).policy_state), values=(float(replica) + 1.,)) for replica in (2, 7))
    coords = coordinates(TASK_IDS, threshold=threshold)
    pieces = fixture(tmp_path, coords=coords, policies=policies, roles=("validation", "certification"), arms=(2, 7))
    callbacks = CampaignCallbacks(provider(tmp_path, pieces))
    source = tmp_path / "bridge-fixture-source.py"
    source.write_bytes(Path(__file__).read_bytes())
    for phase, role in (("validation", "validation"), ("terminal", "certification")):
        path = tmp_path / f"campaign-{phase}-role.json"
        banks = next(binding.manifest.banks for binding in callbacks.current.registry.bindings if binding.role == role)
        publish(path, {"role": phase, "evidence_scope": "engineering", "sample_ids": [bank.bank_id for bank in banks],
            "scheduled_role_registry_hash": callbacks.current.registry.binding_hash(), "stage_id": "campaign"})
        case.manifest["roles"][phase] = reference(tmp_path, path)
        callback = getattr(callbacks, phase)
        setattr(case, f"{phase}_evaluator", callback)
        case.manifest["interfaces"][f"{phase}_evaluator"] = {"source": reference(tmp_path, source), "qualname": callback.__qualname__}
    if threshold != .04:
        with pytest.raises(ValueError, match="selection threshold/task coordinates"):
            case.run()
        assert callbacks.current.estimator_calls == 0 and callbacks.current.cache_hits == 0
        assert callbacks.calls == [("validation", 2)]
        assert not (tmp_path / "campaign/terminal.json").exists()
        return
    result = case.run()
    assert callbacks.calls == [("validation", 2), ("validation", 7), ("terminal", 7)]
    assert result["scientific_admission"] is False and result["decision"] == "TERMINAL_REPORTED_REJECTION"
    assert case.run() == result and len(callbacks.calls) == 3
    assert callbacks.current.estimator_calls == 3


class HostRowFixture:
    def __init__(self, root):
        self.root, self.sources = root, []
        self.calls, self.forbidden, self.fault = 0, False, None

    def __call__(self, policy, request, bank, *, output_directory):
        if self.forbidden:
            raise AssertionError("provider must not repeat")
        self.calls += 1
        if self.fault == "interruption":
            raise RuntimeError("manufactured interrupted callback")
        if self.fault == "mutate-following":
            (self.root / "control-bank-1.json").write_text("changed input during callback")
        publish(output_directory / "diagnostic.json", {"policy": policy.fingerprint(), "request": request.to_dict()})
        sources = [*self.sources, *[dict(binding) for binding in bank.metadata["input_references"].values()],
                   reference(self.root, output_directory / "diagnostic.json")]
        row_ids = tuple(bank.metadata["row_ids"])
        raw = np.full((len(row_ids), 3), policy.values[0] ** 2 + int(bank.bank_id[-1]))
        additions = np.ones_like(raw)
        if self.fault == "negative":
            raw[0, 0] = -1.
        if self.fault == "nonfinite":
            raw[0, 0] = np.nan
        if self.fault == "shape":
            raw = raw[:, :2]
        if self.fault == "references":
            sources = sources[-1:]
        record = BankLossRows(bank.bank_id, row_ids, policy.fingerprint(), ("value", "state", "parameter"),
                              raw, additions, sources)
        if self.fault == "policy":
            record = replace(record, policy_fingerprint="0" * 64)
        if self.fault == "bank":
            record = replace(record, bank_id="other-bank")
        if self.fault == "tasks":
            record = replace(record, task_ids=tuple(reversed(record.task_ids)))
        if self.fault == "rows":
            record = replace(record, row_ids=tuple(f"other-{index}" for index in range(len(row_ids))))
        return record


def dynamic_fixture(root, *, scope="fresh-engineering", roles=("control", "validation")):
    coords, registry, _records = fixture(root, roles=roles, evidence_scope=scope)
    source = root / "dynamic_fixture.py"
    source.write_text("import numpy as np\nfrom dataclasses import replace\n"
                      "from mooneural.training.generic_independent_role_bridge import BankLossRows\n"
                      "from mooneural.training.generic_replication_training_protocol import publish, reference\n\n"
                      + inspect.getsource(HostRowFixture))
    spec = importlib.util.spec_from_file_location("dynamic_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.HostRowFixture(root)
    rows.sources = [reference(root, source)]
    contract = {"callback": {"module": rows.__call__.__module__, "qualname": rows.__call__.__qualname__, "source": rows.sources[0]},
                "source_references": rows.sources, "policy_dimension": len(POLICY.values), "policy_metadata": {},
                "coordinates_hash": coords.binding_hash(), "recipe": {"rule": "fixture", "dimensions": [2, 3]}}
    bindings = []
    for binding in registry.bindings:
        banks = []
        for bank in binding.manifest.banks:
            metadata = dict(bank.metadata)
            metadata.pop("policy_row_hashes")
            metadata.update(provider_binding_hash=stable_hash(contract),
                            input_references={name: reference(root, root / f"{bank.bank_id}.json") for name in bank.input_hashes})
            banks.append(replace(bank, metadata=metadata))
        bindings.append(replace(binding, manifest=replace(binding.manifest, banks=tuple(banks))))
    registry = ScheduledRoleBankRegistry(tuple(bindings))
    return coords, registry, rows, contract


def dynamic_provider(root, pieces, scope="fresh-engineering", **kwargs):
    coords, registry, rows, contract = pieces
    return IndependentBankRoleBridge(root, registry, coords, evidence_scope=scope, row_provider=rows,
                                     provider_binding=contract, cache_directory=root / "dynamic-cache", **kwargs)


@pytest.mark.parametrize("fault", (None, "missing", "path", "digest", "extra-changed-file"))
def test_dynamic_source_membership_uses_complete_canonical_references(tmp_path, monkeypatch, fault):
    pieces = dynamic_fixture(tmp_path)
    current = dynamic_provider(tmp_path, pieces)
    evaluation_request = request(current)
    bank = current._banks[evaluation_request.metadata["bank_ids"][0]]
    record = pieces[2](POLICY, evaluation_request, bank, output_directory=tmp_path / "row")
    sources = [dict(binding) for binding in record.source_references]
    if fault == "missing":
        sources.pop(0)
    elif fault == "path":
        sources[0]["path"] = "wrong-path.py"
    elif fault == "digest":
        sources[0]["sha256"] = "0" * 64
    elif fault == "extra-changed-file":
        (tmp_path / "row/diagnostic.json").write_text("changed output")
    record = replace(record, source_references=list(reversed(sources)) * 64)
    restored = BankLossRows(**record.to_dict())

    def no_pairwise_comparison(*_arguments):
        raise AssertionError("source inventory must not scan every mapping for each required source")

    monkeypatch.setattr(ImmutableJSONMapping, "__eq__", no_pairwise_comparison)
    if fault is None:
        current._validate_dynamic_record(restored, POLICY, bank)
    else:
        message = "artifact changed" if fault == "extra-changed-file" else "output omits input/provider source references"
        with pytest.raises(ValueError, match=message):
            current._validate_dynamic_record(restored, POLICY, bank)


def test_dynamic_changed_compatible_policy_preserves_static_registry(tmp_path):
    pieces = dynamic_fixture(tmp_path)
    current = dynamic_provider(tmp_path, pieces)
    registry_hash = current.registry.binding_hash()
    original = current.evaluate(POLICY, request(current))
    changed_policy = replace(POLICY, values=(3., 4.))
    changed = current.evaluate(changed_policy, request(current, policy=changed_policy))
    assert current.registry.binding_hash() == registry_hash
    assert current.provider_calls == 4 and current.estimator_calls == 2
    assert original.task_upper_mse != changed.task_upper_mse
    assert len(list((tmp_path / "dynamic-cache/dynamic-rows").glob("*/result.json"))) == 4
    for result in (original, changed):
        assert result.raw_records["bank_count"] == 2 and result.raw_records["inner_row_count"] == 6
        assert result.raw_records["engineering_only"] is True
        assert len(result.raw_records["row_receipts"]) == 2
    assert original.raw_records["quality"]["tasks"][TASKS[0]]["mean_normalized_mse"] == .375
    assert all("policy_row_hashes" not in bank.metadata for entry in pieces[1].bindings for bank in entry.manifest.banks)


@pytest.mark.parametrize("scope", ("manufactured", "fresh-finite-benchmark"))
def test_dynamic_all_typed_roles_and_completed_no_call_recovery(tmp_path, monkeypatch, scope):
    pieces = dynamic_fixture(tmp_path, scope=scope, roles=("control", "validation", "certification"))
    current = dynamic_provider(tmp_path, pieces, scope)
    expected = {role: current.evaluate(POLICY, request(current, role)) for role in ("control", "validation", "certification")}
    assert expected["validation"].task_mean_mse[TASKS[0]] == 1.5 / 8.
    assert expected["certification"].passed is False
    assert expected["certification"].hard_vetoes["scientific_qualification"] is False
    pieces[2].forbidden = True

    def forbidden(*args, **kwargs):
        raise AssertionError("estimator must not repeat")

    monkeypatch.setattr(bridge_module, "evaluate_upper_mean_mse", forbidden)
    recovered = dynamic_provider(tmp_path, pieces, scope)
    for role, previous in expected.items():
        assert recovered.evaluate(POLICY, request(recovered, role)).to_dict() == previous.to_dict()
    assert recovered.provider_calls == recovered.estimator_calls == 0
    assert recovered.cache_hits == 3 and recovered.row_cache_hits == 6


@pytest.mark.parametrize("fault", ("dimension", "metadata", "request", "source", "last-input", "callback"))
def test_dynamic_preflight_refuses_before_any_callback(tmp_path, fault):
    pieces = dynamic_fixture(tmp_path)
    current = dynamic_provider(tmp_path, pieces)
    policy = POLICY
    bound = request(current)
    if fault == "dimension":
        policy = replace(policy, values=(1.,))
        bound = request(current, policy=policy)
    elif fault == "metadata":
        from mooneural.training.generic_training_contracts import ImmutableJSONMapping

        current.provider_binding = ImmutableJSONMapping({**pieces[3], "policy_metadata": {"profile": "bound"}})
    elif fault == "request":
        policy = replace(policy, values=(3., 4.))
    elif fault == "source":
        (tmp_path / "dynamic_fixture.py").write_text("changed source")
    elif fault == "last-input":
        (tmp_path / "control-bank-1.json").write_text("changed last input")
    else:
        current.row_provider = lambda *args, **kwargs: None
    with pytest.raises(ValueError):
        current.evaluate(policy, bound)
    assert pieces[2].calls == current.provider_calls == current.estimator_calls == 0
    assert not (tmp_path / "dynamic-cache").exists()


@pytest.mark.parametrize("fault", ("policy", "bank", "tasks", "rows", "negative", "nonfinite", "shape", "references"))
def test_dynamic_invalid_output_retains_intent_and_refuses_retry(tmp_path, fault):
    pieces = dynamic_fixture(tmp_path)
    pieces[2].fault = fault
    current = dynamic_provider(tmp_path, pieces)
    with pytest.raises(ValueError):
        current.evaluate(POLICY, request(current))
    assert current.provider_calls == 1 and current.estimator_calls == 0
    assert len(list((tmp_path / "dynamic-cache/dynamic-rows").glob("*/intent.json"))) == 1
    assert not list((tmp_path / "dynamic-cache/dynamic-rows").glob("*/result.json"))
    recovered = dynamic_provider(tmp_path, pieces)
    with pytest.raises(ValueError, match="scoped repair required"):
        recovered.evaluate(POLICY, request(recovered))
    assert recovered.provider_calls == 0


def test_dynamic_interruption_is_not_silently_retried(tmp_path):
    pieces = dynamic_fixture(tmp_path)
    current = dynamic_provider(tmp_path, pieces)
    pieces[2].fault = "interruption"
    with pytest.raises(RuntimeError, match="interrupted callback"):
        current.evaluate(POLICY, request(current))
    pieces[2].fault = None
    with pytest.raises(ValueError, match="scoped repair required"):
        dynamic_provider(tmp_path, pieces).evaluate(POLICY, request(current))
    assert pieces[2].calls == 1


def test_dynamic_input_changes_between_callbacks_refuse_next_call(tmp_path):
    pieces = dynamic_fixture(tmp_path)
    pieces[2].fault = "mutate-following"
    current = dynamic_provider(tmp_path, pieces)
    with pytest.raises(ValueError, match="artifact changed"):
        current.evaluate(POLICY, request(current))
    assert current.provider_calls == 1 and current.estimator_calls == 0


def test_dynamic_completed_rows_survive_interrupted_aggregation(tmp_path, monkeypatch):
    pieces = dynamic_fixture(tmp_path)
    current = dynamic_provider(tmp_path, pieces)
    estimator = bridge_module.evaluate_upper_mean_mse

    def interrupted(*args, **kwargs):
        raise RuntimeError("manufactured aggregation interruption")

    monkeypatch.setattr(bridge_module, "evaluate_upper_mean_mse", interrupted)
    with pytest.raises(RuntimeError, match="aggregation interruption"):
        current.evaluate(POLICY, request(current))
    pieces[2].forbidden = True
    monkeypatch.setattr(bridge_module, "evaluate_upper_mean_mse", estimator)
    recovered = dynamic_provider(tmp_path, pieces)
    assert recovered.evaluate(POLICY, request(recovered)).raw_records["bank_count"] == 2
    assert recovered.provider_calls == 0 and recovered.row_cache_hits == 2
    assert recovered.estimator_calls == 1


@pytest.mark.parametrize("fault", ("rows", "intent", "output-file"))
def test_dynamic_recovery_checks_row_receipts_and_output_files(tmp_path, fault):
    pieces = dynamic_fixture(tmp_path)
    current = dynamic_provider(tmp_path, pieces)
    current.evaluate(POLICY, request(current))
    receipt = next((tmp_path / "dynamic-cache/dynamic-rows").glob("*/result.json"))
    if fault == "rows":
        payload = json.loads(receipt.read_text())
        payload["rows"]["raw_mse"][0][0] = 1e9
        receipt.write_text(json.dumps(payload))
    elif fault == "intent":
        (receipt.parent / "intent.json").write_text("{}")
    else:
        (receipt.parent / "provider/diagnostic.json").write_text("changed output")
    pieces[2].forbidden = True
    recovered = dynamic_provider(tmp_path, pieces)
    with pytest.raises(ValueError, match="changed"):
        recovered.evaluate(POLICY, request(recovered))
    assert recovered.provider_calls == recovered.estimator_calls == 0


@pytest.mark.parametrize("fault", ("coordinates", "bank-binding", "policy-output-hash", "input-reference", "saved-records", "terminal"))
def test_dynamic_invalid_declaration_refuses(tmp_path, fault):
    pieces = list(dynamic_fixture(tmp_path, roles=("control", "certification") if fault == "terminal" else ("control",)))
    kwargs = {}
    if fault == "coordinates":
        pieces[3]["coordinates_hash"] = "0" * 64
    elif fault == "saved-records":
        kwargs["records"] = fixture(tmp_path / "saved")[2]
    elif fault in ("bank-binding", "policy-output-hash", "input-reference"):
        entry = pieces[1].bindings[0]
        bank = entry.manifest.banks[0]
        metadata = dict(bank.metadata)
        if fault == "bank-binding":
            metadata["provider_binding_hash"] = "0" * 64
        elif fault == "policy-output-hash":
            metadata["policy_row_hashes"] = {}
        else:
            metadata["input_references"] = {}
        bank = replace(bank, metadata=metadata)
        manifest = replace(entry.manifest, banks=(bank, *entry.manifest.banks[1:]))
        pieces[1] = ScheduledRoleBankRegistry((replace(entry, manifest=manifest),))
    with pytest.raises(ValueError):
        dynamic_provider(tmp_path, pieces, **kwargs)
    assert pieces[2].calls == 0


def test_dynamic_binding_is_detached_from_callers_mutable_contract(tmp_path):
    pieces = dynamic_fixture(tmp_path)
    current = dynamic_provider(tmp_path, pieces)
    binding_hash = stable_hash(current.provider_binding)
    pieces[3]["recipe"]["dimensions"].append(99)
    assert stable_hash(current.provider_binding) == binding_hash
    assert current.evaluate(POLICY, request(current)).request.policy_fingerprint == POLICY.fingerprint()
