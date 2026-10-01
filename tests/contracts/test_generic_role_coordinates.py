"""Explicit training/gate coordinates through the existing adapter and roles."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from tests.contracts import test_generic_live_role_lifecycle as live
from tests.contracts.test_generic_finite_objective_runner import arguments, quadratic

from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_independent_role_bridge import (
    GroupedBankLossRows,
    GroupedRoleSpec,
    IndependentBankRoleBridge,
    LossCoordinates,
)
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_replication_runner import validation_score
from mooneural.training.generic_replication_training_protocol import publish, reference
from mooneural.training.generic_role_banks import RoleBank, RoleBankManifest
from mooneural.training.generic_scheduled_role_banks import (
    ScheduledRoleBankRegistry,
    ScheduledRoleBinding,
)
from mooneural.training.generic_training_contracts import PolicyView, stable_hash

PACKET = finite.ROOT / "docs/experiments/generic-neural-solver/phase9-ez-role-coordinate-repair-20260920"


@pytest.fixture
def root():
    parent = finite.ROOT / ".pytest_cache"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="role-coordinate-fixture-", dir=parent) as temporary:
        yield Path(temporary)


def opt_arguments(root):
    args, factory = live.injected_arguments(root)
    coordinates = LossCoordinates(args["task_ids"], tuple(args["denominators"]), (4., 10.), (4., 10.), (4., 10.), 1., minimum_banks=2)
    factory.coords = coordinates
    args["threshold"] = 1.
    args["role_binding"].update(coordinate_mode="separate-training-gates-v1", coordinates=coordinates.to_dict(),
        raw_role_limits={role: [4., 10.] for role in ("control", "validation", "certification")})
    return args, factory


def test_opt_in_training_gate_metadata_and_actual_scheduled_selection(root):
    args, _factory = opt_arguments(root)
    args["objective"] = quadratic
    runner, adapter, provider = finite.build_runner(root / "runtime", **args)
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    raw, normalized, gradients = adapter.evaluate(policy)
    np.testing.assert_allclose(normalized, raw.numpy() / [2., 5.], rtol=0., atol=0.)
    np.testing.assert_allclose(gradients, 2. * (np.array(policy.values)[None, :] - np.array([[0.], [1.]])) / np.array([[2.], [5.]]))
    for row, denominator, gate in zip(adapter.metadata.normalizations, [2., 5.], [4., 10.], strict=True):
        assert row.training.factor == 1. / denominator
        assert row.selection.factor == row.terminal.factor == 1. / gate
    control = provider.control(0, policy, provider.stage, 0)
    validation = provider.validation(0, policy, provider.stage, update_index=0)
    assert control.task_upper_mse == pytest.approx({"equation.a": 2.5, "equation.b": .05})
    assert validation_score(validation, args["task_ids"], args["threshold"]) == pytest.approx(2.5)
    assert validation.provenance["quality"]["threshold_definition"]["mse_matches_training_objective_coordinate"] is False
    with pytest.raises(ValueError, match="raw/D"):
        adapter._validate_outputs((raw, normalized / 2., gradients), gradients=True)
    training = json.loads((root / "runtime/training-configuration.json").read_text())
    assert training["role_coordinate_contract"]["coordinate_mode"] == "separate-training-gates-v1"
    with pytest.raises(ValueError, match="scheduled role"):
        finite.FiniteObjectiveProvider(adapter, provider.roles, provider.stage, ())


@pytest.mark.parametrize("fault", ("absent_mode", "unknown_mode", "training", "unequal_gates", "raw_limits", "missing_limits", "threshold"))
def test_incomplete_or_mismatched_opt_in_refuses_before_factory(root, fault):
    args, factory = opt_arguments(root)
    declaration = args["role_binding"]
    if fault == "absent_mode":
        declaration.pop("coordinate_mode")
    elif fault == "unknown_mode":
        declaration["coordinate_mode"] = "unknown"
    elif fault == "training":
        declaration["coordinates"]["training"][0] = 3.
    elif fault == "unequal_gates":
        declaration["coordinates"]["validation"][0] = 8.
    elif fault == "raw_limits":
        declaration["raw_role_limits"]["control"][0] = 8.
    elif fault == "missing_limits":
        declaration.pop("raw_role_limits")
    else:
        args["threshold"] = .25
    with pytest.raises(ValueError):
        finite.build_runner(root / "runtime", **args)
    assert factory.calls == 0


def test_legacy_adapter_metadata_and_payload_match_frozen_source(root):
    args = arguments(root, objective=quadratic)
    runner, adapter, _provider = finite.build_runner(root / "runtime", **args)
    filename = Path(__file__).resolve().parents[2] / "tests/support/archived_finite.py"
    name = "mooneural.training._frozen_role_coordinate_finite"
    specification = importlib.util.spec_from_file_location(name, filename)
    old = importlib.util.module_from_spec(specification)
    sys.modules[name] = old
    specification.loader.exec_module(old)
    old.ROOT = finite.ROOT
    previous = old.FiniteObjectiveAdapter(adapter.task_ids, adapter.denominators, adapter.policy_dimension,
        adapter.objective, adapter.callback_binding, adapter.metadata.input_binding, adapter.sources,
        adapter.references, adapter.threshold)
    assert previous.metadata.normalization_binding == adapter.metadata.normalization_binding
    assert [row.to_dict() for row in previous.metadata.normalizations] == [row.to_dict() for row in adapter.metadata.normalizations]
    assert previous.adapter_identity() == adapter.adapter_identity()
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    for before, after in zip(previous.evaluate(policy), adapter.evaluate(policy), strict=True):
        np.testing.assert_array_equal(before, after)
    assert "role_coordinate_contract" not in json.loads((root / "runtime/configuration.json").read_text())


def test_opt_in_completed_shared_lifecycle_is_zero_call_recoverable(root, monkeypatch):
    args, _factory = opt_arguments(root)
    runner, _adapter, _provider = finite.build_runner(root / "runtime", **args)
    live.host_updates(runner, monkeypatch)
    result = runner.run()
    restored, adapter, provider = finite.build_runner(root / "runtime", **args)
    for instance, names in ((restored.executor, ("step",)), (provider, ("control", "validation", "certification")),
                            (adapter, ("evaluate", "evaluate_values"))):
        for name in names:
            monkeypatch.setattr(instance, name, live.forbidden)
    cached = restored.run()
    assert cached.states[0].to_dict() == result.states[0].to_dict()
    assert cached.selection == result.selection and cached.evaluations == result.evaluations
    assert provider.bridge.provider_calls == provider.bridge.estimator_calls == 0


@pytest.mark.parametrize("field", ("coordinates", "grouping", "coordinate_mode"))
def test_continuation_cannot_change_role_contract(root, field):
    args, _factory = opt_arguments(root)
    contract = finite._coordinate_contract(args["role_binding"])
    parent = {"role_coordinate_contract": contract}
    finite._check_continuation_coordinates(parent, contract)
    changed = json.loads(json.dumps(contract))
    if field == "coordinates":
        changed[field]["control"][0] *= 2.
    else:
        changed[field] = {"changed": True}
    with pytest.raises(ValueError, match="continuation role"):
        finite._check_continuation_coordinates(parent, changed)
    with pytest.raises(ValueError, match="continuation role"):
        finite._check_continuation_coordinates(parent, None)


def test_training_denominator_changes_do_not_change_gate_values(root):
    args, _factory = opt_arguments(root)
    contract = finite._coordinate_contract(args["role_binding"])
    runner, adapter, provider = finite.build_runner(root / "runtime", **args)
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    result = provider.validation(0, policy, provider.stage, update_index=0)
    assert result.task_mean_mse["equation.a"] == 2.5
    changed = json.loads(json.dumps(contract))
    changed["coordinates"]["training"] = [8., 20.]
    second = finite.FiniteObjectiveAdapter(adapter.task_ids, np.array([8., 20.]), adapter.policy_dimension,
        quadratic, adapter.callback_binding, adapter.metadata.input_binding, adapter.sources, adapter.references,
        1., role_coordinate_contract=changed)
    assert [row.selection.factor for row in second.metadata.normalizations] == [row.selection.factor for row in adapter.metadata.normalizations]
    assert [row.training.factor for row in second.metadata.normalizations] == [.125, .05]


class GroupedLifecycleRows:
    def __init__(self, coordinates, grouping, source):
        self.coordinates, self.grouping, self.source = coordinates, grouping, source

    def __call__(self, policy, request, bank, *, output_directory):
        family = bank.metadata["family"]
        tasks = getattr(self.grouping, f"{family}_task_ids")
        raw = np.full((len(bank.metadata["row_ids"]), len(tasks)), .4)
        if family == "global":
            raw[:, 0] = 8.
        return GroupedBankLossRows(bank.bank_id, tuple(bank.metadata["row_ids"]), policy.fingerprint(), tasks,
            raw, raw, (self.source, *[dict(item) for item in bank.metadata["input_references"].values()]),
            family, tuple(bank.metadata["row_cell_ids"]))


class GroupedLifecycleFactory:
    def __init__(self, root, coordinates, prefix):
        self.root, self.coordinates, self.prefix = root, coordinates, prefix
        tasks = coordinates.task_ids
        families = {f"{prefix}-{ordinal}-{family}-{index}": family for ordinal in range(4)
                    for family in ("global", "local") for index in range(2)}
        self.grouping = GroupedRoleSpec(tasks, tuple(f"cell-{index}" for index in range(9)), tasks[:6], tasks[6:],
            tuple(np.outer([.375, .25, .375], [.375, .25, .375]).reshape(-1)), "stratified", families,
            .02, .03, 2, 2, "manufactured raw records, no scientific qualification")
        self.inputs = {}
        for bank_id in families:
            path = root / f"{bank_id}.json"
            publish(path, {"bank": bank_id, "manufactured": True})
            self.inputs[bank_id] = reference(finite.ROOT, path)

    def __call__(self, adapter, stage, output):
        source = reference(finite.ROOT, Path(__file__))
        rows = GroupedLifecycleRows(self.coordinates, self.grouping, source)
        contract = {"callback": {"module": GroupedLifecycleRows.__module__, "qualname": "GroupedLifecycleRows.__call__", "source": source},
            "source_references": [source], "policy_dimension": adapter.policy_dimension,
            "policy_metadata": dict(adapter.policy_metadata), "coordinates_hash": self.coordinates.binding_hash(),
            "grouping_hash": self.grouping.binding_hash()}
        scopes = [("control", stage.from_round, stage.start_update, stage.start_update),
                  ("control", stage.last_round, stage.stop_update, stage.stop_update),
                  ("validation", None, stage.start_update, stage.stop_update),
                  ("certification", None, stage.start_update, stage.stop_update)]
        entries = []
        for ordinal, (role, number, start, stop) in enumerate(scopes):
            banks = []
            for family in ("global", "local"):
                for index in range(2):
                    bank_id = f"{self.prefix}-{ordinal}-{family}-{index}"
                    cells = self.grouping.cell_ids if family == "global" else ()
                    row_ids = tuple(f"{bank_id}-row-{row}" for row in range(9 if cells else 2))
                    binding = self.inputs[bank_id]
                    banks.append(RoleBank(role, bank_id, stage.start_update * 100 + 10 * ordinal + len(banks),
                        "manufactured", "v1", self.coordinates.binding_hash(), "bank-upper-v1",
                        {"global": binding["sha256"]}, (bank_id,), {"row_ids": row_ids, "row_count": len(row_ids),
                         "row_cell_ids": cells, "family": family, "grouping_hash": self.grouping.binding_hash(),
                         "evidence_scope": "manufactured", "provider_binding_hash": stable_hash(contract),
                         "input_references": {"global": binding}}))
            entries.append(ScheduledRoleBinding(stage.stage_id, 0, role, number, start, stop,
                RoleBankManifest(adapter.task_ids, tuple(banks), {role: len(banks)})))
        registry = ScheduledRoleBankRegistry(tuple(entries))
        bridge = IndependentBankRoleBridge(finite.ROOT, registry, self.coordinates, evidence_scope="manufactured",
            grouping=self.grouping, row_provider=rows, provider_binding=contract, cache_directory=output / "cache")
        return registry, finite.ScheduledRoleProvider(adapter, stage, bridge), {
            "adapter_input_fingerprint": adapter.metadata.input_fingerprint, "registry_hash": registry.binding_hash(),
            "provider_source": finite._reference(finite.__file__), "sources": [source], "row_provider": contract}


def grouped_lifecycle_arguments(root, base, prefix):
    args = dict(base)
    coordinates = LossCoordinates(args["task_ids"], (2.,) * 9, (4.,) * 9, (4.,) * 9, (4.,) * 9, 1., minimum_banks=2)
    factory = GroupedLifecycleFactory(root, coordinates, prefix)
    args.update(role_provider_factory=factory, stage_id=prefix, role_binding={
        "factory_source": finite._reference(__file__), "inputs": factory.inputs, "coordinates": coordinates.to_dict(),
        "coordinate_mode": "separate-training-gates-v1", "raw_role_limits": {role: [4.] * 9 for role in ("control", "validation", "certification")},
        "grouping": factory.grouping.to_dict(), "training_row_ids": ["training-only-row"],
        "measure": {"scientific_admission": False, "production_integrity_evidence": False, "kind": "manufactured-grouped"}})
    return args


def test_actual_grouped_continuation_fresh_banks_57_conjuncts_and_recovery(root, monkeypatch):
    base = arguments(root / "evidence", task_ids=tuple(f"task-{index}" for index in range(9)),
                     denominators=[2.] * 9, updates=1, boundary_every=1)
    parent_args = grouped_lifecycle_arguments(root, base, "parent")
    parent_root = root / "parent-runtime"
    runner, parent_adapter, provider = finite.build_runner(parent_root, **parent_args)
    live.host_updates(runner, monkeypatch)
    parent_result = runner.run()
    parent = parent_result.states[0]
    terminal = provider.certification(0, PolicyView.from_dict(parent.policy_state), provider.stage, update_index=1)
    assert len(terminal.conjuncts) == 57 and not terminal.passed
    reference = finite._write_json(root / "parent-result.json", {"training_completed": True, "final_state": parent.to_dict()})
    controls = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"]).controls
    continuation = {"parent_result": reference, "parent_checkpoint": reference,
        "parent_configuration": finite._reference(parent_root / "configuration.json"),
        "parent_control_archive": [finite._reference(parent_root / "checkpoints/control_evidence" /
            f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in controls]}
    child_args = grouped_lifecycle_arguments(root, base, "child")
    child_args.update(parameters=parent.policy_state["values"], continuation_checkpoint=parent, continuation_binding=continuation)
    assert not set(parent_args["role_binding"]["grouping"]["bank_families"]).intersection(child_args["role_binding"]["grouping"]["bank_families"])
    child_root = root / "child-runtime"
    child, child_adapter, child_provider = finite.build_runner(child_root, **child_args)
    assert child_adapter.metadata.normalization_hash == parent_adapter.metadata.normalization_hash
    assert child_adapter.adapter_identity() != parent_adapter.adapter_identity()
    assert child_provider.roles.binding_hash() != provider.roles.binding_hash()
    calls = live.host_updates(child, monkeypatch)
    child_result = child.run()
    final = child_result.states[0]
    assert calls == [1] and final.update_index == final.optimizer_state["iteration"] == 2
    history = PermanentPassState.from_dict(final.metadata["permanent_pass_rotation"]).controls
    assert [point.evaluation.request.metadata["scheduled_role_scope"]["stage_id"] for point in history] == ["parent", "parent", "child", "child"]
    assert len(child_provider.certification(0, PolicyView.from_dict(final.policy_state), child_provider.stage, update_index=2).conjuncts) == 57
    reconstructed, adapter, restored_provider = finite.build_runner(child_root, **child_args)
    for instance, names in ((reconstructed.executor, ("step",)), (restored_provider, ("control", "validation", "certification")),
                            (adapter, ("evaluate", "evaluate_values"))):
        for name in names:
            monkeypatch.setattr(instance, name, live.forbidden)
    cached = reconstructed.run()
    assert cached.states[0].to_dict() == final.to_dict() and cached.selection == child_result.selection
    assert cached.evaluations == child_result.evaluations and restored_provider.bridge.provider_calls == restored_provider.bridge.estimator_calls == 0
    changed = json.loads(json.dumps(child_args["role_binding"]["grouping"]))
    changed["cell_probabilities"] = [1. / 9] * 9
    bad = {**child_args, "role_binding": {**child_args["role_binding"], "grouping": changed}}
    with pytest.raises(ValueError, match="continuation role"):
        finite.build_runner(root / "bad-child", **bad)
