"""Host fixtures for live role injection and the shared validation-only exit."""

import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests.contracts.test_generic_finite_objective_runner import arguments, forbidden

from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_independent_role_bridge import (
    BankLossRows,
    IndependentBankRoleBridge,
    LossCoordinates,
)
from mooneural.training.generic_replication_runner import (
    GenericReplicationRunner,
    ReplicationRunnerError,
)
from mooneural.training.generic_replication_training_protocol import publish, reference
from mooneural.training.generic_role_banks import RoleBank, RoleBankManifest
from mooneural.training.generic_scheduled_role_banks import (
    ScheduledRoleBankRegistry,
    ScheduledRoleBinding,
)
from mooneural.training.generic_training_contracts import PolicyView, stable_hash


@pytest.fixture
def live_root():
    directory = finite.ROOT / ".pytest_cache"
    directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="live-role-fixture-", dir=directory) as temporary:
        yield Path(temporary)


class FixtureRows:
    def __init__(self, coordinates):
        self.coordinates, self.calls = coordinates, 0
        self.forbidden = False

    def __call__(self, policy, request, bank, *, output_directory):
        if self.forbidden:
            raise AssertionError("role callback repeated")
        self.calls += 1
        normalized = np.array([[2. + abs(policy.values[0]), .1]]) * self.coordinates.threshold
        raw = normalized * np.array(self.coordinates.training)
        return BankLossRows(bank.bank_id, tuple(bank.metadata["row_ids"]), policy.fingerprint(), self.coordinates.task_ids,
                            raw, np.zeros_like(raw), (reference(finite.ROOT, Path(__file__)),
                                *[dict(binding) for binding in bank.metadata["input_references"].values()]))


class RoleFactory:
    def __init__(self, root, coords, fault=None):
        self.root, self.coords, self.fault = root, coords, fault
        self.calls = 0

    def __call__(self, adapter, stage, output):
        self.calls += 1
        assert (output.parent / "training-configuration.json").is_file()
        self.input_fingerprint = adapter.metadata.input_fingerprint
        rows = FixtureRows(self.coords)
        source = reference(finite.ROOT, Path(__file__))
        contract = {"callback": {"module": FixtureRows.__module__, "qualname": "FixtureRows.__call__", "source": source},
                    "source_references": [source], "policy_dimension": adapter.policy_dimension,
                    "policy_metadata": dict(adapter.policy_metadata),
                    "coordinates_hash": self.coords.binding_hash()}
        entries = []
        scopes = [("control", stage.from_round + number, stage.start_update + number * stage.updates_per_round,
                   stage.start_update + number * stage.updates_per_round)
                  for number in range(stage.round_count + 1)] + [("validation", None, stage.start_update, stage.stop_update)]
        for ordinal, (role, number, start, stop) in enumerate(scopes):
            banks = []
            for bank_index in range(2):
                bank_id = f"{role}-{ordinal}-{bank_index}"
                row_id = f"{bank_id}-row"
                input_ref = reference(finite.ROOT, self.root / f"raw-{bank_id}.json")
                banks.append(RoleBank(role, bank_id, 100 + ordinal * 2 + bank_index, "manufactured", "rows-v1",
                    self.coords.binding_hash(), "bank-upper-v1", {"global": input_ref["sha256"]}, (bank_id,),
                    {"row_ids": [row_id], "row_count": 1, "evidence_scope": "fresh-engineering",
                     "provider_binding_hash": stable_hash(contract), "input_references": {"global": input_ref}}))
            manifest = RoleBankManifest(adapter.task_ids, tuple(banks), {role: 2})
            entries.append(ScheduledRoleBinding(stage.stage_id, 0, role, number, start, stop, manifest))
        roles = ScheduledRoleBankRegistry(tuple(entries))
        bridge = IndependentBankRoleBridge(finite.ROOT, roles, self.coords, evidence_scope="fresh-engineering",
            row_provider=rows, provider_binding=contract, cache_directory=output / "cache")
        provider = finite.ScheduledRoleProvider(adapter, stage, bridge)
        binding = {"adapter_input_fingerprint": self.input_fingerprint, "registry_hash": roles.binding_hash(),
                   "provider_source": finite._reference(finite.__file__), "sources": [source], "row_provider": contract}
        if self.fault == "fingerprint":
            binding["adapter_input_fingerprint"] = "0" * 64
        elif self.fault == "registry":
            binding["registry_hash"] = "0" * 64
        elif self.fault == "source":
            binding["provider_source"] = finite._reference(__file__)
        elif self.fault == "adapter":
            adapter.metadata.input_binding["unbound"] = True
        elif self.fault == "inputs":
            (self.root / "raw-inputs.json").write_text("changed during materialization")
        return roles, provider, binding


def injected_arguments(root, *, updates=1, fault=None):
    args = arguments(root, updates=updates, boundary_every=1, threshold=.25)
    coords = LossCoordinates(args["task_ids"], *(tuple(args["denominators"]),) * 4, threshold=.25, minimum_banks=2)
    publish(root / "raw-inputs.json", {"rows": "manufactured independent scope IDs; no statistical-law claim"})
    inputs = {"raw": finite._reference(root / "raw-inputs.json")}
    for ordinal in range(updates + 2):
        role = "control" if ordinal <= updates else "validation"
        for bank_index in range(2):
            bank_id = f"{role}-{ordinal}-{bank_index}"
            path = root / f"raw-{bank_id}.json"
            publish(path, {"bank": bank_id, "row": f"{bank_id}-row"})
            inputs[bank_id] = finite._reference(path)
    factory = RoleFactory(root, coords, fault)
    args.update(role_provider_factory=factory, lifecycle_endpoint="validation-only", role_binding={
        "factory_source": finite._reference(__file__), "inputs": inputs,
        "coordinates": coords.to_dict(), "training_row_ids": ["training-row"],
        "measure": {"population": "manufactured separate role banks", "scientific_admission": False,
                    "production_integrity_evidence": False}})
    return args, factory


def host_updates(runner, monkeypatch, *, fail_on=None):
    calls = []

    def step(state, batch, adapter):
        calls.append(state.update_index)
        assert batch.metadata["policy_fingerprint"] == state.policy_fingerprint
        if len(calls) == fail_on:
            raise RuntimeError("fixture update interrupted")
        policy = replace(PolicyView.from_dict(state.policy_state), values=tuple(value + .1 for value in state.policy_state["values"]))
        coordinator = runner.executor.core.coordinator
        next_rotation = coordinator.read(state).committed_update()
        next_state = replace(state, policy_state=policy.to_dict(), policy_fingerprint=None,
                             optimizer_state={**state.optimizer_state, "iteration": state.update_index + 1},
                             rng_state={**state.rng_state, "next_seed_index": state.update_index + 1})
        return coordinator._store(next_state, next_rotation), {"event": "update"}

    monkeypatch.setattr(runner.executor, "step", step)
    return calls


def test_factory_runs_after_frozen_training_and_binds_real_roles(live_root):
    args, factory = injected_arguments(live_root)
    output = live_root / "runtime"
    runner, adapter, provider = finite.build_runner(output, **args)
    assert factory.calls == 1 and provider.bridge.provider_calls == 0
    assert factory.input_fingerprint == adapter.metadata.input_fingerprint
    training = json.loads((output / "training-configuration.json").read_text())
    config = json.loads((output / "configuration.json").read_text())
    seeds = json.loads((output / "role-seeds.json").read_text())
    assert "role_binding" not in training and "role_manifest_hash" not in training
    assert config["training_configuration"] == finite._reference(output / "training-configuration.json")
    assert config["role_manifest_hash"] == runner.design.role_manifest_sha256 == provider.roles.binding_hash()
    assert stable_hash(seeds) == runner.design.seed_registry_sha256
    assert seeds["roles"][0]["banks"][0]["seed"] == 100
    assert runner.design.permanent_pass_binding["lifecycle_endpoint"] == "validation-only"
    assert runner.design.permanent_pass_binding["role_coordinates"]["threshold"] == .25
    assert adapter.metadata.normalization_binding["measure"] == finite.INJECTED_TRAINING_MEASURE
    assert adapter.metadata.normalizations[0].selection.factor == .5
    assert adapter.metadata.normalizations[1].terminal.factor == .2
    assert runner.states[0].optimizer_state["iteration"] == runner.states[0].update_index == 0
    assert not runner.store.has_initial(0)
    first_identity = adapter.adapter_identity()
    rebuilt, rebuilt_adapter, _provider = finite.build_runner(output, **args)
    assert first_identity == rebuilt_adapter.adapter_identity()
    assert rebuilt.design.binding_hash() == runner.design.binding_hash()


def test_injected_lifecycle_uses_fresh_membership_and_recovers_without_calls(live_root, monkeypatch):
    args, _factory = injected_arguments(live_root)
    output = live_root / "runtime"
    runner, _adapter, provider = finite.build_runner(output, **args)
    calls = host_updates(runner, monkeypatch)
    monkeypatch.setattr(provider, "certification", forbidden)
    result = runner.run()
    state = result.states[0]
    assert calls == [0] and state.update_index == state.optimizer_state["iteration"] == 1
    rotation = runner.executor.core.coordinator.read(state)
    assert rotation.permanent == ("equation.b",)
    assert len(rotation.controls) == 2
    assert [entry["role"] for entry in result.evaluations] == ["validation"]
    assert result.selection["selected_replica"] == 0
    assert provider.bridge.provider_calls == 6
    assert not list((output / "checkpoints/evaluations").rglob("*certification*"))
    rebuilt, adapter, recovered_provider = finite.build_runner(output, **args)
    for method in ("control", "validation", "certification"):
        monkeypatch.setattr(recovered_provider, method, forbidden)
    for method in ("evaluate", "evaluate_values", "compute_task_values_and_gradients"):
        monkeypatch.setattr(adapter, method, forbidden)
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    monkeypatch.setattr(rebuilt, "batch_factory", forbidden)
    recovered = rebuilt.run()
    assert recovered.states[0].to_dict() == state.to_dict()
    assert recovered.selection == result.selection and recovered.evaluations == result.evaluations
    assert recovered_provider.bridge.provider_calls == recovered_provider.bridge.estimator_calls == 0


def test_injected_interruption_resumes_committed_update(live_root, monkeypatch):
    args, _factory = injected_arguments(live_root, updates=2)
    output = live_root / "runtime"
    runner, _adapter, _provider = finite.build_runner(output, **args)
    host_updates(runner, monkeypatch, fail_on=2)
    with pytest.raises(RuntimeError, match="fixture update interrupted"):
        runner.run()
    assert runner.store.recover(0)[0].update_index == 1
    rebuilt, _adapter, provider = finite.build_runner(output, **args)
    calls = host_updates(rebuilt, monkeypatch)
    result = rebuilt.run()
    assert calls == [1] and result.states[0].update_index == 2
    assert provider.bridge.provider_calls == 4
    assert [entry["role"] for entry in result.evaluations] == ["validation"]


@pytest.mark.parametrize("fault", ("coordinate", "threshold", "task_order", "source", "input", "training_rows"))
def test_role_preflight_refuses_before_factory(live_root, fault):
    args, factory = injected_arguments(live_root)
    binding = args["role_binding"]
    if fault == "coordinate":
        binding["coordinates"]["control"][0] *= 2.
    elif fault == "threshold":
        binding["coordinates"]["threshold"] = .5
    elif fault == "task_order":
        binding["coordinates"]["task_ids"].reverse()
    elif fault == "source":
        binding["factory_source"] = finite._reference(finite.__file__)
    elif fault == "input":
        (live_root / "raw-inputs.json").write_text("changed")
    else:
        binding["training_row_ids"] = ["same", "same"]
    with pytest.raises(ValueError):
        finite.build_runner(live_root / "runtime", **args)
    assert factory.calls == 0 and not (live_root / "runtime").exists()


@pytest.mark.parametrize("fault", ("fingerprint", "registry", "source", "adapter", "inputs", "overlap", "undeclared-input"))
def test_factory_mismatch_cannot_publish_final_design(live_root, fault):
    args, factory = injected_arguments(live_root, fault=fault)
    if fault == "overlap":
        args["role_binding"]["training_row_ids"] = ["control-0-0-row"]
    elif fault == "undeclared-input":
        args["role_binding"]["inputs"].pop("control-0-0")
    with pytest.raises(ValueError):
        finite.build_runner(live_root / "runtime", **args)
    assert factory.calls == 1 and not (live_root / "runtime/design.json").exists()


@pytest.mark.parametrize("fault", ("undeclared", "multistage", "unknown", "changed-declaration"))
def test_validation_only_requires_explicit_single_stage_design(live_root, fault):
    args, _factory = injected_arguments(live_root)
    runner, adapter, provider = finite.build_runner(live_root / "runtime", **args)
    design, endpoint, stages = runner.design, "validation-only", runner.stages
    if fault == "undeclared":
        design = replace(design, permanent_pass_binding={"stage_entries": {}})
    elif fault == "multistage":
        later = replace(stages[0], stage_id="later", from_round=1, first_round=2, last_round=2, start_update=1, select_after=False)
        stages = (*stages, later)
        design = replace(design, stages=tuple(asdict(stage) for stage in stages))
    elif fault == "unknown":
        endpoint = "automatic"
    else:
        endpoint = "certification"
    with pytest.raises(ReplicationRunnerError, match="endpoint|one stage"):
        GenericReplicationRunner(runner.executor, {0: adapter}, runner.states, provider, runner.batch_factory,
                                 runner.store, stages, threshold=runner.threshold, design=design, lifecycle_endpoint=endpoint)


def test_validation_only_accepts_provider_without_certification(live_root, monkeypatch):
    args, _factory = injected_arguments(live_root)
    runner, _adapter, provider = finite.build_runner(live_root / "runtime", **args)
    runner = GenericReplicationRunner(runner.executor, runner.adapters, runner.states,
        SimpleNamespace(control=provider.control, validation=provider.validation), runner.batch_factory,
        runner.store, runner.stages, threshold=runner.threshold, design=runner.design, lifecycle_endpoint="validation-only")
    host_updates(runner, monkeypatch)
    assert [item["role"] for item in runner.run().evaluations] == ["validation"]


def test_completed_endpoint_refuses_unexpected_terminal_evidence(live_root, monkeypatch):
    args, _factory = injected_arguments(live_root)
    runner, _adapter, _provider = finite.build_runner(live_root / "runtime", **args)
    host_updates(runner, monkeypatch)
    runner.run()
    publish(runner.store.root / "evaluations/unexpected-certification.json", {"forged": True})
    runner, _adapter, _provider = finite.build_runner(live_root / "runtime", **args)
    with pytest.raises(ReplicationRunnerError, match="inventory mismatch"):
        runner.run()


def test_finite_validation_only_keeps_unused_default_certification_scope(live_root):
    args = arguments(live_root, lifecycle_endpoint="validation-only")
    runner, _adapter, provider = finite.build_runner(live_root / "runtime", **args)
    assert isinstance(provider, finite.FiniteObjectiveProvider)
    assert "certification" in {binding.role for binding in provider.roles.bindings}
    assert runner.lifecycle_endpoint == runner.design.permanent_pass_binding["lifecycle_endpoint"] == "validation-only"
    assert not (runner.store.root / "evaluations").exists()


def test_explicit_default_keeps_existing_configuration_bytes(live_root):
    args = arguments(live_root)
    output = live_root / "runtime"
    default, adapter, provider = finite.build_runner(output, **args)
    configuration = (output / "configuration.json").read_bytes()
    explicit, next_adapter, next_provider = finite.build_runner(output, lifecycle_endpoint="certification", **args)
    assert (output / "configuration.json").read_bytes() == configuration
    assert default.design.binding_hash() == explicit.design.binding_hash()
    assert adapter.adapter_identity() == next_adapter.adapter_identity()
    assert type(provider) is type(next_provider) is finite.FiniteObjectiveProvider
    assert "lifecycle_endpoint" not in default.design.permanent_pass_binding
