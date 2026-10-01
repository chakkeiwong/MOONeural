"""Resource exits preserve committed state and the ordinary final lifecycle."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from tests.contracts.test_generic_replication_runner import (
    FakeEvaluationProvider,
    FakeExecutor,
    checkpoint,
    design_for,
)

from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    GenericReplicationRunner,
    ReplicationRunnerError,
    ReplicationStage,
)
from mooneural.training.generic_training_contracts import PolicyView, stable_hash

POLICY = {"training_cpu_limit": 10., "boundary_cpu_reserve": 2.}


def forbidden(*_args, **_kwargs):
    raise AssertionError("completed recovery must not call numerical work or the CPU meter")


class Meter:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return next(self.values)


class Provider(FakeEvaluationProvider):
    def __init__(self, completion_round=None):
        super().__init__(selected=0, role_hash="4" * 64)
        self.calls = []
        self.completion_round = completion_round

    def control(self, arm_id, policy, stage, round_number):
        self.calls.append(("control", round_number))
        result = super().control(arm_id, policy, stage, round_number)
        if round_number == self.completion_round:
            result = replace(result, task_upper_mse=dict.fromkeys(result.task_upper_mse, .01))
        return result

    def validation(self, arm_id, policy, stage, *, update_index):
        self.calls.append(("validation", update_index))
        return super().validation(arm_id, policy, stage, update_index=update_index)

    def certification(self, arm_id, policy, stage, *, update_index):
        self.calls.append(("certification", update_index))
        return super().certification(arm_id, policy, stage, update_index=update_index)


def make_runner(root, *, cpu=None, policy=POLICY, executor=None, provider=None,
                start_update=0, endpoint="certification", supply_meter=True):
    initial = {0: checkpoint(update_index=start_update)}
    stage = ReplicationStage("budget-stage", 0, 1, 3, start_update, updates_per_round=2, select_after=True)
    permanent = {"stage_entries": {}}
    if policy is not None:
        permanent["budget_stop"] = policy
    if endpoint != "certification":
        permanent["lifecycle_endpoint"] = endpoint
    base = design_for({arm: checkpoint(arm, start_update) for arm in range(5)})
    design = replace(base, stages=(asdict(stage),), expected_selected_replica=None, replica_ids=(0,),
                     initial_state_hashes={"0": stable_hash(initial[0].to_dict())}, permanent_pass_binding=permanent)
    executor = executor or FakeExecutor(role_hash=design.role_manifest_sha256)
    provider = provider or Provider()
    meter = cpu if cpu is not None else (lambda: 0.) if policy is not None else None
    return GenericReplicationRunner(
        executor, {0: object()}, initial, provider, lambda *_args: object(),
        AtomicCheckpointStore(root / "run", "runner-test", design), (stage,),
        threshold=.04, design=design, lifecycle_endpoint=endpoint,
        budget_cpu_seconds=meter if supply_meter else None,
    )


def result_payload(result):
    return {"states": {str(arm): state.to_dict() for arm, state in result.states.items()},
            "selection": result.selection, "evaluations": result.evaluations}


def store_hashes(store):
    return {str(path.relative_to(store.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in store.root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("start_update", (0, 66))
@pytest.mark.parametrize("endpoint", ("certification", "validation-only"))
def test_resource_stop_preserves_unresolved_state_and_final_lifecycle(tmp_path, start_update, endpoint):
    meter = Meter([0., 8.])
    current = make_runner(tmp_path, cpu=meter, start_update=start_update, endpoint=endpoint)
    original_method = current.states[0].method_state
    result = current.run()
    state = result.states[0]
    stop = start_update + 2
    assert current.stages[0].stop_update == start_update + 6
    assert state.update_index == state.optimizer_state["iteration"] == stop
    rotation = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"])
    assert not rotation.complete and not rotation.permanent
    assert state.method_state == original_method
    assert state.rng_state == {"seed": 0}
    assert current.executor.calls == 2 and meter.calls == 2
    assert current.evaluation_provider.calls == [
        ("control", 0), ("control", 1), ("validation", stop),
        *([("certification", stop)] if endpoint == "certification" else []),
    ]
    assert result.selection["candidate_union_ids"] == [0]
    assert result.selection["records"][0]["validation_update_index"] == stop
    assert all(record["checkpoint_reload_equal"] for record in result.evaluations)
    if endpoint == "certification":
        assert result.evaluations[-1]["result"]["conjuncts"] == {"retained-miss": False}
    saved, _events = current.store.recover(0)
    assert state.to_dict() == saved.to_dict()
    receipt = json.loads((current.store.root / "budget-stop.json").read_text())
    assert receipt["state_fingerprint"] == stable_hash(state.to_dict())
    assert receipt["planned_stop_update"] == start_update + 6
    assert dict(current.budget_stop_record) == {key: value for key, value in receipt.items() if key != "receipt_hash"}


def test_completed_budget_stop_recovers_in_fresh_process_without_callbacks(tmp_path):
    current = make_runner(tmp_path, cpu=Meter([0., 9.]))
    result = current.run()
    before = store_hashes(current.store)
    code = """import json, sys
from pathlib import Path
sys.path[:] = json.loads(sys.argv[2])
from tests.contracts.test_generic_replication_budget_stop import make_runner, forbidden, result_payload
current = make_runner(Path(sys.argv[1]), cpu=forbidden)
current.executor.step = current.executor.initialize = forbidden
current.batch_factory = forbidden
for name in ('control', 'validation', 'certification'):
    setattr(current.evaluation_provider, name, forbidden)
current.store.commit_selection = forbidden
print(json.dumps(result_payload(current.run())))
"""
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path), json.dumps(sys.path)],
                               capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(completed.stdout) == json.loads(json.dumps(result_payload(result)))
    assert store_hashes(current.store) == before


@pytest.mark.parametrize("policy", (None, POLICY))
def test_default_or_unspent_budget_runs_to_original_cap(tmp_path, policy):
    meter = None if policy is None else Meter([0., 1., 2.])
    current = make_runner(tmp_path, policy=policy, cpu=meter)
    result = current.run()
    assert result.states[0].update_index == 6
    assert current.executor.calls == 6 and current.budget_stop_record is None
    assert not (current.store.root / "budget-stop.json").exists()
    assert [record["role"] for record in result.evaluations] == ["validation", "certification"]


@pytest.mark.parametrize("completion_round", (0, 1))
def test_controller_completion_precedes_budget_check(tmp_path, completion_round):
    meter = forbidden if completion_round == 0 else Meter([0.])
    current = make_runner(tmp_path, cpu=meter, provider=Provider(completion_round))
    result = current.run()
    assert result.states[0].update_index == completion_round * 2
    assert current.executor.core.coordinator.read(result.states[0]).complete
    assert current.budget_stop_record is None


def test_budget_can_stop_at_committed_entry_without_updates(tmp_path):
    current = make_runner(tmp_path, cpu=lambda: 9.)
    result = current.run()
    assert current.executor.calls == 0 and result.states[0].update_index == 0
    assert current.budget_stop_record["checkpoint_kind"] == "initial"
    assert [record["role"] for record in result.evaluations] == ["validation", "certification"]


def test_stage_entry_stop_uses_stage_entry_marker_and_allows_early_validation(tmp_path, monkeypatch):
    current = make_runner(tmp_path, cpu=lambda: 9.)
    stage = current.stages[0]
    states, _events = current._initialize(stage)
    state = states[0]
    directory = current.store._directory(0, "stage_entries")
    marker = directory / "stage_entry-00000000.complete.json"
    marker.touch()
    read_calls = []

    def read(path, *, arm_id=None, kind=None):
        read_calls.append((path, arm_id, kind))
        return SimpleNamespace(state=state)

    monkeypatch.setattr(current.store, "_read", read)
    current._maybe_budget_stop(0, state, stage, "stage_entry")
    receipt = json.loads((current.store.root / "budget-stop.json").read_text())
    assert receipt["checkpoint_kind"] == "stage_entry"
    assert read_calls == [(marker, 0, "stage_entry")]
    policy = PolicyView.from_dict(state.policy_state)
    validation = current.evaluation_provider.validation(0, policy, stage, update_index=state.update_index)
    current._validate_validation_binding(
        0, policy, validation, stage, state,
    )


def test_interrupted_round_finishes_boundary_before_consulting_budget(tmp_path):
    first = make_runner(tmp_path, cpu=Meter([0.]), executor=FakeExecutor(fail_on_call=2, role_hash="4" * 64))
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        first.run()
    saved, _events = first.store.recover(0)
    assert saved.update_index == 1
    meter = Meter([9.])
    resumed = make_runner(tmp_path, cpu=meter)
    result = resumed.run()
    assert result.states[0].update_index == 2 and resumed.executor.calls == 1
    assert meter.calls == 1
    assert resumed.evaluation_provider.calls == [("control", 1), ("validation", 2), ("certification", 2)]


@pytest.mark.parametrize("point", ("before", "after"))
def test_interrupted_stop_publication_resumes_without_new_updates(tmp_path, monkeypatch, point):
    current = make_runner(tmp_path, cpu=Meter([0., 9.]))
    original = current.store._write_publication_json

    def interrupted(path, payload):
        if path.name == "budget-stop.json":
            if point == "after":
                original(path, payload)
            raise RuntimeError("interrupted stop publication")
        return original(path, payload)

    monkeypatch.setattr(current.store, "_write_publication_json", interrupted)
    with pytest.raises(RuntimeError, match="interrupted stop publication"):
        current.run()
    resumed = make_runner(tmp_path, cpu=forbidden if point == "after" else lambda: 9.)
    result = resumed.run()
    assert result.states[0].update_index == 2 and resumed.executor.calls == 0
    assert resumed.evaluation_provider.calls == [("validation", 2), ("certification", 2)]


@pytest.mark.parametrize("field,value", (
    ("state_fingerprint", "0" * 64), ("update_index", 1),
    ("replication_design_sha256", "0" * 64), ("consumed_cpu_seconds", 0.),
    ("checkpoint_kind", "update"), ("planned_stop_update", 2),
))
def test_changed_stop_receipt_refuses_before_callbacks(tmp_path, field, value):
    current = make_runner(tmp_path, cpu=Meter([0., 9.]))
    current.run()
    path = current.store.root / "budget-stop.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    receipt["receipt_hash"] = stable_hash({key: item for key, item in receipt.items() if key != "receipt_hash"})
    path.write_text(json.dumps(receipt))
    resumed = make_runner(tmp_path, cpu=forbidden)
    with pytest.raises(ReplicationRunnerError, match="invalid budget-stop receipt"):
        resumed.run()
    assert resumed.executor.calls == 0 and resumed.evaluation_provider.calls == []


def test_missing_stop_receipt_cannot_restart_training_after_selection(tmp_path):
    current = make_runner(tmp_path, cpu=Meter([0., 9.]))
    current.run()
    (current.store.root / "budget-stop.json").unlink()
    resumed = make_runner(tmp_path, cpu=forbidden)
    with pytest.raises(ReplicationRunnerError, match="missing its budget-stop receipt"):
        resumed.run()
    assert resumed.executor.calls == 0 and resumed.evaluation_provider.calls == []


@pytest.mark.parametrize("used", (-1., float("nan"), float("inf"), True))
def test_invalid_cpu_measurement_refuses_before_updates(tmp_path, used):
    current = make_runner(tmp_path, cpu=lambda: used)
    with pytest.raises(ReplicationRunnerError, match="finite, nonnegative and cumulative"):
        current.run()
    assert current.executor.calls == 0
    assert not (current.store.root / "budget-stop.json").exists()


def test_decreasing_cpu_measurement_cannot_restart_allowance(tmp_path):
    current = make_runner(tmp_path, cpu=Meter([1., 0.]))
    with pytest.raises(ReplicationRunnerError, match="cumulative across attempts"):
        current.run()
    assert current.executor.calls == 2


@pytest.mark.parametrize("policy", (
    {"training_cpu_limit": 0., "boundary_cpu_reserve": 2.},
    {"training_cpu_limit": 10., "boundary_cpu_reserve": 10.},
    {"training_cpu_limit": float("inf"), "boundary_cpu_reserve": 2.},
    {"training_cpu_limit": 10., "boundary_cpu_reserve": True},
))
def test_invalid_frozen_budget_policy_refuses(tmp_path, policy):
    with pytest.raises((ReplicationRunnerError, TypeError, ValueError)):
        make_runner(tmp_path, policy=policy)


def test_callback_and_frozen_policy_must_be_declared_together(tmp_path):
    with pytest.raises(ReplicationRunnerError, match="frozen budget-stop policy"):
        make_runner(tmp_path, policy=None, cpu=lambda: 0.)
    with pytest.raises(ReplicationRunnerError, match="positive frozen CPU limit/reserve"):
        make_runner(tmp_path, supply_meter=False)


def test_finite_builder_binds_budget_and_recovers_actual_stopped_boundary(tmp_path, monkeypatch):
    from tests.contracts.test_generic_finite_objective_runner import (
        arguments,
        quadratic,
        quadratic_values,
    )

    from mooneural.training import generic_finite_objective_runner as finite

    meter = Meter([0., 9.])
    args = arguments(tmp_path, objective=quadratic, objective_values=quadratic_values,
                     updates=4, boundary_every=1, budget_stop=POLICY, budget_cpu_seconds=meter)
    args["source_evidence"]["sources"].append(finite._reference(__file__))
    output = tmp_path / "runtime"
    current, _adapter, _provider = finite.build_runner(output, **args)
    assert meter.calls == 0
    configuration = json.loads((output / "configuration.json").read_text())
    assert configuration["budget_stop"]["policy"] == POLICY
    assert configuration["budget_stop"]["callback"]["source"] == finite._reference(__file__)
    result = current.run()
    assert result.states[0].update_index == 1 and current.stages[0].stop_update == 4
    assert not current.executor.core.coordinator.read(result.states[0]).complete
    before = store_hashes(current.store)
    rebuilt, adapter, provider = finite.build_runner(output, **args)
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    for name in ("evaluate", "evaluate_values", "compute_task_values_and_gradients"):
        monkeypatch.setattr(adapter, name, forbidden)
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    monkeypatch.setattr(rebuilt, "batch_factory", forbidden)
    monkeypatch.setattr(rebuilt, "budget_cpu_seconds", forbidden)
    recovered = rebuilt.run()
    assert result_payload(recovered) == result_payload(result)
    assert store_hashes(rebuilt.store) == before


def test_finite_builder_refuses_unbound_cpu_callback(tmp_path):
    from tests.contracts.test_generic_finite_objective_runner import arguments

    from mooneural.training import generic_finite_objective_runner as finite

    args = arguments(tmp_path, budget_stop=POLICY, budget_cpu_seconds=Meter([0.]))
    with pytest.raises(ValueError, match="must be included in source_evidence"):
        finite.build_runner(tmp_path / "runtime", **args)
