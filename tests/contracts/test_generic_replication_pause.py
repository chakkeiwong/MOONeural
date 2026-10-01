"""Committed-boundary pauses use host fixtures without numerical kernels."""

import pytest
from tests.contracts.test_generic_replication_budget_stop import (
    POLICY,
    Meter,
    Provider,
    forbidden,
    result_payload,
    store_hashes,
)
from tests.contracts.test_generic_replication_budget_stop import (
    make_runner as budget_runner,
)
from tests.contracts.test_generic_replication_runner import runner as multistage_runner

from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_replication_runner import ReplicationRunnerError


def make_runner(root, **changes):
    return budget_runner(root, **{"policy": None, **changes})


def forbid_calls(runner, monkeypatch):
    monkeypatch.setattr(runner.executor, "initialize", forbidden)
    monkeypatch.setattr(runner.executor, "step", forbidden)
    monkeypatch.setattr(runner, "batch_factory", forbidden)
    monkeypatch.setattr(runner, "budget_cpu_seconds", forbidden)
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(runner.evaluation_provider, name, forbidden)


@pytest.mark.parametrize("start_update", (0, 66))
@pytest.mark.parametrize("endpoint", ("certification", "validation-only"))
def test_pause_commits_control_without_final_roles_or_design_changes(tmp_path, start_update, endpoint):
    current = make_runner(tmp_path, start_update=start_update, endpoint=endpoint)
    design = current.design.to_dict()
    initial = current.states[0].to_dict()
    cutoff = start_update + 2
    result = current.run(through_update_index=cutoff)
    state = result.states[0]
    assert state.update_index == state.optimizer_state["iteration"] == cutoff
    assert state.method_state == initial["method_state"]
    assert state.rng_state == initial["rng_state"]
    assert result.selection is None and result.evaluations == ()
    assert current.evaluation_provider.calls == [("control", 0), ("control", 1)]
    assert current.executor.calls == 2
    assert current.design.to_dict() == design
    assert current.stages[0].stop_update == start_update + 6
    assert not (current.store.root / "selection.json").exists()
    assert not (current.store.root / "budget-stop.json").exists()
    assert not list((current.store.root / "evaluations").rglob("*.json"))
    marker = current.store.root / f"arms/arm-0/boundaries/boundary-{cutoff:08d}.complete.json"
    assert marker.is_file()
    recovered, events = current.store.recover(0)
    assert recovered.to_dict() == state.to_dict()
    assert events[-1]["event"] == "boundary"
    rotation = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"])
    assert [point.update_index for point in rotation.controls] == [start_update, cutoff]


@pytest.mark.parametrize("cutoff", (0, 6))
def test_pause_at_initial_or_final_boundary_still_omits_final_roles(tmp_path, cutoff):
    current = make_runner(tmp_path)
    result = current.run(through_update_index=cutoff)
    assert result.states[0].update_index == cutoff
    assert result.selection is None and result.evaluations == ()
    assert current.executor.calls == cutoff
    assert all(role == "control" for role, _update in current.evaluation_provider.calls)
    assert not (current.store.root / "selection.json").exists()


@pytest.mark.parametrize("rebuild", (False, True))
def test_resume_matches_uninterrupted_state_events_selection_and_evaluations(tmp_path, rebuild):
    uninterrupted = make_runner(tmp_path / "uninterrupted")
    expected = uninterrupted.run()
    current = make_runner(tmp_path / "paused")
    paused = current.run(through_update_index=2)
    if rebuild:
        current = make_runner(tmp_path / "paused")
    resumed = current.run()
    assert result_payload(resumed) == result_payload(expected)
    assert paused.events + resumed.events == expected.events
    assert current.store.recover(0)[1] == uninterrupted.store.recover(0)[1]
    assert current.design.binding_hash() == uninterrupted.design.binding_hash()


def test_repeated_pause_at_same_boundary_makes_zero_calls_and_preserves_receipts(tmp_path, monkeypatch):
    current = make_runner(tmp_path)
    original = current.run(through_update_index=2)
    before = store_hashes(current.store)
    for resumed in (current, make_runner(tmp_path)):
        forbid_calls(resumed, monkeypatch)
        result = resumed.run(through_update_index=2)
        assert result_payload(result) == result_payload(original)
        assert result.events == ()
        assert store_hashes(resumed.store) == before


@pytest.mark.parametrize("cutoff", (-1, 1, 7, True, 2.0, "2"))
def test_invalid_or_unaligned_cutoff_is_refused_before_calls(tmp_path, monkeypatch, cutoff):
    current = make_runner(tmp_path)
    before = store_hashes(current.store)
    forbid_calls(current, monkeypatch)
    with pytest.raises(ReplicationRunnerError, match="aligned boundary"):
        current.run(through_update_index=cutoff)
    assert store_hashes(current.store) == before


def test_multistage_pause_is_refused_before_calls(tmp_path, monkeypatch):
    current, _store, _batches = multistage_runner(tmp_path)
    before = store_hashes(current.store)
    forbid_calls(current, monkeypatch)
    with pytest.raises(ReplicationRunnerError, match="single stage"):
        current.run(through_update_index=2)
    assert store_hashes(current.store) == before


def test_cutoff_behind_recovered_state_is_refused_without_calls(tmp_path, monkeypatch):
    current = make_runner(tmp_path)
    current.run(through_update_index=4)
    resumed = make_runner(tmp_path)
    before = store_hashes(resumed.store)
    forbid_calls(resumed, monkeypatch)
    with pytest.raises(ReplicationRunnerError, match="behind recovered state"):
        resumed.run(through_update_index=2)
    assert store_hashes(resumed.store) == before


@pytest.mark.parametrize("completion_round", (0, 1))
def test_early_permanent_completion_pauses_honestly_and_resumes_final_roles(tmp_path, monkeypatch, completion_round):
    current = make_runner(tmp_path, provider=Provider(completion_round=completion_round))
    paused = current.run(through_update_index=4)
    state = paused.states[0]
    assert state.update_index == completion_round * 2
    assert PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"]).complete
    assert paused.selection is None and paused.evaluations == ()
    assert all(role == "control" for role, _update in current.evaluation_provider.calls)
    repeated = make_runner(tmp_path, provider=Provider(completion_round=completion_round))
    forbid_calls(repeated, monkeypatch)
    assert result_payload(repeated.run(through_update_index=4)) == result_payload(paused)
    resumed = make_runner(tmp_path, provider=Provider(completion_round=completion_round))
    result = resumed.run()
    assert result.states[0].to_dict() == state.to_dict()
    assert resumed.executor.calls == 0
    assert resumed.evaluation_provider.calls == [("validation", state.update_index), ("certification", state.update_index)]


@pytest.mark.parametrize("rebuild", (False, True))
def test_missing_control_receipt_is_completed_before_pausing(tmp_path, rebuild):
    provider = Provider()
    provider.fail_on_control = 2
    current = make_runner(tmp_path, provider=provider)
    with pytest.raises(RuntimeError, match="boundary interruption"):
        current.run(through_update_index=2)
    state, events = current.store.recover(0)
    assert state.update_index == 2 and events[-1]["event"] == "update"
    resumed = make_runner(tmp_path) if rebuild else current
    previous_updates = resumed.executor.calls
    previous_controls = len(resumed.evaluation_provider.calls)
    result = resumed.run(through_update_index=2)
    assert result.states[0].update_index == 2
    assert resumed.executor.calls == previous_updates
    assert resumed.evaluation_provider.calls[previous_controls:] == [("control", 1)]
    assert resumed.store.recover(0)[1][-1]["event"] == "boundary"
    assert result.selection is None and result.evaluations == ()


def test_pause_does_not_create_budget_stop_or_meter_repeated_boundary(tmp_path, monkeypatch):
    meter = Meter([0.])
    current = make_runner(tmp_path, policy=POLICY, cpu=meter)
    original = current.run(through_update_index=2)
    assert meter.calls == 1
    assert current.budget_stop_record is None
    assert not (current.store.root / "budget-stop.json").exists()
    before = store_hashes(current.store)
    resumed = make_runner(tmp_path, policy=POLICY, cpu=forbidden)
    forbid_calls(resumed, monkeypatch)
    assert result_payload(resumed.run(through_update_index=2)) == result_payload(original)
    assert store_hashes(resumed.store) == before


def test_real_budget_stop_before_cutoff_is_retained(tmp_path):
    current = make_runner(tmp_path, policy=POLICY, cpu=Meter([0., 9.]))
    paused = current.run(through_update_index=4)
    assert paused.states[0].update_index == 2
    assert paused.selection is None and paused.evaluations == ()
    assert current.budget_stop_record["update_index"] == 2
    assert (current.store.root / "budget-stop.json").is_file()
    resumed = make_runner(tmp_path, policy=POLICY, cpu=forbidden)
    complete = resumed.run()
    assert complete.states[0].to_dict() == paused.states[0].to_dict()
    assert complete.selection is not None
    assert resumed.executor.calls == 0


def test_explicit_none_preserves_ordinary_multistage_lifecycle(tmp_path):
    baseline, _store, _batches = multistage_runner(tmp_path / "baseline")
    explicit, _store, _batches = multistage_runner(tmp_path / "explicit")
    expected = baseline.run()
    actual = explicit.run(through_update_index=None)
    assert result_payload(actual) == result_payload(expected)
    assert actual.events == expected.events
