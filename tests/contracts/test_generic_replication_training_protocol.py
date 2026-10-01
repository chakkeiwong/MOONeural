"""Synthetic orchestration regressions; no economic or native model updates."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mooneural.training import generic_replication_training_protocol as protocol
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    GenericReplicationRunner,
    ReplicationStage,
)
from tests.contracts.test_generic_replication_calibration import fixture_case
from tests.contracts.test_generic_replication_runner import (
    FakeEvaluationProvider,
    FakeExecutor,
    checkpoint,
    design_for,
)






def fixture_runner(directory, *, updates_per_round=300, executor=None, provider=None):
    initial = {arm_id: replace(checkpoint(arm_id), method_state={
        **checkpoint(arm_id).method_state, "source_replica": arm_id,
    }) for arm_id in range(5)}
    stages = (ReplicationStage("population", 0, 1, 2, 0, updates_per_round=updates_per_round,
                               population_size=5, select_after=True),
              ReplicationStage("continuation", 2, 3, 3, 2 * updates_per_round,
                               updates_per_round=updates_per_round, population_size=1))
    design = replace(design_for(initial), stages=tuple(asdict(stage) for stage in stages))
    executor = executor or FakeExecutor(role_hash=design.role_manifest_sha256)
    provider = provider or FakeEvaluationProvider(role_hash=design.role_manifest_sha256)
    return GenericReplicationRunner(
        executor, {arm_id: object() for arm_id in initial}, initial, provider,
        Mock(return_value=object()), AtomicCheckpointStore(directory, "runner-test", design), stages, design=design,
    )


def saved(root, path, payload):
    protocol.publish(path, payload)
    return protocol.reference(root, path)


def previous_results(root, artifact_root, stage, manifest_hash="a" * 64):
    reviews = {}
    for previous in protocol.STAGES[:protocol.STAGES.index(stage)]:
        result_path = artifact_root / "stages" / previous / "result.json"
        if not result_path.exists():
            saved(root, result_path, {"stage": previous, "status": "PASS", "manifest_sha256": manifest_hash,
                                     "evidence": [], "synthetic_review_fixture": True})
        reviews[previous] = saved(root, artifact_root / "reviews" / f"{previous}.json", {
            "schema": protocol.REVIEW_SCHEMA, "stage": previous, "decision": "PASS",
            "reviewer_id": "separate-synthetic-reviewer", "manifest_sha256": manifest_hash,
            "result": protocol.reference(root, result_path),
        })
    return reviews


def arguments(root, stage, factory):
    artifact_root = root / "attempt"
    design = fixture_runner(root / "unused").design
    return {"root": root, "artifact_root": artifact_root,
            "manifest_ref": {"path": "manifest.json", "sha256": "a" * 64},
            "manifest": {}, "design": design, "stage": stage, "build_runner": factory,
            "executor_id": "synthetic-author", "reviews": previous_results(root, artifact_root, stage)}


@pytest.mark.parametrize("count", (1, 10, 300))
def test_bounded_five_parent_updates_keep_original_stages_and_boundary(tmp_path, count):
    runner = fixture_runner(tmp_path / "store")
    original = tuple(asdict(stage) for stage in runner.stages)
    runner.evaluation_provider.validation = Mock(side_effect=AssertionError("no canary selection"))
    runner.evaluation_provider.certification = Mock(side_effect=AssertionError("no canary certification"))
    states, histories = protocol.run_bounded(runner, count)
    assert set(states) == set(range(5))
    assert all(state.update_index == count for state in states.values())
    assert runner.executor.calls == 5 * count
    assert tuple(asdict(stage) for stage in runner.stages) == original
    assert all(sum(event["event"] == "boundary" for event in events) == int(count == 300) for events in histories.values())
    assert all(state.policy_state["values"][0] == arm_id + count * .5 for arm_id, state in states.items())
    assert runner.store.read_selection() is None


@pytest.mark.parametrize("cut", (1, 3, 7))
def test_uneven_committed_interruption_resumes_without_repeating_update(tmp_path, cut, monkeypatch):
    runner = fixture_runner(tmp_path / "store")
    original = runner.store.commit_update
    committed = []

    def interrupted(arm_id, before, after, event):
        result = original(arm_id, before, after, event)
        committed.append((arm_id, after.update_index))
        if len(committed) == cut:
            raise InterruptedError("after durable update")
        return result

    monkeypatch.setattr(runner.store, "commit_update", interrupted)
    with pytest.raises(InterruptedError):
        protocol.run_bounded(runner, 10)
    resumed = fixture_runner(tmp_path / "store")
    states, histories = protocol.run_bounded(resumed, 10)
    assert resumed.executor.calls == 50 - cut
    assert all(state.update_index == 10 for state in states.values())
    assert all(sum(event["event"] == "update" for event in events) == 10 for events in histories.values())
    for arm_id in states:
        checkpoints = sorted((resumed.store.root / "arms" / f"arm-{arm_id}" / "checkpoints").glob("update-*.json"))
        retained = [json.loads(path.read_text()) for path in checkpoints if not path.name.endswith(".complete.json")]
        assert [record["state"]["update_index"] for record in retained] == list(range(1, 11))


def test_all_permanent_does_not_invent_updates_for_a_canary(tmp_path):
    provider = FakeEvaluationProvider(role_hash="4" * 64)
    original = provider.control
    provider.control = lambda *args: replace(original(*args), task_upper_mse={"task-a": 0.0, "task-b": 0.0})
    runner = fixture_runner(tmp_path / "store", provider=provider)
    with pytest.raises(protocol.TrainingProtocolError, match="before its fixed"):
        protocol.run_bounded(runner, 1)
    assert runner.executor.calls == 0


def test_ten_update_orchestration_reloads_and_is_idempotent(tmp_path):
    constructed = []

    def factory(path):
        runner = fixture_runner(path)
        constructed.append(runner)
        return runner

    options = arguments(tmp_path, "ten", factory)
    result = protocol.execute_stage(**options)
    assert result["status"] == "PASS" and result["exact_reload_comparison"]
    assert result["committed_updates"] == 50 and result["reload_split"] == 5
    assert [runner.executor.calls for runner in constructed] == [50, 25, 25]
    assert protocol.execute_stage(**options) == result
    assert len(constructed) == 3
    assert not (options["artifact_root"] / "stages/full").exists()


def test_ten_resume_after_second_half_has_started(tmp_path):
    paths = []
    interrupted = False

    def factory(path):
        nonlocal interrupted
        paths.append(path)
        runner = fixture_runner(path)
        if len(paths) == 3 and not interrupted:
            original = runner.store.commit_update

            def crash(arm_id, before, after, event):
                nonlocal interrupted
                original(arm_id, before, after, event)
                interrupted = True
                raise InterruptedError("second half after durable update")

            runner.store.commit_update = crash
        return runner

    options = arguments(tmp_path, "ten", factory)
    with pytest.raises(InterruptedError):
        protocol.execute_stage(**options)
    failed = options["artifact_root"] / "stages/ten/invocations/0001.result.json"
    frozen = failed.read_bytes()
    assert json.loads(frozen)["status"] == "INTERRUPTED"
    result = protocol.execute_stage(**options)
    assert result["status"] == "PASS" and result["exact_reload_comparison"]
    assert failed.read_bytes() == frozen


@pytest.mark.parametrize("defect", ("missing", "failed", "self", "wrong-result", "wrong-design", "corrupt-evidence"))
def test_unreviewed_or_changed_canary_cannot_construct_runtime(tmp_path, defect):
    options = arguments(tmp_path, "ten", Mock(side_effect=AssertionError("runtime must not construct")))
    review_ref = options["reviews"]["one"]
    review_path = tmp_path / review_ref["path"]
    review = json.loads(review_path.read_text())
    result_path = options["artifact_root"] / "stages/one/result.json"
    if defect == "missing":
        options["reviews"] = {}
    elif defect == "failed":
        review["decision"] = "FAIL"
    elif defect == "self":
        review["reviewer_id"] = "synthetic-author"
    elif defect == "wrong-result":
        review["result"]["sha256"] = "b" * 64
    elif defect == "wrong-design":
        review["manifest_sha256"] = "b" * 64
    else:
        result = json.loads(result_path.read_text())
        result["evidence"] = [{"path": "missing-raw.json", "sha256": "b" * 64}]
        result_path.write_text(json.dumps(result))
        review["result"] = protocol.reference(tmp_path, result_path)
    review_path.write_text(json.dumps(review))
    if defect != "missing":
        options["reviews"]["one"] = protocol.reference(tmp_path, review_path)
    with pytest.raises((protocol.TrainingProtocolError, FileNotFoundError)):
        protocol.execute_stage(**options)
    options["build_runner"].assert_not_called()


@pytest.mark.parametrize("disposition", ("NOT_EQUIVALENT", "ZERO_VARIANCE", "NONFINITE", "missing", "changed-binding"))
def test_full_source_receipt_failure_precedes_runtime_construction(tmp_path, disposition):
    case = fixture_case(tmp_path, disposition if disposition in ("NOT_EQUIVALENT", "ZERO_VARIANCE", "NONFINITE") else "PASS")
    receipt = case.write()
    options = arguments(tmp_path, "full", Mock(side_effect=AssertionError("no full runtime")))
    options["manifest"] = {"source_calibration_binding": case.binding.to_dict()}
    options["source_receipt"] = receipt.reference()
    if disposition == "missing":
        options["source_receipt"] = None
    if disposition == "changed-binding":
        options["manifest"]["source_calibration_binding"]["source_global_update"] = 99000
    with pytest.raises(ValueError):
        protocol.execute_stage(**options)
    options["build_runner"].assert_not_called()


def test_source_pass_checked_again_on_full_resume(tmp_path, monkeypatch):
    from mooneural.training import generic_replication_calibration as calibration

    case = fixture_case(tmp_path)
    receipt = case.write()
    options = arguments(tmp_path, "full", Mock(side_effect=RuntimeError("synthetic runner stop")))
    options["manifest"] = {"source_calibration_binding": case.binding.to_dict()}
    options["source_receipt"] = receipt.reference()
    verified = Mock(wraps=calibration.require_source_calibration_pass)
    monkeypatch.setattr(calibration, "require_source_calibration_pass", verified)
    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="synthetic runner stop"):
            protocol.execute_stage(**options)
    assert verified.call_count == 2
    assert options["build_runner"].call_count == 2


def test_full_mode_uses_original_runner_and_reports_actual_budget(tmp_path):
    case = fixture_case(tmp_path)
    receipt = case.write()
    constructed = []

    def factory(path):
        runner = fixture_runner(path, updates_per_round=2)
        constructed.append(runner)
        return runner

    options = arguments(tmp_path, "full", factory)
    options["design"] = fixture_runner(tmp_path / "unused", updates_per_round=2).design
    options["manifest"] = {"source_calibration_binding": case.binding.to_dict()}
    options["source_receipt"] = receipt.reference()
    result = protocol.execute_stage(**options)
    assert len(constructed) == 1 and type(constructed[0]) is GenericReplicationRunner
    assert result["committed_updates"] == result["aggregate_budget"] == 22
    assert result["selected_budget"] == 6 and result["selection"]["selected_replica"] == 4
    assert not result["all_permanent_early_stop"]
    assert result["source_calibration_receipt"] == receipt.reference()


@pytest.mark.parametrize("passes", (True, False))
def test_one_update_comparisons_bind_every_parent_before_commit(tmp_path, monkeypatch, passes):
    options = arguments(tmp_path, "one", fixture_runner)
    tolerance_ref = saved(tmp_path, tmp_path / "tolerances.json", {"synthetic_tolerances_only": True})
    options["manifest"] = {"one_update_comparator": {}, "tolerances": tolerance_ref}
    calls = []

    def compare(**arguments):
        before, after = arguments["before"], arguments["after"]
        arm_id = before.method_state["source_replica"]
        calls.append(arm_id)
        raw = saved(tmp_path, tmp_path / f"raw-{arm_id}.json", {
            "synthetic_only": True, "before": before.to_dict(), "after": after.to_dict(),
        })
        return {"decision": "PASS" if passes else "FAIL", "checks": dict.fromkeys(protocol.COMPARISON_CHECKS, passes),
                "evidence": [raw]}

    monkeypatch.setattr(protocol, "load_callable", lambda *args: compare)
    if passes:
        result = protocol.execute_stage(**options)
        assert result["committed_updates"] == 5 and calls == list(range(5))
        assert protocol.execute_stage(**options) == result
        assert len(calls) == 5
    else:
        with pytest.raises(protocol.TrainingProtocolError, match="source comparison failed"):
            protocol.execute_stage(**options)
        store = fixture_runner(options["artifact_root"] / "stages/one/store").store
        assert store.recover(0)[0].update_index == 0
        assert json.loads((options["artifact_root"] / "stages/one/comparisons/arm-0.json").read_text())["measurement"]["decision"] == "FAIL"


@pytest.mark.parametrize("condition", ("open-design", "null-entrypoint", "other-entrypoint", "changed-entrypoint"))
def test_registered_program_refusal_before_native_factory(tmp_path, monkeypatch, condition):
    from mooneural.training import generic_program

    entry = saved(tmp_path, tmp_path / "entry.py", {"host_fixture": True})
    design = SimpleNamespace(master_sha256="a" * 64)
    program = SimpleNamespace(plan_hash=design.master_sha256, complete=set() if condition == "open-design" else {"p5r.design"})

    def require_step(*args, **kwargs):
        assert args == ("p5r.training",) and kwargs["execution"] is True
        if condition == "null-entrypoint":
            raise generic_program.ProgramError("numerical entrypoint is not implemented/registered")
        return {"entrypoint": "other.py" if condition == "other-entrypoint" else entry["path"]}

    program.require_step = require_step
    monkeypatch.setattr(generic_program, "Program", lambda root: program)
    if condition == "changed-entrypoint":
        (tmp_path / "entry.py").write_text("changed")
    with pytest.raises(ValueError):
        protocol.require_registered_program(tmp_path, {"entrypoint": entry}, design)






def test_immutable_publication_resumes_after_link_and_refuses_changed_retry(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    original = protocol.os.link

    def link_then_interrupt(source, destination):
        original(source, destination)
        raise InterruptedError("after link")

    monkeypatch.setattr(protocol.os, "link", link_then_interrupt)
    with pytest.raises(InterruptedError):
        protocol.publish(path, {"value": 1})
    monkeypatch.setattr(protocol.os, "link", original)
    protocol.publish(path, {"value": 1})
    with pytest.raises(protocol.TrainingProtocolError, match="overwrite"):
        protocol.publish(path, {"value": 2})














def test_full_permanent_stop_is_preserved_and_completed_run_is_reused(tmp_path):
    case = fixture_case(tmp_path)
    receipt = case.write()
    constructed = []

    def factory(path):
        provider = FakeEvaluationProvider(role_hash="4" * 64)
        original = provider.control

        def control(arm_id, policy, stage, round_number):
            value = original(arm_id, policy, stage, round_number)
            return replace(value, task_upper_mse={"task-a": 0.0, "task-b": 0.0}) if round_number else value

        provider.control = control
        runner = fixture_runner(path, updates_per_round=2, provider=provider)
        constructed.append(runner)
        return runner

    options = arguments(tmp_path, "full", factory)
    options["design"] = fixture_runner(tmp_path / "unused", updates_per_round=2).design
    options["manifest"] = {"source_calibration_binding": case.binding.to_dict()}
    options["source_receipt"] = receipt.reference()
    result = protocol.execute_stage(**options)
    assert result["all_permanent_early_stop"] is True
    assert result["committed_updates"] == 10 and result["aggregate_budget"] == 22
    assert set(result["updates_per_arm"].values()) == {2}
    assert constructed[0].executor.calls == 10
    assert constructed[0].evaluation_provider.control_calls == 10
    assert len(result["evaluations"]) == 6
    assert protocol.execute_stage(**options) == result
    assert len(constructed) == 1
