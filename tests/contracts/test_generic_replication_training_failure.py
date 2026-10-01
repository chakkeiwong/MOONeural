"""Host terminal-failure evidence; fake updates never execute an economic model."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from mooneural.training import generic_replication_training_protocol as protocol
from mooneural.training.generic_permanent_pass_executor import PermanentPassUpdateError
from mooneural.training.generic_training_contracts import CheckpointState
from tests.contracts.test_generic_replication_training_protocol import (
    arguments,
    fixture_runner,
)


@pytest.fixture
def update_error():
    return PermanentPassUpdateError


def failed_options(tmp_path, update_error, cut):
    constructed, failures = [], []

    def factory(path):
        runner = fixture_runner(path)
        constructed.append(runner)
        original = runner.executor.step

        def step(state, batch, adapter):
            if runner.executor.calls == cut:
                failed = replace(state, method_state={**state.method_state, "rates": {"fake": 0.5}})
                event = {"event": "update", "committed": False, "before": {"active": ["task-a"]},
                         "after": {"active": ["task-a"]}, "method_events": [
                             {"method": "fake", "rate_before": 1.0, "rate_after": 0.5,
                              "error": "synthetic nonfinite direction"}], "error": "all methods failed"}
                error = update_error("synthetic rejected outer update", failed, event)
                failures.append(error)
                raise error
            return original(state, batch, adapter)

        runner.executor.step = step
        return runner

    return arguments(tmp_path, "ten", factory), constructed, failures


@pytest.mark.parametrize("cut", (0, 7))
def test_terminal_failure_retains_exact_uncommitted_state_and_refuses_resume(tmp_path, update_error, cut):
    options, constructed, failures = failed_options(tmp_path, update_error, cut)
    with pytest.raises(update_error) as caught:
        protocol.execute_stage(**options)
    assert caught.value is failures[0]
    stage = options["artifact_root"] / "stages/ten"
    receipt_path = stage / "invocations/0001.result.json"
    original = receipt_path.read_bytes()
    receipt = json.loads(original)
    assert receipt["status"] == "FAILED"
    assert receipt["terminal_numerical_failure"] is True
    assert receipt["resume_permitted"] is False
    assert receipt["failed_update"]["checkpoint"] == caught.value.checkpoint.to_dict()
    assert receipt["failed_update"]["event"] == caught.value.event
    assert receipt["failed_update"]["diagnostic_only"] is True
    assert receipt["failed_update"]["checkpoint_committed"] is False
    assert receipt["inputs"] == json.loads((stage / "invocations/0001.start.json").read_text())
    failed = CheckpointState.from_dict(receipt["failed_update"]["checkpoint"])
    runner = constructed[0]
    persisted, _events = runner.store.recover(failed.method_state["source_replica"])
    assert failed.policy_state == persisted.policy_state
    assert failed.optimizer_state == persisted.optimizer_state
    assert failed.update_index == persisted.update_index
    assert failed.rng_state == persisted.rng_state
    assert failed.metadata == persisted.metadata
    assert persisted.method_state["rates"]["fake"] == 1.0
    assert failed.method_state["rates"]["fake"] == 0.5
    assert sum(sum(event["event"] == "update" for event in runner.store.recover(arm_id)[1])
               for arm_id in runner.states) == cut
    before = {str(path.relative_to(stage)): path.read_bytes() for path in stage.rglob("*") if path.is_file()}
    options["build_runner"] = Mock(side_effect=AssertionError("terminal failure must not rebuild"))
    with pytest.raises(protocol.TrainingProtocolError, match="terminal.*failure"):
        protocol.execute_stage(**options)
    options["build_runner"].assert_not_called()
    assert receipt_path.read_bytes() == original
    assert {str(path.relative_to(stage)): path.read_bytes() for path in stage.rglob("*") if path.is_file()} == before
    assert not (stage / "result.json").exists()
    assert runner.store.read_selection() is None


@pytest.mark.parametrize("kind", (InterruptedError, TimeoutError))
def test_clean_interruption_preserves_commits_and_resumes(tmp_path, kind):
    constructed = []

    def factory(path):
        runner = fixture_runner(path)
        constructed.append(runner)
        if len(constructed) == 1:
            original = runner.store.commit_update

            def interrupt(arm_id, before, after, event):
                original(arm_id, before, after, event)
                raise kind("synthetic resource interruption after durable commit")

            runner.store.commit_update = interrupt
        return runner

    options = arguments(tmp_path, "ten", factory)
    with pytest.raises(kind):
        protocol.execute_stage(**options)
    path = options["artifact_root"] / "stages/ten/invocations/0001.result.json"
    original = path.read_bytes()
    receipt = json.loads(original)
    assert receipt["status"] == "INTERRUPTED"
    assert not receipt.get("terminal_numerical_failure", False)
    result = protocol.execute_stage(**options)
    assert result["status"] == "PASS" and result["exact_reload_comparison"]
    assert [runner.executor.calls for runner in constructed] == [1, 49, 25, 25]
    assert path.read_bytes() == original


def test_failure_receipt_postpublication_interruption_still_blocks_resume(tmp_path, update_error, monkeypatch):
    options, constructed, failures = failed_options(tmp_path, update_error, 0)
    original = protocol.publish

    def interrupt(path, payload):
        original(path, payload)
        if Path(path).name == "0001.result.json":
            raise InterruptedError("after failed receipt publication")

    monkeypatch.setattr(protocol, "publish", interrupt)
    with pytest.raises(InterruptedError):
        protocol.execute_stage(**options)
    monkeypatch.setattr(protocol, "publish", original)
    path = options["artifact_root"] / "stages/ten/invocations/0001.result.json"
    frozen = path.read_bytes()
    assert json.loads(frozen)["failed_update"]["checkpoint"] == failures[0].checkpoint.to_dict()
    with pytest.raises(protocol.TrainingProtocolError, match="terminal.*failure"):
        protocol.execute_stage(**options)
    assert len(constructed) == 1
    assert path.read_bytes() == frozen


def test_legacy_permanent_pass_failure_refuses_resume_without_inventing_evidence(tmp_path):
    options = arguments(tmp_path, "ten", Mock(side_effect=AssertionError("legacy failure must stop")))
    path = options["artifact_root"] / "stages/ten/invocations/0001.result.json"
    protocol.publish(path, {"status": "FAILED", "error_type": "PermanentPassUpdateError", "message": "historical failure"})
    original = path.read_bytes()
    with pytest.raises(protocol.TrainingProtocolError, match="terminal.*failure"):
        protocol.execute_stage(**options)
    options["build_runner"].assert_not_called()
    assert path.read_bytes() == original
    assert not (path.parent / "0002.start.json").exists()


def test_unrelated_checkpoint_event_attributes_are_not_permanent_pass_failure(tmp_path):
    class UnrelatedError(ValueError):
        pass

    error = UnrelatedError("synthetic unrelated preparation failure")
    error.checkpoint, error.event = object(), object()
    options = arguments(tmp_path, "ten", Mock(side_effect=error))
    with pytest.raises(UnrelatedError):
        protocol.execute_stage(**options)
    path = options["artifact_root"] / "stages/ten/invocations/0001.result.json"
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "FAILED" and "failed_update" not in receipt
    assert not receipt.get("terminal_numerical_failure", False)
