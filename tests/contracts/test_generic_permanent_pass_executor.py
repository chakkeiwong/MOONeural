"""Synthetic integration with compiled kernels, not economic conformance."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import tensorflow as tf

from mooneural.training import generic_permanent_pass_executor as executor_module
from mooneural.training.generic_execution_boundary import METHODS
from mooneural.training.generic_permanent_pass_executor import (
    PermanentPassExecutor,
    PermanentPassUpdateError,
)
from tests.support.fixtures import (
    fake_control,
    fixture,
    roles,
)
from mooneural.training.generic_training_contracts import CheckpointState, canonical_json


@pytest.fixture(scope="module")
def runtime():
    return fixture()


def initialize(runtime, values):
    executor, checkpoint, batch = runtime
    control = fake_control(checkpoint, executor.adapter.registry.task_ids, values)
    return executor, executor.initialize(checkpoint, control), batch


@pytest.mark.parametrize("factory,binding,error", (
    (Mock(), None, ValueError),
    (None, {"profile": "test"}, ValueError),
    (3, {"profile": "test"}, TypeError),
    (Mock(), {}, ValueError),
    (Mock(), ["test"], ValueError),
    (Mock(), {"profile": float("nan")}, ValueError),
    (Mock(), {"profile": object()}, TypeError),
))
def test_update_backend_invalid_pair_precedes_executor_construction(factory, binding, error):
    with pytest.raises(error):
        PermanentPassExecutor(None, None, None, threshold=0.04,
                              update_factory=factory, update_binding=binding)


@pytest.mark.parametrize("evaluation_binding", (None, {"profile": "synthetic-evaluation"}))
def test_update_backend_default_hash_matches_pre_edit_executor(evaluation_binding):
    existing, _initial, _batch = fixture()
    snapshot = Path(__file__).resolve().parents[2] / (
        "tests/support/archived_executor.py"
    )
    specification = importlib.util.spec_from_file_location(
        "mooneural.training._pre_native_projection_executor", snapshot
    )
    previous = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(previous)
    arguments = {"threshold": 0.04}
    if evaluation_binding is not None:
        arguments.update(evaluation_factory=Mock(), evaluation_binding=evaluation_binding)
    after = PermanentPassExecutor(existing.adapter, existing.spec, roles(), **arguments)
    before = previous.PermanentPassExecutor(existing.adapter, existing.spec, roles(), **arguments)
    assert before.binding == after.binding
    assert after.update_factory is None
    assert after.commit_guard.experimental_get_tracing_count() == 0


def test_update_backend_profile_snapshot_and_resume_refusal():
    existing, initial, _batch = fixture()
    factory = Mock(side_effect=AssertionError("unexpected numerical factory call"))
    binding = {"profile": "native-diagnostic", "nested": {"order": [4, 5, 6]}}
    custom = PermanentPassExecutor(existing.adapter, existing.spec, roles(), threshold=0.04,
                                  update_factory=factory, update_binding=binding)
    state = custom.initialize(initial, fake_control(initial, existing.adapter.registry.task_ids, [0.2] * 7))
    saved = json.loads(canonical_json(binding))
    binding["nested"]["order"].reverse()
    equivalent = PermanentPassExecutor(existing.adapter, existing.spec, roles(), threshold=0.04,
                                      update_factory=factory, update_binding=saved)
    assert equivalent.binding == custom.binding != existing.binding
    equivalent._validate_numerical_state(state)
    changed = PermanentPassExecutor(existing.adapter, existing.spec, roles(), threshold=0.04,
                                   update_factory=factory, update_binding=binding)
    for incompatible in (existing, changed):
        with pytest.raises(ValueError, match="binding mismatch"):
            incompatible._validate_numerical_state(state)
    factory.assert_not_called()


@pytest.mark.parametrize("unresolved,constraint_count", ((3, 4), (2, 5), (1, 6)))
def test_update_backend_reachable_dispatch_without_numerics(monkeypatch, unresolved, constraint_count):
    existing, initial, _batch = fixture()
    factory = Mock(side_effect=AssertionError("unexpected numerical factory call"))
    custom = PermanentPassExecutor(existing.adapter, existing.spec, roles(), threshold=0.04,
                                  update_factory=factory, update_binding={"profile": "sentinel"})
    state = custom.initialize(initial, fake_control(
        initial, existing.adapter.registry.task_ids, [0.04] * (7 - unresolved) + [0.2] * unresolved
    ))
    boundary = Mock(side_effect=LookupError("stop before kernel construction"))
    monkeypatch.setattr(executor_module, "CompiledBoundary", boundary)
    with pytest.raises(LookupError, match="stop before kernel construction"):
        custom.step(state, ())
    assert len(boundary.call_args.args[3]) == constraint_count
    assert boundary.call_args.kwargs["update_factory"] is factory
    factory.assert_not_called()


def test_update_backend_terminal_state_skips_custom_factory():
    existing, initial, _batch = fixture()
    factory = Mock(side_effect=AssertionError("unexpected numerical factory call"))
    custom = PermanentPassExecutor(existing.adapter, existing.spec, roles(), threshold=0.04,
                                  update_factory=factory, update_binding={"profile": "sentinel"})
    state = custom.initialize(initial, fake_control(initial, existing.adapter.registry.task_ids, [0.04] * 7))
    stopped, event = custom.step(state, ())
    assert stopped is state and event["update_calls"] == 0
    assert custom.kernel_cache == custom.boundaries == {}
    assert custom.commit_guard.experimental_get_tracing_count() == 0
    factory.assert_not_called()


def test_rotating_updates_share_numerical_kernels_without_state_leaks(runtime):
    executor, checkpoint, batch = initialize(runtime, [0.2] * 7)
    original = canonical_json(checkpoint.to_dict())
    partitions = []
    for _step in range(3):
        checkpoint, event = executor.step(checkpoint, batch)
        partitions.append(event["before"]["active"])
        assert len(event["submitted_constraint_dots"]) == 4
        assert all(value >= -1e-10 for value in event["submitted_constraint_dots"])
        assert all(value <= 1e-10 for value in event["actual_constraint_dots"])
    assert len({tuple(partition) for partition in partitions}) == 3
    assert checkpoint.update_index == 27003
    assert checkpoint.optimizer_state["iteration"] == 20
    assert checkpoint.method_state["state"] is None
    assert checkpoint.method_state["preferred"] == "cagrad"
    boundaries = list(executor.boundaries.values())
    assert len({id(boundary.evaluate) for boundary in boundaries}) == 1
    assert len({id(boundary.update) for boundary in boundaries}) == 1
    assert (
        len({id(boundary.methods.functions["cagrad"]) for boundary in boundaries}) == 1
    )
    assert len({id(boundary.methods) for boundary in boundaries}) == len(boundaries)
    assert boundaries[0].evaluate.experimental_get_tracing_count() == 1
    assert boundaries[0].update.experimental_get_tracing_count() == 1
    assert "27000" in original


@pytest.mark.parametrize("unresolved", (1, 2, 3))
def test_one_two_three_unresolved_use_all_remaining_and_preserve_slots(
    runtime, unresolved
):
    executor, parent, batch = initialize(
        runtime, [0.04] * (7 - unresolved) + [0.2] * unresolved
    )
    before = canonical_json(parent.to_dict())
    state, event = executor.step(parent, batch)
    assert len(event["before"]["active"]) == unresolved
    assert len(event["before"]["constraints"]) == 7 - unresolved
    assert event["before"]["temporary_constraints"] == []
    assert state.optimizer_state["iteration"] == 18
    assert (
        state.optimizer_state["opaque_optimizer_metadata"]
        == parent.optimizer_state["opaque_optimizer_metadata"]
    )
    assert (
        state.method_state["opaque_method_metadata"]
        == parent.method_state["opaque_method_metadata"]
    )
    assert state.rng_state["other_rng"] == parent.rng_state["other_rng"]
    assert state.rng_state["next_seed_index"] == 27001
    assert canonical_json(parent.to_dict()) == before


def test_terminal_state_never_constructs_or_calls_a_numerical_boundary():
    executor, initial, batch = fixture()
    state = executor.initialize(
        initial, fake_control(initial, executor.adapter.registry.task_ids, [0.04] * 7)
    )
    stopped, event = executor.step(state, batch)
    assert stopped is state
    assert event["event"] == "stopped_complete"
    assert event["update_calls"] == 0
    assert executor.boundaries == {}
    assert executor.kernel_cache == {}
    assert executor.commit_guard.experimental_get_tracing_count() == 0


def test_boundary_history_and_exact_resume_across_partition_changes(runtime):
    executor, state, batch = initialize(runtime, [0.04] * 4 + [0.2] * 3)
    state, _event = executor.step(state, batch)
    state, boundary = executor.coordinator.boundary(
        state,
        fake_control(
            state, executor.adapter.registry.task_ids, [0.7] * 4 + [0.04, 0.2, 0.2]
        ),
    )
    saved = json.loads(canonical_json(state.to_dict()))
    assert boundary["regressed_permanent_diagnostic"] == list(
        executor.adapter.registry.task_ids[:4]
    )
    expected, expected_event = executor.step(state, batch)
    restored_executor, _initial, _batch = fixture()
    restored = CheckpointState.from_dict(saved)
    actual, actual_event = restored_executor.step(restored, batch)
    assert actual.to_dict() == expected.to_dict()
    assert actual_event == expected_event
    child = executor.coordinator.transfer(actual, attempt_id="child-identity-only")
    parent_next, _event = executor.step(actual, batch)
    child_next, _event = restored_executor.step(child, batch)
    assert (
        replace(child_next, attempt_id=parent_next.attempt_id).to_dict()
        == parent_next.to_dict()
    )


def test_successful_fallback_halves_rate_but_keeps_preferred_and_survives_transfer(
    runtime, monkeypatch
):
    executor, state, batch = initialize(runtime, [0.2] * 7)
    executor.step(state, batch)
    partition = executor.coordinator.read(state).partition
    tasks = executor.adapter.registry.task_ids
    key = (
        tuple(tasks.index(task) for task in partition["active"]),
        tuple(tasks.index(task) for task in partition["constraints"]),
    )
    boundary = executor.boundaries[key]
    source_function = boundary.methods.functions["cagrad"]

    def rejected_method(*arguments):
        result = source_function(*arguments)
        return (*result[:-1], tf.constant(False))

    monkeypatch.setitem(boundary.methods.functions, "cagrad", rejected_method)
    updated, event = executor.step(state, batch)
    assert event["method"] == "pcgrad"
    assert updated.method_state["preferred"] == "cagrad"
    assert (
        updated.method_state["rates"]["cagrad"]
        == state.method_state["rates"]["cagrad"] / 2
    )
    assert (
        updated.optimizer_state["learning_rate"]
        == updated.method_state["rates"]["pcgrad"]
    )
    assert len(updated.method_state["last_events"]) == 2
    updated, _event = executor.coordinator.boundary(
        updated, fake_control(updated, tasks, [0.04] * 5 + [0.2] * 2)
    )
    child = executor.coordinator.transfer(updated, attempt_id="fallback-child")
    next_state, event = executor.step(child, batch)
    assert event["method"] == "cagrad"
    assert next_state.method_state["rates"] == updated.method_state["rates"]
    assert (
        next_state.optimizer_state["learning_rate"]
        == updated.method_state["rates"]["cagrad"]
    )


def test_all_methods_failure_retains_rates_but_not_a_policy_update(runtime):
    executor, state, batch = initialize(runtime, [0.2] * 7)
    before = canonical_json(state.to_dict())
    zero_rows = tf.zeros_like(batch[0]), batch[1], batch[2]
    with pytest.raises(
        PermanentPassUpdateError, match="all compiled methods failed"
    ) as failure:
        executor.step(state, zero_rows)
    failed = failure.value.checkpoint
    assert failed.update_index == state.update_index
    assert failed.policy_state == state.policy_state
    assert failed.optimizer_state == state.optimizer_state
    assert failed.metadata == state.metadata
    assert failed.rng_state == state.rng_state
    assert all(
        failed.method_state["rates"][method] == state.method_state["rates"][method] / 2
        for method in METHODS
    )
    assert failed.method_state["preferred"] == state.method_state["preferred"]
    assert len(failed.method_state["last_events"]) == len(METHODS)
    assert canonical_json(state.to_dict()) == before


def test_source_zero_displacement_veto_is_not_a_paired_round_acceptance():
    executor, initial, _batch = fixture(task_count=4)
    state = executor.initialize(
        initial, fake_control(initial, executor.adapter.registry.task_ids, [0.2] * 4)
    )
    feature = tf.one_hot(0, 8, dtype=tf.float64)
    batch = (
        tf.broadcast_to(feature[None, None, :], [4, 2, 8]),
        tf.constant([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], tf.float64),
        tf.ones([4, 2], tf.float64),
    )
    with pytest.raises(PermanentPassUpdateError, match="zero or tiny") as failure:
        executor.step(state, batch)
    assert failure.value.checkpoint.optimizer_state == state.optimizer_state
    assert failure.value.checkpoint.policy_state == state.policy_state
    assert failure.value.event["committed"] is False


@pytest.mark.parametrize(
    "damage",
    (
        "rng",
        "iteration",
        "slots",
        "gradnorm",
        "rate",
        "learning_rate",
        "binding",
        "preferred",
    ),
)
def test_incomplete_or_incompatible_continuation_state_is_refused(runtime, damage):
    executor, state, batch = initialize(runtime, [0.2] * 7)
    if damage == "rng":
        state = replace(state, rng_state={**state.rng_state, "next_seed_index": 17})
    elif damage == "iteration":
        state = replace(
            state, optimizer_state={**state.optimizer_state, "iteration": True}
        )
    elif damage == "slots":
        state = replace(state, optimizer_state={"iteration": 17})
    elif damage == "gradnorm":
        state = replace(
            state, method_state={**state.method_state, "state": {"weights": [1.0]}}
        )
    elif damage == "rate":
        state = replace(
            state,
            method_state={**state.method_state, "rates": dict.fromkeys(METHODS, -1.0)},
        )
    elif damage == "learning_rate":
        state = replace(
            state, optimizer_state={**state.optimizer_state, "learning_rate": 0.0}
        )
    elif damage == "binding":
        state = replace(
            state,
            metadata={**state.metadata, "permanent_pass_executor_binding": "wrong"},
        )
    elif damage == "preferred":
        state = replace(
            state, method_state={**state.method_state, "preferred": "paired_replay"}
        )
    with pytest.raises(ValueError):
        executor.step(state, batch)
