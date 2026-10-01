"""Regress independent Phase 3 exception fallback and continuation counterexamples."""

import json

import pytest
import tensorflow as tf

from mooneural.training.generic_execution_boundary import METHODS, MethodFallback
from mooneural.training.generic_permanent_pass_executor import PermanentPassUpdateError
from tests.support.fixtures import fake_control, fixture
from mooneural.training.generic_training_contracts import CheckpointState, canonical_json


def exception(kind):
    if kind == "InvalidArgumentError":
        return tf.errors.InvalidArgumentError(None, None, "injected combiner failure")
    return {"ValueError": ValueError, "RuntimeError": RuntimeError}[kind](
        "injected combiner failure"
    )


@pytest.mark.parametrize("preferred", METHODS)
@pytest.mark.parametrize("kind", ("ValueError", "RuntimeError", "InvalidArgumentError"))
@pytest.mark.parametrize(
    "pattern", ("exception_success", "invalid_exception_success", "exhausted")
)
def test_exception_fallback_matches_source_cyclic_order_rates_and_telemetry(
    preferred, kind, pattern
):
    rates = {method: 0.001 * (index + 1) for index, method in enumerate(METHODS)}
    fallback = MethodFallback(3, 8, rates, preferred=preferred)
    start = METHODS.index(preferred)
    order = [METHODS[(start + offset) % len(METHODS)] for offset in range(len(METHODS))]
    calls = []

    def method_function(method):
        def run(*_arguments):
            calls.append(method)
            position = order.index(method)
            if pattern == "invalid_exception_success" and position == 0:
                return (tf.constant(False),)
            if pattern == "exhausted" or position == (
                1 if pattern == "invalid_exception_success" else 0
            ):
                raise exception(kind)
            return (tf.constant(True),)

        return run

    fallback.functions = {method: method_function(method) for method in METHODS}
    arguments = (tf.ones([3, 8], tf.float64), tf.ones([3], tf.float64), 27000)
    if pattern == "exhausted":
        with pytest.raises(ValueError, match="all compiled methods failed"):
            fallback.choose(*arguments)
        failed_count = 5
        assert calls == order
    else:
        failed_count = 2 if pattern == "invalid_exception_success" else 1
        chosen, _output, rate = fallback.choose(*arguments)
        assert chosen == order[failed_count]
        assert float(rate) == rates[chosen]
        assert calls == order[: failed_count + 1]
    assert fallback.preferred == preferred
    assert [event["method"] for event in fallback.events] == calls
    for index, method in enumerate(order):
        assert fallback.rates[method] == rates[method] * (
            0.5 if index < failed_count else 1.0
        )
    for index, event in enumerate(fallback.events[:failed_count]):
        assert event["accepted"] is False
        assert event["previous_rate"] == rates[order[index]]
        assert event["next_rate"] == rates[order[index]] / 2.0
        assert event["reason"] == (
            "invalid_numerical_direction"
            if pattern == "invalid_exception_success" and index == 0
            else f"{kind}:injected combiner failure"
        )


def test_process_interrupt_is_not_a_method_failure():
    fallback = MethodFallback(1, 8, dict.fromkeys(METHODS, 0.001))

    def interrupted(*_arguments):
        raise KeyboardInterrupt()

    fallback.functions["cagrad"] = interrupted
    with pytest.raises(KeyboardInterrupt):
        fallback.choose(tf.ones([1, 8], tf.float64), tf.ones([1], tf.float64), 0)
    assert fallback.rates == dict.fromkeys(METHODS, 0.001)
    assert fallback.events == []


@pytest.fixture(scope="module")
def runtime():
    executor, initial, batch = fixture()
    state = executor.initialize(
        initial, fake_control(initial, executor.adapter.registry.task_ids, [0.2] * 7)
    )
    executor.step(state, batch)
    policy = executor.coordinator.read(state)
    tasks = executor.adapter.registry.task_ids
    key = (
        tuple(tasks.index(task) for task in policy.partition["active"]),
        tuple(tasks.index(task) for task in policy.partition["constraints"]),
    )
    return executor, state, batch, executor.boundaries[key]


@pytest.mark.parametrize("kind", ("ValueError", "RuntimeError", "InvalidArgumentError"))
def test_exhausted_exception_state_serializes_and_retry_preserves_rates(
    runtime, monkeypatch, kind
):
    executor, parent, batch, boundary = runtime
    original = canonical_json(parent.to_dict())
    originals = dict(boundary.methods.functions)

    def throwing(*_arguments):
        raise exception(kind)

    for method in METHODS:
        monkeypatch.setitem(boundary.methods.functions, method, throwing)
    with pytest.raises(PermanentPassUpdateError) as failure:
        executor.step(parent, batch)
    failed = failure.value.checkpoint
    assert failed.policy_state == parent.policy_state
    assert failed.optimizer_state == parent.optimizer_state
    assert failed.rng_state == parent.rng_state
    assert failed.update_index == parent.update_index
    assert failed.metadata == parent.metadata
    assert all(
        failed.method_state["rates"][method] == parent.method_state["rates"][method] / 2
        for method in METHODS
    )
    assert len(failed.method_state["last_events"]) == 5
    assert all(
        event["reason"] == f"{kind}:injected combiner failure"
        for event in failed.method_state["last_events"]
    )
    restored = CheckpointState.from_dict(json.loads(canonical_json(failed.to_dict())))
    child = executor.coordinator.transfer(restored, attempt_id="exception-retry")
    for method, function in originals.items():
        monkeypatch.setitem(boundary.methods.functions, method, function)
    updated, event = executor.step(child, batch)
    assert event["method"] == parent.method_state["preferred"]
    assert updated.method_state["rates"] == failed.method_state["rates"]
    assert updated.method_state["event_count"] == 6
    assert (
        updated.optimizer_state["iteration"] == parent.optimizer_state["iteration"] + 1
    )
    assert updated.update_index == parent.update_index + 1
    assert canonical_json(parent.to_dict()) == original


@pytest.mark.parametrize("kind", ("ValueError", "RuntimeError", "InvalidArgumentError"))
def test_invalid_then_exception_then_success_persists_both_reductions(
    runtime, monkeypatch, kind
):
    executor, parent, batch, boundary = runtime

    def invalid(*_arguments):
        return (tf.constant(False),)

    def throwing(*_arguments):
        raise exception(kind)

    monkeypatch.setitem(boundary.methods.functions, "cagrad", invalid)
    monkeypatch.setitem(boundary.methods.functions, "pcgrad", throwing)
    updated, event = executor.step(parent, batch)
    assert event["method"] == "mgda"
    assert updated.method_state["preferred"] == "cagrad"
    assert (
        updated.method_state["rates"]["cagrad"]
        == parent.method_state["rates"]["cagrad"] / 2
    )
    assert (
        updated.method_state["rates"]["pcgrad"]
        == parent.method_state["rates"]["pcgrad"] / 2
    )
    assert [event["reason"] for event in updated.method_state["last_events"]] == [
        "invalid_numerical_direction",
        f"{kind}:injected combiner failure",
        "valid_direction",
    ]
    restored = CheckpointState.from_dict(json.loads(canonical_json(updated.to_dict())))
    assert restored.method_state == updated.method_state


@pytest.mark.parametrize("kind", ("ValueError", "RuntimeError", "InvalidArgumentError"))
def test_optimizer_exception_stops_without_losing_prior_rate_evidence(
    runtime, monkeypatch, kind
):
    executor, parent, batch, boundary = runtime
    called = []

    def invalid(*_arguments):
        return (tf.constant(False),)

    def failed_optimizer(*_arguments):
        called.append("optimizer")
        raise exception(kind)

    monkeypatch.setitem(boundary.methods.functions, "cagrad", invalid)
    monkeypatch.setattr(boundary, "update", failed_optimizer)
    with pytest.raises(PermanentPassUpdateError) as failure:
        executor.step(parent, batch)
    assert isinstance(failure.value.__cause__, type(exception(kind)))
    failed = failure.value.checkpoint
    assert called == ["optimizer"]
    assert failed.policy_state == parent.policy_state
    assert failed.optimizer_state == parent.optimizer_state
    assert failed.rng_state == parent.rng_state
    assert failed.update_index == parent.update_index
    assert (
        failed.method_state["rates"]["cagrad"]
        == parent.method_state["rates"]["cagrad"] / 2
    )
    assert (
        failed.method_state["rates"]["pcgrad"] == parent.method_state["rates"]["pcgrad"]
    )
    assert failure.value.event["error"] == f"{kind}:injected combiner failure"
    restored = CheckpointState.from_dict(json.loads(canonical_json(failed.to_dict())))
    assert restored.to_dict() == failed.to_dict()
