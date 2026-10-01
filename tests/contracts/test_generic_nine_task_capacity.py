"""Bounded engineering coverage; selectors correspond to the frozen packet plan."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from scipy.optimize import nnls

from mooneural.training import generic_certified_numerics as certified
from mooneural.training import generic_execution_boundary as boundary
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_model_task_executor import ModelTaskExecutor
from mooneural.training.generic_native_projection import (
    make_native_projected_adam_function,
)
from mooneural.training.generic_permanent_pass import PermanentPassState, _partition
from mooneural.training.generic_replication_policy import (
    ReplicationPermanentPassCoordinator,
)
from mooneural.training.generic_statistical_quality import evaluate_upper_mean_mse
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    ControlEvaluation,
    PolicyView,
    ValidationEvaluation,
    canonical_json,
)

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "docs/experiments/generic-neural-solver/phase9-ez-nine-task-capacity-20260920"
COMPLETION = ROOT / "docs/experiments/generic-neural-solver/phase9-ez-nine-task-capacity-completion-20260920"
TASKS = tuple(f"manufactured.{index}" for index in range(9))


def forbidden(*_args, **_kwargs):
    raise AssertionError("unexpected model or optimizer evaluation")


def arguments(*, count=9, objective=forbidden):
    reference = finite._reference(__file__)
    return {"task_ids": TASKS[:count], "denominators": [1.] * count, "threshold": 1.,
        "parameters": [1.] * 8, "objective": objective,
        "data_binding": {"objective_recipe": {"kind": "manufactured-capacity"}, "objective_source": reference},
        "source_evidence": {"plan": reference, "source": reference},
        "learning_rate": .001, "updates": 2, "boundary_every": 1, "lifecycle_endpoint": "validation-only"}


def control(provider, policy, update, *, permanent=8):
    stage = provider.stage
    request = provider.roles.request_for("control", policy, stage_id=stage.stage_id, arm_id=0,
                                         round_number=update, update_index=update)
    values = {task: .1 if index >= len(provider.adapter.task_ids) - permanent else 2.
              for index, task in enumerate(provider.adapter.task_ids)}
    return ControlEvaluation(values, raw_records={"manufactured": True}, request=request)


def test_host_capacity_partition_shapes_and_limits(tmp_path):
    runner, adapter, provider = finite.build_runner(tmp_path / "runner", **arguments())
    assert adapter.task_ids == TASKS and runner.design.task_ids == TASKS
    assert runner.design.permanent_pass_binding["capacity"]["max_constraints"] == 8
    assert boundary.SOURCE_CONSTRAINT_COUNTS == tuple(range(7))
    assert boundary.REACHABLE_CONSTRAINT_COUNTS == (4, 5, 6)
    for permanent_count, active_count, constraint_count in ((0, 3, 6), (7, 2, 7), (8, 1, 8)):
        partition = _partition(TASKS, TASKS[:permanent_count], 5)
        active = tuple(TASKS.index(task) for task in partition["active"])
        constrained = tuple(TASKS.index(task) for task in partition["constraints"])
        assert (len(active), len(constrained)) == (active_count, constraint_count)
        assert sorted(active + constrained) == list(range(9))
        assert set(TASKS[:permanent_count]) <= set(partition["constraints"])
        assert set(partition["temporary_constraints"]) <= set(partition["constraints"])
        function = boundary.make_partition_function(runner.executor.spec, active, constrained)
        values, rows = np.arange(9, dtype=float), np.arange(72, dtype=float).reshape(9, 8)
        actual = function(values, rows)
        for observed, expected in zip(actual, (rows[list(active)], values[list(active)], rows[list(constrained)]), strict=True):
            np.testing.assert_array_equal(observed, expected)
        assert tuple(boundary.subset_masks(constraint_count).shape) == ((1 << constraint_count) - 1, constraint_count)
    for count in (7, 8):
        with pytest.raises(ValueError):
            boundary.make_projected_adam_function(8, count)
        with pytest.raises(ValueError):
            make_native_projected_adam_function(8, count)
    for constructor in (boundary.subset_masks, lambda count: certified.make_certified_projection_function(8, count)):
        for count in (-1, 9, True):
            with pytest.raises(ValueError):
                constructor(count)
    old, _adapter, _provider = finite.build_runner(tmp_path / "seven", **arguments(count=7))
    assert "capacity" not in old.design.permanent_pass_binding
    assert old.executor.core.update_factory is certified.make_certified_projected_adam_function
    with pytest.raises(ValueError, match="nine tasks"):
        finite.build_runner(tmp_path / "ten", **{**arguments(), "task_ids": (*TASKS, "extra"), "denominators": [1.] * 10})
    request = control(provider, PolicyView.from_dict(runner.states[0].policy_state), 0).request
    assert request.task_ids == TASKS


def test_host_capacity_membership_recontrol_and_zero_call_recovery(tmp_path, monkeypatch):
    output = tmp_path / "runner"
    runner, _adapter, provider = finite.build_runner(output, **arguments())
    seed = runner.states[0]
    calls = []

    def fake_control(arm, policy, stage, round_number):
        calls.append(round_number)
        return control(provider, policy, round_number)

    def fake_validation(arm, policy, stage, *, update_index):
        request = provider.roles.request_for("validation", policy, stage_id=stage.stage_id, arm_id=arm,
                                             round_number=None, update_index=update_index)
        return ValidationEvaluation(dict.fromkeys(TASKS, 2.), dict.fromkeys(TASKS, 1),
            {"scheduled_role_registry_hash": provider.roles.binding_hash(),
             "role_bank_manifest_hash": request.metadata["role_bank_manifest_hash"]}, request=request)

    def fake_step(state, batch, adapter):
        if state.update_index == 1 and not (output / "resume").exists():
            raise RuntimeError("manufactured interruption")
        coordinator = runner.executor.core.coordinator
        assert len(coordinator.read(state).partition["constraints"]) == 8
        return coordinator.commit(state, replace(state, update_index=state.update_index + 1,
            rng_state={**state.rng_state, "next_seed_index": state.update_index + 1})), {"event": "update"}

    monkeypatch.setattr(provider, "control", fake_control)
    monkeypatch.setattr(provider, "validation", fake_validation)
    monkeypatch.setattr(runner.executor, "step", fake_step)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    assert runner.store.recover(0)[0].update_index == 1
    (output / "resume").touch()
    runner, _adapter, provider = finite.build_runner(output, **arguments())
    monkeypatch.setattr(provider, "control", fake_control)
    monkeypatch.setattr(provider, "validation", fake_validation)
    monkeypatch.setattr(runner.executor, "step", fake_step)
    result = runner.run()
    assert calls == [0, 1, 2]
    restored, _adapter, provider = finite.build_runner(output, **arguments())
    monkeypatch.setattr(restored.executor, "step", forbidden)
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    replay = restored.run()
    assert replay.states[0].to_dict() == result.states[0].to_dict()
    assert replay.selection == result.selection and replay.evaluations == result.evaluations
    coordinator = restored.executor.core.coordinator
    state = result.states[0]
    entry, _event = coordinator.stage_entry(state, control(provider, PolicyView.from_dict(state.policy_state), 2),
                                           stage_id=provider.stage.stage_id)
    assert entry.update_index == state.update_index and len(coordinator.read(entry).controls) == 4
    completed = restored.executor.initialize(seed, control(provider, PolicyView.from_dict(seed.policy_state), 0, permanent=9))
    assert coordinator.read(completed).complete
    stopped, event = restored.executor.core.step(completed, None)
    assert stopped.to_dict() == completed.to_dict() and event["update_calls"] == 0


def test_host_capacity_nine_task_multiplicity_and_order():
    from scipy.stats import t as student_t

    records = [{"central_normalized_mse": dict.fromkeys(TASKS, value),
                "conservative_normalized_mse": dict.fromkeys(TASKS, value)} for value in (.1, .2, .3)]
    summary = evaluate_upper_mean_mse(records, anchor_ids=["a", "b", "c"], task_order=TASKS, minimum_anchors=3)
    assert tuple(summary["tasks"]) == TASKS
    assert all(row["critical_value"] == float(student_t.ppf(1. - .05 / 9., 2)) for row in summary["tasks"].values())
    with pytest.raises(ValueError, match="order"):
        evaluate_upper_mean_mse(records, anchor_ids=["a", "b", "c"], task_order=tuple(reversed(TASKS)), minimum_anchors=3)


@lru_cache(None)
def projector(count, dimension=8):
    return certified.make_certified_projection_function(dimension, count)


@lru_cache(None)
def _cached_adam(dimension, count, config):
    return certified.make_certified_projected_adam_function(dimension, count, config)


def adam(dimension, count, config=boundary.DEFAULT_ADAM):
    return _cached_adam(dimension, count, config)


def projected(direction, rows):
    return projector(len(rows), len(direction))(direction, rows, boundary.subset_masks(len(rows)),
                                               boundary.jacobi_schedule(), tf.constant(1e-12, tf.float64))


def timed(label, function, *args):
    started_cpu, started_wall = time.process_time(), time.monotonic()
    print(json.dumps({"event": "call_start", "label": label}), flush=True)
    try:
        return function(*args)
    finally:
        print(json.dumps({"event": "call_end", "label": label,
                          "cpu_seconds": time.process_time() - started_cpu,
                          "wall_seconds": time.monotonic() - started_wall}), flush=True)


def operands(count):
    declaration = json.loads((ROOT / "tests/fixtures/capacity-operands.json").read_text())
    return [np.asarray(declaration[name], dtype=float) for name in ("parameters", "first_moment", "second_moment")] + [
        np.int64(declaration["iteration"]), np.asarray(declaration["direction"]), np.eye(8)[:count], declaration["learning_rate"]]


def assert_projection(direction, rows, analytic=None):
    output = projected(direction, rows)
    assert bool(output[-1]) and float(output[2]) <= 1e-12
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    normalized = rows / np.where(norms > 0, norms, 1.)
    if analytic is None:
        multipliers, _residual = nnls(normalized.T, -direction, maxiter=10000)
        analytic = direction + normalized.T @ multipliers
    np.testing.assert_allclose(output[0], analytic, rtol=2e-9, atol=2e-10)
    assert np.min(normalized @ output[0].numpy(), initial=0.) >= -1e-10
    return output


@pytest.mark.parametrize("count", [pytest.param(7, id="seven_constraint"), pytest.param(8, id="eight_constraint")])
def test_actual_projection(count):
    direction = np.asarray(operands(count)[4])
    expected = direction.copy()
    expected[:count] = np.maximum(expected[:count], 0.)
    assert_projection(direction, np.eye(8)[:count], expected)
    if count == 8:
        rows = np.eye(8)
        rows[1], rows[2], rows[3] = rows[0], -rows[0], np.zeros(8)
        assert_projection(direction, rows)
        assert_projection(direction, np.eye(8) * np.array([1e-12, 1e12, 1e-6, 1e6, 1e-3, 1e3, 1., 2.])[:, None])
        for separation in (2.00001e-6, 1.99999e-6):
            rows = np.zeros((8, 8))
            rows[0, 0], rows[1, 0], rows[1, 1] = 1., -1., separation
            cutoff_direction = np.array([1., -2., 3., 0., 0., 0., 0., 0.])
            assert_projection(cutoff_direction, rows, np.array([0., 0., 3., 0., 0., 0., 0., 0.]))
    invalid = projected(np.full(8, np.nan), np.eye(8)[:count])
    assert not bool(invalid[-1])


@pytest.mark.parametrize("count", [pytest.param(7, id="seven_constraint"), pytest.param(8, id="eight_constraint")])
def test_actual_adam(count):
    args = operands(count)
    args[4] = np.full(8, .01)
    args[1][0] = .4
    update = timed(f"adam_{count}_factory", adam, 8, count)
    output = timed(f"adam_{count}_first_call_trace_compile_execute", update, *args)
    assert bool(output[-1]) and int(output[3]) == int(args[3]) + 1
    before = args[4].copy()
    before[:count] = np.maximum(before[:count], 0.)
    after = -output[5].numpy()
    after[:count] = np.maximum(after[:count], 0.)
    np.testing.assert_array_equal(output[4], before)
    np.testing.assert_allclose(-output[6].numpy(), after, rtol=1e-9, atol=1e-12)
    assert np.any(args[5] @ output[5].numpy() > 1e-10)
    assert np.max(output[8]) <= 1e-10 and np.min(output[7]) >= -1e-10
    assert np.linalg.norm(output[6]) > 0
    assert not np.array_equal(output[1], args[1]) and not np.array_equal(output[2], args[2])
    damaged = list(args)
    damaged[4] = np.full(8, np.nan)
    rejected = timed(f"adam_{count}_rollback_cached_call", update, *damaged)
    assert not bool(rejected[-1])
    for actual, original in zip(rejected[:4], args[:4], strict=True):
        np.testing.assert_array_equal(actual, original)


@tf.function(input_signature=[tf.TensorSpec([8], tf.float64)], autograph=False)
def manufactured_objective(parameters):
    raw = tf.concat([[tf.reduce_sum(tf.square(parameters)) + 2.], tf.square(parameters) + .1], axis=0)
    gradients = tf.concat([2. * parameters[None, :], 2. * tf.linalg.diag(parameters)], axis=0)
    return raw, raw, gradients


def test_eight_constraint_actual_transaction_state_resume(tmp_path):
    runner, adapter, provider = finite.build_runner(tmp_path / "runner", **arguments(objective=manufactured_objective))

    def executor():
        return ModelTaskExecutor(adapter.metadata, runner.executor.spec, provider.roles, threshold=1.,
            model_binding=adapter.adapter_identity(), coordinator_factory=ReplicationPermanentPassCoordinator,
            update_factory=adam, update_binding=certified.numerical_binding(),
            method_factory=certified.make_certified_method_function, method_binding=certified.numerical_binding())

    core = executor()
    args = operands(8)
    args[1][0] = .4
    initial = replace(runner.states[0], optimizer_state={"first_moment": args[1].tolist(), "second_moment": args[2].tolist(),
                      "iteration": int(args[3]), "learning_rate": args[6]})
    state = core.initialize(initial, control(provider, PolicyView.from_dict(initial.policy_state), 0))
    first, event = timed("transaction_first_step", core.step, state,
                         adapter.prepare_batch(PolicyView.from_dict(state.policy_state), 0), adapter)
    assert event["method"] == "cagrad" and event["committed"]
    first, _event = core.core.coordinator.boundary(first, control(provider, PolicyView.from_dict(first.policy_state), 1))
    first, _event = core.core.coordinator.stage_entry(first, control(provider, PolicyView.from_dict(first.policy_state), 1),
                                                    stage_id=provider.stage.stage_id)
    saved = CheckpointState.from_dict(json.loads(canonical_json(first.to_dict())))
    batch = adapter.prepare_batch(PolicyView.from_dict(saved.policy_state), 1)
    expected, _event = timed("transaction_continued_step", core.step, first, batch, adapter)
    resumed, event = timed("transaction_restored_step", executor().step, saved, batch, adapter)
    assert resumed.to_dict() == expected.to_dict() and event["optimizer_iteration"] == int(args[3]) + 2
    assert resumed.rng_state["next_seed_index"] == 2
    assert len(PermanentPassState.from_dict(resumed.metadata["permanent_pass_rotation"]).permanent) == 8
