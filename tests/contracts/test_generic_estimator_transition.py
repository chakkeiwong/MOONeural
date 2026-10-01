"""Explicit estimator changes preserve the shared continuation state."""

import copy
import importlib.util
from dataclasses import replace

import numpy as np
import pytest
from tests.contracts.test_generic_checkpoint_continuation import (
    fixture_arguments,
    host_runtime,
    numerical,
    schedule,
)
from tests.contracts.test_generic_finite_objective_runner import forbidden

from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_training_contracts import PolicyView, stable_hash


def complete_objective(parameters):
    import tensorflow as tf

    centers = tf.cast(tf.range(7), tf.float64)[:, None]
    residuals = parameters[None, :] - centers
    raw = tf.reduce_mean(residuals ** 2, axis=1)
    return raw, raw, 2. * residuals / parameters.shape[0]


def old_estimator(parameters, update_index):
    return tuple(value * (1. + .1 * (update_index % 2)) for value in complete_objective(parameters))


def new_estimator(parameters, update_index):
    return complete_objective(parameters)


@pytest.fixture
def transition_parent(tmp_path, monkeypatch):
    args = fixture_arguments(tmp_path / "evidence", 5)
    source = finite._reference(__file__)
    args.update(objective=complete_objective, training_objective=old_estimator)
    args["data_binding"].update(objective_source=source, training_objective_source=source)
    args["source_evidence"]["sources"].append(source)
    output = tmp_path / "parent"
    runner, _adapter, provider = finite.build_runner(output, **args)
    host_runtime(runner, provider, monkeypatch)
    parent = runner.run().states[0]
    reference = finite._write_json(tmp_path / "parent-result.json", {"training_completed": True, "final_state": parent.to_dict()})
    controls = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"]).controls
    binding = {"parent_result": reference, "parent_checkpoint": reference,
        "parent_configuration": finite._reference(output / "configuration.json"),
        "parent_control_archive": [finite._reference(output / "checkpoints/control_evidence" /
            f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in controls]}
    child = {**args, "parameters": parent.policy_state["values"], "updates": 2,
        "training_objective": new_estimator, "training_schedule": schedule(7),
        "stage_id": "changed-estimator", "continuation_checkpoint": parent,
        "continuation_binding": binding, "estimator_transition": {
            "effective_update": 5, "reason": "Enumerate the same manufactured population",
            "recipe": {"population_rows": 3, "chunks": [[0], [1, 2]], "weights": [1. / 3., 2. / 3.]}}}
    return parent, child, output


def test_transition_preserves_state_and_dispatches_declared_callback(transition_parent, tmp_path, monkeypatch):
    parent, args, _parent_root = transition_parent
    runner, adapter, provider = finite.build_runner(tmp_path / "child", **args)
    assert numerical(runner.states[0]) == numerical(parent)
    assert not runner.design.optimizer_binding["fresh_moments"]
    binding = adapter.training_binding["estimator_transition"]
    assert binding["effective_update"] == 5
    assert binding["old_callback"]["qualname"] == "old_estimator"
    assert binding["new_callback"]["qualname"] == "new_estimator"
    assert binding["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
    assert runner.checkpoint_continuation["estimator_transition"] == binding
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    with pytest.raises(ValueError, match="before.*estimator transition"):
        adapter.prepare_batch(policy, 4)
    batch = adapter.prepare_batch(policy, 5)
    evaluation = adapter.compute_task_values_and_gradients(policy, batch, 5)
    expected = complete_objective(np.asarray(policy.values))
    np.testing.assert_array_equal(evaluation.values, expected[1])
    np.testing.assert_array_equal(evaluation.rows, expected[2])
    calls = host_runtime(runner, provider, monkeypatch)
    result = runner.run()
    assert calls["updates"] == [5, 6]
    assert calls["controls"] == [5, 6, 7]
    assert result.states[0].optimizer_state["iteration"] == 7
    prior = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"])
    actual = runner.executor.core.coordinator.read(result.states[0])
    assert actual.controls[:len(prior.controls)] == prior.controls
    assert actual.permanent == prior.permanent


def test_changed_stage_interrupted_and_completed_recovery(transition_parent, tmp_path, monkeypatch):
    _parent, args, _parent_root = transition_parent
    runner, _adapter, provider = finite.build_runner(tmp_path / "expected", **args)
    host_runtime(runner, provider, monkeypatch)
    expected = runner.run()
    output = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(output, **args)
    host_runtime(runner, provider, monkeypatch, fail_at=6)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    runner, _adapter, provider = finite.build_runner(output, **args)
    calls = host_runtime(runner, provider, monkeypatch)
    actual = runner.run()
    assert calls["updates"] == [6]
    assert numerical(actual.states[0]) == numerical(expected.states[0])
    runner, adapter, provider = finite.build_runner(output, **args)
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    monkeypatch.setattr(runner.executor, "step", forbidden)
    monkeypatch.setattr(runner, "batch_factory", forbidden)
    monkeypatch.setattr(adapter, "evaluate", forbidden)
    assert runner.run().states[0].to_dict() == actual.states[0].to_dict()
    changed = {**args, "estimator_transition": {**args["estimator_transition"], "reason": "changed binding"}}
    with pytest.raises(ValueError):
        finite.build_runner(output, **changed)


@pytest.mark.parametrize("fault", ["missing", "clock", "empty_recipe", "reason", "field", "prefix", "data", "scale", "full_objective", "backend"])
def test_transition_refusals_preserve_parent(transition_parent, tmp_path, fault):
    parent, original, _parent_root = transition_parent
    args = {**original, "estimator_transition": copy.deepcopy(original["estimator_transition"])}
    before = parent.to_dict()
    if fault == "missing":
        args.pop("estimator_transition")
    elif fault == "clock":
        args["estimator_transition"]["effective_update"] = 4
    elif fault == "empty_recipe":
        args["estimator_transition"]["recipe"] = {}
    elif fault == "reason":
        args["estimator_transition"]["reason"] = ""
    elif fault == "field":
        args["estimator_transition"]["reset_adam"] = True
    elif fault == "prefix":
        args["training_schedule"] = ({**schedule(7)[0], "row_indices": [99]}, *schedule(7)[1:])
    elif fault == "data":
        args["data_binding"] = {**args["data_binding"], "fixed_data": [99.]}
    elif fault == "scale":
        args["denominators"] = [2.] * 7
    elif fault == "full_objective":
        args["objective"] = forbidden
    else:
        args["continuation_checkpoint"] = replace(parent, optimizer_state={**parent.optimizer_state, "iteration": 4})
    with pytest.raises((ValueError, KeyError)):
        finite.build_runner(tmp_path / fault, **args)
    assert parent.to_dict() == before


def test_transition_requires_verified_parent(tmp_path):
    args = fixture_arguments(tmp_path / "evidence", 1)
    args["estimator_transition"] = {"effective_update": 0, "reason": "no parent", "recipe": {"kind": "full"}}
    with pytest.raises(ValueError, match="continuation parent"):
        finite.build_runner(tmp_path / "child", **args)


def test_replacement_source_is_bound_without_exempting_economic_data(transition_parent, tmp_path):
    parent, original, _parent_root = transition_parent
    source = tmp_path / "replacement.py"
    source.write_text("def evaluate(parameters, update_index):\n    return declared(parameters, update_index)\n")
    specification = importlib.util.spec_from_file_location("replacement", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.declared = new_estimator
    reference = finite._reference(source)
    args = {**original, "training_objective": module.evaluate,
        "data_binding": {**original["data_binding"], "training_objective_source": reference},
        "source_evidence": {**original["source_evidence"],
            "sources": [*original["source_evidence"]["sources"], reference]}}
    runner, adapter, _provider = finite.build_runner(tmp_path / "new-source", **args)
    transition = adapter.training_binding["estimator_transition"]
    assert transition["old_callback"]["source"] != reference
    assert transition["new_callback"]["source"] == reference
    assert numerical(runner.states[0]) == numerical(parent)
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    adapter.compute_task_values_and_gradients(policy, adapter.prepare_batch(policy, 5), 5)
    args["data_binding"] = {**args["data_binding"], "fixed_data": [99.]}
    with pytest.raises(ValueError, match="fixed objective data"):
        finite.build_runner(tmp_path / "changed-data", **args)


def test_later_unchanged_stage_inherits_transition_and_recovers(transition_parent, tmp_path, monkeypatch):
    _parent, args, _parent_root = transition_parent
    first_output = tmp_path / "first"
    runner, adapter, provider = finite.build_runner(first_output, **args)
    host_runtime(runner, provider, monkeypatch)
    state = runner.run().states[0]
    prior_transition = adapter.training_binding["estimator_transition"]
    prior = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"])
    reference = finite._write_json(tmp_path / "first-result.json", {"training_completed": True, "final_state": state.to_dict()})
    binding = {"parent_result": reference, "parent_checkpoint": reference,
        "parent_configuration": finite._reference(first_output / "configuration.json"),
        "parent_control_archive": [finite._reference(first_output / "checkpoints/control_evidence" /
            f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in prior.controls]}
    later = {**args, "parameters": state.policy_state["values"], "updates": 1,
        "stage_id": "unchanged-after-transition", "training_schedule": schedule(8),
        "continuation_checkpoint": state, "continuation_binding": binding}
    later.pop("estimator_transition")
    output = tmp_path / "later"
    runner, adapter, provider = finite.build_runner(output, **later)
    assert adapter.training_binding["estimator_transition"] == prior_transition
    assert numerical(runner.states[0]) == numerical(state)
    host_runtime(runner, provider, monkeypatch)
    result = runner.run()
    assert runner.executor.core.coordinator.read(result.states[0]).controls[:len(prior.controls)] == prior.controls
    rebuilt, _adapter, provider = finite.build_runner(output, **later)
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    monkeypatch.setattr(rebuilt, "batch_factory", forbidden)
    assert rebuilt.run().states[0].to_dict() == result.states[0].to_dict()
