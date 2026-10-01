"""Declared preference changes through the real builder, with numerical leaves forbidden."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from tests.contracts.test_generic_checkpoint_continuation import (
    fixture_arguments,
    host_runtime,
    numerical,
    parent_run,
    schedule,
)
from tests.contracts.test_generic_estimator_transition import transition_parent
from tests.contracts.test_generic_learning_rate_transition import (
    disable_runtime,
    rate_parent,
    retain_parent,
)

from mooneural.training import generic_execution_boundary as boundary_module
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training import generic_permanent_pass_executor as executor_module
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    canonical_json,
    stable_hash,
)

__all__ = ["parent_run", "rate_parent", "transition_parent"]


def declaration(clock, old="cagrad", new="pcgrad"):
    return {"effective_update": clock, "old_preferred": old, "new_preferred": new,
            "reason": "Manufactured prospective method-policy experiment"}


@pytest.fixture
def method_parent(rate_parent):
    parent, original, runtime = rate_parent
    args = {key: value for key, value in original.items() if key != "learning_rate_transition"}
    args.update(learning_rate=.01, stage_id="method-change-at-five", method_policy_transition=declaration(5))
    return parent, args, runtime


def without_preference(state):
    value = numerical(state)
    value["method_state"].pop("preferred")
    return value


def test_only_preference_changes_and_actual_history_is_retained(method_parent, tmp_path, monkeypatch):
    parent, args, parent_root = method_parent
    before = {str(path): path.read_bytes() for path in parent_root.rglob("*") if path.is_file()}
    runner, _adapter, provider = finite.build_runner(tmp_path / "child", **args)
    initial = runner.states[0]
    assert "replication_design" not in initial.metadata
    assert without_preference(initial) == without_preference(parent)
    assert initial.method_state["preferred"] == "pcgrad"
    assert initial.metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]
    receipt = runner.design.optimizer_binding["method_policy_transition"]
    assert receipt["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
    assert receipt["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    assert receipt["method_rates"] == parent.method_state["rates"]
    assert receipt["saved_learning_rate"] == parent.optimizer_state["learning_rate"]
    assert receipt["previous_transition"] is None
    assert canonical_json(receipt) == canonical_json(initial.metadata["checkpoint_continuation"]["method_policy_transition"])
    assert receipt == runner.checkpoint_continuation["method_policy_transition"]
    assert not runner.design.optimizer_binding["fresh_moments"]
    prior = runner.executor.core.coordinator.read(initial)
    calls = host_runtime(runner, provider, monkeypatch)
    outcome = runner.run()
    after = runner.executor.core.coordinator.read(outcome.states[0])
    assert calls["updates"] == [5, 6] and calls["controls"] == [5, 6, 7]
    assert calls["partitions"][0] == prior.partition
    assert after.controls[:len(prior.controls)] == prior.controls and after.permanent == prior.permanent
    assert before == {str(path): path.read_bytes() for path in parent_root.rglob("*") if path.is_file()}


def test_interruption_completed_and_fresh_process_recovery(method_parent, tmp_path, monkeypatch):
    _parent, args, _root = method_parent
    expected_runner, _adapter, provider = finite.build_runner(tmp_path / "expected", **args)
    host_runtime(expected_runner, provider, monkeypatch)
    expected = expected_runner.run()
    output = tmp_path / "interrupted"
    runner, _adapter, provider = finite.build_runner(output, **args)
    host_runtime(runner, provider, monkeypatch, fail_at=6)
    with pytest.raises(RuntimeError, match="manufactured interruption"):
        runner.run()
    runner, _adapter, provider = finite.build_runner(output, **args)
    calls = host_runtime(runner, provider, monkeypatch)
    actual = runner.run()
    assert calls["updates"] == [6] and calls["controls"] == [7]
    assert numerical(actual.states[0]) == numerical(expected.states[0])
    runner, adapter, provider = finite.build_runner(output, **args)
    disable_runtime(runner, adapter, provider, monkeypatch)
    recovered = runner.run()
    assert recovered.states[0].to_dict() == actual.states[0].to_dict()
    assert recovered.selection == actual.selection and recovered.evaluations == actual.evaluations
    snapshot = {str(path): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    changed = {**args, "method_policy_transition": {**args["method_policy_transition"], "reason": "Changed"}}
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        finite.build_runner(output, **changed)
    assert snapshot == {str(path): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    payload = {key: value for key, value in args.items()
               if key not in ("objective", "training_objective", "continuation_checkpoint")}
    payload["continuation_checkpoint"] = args["continuation_checkpoint"].to_dict()
    expected_result = {"state": actual.states[0].to_dict(), "selection": actual.selection,
                       "evaluations": actual.evaluations}
    request = finite._write_json(tmp_path / "fresh-recovery.json", {"args": payload, "expected": expected_result})
    program = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from tests.contracts.test_generic_finite_objective_runner import forbidden
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_training_contracts import CheckpointState, canonical_json
payload = json.loads(Path(sys.argv[1]).read_text())
args = payload['args']
args['continuation_checkpoint'] = CheckpointState.from_dict(args['continuation_checkpoint'])
args['training_schedule'] = tuple(args['training_schedule'])
args.update(objective=forbidden, training_objective=forbidden)
runner, adapter, provider = finite.build_runner(Path(sys.argv[2]), **args)
for name in ('control', 'validation', 'certification'):
    setattr(provider, name, forbidden)
runner.executor.step = forbidden
runner.batch_factory = forbidden
adapter.objective = adapter.objective_values = adapter.training_objective = forbidden
outcome = runner.run()
actual = {'state': outcome.states[0].to_dict(), 'selection': outcome.selection, 'evaluations': outcome.evaluations}
assert canonical_json(actual) == canonical_json(payload['expected'])
print('FRESH_COMPLETED_ZERO_CALL_RECOVERY_PASS')
"""
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": os.pathsep.join((
        str(finite.ROOT / "src"), str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")))}
    completed = subprocess.run([sys.executable, "-c", program, str(finite.ROOT / request["path"]), str(output),
                                str(Path(finite.__file__).parents[2])],
                               env=environment, capture_output=True, text=True, timeout=25, check=False)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "FRESH_COMPLETED_ZERO_CALL_RECOVERY_PASS" in completed.stdout


@pytest.mark.parametrize("second_change", [False, True])
def test_later_ordinary_or_explicit_transition_retains_lineage(method_parent, tmp_path, monkeypatch, second_change):
    _parent, args, _root = method_parent
    first = tmp_path / "first"
    runner, _adapter, provider = finite.build_runner(first, **args)
    host_runtime(runner, provider, monkeypatch)
    parent = runner.run().states[0]
    inherited = runner.design.optimizer_binding["method_policy_transition"]
    later = {**args, "parameters": parent.policy_state["values"], "stage_id": "later-method-stage", "updates": 1,
             "training_schedule": schedule(8), "continuation_checkpoint": parent,
             "continuation_binding": retain_parent(tmp_path / "saved", parent, first)}
    later.pop("method_policy_transition")
    if second_change:
        later["method_policy_transition"] = declaration(7, "pcgrad", "mgda")
    runner, _adapter, provider = finite.build_runner(tmp_path / "later", **later)
    receipt = runner.design.optimizer_binding["method_policy_transition"]
    if second_change:
        assert receipt["previous_transition"] == inherited
        assert without_preference(runner.states[0]) == without_preference(parent)
        assert runner.states[0].method_state["preferred"] == "mgda"
    else:
        assert receipt == inherited and numerical(runner.states[0]) == numerical(parent)
    host_runtime(runner, provider, monkeypatch)
    outcome = runner.run()
    runner, adapter, provider = finite.build_runner(tmp_path / "later", **later)
    disable_runtime(runner, adapter, provider, monkeypatch)
    assert runner.run().states[0].to_dict() == outcome.states[0].to_dict()


@pytest.mark.parametrize("fault", ["old", "clock", "bool_clock", "float_clock", "unknown", "same", "reason",
                                  "extra", "missing", "rate", "estimator", "moments", "undeclared"])
def test_bad_or_undeclared_changes_refuse_before_writes(method_parent, tmp_path, fault):
    parent, original, runtime = method_parent
    args = {**original, "method_policy_transition": dict(original["method_policy_transition"])}
    changes = {"old": ("old_preferred", "mgda"), "clock": ("effective_update", 6),
               "bool_clock": ("effective_update", True), "float_clock": ("effective_update", 5.),
               "unknown": ("new_preferred", "unqualified"), "same": ("new_preferred", "cagrad"),
               "reason": ("reason", " "), "extra": ("reset_moments", True)}
    if fault in changes:
        key, value = changes[fault]
        args["method_policy_transition"][key] = value
    elif fault == "missing":
        args["method_policy_transition"].pop("old_preferred")
    elif fault == "rate":
        args["learning_rate_transition"] = {"effective_update": 5, "reason": "Combined"}
    elif fault == "estimator":
        args["estimator_transition"] = {"effective_update": 5, "reason": "Combined", "recipe": {"new": True}}
    elif fault == "moments":
        args["continuation_checkpoint"] = replace(parent, optimizer_state={**parent.optimizer_state, "first_moment": [0., 0.]})
    else:
        changed = replace(parent, method_state={**parent.method_state, "preferred": "pcgrad"})
        args.update(continuation_checkpoint=changed,
                    continuation_binding=retain_parent(tmp_path / "edited", changed, runtime))
        args.pop("method_policy_transition")
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        finite.build_runner(output, **args)
    assert not output.exists()


def test_default_and_no_parent(method_parent, tmp_path):
    parent, args, _root = method_parent
    original = {key: value for key, value in args.items() if key != "method_policy_transition"}
    runner, _adapter, _provider = finite.build_runner(tmp_path / "inherited", **original)
    assert numerical(runner.states[0]) == numerical(parent)
    assert "method_policy_transition" not in runner.design.optimizer_binding
    fresh = fixture_arguments(tmp_path / "evidence", 1)
    runner, _adapter, _provider = finite.build_runner(tmp_path / "fresh", **fresh)
    assert runner.states[0].method_state["preferred"] == "cagrad"
    assert "method_policy_transition" not in runner.design.optimizer_binding
    with pytest.raises(ValueError, match="verified continuation parent"):
        finite.build_runner(tmp_path / "no-parent", **fresh, method_policy_transition=declaration(0))


def test_inherited_rate_and_estimator_lineages_survive(transition_parent, tmp_path, monkeypatch):
    _parent, args, _root = transition_parent
    first = tmp_path / "estimator"
    runner, adapter, provider = finite.build_runner(first, **args)
    host_runtime(runner, provider, monkeypatch)
    parent = runner.run().states[0]
    estimator = adapter.training_binding["estimator_transition"]
    second_args = {key: value for key, value in args.items() if key != "estimator_transition"}
    second_args.update(parameters=parent.policy_state["values"], stage_id="rate-after-estimator", updates=1,
        training_schedule=schedule(8), learning_rate=.005, continuation_checkpoint=parent,
        continuation_binding=retain_parent(tmp_path / "estimator-parent", parent, first),
        learning_rate_transition={"effective_update": 7, "reason": "Existing inherited half rate"})
    second = tmp_path / "rate"
    runner, _adapter, provider = finite.build_runner(second, **second_args)
    host_runtime(runner, provider, monkeypatch)
    parent = runner.run().states[0]
    rate = runner.design.optimizer_binding["learning_rate_transition"]
    third_args = {key: value for key, value in second_args.items() if key != "learning_rate_transition"}
    third_args.update(parameters=parent.policy_state["values"], stage_id="method-after-rate", training_schedule=schedule(9),
        continuation_checkpoint=parent, continuation_binding=retain_parent(tmp_path / "rate-parent", parent, second),
        method_policy_transition=declaration(8))
    runner, adapter, _provider = finite.build_runner(tmp_path / "method", **third_args)
    assert adapter.training_binding["estimator_transition"] == estimator
    assert runner.design.optimizer_binding["learning_rate_transition"] == rate
    assert without_preference(runner.states[0]) == without_preference(parent)


@pytest.mark.parametrize("fault", ["config", "preference", "clock", "methods"])
def test_inconsistent_lineage_refused(method_parent, tmp_path, monkeypatch, fault):
    _parent, args, _root = method_parent
    output = tmp_path / "first"
    runner, _adapter, provider = finite.build_runner(output, **args)
    host_runtime(runner, provider, monkeypatch)
    parent = runner.run().states[0]
    config = json.loads((output / "configuration.json").read_text())
    if fault == "config":
        config["checkpoint_continuation"]["method_policy_transition"]["reason"] = "Changed"
    else:
        metadata = copy.deepcopy(parent.to_dict()["metadata"])
        changes = {"preference": ("new_preferred", "cagrad"), "clock": ("effective_update", 8), "methods": ("methods", [])}
        key, value = changes[fault]
        for receipt in (config["checkpoint_continuation"]["method_policy_transition"],
                        metadata["checkpoint_continuation"]["method_policy_transition"],
                        metadata["replication_design"]["optimizer_binding"]["method_policy_transition"]):
            receipt[key] = value
        parent = replace(parent, metadata=metadata)
    with pytest.raises(ValueError, match="lineage"):
        finite._method_policy_transition(parent, config, None, boundary_module.METHODS)


def test_early_complete_transition_recovery_never_dispatches(method_parent, tmp_path, monkeypatch):
    parent, args, _root = method_parent
    output = tmp_path / "early"
    runner, _adapter, provider = finite.build_runner(output, **args)
    calls = host_runtime(runner, provider, monkeypatch)
    control = provider.control
    monkeypatch.setattr(provider, "control", lambda *arguments: replace(
        control(*arguments), task_upper_mse=dict.fromkeys(runner.design.task_ids, .1)))
    outcome = runner.run()
    state = outcome.states[0]
    assert state.update_index == parent.update_index and calls["updates"] == []
    config = json.loads((output / "configuration.json").read_text())
    receipt = finite._method_policy_transition(state, config, None, boundary_module.METHODS)
    assert receipt["effective_update"] == state.update_index
    runner, adapter, provider = finite.build_runner(output, **args)
    disable_runtime(runner, adapter, provider, monkeypatch)
    assert runner.run().states[0].to_dict() == state.to_dict()


@pytest.mark.parametrize("fallback", [False, True])
def test_source_dispatch_keeps_slots_clock_and_fallback_rates(method_parent, tmp_path, monkeypatch, fallback):
    import tensorflow as tf

    parent, args, _root = method_parent
    runner, _adapter, provider = finite.build_runner(tmp_path / "dispatch", **args)
    host_runtime(runner, provider, monkeypatch)
    states, _events = runner._initialize(runner.stages[0])
    state = states[0]
    core, observed = runner.executor.core, {}

    def method_factory(method, _active_count, _parameter_dim):
        def direction(*_arguments):
            observed.setdefault("methods", []).append(method)
            return (None, tf.constant([1., 2.], tf.float64), None, None, None,
                    tf.constant(not fallback or method != "pcgrad"))
        return direction

    def optimizer_leaf(*arguments):
        observed["arguments"] = arguments
        raise LookupError("intercept before numerical optimizer")

    def boundary_factory(adapter, spec, active, constraints, rates, **options):
        observed["adam"] = options["adam"]
        instance = object.__new__(boundary_module.CompiledBoundary)
        instance.methods = boundary_module.MethodFallback(len(active), spec.parameter_dim, rates,
            preferred=options["preferred"], method_factory=method_factory)
        instance.evaluate = lambda *_args: (None, None, "values", "rows", tf.constant(True))
        instance.partition = lambda *_args: ("active rows", tf.constant([2.] * len(active), tf.float64), "constraints")
        instance.update = optimizer_leaf
        instance.last_arguments = {}
        observed["boundary"] = instance
        return instance

    def permutation(active_count, update_index):
        observed["permutation_clock"] = update_index

    monkeypatch.setattr(executor_module, "CompiledBoundary", boundary_factory)
    monkeypatch.setattr(boundary_module, "pcgrad_permutations", permutation)
    with pytest.raises(executor_module.PermanentPassUpdateError, match="intercept before numerical optimizer"):
        core.step(state, ())
    assert observed["adam"] == boundary_module.DEFAULT_ADAM
    arguments = observed["arguments"]
    for actual, expected in zip(arguments[:3], (parent.policy_state["values"],
            parent.optimizer_state["first_moment"], parent.optimizer_state["second_moment"]), strict=True):
        assert actual.numpy().tolist() == list(expected)
    assert int(arguments[3]) == parent.optimizer_state["iteration"]
    assert observed["permutation_clock"] == parent.update_index
    selected = "mgda" if fallback else "pcgrad"
    assert observed["methods"] == (["pcgrad", "mgda"] if fallback else ["pcgrad"])
    assert float(arguments[-1]) == parent.method_state["rates"][selected]
    expected_rates = dict(parent.method_state["rates"])
    if fallback:
        expected_rates["pcgrad"] *= .5
    assert observed["boundary"].methods.rates == expected_rates
    assert observed["boundary"].methods.preferred == "pcgrad"
    assert canonical_json(state.rng_state) == canonical_json(parent.rng_state)
