"""Bounded host lifecycle and intercepted dispatch; no numerical optimizer calls."""

import copy
import math
from dataclasses import replace

import pytest
from tests.contracts.test_generic_checkpoint_continuation import (
    fixture_arguments,
    host_runtime,
    numerical,
    parent_run,
    schedule,
)
from tests.contracts.test_generic_estimator_transition import transition_parent
from tests.contracts.test_generic_finite_objective_runner import forbidden

from mooneural.training import generic_execution_boundary as boundary_module
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training import generic_permanent_pass_executor as executor_module
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_training_contracts import canonical_json, stable_hash

__all__ = ["parent_run", "transition_parent"]


def retain_parent(root, state, runtime):
    reference = finite._write_json(root / "parent-result.json", {
        "training_completed": True, "final_state": state.to_dict()})
    controls = PermanentPassState.from_dict(state.metadata["permanent_pass_rotation"]).controls
    return {"parent_result": reference, "parent_checkpoint": reference,
        "parent_configuration": finite._reference(runtime / "configuration.json"),
        "parent_control_archive": [finite._reference(runtime / "checkpoints/control_evidence" /
            f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in controls]}


@pytest.fixture
def rate_parent(parent_run, tmp_path):
    parent, original, runtime = parent_run
    rates = {method: rate / (2. ** index) for index, (method, rate) in enumerate(parent.method_state["rates"].items())}
    parent = replace(parent, optimizer_state={**parent.optimizer_state, "learning_rate": .0025},
                     method_state={**parent.method_state, "rates": rates})
    binding = retain_parent(tmp_path / "fallback", parent, runtime)
    args = {**original, "updates": 2, "training_schedule": schedule(7),
            "learning_rate": .005, "continuation_checkpoint": parent, "continuation_binding": binding,
            "stage_id": "rate-change-at-five", "learning_rate_transition": {
                "effective_update": 5, "reason": "Manufactured half-rate continuation"}}
    return parent, args, runtime


def without_rates(state):
    value = numerical(state)
    value["optimizer_state"].pop("learning_rate")
    value["method_state"].pop("rates")
    return value


def disable_runtime(runner, adapter, provider, monkeypatch):
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    monkeypatch.setattr(runner.executor, "step", forbidden)
    monkeypatch.setattr(runner, "batch_factory", forbidden)
    monkeypatch.setattr(adapter, "objective", forbidden)
    monkeypatch.setattr(adapter, "objective_values", forbidden)
    monkeypatch.setattr(adapter, "training_objective", forbidden)


def test_transition_preserves_state_fallback_ratios_and_parent(rate_parent, tmp_path, monkeypatch):
    parent, args, parent_root = rate_parent
    before = {str(path): path.read_bytes() for path in parent_root.rglob("*") if path.is_file()}
    runner, _adapter, provider = finite.build_runner(tmp_path / "child", **args)
    initial = runner.states[0]
    transition = runner.design.optimizer_binding["learning_rate_transition"]
    assert without_rates(initial) == without_rates(parent)
    assert initial.optimizer_state["learning_rate"] == .00125
    assert initial.method_state["rates"] == {method: rate * .5 for method, rate in parent.method_state["rates"].items()}
    assert transition["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
    assert transition["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    assert transition["ratio"] == .5 and transition["previous_transition"] is None
    assert transition["old_optimizer"]["adam"] == transition["new_optimizer"]["adam"]
    assert transition["old_optimizer"]["adam"] == runner.design.optimizer_binding["adam"]
    assert transition == runner.checkpoint_continuation["learning_rate_transition"]
    assert canonical_json(transition) == canonical_json(initial.metadata["checkpoint_continuation"]["learning_rate_transition"])
    assert not runner.design.optimizer_binding["fresh_moments"]
    assert initial.metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]
    calls = host_runtime(runner, provider, monkeypatch)
    result = runner.run()
    assert calls["controls"] == [5, 6, 7] and calls["updates"] == [5, 6]
    prior = runner.executor.core.coordinator.read(initial)
    after = runner.executor.core.coordinator.read(result.states[0])
    assert after.controls[:len(prior.controls)] == prior.controls
    assert after.permanent == prior.permanent and calls["partitions"][0] == prior.partition
    assert before == {str(path): path.read_bytes() for path in parent_root.rglob("*") if path.is_file()}


def test_interruption_and_completed_recovery_never_reapply_rate(rate_parent, tmp_path, monkeypatch):
    _parent, args, _parent_root = rate_parent
    uninterrupted, _adapter, provider = finite.build_runner(tmp_path / "expected", **args)
    host_runtime(uninterrupted, provider, monkeypatch)
    expected = uninterrupted.run()
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
    rebuilt, adapter, provider = finite.build_runner(output, **args)
    disable_runtime(rebuilt, adapter, provider, monkeypatch)
    recovered = rebuilt.run()
    assert recovered.states[0].to_dict() == actual.states[0].to_dict()
    assert recovered.selection == actual.selection and recovered.evaluations == actual.evaluations
    changed = {**args, "learning_rate_transition": {**args["learning_rate_transition"], "reason": "different declaration"}}
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        finite.build_runner(output, **changed)


@pytest.mark.parametrize("second_change", [False, True])
def test_later_stage_inherits_or_extends_lineage(rate_parent, tmp_path, monkeypatch, second_change):
    _parent, args, _root = rate_parent
    first = tmp_path / "first"
    runner, _adapter, provider = finite.build_runner(first, **args)
    host_runtime(runner, provider, monkeypatch)
    parent = runner.run().states[0]
    inherited = runner.design.optimizer_binding["learning_rate_transition"]
    later = {**args, "parameters": parent.policy_state["values"], "updates": 1, "training_schedule": schedule(8),
        "continuation_checkpoint": parent, "continuation_binding": retain_parent(tmp_path / "retained", parent, first),
        "stage_id": "later-rate-stage"}
    later.pop("learning_rate_transition")
    if second_change:
        later.update(learning_rate=.0025, learning_rate_transition={"effective_update": 7, "reason": "Second declared reduction"})
    output = tmp_path / "later"
    runner, _adapter, provider = finite.build_runner(output, **later)
    lineage = runner.design.optimizer_binding["learning_rate_transition"]
    if second_change:
        assert lineage["previous_transition"] == inherited
        assert lineage["old_optimizer"]["base_learning_rate"] == .005
        assert without_rates(runner.states[0]) == without_rates(parent)
        assert runner.states[0].method_state["rates"] == {method: rate * .5 for method, rate in parent.method_state["rates"].items()}
    else:
        assert lineage == inherited
        assert numerical(runner.states[0]) == numerical(parent)
    host_runtime(runner, provider, monkeypatch)
    expected = runner.run()
    runner, adapter, provider = finite.build_runner(output, **later)
    disable_runtime(runner, adapter, provider, monkeypatch)
    assert runner.run().states[0].to_dict() == expected.states[0].to_dict()


@pytest.mark.parametrize("fault", ["undeclared", "clock", "reason", "extra", "same", "ineffective", "zero",
    "negative", "nan", "inf", "underflow", "overflow", "moments", "data", "prefix", "beta", "estimator"])
def test_invalid_transition_refused(rate_parent, tmp_path, monkeypatch, fault):
    parent, original, _root = rate_parent
    args = {**original, "learning_rate_transition": dict(original["learning_rate_transition"])}
    if fault == "undeclared":
        args.pop("learning_rate_transition")
    elif fault == "clock":
        args["learning_rate_transition"]["effective_update"] = 4
    elif fault == "reason":
        args["learning_rate_transition"]["reason"] = " "
    elif fault == "extra":
        args["learning_rate_transition"]["reset_moments"] = True
    elif fault in ("same", "ineffective", "zero", "negative", "nan", "inf", "underflow", "overflow"):
        args["learning_rate"] = {"same": .01, "ineffective": math.nextafter(.01, 1.), "zero": 0.,
            "negative": -1., "nan": math.nan, "inf": math.inf, "underflow": 1e-300, "overflow": 1e40}[fault]
    elif fault == "moments":
        args["continuation_checkpoint"] = replace(parent, optimizer_state={**parent.optimizer_state, "first_moment": [0., 0.]})
    elif fault == "data":
        args["data_binding"] = {**args["data_binding"], "fixed_data": [99., 100.]}
    elif fault == "prefix":
        args["training_schedule"] = ({**args["training_schedule"][0], "row_indices": [99]}, *args["training_schedule"][1:])
    elif fault == "beta":
        monkeypatch.setattr(boundary_module, "DEFAULT_ADAM", replace(boundary_module.DEFAULT_ADAM, beta2=.99))
    else:
        args["estimator_transition"] = {"effective_update": 5, "reason": "Combined change", "recipe": {"kind": "new"}}
    with pytest.raises(ValueError):
        finite.build_runner(tmp_path / "rejected", **args)


def test_no_parent_and_absent_option_default(fixture_arguments_root, tmp_path):
    args = fixture_arguments_root
    runner, _adapter, _provider = finite.build_runner(tmp_path / "default", **args)
    assert "learning_rate_transition" not in runner.design.optimizer_binding
    assert runner.states[0].method_state["rates"] == dict.fromkeys(boundary_module.METHODS, args["learning_rate"])
    with pytest.raises(ValueError, match="verified continuation parent"):
        finite.build_runner(tmp_path / "invalid", **args, learning_rate_transition={"effective_update": 0, "reason": "No parent"})


@pytest.fixture
def fixture_arguments_root(tmp_path):
    return fixture_arguments(tmp_path / "evidence", 2)


def test_existing_estimator_lineage_survives_rate_change(transition_parent, tmp_path, monkeypatch):
    _parent, args, _root = transition_parent
    first = tmp_path / "estimator"
    runner, adapter, provider = finite.build_runner(first, **args)
    host_runtime(runner, provider, monkeypatch)
    state = runner.run().states[0]
    estimator_lineage = adapter.training_binding["estimator_transition"]
    later = {**args, "parameters": state.policy_state["values"], "updates": 1, "training_schedule": schedule(8),
        "learning_rate": .005, "stage_id": "rate-after-estimator", "continuation_checkpoint": state,
        "continuation_binding": retain_parent(tmp_path / "retained-estimator", state, first),
        "learning_rate_transition": {"effective_update": 7, "reason": "Rate only; estimator inherited"}}
    later.pop("estimator_transition")
    runner, adapter, _provider = finite.build_runner(tmp_path / "later", **later)
    assert adapter.training_binding["estimator_transition"] == estimator_lineage
    assert without_rates(runner.states[0]) == without_rates(state)


def test_inconsistent_lineage_refused(rate_parent, tmp_path, monkeypatch):
    _parent, args, _root = rate_parent
    first = tmp_path / "first"
    runner, _adapter, provider = finite.build_runner(first, **args)
    host_runtime(runner, provider, monkeypatch)
    parent = runner.run().states[0]
    config = copy.deepcopy(dict(finite.json.loads((first / "configuration.json").read_text())))
    config["checkpoint_continuation"]["learning_rate_transition"]["reason"] = "corrupt lineage"
    with pytest.raises(ValueError, match="lineage differs"):
        finite._rate_transition(parent, config, .005, None)


def test_zero_update_completion_at_transition_clock_does_not_reapply(rate_parent, tmp_path, monkeypatch):
    parent, args, _root = rate_parent
    output = tmp_path / "early-complete"
    runner, _adapter, provider = finite.build_runner(output, **args)
    calls = host_runtime(runner, provider, monkeypatch)
    original_control = provider.control

    def all_pass(*arguments):
        control = original_control(*arguments)
        return replace(control, task_upper_mse=dict.fromkeys(runner.design.task_ids, .1))

    monkeypatch.setattr(provider, "control", all_pass)
    result = runner.run()
    state = result.states[0]
    assert state.update_index == parent.update_index and calls["updates"] == []
    runner.design.validate_checkpoint(state)
    config = finite.json.loads((output / "configuration.json").read_text())
    before = state.to_dict()
    inherited = finite._rate_transition(state, config, .005, None)
    assert canonical_json(inherited) == canonical_json(runner.design.optimizer_binding["learning_rate_transition"])
    assert inherited["effective_update"] == state.update_index
    assert state.to_dict() == before
    rebuilt, adapter, provider = finite.build_runner(output, **args)
    disable_runtime(rebuilt, adapter, provider, monkeypatch)
    assert rebuilt.run().states[0].to_dict() == before


@pytest.mark.parametrize("fallback", [False, True])
def test_actual_dispatch_passes_retained_slots_and_selected_rate(rate_parent, tmp_path, monkeypatch, fallback):
    import tensorflow as tf

    parent, args, _root = rate_parent
    runner, _adapter, provider = finite.build_runner(tmp_path / "dispatch", **args)
    host_runtime(runner, provider, monkeypatch)
    states, _events = runner._initialize(runner.stages[0])
    state = states[0]
    core = runner.executor.core
    observed = {}

    def method_factory(method, _active_count, _parameter_dim):
        def direction(*_arguments):
            observed.setdefault("methods", []).append(method)
            return (None, tf.constant([1., 2.], tf.float64), None, None, None,
                    tf.constant(not fallback or method != "cagrad"))
        return direction

    def optimizer_leaf(*arguments):
        observed["optimizer_arguments"] = arguments
        raise LookupError("intercept before numerical optimizer")

    def boundary_factory(adapter, spec, active, constraints, rates, **options):
        observed["adam"] = options["adam"]
        observed["initial_rates"] = dict(rates)
        instance = object.__new__(boundary_module.CompiledBoundary)
        instance.methods = boundary_module.MethodFallback(len(active), spec.parameter_dim, rates,
            preferred=options["preferred"], method_factory=method_factory)
        instance.evaluate = lambda *_args: (None, None, "values", "rows", tf.constant(True))
        instance.partition = lambda *_args: ("active rows", tf.constant([2.] * len(active), tf.float64), "constraints")
        instance.update = optimizer_leaf
        instance.last_arguments = {}
        return instance

    monkeypatch.setattr(executor_module, "CompiledBoundary", boundary_factory)
    monkeypatch.setattr(boundary_module, "pcgrad_permutations", lambda *_args: None)
    with pytest.raises(executor_module.PermanentPassUpdateError, match="intercept before numerical optimizer"):
        core.step(state, ())
    assert observed["initial_rates"] == state.method_state["rates"]
    assert observed["adam"] == boundary_module.DEFAULT_ADAM
    passed = observed["optimizer_arguments"]
    assert passed[0].numpy().tolist() == list(parent.policy_state["values"])
    assert passed[1].numpy().tolist() == list(parent.optimizer_state["first_moment"])
    assert passed[2].numpy().tolist() == list(parent.optimizer_state["second_moment"])
    assert int(passed[3]) == parent.optimizer_state["iteration"]
    selected = boundary_module.METHODS[1] if fallback else "cagrad"
    assert float(passed[-1]) == state.method_state["rates"][selected]
    assert observed["methods"] == (["cagrad", selected] if fallback else ["cagrad"])
    assert canonical_json(state.rng_state) == canonical_json(parent.rng_state)
