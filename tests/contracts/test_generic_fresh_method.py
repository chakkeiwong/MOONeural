"""Fresh shared-method binding with objective and optimizer kernels forbidden."""

import json
from dataclasses import replace

import pytest
from tests.contracts.test_generic_checkpoint_continuation import numerical, parent_run
from tests.contracts.test_generic_finite_objective_runner import arguments

from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_execution_boundary import METHODS
from mooneural.training.generic_training_contracts import (
    ControlEvaluation,
    PolicyView,
    stable_hash,
)

__all__ = ["parent_run"]


def test_default_none_and_explicit_cagrad_preserve_legacy_bindings(tmp_path):
    args = arguments(tmp_path)
    output = tmp_path / "runtime"
    baseline, _adapter, _provider = finite.build_runner(output, **args)
    configuration = (output / "configuration.json").read_bytes()
    assert "initial_preferred_method" not in json.loads(configuration)
    assert baseline.states[0].method_state["preferred"] == "cagrad"
    for method in (None, "cagrad"):
        rebuilt, _adapter, _provider = finite.build_runner(output, **args, initial_preferred_method=method)
        assert rebuilt.states[0].to_dict() == baseline.states[0].to_dict()
        assert rebuilt.design.binding_hash() == baseline.design.binding_hash()
        assert (output / "configuration.json").read_bytes() == configuration


def test_pcgrad_has_fresh_zero_state_and_configuration_design_bindings(tmp_path):
    args = arguments(tmp_path)
    output = tmp_path / "runtime"
    runner, _adapter, provider = finite.build_runner(output, **args, initial_preferred_method="pcgrad")
    initial = runner.states[0]
    assert initial.update_index == initial.optimizer_state["iteration"] == 0
    assert initial.policy_state["values"] == tuple(args["parameters"])
    assert initial.optimizer_state["first_moment"] == initial.optimizer_state["second_moment"] == [0., 0.]
    assert initial.optimizer_state["learning_rate"] == args["learning_rate"]
    assert initial.method_state == {"preferred": "pcgrad", "rates": dict.fromkeys(METHODS, args["learning_rate"]),
                                    "gradnorm_reset_per_call": True, "state": None}
    assert initial.rng_state == {"pcgrad_seed": 20260722, "next_seed_index": 0}
    assert "permanent_pass_rotation" not in initial.metadata
    assert "checkpoint_continuation" not in initial.metadata
    configuration = json.loads((output / "configuration.json").read_text())
    assert configuration["initial_preferred_method"] == "pcgrad"
    assert runner.design.target_hashes["configuration"] == finite._reference(output / "configuration.json")["sha256"]
    assert runner.design.initial_state_hashes["0"] == stable_hash(initial.to_dict())
    assert runner.design.optimizer_binding["preferred_method"] == "pcgrad"
    assert runner.design.optimizer_binding["fresh_moments"] is True
    runner.design.validate_initial_states({0: initial})
    changed = replace(initial, method_state={**initial.method_state, "preferred": "cagrad"})
    with pytest.raises(ValueError, match="initial state identity"):
        runner.design.validate_initial_states({0: changed})
    stage, = runner.stages
    request = provider.roles.request_for("control", PolicyView.from_dict(initial.policy_state),
        stage_id=stage.stage_id, arm_id=0, round_number=stage.from_round, update_index=0)
    control = ControlEvaluation(dict.fromkeys(args["task_ids"], 2.), request=request)
    coordinator = runner.executor.core.coordinator
    initialized = coordinator.initialize(initial, control)
    membership = coordinator.read(initialized)
    assert membership.update_index == 0 and membership.permanent == ()
    assert len(membership.controls) == 1
    assert initialized.optimizer_state == initial.optimizer_state
    assert initialized.method_state == initial.method_state


@pytest.mark.parametrize("method", ("adam", "PCGrad", "", False, 3, [], {}))
def test_invalid_fresh_method_is_refused_before_writing(tmp_path, method):
    args = arguments(tmp_path)
    output = tmp_path / "runtime"
    with pytest.raises(ValueError, match="initial_preferred_method must name a shared method"):
        finite.build_runner(output, **args, initial_preferred_method=method)
    assert not output.exists()


@pytest.mark.parametrize("method", METHODS)
def test_explicit_initial_method_cannot_override_continuation(parent_run, tmp_path, method):
    parent, args, _parent_root = parent_run
    original = parent.to_dict()
    output = tmp_path / "child"
    with pytest.raises(ValueError, match="only supported for fresh attempts"):
        finite.build_runner(output, **args, initial_preferred_method=method)
    assert not output.exists()
    assert parent.to_dict() == original


def test_ordinary_continuation_retains_parent_state(parent_run, tmp_path):
    parent, args, _parent_root = parent_run
    output = tmp_path / "child"
    runner, _adapter, _provider = finite.build_runner(output, **args)
    assert numerical(runner.states[0]) == numerical(parent)
    assert runner.states[0].metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]
    assert runner.design.optimizer_binding["fresh_moments"] is False
    rebuilt, _adapter, _provider = finite.build_runner(output, **args, initial_preferred_method=None)
    assert rebuilt.states[0].to_dict() == runner.states[0].to_dict()
    assert rebuilt.design.binding_hash() == runner.design.binding_hash()
