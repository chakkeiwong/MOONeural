"""Fixed-objective contracts; shared-engine numerical execution is opt-in.

Scheduled-gradient repair preflight: exercise the existing67 tests plus small
manufactured scheduled objectives, source/schedule refusals, ordinary raw/D,
full-population role measurements and completed replay without callbacks.
Minibatch losses deliberately differ from full losses, exposing accidental
cross-measure parity or retirement. No model calls or scientific claims.
CPU-only focused worker allowance90 seconds including startup and failures.
"""

import ast
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_certified_numerics import (
    make_certified_method_function,
    make_certified_projected_adam_function,
)
from mooneural.training.generic_replication_design import ReplicationDesignBinding
from mooneural.training.generic_replication_policy import (
    ReplicationPermanentPassCoordinator,
)
from mooneural.training.generic_replication_runner import (
    AtomicCheckpointStore,
    GenericReplicationRunner,
)
from mooneural.training.generic_scheduled_role_banks import ScheduledRoleBankRegistry
from mooneural.training.generic_training_contracts import PolicyView, stable_hash


def forbidden(*_args, **_kwargs):
    raise AssertionError("objective or update called during host construction/recovery")


def quadratic(parameters):
    import tensorflow as tf

    differences = parameters[None, :] - tf.constant([[0.], [1.]], tf.float64)
    denominators = tf.constant([2., 5.], tf.float64)
    raw = tf.reduce_sum(differences**2, axis=1)
    return raw, raw / denominators, 2. * differences / denominators[:, None]


def zero_objective(parameters):
    import tensorflow as tf

    return tf.zeros([2], tf.float64), tf.zeros([2], tf.float64), tf.zeros([2, parameters.shape[0]], tf.float64)


def quadratic_values(parameters):
    import tensorflow as tf

    raw = tf.stack([tf.reduce_sum(parameters**2), tf.reduce_sum((parameters - 1.)**2)])
    return raw, raw / tf.constant([2., 5.], tf.float64)


def scheduled_quadratic(parameters, update_index):
    return tuple(value * ((update_index + 1) / 4.) for value in quadratic(parameters))


def arguments(tmp_path, **changes):
    evidence = tmp_path / "plan.md"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("Manufactured finite-objective engineering diagnostic.\n")
    source = finite._reference(__file__)
    args = {
        "task_ids": ("equation.a", "equation.b"), "denominators": [2., 5.], "threshold": 1.,
        "parameters": [3., 4.], "objective": forbidden,
        "data_binding": {"objective_recipe": {"kind": "manufactured-quadratic", "centers": [0., 1.]},
                         "objective_source": source, "fixed_data": [2., 5.], "profile": "finite-v1"},
        "source_evidence": {"plan": finite._reference(evidence), "sources": [source]},
        "learning_rate": .01, "updates": 4, "boundary_every": 2,
    }
    args.update(changes)
    return args


def scheduled_arguments(tmp_path, **changes):
    args = arguments(tmp_path, updates=2, boundary_every=1, objective_values=quadratic_values)
    args["training_objective"] = scheduled_quadratic
    args.update(changes)
    args["data_binding"]["training_objective_source"] = finite._reference(__file__)
    args.setdefault("training_schedule", tuple({
        "update_index": index, "row_indices": [index, index + 2],
        "seed": 2026091901, "sampling": "caller-declared fixed test schedule",
    } for index in range(args["updates"])))
    return args


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    output = tmp_path_factory.mktemp("finite-objective")
    args = arguments(output)
    runner, adapter, provider = finite.build_runner(output / "runtime", **args)
    return output / "runtime", args, runner, adapter, provider


def test_constructor_uses_common_engine_without_evaluation(built):
    _output, args, runner, adapter, provider = built
    assert type(runner) is GenericReplicationRunner
    assert type(runner.store) is AtomicCheckpointStore
    assert runner.adapters == {0: adapter}
    assert runner.evaluation_provider is provider
    assert type(runner.executor.core.coordinator) is ReplicationPermanentPassCoordinator
    assert runner.executor.core.method_factory is make_certified_method_function
    assert runner.executor.core.update_factory is make_certified_projected_adam_function
    assert runner.threshold == runner.design.threshold == args["threshold"]
    assert runner.executor.spec.task_count == 2
    assert runner.executor.spec.hard_check_count == 0
    assert not runner.store.has_initial(0)


def test_boundaries_and_early_exit_scopes_are_exact(built):
    _output, _args, runner, _adapter, provider = built
    assert isinstance(provider.roles, ScheduledRoleBankRegistry)
    stage, = runner.stages
    assert (stage.round_count, stage.updates_per_round, stage.total_updates, stage.population_size) == (2, 2, 4, 1)
    assert stage.select_after is True
    controls = [binding for binding in provider.roles.bindings if binding.role == "control"]
    assert [(binding.round_number, binding.min_update, binding.max_update) for binding in controls] == [
        (0, 0, 0), (1, 2, 2), (2, 4, 4)]
    assert [(binding.role, binding.min_update, binding.max_update) for binding in provider.roles.bindings
            if binding.role != "control"] == [("validation", 0, 4), ("certification", 0, 4)]


def test_all_role_descriptors_admit_same_finite_training_population(built):
    _output, _args, runner, adapter, provider = built
    underlying, jobs = set(), set()
    for binding in provider.roles.bindings:
        bank, = binding.manifest.banks
        descriptor = json.loads(finite._checked(bank.metadata["global_archive"]).read_text())
        assert descriptor["input_fingerprint"] == adapter.metadata.input_fingerprint
        for key in ("held_out", "independent_roles", "scientific_admission", "production_integrity_evidence"):
            assert descriptor["measure"][key] is False
        assert bank.input_hashes["local"] == descriptor["data_binding"]["sha256"]
        underlying.add(bank.input_hashes["local"])
        jobs.add(bank.input_hashes["global"])
    assert len(underlying) == 1
    assert len(jobs) == 5
    assert runner.design.permanent_pass_binding["measure"]["scientific_admission"] is False


def test_parameters_scales_and_moments_are_fresh_and_bound(built):
    _output, args, runner, adapter, _provider = built
    initial = runner.states[0]
    policy = PolicyView.from_dict(initial.policy_state)
    np.testing.assert_array_equal(policy.values, args["parameters"])
    np.testing.assert_array_equal(adapter.denominators, args["denominators"])
    assert not adapter.denominators.flags.writeable
    assert policy.metadata["finite_objective_input_fingerprint"] == adapter.metadata.input_fingerprint
    assert initial.optimizer_state["iteration"] == 0
    assert initial.optimizer_state["learning_rate"] == args["learning_rate"]
    for moment in ("first_moment", "second_moment"):
        assert initial.optimizer_state[moment] == [0., 0.]
    assert initial.method_state["gradnorm_reset_per_call"] is True
    assert initial.method_state["preferred"] == "cagrad"
    assert initial.rng_state["pcgrad_seed"] == 20260722
    assert all(rate == args["learning_rate"] for rate in initial.method_state["rates"].values())
    for normalization, denominator in zip(adapter.metadata.normalizations, args["denominators"], strict=True):
        assert normalization.training.factor == normalization.selection.factor == normalization.terminal.factor == 1. / denominator


def test_constructor_supports_caller_ordered_seven_tasks(tmp_path):
    task_ids = tuple(f"task.{index}" for index in reversed(range(7)))
    args = arguments(tmp_path, task_ids=task_ids, denominators=[2.] * 7)
    runner, adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    assert runner.design.task_ids == task_ids
    assert adapter.metadata.registry.task_ids == task_ids
    assert runner.executor.spec.task_count == 7


def test_actual_controls_equal_objective_and_determine_membership(built, monkeypatch):
    _output, _args, runner, adapter, provider = built
    calls = []

    def measured(parameters):
        calls.append(parameters.numpy())
        return quadratic(parameters)

    monkeypatch.setattr(adapter, "objective", measured)
    original = PolicyView.from_dict(runner.states[0].policy_state)
    policy = replace(original, values=(1., 1.))
    stage, = runner.stages
    control = provider.control(0, policy, stage, 0)
    boundary = provider.control(0, policy, stage, 1)
    validation = provider.validation(0, policy, stage, update_index=2)
    certification = provider.certification(0, policy, stage, update_index=2)
    expected = {"equation.a": 1., "equation.b": 0.}
    assert control.task_upper_mse == boundary.task_upper_mse == validation.task_mean_mse == expected
    assert control.raw_records["raw_losses"] == {"equation.a": 2., "equation.b": 0.}
    assert all(certification.conjuncts.values())
    assert certification.hard_vetoes["production_integrity_evidence"] is False
    assert certification.passed is False
    assert len(calls) == 4
    state = runner.executor.core.coordinator.initialize(
        replace(runner.states[0], policy_state=policy.to_dict(), policy_fingerprint=None), control)
    assert runner.executor.core.coordinator.read(state).permanent == ("equation.a", "equation.b")
    higher = replace(policy, values=(3., 4.))
    control = provider.control(0, higher, stage, 0)
    assert control.task_upper_mse == {"equation.a": 12.5, "equation.b": 2.6}
    state = runner.executor.core.coordinator.initialize(
        replace(runner.states[0], policy_state=higher.to_dict(), policy_fingerprint=None), control)
    assert runner.executor.core.coordinator.read(state).permanent == ()


@pytest.mark.parametrize("kind", ("arity", "numpy", "float32", "raw_shape", "normalized_shape", "gradient_shape",
                                 "nan_raw", "nan_gradient", "negative", "normalization", "tiny_normalization"))
def test_callback_output_contract_refusals(built, monkeypatch, kind):
    import tensorflow as tf

    _output, _args, runner, adapter, _provider = built

    def invalid(parameters):
        raw, normalized, gradients = quadratic(parameters)
        if kind == "arity":
            return raw, normalized
        if kind == "numpy":
            return raw.numpy(), normalized, gradients
        if kind == "float32":
            return raw, tf.cast(normalized, tf.float32), gradients
        if kind == "raw_shape":
            return raw[None, :], normalized, gradients
        if kind == "normalized_shape":
            return raw, normalized[None, :], gradients
        if kind == "gradient_shape":
            return raw, normalized, gradients[:, :1]
        if kind == "nan_raw":
            return raw * tf.constant(np.nan, tf.float64), normalized, gradients
        if kind == "nan_gradient":
            return raw, normalized, gradients * tf.constant(np.nan, tf.float64)
        if kind == "negative":
            return -raw, -normalized, gradients
        if kind == "normalization":
            return raw, normalized / tf.constant([2., 5.], tf.float64), gradients
        return tf.constant([2e-20, 5e-20], tf.float64), tf.zeros([2], tf.float64), gradients

    monkeypatch.setattr(adapter, "objective", invalid)
    with pytest.raises(ValueError):
        adapter.evaluate(PolicyView.from_dict(runner.states[0].policy_state))


def test_original_callback_exception_is_preserved(built, monkeypatch):
    _output, _args, runner, adapter, provider = built
    failure = RuntimeError("manufactured objective failure")

    def broken(_parameters):
        raise failure

    monkeypatch.setattr(adapter, "objective", broken)
    with pytest.raises(RuntimeError) as error:
        provider.control(0, PolicyView.from_dict(runner.states[0].policy_state), runner.stages[0], 0)
    assert error.value is failure


def test_policy_batch_and_role_coordinates_refuse_stale_inputs(built):
    _output, _args, runner, adapter, provider = built
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    stage, = runner.stages
    batch = runner.batch_factory(0, policy, stage, 0)
    assert batch.metadata["update_index"] == 0
    assert batch.metadata["policy_fingerprint"] == policy.fingerprint()
    assert batch.metadata["input_binding"] == adapter.metadata.input_binding
    with pytest.raises(ValueError, match="prepared policy/update"):
        adapter.compute_task_values_and_gradients(policy, batch, 1)
    with pytest.raises(ValueError, match="prepared policy/update"):
        adapter.compute_task_values_and_gradients(replace(policy, values=(2., 3.)), batch, 0)
    with pytest.raises(ValueError, match="policy/data binding"):
        adapter.prepare_batch(replace(policy, metadata={"finite_objective_input_fingerprint": "0" * 64}), 0)
    with pytest.raises(ValueError, match="dimension"):
        adapter.prepare_batch(replace(policy, values=(1.,)), 0)
    for update in (-1, 4, True):
        with pytest.raises(ValueError, match="outside frozen"):
            runner.batch_factory(0, policy, stage, update)
    with pytest.raises(ValueError, match="outside frozen"):
        provider.control(1, policy, stage, 0)
    with pytest.raises(ValueError, match="scope is not registered"):
        provider.control(0, policy, stage, 3)
    with pytest.raises(ValueError, match="outside the scheduled role range"):
        provider.validation(0, policy, stage, update_index=5)


def test_design_sources_data_and_callback_are_frozen(built):
    output, args, runner, _adapter, _provider = built
    design = ReplicationDesignBinding.from_dict(json.loads((output / "design.json").read_text()))
    assert design.binding_hash() == runner.design.binding_hash()
    configuration = json.loads((output / "configuration.json").read_text())
    assert configuration["data_binding"] == args["data_binding"]
    assert configuration["source_evidence"] == args["source_evidence"]
    assert configuration["callback"]["recipe_hash"] == stable_hash(args["data_binding"]["objective_recipe"])
    assert configuration["callback"]["source"] == args["data_binding"]["objective_source"]
    for relative, digest in runner.design.generic_code_hashes.items():
        assert finite._reference(output / "source" / relative)["sha256"] == digest
    checkpoint = runner.design.bind_checkpoint(runner.states[0])
    runner.design.validate_checkpoint(checkpoint)
    changed = replace(runner.design, target_hashes={**runner.design.target_hashes, "data_binding": "0" * 64})
    with pytest.raises(ValueError, match="design hash mismatch"):
        changed.validate_checkpoint(checkpoint)


def test_same_build_has_identical_bindings_and_no_callback_execution(built):
    output, args, runner, _adapter, _provider = built
    before = {str(path): finite._reference(path)["sha256"] for path in output.rglob("*") if path.is_file()}
    rebuilt, _adapter, _provider = finite.build_runner(output, **args)
    assert rebuilt.design.binding_hash() == runner.design.binding_hash()
    assert before == {str(path): finite._reference(path)["sha256"] for path in output.rglob("*") if path.is_file()}


@pytest.mark.parametrize("key,value", (("learning_rate", .02), ("updates", 6), ("boundary_every", 1),
                                      ("parameters", [3., 5.]), ("denominators", [2., 6.]),
                                      ("threshold", .01), ("task_ids", ("equation.b", "equation.a")),
                                      ("objective", quadratic)))
def test_rebuild_refuses_changed_configuration(built, key, value):
    output, args, _runner, _adapter, _provider = built
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        finite.build_runner(output, **{**args, key: value})


@pytest.mark.parametrize("field", ("objective_recipe", "fixed_data", "profile"))
def test_rebuild_refuses_changed_data_recipe_or_profile(built, field):
    output, args, _runner, _adapter, _provider = built
    data = {**args["data_binding"], field: {"changed": True}}
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        finite.build_runner(output, **{**args, "data_binding": data})


@pytest.mark.parametrize("changes", (
    {"task_ids": []}, {"task_ids": ["same", "same"]}, {"task_ids": [" ", "b"]},
    {"task_ids": [f"task.{index}" for index in range(8)]},
    {"parameters": [[3., 4.]]}, {"parameters": [np.inf, 4.]}, {"parameters": [True, False]},
    {"denominators": [0., 5.]}, {"denominators": [2.]}, {"denominators": [1e-320, 5.]},
    {"threshold": 0.}, {"threshold": True}, {"learning_rate": np.nan},
    {"updates": 0}, {"updates": True}, {"boundary_every": 3}, {"boundary_every": False},
    {"objective": None}, {"data_binding": {}}, {"source_evidence": {}},
))
def test_invalid_inputs_refuse_before_artifacts(tmp_path, changes):
    args = arguments(tmp_path, **changes)
    with pytest.raises((TypeError, ValueError)):
        finite.build_runner(tmp_path / "runtime", **args)
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("kind", ("recipe", "source_missing", "source_not_declared", "source_wrong", "evidence_changed"))
def test_missing_or_incorrect_callback_provenance_refuses(tmp_path, kind):
    args = arguments(tmp_path)
    if kind == "recipe":
        args["data_binding"].pop("objective_recipe")
    elif kind == "source_missing":
        args["data_binding"].pop("objective_source")
    elif kind == "source_not_declared":
        args["source_evidence"]["sources"] = []
    elif kind == "source_wrong":
        args["data_binding"]["objective_source"] = args["source_evidence"]["plan"]
    else:
        args["source_evidence"]["plan"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        finite.build_runner(tmp_path / "runtime", **args)
    assert not (tmp_path / "runtime").exists()


def test_changed_declared_data_refuses_before_objective(tmp_path):
    args = arguments(tmp_path)
    data = tmp_path / "data.txt"
    data.write_text("original finite data")
    args["data_binding"]["dataset"] = finite._reference(data)
    runner, adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    data.write_text("changed finite data")
    with pytest.raises(ValueError, match="bound file changed"):
        adapter.evaluate(PolicyView.from_dict(runner.states[0].policy_state))


def test_changed_sources_and_scope_descriptor_refuse_before_objective(built, monkeypatch):
    _output, _args, runner, adapter, provider = built
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    sources = dict(adapter.sources)
    sources[next(iter(sources))] = "0" * 64
    with monkeypatch.context() as patch:
        patch.setattr(adapter, "sources", sources)
        with pytest.raises(ValueError, match="bound file changed"):
            adapter.evaluate(policy)
    references = [dict(reference) for reference in provider.scope_references]
    references[0]["sha256"] = "0" * 64
    monkeypatch.setattr(provider, "scope_references", references)
    with pytest.raises(ValueError, match="bound file changed"):
        provider.control(0, policy, runner.stages[0], 0)


def test_runner_contains_no_model_branch_or_private_optimizer():
    parsed = ast.parse(Path(finite.__file__).read_text())
    imports = [node.module for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)]
    assert not any(module and ("sgu" in module or "rotemberg" in module) for module in imports)
    # The upstream runner now walks checkpoint-transition history with a host
    # while loop. Construction must still delegate every optimizer update.
    constructor = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "build_runner")
    assert not any(isinstance(node, ast.While) for node in ast.walk(constructor))
    assert not any(isinstance(node, ast.Attribute) and node.attr in ("apply_gradients", "minimize") for node in ast.walk(parsed))


def test_loss_only_roles_do_not_call_full_gradients_and_bind_same_recipe(tmp_path):
    args = arguments(tmp_path, objective_values=quadratic_values)
    runner, adapter, provider = finite.build_runner(tmp_path / "runtime", **args)
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    stage, = runner.stages
    control = provider.control(0, policy, stage, 0)
    validation = provider.validation(0, policy, stage, update_index=0)
    certification = provider.certification(0, policy, stage, update_index=0)
    assert control.task_upper_mse == validation.task_mean_mse == {"equation.a": 12.5, "equation.b": 2.6}
    assert certification.passed is False
    assert adapter.values_binding["recipe_hash"] == adapter.callback_binding["recipe_hash"]
    assert adapter.values_binding["source"] == adapter.callback_binding["source"]
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        finite.build_runner(tmp_path / "runtime", **{**args, "objective_values": None})


def test_loss_only_values_and_analytic_gradients_match_full_callback(tmp_path):
    args = arguments(tmp_path, objective=quadratic, objective_values=quadratic_values)
    runner, adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    for values in ((3., 4.), (1., 1.), (0., 0.), (1e-10, -1e-10)):
        policy = replace(PolicyView.from_dict(runner.states[0].policy_state), values=values)
        full = adapter.evaluate(policy)
        only_values = adapter.evaluate_values(policy)
        for loss, value in zip(full[:2], only_values, strict=True):
            np.testing.assert_array_equal(loss.numpy(), value.numpy())
        expected = 2. * (np.array(values)[None, :] - np.array([[0.], [1.]])) / np.array([[2.], [5.]])
        np.testing.assert_array_equal(full[2].numpy(), expected)


@pytest.mark.parametrize("values_first", (True, False))
def test_overlapping_full_and_loss_only_disagreement_refuses(tmp_path, values_first):
    def inconsistent(parameters):
        raw, normalized = quadratic_values(parameters)
        return raw * 2., normalized * 2.

    args = arguments(tmp_path, objective=quadratic, objective_values=inconsistent)
    runner, adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    first, second = (adapter.evaluate_values, adapter.evaluate) if values_first else (adapter.evaluate, adapter.evaluate_values)
    first(policy)
    with pytest.raises(ValueError, match="callback parity mismatch"):
        second(policy)


def test_loss_only_normalization_refuses(tmp_path):
    def inconsistent(parameters):
        raw, normalized = quadratic_values(parameters)
        return raw, normalized * 2.

    args = arguments(tmp_path, objective_values=inconsistent)
    runner, _adapter, provider = finite.build_runner(tmp_path / "runtime", **args)
    with pytest.raises(ValueError, match="normalization"):
        provider.control(0, PolicyView.from_dict(runner.states[0].policy_state), runner.stages[0], 0)


@pytest.mark.numerics
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GENERIC_FINITE_OBJECTIVE_RUN_NUMERICAL") != "1", reason="bounded numerical opt-in required")
def test_tiny_actual_updates_boundary_receipts_and_completed_recovery(tmp_path, monkeypatch):
    args = arguments(tmp_path, objective=quadratic, objective_values=quadratic_values, updates=2, boundary_every=1)
    runner, _adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    result = runner.run()
    state = result.states[0]
    assert state.update_index == state.optimizer_state["iteration"] == 2
    assert not np.array_equal(state.policy_state["values"], args["parameters"])
    assert [event["event"] for event in result.events] == ["initialization", "update", "boundary", "update", "boundary"]
    assert [receipt["role"] for receipt in result.evaluations] == ["validation", "certification"]
    for control in runner.executor.core.coordinator.read(state).controls:
        evaluation = runner.store.rehydrate_control(control.evaluation)
        assert evaluation.task_upper_mse == evaluation.raw_records["normalized_losses"]
        assert evaluation.raw_records["measure"]["held_out"] is False
    certification = result.evaluations[-1]["result"]
    assert certification["passed"] is False
    assert certification["hard_vetoes"]["production_integrity_evidence"] is False
    checkpoint, _events = runner.store.recover(0, through_update_index=1)
    assert checkpoint.update_index == 1
    rebuilt, adapter, provider = finite.build_runner(tmp_path / "runtime", **args)
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    monkeypatch.setattr(adapter, "evaluate", forbidden)
    monkeypatch.setattr(adapter, "evaluate_values", forbidden)
    monkeypatch.setattr(adapter, "compute_task_values_and_gradients", forbidden)
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    monkeypatch.setattr(rebuilt, "batch_factory", forbidden)
    recovered = rebuilt.run()
    assert recovered.states[0].to_dict() == state.to_dict()
    assert recovered.evaluations == result.evaluations
    assert recovered.selection == result.selection


@pytest.mark.numerics
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GENERIC_FINITE_OBJECTIVE_RUN_NUMERICAL") != "1", reason="bounded numerical opt-in required")
def test_real_initial_convergence_does_not_fake_an_update_or_scientific_pass(tmp_path, monkeypatch):
    args = arguments(tmp_path, objective=zero_objective)
    runner, _adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    monkeypatch.setattr(runner.executor, "step", forbidden)
    result = runner.run()
    assert result.states[0].update_index == 0
    assert all(result.evaluations[-1]["result"]["conjuncts"].values())
    assert result.evaluations[-1]["result"]["passed"] is False
    assert [event["event"] for event in result.events] == ["initialization"]


def test_scheduled_training_constructor_freezes_callback_and_schedule_without_calls(tmp_path):
    args = scheduled_arguments(tmp_path, training_objective=forbidden)
    output = tmp_path / "runtime"
    runner, adapter, _provider = finite.build_runner(output, **args)
    configuration = json.loads((output / "configuration.json").read_text())
    training = configuration["training"]
    assert training["schedule"] == list(args["training_schedule"])
    assert training["schedule_hash"] == stable_hash(args["training_schedule"])
    assert training["callback"]["source"] == args["data_binding"]["training_objective_source"]
    assert adapter.metadata.input_binding["training"] == adapter.adapter_identity()["training"] == training
    assert runner.design.target_hashes["training_schedule"] == training["schedule_hash"]
    assert runner.design.target_hashes["training_objective"] == stable_hash(training["callback"])
    assert training["measure"]["scientific_admission"] is False
    assert adapter.metadata.normalization_binding["training_measure"] == finite.TRAINING_MEASURE
    assert not runner.store.has_initial(0)
    args["training_schedule"][0]["row_indices"][0] = 99
    assert adapter.training_binding["schedule"][0]["row_indices"][0] == 0
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        finite.build_runner(output, **args)


@pytest.mark.parametrize("kind", ("missing_callback", "missing_schedule", "not_tuple", "length", "empty_entry",
                                 "nonmapping", "missing_source", "wrong_source", "bad_hash", "bad_index"))
def test_invalid_training_pair_schedule_and_source_refuse_before_artifacts(tmp_path, kind):
    args = scheduled_arguments(tmp_path)
    if kind == "missing_callback":
        args.pop("training_objective")
    elif kind == "missing_schedule":
        args.pop("training_schedule")
    elif kind == "not_tuple":
        args["training_schedule"] = list(args["training_schedule"])
    elif kind == "length":
        args["training_schedule"] = args["training_schedule"][:1]
    elif kind == "empty_entry":
        args["training_schedule"] = ({}, args["training_schedule"][1])
    elif kind == "nonmapping":
        args["training_schedule"] = (1, args["training_schedule"][1])
    elif kind == "missing_source":
        args["data_binding"].pop("training_objective_source")
    elif kind == "wrong_source":
        args["data_binding"]["training_objective_source"] = args["source_evidence"]["plan"]
    elif kind == "bad_hash":
        args["data_binding"]["training_objective_source"]["sha256"] = "0" * 64
    else:
        args["training_schedule"][0]["update_index"] = 1
    with pytest.raises((ValueError, TypeError)):
        finite.build_runner(tmp_path / "runtime", **args)
    assert not (tmp_path / "runtime").exists()


def test_training_callback_source_must_be_in_source_evidence(tmp_path):
    args = scheduled_arguments(tmp_path, training_objective=finite._array)
    args["data_binding"]["training_objective_source"] = finite._reference(finite.__file__)
    with pytest.raises(ValueError, match="training_objective_source must be included"):
        finite.build_runner(tmp_path / "runtime", **args)


def test_scheduled_batch_binds_exact_entry_and_refuses_stale_or_exhausted_requests(tmp_path):
    args = scheduled_arguments(tmp_path, training_objective=forbidden)
    runner, adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    stage, = runner.stages
    first = runner.batch_factory(0, policy, stage, 0)
    second = runner.batch_factory(0, policy, stage, 1)
    assert first.metadata["training_schedule_entry"] == args["training_schedule"][0]
    assert second.metadata["training_schedule_entry"] == args["training_schedule"][1]
    assert first.metadata["training_schedule_hash"] == stable_hash(args["training_schedule"])
    assert first.metadata["training_schedule_entry_hash"] == stable_hash(args["training_schedule"][0])
    assert first.metadata["input_binding"] == adapter.metadata.input_binding
    altered = replace(first, metadata={**first.metadata, "training_schedule_entry": args["training_schedule"][1]})
    for batch, update in ((first, 1), (second, 0), (altered, 0)):
        with pytest.raises(ValueError, match="prepared policy/update binding mismatch"):
            adapter.compute_task_values_and_gradients(policy, batch, update)
    with pytest.raises(ValueError, match="training schedule exhausted"):
        adapter.prepare_batch(policy, 2)
    for update in (-1, True):
        with pytest.raises(ValueError):
            adapter.prepare_batch(policy, update)
    with pytest.raises(ValueError, match="outside frozen update allowance"):
        runner.batch_factory(0, policy, stage, 2)


def test_scheduled_losses_do_not_change_full_control_membership_or_parity_cache(tmp_path):
    args = scheduled_arguments(tmp_path)
    runner, adapter, provider = finite.build_runner(tmp_path / "runtime", **args)
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    stage, = runner.stages
    control = provider.control(0, policy, stage, 0)
    initialized = runner.executor.core.coordinator.initialize(runner.states[0], control)
    cached = adapter._last_measurements["values"]
    batch = adapter.prepare_batch(policy, 0)
    evaluated = adapter.compute_task_values_and_gradients(policy, batch, 0)
    np.testing.assert_allclose(evaluated.tensors.values, [3.125, .65], rtol=0., atol=1e-15)
    assert control.task_upper_mse == {"equation.a": 12.5, "equation.b": 2.6}
    assert runner.executor.core.coordinator.read(initialized).permanent == ()
    assert adapter._last_measurements["values"] is cached
    assert "full" not in adapter._last_measurements
    validation = provider.validation(0, policy, stage, update_index=0)
    certification = provider.certification(0, policy, stage, update_index=0)
    assert validation.task_mean_mse == control.task_upper_mse
    assert not any(certification.conjuncts.values())
    assert certification.hard_vetoes["production_integrity_evidence"] is False


@pytest.mark.parametrize("kind", ("normalization", "gradient_shape", "gradient_nan", "float32", "arity"))
def test_scheduled_objective_keeps_full_denominators_and_output_contract(tmp_path, kind):
    import tensorflow as tf

    def invalid(parameters, update_index):
        raw, normalized, gradients = scheduled_quadratic(parameters, update_index)
        if kind == "normalization":
            return raw, normalized / 2., gradients
        if kind == "gradient_shape":
            return raw, normalized, gradients[:, :1]
        if kind == "gradient_nan":
            return raw, normalized, gradients * tf.constant(np.nan, tf.float64)
        if kind == "float32":
            return raw, tf.cast(normalized, tf.float32), gradients
        return raw, normalized

    args = scheduled_arguments(tmp_path, training_objective=invalid)
    runner, adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    with pytest.raises(ValueError):
        adapter.compute_task_values_and_gradients(policy, adapter.prepare_batch(policy, 0), 0)


def test_scheduled_artifact_reference_is_verified_before_training(tmp_path):
    args = scheduled_arguments(tmp_path, training_objective=forbidden)
    archive = tmp_path / "scheduled-data.json"
    archive.write_text('{"indices":[0,2]}')
    args["training_schedule"][0]["data"] = finite._reference(archive)
    runner, adapter, _provider = finite.build_runner(tmp_path / "runtime", **args)
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    archive.write_text('{"indices":[1,3]}')
    with pytest.raises(ValueError, match="bound file changed"):
        adapter.compute_task_values_and_gradients(policy, adapter.prepare_batch(policy, 0), 0)


def test_training_callback_change_and_checkpoint_schedule_mismatch_refuse(tmp_path):
    args = scheduled_arguments(tmp_path)
    output = tmp_path / "runtime"
    runner, _adapter, _provider = finite.build_runner(output, **args)
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        finite.build_runner(output, **{**args, "training_objective": forbidden})
    checkpoint = runner.design.bind_checkpoint(runner.states[0])
    changed = replace(runner.design, target_hashes={**runner.design.target_hashes, "training_schedule": "0" * 64})
    with pytest.raises(ValueError, match="design hash mismatch"):
        changed.validate_checkpoint(checkpoint)


@pytest.mark.numerics
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GENERIC_FINITE_OBJECTIVE_RUN_NUMERICAL") != "1", reason="bounded numerical opt-in required")
@pytest.mark.parametrize("updates", (1, 2))
def test_actual_scheduled_updates_full_controls_and_completed_replay(tmp_path, monkeypatch, updates):
    observed_updates = []

    def measured_training(parameters, update_index):
        observed_updates.append(update_index)
        return scheduled_quadratic(parameters, update_index)

    args = scheduled_arguments(tmp_path, updates=updates, training_objective=measured_training)
    output = tmp_path / "runtime"
    runner, _adapter, _provider = finite.build_runner(output, **args)
    result = runner.run()
    assert observed_updates == list(range(updates))
    state = result.states[0]
    assert state.update_index == state.optimizer_state["iteration"] == updates
    assert not np.array_equal(state.policy_state["values"], args["parameters"])
    assert runner.executor.core.coordinator.read(state).permanent == ()
    policy = PolicyView.from_dict(state.policy_state)
    expected = dict(zip(args["task_ids"], quadratic_values(np.asarray(policy.values))[1].numpy(), strict=True))
    for task, value in expected.items():
        np.testing.assert_allclose(result.evaluations[0]["result"]["task_mean_mse"][task], value, rtol=2e-14, atol=0.)
    assert result.evaluations[-1]["result"]["passed"] is False
    checkpoint, _events = runner.store.recover(0, through_update_index=1)
    assert checkpoint.update_index == 1
    rebuilt, adapter, provider = finite.build_runner(output, **args)
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    for name in ("objective", "objective_values", "training_objective", "evaluate", "evaluate_values",
                 "compute_task_values_and_gradients"):
        monkeypatch.setattr(adapter, name, forbidden)
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    monkeypatch.setattr(rebuilt, "batch_factory", forbidden)
    replay = rebuilt.run()
    assert replay.states[0].to_dict() == state.to_dict()
    assert replay.evaluations == result.evaluations
    assert replay.selection == result.selection
    assert observed_updates == list(range(updates))
