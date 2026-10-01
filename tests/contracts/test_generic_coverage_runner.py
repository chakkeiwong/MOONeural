"""Host contracts for auxiliary coverage training; numerical runs are opt-in."""

import ast
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mooneural.training import generic_coverage_runner as coverage
from mooneural.training.generic_certified_numerics import (
    make_certified_method_function,
    make_certified_projected_adam_function,
)
from mooneural.training.generic_coverage_initialization import (
    CoverageGroup,
    CoverageTrainingData,
)
from mooneural.training.generic_policy_coordinates import AffinePolicyCoordinates
from mooneural.training.generic_replication_design import ReplicationDesignBinding
from mooneural.training.generic_replication_runner import GenericReplicationRunner
from mooneural.training.generic_scheduled_role_banks import ScheduledRoleBankRegistry
from mooneural.training.generic_training_contracts import PolicyView, stable_hash


def forbidden(*_args, **_kwargs):
    raise AssertionError("host coverage contracts must not evaluate or update policies")


def arguments(tmp_path, *, width=4, updates=50, rate=1e-4, groups=20):
    generator = np.random.default_rng(1049)
    collection = []
    for index in range(groups):
        inputs = generator.normal(size=(2 + index % 3, 3))
        values = np.stack([1. + inputs[:, 0], 2. + inputs[:, 2]], axis=1)
        jacobians = np.broadcast_to(np.array([[1., .3, 0.], [.2, 0., 1.]]), (len(inputs), 2, 3)).copy()
        collection.append(CoverageGroup(f"group-{index}", "successor" if index % 2 else "current", "training",
                                        inputs, values, jacobians, np.arange(1, len(inputs) + 1), 1.))
    data = CoverageTrainingData(tuple(collection))
    profile = AffinePolicyCoordinates((0.,) * 3, (1.,) * 3, (0.,) * 2, (1.,) * 2, data.binding_hash())
    parameters = generator.normal(0., .04, 3 * width + 2 * width + width**2 + 2 * width + 2)
    evidence = tmp_path / "plan.md"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("Host fixture: no model evaluation or scientific claim.\n")
    reference = coverage._reference(evidence)
    source_evidence = {"initialization": reference, "plan": reference, "master": reference,
                       "sources": [coverage._reference(Path(coverage.__file__))]}
    return (tmp_path / "runtime", profile, width, data, parameters, np.array([2., 3., 4.]), 2, rate, updates, source_evidence)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    args = arguments(tmp_path_factory.mktemp("coverage-runner"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(coverage.CoverageAdapter, "evaluate", forbidden)
        patch.setattr(coverage.CoverageAdapter, "compute_task_values_and_gradients", forbidden)
        components = coverage.build_runner(*args)
    components[1].evaluate = forbidden
    components[1].compute_task_values_and_gradients = forbidden
    return args, components


def test_actual_common_engine_and_fifty_update_schedule(built):
    _args, (runner, adapter, provider) = built
    assert type(runner) is GenericReplicationRunner
    assert runner.adapters == {0: adapter}
    assert runner.evaluation_provider is provider
    assert isinstance(provider.roles, ScheduledRoleBankRegistry)
    stage, = runner.stages
    assert (stage.round_count, stage.updates_per_round, stage.total_updates, stage.population_size) == (5, 10, 50, 1)
    assert stage.select_after is True
    assert runner.threshold == runner.design.threshold == .01
    assert runner.executor.spec.task_count == 3
    assert runner.executor.spec.hard_check_count == 0
    assert runner.executor.core.method_factory is make_certified_method_function
    assert runner.executor.core.update_factory is make_certified_projected_adam_function
    assert [(item.round_number, item.min_update, item.max_update) for item in provider.roles.bindings if item.role == "control"] == [
        (number, number * 10, number * 10) for number in range(6)]
    assert [(item.role, item.min_update, item.max_update) for item in provider.roles.bindings if item.role != "control"] == [
        ("validation", 0, 50), ("certification", 0, 50)]
    assert not runner.store.has_initial(0)


def test_training_population_reuse_is_explicit_and_scopes_have_materialized_descriptors(built):
    args, (runner, adapter, provider) = built
    assert tuple(adapter.metadata.registry.task_ids) == coverage.TASK_IDS
    assert len(adapter.data.groups) == 20
    assert adapter.sample_count == sum(len(group.inputs) for group in args[3].groups)
    underlying = set()
    scopes = set()
    for binding in provider.roles.bindings:
        bank, = binding.manifest.banks
        record = json.loads(coverage._checked(bank.metadata["global_archive"]).read_text())
        assert record["measure"] == coverage.MEASURE
        assert record["measure"]["independent_roles"] is False
        assert record["measure"]["held_out"] is False
        assert "no Student-t" in record["measure"]["estimator"]
        assert bank.input_hashes["global"] == bank.metadata["global_archive"]["sha256"]
        assert bank.input_hashes["local"] == record["data"]["sha256"]
        underlying.add(bank.input_hashes["local"])
        scopes.add(bank.input_hashes["global"])
    assert len(underlying) == 1
    assert len(scopes) == 8
    assert runner.design.permanent_pass_binding["measure"]["scientific_admission"] is False
    assert runner.design.permanent_pass_binding["measure"]["production_integrity_evidence"] is False


def test_supplied_coordinates_values_scales_and_fresh_moments(built):
    args, (runner, adapter, _provider) = built
    initial = runner.states[0]
    policy = PolicyView.from_dict(initial.policy_state)
    np.testing.assert_array_equal(policy.values, args[4])
    np.testing.assert_array_equal(adapter.denominators, args[5])
    assert not adapter.denominators.flags.writeable
    assert policy.metadata["coordinate_profile_hash"] == args[1].binding_hash()
    assert policy.metadata["coverage_input_fingerprint"] == adapter.metadata.input_fingerprint
    assert initial.optimizer_state["iteration"] == 0
    assert initial.optimizer_state["learning_rate"] == args[7]
    for name in ("first_moment", "second_moment"):
        assert initial.optimizer_state[name] == [0.] * adapter.policy_dimension
    assert all(value == args[7] for value in initial.method_state["rates"].values())
    assert adapter.metadata.normalization_binding["scientific_terminal_limits"] is None
    np.testing.assert_array_equal([item.training.factor for item in adapter.metadata.normalizations], 1. / args[5])


def test_native_objective_signature_is_bound_without_tracing(built):
    _args, (_runner, adapter, _provider) = built
    assert adapter._native.experimental_get_tracing_count() == 0
    assert tuple(adapter._native.input_signature[0].shape) == (adapter.policy_dimension,)
    assert adapter._native.input_signature[0].dtype.name == "float64"


def test_control_validation_and_certification_dispatch_to_actual_objective(built, monkeypatch):
    _args, (runner, adapter, provider) = built
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    stage = runner.stages[0]
    failure = RuntimeError("objective entrypoint reached without execution")
    observed = []

    def measured(proposed):
        observed.append(proposed.fingerprint())
        raise failure

    monkeypatch.setattr(adapter, "evaluate", measured)
    for method, keywords in ((provider.control, {"round_number": 0}), (provider.control, {"round_number": 1}),
                             (provider.validation, {"update_index": 10}), (provider.certification, {"update_index": 50})):
        with pytest.raises(RuntimeError) as outcome:
            method(0, policy, stage, **keywords)
        assert outcome.value is failure
    assert observed == [policy.fingerprint()] * 4


def test_batch_policy_update_and_request_binding_refusals(built):
    _args, (runner, adapter, provider) = built
    policy = PolicyView.from_dict(runner.states[0].policy_state)
    stage = runner.stages[0]
    batch = runner.batch_factory(0, policy, stage, 10)
    assert batch.metadata["update_index"] == 10
    assert batch.metadata["policy_fingerprint"] == policy.fingerprint()
    assert batch.task_ids == coverage.TASK_IDS
    for update in (-1, 50, True):
        with pytest.raises(ValueError, match="outside frozen"):
            runner.batch_factory(0, policy, stage, update)
    wrong = replace(policy, metadata={**policy.metadata, "coordinate_profile_hash": "0" * 64})
    with pytest.raises(ValueError, match="profile mismatch"):
        adapter.prepare_batch(wrong, 0)
    with pytest.raises(ValueError, match="outside frozen"):
        provider.control(1, policy, stage, 0)
    with pytest.raises(ValueError, match="scope is not registered"):
        provider.control(0, policy, stage, 6)


def test_design_inputs_evidence_and_sources_are_frozen(built):
    args, (runner, _adapter, _provider) = built
    restored = ReplicationDesignBinding.from_dict(json.loads((args[0] / "design.json").read_text()))
    assert restored.binding_hash() == runner.design.binding_hash()
    configuration = json.loads((args[0] / "configuration.json").read_text())
    assert configuration["source_evidence"] == args[-1]
    assert configuration["data_hash"] == args[3].binding_hash()
    assert runner.design.master_sha256 == args[-1]["master"]["sha256"]
    assert runner.design.target_hashes["configuration"] == coverage._reference(args[0] / "configuration.json")["sha256"]
    for relative, digest in runner.design.generic_code_hashes.items():
        assert coverage._reference(args[0] / "source" / relative)["sha256"] == digest
    assert len(list((args[0] / "evidence").glob("*.snapshot"))) == 4
    checkpoint = runner.design.bind_checkpoint(runner.states[0])
    runner.design.validate_checkpoint(checkpoint)
    changed = replace(runner.design, target_hashes={**runner.design.target_hashes, "data": "0" * 64})
    with pytest.raises(ValueError, match="design hash mismatch"):
        changed.validate_checkpoint(checkpoint)


def test_identical_build_replays_without_objective_calls(built):
    args, (runner, _adapter, _provider) = built
    before = {str(path): coverage._reference(path)["sha256"] for path in args[0].rglob("*") if path.is_file()}
    rebuilt, adapter, _provider = coverage.build_runner(*args)
    assert rebuilt.design.binding_hash() == runner.design.binding_hash()
    assert adapter._native.experimental_get_tracing_count() == 0
    assert before == {str(path): coverage._reference(path)["sha256"] for path in args[0].rglob("*") if path.is_file()}


@pytest.mark.parametrize("index,value", [(7, 3e-4), (8, 10), (6, 1), (5, [1., 2., 3.])])
def test_existing_design_refuses_changed_hyperparameters(built, index, value):
    args, _components = built
    changed = list(args)
    changed[index] = value
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        coverage.build_runner(*changed)


@pytest.mark.parametrize("kind", ("profile", "width", "parameters", "denominators", "rate", "updates", "evidence"))
def test_invalid_inputs_refuse_before_output(tmp_path, kind):
    args = list(arguments(tmp_path, groups=2))
    if kind == "profile":
        args[1] = replace(args[1], training_binding="0" * 64)
    elif kind == "width":
        args[2] = True
    elif kind == "parameters":
        args[4][0] = np.nan
    elif kind == "denominators":
        args[5][1] = 0.
    elif kind == "rate":
        args[7] = 0.
    elif kind == "updates":
        args[8] = 51
    else:
        args[-1]["plan"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        coverage.build_runner(*args)
    assert not args[0].exists()


def test_changed_source_is_refused_before_evaluation(built, monkeypatch):
    _args, (_runner, adapter, _provider) = built
    sources = dict(adapter.sources)
    first = next(iter(sources))
    sources[first] = "0" * 64
    monkeypatch.setattr(adapter, "_native", forbidden)
    with pytest.raises(ValueError, match="bound file changed"):
        coverage._check_sources(sources)


def test_recovery_requires_completed_checkpoint(built):
    args, _components = built
    with pytest.raises(ValueError, match="completed auxiliary checkpoint required"):
        coverage.recover_completed(args[0])


def test_runner_has_no_model_specific_import_or_optimizer_loop():
    parsed = ast.parse(Path(coverage.__file__).read_text())
    imports = [node.module for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)]
    assert not any(module and ("sgu" in module or "rotemberg" in module) for module in imports)
    assert not any(isinstance(node, ast.While) for node in ast.walk(parsed))
    assert not any(isinstance(node, ast.Attribute) and node.attr in ("apply_gradients", "minimize") for node in ast.walk(parsed))


@pytest.mark.numerics
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GENERIC_COVERAGE_RUN_NUMERICAL") != "1", reason="parent owns the numerical launch allowance")
def test_real_auxiliary_update_boundary_receipts_and_recovery(tmp_path):
    args = arguments(tmp_path, updates=2, groups=2)
    runner, adapter, _provider = coverage.build_runner(*args)
    initial = np.asarray(PolicyView.from_dict(runner.states[0].policy_state).values)
    result = runner.run()
    assert result.states[0].update_index == 2
    assert not np.array_equal(initial, result.states[0].policy_state["values"])
    assert adapter._native.experimental_get_tracing_count() == 1
    assert [event["event"] for event in result.events] == ["initialization", "update", "update", "boundary"]
    assert [receipt["role"] for receipt in result.evaluations] == ["validation", "certification"]
    certification = result.evaluations[-1]["result"]
    assert certification["passed"] is False
    assert certification["hard_vetoes"]["production_integrity_evidence"] is False
    assert certification["upper_records"]["measure"]["held_out"] is False
    ten_style, _events = runner.store.recover(0, through_update_index=1)
    assert ten_style.update_index == 1
    recovered = coverage.recover_completed(args[0])
    assert recovered.states[0].to_dict() == result.states[0].to_dict()
    assert recovered.selection == result.selection
    assert recovered.evaluations == result.evaluations


@pytest.mark.numerics
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GENERIC_COVERAGE_RUN_NUMERICAL") != "1", reason="parent owns the numerical launch allowance")
def test_exact_regression_convergence_can_stop_at_initial_control(tmp_path):
    args = list(arguments(tmp_path, updates=50, groups=2))
    groups = tuple(replace(group, values=np.zeros_like(group.values), raw_jacobians=np.zeros_like(group.raw_jacobians))
                   for group in args[3].groups)
    args[3] = CoverageTrainingData(groups)
    args[1] = replace(args[1], training_binding=args[3].binding_hash())
    args[4] = np.zeros_like(args[4])
    runner, _adapter, _provider = coverage.build_runner(*args)
    result = runner.run()
    assert result.states[0].update_index == 0
    assert all(result.evaluations[-1]["result"]["conjuncts"].values())
    assert result.evaluations[-1]["result"]["passed"] is False
    recovered = coverage.recover_completed(args[0])
    assert stable_hash(recovered.states[0].to_dict()) == stable_hash(result.states[0].to_dict())
