"""Residual curvature wiring, coordinate refusals and atomic runner recovery."""

import json
from collections.abc import Mapping
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow as tf
from tests.contracts import test_generic_postproposal as legacy
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training import generic_postproposal as postproposal
from mooneural.training.generic_training_contracts import canonical_json

DENOMINATORS = np.array([2., 5.])


def batch_identity(batch, update_index):
    return dict(batch) if isinstance(batch, Mapping) else {"update": update_index, "rows": []}


def objective(parameters):
    differences = np.asarray(parameters) - 1.
    raw = differences**2
    return tuple(tf.constant(value, tf.float64) for value in (
        raw, raw / DENOMINATORS, np.diag(2 * differences / DENOMINATORS)))


def losses(parameters, batch, update_index):
    return objective(parameters)[:2]


def residual_provider(parameters, batch, update_index, context):
    residuals = [np.array([[(value - 1.) / np.sqrt(denominator)]])
        for value, denominator in zip(parameters, DENOMINATORS, strict=True)]
    responses = [(row / np.sqrt(denominator))[None, None, :]
        for row, denominator in zip(np.eye(2), DENOMINATORS, strict=True)]
    return {"residuals": residuals, "responses": responses, "probabilities": np.ones(1),
        "basis": np.eye(2), "context": context}


def guard(provider=residual_provider, callback=losses, **kwargs):
    return postproposal.GuardedPostproposal(callback, batch_identity, DENOMINATORS, [10., 10.],
        binding={"criterion": "manufactured-quadratic"}, residual_provider=provider,
        residual_binding={"recipe": "two-independent-signed-quadratics"}, curvature_relative_radius=2., **kwargs)


def source_proposal(parameters):
    displacement = np.array([.1, .2])
    return tuple(tf.constant(value) for value in (
        parameters + displacement, np.array([.3, .4]), np.array([.5, .6]), np.int64(7),
        np.array([1., 1.]), displacement, displacement, np.zeros(1), np.zeros(1), np.float64(0), True))


def call(instance, *, batch=None, optimizer=None):
    parameters = np.array([3., 4.])
    raw, normalized, gradients = objective(parameters)
    batch = {} if batch is None else batch
    return instance(parameters=parameters, batch=batch, update_index=0, raw_losses=raw,
        normalized_losses=normalized, gradient_rows=gradients,
        optimizer=source_proposal(parameters) if optimizer is None else optimizer,
        active_indices=(0,), constraint_indices=(1,), gradient_batch_binding=batch_identity(batch, 0))


def test_curvature_replaces_only_parameters_displacement_and_dots():
    original = source_proposal(np.array([3., 4.]))
    updated, receipt = call(guard(), optimizer=original)
    assert receipt["accepted"] and receipt["loss_calls"] == 2
    assert receipt["direction_kind"] == "residual-curvature-common-descent"
    assert receipt["curvature"]["provider_calls"] == receipt["curvature"]["solver_calls"] == 1
    assert receipt["source_moments_preserved"]
    assert np.all(np.asarray(objective(updated[0])[0]) < np.asarray(objective([3., 4.])[0]))
    np.testing.assert_array_equal(updated[6], np.asarray(updated[0]) - np.array([3., 4.]))
    np.testing.assert_array_equal(updated[8], np.asarray(updated[6])[1:])
    np.testing.assert_array_equal(receipt["rounded_displacement"], updated[6])
    np.testing.assert_array_equal(receipt["actual_all_task_gradient_dots"],
        np.asarray(objective([3., 4.])[2]) @ np.asarray(updated[6]))
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert updated[index] is original[index]
    canonical_json(receipt)


@pytest.mark.parametrize("field", ("parameters_hash", "batch_hash", "update_index", "D", "profile_hash", "descent_indices"))
def test_stale_response_context_refuses_before_solver(field):
    def stale(parameters, batch, update_index, context):
        payload = residual_provider(parameters, batch, update_index, context)
        payload["context"] = {**dict(context), field: "stale"}
        return payload

    with pytest.raises(postproposal.PostproposalRejected, match="stale") as failure:
        call(guard(stale))
    assert failure.value.receipt["curvature"]["solver_calls"] == 0
    assert failure.value.receipt["loss_calls"] == 1


@pytest.mark.parametrize("kind", ("normalization", "gradient", "basis", "probabilities", "shape", "dtype", "nonfinite", "missing_task"))
def test_invalid_residual_provider_refuses_before_solver(kind):
    def invalid(parameters, batch, update_index, context):
        payload = residual_provider(parameters, batch, update_index, context)
        if kind == "normalization":
            payload["residuals"][0] *= 2.
        elif kind == "gradient":
            payload["responses"][0] *= 2.
        elif kind == "basis":
            payload["basis"] *= 2.
        elif kind == "probabilities":
            payload["probabilities"] *= 2.
        elif kind == "shape":
            payload["responses"][0] = np.ones((2, 1, 2))
        elif kind == "dtype":
            payload["basis"] = payload["basis"].astype(np.float32)
        elif kind == "nonfinite":
            payload["responses"][0][0, 0, 0] = np.nan
        elif kind == "missing_task":
            payload["residuals"].pop()
        return payload

    with pytest.raises(postproposal.PostproposalRejected) as failure:
        call(guard(invalid))
    assert failure.value.receipt["curvature"]["solver_calls"] == 0
    assert failure.value.receipt["loss_calls"] == 1


@pytest.mark.parametrize("kind", ("parameters", "batch", "callback", "profile"))
def test_provider_mutation_is_detected(kind):
    def mutation(parameters, batch, update_index, context):
        payload = residual_provider(parameters, batch, update_index, context)
        if kind == "parameters":
            parameters.setflags(write=True)
            parameters[0] += 1.
        elif kind == "batch":
            batch["changed"] = True
        elif kind == "callback":
            instance.residual_provider = residual_provider
        else:
            instance.binding["curvature_direction"]["relative_weight_radius"] = 99.
        return payload

    instance = guard(mutation)
    with pytest.raises(postproposal.PostproposalRejected, match="changed|mutated"):
        call(instance)


def test_protected_task_must_decrease_and_finite_fraction_recovers():
    def nonlinear(parameters, batch, update_index):
        raw = np.asarray(objective(parameters)[0]).copy()
        raw[1] += .2 * (parameters[1] - 4.)**4
        return raw, raw / DENOMINATORS

    _updated, receipt = call(guard(callback=nonlinear))
    assert not receipt["candidates"][0]["accepted"]
    assert receipt["candidates"][0]["raw"][1] > 9.
    assert receipt["accepted_fraction"] == .5
    assert receipt["loss_calls"] == 3


def test_nonfinite_solver_result_has_no_candidate_calls(monkeypatch):
    monkeypatch.setattr(postproposal, "curvature_progress", lambda *args: {"direction": np.array([np.nan, 0.])})
    with pytest.raises(postproposal.PostproposalRejected, match="finite common descent") as failure:
        call(guard())
    assert failure.value.receipt["loss_calls"] == 1
    assert failure.value.receipt["curvature"]["solver_calls"] == 1


def test_failed_protected_dot_clears_candidate_acceptance(monkeypatch):
    def nonlinear(parameters, batch, update_index):
        raw = np.asarray(objective(parameters)[0]).copy()
        raw[1] -= 11. * (parameters[1] - 4.)**2
        return raw, raw / DENOMINATORS

    instance = guard(callback=nonlinear)
    monkeypatch.setattr(instance, "_curvature_direction", lambda *args: np.array([-.1, 1.]))
    with pytest.raises(postproposal.PostproposalRejected, match="budget exhausted") as failure:
        call(instance)
    candidate = failure.value.receipt["candidates"][0]
    assert candidate["raw"][0] < 4. and candidate["raw"][1] < 9.
    assert candidate["protected_dots"][0] > 0
    assert not candidate["accepted"]
    assert "protected-dot" in candidate["reason"]


def runner_arguments():
    source = finite._reference(__file__)
    instance = postproposal.GuardedPostproposal(losses, batch_identity, DENOMINATORS, DENOMINATORS,
        binding={"criterion": "manufactured-runner-quadratic"}, residual_provider=residual_provider,
        residual_binding={"source": source}, curvature_relative_radius=.1)
    return {"task_ids": ("a", "b"), "denominators": DENOMINATORS, "threshold": 1.,
        "parameters": [3., 4.], "objective": objective, "learning_rate": .01,
        "updates": 1, "boundary_every": 1, "postproposal": instance,
        "data_binding": {"objective_recipe": {"kind": "manufactured"}, "objective_source": source,
            "postproposal_source": source, "postproposal_residual_source": source},
        "source_evidence": {"source": source}}


def test_actual_runner_transaction_checkpoint_and_zero_call_recovery(tmp_path, monkeypatch):
    arguments = runner_arguments()
    runner, _adapter, _provider = finite.build_runner(tmp_path / "runtime", **arguments)
    result = runner.run()
    assert result.states[0].update_index == result.states[0].optimizer_state["iteration"] == 1
    transaction = json.loads(next((tmp_path / "runtime/checkpoints/arms/arm-0/checkpoints").glob("update-????????.json")).read_text())
    receipt = transaction["event"]["postproposal"]
    assert receipt["accepted"] and receipt["source_moments_preserved"]
    assert "residual_callback" in runner.design.to_dict()["optimizer_binding"]["postproposal"]
    rebuilt, adapter, provider = finite.build_runner(tmp_path / "runtime", **arguments)
    forbidden = Mock(side_effect=AssertionError("unexpected recovery numerical call"))
    for name in ("control", "validation", "certification"):
        monkeypatch.setattr(provider, name, forbidden)
    for name in ("evaluate", "evaluate_values", "compute_task_values_and_gradients"):
        monkeypatch.setattr(adapter, name, forbidden)
    monkeypatch.setattr(rebuilt.executor, "step", forbidden)
    monkeypatch.setattr(postproposal, "curvature_progress", forbidden)
    recovered = rebuilt.run()
    assert recovered.states[0].to_dict() == result.states[0].to_dict()
    assert recovered.evaluations == result.evaluations
    assert recovered.selection == result.selection
    forbidden.assert_not_called()


def test_runner_refuses_missing_residual_source_before_output(tmp_path):
    arguments = runner_arguments()
    arguments["data_binding"].pop("postproposal_residual_source")
    with pytest.raises(ValueError, match="postproposal_residual_source"):
        finite.build_runner(tmp_path / "runtime", **arguments)
    assert not (tmp_path / "runtime").exists()


def test_absent_option_preserves_binding_and_default_outputs():
    implicit = postproposal.GuardedPostproposal(losses, batch_identity, DENOMINATORS, [10., 10.], binding={"same": True})
    explicit = postproposal.GuardedPostproposal(losses, batch_identity, DENOMINATORS, [10., 10.], binding={"same": True},
        residual_provider=None, residual_binding=None, curvature_relative_radius=None)
    assert implicit.binding_hash == explicit.binding_hash
    assert "curvature_direction" not in implicit.binding
    first, first_receipt = call(implicit)
    second, second_receipt = call(explicit)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert first_receipt == second_receipt


def test_component_and_curvature_profiles_are_exclusive():
    with pytest.raises(ValueError, match="exclusive"):
        guard(component_rows=lambda *args: None, component_binding={"source": "fixture"})


@pytest.mark.parametrize("options", ({}, {"refinement": True}, {"search_profile": True}))
def test_unsupported_retained_curvature_transition_refuses(options):
    instance = guard()
    with pytest.raises(ValueError, match="retained curvature profile transitions"):
        finite._archived_postproposal(instance, {"profile": instance.binding}, **options)


def test_actual_curvature_rejection_rolls_back_complete_state(monkeypatch):
    existing, initial, batch = legacy.fixture(task_count=2)

    def wrong_residuals(parameters, batch, update_index, context):
        return {"residuals": [np.full((1, 1), 129.)] * 2,
            "responses": [np.ones((1, 1, len(parameters)))] * 2,
            "probabilities": np.ones(1), "basis": np.eye(len(parameters)), "context": context}

    instance = postproposal.GuardedPostproposal(legacy.quadratic_batch_losses, legacy.batch_identity,
        [1., 1.], [.04, .04], binding={"fixture": "rollback"}, residual_provider=wrong_residuals,
        residual_binding={"fixture": "intentionally-wrong-norms"}, curvature_relative_radius=.1)
    executor = legacy.PermanentPassExecutor(existing.adapter, existing.spec, legacy.roles(), threshold=.04,
        postproposal=instance)
    parent = executor.initialize(initial, legacy.fake_control(initial, existing.adapter.registry.task_ids, [.2, .2]))
    saved = canonical_json(parent.to_dict())
    original_choose = legacy.MethodFallback.choose

    def mutate_rates(method, *arguments):
        result = original_choose(method, *arguments)
        method.rates["cagrad"] *= .5
        method.preferred = "pcgrad"
        return result

    monkeypatch.setattr(legacy.MethodFallback, "choose", mutate_rates)
    with pytest.raises(legacy.PermanentPassUpdateError, match="residual squared norms") as failure:
        executor.step(parent, batch)
    assert failure.value.checkpoint is parent
    assert canonical_json(failure.value.checkpoint.to_dict()) == saved
    assert failure.value.event["transaction_rolled_back"]
    assert failure.value.event["postproposal"]["curvature"]["provider_calls"] == 1
    assert failure.value.event["postproposal"]["curvature"]["solver_calls"] == 0
    boundary, = executor.boundaries.values()
    assert boundary.methods.rates == parent.method_state["rates"]
    assert boundary.methods.preferred == parent.method_state["preferred"]


@pytest.mark.parametrize("radius", (None, 0., -1., np.nan, True))
def test_explicit_positive_radius_required(radius):
    with pytest.raises(ValueError, match="positive relative radius"):
        postproposal.GuardedPostproposal(losses, batch_identity, DENOMINATORS, [10., 10.], binding={"same": True},
            residual_provider=residual_provider, residual_binding={"recipe": "fixture"}, curvature_relative_radius=radius)
