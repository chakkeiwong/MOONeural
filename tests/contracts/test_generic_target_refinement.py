"""Weighted targets and finite guards against independently known curved losses."""

import copy

import numpy as np
import pytest
import tensorflow as tf
from tests.contracts import test_generic_postproposal as fixtures
from tests.contracts import test_generic_required_component_progress as targets
from mooneural.training.generic_postproposal import (
    GuardedPostproposal,
    PostproposalRejected,
)
from mooneural.training.generic_relative_progress import (
    make_relative_progress_function,
    relative_progress_impl,
)


def weighted_kernel(**options):
    return make_relative_progress_function(1, explicit_component_targets=True,
        explicit_component_weights=True, **options)


def test_all_one_weights_preserve_every_unweighted_result_and_graph_signature():
    plain = make_relative_progress_function(1, explicit_component_targets=True)
    weighted = weighted_kernel()
    for value in (.5, 1. - 1e-12):
        for required in (False, True):
            arguments = (*targets.operands(value), np.array([required]))
            old, new = plain(*arguments), weighted(*arguments, np.ones(1))
            assert new["valid"]
            for key in old:
                np.testing.assert_array_equal(new[key], old[key])
    assert len(plain.input_signature) == 8 and len(weighted.input_signature) == 9
    assert weighted.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("factor", [1e-100, 1., 1.7e308])
def test_weighted_direction_and_actual_margin_match_analytic_solution(factor):
    arguments = list(targets.operands())
    for index in (0, 1, 3, 4):
        arguments[index] *= factor
    result = weighted_kernel()(*arguments, np.array([True]), np.array([4.]))
    assert result["valid"]
    expected = -.1 * np.array([1., 2.]) / np.sqrt(5.)
    np.testing.assert_allclose(result["direction"], expected, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(result["minimum_fractional_progress"], .1 / np.sqrt(5.), rtol=1e-8)
    np.testing.assert_allclose(result["weighted_minimum_fractional_progress"], .1 / np.sqrt(5.), rtol=1e-8)
    result = weighted_kernel()(*targets.operands(), np.array([True]), np.array([.5]))
    assert result["valid"]
    assert float(result["minimum_fractional_progress"]) < float(result["weighted_minimum_fractional_progress"])
    np.testing.assert_allclose(result["minimum_fractional_progress"],
        np.min(-np.asarray(result["direction"]) / [1., .5]), rtol=1e-8)


@pytest.mark.parametrize("weight", [1e-250, 1e308])
def test_unrepresentable_target_ratio_refuses_without_dropping_required_row(weight):
    arguments = list(targets.operands())
    arguments[3] = np.array([1e-100 if weight < 1. else .5])
    if weight > 1.:
        arguments[4] = np.array([[0., 1e-100]])
    result = weighted_kernel()(*arguments, np.array([True]), np.array([weight]))
    assert bool(result["relative_target_rows"][1])
    assert not result["valid"] and not result["targets_valid"]
    assert int(result["iterations"]) == 0


@pytest.mark.parametrize("weights", [np.array([0.]), np.array([-1.]), np.array([np.inf]), np.array([np.nan]),
    np.array([1]), np.array([1.], np.float32), np.ones((1, 1)), np.ones(2)])
def test_bad_weights_refuse(weights):
    with pytest.raises((TypeError, ValueError, tf.errors.InvalidArgumentError)):
        relative_progress_impl(*targets.operands(), required_components=np.array([True]),
            component_target_weights=weights)


def curved_guard(*, shifted=False, tied=False, curvature=.6, rival_quartic=0., corrupt_cells=False,
                 fractions=(1., .5, .25), **options):
    size = 3 if shifted or tied else 2
    parent = np.array([10., 2., 2. if tied else 1.5][:size])
    cells = tuple(f"cell-{index}" for index in range(size))
    calls = {"provider": 0, "loss": 0, "components": 0}

    def values(parameters):
        raw = parent + parameters
        raw[1] += curvature * parameters[0] ** 2 + rival_quartic * parameters[1] ** 4
        if shifted:
            raw[2] += .8 * parameters[1] ** 2
        elif tied:
            raw[2] += curvature * parameters[0] ** 2
        return raw

    def losses(parameters, batch, update):
        calls["loss"] += 1
        raw = np.array([np.max(values(parameters))])
        return raw, raw / 100.

    def components(parameters, batch, update, context):
        calls["provider"] += 1
        assert np.array_equal(parameters, np.zeros(size))
        return {"owner_indices": np.zeros(size, np.int32), "cell_ids": cells,
            "raw_values": parent.copy(), "normalized_values": parent / 100.,
            "normalized_gradients": np.eye(size) / 100., "context": context}

    def cell_losses(parameters, batch, update, context, owners, ids):
        calls["components"] += 1
        assert ids == cells
        raw = values(parameters)
        return {"owner_indices": owners, "cell_ids": tuple(reversed(ids)) if corrupt_cells else ids, "raw_values": raw,
            "normalized_values": raw / 100., "context": context}

    guard = GuardedPostproposal(losses, lambda batch, update: batch, [100.], [1.],
        binding={"fixture": "cross-curvature"}, fractions=fractions, component_rows=components,
        component_binding={"fixture": "cross-curvature"}, component_direction="relative_public_progress",
        coupled_component_guard=True, component_loss_values=cell_losses,
        component_loss_binding={"fixture": "cross-curvature"}, **options)
    return guard, calls, parent


def apply(guard, parent):
    parameters = np.zeros(len(parent))
    displacement = parameters.copy()
    displacement[0] = -1.
    source = fixtures.proposal(parameters=parameters, displacement=displacement, descent=parameters)
    updated, receipt = guard(parameters=parameters, batch={}, update_index=0, raw_losses=np.array([10.]),
        normalized_losses=np.array([.1]), gradient_rows=np.eye(len(parent))[:1] / 100., optimizer=source,
        active_indices=(0,), constraint_indices=(), gradient_batch_binding={})
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert updated[index] is source[index]
    return updated, receipt


def test_refinement_resolves_cross_curvature_using_one_provider_and_original_radius():
    baseline, old_calls, parent = curved_guard()
    _old, old_receipt = apply(baseline, parent)
    enabled, calls, parent = curved_guard(finite_target_refinement=True)
    updated, receipt = apply(enabled, parent)
    assert old_receipt["accepted_fraction"] == .25 and receipt["accepted_fraction"] == 1.
    assert receipt["direction_kind"] == "same-parent-target-refinement"
    assert calls == {"provider": 1, "loss": 3, "components": 2}
    assert old_calls["provider"] == 1 and receipt["component_fallback"]["witness_calls"] == 2
    detail = receipt["target_refinement"]
    assert detail["component_target_weights"] == [1., 3.]
    assert detail["cell_id"] == "cell-1" and detail["witness_calls"] == 1
    np.testing.assert_allclose(updated[0], -np.array([10., 6.]) / np.sqrt(136.), rtol=1e-8)
    np.testing.assert_allclose(detail["source_norm"], 1., rtol=1e-12)
    assert receipt["candidates"][-1]["component"]["strict_decrease"]


def test_changed_limiting_cell_is_checked_and_does_not_trigger_second_refinement():
    guard, calls, parent = curved_guard(shifted=True, finite_target_refinement=True)
    _updated, receipt = apply(guard, parent)
    assert receipt["accepted_fraction"] == .5
    first_refined = receipt["candidates"][1]["component"]["raw_values"]
    assert first_refined[1] < parent[1] and first_refined[2] > parent[2]
    assert not receipt["candidates"][1]["accepted"]
    assert np.all(np.array(receipt["candidates"][-1]["component"]["raw_values"]) < parent)
    assert calls["provider"] == 1 and receipt["component_fallback"]["witness_calls"] == 2


def test_equal_remainders_select_first_supplied_cell():
    guard, _calls, parent = curved_guard(tied=True, finite_target_refinement=True)
    _updated, receipt = apply(guard, parent)
    detail = receipt["target_refinement"]
    assert detail["ratios"][1] == detail["ratios"][2]
    assert detail["component_index"] == 1


def test_invalid_refined_solve_resumes_original_grid(monkeypatch):
    guard, calls, parent = curved_guard(finite_target_refinement=True)
    invalid = weighted_kernel(max_iterations=0)
    monkeypatch.setattr(guard, "target_refinement_solver", invalid)
    monkeypatch.setattr(guard, "_target_refinement_solver", invalid)
    _updated, receipt = apply(guard, parent)
    assert receipt["accepted_fraction"] == .25
    assert not receipt["target_refinement"]["valid"]
    assert calls == {"provider": 1, "loss": 4, "components": 3}
    assert all(trial["direction_kind"] != "same-parent-target-refinement" for trial in receipt["candidates"])


def test_refined_grid_exhaustion_resumes_original_fractions():
    guard, calls, parent = curved_guard(rival_quartic=1000., finite_target_refinement=True)
    _updated, receipt = apply(guard, parent)
    assert receipt["target_refinement"]["valid"]
    assert receipt["accepted_fraction"] == .25
    assert receipt["direction_kind"] == "supplied-component-relative-public-progress"
    assert [trial["direction_kind"] for trial in receipt["candidates"][1:4]] == ["same-parent-target-refinement"] * 3
    assert calls == {"provider": 1, "loss": 7, "components": 6}
    assert receipt["component_fallback"]["provider_calls"] == 1
    assert receipt["component_fallback"]["witness_calls"] == 2


def test_malformed_cell_identity_refuses_instead_of_falling_back():
    guard, calls, parent = curved_guard(corrupt_cells=True, finite_target_refinement=True)
    with pytest.raises(PostproposalRejected) as failure:
        apply(guard, parent)
    assert failure.value.receipt["failure_kind"] == "contract_error"
    assert calls == {"provider": 1, "loss": 2, "components": 1}
    assert "target_refinement" not in failure.value.receipt


def test_exhaustion_retains_every_rejected_trial_and_exact_caps():
    guard, calls, parent = curved_guard(shifted=True, fractions=(1.,), finite_target_refinement=True)
    with pytest.raises(PostproposalRejected, match="budget exhausted") as failure:
        apply(guard, parent)
    receipt = failure.value.receipt
    assert len(receipt["candidates"]) == 2 and not any(trial["accepted"] for trial in receipt["candidates"])
    assert calls == {"provider": 1, "loss": 3, "components": 2}
    assert guard.binding["maximum_loss_calls"] == 3
    assert guard.binding["component_fallback"]["maximum_component_loss_calls"] == 2


@pytest.mark.parametrize("slope_row", [[0., 0.], [1., -1.], [np.inf, 0.]])
def test_unreliable_slopes_never_create_an_extra_solve(slope_row):
    guard, _calls, _parent = curved_guard(finite_target_refinement=True)
    receipt = {"component_fallback": {"required_components": [True], "owner_indices": [0],
        "normalized_gradients": [slope_row], "normalized_values": [.02]}, "candidates": [{}]}
    result = guard._refine_targets(np.array([.1]), np.array([[.01, 0.]]), np.array([True]),
        np.array([-1., 0.]), receipt,
        {"rounded_displacement": [1., 1.], "component": {"raw_values": [2.1]}, "fraction": 1.})
    assert result is None and receipt["target_refinement"]["witness_calls"] == 0
    assert guard.target_refinement_solver.experimental_get_tracing_count() == 0


def test_omitted_and_false_options_are_identical_and_mutation_is_rejected():
    omitted, _calls, parent = curved_guard()
    explicit, _calls, _parent = curved_guard(finite_target_refinement=False)
    old, old_receipt = apply(omitted, parent)
    new, new_receipt = apply(explicit, parent)
    assert omitted.binding == explicit.binding and old_receipt == new_receipt
    for previous, current in zip(old, new, strict=True):
        np.testing.assert_array_equal(previous, current)
    original = copy.deepcopy(explicit.binding)
    explicit.finite_target_refinement = True
    with pytest.raises(ValueError, match="mutated"):
        explicit.validate_binding()
    assert explicit.binding == original


def test_target_refinement_requires_coupled_guard_before_any_callback():
    with pytest.raises(ValueError, match="coupled"):
        GuardedPostproposal(lambda *args: None, lambda *args: None, [1.], [1.],
            binding={"fixture": True}, finite_target_refinement=True)
