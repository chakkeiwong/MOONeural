"""Optional protected-task preferences preserve membership and source updates."""

import copy

import numpy as np
import pytest
import tensorflow as tf
from tests.contracts import test_generic_postproposal as fixtures
from mooneural.training.generic_execution_boundary import MethodFallback
from mooneural.training.generic_grouped_components import (
    make_grouped_component_provider,
)
from mooneural.training.generic_postproposal import (
    GuardedPostproposal,
    PostproposalRejected,
)
from mooneural.training.generic_training_contracts import canonical_json, stable_hash


def cells(parameters):
    return np.array([(.7 + parameters[2]) ** 2, (.3 + parameters[1]) ** 2])


def losses(parameters, batch, update_index):
    raw = np.array([2. + parameters[0], max(cells(parameters))])
    return raw, raw / [2., 3.]


def components(parameters, batch, update_index, context):
    raw = cells(parameters)
    return {"owner_indices": np.array([1, 1]), "cell_ids": ["maximum", "hidden"],
        "raw_values": raw, "normalized_values": raw / 3.,
        "normalized_gradients": np.array([[0., 0., 2. * (.7 + parameters[2]) / 3.],
                                          [0., 2. * (.3 + parameters[1]) / 3., 0.]]),
        "context": context}


def guard(provider=components, **options):
    return GuardedPostproposal(losses, fixtures.batch_identity, [2., 3.], [1., 1.],
        binding={"criterion": "protected-two-cell-square"}, component_rows=provider,
        component_binding={"source": "analytic-square-components"},
        fractions=options.pop("fractions", (1., .5, .25)), **options)


def apply(postproposal, *, displacement=(-.1, .15, 0.), gradient_rows=None, descent=(1., 0., 0.)):
    source = fixtures.proposal((0., 0., 0.), displacement, descent)
    raw, normalized = losses(np.zeros(3), (), 6)
    updated, receipt = postproposal(parameters=np.zeros(3), batch=(), update_index=6,
        raw_losses=raw, normalized_losses=normalized,
        gradient_rows=np.array([[.5, 0., 0.], [0., 0., 1.4 / 3.]]) if gradient_rows is None else gradient_rows,
        optimizer=source, active_indices=(0,), constraint_indices=(1,),
        gradient_batch_binding=fixtures.batch_identity((), 6))
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert updated[index] is source[index]
    return np.asarray(updated[0]), receipt


def test_optional_mode_repairs_hidden_growth_without_redeclaring_active_membership():
    old_parameters, old = apply(guard())
    np.testing.assert_allclose(old_parameters, [-.1, .15, 0.])
    assert cells(old_parameters)[1] > cells(np.zeros(3))[1]
    assert not old["component_fallback"]["invoked"]
    parameters, receipt = apply(guard(extra_descent_indices=(1,)))
    assert np.all(cells(parameters) < cells(np.zeros(3)))
    assert losses(parameters, (), 6)[0][0] < 2.
    assert receipt["component_trigger"] == "extra-descent-preference"
    detail = receipt["component_fallback"]
    assert detail["context"]["active_indices"] == [0]
    assert detail["context"]["extra_descent_indices"] == [1]
    assert detail["provider_calls"] == detail["witness_calls"] == 1
    assert detail["radius_source"] == "source-final-displacement-norm"
    assert receipt["loss_calls"] == 2


@pytest.mark.parametrize("descent", [(0., 0., 0.), (1.7e308, 1.7e308, 1.7e308)])
def test_extra_mode_never_requires_or_traces_the_unused_legacy_reference(descent):
    enabled = guard(extra_descent_indices=(1,))
    parameters, receipt = apply(enabled, descent=descent)
    assert np.all(cells(parameters) < cells(np.zeros(3)))
    assert enabled.blend.experimental_get_tracing_count() == 0
    assert receipt["original_segment_valid"] is None and receipt["blend_fraction"] is None
    with pytest.raises(PostproposalRejected, match="reference direction"):
        apply(guard(), descent=descent)


@pytest.mark.parametrize("displacement", [(0., 0., 0.), (float("nan"), 0., 0.),
                                        (float("inf"), 0., 0.), (1.7e308, 1.7e308, 1.7e308)])
def test_extra_mode_retains_final_source_displacement_validity(displacement):
    enabled = guard(extra_descent_indices=(1,))
    with pytest.raises((ValueError, PostproposalRejected)):
        apply(enabled, displacement=displacement)
    assert enabled.component_descent.experimental_get_tracing_count() == 0
    assert enabled.blend.experimental_get_tracing_count() == 0


@pytest.mark.parametrize("finite", [False, True])
def test_empty_preference_is_exact_default_and_stronger_mode_has_bounded_calls(finite):
    old = guard(finite_component_fallback=finite)
    empty = guard(finite_component_fallback=finite, extra_descent_indices=())
    assert old.binding == empty.binding and old.binding_hash == empty.binding_hash
    previous, previous_receipt = apply(old)
    current, current_receipt = apply(empty)
    np.testing.assert_array_equal(current, previous)
    assert current_receipt == previous_receipt
    strong = guard(finite_component_fallback=finite, extra_descent_indices=[1])
    assert strong.binding["maximum_loss_calls"] == 4
    parameters, receipt = apply(strong, displacement=(-.1, 2., 0.))
    assert receipt["accepted_fraction"] == .5
    assert [candidate["accepted"] for candidate in receipt["candidates"]] == [False, True]
    assert receipt["loss_calls"] == 3
    assert np.all(cells(parameters) < cells(np.zeros(3)))


@pytest.mark.parametrize("fault", ["opposing", "zero"])
def test_failed_extra_witness_never_falls_back_to_original_harmful_segment(fault):
    def invalid(parameters, batch, update_index, context):
        value = components(parameters, batch, update_index, context)
        value["normalized_gradients"][1] = [0., 0., -1.] if fault == "opposing" else [0., 0., 0.]
        return value

    with pytest.raises(PostproposalRejected, match="invalid supplied-component") as failed:
        apply(guard(invalid, finite_component_fallback=True, extra_descent_indices=(1,)))
    assert failed.value.receipt["loss_calls"] == 1
    assert failed.value.receipt["candidates"] == []
    assert not failed.value.receipt["component_fallback"]["valid"]


def test_strict_extra_public_decrease_survives_a_feasible_threshold():
    candidate_guard = guard(extra_descent_indices=(1,), fractions=(1.,))
    with pytest.raises(PostproposalRejected, match="budget exhausted") as failed:
        apply(candidate_guard, displacement=(-.1, 2., 0.))
    candidate = failed.value.receipt["candidates"][0]
    assert .49 < candidate["raw"][1] < 1.
    assert not candidate["accepted"]


@pytest.mark.parametrize("indices", [(1, 1), (-1,), (2,), (True,), (1.,), {1}])
def test_invalid_preferences_rejected_before_any_numerics(indices):
    with pytest.raises(ValueError, match="extra descent"):
        guard(extra_descent_indices=indices)


def test_extra_descent_requires_provider_and_is_bound_against_mutation():
    with pytest.raises(ValueError, match="provider"):
        GuardedPostproposal(losses, fixtures.batch_identity, [2., 3.], [1., 1.],
                            binding={"criterion": "analytic"}, extra_descent_indices=(1,))
    value = guard(extra_descent_indices=(1,))
    value.extra_descent_indices = (0,)
    with pytest.raises(ValueError):
        value.validate_binding()


def test_preferred_task_already_active_is_not_duplicated_or_removed_from_original_constraints():
    def active_component(parameters, batch, update_index, context):
        raw = np.array([2. + parameters[0]])
        return {"owner_indices": np.array([0]), "cell_ids": ["active"], "raw_values": raw,
            "normalized_values": raw / 2., "normalized_gradients": np.array([[.5, 0., 0.]]), "context": context}

    parameters, receipt = apply(guard(active_component, extra_descent_indices=(0,)))
    assert parameters[0] < 0. and parameters[2] == 0.
    assert receipt["component_fallback"]["context"]["active_indices"] == [0]
    assert receipt["component_fallback"]["context"]["extra_descent_indices"] == [0]


def make_provider(*, owner_cap=2):
    def raw_rows(parameters):
        return tf.stack((tf.stack((2. + parameters[0], (.7 + parameters[2]) ** 2)),
                         tf.stack((1.9 + parameters[0], (.3 + parameters[1]) ** 2))))

    profile = {"task_ids": ["active", "protected"],
        "aggregation": {"active": "max_group_mean", "protected": "max_group_mean"},
        "max_active_owners": owner_cap, "max_component_rows": 4, "top_k": 2,
        "source": {"analytic": "two-squares"}, "input_binding": {"rows": ["maximum", "hidden"]}}
    return make_grouped_component_provider(raw_rows, ["row-0", "row-1"], np.array([.25, .75]),
        np.array([0, 1]), ["maximum", "hidden"], np.array([2., 3.]), np.array([0, 1]), profile)


def provider_context():
    return {"parameters_hash": stable_hash(np.zeros(3).tolist()), "batch_hash": stable_hash(()),
        "update_index": 6, "coordinate_mode": "raw_over_D", "D": [2., 3.],
        "active_indices": [0], "extra_descent_indices": [1], "profile_hash": "analytic-guard"}


def test_grouped_provider_selects_union_with_nonunit_D_and_truthful_membership():
    provider = make_provider()
    context = provider_context()
    before = copy.deepcopy(context)
    payload = provider(np.zeros(3), (), 6, context)
    assert context == before
    assert payload["owner_indices"].tolist() == [0, 0, 1, 1]
    np.testing.assert_allclose(payload["normalized_values"], np.array([2., 1.9, .49, .09]) / [2., 2., 3., 3.])
    np.testing.assert_allclose(payload["normalized_gradients"],
        [[.5, 0., 0.], [.5, 0., 0.], [0., 0., 1.4 / 3.], [0., .2, 0.]], atol=1e-15)
    assert provider.receipts[0]["reverse_calls"] == 4
    assert provider.receipts[0]["forward_calls"] == 1
    context["active_indices"] = [0, 1]
    rotated = provider(np.zeros(3), (), 6, context)
    np.testing.assert_array_equal(rotated["normalized_gradients"], payload["normalized_gradients"])


def test_extra_provider_budget_refuses_before_graph_execution():
    provider = make_provider(owner_cap=1)
    with pytest.raises(ValueError, match="budget exceeded"):
        provider(np.zeros(3), (), 6, provider_context())
    assert provider.graph.experimental_get_tracing_count() == 0
    assert provider.receipts == []


def test_zero_extra_public_gradient_refuses_without_claiming_infeasibility():
    with pytest.raises(PostproposalRejected, match="invalid supplied-component"):
        apply(guard(extra_descent_indices=(1,)), gradient_rows=np.array([[.5, 0., 0.], [0., 0., 0.]]))


def test_extra_mode_provider_failure_rolls_back_actual_executor_and_method_state(monkeypatch):
    existing, initial, batch = fixtures.fixture(task_count=2)

    def malformed(*arguments):
        raise ValueError("extra-descent malformed component")

    enabled = GuardedPostproposal(fixtures.quadratic_batch_losses, fixtures.batch_identity,
        [1., 1.], [.04, .04], binding={"fixture": "quadratic-extra-descent"},
        component_rows=malformed, component_binding={"source": "malformed-analytic"}, extra_descent_indices=(1,))
    executor = fixtures.PermanentPassExecutor(existing.adapter, existing.spec, fixtures.roles(),
                                            threshold=.04, postproposal=enabled)
    parent = executor.initialize(initial, fixtures.fake_control(initial, existing.adapter.registry.task_ids, [.2, .001]))
    before = canonical_json(parent.to_dict())
    source_choose = MethodFallback.choose

    def changed_method(method, *arguments):
        result = source_choose(method, *arguments)
        method.rates["cagrad"] *= .5
        method.preferred = "pcgrad"
        method.events.append({"fixture": "uncommitted"})
        return result

    monkeypatch.setattr(MethodFallback, "choose", changed_method)
    with pytest.raises(fixtures.PermanentPassUpdateError, match="extra-descent malformed") as failed:
        executor.step(parent, batch)
    assert failed.value.checkpoint is parent and canonical_json(parent.to_dict()) == before
    assert failed.value.event["transaction_rolled_back"] and not failed.value.event["committed"]
    receipt = failed.value.event["postproposal"]
    assert receipt["loss_calls"] == 1 and receipt["component_trigger"] == "extra-descent-preference"
    boundary, = executor.boundaries.values()
    assert boundary.methods.rates == parent.method_state["rates"]
    assert boundary.methods.preferred == parent.method_state["preferred"]
    assert boundary.methods.events == []
