"""Qualified relative direction through the shared finite guard call chain."""

import numpy as np
import pytest
from tests.contracts import test_generic_postproposal as fixtures
from tests.contracts import test_generic_protected_descent as protected
from mooneural.training.generic_postproposal import (
    GuardedPostproposal,
    PostproposalRejected,
)


def losses(parameters, batch, update_index):
    raw = (np.array([2., .7]) + parameters) ** 2
    return raw, raw / [2., 3.]


def components(parameters, batch, update_index, context):
    owners = np.array(sorted(set(context["active_indices"]).union(context.get("extra_descent_indices", ()))),
                      dtype=np.int32)
    raw, normalized = losses(parameters, batch, update_index)
    return {"owner_indices": owners, "cell_ids": [f"owner-{owner}" for owner in owners],
        "raw_values": raw[owners], "normalized_values": normalized[owners],
        "normalized_gradients": np.diag(2. * (np.array([2., .7]) + parameters) / [2., 3.])[owners],
        "context": context}


def guard(**options):
    return GuardedPostproposal(losses, fixtures.batch_identity, [2., 3.], [1., 1.],
        binding={"criterion": "analytic-two-squares"}, component_rows=components,
        component_binding={"source": "analytic-two-squares"}, component_direction="relative_public_progress",
        fractions=options.pop("fractions", (1., .5, .25)), **options)


def apply(enabled, *, displacement=(-.1, 0.), descent=(0., 0.), gradients=None):
    source = fixtures.proposal((0., 0.), displacement, descent)
    raw, normalized = losses(np.zeros(2), (), 6)
    updated, receipt = enabled(parameters=np.zeros(2), batch=(), update_index=6,
        raw_losses=raw, normalized_losses=normalized,
        gradient_rows=np.diag([2., 1.4 / 3.]) if gradients is None else gradients,
        optimizer=source, active_indices=(0,), constraint_indices=(1,),
        gradient_batch_binding=fixtures.batch_identity((), 6))
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert updated[index] is source[index]
    return np.asarray(updated[0]), receipt


@pytest.mark.parametrize("extras", [(), (1,)])
@pytest.mark.parametrize("finite", [False, True])
def test_relative_direction_dispatches_once_with_truthful_membership_and_source_slots(extras, finite):
    enabled = guard(extra_descent_indices=extras, finite_component_fallback=finite)
    parameters, receipt = apply(enabled, descent=(1.7e308, 1.7e308))
    detail = receipt["component_fallback"]
    assert enabled.blend.experimental_get_tracing_count() == 0
    assert enabled.component_descent.experimental_get_tracing_count() == 1
    assert receipt["direction_kind"] == "supplied-component-relative-public-progress"
    assert receipt["component_trigger"] == "relative-public-progress-preference"
    assert receipt["blend_fraction"] is receipt["original_segment_valid"] is None
    assert detail["provider_calls"] == detail["witness_calls"] == 1
    assert detail["context"]["active_indices"] == [0]
    assert detail["context"].get("extra_descent_indices", []) == list(extras)
    assert detail["valid"] and detail["targets_valid"]
    assert receipt["loss_calls"] == 2 and enabled.binding["maximum_loss_calls"] == 4
    np.testing.assert_allclose(np.linalg.norm(parameters), .1, rtol=1e-12)
    np.testing.assert_array_equal(parameters, detail["direction"])
    if extras:
        np.testing.assert_allclose(parameters[0] / parameters[1], 1. / .35, rtol=1e-10)
    else:
        assert parameters[1] == 0.
    profile = enabled.binding["component_fallback"]
    assert profile["component_direction"] == "relative_public_progress"
    assert profile["solver"] == "float64-feasible-seed-primal-active-set"
    assert "active_target" not in profile and "finite_trigger_radius" not in profile


def test_backtracking_keeps_one_provider_and_the_finite_loss_cap():
    parameters, receipt = apply(guard(extra_descent_indices=(1,)), displacement=(-5., 0.))
    assert receipt["accepted_fraction"] == .5 and receipt["loss_calls"] == 3
    assert [candidate["accepted"] for candidate in receipt["candidates"]] == [False, True]
    assert np.all(losses(parameters, (), 6)[0] < losses(np.zeros(2), (), 6)[0])
    assert receipt["component_fallback"]["provider_calls"] == 1
    with pytest.raises(PostproposalRejected, match="budget exhausted") as failed:
        apply(guard(extra_descent_indices=(1,), fractions=(1.,)), displacement=(-5., 0.))
    assert failed.value.receipt["loss_calls"] == 2


def test_invalid_relative_direction_never_falls_back_to_legacy_proposal():
    enabled = guard()
    with pytest.raises(PostproposalRejected, match="invalid supplied-component relative") as failed:
        apply(enabled, gradients=np.zeros((2, 2)))
    assert failed.value.receipt["loss_calls"] == 1 and failed.value.receipt["candidates"] == []
    assert enabled.blend.experimental_get_tracing_count() == 0


@pytest.mark.parametrize("extras", [(), (1,)])
@pytest.mark.parametrize("finite", [False, True])
def test_explicit_equality_is_byte_identical_to_omitted_default(extras, finite):
    options = {"extra_descent_indices": extras, "finite_component_fallback": finite}
    original, explicit = protected.guard(**options), protected.guard(**options, component_direction="equality")
    assert original.binding == explicit.binding and original.binding_hash == explicit.binding_hash
    old, old_receipt = protected.apply(original)
    new, new_receipt = protected.apply(explicit)
    np.testing.assert_array_equal(old, new)
    assert old_receipt == new_receipt


@pytest.mark.parametrize("direction", ["unknown", "", None])
def test_invalid_direction_is_rejected_before_callbacks(direction):
    with pytest.raises(ValueError, match="component direction"):
        protected.guard(component_direction=direction)


def test_relative_direction_requires_provider_and_rejects_profile_mutation():
    with pytest.raises(ValueError, match="provider"):
        GuardedPostproposal(losses, fixtures.batch_identity, [2., 3.], [1., 1.],
            binding={"criterion": "analytic"}, component_direction="relative_public_progress")
    enabled = guard()
    enabled.component_direction = "equality"
    with pytest.raises(ValueError, match="mutated"):
        enabled.validate_binding()
    enabled = guard()
    enabled.binding["component_fallback"]["max_iterations"] = 1000
    with pytest.raises(ValueError, match="mutated"):
        enabled.validate_binding()
