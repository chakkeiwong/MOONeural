"""Analytic finite-trigger screening and transaction checks, without economics."""

import json

import numpy as np
import pytest
import tensorflow as tf
from tests.contracts import test_generic_postproposal as fixtures
from mooneural.training.generic_execution_boundary import MethodFallback
from mooneural.training.generic_permanent_pass_executor import (
    PermanentPassExecutor,
    PermanentPassUpdateError,
)
from mooneural.training.generic_postproposal import (
    GuardedPostproposal,
    PostproposalRejected,
)
from mooneural.training.generic_training_contracts import canonical_json, stable_hash

FRACTIONS = (1., .5, .25)
ORIGINAL = "original-segment"
COMPONENT = "supplied-component-equality-witness"


def maximum_losses(parameters, batch, update_index):
    """The rival 1.9+y overtakes 2+x along (-.1, .3, 0)."""
    raw = np.array([max(2. + parameters[0], 1.9 + parameters[1]), .1 + parameters[2]])
    return raw, raw / [2., 3.]


def rival_component(parameters, batch, update_index, context):
    denominator = context["D"][0]
    raw = np.array([1.9 + parameters[1]])
    return {
        "owner_indices": np.array([0]), "cell_ids": ["rival-y"],
        "raw_values": raw, "normalized_values": raw / denominator,
        "normalized_gradients": np.array([[0., 1. / denominator, 0.]]),
        "context": context,
    }


def make_guard(callback=maximum_losses, provider=rival_component, **options):
    return GuardedPostproposal(
        callback, fixtures.batch_identity, [2., 3.], [1., 1.],
        binding={"criterion": "analytic-two-cell-maximum"}, fractions=options.pop("fractions", FRACTIONS),
        component_rows=provider, component_binding={"source": "analytic-rival-gradient"}, **options,
    )


def source_proposal(displacement=(-.1, .3, 0.), descent=(1., 0., 0.), parameters=(0., 0., 0.)):
    values = list(fixtures.proposal(parameters, displacement, descent))
    values[1] = tf.constant([.3, .4, .7], tf.float64)
    values[2] = tf.constant([.5, .6, .8], tf.float64)
    return tuple(values)


def apply_guard(guard, optimizer=None, *, parameters=None, rows=None):
    return guard(
        parameters=np.zeros(3) if parameters is None else parameters,
        batch=(), update_index=6, raw_losses=np.array([2., .1]),
        normalized_losses=np.array([2., .1]) / guard.denominators,
        gradient_rows=np.array([[.5, 0., 0.], [0., 0., 1. / 3.]]) if rows is None else rows,
        optimizer=source_proposal() if optimizer is None else optimizer,
        active_indices=(0,), constraint_indices=(1,),
        gradient_batch_binding=fixtures.batch_identity((), 6),
    )


def assert_source_preserved(updated, source):
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert updated[index] is source[index]
    assert int(updated[3]) == 7


def assert_finite_trigger(receipt, *, valid=True):
    assert receipt["original_segment_valid"] is True
    assert receipt["component_trigger"] == "first-finite-rejection"
    details = receipt["component_fallback"]
    assert details["provider_calls"] == details["witness_calls"] == 1
    assert details["valid"] is valid
    assert details["radius_source"] == "original-valid-segment-norm"


@pytest.mark.parametrize("outcome", ["full", "backtrack", "exhaustion"])
def test_omission_and_false_have_identical_legacy_binding_receipts_and_outputs(outcome):
    def forbidden(*arguments):
        raise AssertionError("disabled finite trigger must remain lazy")

    def losses(parameters, batch, update_index):
        raw, _normalized = maximum_losses(parameters, batch, update_index)
        if outcome == "exhaustion":
            raw[1] += 1e8 * parameters[0] ** 2
        return raw, raw / [2., 3.]

    legacy = make_guard(losses, forbidden)
    explicit = make_guard(losses, forbidden, finite_component_fallback=False)
    assert legacy.binding == explicit.binding
    assert legacy.binding_hash == explicit.binding_hash == stable_hash(legacy.binding)
    assert legacy.binding["maximum_loss_calls"] == 1 + len(FRACTIONS)
    assert legacy.binding["component_fallback"]["trigger"] == "invalid-original-linear-segment-only"
    optimizer = source_proposal(displacement=(-.1, 0., 0.)) if outcome == "full" else source_proposal()
    if outcome == "exhaustion":
        with pytest.raises(PostproposalRejected, match="budget exhausted") as previous:
            apply_guard(legacy, optimizer)
        with pytest.raises(PostproposalRejected, match="budget exhausted") as current:
            apply_guard(explicit, optimizer)
        receipt = current.value.receipt
        assert receipt == previous.value.receipt
    else:
        before, previous_receipt = apply_guard(legacy, optimizer)
        after, receipt = apply_guard(explicit, optimizer)
        assert receipt == previous_receipt
        assert receipt["accepted_fraction"] == (1. if outcome == "full" else .25)
        for actual, expected in zip(after, before, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert_source_preserved(after, optimizer)
    assert "component_trigger" not in receipt and "original_segment_valid" not in receipt
    assert "direction_kind" not in receipt
    assert all("direction_kind" not in trial for trial in receipt["candidates"])
    assert receipt["component_fallback"]["provider_calls"] == 0
    assert explicit.component_descent.experimental_get_tracing_count() == 0


def test_enabled_requires_provider_and_false_without_provider_is_exact_legacy():
    with pytest.raises(ValueError, match="component provider"):
        fixtures.make_guard(finite_component_fallback=True)
    previous, explicit = fixtures.make_guard(), fixtures.make_guard(finite_component_fallback=False)
    assert previous.binding == explicit.binding
    before, before_receipt = fixtures.call(previous)
    after, after_receipt = fixtures.call(explicit)
    assert before_receipt == after_receipt
    for actual, expected in zip(after, before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_enabled_full_original_step_keeps_provider_and_witness_lazy():
    def forbidden(*arguments):
        raise AssertionError("full-step success must not request component rows")

    guard = make_guard(provider=forbidden, finite_component_fallback=True)
    source = source_proposal(displacement=(-.1, 0., 0.))
    updated, receipt = apply_guard(guard, source)
    np.testing.assert_array_equal(updated[0], source[0])
    assert_source_preserved(updated, source)
    assert receipt["accepted_fraction"] == 1. and receipt["loss_calls"] == 2
    assert receipt["original_segment_valid"] is True
    assert receipt["direction_kind"] == ORIGINAL
    assert [trial["direction_kind"] for trial in receipt["candidates"]] == [ORIGINAL]
    assert "component_trigger" not in receipt
    assert receipt["component_fallback"] == {"invoked": False, "provider_calls": 0, "witness_calls": 0}
    assert guard.component_descent.experimental_get_tracing_count() == 0


@pytest.mark.parametrize("blend_required", [False, True])
def test_analytic_maximum_rescue_uses_original_segment_radius_and_preserves_source(blend_required):
    """Equalities give dx=dy<0 and dz=0, scaled to the blended radius."""
    points, requests = [], []

    def losses(parameters, batch, update_index):
        points.append(parameters.copy())
        return maximum_losses(parameters, batch, update_index)

    def provider(parameters, batch, update_index, context):
        assert not parameters.flags.writeable
        assert context["parameters_hash"] == stable_hash([0., 0., 0.])
        assert context["batch_hash"] == stable_hash(fixtures.batch_identity((), 6))
        assert context["update_index"] == 6 and context["coordinate_mode"] == "raw_over_D"
        assert list(context["D"]) == [2., 3.]
        requests.append(context)
        return rival_component(parameters, batch, update_index, context)

    source = source_proposal(displacement=(.1 if blend_required else -.1, .3, 0.))
    guard = make_guard(losses, provider, finite_component_fallback=True)
    updated, receipt = apply_guard(guard, source)
    source_radius = np.sqrt(.1)
    mixing = (.1 + .1 * source_radius) / (.1 + source_radius) if blend_required else 0.
    segment = (1. - mixing) * np.asarray(source[6]) + mixing * np.array([-source_radius, 0., 0.])
    radius = np.linalg.norm(segment)
    expected = np.array([-radius / np.sqrt(2.), -radius / np.sqrt(2.), 0.])
    np.testing.assert_allclose(points, [np.zeros(3), segment, expected], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(updated[0], expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(updated[0], updated[6])
    np.testing.assert_allclose(updated[8], [0.], atol=1e-12)
    assert receipt["blend_fraction"] == pytest.approx(mixing, abs=1e-12)
    assert receipt["component_fallback"]["source_norm"] == pytest.approx(radius, rel=1e-11, abs=1e-12)
    if blend_required:
        assert radius < .8 * source_radius
    assert receipt["candidates"][0]["raw"][0] > 2.
    assert receipt["candidates"][1]["raw"] == pytest.approx([2. - radius / np.sqrt(2.), .1])
    assert [trial["direction_kind"] for trial in receipt["candidates"]] == [ORIGINAL, COMPONENT]
    assert receipt["direction_kind"] == COMPONENT and receipt["accepted_fraction"] == 1.
    assert receipt["loss_calls"] == len(points) == 3 and len(requests) == 1
    assert receipt["component_fallback"]["normalized_values"] == pytest.approx([1.9 / 2.])
    assert_finite_trigger(receipt)
    assert_source_preserved(updated, source)


@pytest.mark.parametrize("failed_task", ["active", "protected"])
def test_all_component_fractions_fail_then_original_viable_shorter_step_is_retained(failed_task):
    """A one-sided quadratic is flat at the parent and only penalizes dy<0."""
    points = []

    def losses(parameters, batch, update_index):
        points.append(parameters.copy())
        penalty = 400. * min(parameters[1], 0.) ** 2
        raw, _normalized = maximum_losses(parameters, batch, update_index)
        raw[0 if failed_task == "active" else 1] += penalty
        return raw, raw / [2., 3.]

    guard = make_guard(losses, finite_component_fallback=True)
    source = source_proposal()
    updated, receipt = apply_guard(guard, source)
    witness = np.array([-np.sqrt(.05), -np.sqrt(.05), 0.])
    expected_points = [np.zeros(3), np.asarray(source[6])]
    expected_points.extend(fraction * witness for fraction in FRACTIONS)
    expected_points.extend(fraction * np.asarray(source[6]) for fraction in FRACTIONS[1:])
    np.testing.assert_allclose(points, expected_points, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(updated[0], [-.025, .075, 0.], rtol=1e-11, atol=1e-12)
    assert [trial["direction_kind"] for trial in receipt["candidates"]] == [
        ORIGINAL, COMPONENT, COMPONENT, COMPONENT, ORIGINAL, ORIGINAL,
    ]
    assert [trial["fraction"] for trial in receipt["candidates"]] == [1., 1., .5, .25, .5, .25]
    assert [trial["accepted"] for trial in receipt["candidates"]] == [False] * 5 + [True]
    assert receipt["direction_kind"] == ORIGINAL and receipt["accepted_fraction"] == .25
    assert receipt["candidates"][-1]["raw"] == pytest.approx([1.975, .1])
    assert receipt["loss_calls"] == len(points) == guard.binding["maximum_loss_calls"] == 7
    if failed_task == "protected":
        assert receipt["candidates"][3]["raw"][1] == pytest.approx(1.35)
        assert receipt["candidates"][3]["raw_over_T"][1] > 1.
        assert receipt["candidates"][3]["raw"][1] / guard.denominators[1] < 1.
    assert_finite_trigger(receipt)
    assert_source_preserved(updated, source)


@pytest.mark.parametrize("geometry", ["opposing", "zero"])
def test_invalid_equality_geometry_preserves_original_shorter_fractions(geometry):
    slope = -2. if geometry == "opposing" else 0.

    def losses(parameters, batch, update_index):
        raw, _normalized = maximum_losses(parameters, batch, update_index)
        raw[0] = max(raw[0], 1.9 + slope * parameters[0])
        return raw, raw / [2., 3.]

    def provider(parameters, batch, update_index, context):
        payload = rival_component(parameters, batch, update_index, context)
        payload["cell_ids"] = [geometry]
        payload["normalized_gradients"] = np.array([[slope / 2., 0., 0.]])
        return payload

    source = source_proposal()
    updated, receipt = apply_guard(make_guard(losses, provider, finite_component_fallback=True), source)
    assert_finite_trigger(receipt, valid=False)
    assert receipt["loss_calls"] == 4 and receipt["accepted_fraction"] == .25
    assert [trial["direction_kind"] for trial in receipt["candidates"]] == [ORIGINAL] * 3
    assert receipt["direction_kind"] == ORIGINAL
    np.testing.assert_allclose(updated[0], [-.025, .075, 0.], rtol=1e-11, atol=1e-12)
    assert_source_preserved(updated, source)


@pytest.mark.parametrize("fault", ["context", "normalization", "shape", "nonfinite", "owner"])
def test_malformed_provider_stops_after_first_finite_refusal_without_shorter_trials(fault):
    def provider(parameters, batch, update_index, context):
        payload = rival_component(parameters, batch, update_index, context)
        if fault == "context":
            payload["context"] = json.loads(canonical_json(context))
            payload["context"]["update_index"] += 1
        elif fault == "normalization":
            payload["normalized_values"] /= context["D"][0]
        elif fault == "shape":
            payload["normalized_gradients"] = np.array([[0., .5]])
        elif fault == "nonfinite":
            payload["normalized_gradients"][0, 1] = np.nan
        else:
            payload["owner_indices"] = np.array([1])
        return payload

    source = source_proposal()
    before = [np.asarray(value).copy() for value in source]
    with pytest.raises(PostproposalRejected) as failure:
        apply_guard(make_guard(provider=provider, finite_component_fallback=True), source)
    receipt = failure.value.receipt
    assert receipt["failure_kind"] == "contract_error" and not receipt["accepted"]
    assert receipt["original_segment_valid"] is True
    assert receipt["component_trigger"] == "first-finite-rejection"
    assert receipt["loss_calls"] == 2 and len(receipt["candidates"]) == 1
    assert receipt["candidates"][0]["direction_kind"] == ORIGINAL
    assert receipt["component_fallback"]["provider_calls"] == 1
    assert receipt["component_fallback"]["witness_calls"] == 0
    for actual, saved in zip(source, before, strict=True):
        np.testing.assert_array_equal(actual, saved)


def test_nonfinite_first_original_trial_is_counted_and_can_be_rescued():
    def losses(parameters, batch, update_index):
        raw, _normalized = maximum_losses(parameters, batch, update_index)
        if parameters[1] > .2:
            raw[0] = np.nan
        return raw, raw / [2., 3.]

    source = source_proposal()
    updated, receipt = apply_guard(make_guard(losses, finite_component_fallback=True), source)
    assert receipt["loss_calls"] == 3 and receipt["candidates"][0]["numerical_failure"]
    assert receipt["direction_kind"] == COMPONENT and receipt["accepted_fraction"] == 1.
    assert_finite_trigger(receipt)
    assert_source_preserved(updated, source)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fraction_count", [1, 3, 11])
def test_nonfinite_exhaustion_counts_every_trial_and_enforces_optional_ceiling(enabled, fraction_count):
    calls, requests = [], []
    fractions = tuple(2. ** -index for index in range(fraction_count))

    def losses(parameters, batch, update_index):
        calls.append(parameters.copy())
        raw = np.array([2., .1]) if np.array_equal(parameters, np.zeros(3)) else np.array([np.nan, .1])
        return raw, raw / [2., 3.]

    def provider(parameters, batch, update_index, context):
        requests.append(update_index)
        return rival_component(parameters, batch, update_index, context)

    guard = make_guard(losses, provider, fractions=fractions, finite_component_fallback=enabled)
    with pytest.raises(PostproposalRejected, match="budget exhausted") as failure:
        apply_guard(guard)
    receipt = failure.value.receipt
    maximum = 1 + fraction_count * (2 if enabled else 1)
    assert receipt["loss_calls"] == len(calls) == guard.binding["maximum_loss_calls"] == maximum
    assert len(receipt["candidates"]) == maximum - 1
    assert all(trial["numerical_failure"] and not trial["accepted"] for trial in receipt["candidates"])
    assert requests == ([6] if enabled else [])
    if enabled:
        assert_finite_trigger(receipt)
        assert [trial["direction_kind"] for trial in receipt["candidates"]] == (
            [ORIGINAL] + [COMPONENT] * fraction_count + [ORIGINAL] * (fraction_count - 1)
        )
    else:
        assert receipt["component_fallback"]["provider_calls"] == 0


@pytest.mark.parametrize("failure_kind", ["nonfinite-parent", "bad-candidate-normalization"])
def test_parent_or_loss_contract_failure_cannot_trigger_component_search(failure_kind):
    def forbidden(*arguments):
        raise AssertionError("a loss contract error must not request components")

    def losses(parameters, batch, update_index):
        raw, normalized = maximum_losses(parameters, batch, update_index)
        if failure_kind == "nonfinite-parent":
            return np.full(2, np.nan), np.full(2, np.nan)
        if np.any(parameters):
            normalized /= [2., 3.]
        return raw, normalized

    with pytest.raises(PostproposalRejected) as failure:
        apply_guard(make_guard(losses, forbidden, finite_component_fallback=True))
    receipt = failure.value.receipt
    assert receipt["loss_calls"] == (1 if failure_kind == "nonfinite-parent" else 2)
    assert not receipt["accepted"] and receipt["component_fallback"]["provider_calls"] == 0


def test_invalid_original_linear_segment_keeps_one_witness_at_source_proposal_radius():
    source = source_proposal(displacement=(.1, .3, 0.), descent=(-1., 0., 0.))
    previous, before_receipt = apply_guard(make_guard(), source)
    updated, receipt = apply_guard(make_guard(finite_component_fallback=True), source)
    for actual, expected in zip(updated, previous, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert receipt["original_segment_valid"] is False and "component_trigger" not in receipt
    assert receipt["loss_calls"] == before_receipt["loss_calls"] == 2
    assert receipt["component_fallback"]["provider_calls"] == receipt["component_fallback"]["witness_calls"] == 1
    assert receipt["component_fallback"]["source_norm"] == pytest.approx(np.sqrt(.1))
    assert [trial["direction_kind"] for trial in receipt["candidates"]] == [COMPONENT]
    assert_source_preserved(updated, source)


def test_invalid_original_segment_exhausts_its_single_witness_without_second_provider_call():
    def losses(parameters, batch, update_index):
        raw, _normalized = maximum_losses(parameters, batch, update_index)
        raw[1] += 400. * min(parameters[1], 0.) ** 2
        return raw, raw / [2., 3.]

    guard = make_guard(losses, finite_component_fallback=True)
    source = source_proposal(displacement=(.1, .3, 0.), descent=(-1., 0., 0.))
    with pytest.raises(PostproposalRejected, match="budget exhausted") as failure:
        apply_guard(guard, source)
    receipt = failure.value.receipt
    assert receipt["original_segment_valid"] is False and "component_trigger" not in receipt
    assert receipt["component_fallback"]["provider_calls"] == receipt["component_fallback"]["witness_calls"] == 1
    assert receipt["loss_calls"] == 1 + len(FRACTIONS) < guard.binding["maximum_loss_calls"]
    assert [trial["direction_kind"] for trial in receipt["candidates"]] == [COMPONENT] * len(FRACTIONS)
    assert not any(trial["accepted"] for trial in receipt["candidates"])


def test_rounded_component_refusals_preserve_viable_original_shorter_fraction():
    """Witness dx rounds away; original half-step retains dx=-2, dy=.5."""
    parameters = np.array([1e16, 0., 0.])
    points = []

    def losses(candidate, batch, update_index):
        points.append(candidate.copy())
        displacement = candidate - parameters
        raw = np.array([
            max(2. + displacement[0] - displacement[1], 1.7 - .1 * displacement[0] + .001 * displacement[2]),
            .1 + .01 * (displacement[0] + displacement[1]),
        ])
        return raw, raw / [2., 3.]

    def provider(candidate, batch, update_index, context):
        return {
            "owner_indices": np.array([0]), "cell_ids": ["rival-negative-x"],
            "raw_values": np.array([1.7]), "normalized_values": np.array([1.7 / 2.]),
            "normalized_gradients": np.array([[-.05, 0., .0005]]), "context": context,
        }

    source = source_proposal(displacement=(-4., 1., 1.), descent=(1., -1., 0.), parameters=parameters)
    rows = np.array([[.5, -.5, 0.], [.01 / 3., .01 / 3., 0.]])
    guard = make_guard(losses, provider, finite_component_fallback=True)
    updated, receipt = apply_guard(guard, source, parameters=parameters, rows=rows)
    assert_finite_trigger(receipt)
    assert receipt["loss_calls"] == len(points) == 6 < guard.binding["maximum_loss_calls"]
    assert receipt["accepted"] and receipt["accepted_fraction"] == .5
    assert receipt["direction_kind"] == ORIGINAL and receipt["blend_fraction"] == 0.
    assert [trial["accepted"] for trial in receipt["candidates"]] == [False] * 4 + [True]
    assert [trial["direction_kind"] for trial in receipt["candidates"]] == [
        ORIGINAL, COMPONENT, COMPONENT, COMPONENT, ORIGINAL,
    ]
    assert [trial["fraction"] for trial in receipt["candidates"]] == [1., 1., .5, .25, .5]
    assert receipt["candidates"][0]["raw"][0] == pytest.approx(2.101)
    for trial, candidate in zip(receipt["candidates"][1:4], points[2:5], strict=True):
        displacement = candidate - parameters
        assert displacement[0] == 0. and displacement[1] > 0.
        assert trial["raw"][0] < 2. and trial["raw"][1] < 1.
        assert trial["reason"] == "rounded displacement failed source protected-dot validity"
        assert trial["protected_dots"] == pytest.approx([displacement[1] / np.sqrt(2.)])
        assert trial["protected_dots"][0] > 1e-10
    np.testing.assert_array_equal(updated[0], parameters + [-2., .5, .5])
    np.testing.assert_array_equal(updated[6], [-2., .5, .5])
    np.testing.assert_allclose(updated[8], [-1.5 / np.sqrt(2.)], rtol=1e-11, atol=1e-12)
    assert receipt["candidates"][-1]["raw"] == pytest.approx([1.9005, .085])
    assert_source_preserved(updated, source)


def test_malformed_finite_trigger_rolls_back_actual_executor_and_method_state(monkeypatch):
    existing, initial, batch = fixtures.fixture(task_count=2)
    parent_parameters = np.array(initial.policy_state["values"])

    def losses(parameters, supplied_batch, update_index):
        raw, _normalized = fixtures.quadratic_batch_losses(parameters, supplied_batch, update_index)
        displacement = parameters - parent_parameters
        raw += 1e24 * np.sum(displacement ** 2) ** 2
        return raw, raw.copy()

    def malformed_provider(*arguments):
        raise ValueError("finite-trigger malformed component")

    guard = GuardedPostproposal(
        losses, fixtures.batch_identity, [1., 1.], [.04, .04],
        binding={"fixture": "quadratic-with-zero-derivative-quartic"},
        component_rows=malformed_provider, component_binding={"source": "malformed-fixture"},
        finite_component_fallback=True,
    )
    executor = PermanentPassExecutor(existing.adapter, existing.spec, fixtures.roles(), threshold=.04, postproposal=guard)
    parent = executor.initialize(initial, fixtures.fake_control(initial, existing.adapter.registry.task_ids, [.2, .2]))
    saved = canonical_json(parent.to_dict())
    source_choose = MethodFallback.choose

    def changed_method_state(method, *arguments):
        result = source_choose(method, *arguments)
        method.rates["cagrad"] *= .5
        method.preferred = "pcgrad"
        method.events.append({"fixture": "uncommitted-state-change"})
        return result

    monkeypatch.setattr(MethodFallback, "choose", changed_method_state)
    with pytest.raises(PermanentPassUpdateError, match="finite-trigger malformed component") as failure:
        executor.step(parent, batch)
    assert failure.value.checkpoint is parent and canonical_json(parent.to_dict()) == saved
    assert failure.value.event["transaction_rolled_back"] and not failure.value.event["committed"]
    receipt = failure.value.event["postproposal"]
    assert receipt["failure_kind"] == "contract_error" and receipt["loss_calls"] == 2
    assert receipt["original_segment_valid"] is True
    assert receipt["component_trigger"] == "first-finite-rejection"
    assert receipt["component_fallback"]["provider_calls"] == 1
    assert receipt["component_fallback"]["witness_calls"] == 0
    boundary, = executor.boundaries.values()
    assert boundary.methods.rates == parent.method_state["rates"]
    assert boundary.methods.preferred == parent.method_state["preferred"]
    assert boundary.methods.events == []
