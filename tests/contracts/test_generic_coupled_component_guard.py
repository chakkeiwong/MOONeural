"""Nonlinear cell descent, fixed identities and loss-only aggregation."""

import copy

import numpy as np
import pytest
import tensorflow as tf
from tests.contracts import test_generic_grouped_components as grouped
from tests.contracts import test_generic_postproposal as fixtures
from mooneural.training.generic_grouped_components import make_grouped_component_provider
from mooneural.training.generic_postproposal import (
    GuardedPostproposal,
    PostproposalRejected,
)
from mooneural.training.generic_training_contracts import stable_hash


def loss_context(payload, candidate):
    return {**payload["context"], "candidate_parameters_hash": stable_hash(candidate.tolist()),
        "owner_indices": payload["owner_indices"].tolist(), "cell_ids": payload["cell_ids"]}


def test_loss_only_preserves_conditional_weighting_and_parent_cell_order_without_reverses():
    provider, forwards = grouped.analytic_provider()
    parent = np.array([.2, -.3])
    payload = grouped.request(provider, parent, [1, 3])
    context = loss_context(payload, parent)
    batch = {"pool": "unequal-mass", "update": 21}
    assert provider.loss_graph.experimental_get_tracing_count() == 0
    result = provider.loss_only(parent, batch, 21, context, payload["owner_indices"], payload["cell_ids"])
    np.testing.assert_array_equal(result["raw_values"], payload["raw_values"])
    assert len(provider.receipts) == 1
    candidate = np.array([4., -3.])
    result = provider.loss_only(candidate, batch, 21, loss_context(payload, candidate),
        payload["owner_indices"], payload["cell_ids"])
    owners, cells, raw, _gradients = grouped.analytic_expected(candidate, [1, 3], top_k=3)
    lookup = dict(zip(zip(owners, cells, strict=True), raw, strict=True))
    expected = [lookup[pair] for pair in zip(payload["owner_indices"], payload["cell_ids"], strict=True)]
    np.testing.assert_allclose(result["raw_values"], expected, rtol=1e-13)
    assert provider.loss_graph.experimental_get_tracing_count() == 1 and int(forwards) == 3
    assert all(receipt["reverse_calls"] == 0 for receipt in provider.loss_only_receipts)
    for field in ("D", "profile_hash", "candidate_parameters_hash", "cell_ids"):
        bad = copy.deepcopy(loss_context(payload, candidate))
        bad[field] = []
        with pytest.raises(ValueError):
            provider.loss_only(candidate, batch, 21, bad, payload["owner_indices"], payload["cell_ids"])
    assert int(forwards) == 3


def nonlinear_guard(*, enabled=True, loss_callback=None, fractions=(1., .25, .0625)):
    def raw_rows(parameters):
        return tf.stack((10. + parameters[0], 2. + parameters[1] + 10. * parameters[1] ** 2))[:, None]

    profile = {"task_ids": ["maximum"], "aggregation": {"maximum": "max_group_mean"},
        "top_k": 2, "max_active_owners": 1, "max_component_rows": 2,
        "source": {"fixture": "quadratic"}, "input_binding": {"pool": "two-cells"}}
    provider = make_grouped_component_provider(raw_rows, ["row0", "row1"], np.array([.4, .6]),
        np.array([0, 1]), ["winner", "rival"], np.array([100.]), np.array([0]), profile)

    def losses(parameters, batch, update):
        raw = np.array([max(10. + parameters[0], 2. + parameters[1] + 10. * parameters[1] ** 2)])
        return raw, raw / 100.

    options = {"coupled_component_guard": True, "component_loss_values": loss_callback or provider.loss_only,
               "component_loss_binding": provider.binding} if enabled else {}
    guard = GuardedPostproposal(losses, lambda batch, update: batch, [100.], [1.],
        binding={"fixture": "quadratic"}, fractions=fractions, component_rows=provider,
        component_binding=provider.binding, component_direction="relative_public_progress", **options)
    return guard, provider


def apply(guard):
    source = fixtures.proposal(displacement=(-1., 0.))
    result, receipt = guard(parameters=np.zeros(2), batch={}, update_index=0, raw_losses=np.array([10.]),
        normalized_losses=np.array([.1]), gradient_rows=np.array([[.01, 0.]]), optimizer=source,
        active_indices=(0,), constraint_indices=(), gradient_batch_binding={})
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert result[index] is source[index]
    return result, receipt


def test_coupled_guard_rejects_public_decrease_with_cell_growth_then_backtracks():
    guard, provider = nonlinear_guard()
    updated, receipt = apply(guard)
    assert receipt["accepted_fraction"] == .25
    assert receipt["candidates"][0]["raw"][0] < 10.
    assert receipt["candidates"][0]["component"]["raw_values"][1] > 2.
    assert [candidate["accepted"] for candidate in receipt["candidates"]] == [False, True]
    assert receipt["component_fallback"]["required_components"] == [True, True]
    assert receipt["component_guard"]["loss_calls"] == 2 and receipt["loss_calls"] == 3
    assert provider.receipts[0]["reverse_calls"] == 2 and len(provider.receipts) == 1
    assert len(provider.loss_only_receipts) == 2
    assert receipt["component_guard"]["cell_ids"] == ["winner", "rival"]
    assert 2. + float(updated[0][1]) + 10. * float(updated[0][1]) ** 2 < 2.
    old, old_provider = nonlinear_guard(enabled=False)
    _updated, old_receipt = apply(old)
    assert old_receipt["accepted_fraction"] == 1.
    assert old_provider.loss_only_receipts == [] and "component_guard" not in old_receipt


@pytest.mark.parametrize("fault", ["nonfinite", "dtype", "shape", "cell", "owner", "context", "normalization", "batch"])
def test_bad_cell_receipts_never_accept_and_only_nonfinite_trials_can_backtrack(fault):
    _reference_guard, provider = nonlinear_guard()
    calls = []

    def callback(parameters, batch, update, context, owners, cells):
        payload = provider.loss_only(parameters, batch, update, context, owners, cells)
        calls.append(context)
        if fault == "nonfinite" and len(calls) == 1:
            payload["raw_values"] = np.full(2, np.nan)
        elif fault == "dtype":
            payload["raw_values"] = payload["raw_values"].astype(np.float32)
        elif fault == "shape":
            payload["raw_values"] = payload["raw_values"][:1]
        elif fault == "cell":
            payload["cell_ids"] = ["rival", "winner"]
        elif fault == "owner":
            payload["owner_indices"] = np.array([0, 1])
        elif fault == "context":
            payload["context"] = {**context, "parameters_hash": "stale"}
        elif fault == "normalization":
            payload["normalized_values"] *= 2.
        elif fault == "batch":
            batch["changed"] = True
        return payload

    guard, _provider = nonlinear_guard(loss_callback=callback)
    if fault == "nonfinite":
        _result, receipt = apply(guard)
        assert receipt["accepted_fraction"] == .25
        assert receipt["candidates"][0]["numerical_failure"]
    else:
        with pytest.raises(PostproposalRejected) as rejected:
            apply(guard)
        assert rejected.value.receipt["failure_kind"] == "contract_error"
        assert not any(candidate["accepted"] for candidate in rejected.value.receipt["candidates"])
        assert len(calls) == 1


def test_exhausted_cell_fractions_refuse_without_weaker_fallback():
    guard, _provider = nonlinear_guard(fractions=(1.,))
    with pytest.raises(PostproposalRejected, match="budget exhausted") as rejected:
        apply(guard)
    assert rejected.value.receipt["component_guard"]["loss_calls"] == 1
    assert not rejected.value.receipt["accepted"]


def test_threshold_mask_includes_equality_uses_T_and_can_be_empty():
    for limit, expected in ((1., [True, False, True, True]), (11., [False] * 4)):
        def components(parameters, batch, update, context):
            raw = np.array([10., .999, 1., 1.001]) + parameters[0]
            return {"owner_indices": np.zeros(4, np.int32), "cell_ids": ["max", "below", "equal", "above"],
                "raw_values": raw, "normalized_values": raw / 100.,
                "normalized_gradients": np.tile([.01, 0.], (4, 1)), "context": context}

        def cells(parameters, batch, update, context, owners, ids):
            assert ids == ("max", "equal", "above")
            raw = np.array([10., 1., 1.001]) + parameters[0]
            return {"owner_indices": owners, "cell_ids": ids, "raw_values": raw,
                "normalized_values": raw / 100., "context": context}

        guard = GuardedPostproposal(lambda parameters, batch, update: (
            np.array([10. + parameters[0]]), np.array([(10. + parameters[0]) / 100.])),
            lambda batch, update: batch, [100.], [limit], binding={"fixture": "threshold"},
            component_rows=components, component_binding={"fixture": "threshold"},
            component_direction="relative_public_progress", coupled_component_guard=True,
            component_loss_values=cells, component_loss_binding={"fixture": "threshold"})
        _updated, receipt = apply(guard)
        assert receipt["component_fallback"]["required_components"] == expected
        assert receipt["component_guard"]["loss_calls"] == int(limit == 1.)


def test_missing_callback_and_nonrelative_mode_fail_before_numerical_work():
    with pytest.raises(ValueError, match="gradient and loss-only"):
        GuardedPostproposal(lambda *args: None, lambda *args: None, [1.], [1.],
            binding={"test": True}, coupled_component_guard=True)
