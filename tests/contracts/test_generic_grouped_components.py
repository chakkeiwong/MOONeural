"""Analytic grouped parameter VJPs and bounded supplied-row integration."""

import json

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_component_descent import make_component_descent_function
from mooneural.training.generic_grouped_components import make_grouped_component_provider
from mooneural.training.generic_postproposal import GuardedPostproposal
from mooneural.training.generic_training_contracts import (
    ImmutableJSONMapping,
    stable_hash,
)


def profile(tasks=("weighted", "first", "unused", "second"), **overrides):
    result = {"task_ids": list(tasks), "aggregation": dict.fromkeys(tasks, "max_group_mean"),
        "max_active_owners": 3, "max_component_rows": 6,
        "source": {"fixture": "analytic-quadratics"}, "input_binding": {"pool": "unequal-mass"}}
    result["aggregation"][tasks[0]] = "weighted_mean"
    result.update(overrides)
    return result


def analytic_provider(*, maximum_owners=3, all_max=False, column_order=(0, 1, 2), profile_overrides=None):
    forwards = tf.Variable(0, dtype=tf.int32, trainable=False)

    def raw_rows(parameters):
        with tf.control_dependencies([forwards.assign_add(1)]):
            first = tf.constant([1., 2., 3., 4.], tf.float64) * (
                parameters[0] - tf.constant([0., 1., -1., 2.], tf.float64)) ** 2 + tf.constant([1., 2., 3., 4.], tf.float64)
            second = tf.constant([2., 1., 4., 3.], tf.float64) * (
                parameters[1] + tf.constant([1., -1., .5, 2.], tf.float64)) ** 2 + tf.constant([2., 1., 3., 1.], tf.float64)
            weighted = tf.reduce_sum(parameters**2) + tf.constant([5., 6., 7., 8.], tf.float64)
            return tf.gather(tf.stack((first, second, weighted), axis=1), column_order, axis=1)

    declaration = profile(max_active_owners=maximum_owners)
    if all_max:
        declaration["aggregation"]["weighted"] = "max_group_mean"
    if profile_overrides is not None:
        declaration.update(profile_overrides)
    provider = make_grouped_component_provider(raw_rows, ["r0", "r1", "r2", "r3"],
        np.array([.1, .3, .2, .4]), np.array([0, 0, 1, 2]), ["low", "middle", "high"],
        np.array([7., 2., 11., 5.]), np.array([1, 3, 0])[list(column_order)], declaration)
    return provider, forwards


def request(provider, parameters, active, *, fault=None):
    parameters = np.asarray(parameters, np.float64)
    batch = {"pool": "unequal-mass", "update": 21}
    context = {"parameters_hash": stable_hash(parameters.tolist()), "batch_hash": stable_hash(batch),
        "update_index": 21, "D": provider.denominators.tolist(), "coordinate_mode": "raw_over_D",
        "active_indices": active, "profile_hash": stable_hash({"fixture": "guard-profile"})}
    if fault == "parent":
        context["parameters_hash"] = "stale"
    elif fault == "batch":
        batch["pool"] = "different"
    elif fault == "update":
        context["update_index"] = 20
    elif fault == "D":
        context["D"][0] *= 2.
    elif fault == "coordinate":
        context["coordinate_mode"] = "raw_over_T"
    return provider(parameters, batch, 21, ImmutableJSONMapping(context))


def analytic_expected(parameters, owners, *, top_k=2):
    first, second = parameters
    raw_first = np.array([(first - 0.) ** 2 + 1., 2. * (first - 1.) ** 2 + 2.,
                          3. * (first + 1.) ** 2 + 3., 4. * (first - 2.) ** 2 + 4.])
    raw_second = np.array([2. * (second + 1.) ** 2 + 2., (second - 1.) ** 2 + 1.,
                           4. * (second + .5) ** 2 + 3., 3. * (second + 2.) ** 2 + 1.])
    row_values = {1: raw_first, 3: raw_second, 0: first**2 + second**2 + np.array([5., 6., 7., 8.])}
    row_derivatives = {1: np.column_stack(([2. * first, 4. * (first - 1.), 6. * (first + 1.), 8. * (first - 2.)], np.zeros(4))),
        3: np.column_stack((np.zeros(4), [4. * (second + 1.), 2. * (second - 1.), 8. * (second + .5), 6. * (second + 2.)])),
        0: np.tile([2. * first, 2. * second], (4, 1))}
    denominators, names = {0: 7., 1: 2., 3: 5.}, ["low", "middle", "high"]
    values, gradients, cells, selected_owners = [], [], [], []
    for owner in sorted(owners):
        raw, derivative = row_values[owner], row_derivatives[owner]
        means = np.array([.25 * raw[0] + .75 * raw[1], raw[2], raw[3]])
        slopes = np.stack((.25 * derivative[0] + .75 * derivative[1], derivative[2], derivative[3]))
        for cell in sorted(range(3), key=lambda index: (-means[index], index))[:top_k]:
            selected_owners.append(owner)
            cells.append(names[cell])
            values.append(means[cell])
            gradients.append(slopes[cell] / denominators[owner])
    return np.array(selected_owners), cells, np.array(values), np.array(gradients)


def test_serial_matrix_vjp_matches_independent_nonunit_D_unequal_mass_derivatives():
    provider, forwards = analytic_provider()
    assert int(forwards) == 0 and provider.graph.experimental_get_tracing_count() == 0
    for active in ([0, 1], [3, 1, 0], [3]):
        result = request(provider, [.2, -.3], active)
        owners = sorted(set(active) & {1, 3})
        expected_owners, cells, raw, gradients = analytic_expected([.2, -.3], owners)
        np.testing.assert_array_equal(result["owner_indices"], expected_owners)
        assert result["cell_ids"] == cells
        np.testing.assert_allclose(result["raw_values"], raw, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(result["normalized_values"], raw / provider.denominators[expected_owners], rtol=1e-13)
        np.testing.assert_allclose(result["normalized_gradients"], gradients, rtol=1e-12, atol=1e-14)
        assert provider.receipts[-1]["reverse_calls"] == 2 * len(owners)
        assert provider.receipts[-1]["forward_calls"] == 1
    assert int(forwards) == 3 and provider.graph.experimental_get_tracing_count() == 1
    json.dumps(provider.receipts, allow_nan=False)


def test_three_owners_compute_six_reverses_and_callback_column_permutation_matches():
    original, forwards = analytic_provider(all_max=True)
    permuted, other_forwards = analytic_provider(all_max=True, column_order=(2, 0, 1))
    first = request(original, [.2, -.3], [3, 0, 1])
    second = request(permuted, [.2, -.3], [1, 3, 0])
    assert original.receipts[-1]["reverse_calls"] == permuted.receipts[-1]["reverse_calls"] == 6
    assert int(forwards) == int(other_forwards) == 1
    assert first["cell_ids"] == second["cell_ids"]
    for name in ("owner_indices", "raw_values", "normalized_values", "normalized_gradients"):
        np.testing.assert_allclose(first[name], second[name], rtol=1e-12, atol=1e-14)


def test_empty_or_weighted_selection_invokes_no_callback_or_graph():
    provider, forwards = analytic_provider()
    for active in ([], [0], [2]):
        result = request(provider, [.2, -.3], active)
        assert result["normalized_gradients"].shape == (0, 2)
        assert not result["cell_ids"] and result["owner_indices"].size == 0
        assert provider.receipts[-1]["forward_calls"] == provider.receipts[-1]["reverse_calls"] == 0
    assert int(forwards) == 0 and provider.graph.experimental_get_tracing_count() == 0


def test_fixed_top_two_reports_omitted_ties_without_extra_reverses():
    def raw_rows(parameters):
        return tf.stack((2. + parameters[0], 2. + parameters[1], 2. + parameters[0] + parameters[1]))[:, None]

    declaration = profile(tasks=("maximum",))
    declaration["aggregation"]["maximum"] = "max_group_mean"
    provider = make_grouped_component_provider(raw_rows, ["r0", "r1", "r2"], np.array([.1, .3, .6]),
        np.arange(3), ["first", "second", "omitted"], np.array([2.]), np.array([0]), declaration)
    result = request(provider, [0., 0.], [0])
    assert result["cell_ids"] == ["first", "second"]
    np.testing.assert_allclose(result["normalized_gradients"], [[.5, 0.], [0., .5]], atol=1e-15)
    receipt = provider.receipts[-1]
    assert receipt["maximum_tie_counts"] == receipt["cutoff_tie_counts"] == [3]
    assert receipt["omitted_tied_cell_ids"] == [["omitted"]]
    assert receipt["reverse_calls"] == 2


def test_correct_tied_provider_does_not_claim_equality_witness_feasibility():
    def raw_rows(parameters):
        return (2. + parameters)[:, None]

    declaration = profile(tasks=("maximum",))
    declaration["aggregation"]["maximum"] = "max_group_mean"
    provider = make_grouped_component_provider(raw_rows, ["r0", "r1"], np.array([.2, .8]), np.array([0, 1]),
        ["first", "second"], np.array([1.]), np.array([0]), declaration)
    result = request(provider, [0., 0.], [0])
    np.testing.assert_allclose(result["normalized_gradients"], np.eye(2), atol=1e-15)
    witness = make_component_descent_function(1)(np.array([[.5, .5]]), np.array([True]),
        result["normalized_gradients"], np.array([-.1, -.1]))
    assert not bool(witness["valid"]) and provider.receipts[-1]["completed"]
    assert np.all(result["normalized_gradients"] @ np.array([-1., -1.]) < 0.)


@pytest.mark.parametrize("fault", ["parent", "batch", "update", "D", "coordinate", "duplicate_owner", "owner_range", "budget"])
def test_context_and_owner_errors_refuse_before_forward(fault):
    provider, forwards = analytic_provider(maximum_owners=1 if fault == "budget" else 3)
    active = [1, 3]
    if fault == "duplicate_owner":
        active = [1, 1]
    elif fault == "owner_range":
        active = [4]
    with pytest.raises(ValueError):
        request(provider, [.2, -.3], active, fault=fault)
    assert int(forwards) == 0 and provider.graph.experimental_get_tracing_count() == 0


@pytest.mark.parametrize("fault", ["disconnected", "nonfinite"])
def test_invalid_raw_callback_is_not_a_successful_zero_derivative(fault):
    def raw_rows(parameters):
        if fault == "disconnected":
            return tf.ones([2, 1], tf.float64)
        return tf.fill([2, 1], tf.constant(float("nan"), tf.float64)) + tf.reduce_sum(parameters) * 0.

    declaration = profile(tasks=("maximum",))
    declaration["aggregation"]["maximum"] = "max_group_mean"
    provider = make_grouped_component_provider(raw_rows, ["r0", "r1"], np.array([.2, .8]), np.array([0, 1]),
        ["first", "second"], np.array([1.]), np.array([0]), declaration)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="disconnected|nonfinite"):
        request(provider, [0., 0.], [0])
    assert not provider.receipts[-1]["completed"]
    assert provider.receipts[-1]["reverse_calls"] is None


def test_frozen_callback_binding_and_immutable_inputs():
    provider, forwards = analytic_provider()
    with pytest.raises(ValueError):
        provider.probabilities[0] = .9
    provider.raw_rows_fn = lambda parameters: tf.ones([4, 3], tf.float64)
    with pytest.raises(ValueError, match="callback/profile changed"):
        request(provider, [.2, -.3], [1])
    assert int(forwards) == 0


@pytest.mark.parametrize("invalid_segment", [False, True])
def test_guard_passes_actual_partition_and_preserves_slots_with_lazy_grouped_provider(invalid_segment):
    forwards = tf.Variable(0, dtype=tf.int32, trainable=False)

    def raw_rows(parameters):
        with tf.control_dependencies([forwards.assign_add(1)]):
            return tf.stack((2. + parameters[0], 1.9 + parameters[3]))[:, None]

    tasks = ("objective", "first-protected", "second-protected")
    declaration = profile(tasks=tasks, aggregation={"objective": "max_group_mean",
        "first-protected": "weighted_mean", "second-protected": "weighted_mean"})
    denominators = np.array([2., 3., 5.])
    provider = make_grouped_component_provider(raw_rows, ["first-row", "second-row"], np.array([.3, .7]),
        np.array([0, 1]), ["winner", "rival"], denominators, np.array([0]), declaration)

    def losses(parameters, batch, update):
        raw = np.array([max(2. + parameters[0], 1.9 + parameters[3]), .1 + parameters[1], .2 + parameters[2]])
        return raw, raw / denominators

    guard = GuardedPostproposal(losses, lambda batch, update: batch, denominators, np.ones(3),
        binding={"source": "analytic-max-and-weighted"}, component_rows=provider, component_binding=provider.binding)
    displacement = tf.constant([1., 0., 0., 0.], tf.float64)
    descent = -displacement if invalid_segment else displacement
    original = (displacement, tf.ones(4, tf.float64), tf.fill([4], tf.constant(.5, tf.float64)),
        tf.constant(7, tf.int64), descent, displacement, displacement, tf.zeros(2, tf.float64),
        tf.zeros(2, tf.float64), tf.constant(0., tf.float64), tf.constant(True))
    batch = {"pool": "analytic-max-and-weighted", "update": 21}
    raw = np.array([2., .1, .2])
    updated, receipt = guard(parameters=np.zeros(4), batch=batch, update_index=21, raw_losses=raw,
        normalized_losses=raw / denominators,
        gradient_rows=np.array([[.5, 0., 0., 0.], [0., 1. / 3., 0., 0.], [0., 0., .2, 0.]]),
        optimizer=original, active_indices=(0,), constraint_indices=(1, 2), gradient_batch_binding=batch)
    assert receipt["accepted"] and receipt["loss_calls"] == 2
    assert receipt["component_fallback"]["provider_calls"] == int(invalid_segment)
    assert int(forwards) == int(invalid_segment)
    for index in (1, 2, 3, 4, 5, 7, 9, 10):
        assert updated[index] is original[index]
    if invalid_segment:
        assert provider.receipts[-1]["context"]["active_indices"] == [0]
        assert provider.receipts[-1]["reverse_calls"] == 2
    else:
        assert not provider.receipts and provider.graph.experimental_get_tracing_count() == 0


def test_omitted_top_k_preserves_legacy_binding_and_explicit_two_preserves_numerics():
    provider, forwards = analytic_provider()
    expected = {"schema": "generic_neural_solver.grouped_components.v1", "profile": profile(),
        "row_ids": ["r0", "r1", "r2", "r3"], "cell_ids": ["low", "middle", "high"],
        "probabilities": [.1, .3, .2, .4], "group_ids": [0, 0, 1, 2], "D": [7., 2., 11., 5.],
        "owner_indices": [1, 3, 0], "coordinate_mode": "raw_over_D", "top_k": 2,
        "tie_policy": "declared-cell-order-report-omitted", "winner_reuse": False,
        "graph": "persistent-full-matrix-tape-serial-selected-vjp", "empty_selection": "zero-calls",
        "callback": {"module": provider.raw_rows_fn.__module__, "qualname": "analytic_provider.<locals>.raw_rows"}}
    assert stable_hash(provider.binding) == stable_hash(expected)
    assert "top_k" not in provider.profile
    explicit, other_forwards = analytic_provider(profile_overrides={"top_k": 2})
    assert explicit.profile["top_k"] == explicit.binding["top_k"] == 2
    assert stable_hash(explicit.binding) != stable_hash(provider.binding)
    original = request(provider, [.2, -.3], [1, 3])
    selected = request(explicit, [.2, -.3], [1, 3])
    assert original["cell_ids"] == selected["cell_ids"]
    for name in ("owner_indices", "raw_values", "normalized_values", "normalized_gradients"):
        np.testing.assert_array_equal(original[name], selected[name])
    assert provider.receipts[-1]["reverse_calls"] == explicit.receipts[-1]["reverse_calls"] == 4
    assert int(forwards) == int(other_forwards) == 1


@pytest.mark.parametrize("column_order", [(0, 1, 2), (2, 0, 1)])
def test_top_three_analytic_values_gradients_order_and_dynamic_counts(column_order):
    provider, forwards = analytic_provider(all_max=True, column_order=column_order,
        profile_overrides={"top_k": 3, "max_component_rows": 9})
    assert provider.binding["top_k"] == provider.profile["top_k"] == 3
    for active in ([0, 1], [3, 1, 0], [3]):
        result = request(provider, [.2, -.3], active)
        expected_owners, cells, raw, gradients = analytic_expected([.2, -.3], active, top_k=3)
        np.testing.assert_array_equal(result["owner_indices"], expected_owners)
        assert result["cell_ids"] == cells
        np.testing.assert_allclose(result["raw_values"], raw, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(result["normalized_values"], raw / provider.denominators[expected_owners], rtol=1e-13)
        np.testing.assert_allclose(result["normalized_gradients"], gradients, rtol=1e-12, atol=1e-14)
        assert provider.receipts[-1]["reverse_calls"] == 3 * len(active)
        assert provider.receipts[-1]["forward_calls"] == 1
        assert provider.receipts[-1]["omitted_tied_cell_ids"] == [[] for _owner in active]
    assert int(forwards) == 3 and provider.graph.experimental_get_tracing_count() == 1
    result = request(provider, [.2, -.3], [])
    assert result["normalized_gradients"].shape == (0, 2)
    assert provider.receipts[-1]["reverse_calls"] == provider.receipts[-1]["forward_calls"] == 0
    assert int(forwards) == 3


@pytest.mark.parametrize("top_k", [0, -1, True, False, 2., "3", None, np.int64(3)])
def test_top_k_requires_positive_plain_integer_before_callback_or_trace(top_k):
    def forbidden(parameters):
        raise AssertionError("invalid top_k must not call the raw-row callback")

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        make_grouped_component_provider(forbidden, ["row"], np.array([1.]), np.array([0]), ["cell"],
            np.array([1.]), np.array([0]), profile(tasks=("maximum",), top_k=top_k))


@pytest.mark.parametrize("cap", [2, 8])
def test_top_three_respects_declared_reverse_cap_before_any_forward(cap):
    provider, forwards = analytic_provider(all_max=True, profile_overrides={"top_k": 3, "max_component_rows": cap})
    active = [1] if cap == 2 else [0, 1, 3]
    with pytest.raises(ValueError, match="owner/reverse-call budget exceeded"):
        request(provider, [.2, -.3], active)
    assert int(forwards) == 0 and provider.graph.experimental_get_tracing_count() == 0
    assert provider.receipts == []


@pytest.mark.parametrize("top_k", [1, 3])
def test_single_group_uses_one_reverse_and_captured_top_k_is_immutable(top_k):
    forwards = tf.Variable(0, dtype=tf.int32, trainable=False)

    def raw_rows(parameters):
        with tf.control_dependencies([forwards.assign_add(1)]):
            return tf.stack((2. + parameters[0], 4. + 3. * parameters[0]))[:, None]

    declaration = profile(tasks=("maximum",), aggregation={"maximum": "max_group_mean"},
                          top_k=top_k, max_component_rows=1)
    provider = make_grouped_component_provider(raw_rows, ["first", "second"], np.array([.25, .75]),
        np.array([0, 0]), ["only"], np.array([5.]), np.array([0]), declaration)
    saved_binding = stable_hash(provider.binding)
    declaration["top_k"] = 7
    assert provider.profile["top_k"] == provider.binding["top_k"] == top_k
    assert stable_hash(provider.binding) == saved_binding
    with pytest.raises(TypeError):
        provider.profile["top_k"] = 7
    result = request(provider, [.2], [0])
    assert result["cell_ids"] == ["only"]
    np.testing.assert_allclose(result["raw_values"], [4.], atol=1e-14)
    np.testing.assert_allclose(result["normalized_values"], [.8], atol=1e-14)
    np.testing.assert_allclose(result["normalized_gradients"], [[.5]], atol=1e-14)
    assert int(forwards) == provider.receipts[-1]["forward_calls"] == provider.receipts[-1]["reverse_calls"] == 1


def test_top_three_keeps_fixed_tie_order_and_reports_fourth_omission():
    def raw_rows(parameters):
        return (2. + parameters)[:, None]

    declaration = profile(tasks=("maximum",), aggregation={"maximum": "max_group_mean"},
                          top_k=3, max_component_rows=3)
    provider = make_grouped_component_provider(raw_rows, ["r0", "r1", "r2", "r3"], np.array([.1, .2, .3, .4]),
        np.arange(4), ["first", "second", "third", "omitted"], np.array([2.]), np.array([0]), declaration)
    result = request(provider, np.zeros(4), [0])
    assert result["cell_ids"] == ["first", "second", "third"]
    np.testing.assert_allclose(result["normalized_gradients"], .5 * np.eye(4)[:3], atol=1e-15)
    receipt = provider.receipts[-1]
    assert receipt["maximum_tie_counts"] == receipt["cutoff_tie_counts"] == [4]
    assert receipt["omitted_tied_cell_ids"] == [["omitted"]]
    assert receipt["reverse_calls"] == 3
