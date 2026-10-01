"""Complete nine-cell coverage against closed-form weighted values and slopes."""

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_grouped_components import make_grouped_component_provider
from mooneural.training.generic_training_contracts import (
    ImmutableJSONMapping,
    stable_hash,
)

TASKS = tuple(f"task-{index}" for index in range(9))
OWNERS = (0, 4, 8)
CELLS = tuple(f"cell-{index:02d}" for index in range(9))
DENOMINATORS = np.array([2., 3., 5., 7., 11., 13., 17., 19., 23.])


def analytic_provider(*, row_order=tuple(range(18)), column_order=(2, 0, 1), ties=False,
                      row_cap=27, owner_cap=3):
    forwards = tf.Variable(0, dtype=tf.int32, trainable=False)
    reverses = tf.Variable(0, dtype=tf.int32, trainable=False)
    groups = np.repeat(np.arange(9, dtype=np.int32), 2)
    members = np.tile([0., 1.], 9)
    masses = np.array([1., 2., 3., 4., 5., 6., 7., 8., 28.]) / 64.
    probabilities = np.repeat(masses, 2) * np.tile([.25, .75], 9)
    row_ids = [f"{cell}-member-{member}" for cell in CELLS for member in range(2)]

    @tf.custom_gradient
    def counted_identity(values):
        def reverse(upstream):
            with tf.control_dependencies([reverses.assign_add(1)]):
                return tf.identity(upstream)

        return tf.identity(values), reverse

    def raw_rows(parameters):
        with tf.control_dependencies([forwards.assign_add(1)]):
            cell = tf.constant(groups, tf.float64)
            member = tf.constant(members, tf.float64)
            first, second, third = tf.unstack(parameters, num=3)
            if ties:
                first_loss = 8. + 4. * (member - .75) + (cell + 1.) * first + (member + 1.) * second
                second_loss = 16. + 8. * (member - .75) + (cell + 1.) * second + (2. * member - 1.) * third
                third_loss = 32. + 12. * (member - .75) + (cell + 1.) * third + (member + 1.) * first
            else:
                residue = tf.constant(groups % 3, tf.float64)
                first_loss = 3. + cell / 2. + (first + cell / 8. + member) ** 2 + 2. * second ** 2 + third / 4.
                second_loss = (7. + (8. - cell) / 2. + (2. * second - cell / 4. + member / 2.) ** 2
                    + first * third + (member + 1.) * first ** 2)
                third_loss = (5. + residue / 2. + (third + residue / 4. - member / 2.) ** 2
                    + (first - second) ** 2 + member * second / 8.)
            matrix = tf.stack((first_loss, second_loss, third_loss), axis=1)
            return counted_identity(tf.gather(tf.gather(matrix, row_order, axis=0), column_order, axis=1))

    declaration = {
        "task_ids": list(TASKS),
        "aggregation": {task: "max_group_mean" if index in OWNERS else "weighted_mean"
                        for index, task in enumerate(TASKS)},
        "top_k": 9, "max_active_owners": owner_cap, "max_component_rows": row_cap,
        "source": {"fixture": "closed-form-two-member-cells"},
        "input_binding": {"pool": "nine-cells-unequal-masses-and-conditional-probabilities"},
    }
    provider = make_grouped_component_provider(
        raw_rows, [row_ids[index] for index in row_order], probabilities[list(row_order)],
        groups[list(row_order)], CELLS, DENOMINATORS, np.array(OWNERS)[list(column_order)], declaration,
    )
    return provider, forwards, reverses


def request(provider, parameters):
    parameters = np.asarray(parameters, np.float64)
    batch = {"row_ids": list(provider.row_ids), "pool": "complete-cell-fixture"}
    context = ImmutableJSONMapping({
        "parameters_hash": stable_hash(parameters.tolist()), "batch_hash": stable_hash(batch),
        "update_index": 50, "coordinate_mode": "raw_over_D", "D": DENOMINATORS.tolist(),
        "active_indices": [8, 0, 4], "profile_hash": stable_hash(provider.binding),
    })
    return provider(parameters, batch, 50, context)


def analytic_cell(parameters, owner, cell):
    """Conditional E[member]=E[member**2]=3/4; cell mass cancels."""
    first, second, third = parameters
    if owner == 0:
        shifted = first + cell / 8.
        value = 3. + cell / 2. + shifted ** 2 + 1.5 * shifted + .75 + 2. * second ** 2 + third / 4.
        derivative = [2. * shifted + 1.5, 4. * second, .25]
    elif owner == 4:
        shifted = 2. * second - cell / 4.
        value = 7. + (8. - cell) / 2. + shifted ** 2 + .75 * shifted + .1875 + first * third + 1.75 * first ** 2
        derivative = [third + 3.5 * first, 4. * shifted + 1.5, first]
    else:
        shifted = third + (cell % 3) / 4.
        value = 5. + (cell % 3) / 2. + shifted ** 2 - .75 * shifted + .1875 + (first - second) ** 2 + 3. * second / 32.
        derivative = [2. * (first - second), -2. * (first - second) + 3. / 32., 2. * shifted - .75]
    return value, np.array(derivative)


def assert_complete_serial_call(provider, forwards, reverses, result):
    assert result["normalized_gradients"].shape == (27, 3)
    assert len(result["cell_ids"]) == 27
    pairs = list(zip(result["owner_indices"].tolist(), result["cell_ids"], strict=True))
    assert len(set(pairs)) == 27
    assert set(pairs) == {(owner, cell) for owner in OWNERS for cell in CELLS}
    for name in ("raw_values", "normalized_values", "normalized_gradients"):
        assert result[name].dtype == np.float64
    receipt, = provider.receipts
    assert receipt["completed"] and receipt["graph_calls"] == 1
    assert receipt["forward_calls"] == int(forwards) == 1
    assert receipt["reverse_calls"] == int(reverses) == 27
    assert receipt["selected_owners"] == list(OWNERS)
    assert receipt["omitted_tied_cell_ids"] == [[], [], []]
    assert provider.binding["top_k"] == provider.profile["top_k"] == 9
    assert provider.profile["max_component_rows"] == 27
    assert provider.graph.experimental_get_tracing_count() == 1
    graph = provider.graph.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
    loops = [node for node in nodes if node.op in ("While", "StatelessWhile")]
    assert len(loops) == 1 and loops[0].attr["parallel_iterations"].i == 1
    assert int(forwards) == 1 and int(reverses) == 27


@pytest.mark.parametrize("row_order,column_order", [
    (tuple(range(18)), (2, 0, 1)),
    (tuple(range(17, -1, -2)) + tuple(range(16, -1, -2)), (1, 2, 0)),
])
def test_all_nine_cells_match_analytic_weighted_values_and_derivatives_under_permutation(row_order, column_order):
    provider, forwards, reverses = analytic_provider(row_order=row_order, column_order=column_order)
    assert int(forwards) == int(reverses) == provider.graph.experimental_get_tracing_count() == 0
    parameters = np.array([.25, -.5, .75])
    result = request(provider, parameters)
    expected_owners, expected_cells, expected_values, expected_gradients, group_means = [], [], [], [], []
    for owner in OWNERS:
        cells = [analytic_cell(parameters, owner, cell) for cell in range(9)]
        group_means.append([value for value, _derivative in cells])
        for cell in sorted(range(9), key=lambda index: (-cells[index][0], index)):
            value, derivative = cells[cell]
            expected_owners.append(owner)
            expected_cells.append(CELLS[cell])
            expected_values.append(value)
            expected_gradients.append(derivative / DENOMINATORS[owner])
    np.testing.assert_array_equal(result["owner_indices"], expected_owners)
    assert result["cell_ids"] == expected_cells
    np.testing.assert_allclose(result["raw_values"], expected_values, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(result["normalized_values"],
        np.array(expected_values) / DENOMINATORS[expected_owners], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(result["normalized_gradients"], expected_gradients, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(provider.receipts[-1]["group_raw_means"], group_means, rtol=1e-13, atol=1e-14)
    assert_complete_serial_call(provider, forwards, reverses, result)


def test_all_cell_ties_preserve_declared_order_and_distinct_derivatives_without_omissions():
    provider, forwards, reverses = analytic_provider(ties=True, row_order=tuple(range(17, -1, -1)))
    result = request(provider, np.zeros(3))
    np.testing.assert_array_equal(result["owner_indices"], np.repeat(OWNERS, 9))
    assert result["cell_ids"] == list(CELLS) * 3
    np.testing.assert_array_equal(result["raw_values"], np.repeat([8., 16., 32.], 9))
    np.testing.assert_allclose(result["normalized_values"], np.repeat([8. / 2., 16. / 11., 32. / 23.], 9), rtol=1e-13)
    expected_gradients = (
        [[cell + 1., 1.75, 0.] for cell in range(9)]
        + [[0., cell + 1., .5] for cell in range(9)]
        + [[1.75, 0., cell + 1.] for cell in range(9)]
    )
    np.testing.assert_allclose(result["normalized_gradients"],
        np.array(expected_gradients) / np.repeat(DENOMINATORS[list(OWNERS)], 9)[:, None], rtol=1e-12, atol=1e-14)
    assert provider.receipts[-1]["maximum_tie_counts"] == [9, 9, 9]
    assert provider.receipts[-1]["cutoff_tie_counts"] == [9, 9, 9]
    assert_complete_serial_call(provider, forwards, reverses, result)


@pytest.mark.parametrize("row_cap,owner_cap", [(26, 3), (27, 2)])
def test_complete_cell_call_caps_reject_before_callback_trace_or_any_compute(row_cap, owner_cap):
    provider, forwards, reverses = analytic_provider(row_cap=row_cap, owner_cap=owner_cap)
    with pytest.raises(ValueError, match="owner/reverse-call budget exceeded"):
        request(provider, [.25, -.5, .75])
    assert int(forwards) == int(reverses) == 0
    assert provider.graph.experimental_get_tracing_count() == 0
    assert provider.receipts == []
