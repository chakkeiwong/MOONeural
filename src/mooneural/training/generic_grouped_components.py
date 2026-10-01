"""Bounded component gradients from an existing grouped raw-row callback."""

import json
from collections.abc import Mapping

import numpy as np
import tensorflow as tf

from .generic_training_contracts import (
    ImmutableJSONMapping,
    canonical_json,
    stable_hash,
)


def _frozen_array(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _group_means(raw_rows, probabilities, groups, group_count, denominators):
    probabilities_tensor = tf.constant(probabilities, tf.float64)
    groups_tensor = tf.constant(groups, tf.int32)
    mass = tf.math.unsorted_segment_sum(probabilities_tensor, groups_tensor, group_count)
    totals = tf.math.unsorted_segment_sum(raw_rows * probabilities_tensor[:, None], groups_tensor, group_count)
    raw_means = totals / mass[:, None]
    return raw_means, raw_means / tf.constant(denominators, tf.float64)[None, :]


class GroupedComponentProvider:
    """Supply a declared number of components for active maximum tasks.

    ``owner_indices`` maps raw callback columns into ``max_profile['task_ids']``.
    The profile also requires aggregation (a task-ID mapping), max_active_owners,
    max_component_rows, source and input_binding. Optional top_k is a positive
    integer, defaulting to two without adding a key to the supplied profile.
    The batch argument is the
    canonical gradient-batch descriptor whose hash appears in the guard context.
    ``binding`` is passed as the guard's component_binding; ``receipts`` records
    selections, omitted ties and completed forward/reverse counts.
    """

    def __init__(self, raw_rows_fn, row_ids, probabilities, group_ids, cell_ids, D, owner_indices, max_profile):
        required = {"task_ids", "aggregation", "max_active_owners", "max_component_rows", "source", "input_binding"}
        if (not callable(raw_rows_fn) or not isinstance(max_profile, Mapping)
                or set(max_profile) not in (required, required | {"top_k"})):
            raise ValueError("raw-row callback and complete explicit maximum-component profile required")
        top_k = max_profile.get("top_k", 2)
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        profile = ImmutableJSONMapping(max_profile)
        tasks = tuple(profile["task_ids"])
        if (not tasks or any(not isinstance(task, str) or not task for task in tasks)
                or len(set(tasks)) != len(tasks) or not isinstance(profile["aggregation"], Mapping)
                or set(profile["aggregation"]) != set(tasks)
                or any(value not in ("max_group_mean", "weighted_mean") for value in profile["aggregation"].values())
                or not profile["source"] or not profile["input_binding"]):
            raise ValueError("unique public tasks, declared aggregations and source/input bindings required")
        for name in ("max_active_owners", "max_component_rows"):
            if type(profile[name]) is not int or profile[name] < 1:
                raise ValueError("positive explicit owner and reverse-call caps required")
        row_ids, cell_ids = tuple(row_ids), tuple(cell_ids)
        if (not row_ids or not cell_ids or len(set(row_ids)) != len(row_ids) or len(set(cell_ids)) != len(cell_ids)
                or any(not isinstance(value, str) or not value for value in (*row_ids, *cell_ids))):
            raise ValueError("nonempty unique ordered row and cell IDs required")
        probabilities, denominators = np.asarray(probabilities), np.asarray(D)
        groups, owners = np.asarray(group_ids), np.asarray(owner_indices)
        if (probabilities.dtype != np.float64 or probabilities.shape != (len(row_ids),)
                or not np.isfinite(probabilities).all() or np.any(probabilities <= 0.)
                or not np.isclose(probabilities.sum(), 1., rtol=0., atol=1e-12)
                or denominators.dtype != np.float64 or denominators.shape != (len(tasks),)
                or not np.isfinite(denominators).all() or np.any(denominators <= 0.)):
            raise ValueError("positive finite float64 probabilities summing to one and public D required")
        if (groups.shape != probabilities.shape or groups.dtype.kind not in "iu"
                or set(groups.tolist()) != set(range(len(cell_ids)))
                or owners.ndim != 1 or not owners.size or owners.dtype.kind not in "iu"
                or len(set(owners.tolist())) != owners.size or np.any(owners < 0) or np.any(owners >= len(tasks))):
            raise ValueError("nonempty groups and unique callback-column/public-owner mapping required")
        self.raw_rows_fn = raw_rows_fn
        self._raw_rows_fn = raw_rows_fn
        self.profile, self.tasks, self.row_ids, self.cell_ids = profile, tasks, row_ids, cell_ids
        self.probabilities = _frozen_array(probabilities, np.float64)
        self.group_ids = _frozen_array(groups, np.int32)
        self.denominators = _frozen_array(denominators, np.float64)
        self.owner_indices = _frozen_array(owners, np.int32)
        self._eligible = tuple(owner for owner in owners.tolist() if profile["aggregation"][tasks[owner]] == "max_group_mean")
        implementation = getattr(raw_rows_fn, "python_function", raw_rows_fn)
        self.binding = ImmutableJSONMapping({"schema": "generic_neural_solver.grouped_components.v1",
            "profile": profile, "row_ids": list(row_ids), "cell_ids": list(cell_ids),
            "probabilities": probabilities.tolist(), "group_ids": groups.tolist(), "D": denominators.tolist(),
            "owner_indices": owners.tolist(), "coordinate_mode": "raw_over_D", "top_k": top_k,
            "tie_policy": "declared-cell-order-report-omitted", "winner_reuse": False,
            "graph": "persistent-full-matrix-tape-serial-selected-vjp", "empty_selection": "zero-calls",
            "callback": {"module": getattr(implementation, "__module__", type(implementation).__module__),
                         "qualname": getattr(implementation, "__qualname__", type(implementation).__qualname__)}})
        self._binding_hash = stable_hash(self.binding)
        self.receipts = []
        self.graph = self._make_graph()
        self._graph = self.graph
        self.loss_graph = self._make_loss_graph()
        self._loss_graph = self.loss_graph
        self.loss_only_receipts = []

    def _make_graph(self):
        raw_rows_fn = self.raw_rows_fn
        probabilities, groups = self.probabilities, self.group_ids
        denominators, owners = self.denominators, self.owner_indices
        row_count, group_count, column_count = len(self.row_ids), len(self.cell_ids), len(owners)
        selected_per_owner = min(self.binding["top_k"], group_count)
        columns = np.full(len(self.tasks), -1, np.int32)
        columns[owners] = np.arange(column_count)
        eligible = np.array([owner in self._eligible for owner in range(len(self.tasks))])
        owner_cap, row_cap = self.profile["max_active_owners"], self.profile["max_component_rows"]

        @tf.function(autograph=False, jit_compile=False, input_signature=[
            tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.int32)])
        def selected_components(parameters, selected_owners):
            checks = [tf.debugging.assert_positive(tf.size(parameters)),
                tf.debugging.assert_all_finite(parameters, "nonfinite component parent"),
                tf.debugging.assert_positive(tf.size(selected_owners)),
                tf.debugging.assert_non_negative(selected_owners),
                tf.debugging.assert_less(selected_owners, len(denominators)),
                tf.debugging.assert_less_equal(tf.size(selected_owners), owner_cap),
                tf.debugging.assert_less_equal(tf.size(selected_owners) * selected_per_owner, row_cap),
                tf.debugging.assert_equal(tf.size(tf.unique(selected_owners)[0]), tf.size(selected_owners))]
            with tf.control_dependencies(checks):
                selected_columns = tf.gather(tf.constant(columns), selected_owners)
                eligible_owners = tf.gather(tf.constant(eligible), selected_owners)
            with tf.control_dependencies([tf.debugging.assert_equal(eligible_owners, tf.ones_like(eligible_owners))]):
                parameters = tf.identity(parameters)
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(parameters)
                raw_rows = tf.convert_to_tensor(raw_rows_fn(parameters))
                if raw_rows.dtype != tf.float64:
                    raise TypeError("raw-row callback must return float64 losses")
                raw_rows = tf.ensure_shape(raw_rows, [row_count, column_count])
                with tf.control_dependencies([tf.debugging.assert_all_finite(raw_rows, "nonfinite component raw rows"),
                                               tf.debugging.assert_non_negative(raw_rows)]):
                    raw_means, normalized_means = _group_means(
                        raw_rows, probabilities, groups, group_count, denominators[owners])
                    normalized_means = tf.debugging.check_numerics(
                        normalized_means, "nonfinite component means")
            selected_means = tf.transpose(tf.gather(raw_means, selected_columns, axis=1))
            order = tf.argsort(selected_means, axis=1, direction="DESCENDING", stable=True)
            selected_cells = order[:, :selected_per_owner]
            repeated_columns = tf.repeat(selected_columns, selected_per_owner)
            pairs = tf.stack((tf.reshape(selected_cells, [-1]), repeated_columns), axis=1)
            reverse_count = tf.shape(pairs)[0]
            derivatives = tf.TensorArray(tf.float64, size=reverse_count, element_shape=tf.TensorShape([None]))

            def reverse_one(index, values):
                selector = tf.scatter_nd(pairs[index:index + 1], tf.ones([1], tf.float64), tf.shape(normalized_means))
                derivative = tape.gradient(normalized_means, parameters, output_gradients=selector)
                if derivative is None:
                    raise ValueError("disconnected raw-row callback parameter gradient")
                derivative = tf.debugging.check_numerics(derivative, "nonfinite selected component gradient")
                return index + 1, values.write(index, derivative)

            _, derivatives = tf.while_loop(lambda index, values: index < reverse_count, reverse_one,
                (tf.constant(0), derivatives), parallel_iterations=1)
            cutoff = tf.gather(selected_means, selected_cells[:, -1], batch_dims=1)[:, None]
            at_cutoff = selected_means == cutoff
            selected_mask = tf.reduce_any(tf.one_hot(selected_cells, group_count, on_value=True, off_value=False), axis=1)
            maxima = tf.reduce_max(selected_means, axis=1, keepdims=True)
            return {"owner_indices": tf.repeat(selected_owners, selected_per_owner),
                "cell_indices": tf.reshape(selected_cells, [-1]), "raw_values": tf.gather_nd(raw_means, pairs),
                "normalized_values": tf.gather_nd(normalized_means, pairs), "normalized_gradients": derivatives.stack(),
                "selected_owners": selected_owners,
                "group_raw_means": selected_means,
                "maximum_tie_counts": tf.reduce_sum(tf.cast(selected_means == maxima, tf.int32), axis=1),
                "cutoff_tie_counts": tf.reduce_sum(tf.cast(at_cutoff, tf.int32), axis=1),
                "omitted_ties": at_cutoff & ~selected_mask,
                "forward_calls": tf.constant(1), "reverse_calls": reverse_count}

        return selected_components

    def _make_loss_graph(self):
        raw_rows_fn = self.raw_rows_fn
        probabilities, groups = self.probabilities, self.group_ids
        denominators, owners = self.denominators, self.owner_indices
        row_count, group_count, column_count = len(self.row_ids), len(self.cell_ids), len(owners)
        columns = np.full(len(self.tasks), -1, np.int32)
        columns[owners] = np.arange(column_count)

        @tf.function(autograph=False, jit_compile=False, input_signature=[
            tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.int32),
            tf.TensorSpec([None], tf.int32)])
        def selected_losses(parameters, selected_owners, selected_cells):
            checks = [tf.debugging.assert_positive(tf.size(parameters)),
                tf.debugging.assert_all_finite(parameters, "nonfinite component loss parent"),
                tf.debugging.assert_equal(tf.size(selected_owners), tf.size(selected_cells)),
                tf.debugging.assert_positive(tf.size(selected_owners)),
                tf.debugging.assert_non_negative(selected_owners),
                tf.debugging.assert_less(selected_owners, len(self.tasks)),
                tf.debugging.assert_non_negative(selected_cells),
                tf.debugging.assert_less(selected_cells, group_count),
                tf.debugging.assert_less_equal(tf.size(selected_owners), self.profile["max_component_rows"])]
            with tf.control_dependencies(checks):
                selected_columns = tf.gather(tf.constant(columns), selected_owners)
                raw_rows = tf.convert_to_tensor(raw_rows_fn(parameters))
                if raw_rows.dtype != tf.float64:
                    raise TypeError("raw-row callback must return float64 losses")
                raw_rows = tf.ensure_shape(raw_rows, [row_count, column_count])
                raw_means, normalized_means = _group_means(
                    raw_rows, probabilities, groups, group_count, denominators[owners])
                pairs = tf.stack((selected_cells, selected_columns), axis=1)
                return {"raw_values": tf.gather_nd(raw_means, pairs),
                    "normalized_values": tf.gather_nd(normalized_means, pairs), "forward_calls": tf.constant(1)}

        return selected_losses

    def __call__(self, parameters, batch, update_index, context):
        if (self.raw_rows_fn is not self._raw_rows_fn or self.graph is not self._graph
                or stable_hash(self.binding) != self._binding_hash):
            raise ValueError("grouped component callback/profile changed")
        parameters = np.asarray(parameters)
        if (parameters.dtype != np.float64 or parameters.ndim != 1 or not parameters.size
                or not np.isfinite(parameters).all() or type(update_index) is not int):
            raise ValueError("finite float64 parent vector and integer update required")
        if (not isinstance(context, Mapping) or context.get("parameters_hash") != stable_hash(parameters.tolist())
                or context.get("batch_hash") != stable_hash(batch) or context.get("update_index") != update_index
                or context.get("coordinate_mode") != "raw_over_D"
                or canonical_json(context.get("D")) != canonical_json(self.denominators.tolist())
                or not isinstance(context.get("profile_hash"), str) or not context["profile_hash"]):
            raise ValueError("grouped components require exact parent/batch/update/D/profile context")
        active = context.get("active_indices")
        if (not isinstance(active, (list, tuple)) or any(type(owner) is not int or not 0 <= owner < len(self.tasks) for owner in active)
                or len(set(active)) != len(active)):
            raise ValueError("unique current public active indices required")
        extra = context.get("extra_descent_indices", ())
        if (not isinstance(extra, (list, tuple))
                or any(type(owner) is not int or not 0 <= owner < len(self.tasks) for owner in extra)
                or len(set(extra)) != len(extra)):
            raise ValueError("unique extra public descent indices required")
        selected = sorted(set(active).union(extra).intersection(self._eligible))
        if (len(selected) > self.profile["max_active_owners"]
                or len(selected) * min(self.binding["top_k"], len(self.cell_ids)) > self.profile["max_component_rows"]):
            raise ValueError("declared component owner/reverse-call budget exceeded")
        receipt = {"context": json.loads(canonical_json(context)), "provider_binding_hash": self._binding_hash, "selected_owners": selected,
            "graph_calls": 0, "forward_calls": 0, "reverse_calls": 0, "completed": False}
        self.receipts.append(receipt)
        if not selected:
            receipt["completed"] = True
            return {"owner_indices": np.empty(0, np.int64), "cell_ids": [], "raw_values": np.empty(0, np.float64),
                "normalized_values": np.empty(0, np.float64),
                "normalized_gradients": np.empty((0, parameters.size), np.float64), "context": context}
        receipt.update(graph_calls=1, forward_calls=None, reverse_calls=None)
        result = self.graph(parameters, np.array(selected, np.int32))
        values = {name: np.asarray(value) for name, value in result.items()}
        cells = [self.cell_ids[index] for index in values["cell_indices"]]
        receipt.update(completed=True, forward_calls=int(values["forward_calls"]), reverse_calls=int(values["reverse_calls"]),
            cell_ids=cells)
        receipt.update({"maximum_tie_counts": values["maximum_tie_counts"].tolist(),
            "cutoff_tie_counts": values["cutoff_tie_counts"].tolist(),
            "omitted_tied_cell_ids": [[cell for cell, omitted in zip(self.cell_ids, mask, strict=True) if omitted]
                                     for mask in values["omitted_ties"]],
            "group_raw_means": values["group_raw_means"].tolist(),
            "trace_count": self.graph.experimental_get_tracing_count()})
        return {"owner_indices": values["owner_indices"], "cell_ids": cells, "raw_values": values["raw_values"],
            "normalized_values": values["normalized_values"], "normalized_gradients": values["normalized_gradients"],
            "context": context}

    def loss_only(self, parameters, batch, update_index, context, owner_indices, cell_ids):
        if (self.raw_rows_fn is not self._raw_rows_fn or self.loss_graph is not self._loss_graph
                or stable_hash(self.binding) != self._binding_hash):
            raise ValueError("grouped loss callback/profile changed")
        parameters = np.asarray(parameters)
        if (parameters.dtype != np.float64 or parameters.ndim != 1 or not parameters.size
                or not np.isfinite(parameters).all() or type(update_index) is not int):
            raise ValueError("finite float64 component loss vector and integer update required")
        if (not isinstance(context, Mapping) or context.get("candidate_parameters_hash") != stable_hash(parameters.tolist())
                or context.get("batch_hash") != stable_hash(batch) or context.get("update_index") != update_index
                or context.get("coordinate_mode") != "raw_over_D"
                or canonical_json(context.get("D")) != canonical_json(self.denominators.tolist())
                or not isinstance(context.get("parameters_hash"), str) or not context["parameters_hash"]
                or not isinstance(context.get("profile_hash"), str) or not context["profile_hash"]):
            raise ValueError("grouped losses require exact parent/batch/update/coordinate context")
        owners = np.asarray(owner_indices)
        cells = tuple(cell_ids) if isinstance(cell_ids, (list, tuple)) else None
        if (owners.ndim != 1 or owners.dtype.kind not in "iu" or cells is None or len(cells) != owners.size
                or any(not isinstance(cell, str) or cell not in self.cell_ids for cell in cells)
                or np.any(owners < 0) or np.any(owners >= len(self.tasks))
                or len(set(zip(owners.tolist(), cells, strict=True))) != owners.size):
            raise ValueError("unique fixed grouped component identities required")
        eligible = set(self._eligible)
        if any(int(owner) not in eligible for owner in owners.tolist()):
            raise ValueError("fixed grouped component owner is not eligible")
        if (len(set(owners.tolist())) > self.profile["max_active_owners"]
                or owners.size > self.profile["max_component_rows"]
                or tuple(context.get("owner_indices", ())) != tuple(owners.tolist())
                or tuple(context.get("cell_ids", ())) != cells):
            raise ValueError("fixed component identities differ from context or declared budget")
        cell_indices = np.asarray([self.cell_ids.index(cell) for cell in cells], dtype=np.int32)
        receipt = {"context": json.loads(canonical_json(context)), "provider_binding_hash": self._binding_hash,
            "owner_indices": owners.tolist(), "cell_ids": list(cells), "forward_calls": 0,
            "reverse_calls": 0, "completed": False}
        self.loss_only_receipts.append(receipt)
        if not owners.size:
            receipt["completed"] = True
            return {"owner_indices": owners.astype(np.int32), "cell_ids": [],
                "raw_values": np.empty(0, np.float64), "normalized_values": np.empty(0, np.float64), "context": context}
        receipt["forward_calls"] = 1
        result = self.loss_graph(parameters, owners.astype(np.int32), cell_indices)
        values = {name: np.asarray(value) for name, value in result.items()}
        receipt.update(forward_calls=int(values["forward_calls"]), completed=True,
                       trace_count=self.loss_graph.experimental_get_tracing_count())
        return {"owner_indices": owners.astype(np.int32), "cell_ids": list(cells),
            "raw_values": values["raw_values"], "normalized_values": values["normalized_values"],
            "context": context}


def make_grouped_component_provider(raw_rows_fn, row_ids, probabilities, group_ids, cell_ids, D, owner_indices, max_profile):
    return GroupedComponentProvider(raw_rows_fn, row_ids, probabilities, group_ids, cell_ids, D, owner_indices, max_profile)
