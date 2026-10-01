"""Deterministic fixed-population accumulation before a shared optimizer update."""

from __future__ import annotations

import math

import numpy as np


def _float64_array(value, name, rank):
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or array.ndim != rank:
        raise ValueError(f"{name} must be a rank-{rank} float64 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def make_weighted_objective(evaluate_batch, values_batch, batches, weights):
    """Return source-bindable ``evaluate`` and ``losses`` callbacks.

    Each call traverses every batch in its supplied order at the same flat
    float64 parameters. Batch callbacks return float64 raw[T], normalized[T]
    and, for evaluation, gradients[T,P] of the normalized objectives. All three
    arrays use the same positive batch probabilities, which must sum to one
    within absolute float64 tolerance 1e-12; weights are never renormalized.
    For empirical row means, supply batch row counts divided by total rows.
    This preserves raw/D and its gradient when every batch uses the same D.

    The host validates via NumPy and accumulates with array arithmetic, retaining
    NumPy arrays or eager tensor outputs without a framework import. This is
    host orchestration, not a traced function. Task and parameter counts are
    shared by both callbacks and fixed by the first successful complete call.
    Construction invokes neither callback. Callback errors propagate unchanged.

    Batch order and weights are captured at construction; batch contents remain
    caller-owned constants. The caller binds batch contents, weights, callback
    sources, recipe and normalization and must preserve their immutability.
    No optimizer, stochastic sampling, normalization or scientific acceptance
    is introduced here.
    """
    if not callable(evaluate_batch) or not callable(values_batch):
        raise TypeError("evaluate_batch and values_batch must be callable")
    fixed_batches = tuple(batches)
    supplied_weights = np.asarray(weights)
    if (not fixed_batches or supplied_weights.shape != (len(fixed_batches),)
            or supplied_weights.dtype.kind not in "fiu"):
        raise ValueError("weights must be a real vector matching nonempty batches")
    with np.errstate(over="ignore", invalid="ignore"):
        fixed_weights = np.array(supplied_weights, dtype=np.float64, copy=True)
    if not np.isfinite(fixed_weights).all() or not np.all(fixed_weights > 0.):
        raise ValueError("weights must be finite and strictly positive")
    try:
        total_weight = math.fsum(fixed_weights)
    except OverflowError as error:
        raise ValueError("weights must sum to one") from error
    if not math.isclose(total_weight, 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("weights must sum to one without renormalization")
    fixed_weights.setflags(write=False)
    signature = None

    def accumulate(parameters, callback, gradients):
        nonlocal signature

        parameter_values = _float64_array(parameters, "parameters", 1)
        parameter_count = parameter_values.size
        if not parameter_count:
            raise ValueError("parameters must be nonempty")
        if signature is not None and parameter_count != signature[1]:
            raise ValueError("parameter count changed across objective calls")
        task_count = None if signature is None else signature[0]
        totals = None
        for batch, weight in zip(fixed_batches, fixed_weights, strict=True):
            result = callback(parameters, batch)
            if not isinstance(result, (tuple, list)) or len(result) != (3 if gradients else 2):
                raise ValueError("batch callback output count mismatch")
            arrays = []
            for index, value in enumerate(result):
                name = ("raw", "normalized", "gradients")[index]
                if not hasattr(value, "dtype") or not hasattr(value, "shape"):
                    raise ValueError(f"{name} must be a float64 array or eager tensor")
                arrays.append(_float64_array(value, name, 2 if index == 2 else 1))
            if task_count is None:
                task_count = arrays[0].size
            if not task_count or arrays[0].shape != (task_count,) or arrays[1].shape != (task_count,):
                raise ValueError("raw and normalized task counts must be nonempty and consistent")
            if gradients and arrays[2].shape != (task_count, parameter_count):
                raise ValueError("gradient shape must match task and parameter counts")
            with np.errstate(over="ignore", invalid="ignore"):
                weighted = tuple(value * weight for value in result)
                totals = weighted if totals is None else tuple(
                    total + value for total, value in zip(totals, weighted, strict=True))
            for index, total in enumerate(totals):
                _float64_array(total, "weighted accumulation", 2 if index == 2 else 1)
        signature = (task_count, parameter_count)
        return totals

    def evaluate(parameters):
        return accumulate(parameters, evaluate_batch, True)

    def losses(parameters):
        return accumulate(parameters, values_batch, False)

    return {"evaluate": evaluate, "losses": losses}
