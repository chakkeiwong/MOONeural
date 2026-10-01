"""Fixed-policy coordinate and shared-direction diagnostics, without updates."""

from __future__ import annotations

import hashlib

import numpy as np


def scale_profiles(task_ids, training_scales, terminal_limits):
    tasks = tuple(task_ids)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("task identifiers must be nonempty and unique")
    vectors = []
    for values in (training_scales, terminal_limits):
        if set(values) != set(tasks):
            raise ValueError("scale task inventory mismatch")
        vector = np.asarray([values[task] for task in tasks], dtype=np.float64)
        if not np.all(np.isfinite(vector) & (vector > 0.0)):
            raise ValueError("scale denominators must be finite and positive")
        vectors.append(vector)
    scales, limits = vectors
    return {
        "calibration": {"training": scales, "selection": scales},
        "terminal_limit": {"training": limits, "selection": limits},
        "gate_0_04": {"training": limits / 0.04, "selection": limits / 0.04},
        "selector_only": {"training": scales, "selection": limits},
    }


def coordinate_report(raw_values, raw_gradients, denominators, terminal_limits):
    values = np.asarray(raw_values, dtype=np.float64)
    gradients = np.asarray(raw_gradients, dtype=np.float64)
    scales = np.asarray(denominators, dtype=np.float64)
    limits = np.asarray(terminal_limits, dtype=np.float64)
    if (values.ndim != 1 or gradients.ndim != 2 or gradients.shape[0] != values.size
            or scales.shape != values.shape or limits.shape != values.shape):
        raise ValueError("coordinate shape mismatch")
    if (not all(np.isfinite(array).all() for array in (values, gradients, scales, limits))
            or np.any(values < 0.0) or np.any(scales <= 0.0) or np.any(limits <= 0.0)):
        raise ValueError("invalid coordinate values or denominators")
    return {
        "raw_mse": values.tolist(), "raw_rms": np.sqrt(values).tolist(),
        "training_denominators": scales.tolist(),
        "training_normalized_mse": (values / scales).tolist(),
        "terminal_raw_mse_limits": limits.tolist(),
        "terminal_boundary_in_training_coordinates": (limits / scales).tolist(),
        "descriptive_gate_ratio": (values / limits).tolist(),
        "raw_gradient_norm": np.linalg.norm(gradients, axis=1).tolist(),
        "training_gradient_norm": np.linalg.norm(gradients / scales[:, None], axis=1).tolist(),
    }


def candidate_ranking(candidate_ids, raw_values, selection_denominators):
    identifiers = tuple(candidate_ids)
    values = np.asarray(raw_values, dtype=np.float64)
    scales = np.asarray(selection_denominators, dtype=np.float64)
    if (len(set(identifiers)) != len(identifiers) or values.ndim != 2
            or values.shape != (len(identifiers), scales.size) or scales.ndim != 1
            or not identifiers or scales.size == 0):
        raise ValueError("candidate inventory or shape mismatch")
    if (not np.isfinite(values).all() or not np.isfinite(scales).all()
            or np.any(values < 0.0) or np.any(scales <= 0.0)):
        raise ValueError("invalid candidate values or selector denominators")
    ratios = values / scales
    scores = ratios.max(axis=1)
    return [{"candidate_id": identifiers[index], "minimax": float(scores[index]),
             "worst_task_index": int(np.argmax(ratios[index]))}
            for index in sorted(range(len(identifiers)), key=lambda index: (scores[index], identifiers[index]))]


def direction_report(task_ids, raw_values, raw_gradients, profiles, *, pcgrad_seed):
    """Use the shared all-unpassed scheduler and numerical MOO/projector kernels."""
    import math

    import tensorflow as tf

    from .generic_execution_boundary import (
        METHODS,
        jacobi_schedule,
        make_method_function,
        pcgrad_permutations,
        subset_masks,
    )
    from .generic_permanent_pass import _partition
    from .generic_xla_kernels import project_cone_impl

    tasks = tuple(task_ids)
    raw_values = np.asarray(raw_values, dtype=np.float64)
    raw_gradients = np.asarray(raw_gradients, dtype=np.float64)
    active_count = min(3, len(tasks))
    constraint_count = len(tasks) - active_count
    parameter_dim = raw_gradients.shape[1]
    masks, schedule = subset_masks(constraint_count), jacobi_schedule()
    methods = {name: make_method_function(name, active_count, parameter_dim) for name in METHODS}

    @tf.function(input_signature=(
        tf.TensorSpec([parameter_dim], tf.float64),
        tf.TensorSpec([constraint_count, parameter_dim], tf.float64),
    ), autograph=False, jit_compile=True)
    def project(direction, constraints):
        return project_cone_impl(direction, constraints, masks, schedule, tf.constant(1e-12, tf.float64))

    partitions = [_partition(tasks, (), index) for index in range(math.comb(len(tasks), active_count))]
    baseline_directions = {}
    records = {}
    for profile_name, profile in profiles.items():
        normalized_values = raw_values / profile["training"]
        normalized_gradients = raw_gradients / profile["training"][:, None]
        profile_records = []
        for index, partition in enumerate(partitions):
            active = [tasks.index(task) for task in partition["active"]]
            constraints = [tasks.index(task) for task in partition["constraints"]]
            rows = tf.constant(normalized_gradients[active])
            losses = tf.constant(normalized_values[active])
            permutations = pcgrad_permutations(active_count, index, seed=pcgrad_seed)
            for method, function in methods.items():
                coefficients, direction, _weights, _initial, _step, valid = function(
                    rows, losses, permutations, tf.ones([active_count], tf.float64),
                    losses, tf.constant(0, tf.int64),
                )
                record = {"partition_index": index, "method": method, "valid": bool(valid.numpy())}
                if record["valid"]:
                    projected, constraint_products, residual, valid_projection = project(
                        direction, tf.constant(normalized_gradients[constraints]),
                    )
                    record["valid"] = bool(valid_projection.numpy())
                    if record["valid"]:
                        original, corrected = direction.numpy(), projected.numpy()
                        baseline_key = (index, method)
                        if profile_name == "calibration":
                            baseline_directions[baseline_key] = corrected
                        baseline = baseline_directions.get(baseline_key)
                        norm = float(np.linalg.norm(corrected))
                        baseline_norm = 0.0 if baseline is None else float(np.linalg.norm(baseline))
                        record.update({
                            "coefficients": coefficients.numpy().tolist(),
                            "unadjusted_active_gradient_norms": np.linalg.norm(normalized_gradients[active], axis=1).tolist(),
                            "preprojection_contribution_norms": None if method == "pcgrad" else np.linalg.norm(coefficients.numpy()[:, None] * normalized_gradients[active], axis=1).tolist(),
                            "contribution_basis": "adjusted per-task PCGrad rows unavailable from shared API" if method == "pcgrad" else "coefficients times original active gradients before cone projection",
                            "direction_norm": float(np.linalg.norm(original)),
                            "projected_norm": norm,
                            "projection_correction_norm": float(np.linalg.norm(corrected - original)),
                            "constraint_directional_products": constraint_products.numpy().tolist(),
                            "pinv_residual": float(residual.numpy()),
                            "raw_directional_derivative_for_negative_direction": (-raw_gradients @ corrected).tolist(),
                            "projected_cosine_to_calibration": None if norm == 0.0 or baseline_norm == 0.0 else float(corrected @ baseline / norm / baseline_norm),
                            "direction_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                            "projected_sha256": hashlib.sha256(corrected.tobytes()).hexdigest(),
                        })
                profile_records.append(record)
        records[profile_name] = profile_records
    return {
        "scenario": "all tasks unpassed; no statistical membership claim",
        "method_state": "reset per probe; no optimizer update", "pcgrad_seed": pcgrad_seed,
        "partitions": partitions,
        "active_counts": {task: sum(task in part["active"] for part in partitions) for task in tasks},
        "constraint_counts": {task: sum(task in part["constraints"] for part in partitions) for task in tasks},
        "records": records,
        "tracing_counts": {**{name: function.experimental_get_tracing_count() for name, function in methods.items()},
                           "project": project.experimental_get_tracing_count()},
    }
