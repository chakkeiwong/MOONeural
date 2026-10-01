"""Common descent in saved residual-curvature coordinates using the shared QP."""

import numpy as np

from .generic_relative_progress import make_relative_progress_function
from .generic_residual_subspace import _finite


def curvature_progress(residuals, responses, probabilities, basis, radius):
    """Return one locally scaled direction; validity does not imply convergence."""
    probabilities = _finite(probabilities, "probabilities", 1)
    basis = _finite(basis, "basis", 2)
    if (not residuals or len(residuals) != len(responses) or np.any(probabilities <= 0)
            or abs(probabilities.sum() - 1.) > 64 * np.finfo(np.float64).eps
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError("normalized measure, matched tasks and positive radius required")
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), rtol=3e-10, atol=1e-10)
    matrices, values, gradients = [], [], []
    for residual, response in zip(residuals, responses, strict=True):
        residual = _finite(residual, "residual", 2)
        response = _finite(response, "response", 3)
        if (response.shape[:2] != residual.shape or len(residual) != len(probabilities)
                or response.shape[2] != basis.shape[1]):
            raise ValueError("response and basis dimensions differ")
        matrices.append((response * np.sqrt(probabilities[:, None, None])).reshape(-1, basis.shape[1]))
        values.append(np.einsum("b,bc,bc->", probabilities, residual, residual))
        gradients.append(2 * np.einsum("b,bc,bck->k", probabilities, residual, response))
    matrix = np.concatenate(matrices)
    _left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    cutoff = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.sum(singular > cutoff))
    if rank != basis.shape[1]:
        raise ValueError("full numerical response rank required")
    whitening = right.T / singular
    np.testing.assert_allclose((matrix @ whitening).T @ (matrix @ whitening), np.eye(rank), rtol=3e-10, atol=1e-10)
    values, gradients = np.array(values), np.array(gradients)
    count = len(values)
    source = np.zeros(rank)
    source[0] = 1.
    solver = make_relative_progress_function(count)
    solution = solver(values, gradients @ whitening, np.ones(count, dtype=bool), np.empty(0, dtype=np.float64),
        np.empty((0, rank), dtype=np.float64), np.empty(0, dtype=np.int32), source)
    solution = {key: value.numpy() for key, value in solution.items()}
    if not bool(solution["valid"]) or not bool(solution["converged"]):
        raise ValueError("shared relative-progress solver refused curvature coordinates")
    coefficients = whitening @ solution["direction"]
    slopes = gradients @ coefficients
    curvature = np.array([np.dot(part @ coefficients, part @ coefficients) for part in matrices])
    norm = np.linalg.norm(basis @ coefficients)
    if (not np.isfinite(slopes).all() or not np.all(slopes < 0) or not np.isfinite(curvature).all()
            or np.any(curvature < 0) or not np.any(curvature > 0) or not np.isfinite(norm) or norm <= 0):
        raise ValueError("finite common descent and positive curvature/norm required")
    quadratic_scale = np.min(-slopes[curvature > 0] / (2 * curvature[curvature > 0]))
    radius_scale = radius / norm
    scale = min(quadratic_scale, radius_scale)
    change = scale * slopes + scale**2 * curvature
    if not np.isfinite(change).all() or not np.all(change < 0):
        raise ValueError("predicted common decrease not established")
    if np.any(change > .5 * scale * slopes + 64 * np.finfo(np.float64).eps * np.abs(scale * slopes)):
        raise ValueError("common quadratic scale inequality differs")
    final_coefficients = scale * coefficients
    return {"coefficients": final_coefficients, "direction": basis @ final_coefficients,
        "unscaled_coefficients": coefficients, "whitening": whitening, "singular": singular,
        "baseline": values, "predicted": values + change, "slopes": slopes, "curvature": curvature,
        "rank": rank, "rank_cutoff": float(cutoff), "scale": float(scale), "radius": float(radius),
        "quadratic_scale": float(quadratic_scale), "radius_scale": float(radius_scale),
        "radius_binding": bool(radius_scale < quadratic_scale), "solver": solution}
