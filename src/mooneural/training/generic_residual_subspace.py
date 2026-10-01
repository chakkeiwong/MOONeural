"""Saved-array least squares in an explicit neural-weight subspace."""

import numpy as np


def _finite(value, name, rank):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim != rank or not all(array.shape):
        raise ValueError(f"{name} must be a nonempty real rank-{rank} array")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float64")
    return array


def gradient_subspace(gradients):
    """Return orthonormal weight columns spanning the supplied gradient rows."""
    gradients = _finite(gradients, "gradients", 2)
    magnitudes = np.max(np.abs(gradients), axis=1)
    active = magnitudes > 0
    if not active.any():
        raise ValueError("gradient subspace has zero rank")
    scaled = gradients[active] / magnitudes[active, None]
    scaled /= np.linalg.norm(scaled, axis=1)[:, None]
    _left, singular, right = np.linalg.svd(scaled, full_matrices=False)
    cutoff = max(scaled.shape) * np.finfo(np.float64).eps * singular[0]
    selected = singular > cutoff
    basis = right[selected].T
    pivot = np.argmax(np.abs(basis), axis=0)
    basis *= np.where(basis[pivot, np.arange(basis.shape[1])] < 0, -1., 1.)
    return {"basis": basis, "singular": singular, "cutoff": float(cutoff), "rank": int(selected.sum())}


def fit_residual_subspace(residuals, responses, probabilities):
    """Fit one equal sum of already normalized task losses, with no damping."""
    if not residuals or len(residuals) != len(responses):
        raise ValueError("matching nonempty residual and response tasks required")
    probabilities = _finite(probabilities, "probabilities", 1)
    if np.any(probabilities <= 0) or abs(probabilities.sum() - 1.) > 64 * np.finfo(np.float64).eps:
        raise ValueError("positive normalized outer probabilities required")
    matrices, vectors, sizes = [], [], []
    dimension = None
    for residual, response in zip(residuals, responses, strict=True):
        residual = _finite(residual, "residual", 2)
        response = _finite(response, "response", 3)
        if response.shape[:2] != residual.shape or len(residual) != len(probabilities):
            raise ValueError("residual/response/outer row shape mismatch")
        if dimension is not None and response.shape[2] != dimension:
            raise ValueError("response subspace dimensions differ")
        dimension = response.shape[2]
        vectors.append((residual * np.sqrt(probabilities[:, None])).reshape(-1))
        matrices.append((response * np.sqrt(probabilities[:, None, None])).reshape(-1, dimension))
        sizes.append(residual.size)
    vector, matrix = np.concatenate(vectors), np.concatenate(matrices)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    cutoff = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    active = singular > cutoff
    if not active.any():
        raise ValueError("residual response has zero rank")
    coefficients = -(right[active].T @ ((left[:, active].T @ vector) / singular[active]))
    predicted = vector + matrix @ coefficients
    if not np.isfinite(coefficients).all() or not np.isfinite(predicted).all():
        raise ValueError("nonfinite subspace fit")
    boundaries = np.cumsum(sizes)[:-1]
    before = np.array([np.dot(part, part) for part in np.split(vector, boundaries)])
    after = np.array([np.dot(part, part) for part in np.split(predicted, boundaries)])
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("subspace loss overflow")
    return {"coefficients": coefficients, "singular": singular, "rank": int(active.sum()), "cutoff": float(cutoff),
        "baseline_losses": before, "predicted_losses": after, "orthogonal_squared_norm": float(predicted @ predicted),
        "matrix_shape": matrix.shape}
