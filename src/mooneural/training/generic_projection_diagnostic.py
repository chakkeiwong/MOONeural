"""Recover fixed-row derivative coordinates and integrate fresh sign projections."""

import numpy as np


def projection_diagnostic(projections, residuals, responses):
    """Analyze already normalized [row, projection] residuals and directions.

    Conditional variances describe new independent signs at the supplied fixed
    coordinates. They do not correct adaptation to the observed projections.
    """
    projections, residuals, responses = [np.asarray(value, dtype=np.float64)
        for value in (projections, residuals, responses)]
    if (projections.ndim != 3 or not all(projections.shape)
            or residuals.shape != projections.shape[:2] or responses.ndim != 3
            or responses.shape[:2] != residuals.shape or not responses.shape[2]
            or not all(np.isfinite(value).all() for value in (projections, residuals, responses))
            or not np.all(np.abs(projections) == 1)):
        raise ValueError("matched finite Rademacher projections and responses required")
    count, dimension = projections.shape[1:]
    design = projections / np.sqrt(count)
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    cutoff = max(count, dimension) * np.finfo(np.float64).eps * singular[:, :1]
    if count < dimension or np.any(np.sum(singular > cutoff, axis=1) != dimension):
        raise ValueError("full projection column rank required")
    targets = np.concatenate((residuals[:, :, None], responses), axis=2)
    recovered = right.swapaxes(1, 2) @ ((left.swapaxes(1, 2) @ targets) / singular[:, :, None])
    reconstructed = design @ recovered
    if not np.allclose(reconstructed, targets, rtol=3e-10, atol=1e-10):
        raise ValueError("saved projected coordinates do not reconstruct")
    derivative, direction = recovered[:, :, 0], recovered[:, :, 1:]
    loss = np.sum(derivative**2, axis=1)
    norm = np.sum(direction**2, axis=1)
    product = np.einsum("bd,bdk->bk", derivative, direction)
    sampled_loss = np.sum(residuals**2, axis=1)
    sampled_slope = 2 * np.einsum("bp,bpk->bk", residuals, responses)
    loss_variance = 2 / count * (loss**2 - np.sum(derivative**4, axis=1))
    slope_variance = 4 / count * (loss[:, None] * norm + product**2
        - 2 * np.einsum("bd,bdk->bk", derivative**2, direction**2))
    return {
        "derivative": derivative, "direction": direction,
        "sampled_loss": sampled_loss, "integrated_loss": loss,
        "sampled_slope": sampled_slope, "integrated_slope": 2 * product,
        "conditional_loss_variance": np.maximum(loss_variance, 0),
        "conditional_slope_variance": np.maximum(slope_variance, 0),
        "gram_eigenvalues": singular**2,
        "maximum_reconstruction_error": float(np.max(np.abs(reconstructed - targets))),
    }
