"""Append only independent candidate directions to a preserved weight basis."""

import numpy as np

from .generic_residual_subspace import _finite


def augment_subspace(existing_basis, candidate_rows):
    """Keep the existing columns exactly and return an orthonormal complement."""
    basis = _finite(existing_basis, "existing_basis", 2)
    rows = _finite(candidate_rows, "candidate_rows", 2)
    if rows.shape[1] != basis.shape[0] or basis.shape[1] > basis.shape[0]:
        raise ValueError("candidate rows and basis must share packed-weight coordinates")
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), rtol=3e-10, atol=1e-10):
        raise ValueError("existing basis must be orthonormal")
    magnitudes = np.max(np.abs(rows), axis=1)
    scaled = rows / np.where(magnitudes > 0, magnitudes, 1.)[:, None]
    norms = np.linalg.norm(scaled, axis=1)
    columns = (scaled / np.where(norms > 0, norms, 1.)[:, None]).T
    reference_norm = np.linalg.norm(columns, ord=2)
    complement = columns - basis @ (basis.T @ columns)
    complement -= basis @ (basis.T @ complement)
    left, singular, _right = np.linalg.svd(complement, full_matrices=False)
    cutoff = max(columns.shape) * np.finfo(np.float64).eps * reference_norm
    selected = singular > cutoff
    addition = left[:, selected]
    if addition.shape[1]:
        pivots = np.argmax(np.abs(addition), axis=0)
        addition *= np.where(addition[pivots, np.arange(addition.shape[1])] < 0, -1., 1.)
    combined = np.concatenate((basis, addition), axis=1)
    if not np.allclose(combined.T @ combined, np.eye(combined.shape[1]), rtol=3e-10, atol=1e-10):
        raise ValueError("augmented basis orthogonality not established")
    return {"basis": combined, "complement": addition, "added_rank": int(selected.sum()),
        "singular": singular, "rank_cutoff": float(cutoff),
        "candidate_complement_norms": np.linalg.norm(complement, axis=0)}
