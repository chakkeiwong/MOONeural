"""Exact small references for fixed-subspace normalized residual proposals."""

import numpy as np
import pytest

from mooneural.training.generic_residual_subspace import (
    fit_residual_subspace,
    gradient_subspace,
)


def test_gradient_span_is_orthonormal_and_preserves_scaled_rows():
    gradients = np.array([[2., 1., 0., -1.], [-1., 3., 2., 0.], [0., 0., 0., 0.]])
    result = gradient_subspace(gradients)
    basis = result["basis"]
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(gradients @ basis @ basis.T, gradients, atol=1e-14)
    other = gradient_subspace(gradients * np.array([1e-200, 1e200, 1.])[:, None])["basis"]
    np.testing.assert_allclose(other @ other.T, basis @ basis.T, atol=1e-14)
    assert result["rank"] == 2


def test_full_rank_fit_reduces_exact_original_weighted_tasks():
    residuals = [np.array([[2., -1.], [.4, 1.]]), np.array([[.2], [-.8]])]
    responses = [np.array([[[1., 0.], [0., 1.]], [[2., 1.], [-1., 3.]]]), np.array([[[1., 2.]], [[2., -1.]]])]
    probabilities = np.array([.2, .8])
    result = fit_residual_subspace(residuals, responses, probabilities)
    matrix = np.concatenate([(response * np.sqrt(probabilities[:, None, None])).reshape(-1, 2) for response in responses])
    vector = np.concatenate([(residual * np.sqrt(probabilities[:, None])).reshape(-1) for residual in residuals])
    expected = np.linalg.solve(matrix.T @ matrix, -matrix.T @ vector)
    np.testing.assert_allclose(result["coefficients"], expected, atol=1e-14)
    predicted = [residual + np.einsum("bcd,d->bc", response, expected)
                 for residual, response in zip(residuals, responses, strict=True)]
    for name, values in (("baseline_losses", residuals), ("predicted_losses", predicted)):
        expected_losses = [np.sum(probabilities * np.sum(value**2, axis=1)) for value in values]
        np.testing.assert_allclose(result[name], expected_losses, atol=1e-14)
    assert result["orthogonal_squared_norm"] <= np.sum(result["baseline_losses"])


def test_rank_deficient_response_uses_minimum_norm_without_regularization():
    result = fit_residual_subspace([np.array([[3.], [6.]])],
        [np.array([[[1., 2.]], [[2., 4.]]])], np.array([.3, .7]))
    np.testing.assert_allclose(result["coefficients"], [-.6, -1.2], atol=1e-14)
    np.testing.assert_allclose(result["predicted_losses"], [0.], atol=1e-28)
    assert result["rank"] == 1


@pytest.mark.parametrize("gradients", (np.zeros((2, 3)), np.array([[np.nan]]), np.ones(2)))
def test_invalid_gradient_basis_refuses(gradients):
    with pytest.raises(ValueError):
        gradient_subspace(gradients)


@pytest.mark.parametrize("failure", ("shape", "probability", "zero_rank", "nonfinite"))
def test_invalid_fit_refuses(failure):
    residual = np.ones((2, 1))
    response = np.ones((2, 1, 2))
    probability = np.array([.5, .5])
    if failure == "shape":
        response = response[:1]
    elif failure == "probability":
        probability *= 2.
    elif failure == "zero_rank":
        response *= 0.
    else:
        response[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_residual_subspace([residual], [response], probability)
