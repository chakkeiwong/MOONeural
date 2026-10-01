"""Analytic quadratic cases for the shared curvature-coordinate direction."""

import numpy as np
import pytest

from mooneural.training.generic_curvature_progress import curvature_progress


def example(radius=10.):
    residuals = [np.array([[2.], [1.]]), np.array([[1.], [3.]])]
    responses = [np.array([[[1., 0.]], [[2., 0.]]]), np.array([[[0., 2.]], [[0., 1.]]])]
    probability = np.array([.3, .7])
    return curvature_progress(residuals, responses, probability, np.eye(2), radius), residuals, responses, probability


def test_exact_quadratic_tasks_obey_common_descent_and_whitening():
    result, residuals, responses, probability = example()
    assert result["solver"]["valid"] and result["solver"]["converged"]
    assert np.all(result["slopes"] < 0)
    assert np.all(result["predicted"] < result["baseline"])
    for task, (residual, response) in enumerate(zip(residuals, responses, strict=True)):
        actual = residual + response @ result["coefficients"]
        expected = np.einsum("b,bc,bc->", probability, actual, actual)
        np.testing.assert_allclose(expected, result["predicted"][task], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.sum(result["curvature"]), 1., rtol=1e-10)


def test_radius_contracts_uniformly_and_preserves_all_descent():
    result, _residuals, _responses, _probability = example(.001)
    assert result["radius_binding"]
    np.testing.assert_allclose(np.linalg.norm(result["direction"]), .001, rtol=1e-12)
    assert np.all(result["predicted"] < result["baseline"])


def test_singular_response_refuses_without_ridge():
    with pytest.raises(ValueError, match="full numerical response rank"):
        curvature_progress([np.ones((2, 1))], [np.ones((2, 1, 2))], np.array([.5, .5]), np.eye(2), 1.)


def test_conflicting_tasks_refuse_common_descent():
    with pytest.raises(ValueError, match="solver refused"):
        curvature_progress([np.ones((2, 1)), -np.ones((2, 1))],
            [np.ones((2, 1, 1)), np.ones((2, 1, 1))], np.array([.5, .5]), np.ones((1, 1)), 1.)


@pytest.mark.parametrize("radius", (0., -1., np.inf, np.nan))
def test_invalid_radius_refuses(radius):
    with pytest.raises(ValueError, match="positive radius"):
        example(radius)
