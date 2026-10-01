"""Exact sign enumeration and malformed-cache checks for projection diagnostics."""

import itertools

import numpy as np
import pytest

from mooneural.training.generic_projection_diagnostic import projection_diagnostic


def fixture():
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    signs = np.concatenate((signs, signs[:3]))[None]
    derivative = np.array([[2., -3., 0.5]]) / np.sqrt(7.)
    directions = np.array([[[1., 0.], [4., 0.], [-2., 0.]]]) / np.sqrt(7.)
    residual = (signs @ derivative[:, :, None])[:, :, 0] / np.sqrt(signs.shape[1])
    response = signs @ directions / np.sqrt(signs.shape[1])
    return signs, derivative, directions, residual, response


def test_enumerated_conditional_moments_and_nonunit_denominator():
    signs, derivative, directions, residual, response = fixture()
    result = projection_diagnostic(signs, residual, response)
    exact_signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    projected = exact_signs @ derivative[0]
    projected_directions = exact_signs @ directions[0]
    losses = projected**2
    slopes = 2 * projected[:, None] * projected_directions
    np.testing.assert_allclose(result["derivative"], derivative, atol=1e-14)
    np.testing.assert_allclose(result["direction"], directions, atol=1e-14)
    np.testing.assert_allclose(result["integrated_loss"], losses.mean(), rtol=1e-13)
    np.testing.assert_allclose(result["integrated_slope"][0], slopes.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(result["conditional_loss_variance"], losses.var() / len(signs[0]), rtol=1e-13)
    np.testing.assert_allclose(result["conditional_slope_variance"][0], slopes.var(axis=0) / len(signs[0]), atol=1e-13)
    assert not np.allclose(result["sampled_loss"], result["integrated_loss"])


def test_projection_count_changes_variance_not_integrated_mean():
    signs, _, _, residual, response = fixture()
    original = projection_diagnostic(signs, residual, response)
    doubled = projection_diagnostic(np.tile(signs, (1, 2, 1)),
        np.tile(residual, (1, 2)) / np.sqrt(2), np.tile(response, (1, 2, 1)) / np.sqrt(2))
    np.testing.assert_allclose(doubled["integrated_loss"], original["integrated_loss"])
    np.testing.assert_allclose(doubled["conditional_loss_variance"], original["conditional_loss_variance"] / 2)


def test_row_direction_and_projection_permutations():
    signs, _, _, residual, response = fixture()
    signs = np.repeat(signs, 2, axis=0)
    residual = np.concatenate((residual, 2 * residual))
    response = np.concatenate((response, 3 * response))
    original = projection_diagnostic(signs, residual, response)
    reversed_result = projection_diagnostic(signs[::-1, ::-1], residual[::-1, ::-1], response[::-1, ::-1, ::-1])
    np.testing.assert_allclose(reversed_result["integrated_slope"], original["integrated_slope"][::-1, ::-1], atol=1e-13)


@pytest.mark.parametrize("defect", ("rank", "nonfinite", "shape", "not_signs", "too_few"))
def test_invalid_projection_input_refuses(defect):
    signs, _, _, residual, response = fixture()
    if defect == "rank":
        signs[:] = 1
    elif defect == "nonfinite":
        residual[0, 0] = np.nan
    elif defect == "shape":
        response = response[:, :-1]
    elif defect == "not_signs":
        signs[0, 0, 0] = 0
    else:
        signs, residual, response = signs[:, :2], residual[:, :2], response[:, :2]
    with pytest.raises(ValueError):
        projection_diagnostic(signs, residual, response)


def test_incompatible_saved_vectors_refuse():
    signs, _, _, residual, response = fixture()
    residual[0, 0] += 0.1
    with pytest.raises(ValueError, match="reconstruct"):
        projection_diagnostic(signs, residual, response)
