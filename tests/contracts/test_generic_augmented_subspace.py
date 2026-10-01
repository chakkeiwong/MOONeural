"""Nested search-space algebra, scale invariance and near-span refusal."""

import numpy as np
import pytest

from mooneural.training.generic_augmented_subspace import augment_subspace


def test_prefix_is_exact_and_complement_spans_only_new_directions():
    original = np.eye(5)[:, :2]
    rows = np.array([[1., 2., 3., 0., 0.], [2., 3., 0., 4., 0.]])
    result = augment_subspace(original, rows)
    np.testing.assert_array_equal(result["basis"][:, :2], original)
    np.testing.assert_allclose(result["basis"].T @ result["basis"], np.eye(4), atol=2e-15)
    np.testing.assert_allclose(result["basis"] @ (result["basis"].T @ rows.T), rows.T, atol=3e-15)
    assert result["added_rank"] == 2
    assert not np.any(result["basis"][4])


def test_duplicate_and_zero_candidates_do_not_inflate_rank():
    original = np.eye(4)[:, :1]
    rows = np.array([[1., 2., 0., 0.], [2., 4., 0., 0.], [0., 0., 0., 0.]])
    result = augment_subspace(original, rows)
    assert result["added_rank"] == 1
    assert result["basis"].shape == (4, 2)


def test_scaled_candidate_rows_preserve_projected_space_without_overflow():
    original = np.eye(4)[:, :1]
    rows = np.array([[1., 2., 0., 0.], [1., 0., 3., 0.]])
    first = augment_subspace(original, rows)["basis"]
    second = augment_subspace(original, rows * np.array([1e250, -1e-250])[:, None])["basis"]
    np.testing.assert_allclose(first @ first.T, second @ second.T, atol=2e-15)


def test_roundoff_components_of_existing_span_are_not_amplified():
    original, _upper = np.linalg.qr(np.array([[1., 2.], [3., 7.], [5., 11.], [13., 17.]]))
    result = augment_subspace(original, (original @ np.array([[2., 3.], [5., 7.]])).T)
    assert result["added_rank"] == 0
    np.testing.assert_array_equal(result["basis"], original)


def test_zero_complement_is_a_valid_outcome():
    result = augment_subspace(np.eye(3), np.zeros((2, 3)))
    assert result["added_rank"] == 0
    assert result["complement"].shape == (3, 0)
    np.testing.assert_array_equal(result["basis"], np.eye(3))


@pytest.mark.parametrize("basis,rows", (
    (np.ones((3, 2)), np.ones((1, 3))),
    (np.eye(3)[:, :1], np.ones((1, 4))),
    (np.eye(3)[:, :1], np.array([[np.nan, 1., 2.]])),
))
def test_invalid_coordinates_refuse(basis, rows):
    with pytest.raises(ValueError):
        augment_subspace(basis, rows)
