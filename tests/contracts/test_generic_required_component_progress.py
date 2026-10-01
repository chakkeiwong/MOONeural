"""Independent targets, scaling and saved-direction checks for explicit cell masks."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training.generic_relative_progress import (
    make_relative_progress_function,
    relative_progress_impl,
)

ROOT = Path(__file__).resolve().parents[2]


def operands(component_value=.5):
    return (np.array([1.]), np.array([[1., 0.]]), np.array([True]),
        np.array([component_value]), np.array([[0., 1.]]), np.array([0], np.int32), np.array([.1, 0.]))


def test_required_nonmaximal_cell_gets_its_own_fractional_target():
    kernel = make_relative_progress_function(1, explicit_component_targets=True)
    result = kernel(*operands(), np.array([True]))
    assert result["valid"]
    expected = -.1 * np.array([1., .5]) / np.sqrt(1.25)
    np.testing.assert_allclose(result["direction"], expected, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(-np.asarray(result["direction"]) / [1., .5], .1 / np.sqrt(1.25), rtol=1e-8)


def test_false_mask_preserves_old_direction_and_near_ties_cannot_be_disabled():
    original = make_relative_progress_function(1)
    explicit = make_relative_progress_function(1, explicit_component_targets=True)
    for value in (.5, 1. - 1e-12):
        old = original(*operands(value))
        new = explicit(*operands(value), np.array([False]))
        assert old["valid"] and new["valid"]
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])
    assert explicit.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("factor", [1e-100, 1e100, 1.7e308])
def test_owner_rescaling_preserves_nonmaximal_cell_progress(factor):
    data = list(operands())
    for index in (0, 1, 3, 4):
        data[index] = data[index] * factor
    result = make_relative_progress_function(1, explicit_component_targets=True)(*data, np.array([True]))
    assert result["valid"]
    expected = -.1 * np.array([1., .5]) / np.sqrt(1.25)
    np.testing.assert_allclose(result["direction"], expected, rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_required_cell_value_cannot_be_omitted(value):
    result = make_relative_progress_function(1, explicit_component_targets=True)(*operands(value), np.array([True]))
    assert not result["valid"]
    assert not result["seed_valid"]
    assert result["iterations"] == 0


@pytest.mark.parametrize("mask", [np.array([1]), np.array([[True]]), np.array([True, False])])
def test_required_mask_must_match_component_shape_and_dtype(mask):
    with pytest.raises((TypeError, ValueError, tf.errors.InvalidArgumentError)):
        relative_progress_impl(*operands(), required_components=mask)


def test_zero_iteration_budget_rejects_new_required_rows():
    result = make_relative_progress_function(1, explicit_component_targets=True, max_iterations=0)(
        *operands(), np.array([True]))
    assert not result["valid"]
    assert result["iterations"] == 0
