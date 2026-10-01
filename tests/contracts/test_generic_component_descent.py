"""Equality witness qualification without model or optimizer calls."""

import json
from pathlib import Path

import numpy as np
import pytest

from mooneural.training.generic_component_descent import make_component_descent_function


@pytest.mark.parametrize("task_count", [7, 9])
def test_public_and_component_equalities_preserve_source_norm(task_count):
    public = np.eye(task_count + 1)[:task_count]
    active = np.arange(task_count) < 3
    components = np.eye(task_count + 1)[[task_count, 0]]
    source = np.full(task_count + 1, .003)
    result = make_component_descent_function(task_count)(public, active, components, source)
    witness = np.zeros(task_count + 1)
    witness[:3], witness[-1] = -1., -1.
    expected = witness * np.linalg.norm(source) / np.linalg.norm(witness)
    assert bool(result["valid"])
    assert int(result["rank"]) == task_count + 1
    np.testing.assert_allclose(result["direction"], expected, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(result["equality_dots"], np.r_[-active.astype(float), -np.ones(2)], atol=1e-12)
    assert float(result["relative_residual"]) < 1e-12


def test_positive_row_rescaling_permutation_and_bounded_polymorphism():
    function = make_component_descent_function(3)
    public = np.eye(4)[:3]
    active = np.array([True, False, True])
    components = np.array([[0., 0., 0., 1.], [1., 0., 0., 0.]])
    source = np.full(4, .1)
    original = function(public, active, components, source)
    order = [2, 0, 1]
    scaled = function(public[order] * np.array([[1e-200], [1e200], [3.]]), active[order],
                      components[::-1] * np.array([[1e200], [1e-200]]), source)
    redundant = function(public, active, components[:1], source)
    for result in (original, scaled, redundant):
        assert bool(result["valid"])
        np.testing.assert_allclose(result["direction"], original["direction"], atol=1e-15)
    assert function.experimental_get_tracing_count() == 1
    assert not function.get_concrete_function().variables


def test_consistent_rank_deficiency_and_zero_protected_row_are_allowed():
    function = make_component_descent_function(3)
    result = function(np.array([[2., 0.], [4., 0.], [0., 0.]]), np.array([True, True, False]),
                      np.array([[8., 0.]]), np.array([.2, 0.]))
    assert bool(result["valid"]) and int(result["rank"]) == 1
    np.testing.assert_allclose(result["direction"], [-.2, 0.], atol=1e-16)


@pytest.mark.parametrize("fault", ["opposing", "zero_active", "zero_component", "nan", "infinite", "zero_step"])
def test_inconsistent_or_nonfinite_witness_is_invalid_without_infeasibility_claim(fault):
    public = np.eye(2)
    components = np.array([[1., 0.]])
    source = np.array([.1, 0.])
    if fault == "opposing":
        components *= -1.
    elif fault == "zero_active":
        public[0] = 0.
    elif fault == "zero_component":
        components[:] = 0.
    elif fault == "nan":
        components[0, 0] = np.nan
    elif fault == "infinite":
        source[0] = np.inf
    else:
        source[:] = 0.
    result = make_component_descent_function(2)(public, np.array([True, False]), components, source)
    assert not bool(result["valid"])


def test_achieved_protected_dots_use_relative_roundoff_allowance():
    public = np.array([[1., 2., 3.], [2., -1., 1.]])
    active = np.array([True, False])
    source = np.array([1e-9, 2e-9, -1e-9])
    result = make_component_descent_function(2)(public, active, np.empty((0, 3)), source)
    assert bool(result["valid"])
    assert float(result["dots"][0]) < 0.
    assert abs(float(result["dots"][1])) <= 1e-12 * np.linalg.norm(source)
