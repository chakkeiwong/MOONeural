"""Independent feasibility, scaling, failure and saved-reference checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mooneural.training.generic_relative_progress import make_relative_progress_function

ROOT = Path(__file__).resolve().parents[2]


def evaluate(values, rows, mask, *, component_values=(), components=None, owners=(), radius=.1,
             max_iterations=200):
    rows = np.asarray(rows, np.float64)
    if components is None:
        components = np.empty((0, rows.shape[1]), np.float64)
    source = np.zeros(rows.shape[1], np.float64)
    source[0] = radius
    kernel = make_relative_progress_function(len(values), max_iterations=max_iterations)
    result = kernel(np.asarray(values, np.float64), rows, np.asarray(mask, bool),
        np.asarray(component_values, np.float64), np.asarray(components, np.float64),
        np.asarray(owners, np.int32), source)
    return {name: value.numpy() for name, value in result.items()}


def require_valid(result, radius=.1):
    assert result["valid"]
    assert result["converged"]
    np.testing.assert_allclose(np.linalg.norm(result["direction"]), radius, rtol=1e-10)
    assert result["primal_violation"] <= 1e-10
    assert result["full_row_violation"] <= radius * 1e-10
    assert result["stationarity_residual"] <= 1e-10
    assert result["complementarity_residual"] <= 1e-10


def test_stationary_full_face_releases_negative_multiplier_against_subset_oracle():
    path = ROOT / "tests/fixtures/relative-progress.json"
    fixture = json.loads(path.read_text())
    rows = np.asarray(fixture["gradients"], np.float64)
    values = np.asarray(fixture["values"], np.float64)
    mask = np.asarray(fixture["mask"], bool)
    radius = np.linalg.norm(fixture["source_displacement"])
    unit_rows = rows / np.linalg.norm(rows, axis=1)[:, None]
    seed = np.linalg.lstsq(unit_rows, -mask.astype(float), rcond=None)[0]
    seed /= np.linalg.norm(seed)
    targets = np.where(mask, values / np.linalg.norm(rows, axis=1), 0.)
    scale = np.min(-(unit_rows @ seed)[mask] / targets[mask])
    bounds = -scale * targets
    candidates = []
    for bits in range(1, 2 ** len(values)):
        selected = np.array([bool(bits & (1 << index)) for index in range(len(values))])
        active_rows = unit_rows[selected]
        coordinates = np.linalg.lstsq(active_rows, bounds[selected], rcond=None)[0]
        multipliers = np.linalg.lstsq(active_rows.T, -coordinates, rcond=None)[0]
        slack = unit_rows @ coordinates - bounds
        if (np.max(slack) <= 1e-10 and np.min(multipliers) >= -1e-10
                and np.linalg.norm(coordinates + active_rows.T @ multipliers) <= 1e-10
                and np.max(np.abs(multipliers * slack[selected])) <= 1e-10):
            candidates.append(coordinates)
    assert candidates
    reference = min(candidates, key=np.linalg.norm)
    reference = reference / np.linalg.norm(reference) * radius
    result = evaluate(values, rows, mask, radius=radius)
    require_valid(result, radius)
    assert result["iterations"] < 200
    np.testing.assert_allclose(result["direction"], reference, rtol=1e-9, atol=radius * 1e-11)


def test_unequal_loss_sensitivities_have_equal_fractional_progress():
    result = evaluate([1., 1.], [[1., 0.], [0., 2.]], [True, True])
    require_valid(result)
    expected = -.1 * np.array([1., .5]) / np.sqrt(1.25)
    np.testing.assert_allclose(result["direction"], expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result["minimum_fractional_progress"], .1 / np.sqrt(1.25), rtol=1e-9)


def test_protected_slope_inequality_permits_useful_descent():
    result = evaluate([1., .01], [[1., 1.], [0., 1.]], [True, False])
    require_valid(result)
    np.testing.assert_allclose(result["direction"], -.1 / np.sqrt(2.) * np.ones(2), rtol=1e-8, atol=1e-11)
    assert result["unit_direction_dots"][1] < 0.


def test_protected_slope_can_bind_without_losing_radius():
    result = evaluate([1., .01], [[1., 1.], [0., -1.]], [True, False])
    require_valid(result)
    np.testing.assert_allclose(result["direction"], [-.1, 0.], rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize("copies", [1, 2, 5])
def test_redundant_tied_component_rows_preserve_the_analytical_minimum(copies):
    result = evaluate([1., 1.], [[1., 0.], [0., 2.]], [True, True],
        component_values=np.ones(copies), components=np.tile([1., 0.], (copies, 1)), owners=np.zeros(copies, int))
    require_valid(result)
    assert result["tied_components"].all()
    np.testing.assert_allclose(result["direction"], -.1 * np.array([1., .5]) / np.sqrt(1.25), rtol=1e-8, atol=1e-11)


def test_below_maximum_component_is_nonpositive_without_equal_margin():
    result = evaluate([1.], [[1., 0.]], [True], component_values=[.5], components=[[0., 1.]], owners=[0])
    require_valid(result)
    np.testing.assert_allclose(result["direction"], [-.1, 0.], rtol=1e-8, atol=1e-11)
    assert not result["tied_components"][0]
    assert result["unit_direction_dots"][1] <= 1e-11


@pytest.mark.parametrize("gap,tied", [(0., True), (1e-12, True), (1e-8, False)])
def test_near_tie_changes_only_the_declared_direction_constraint(gap, tied):
    result = evaluate([1.], [[1., 0.]], [True], component_values=[1. - gap], components=[[0., 1.]], owners=[0])
    require_valid(result)
    assert bool(result["tied_components"][0]) is tied
    expected = -.1 / np.sqrt(2.) * np.ones(2) if tied else np.array([-.1, 0.])
    np.testing.assert_allclose(result["direction"], expected, rtol=1e-8, atol=1e-11)


def test_inconsistent_equality_seed_rejects_without_claiming_inequality_infeasibility():
    result = evaluate([1.], [[.5, .5]], [True], component_values=[1., 1.],
        components=[[1., 0.], [0., 1.]], owners=[0, 0])
    assert not result["seed_valid"]
    assert not result["valid"]
    assert result["iterations"] == 0
    feasible = np.array([-1., -1.])
    assert np.all(np.array([[.5, .5], [1., 0.], [0., 1.]]) @ feasible <= -1.)


@pytest.mark.parametrize("factor", [1e-6, 1e6])
def test_positive_owner_loss_and_gradient_rescaling_preserves_direction(factor):
    baseline = evaluate([1., 1.], [[1., 0.], [0., 2.]], [True, True],
        component_values=[1.], components=[[1., 0.]], owners=[0])
    result = evaluate([factor, 1.], [[factor, 0.], [0., 2.]], [True, True],
        component_values=[factor], components=[[factor, 0.]], owners=[0])
    require_valid(result)
    np.testing.assert_allclose(result["direction"], baseline["direction"], rtol=1e-8, atol=1e-11)


def test_row_permutation_preserves_direction():
    baseline = evaluate([1., 1.], [[1., 0.], [0., 2.]], [True, True],
        component_values=[1., .5], components=[[1., 0.], [0., 2.]], owners=[0, 1])
    result = evaluate([1., 1.], [[0., 2.], [1., 0.]], [True, True],
        component_values=[.5, 1.], components=[[0., 2.], [1., 0.]], owners=[0, 1])
    require_valid(result)
    np.testing.assert_allclose(result["direction"], baseline["direction"], rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize("radius", [0., float("nan"), float("inf")])
def test_invalid_source_radius_cannot_nominate(radius):
    assert not evaluate([1.], [[1., 0.]], [True], radius=radius)["valid"]


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_descent_loss_cannot_nominate(value):
    assert not evaluate([value], [[1., 0.]], [True])["valid"]


@pytest.mark.parametrize("row", [[0., 0.], [float("nan"), 0.], [float("inf"), 0.]])
def test_invalid_descent_gradient_cannot_nominate(row):
    assert not evaluate([1.], [row], [True])["valid"]


def test_component_above_public_maximum_cannot_nominate():
    assert not evaluate([1.], [[1., 0.]], [True], component_values=[2.], components=[[0., 1.]], owners=[0])["valid"]


def test_component_owned_by_nondescent_task_cannot_nominate():
    assert not evaluate([1., 1.], [[1., 0.], [0., 1.]], [True, False],
        component_values=[.5], components=[[0., 1.]], owners=[1])["valid"]


def test_empty_descent_mask_cannot_nominate():
    assert not evaluate([1.], [[1., 0.]], [False])["valid"]


def test_iteration_exhaustion_is_explicit_and_cannot_nominate():
    result = evaluate([1., 1.], [[1., 0.], [0., 2.]], [True, True], max_iterations=0)
    assert result["seed_valid"]
    assert not result["converged"]
    assert not result["valid"]
    assert result["iterations"] == 0


def test_stable_signature_does_not_retrace_for_parameter_count():
    kernel = make_relative_progress_function(1)
    for width in (2, 3):
        result = kernel(np.ones(1), np.ones((1, width)), np.ones(1, bool), np.empty(0),
            np.empty((0, width)), np.empty(0, np.int32), np.ones(width) * .01)
        assert result["valid"]
    assert kernel.experimental_get_tracing_count() == 1




@pytest.mark.parametrize("size", [1e300, 1.7e308])
def test_finite_extreme_row_retains_its_relative_target_with_an_ordinary_row(size):
    values = np.array([size, .01])
    rows = np.array([[size, size], [0., 1.]])
    result = evaluate(values, rows, [True, True])
    require_valid(result)
    assert result["targets_valid"]
    fractional_decrease = -(rows / values[:, None]) @ result["direction"]
    np.testing.assert_allclose(result["minimum_fractional_progress"], np.min(fractional_decrease), rtol=1e-9)
    np.testing.assert_allclose(result["direction"], [-.1 / np.sqrt(2.), -.1 / np.sqrt(2.)], rtol=1e-8, atol=1e-11)


def test_extreme_owner_component_ratio_matches_unscaled_direction():
    baseline = evaluate([1., .01], [[1., 1.], [0., 1.]], [True, True],
        component_values=[1.], components=[[1., 1.]], owners=[0])
    result = evaluate([1.7e308, .01], [[1.7e308, 1.7e308], [0., 1.]], [True, True],
        component_values=[1.7e308], components=[[1.7e308, 1.7e308]], owners=[0])
    require_valid(result)
    np.testing.assert_allclose(result["direction"], baseline["direction"], rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(result["minimum_fractional_progress"], baseline["minimum_fractional_progress"], rtol=1e-9)


@pytest.mark.parametrize("loss,gradient", [(1e-300, 1e300), (1e300, 1e-300)])
def test_unrepresentable_required_target_cannot_hide_behind_a_healthy_row(loss, gradient):
    result = evaluate([loss, 1.], [[gradient, 0.], [0., 1.]], [True, True])
    assert not result["targets_valid"]
    assert not result["seed_valid"]
    assert not result["valid"]
    assert result["iterations"] == 0
