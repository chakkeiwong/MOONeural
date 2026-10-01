"""Coordinate choices change selectors and gradients independently."""

import numpy as np
import pytest

from mooneural.training.generic_scale_diagnostics import (
    candidate_ranking,
    coordinate_report,
    scale_profiles,
)


def test_selector_only_changes_ranking_without_changing_training_geometry():
    profiles = scale_profiles(("first", "second"), {"first": 1.0, "second": 100.0},
                              {"first": 1.0, "second": 1.0})
    raw_values = np.array([[0.8, 8.0], [1.0, 0.1]])
    rankings = {name: candidate_ranking(("left", "right"), raw_values, profile["selection"])
                for name, profile in profiles.items()}
    assert rankings["calibration"][0]["candidate_id"] == "left"
    assert all(rankings[name][0]["candidate_id"] == "right" for name in ("terminal_limit", "gate_0_04", "selector_only"))
    np.testing.assert_array_equal(profiles["calibration"]["training"], profiles["selector_only"]["training"])
    assert rankings["gate_0_04"][0]["minimax"] == pytest.approx(0.04 * rankings["terminal_limit"][0]["minimax"])


def test_values_and_gradients_use_mse_denominators_once():
    report = coordinate_report([4.0, 9.0], [[3.0, 4.0], [0.0, 12.0]], [2.0, 3.0], [0.5, 1.0])
    assert report["raw_rms"] == [2.0, 3.0]
    assert report["training_normalized_mse"] == [2.0, 3.0]
    assert report["training_gradient_norm"] == [2.5, 4.0]
    assert report["descriptive_gate_ratio"] == [8.0, 9.0]
    assert report["terminal_boundary_in_training_coordinates"] == [0.25, 1.0 / 3.0]


@pytest.mark.parametrize("bad_scales", ({"first": 1.0}, {"first": 1.0, "second": 0.0},
                                       {"first": 1.0, "second": float("nan")}))
def test_missing_or_invalid_scales_are_rejected(bad_scales):
    with pytest.raises(ValueError):
        scale_profiles(("first", "second"), bad_scales, {"first": 1.0, "second": 1.0})


def test_ranking_preserves_duplicate_policies_but_rejects_duplicate_ids():
    rows = [[1.0, 2.0], [1.0, 2.0]]
    assert [record["candidate_id"] for record in candidate_ranking(("b", "a"), rows, [1.0, 1.0])] == ["a", "b"]
    with pytest.raises(ValueError, match="inventory"):
        candidate_ranking(("a", "a"), rows, [1.0, 1.0])


@pytest.mark.parametrize("values", ([float("nan"), 1.0], [-1.0, 1.0]))
def test_invalid_metrics_are_rejected(values):
    with pytest.raises(ValueError):
        coordinate_report(values, [[1.0], [1.0]], [1.0, 1.0], [1.0, 1.0])
