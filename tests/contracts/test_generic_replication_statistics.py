"""Synthetic arithmetic checks; these are not independent scientific evidence."""

import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from mooneural.training.generic_replication_statistics import (
    PHASE5R_METRIC_ORDER,
    PHASE5R_METRIC_PATHS,
    ReplicationBankVectors,
    ReplicationStatisticsSpec,
    _interval_geometry,
    evaluate_paired_replication,
    evaluate_source_calibration,
    phase5r_statistics_spec,
)

ROOT = Path(__file__).resolve().parents[2]
ACCEPTED_PATH = ROOT / (
    "docs/plans/artifacts/rotemberg-public-explicit-state-selected-replica4-"
    "round331-380-continuation-2026-08-03/certification.json"
)
PATTERN = np.arange(20, dtype=float) - 9.5


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def identity_fields():
    return {
        "design_sha256": digest("synthetic-qualified-design"),
        "source_endpoint_sha256": digest("synthetic-frozen-source-endpoint"),
        "bank_registry_sha256": digest("synthetic-seed-registry"),
        "calibration_bank_ids": tuple(f"calibration-{index}" for index in range(20)),
        "comparison_bank_ids": tuple(f"comparison-{index}" for index in range(20)),
        "excluded_bank_ids": ("training-0", "validation-0", "selection-0", "certification-0"),
        "calibration_coordinate_sha256s": tuple(digest(f"calibration-input-target-{index}") for index in range(20)),
        "comparison_coordinate_sha256s": tuple(digest(f"comparison-input-target-{index}") for index in range(20)),
    }


def make_spec(accepted=None):
    if accepted is None:
        accepted = [0.015 + index * 0.0005 for index in range(19)]
        accepted[16] = 0.05371185695178999
    count = len(accepted)
    paths = PHASE5R_METRIC_PATHS if count == 19 else tuple((f"metric-{index}", "value") for index in range(count))
    order = PHASE5R_METRIC_ORDER if count == 19 else tuple(f"metric-{index}" for index in range(count))
    return ReplicationStatisticsSpec(
        **identity_fields(),
        accepted_target_sha256=digest("synthetic-accepted-target"),
        metric_order=order,
        metric_paths=paths,
        boolean_paths=tuple(path[:-1] + ("passed",) for path in paths),
        accepted_values=tuple(accepted),
        accepted_booleans=tuple(value <= 0.04 for value in accepted),
    )


def calibration_values(spec):
    slopes = np.linspace(0.0001, 0.0002, len(spec.metric_order))
    return np.asarray(spec.accepted_values) + np.outer(PATTERN, slopes)


def vectors(spec, values, *, paired=False, generic=False):
    return ReplicationBankVectors(
        spec_sha256=spec.binding_hash(),
        endpoint_sha256=digest("generic-endpoint") if generic else spec.source_endpoint_sha256,
        metric_order=spec.metric_order,
        bank_ids=spec.comparison_bank_ids if paired else spec.calibration_bank_ids,
        coordinate_sha256s=spec.comparison_coordinate_sha256s if paired else spec.calibration_coordinate_sha256s,
        values=values,
    )


def source_result(spec, values=None):
    return evaluate_source_calibration(spec, vectors(spec, calibration_values(spec) if values is None else values))


def paired_result(calibration, source, generic):
    return evaluate_paired_replication(
        calibration,
        vectors(calibration.spec, source, paired=True),
        vectors(calibration.spec, generic, paired=True, generic=True),
        generic_endpoint_sha256=digest("generic-endpoint"),
    )


def test_source_moments_match_hand_derived_linear_bank_family():
    spec = make_spec()
    result = source_result(spec).to_dict()
    slopes = np.linspace(0.0001, 0.0002, 19)
    expected_sd = slopes * math.sqrt(35)
    expected_lower = expected_sd * math.sqrt(19 / 30.14352720564616)

    assert sum(PATTERN) == 0
    assert sum(PATTERN ** 2) == 665
    np.testing.assert_allclose(result["U_source_current"], spec.accepted_values, atol=1e-17, rtol=0)
    np.testing.assert_allclose(result["sigma_point"], expected_sd, rtol=2e-14)
    np.testing.assert_allclose(result["sigma_gate"], expected_lower, rtol=2e-14)
    np.testing.assert_allclose(result["coefficient_of_variation"], expected_sd / np.abs(spec.accepted_values), rtol=2e-14)
    assert np.all(np.asarray(result["sigma_gate"]) < result["sigma_point"])
    assert result["status"] == "SOURCE_PROFILE_PASS"
    assert result["B_source_current"] == [True] * 16 + [False, True, True]
    assert result["source_profile"]["profile_passed"] is True
    assert result["U_accepted"][16] == 0.05371185695178999
    assert result["degrees_of_freedom"] == 19
    assert result["scope"] == "supplied_bank_vector_arithmetic"
    assert len(result["source_banks"]["values"]) == 20


def test_source_moments_match_scalar_statistics_reference_on_nonlinear_vectors():
    spec = make_spec()
    perturbations = np.sin(np.arange(380).reshape(20, 19)) * 0.0002
    values = np.asarray(spec.accepted_values) + perturbations
    result = source_result(spec, values).to_dict()
    expected_mean = [statistics.fmean(column) for column in values.T]
    expected_sd = [statistics.stdev(column) for column in values.T]
    expected_lower = [math.sqrt(19 * value * value / stats.chi2.isf(0.05, 19)) for value in expected_sd]

    np.testing.assert_allclose(result["U_source_current"], expected_mean, rtol=1e-14)
    np.testing.assert_allclose(result["sigma_point"], expected_sd, rtol=2e-14)
    np.testing.assert_allclose(result["sigma_gate"], expected_lower, rtol=2e-14)


def test_paired_differences_covariance_and_simultaneous_intervals_match_reference():
    spec = make_spec()
    calibration = source_result(spec)
    source = calibration_values(spec)
    slopes = np.arange(1, 20) * 1e-7
    shifts = np.arange(-9, 10) * 1e-7
    generic = source + shifts + np.outer(PATTERN, slopes)
    result = paired_result(calibration, source, generic)
    differences = [[float(generic[row, column]) - float(source[row, column]) for column in range(19)] for row in range(20)]
    means = [statistics.fmean(column) for column in zip(*differences)]
    covariance = [[
        math.fsum((row[left] - means[left]) * (row[right] - means[right]) for row in differences) / 19
        for right in range(19)
    ] for left in range(19)]
    intervals = [stats.t.interval(
        confidence=1 - 0.05 / 19,
        df=19,
        loc=statistics.fmean(column),
        scale=statistics.stdev(column) / math.sqrt(20),
    ) for column in zip(*differences)]

    np.testing.assert_array_equal(result["D"], differences)
    np.testing.assert_allclose(result["mean_D"], means, atol=2e-21, rtol=0)
    np.testing.assert_allclose(result["covariance_D"], covariance, rtol=2e-14)
    np.testing.assert_allclose(result["covariance_D"], 35 * np.outer(slopes, slopes), rtol=1e-11)
    np.testing.assert_allclose(result["confidence_intervals"], intervals, rtol=2e-13)
    assert result["student_t_critical"] == pytest.approx(3.458527135250202, rel=1e-14)
    assert result["per_coordinate_alpha"] == 0.05 / 19
    assert result["degrees_of_freedom"] == 19
    assert np.linalg.matrix_rank(result["covariance_D"]) < 19
    assert np.asarray(result["covariance_D"]).shape == (19, 19)
    assert result["status"] == "EQUIVALENT"
    assert result["paired_passed"] is True
    assert result["source_calibration_sha256"] == calibration.to_dict()["result_sha256"]


def test_exactly_equal_pairs_with_zero_variance_pass_valid_calibration():
    spec = make_spec()
    calibration = source_result(spec)
    paired = np.tile(spec.accepted_values, (20, 1))
    result = paired_result(calibration, paired, paired)

    assert result["status"] == "EQUIVALENT"
    np.testing.assert_array_equal(result["covariance_D"], np.zeros((19, 19)))
    np.testing.assert_array_equal(result["confidence_intervals"], np.zeros((19, 2)))
    assert result["interval_geometry"] == ["WITHIN_ENVELOPE"] * 19


@pytest.mark.parametrize("case", ("constant", "zero_mean", "all_zero", "nan", "inf", "-inf", "overflow"))
def test_invalid_source_variance_or_cv_is_inconclusive_and_never_rescued_by_pairs(case):
    spec = make_spec()
    values = calibration_values(spec)
    if case == "constant":
        values[:, 0] = 0.05371185695178999
    elif case == "zero_mean":
        values[:, 0] = PATTERN
    elif case == "all_zero":
        values[:, 0] = 0
    elif case == "overflow":
        values[:, 0] = PATTERN * 1e200
    else:
        values[0, 0] = float(case)
    calibration = source_result(spec, values)
    result = calibration.to_dict()
    paired = np.tile(spec.accepted_values, (20, 1))

    assert result["status"] == "INCONCLUSIVE"
    assert result["statistics_valid"] is False
    assert result["sigma_stable"][0] is False
    assert result["source_profile"]["profile_matches"][0] is None
    assert paired_result(calibration, paired, paired)["status"] == "INCONCLUSIVE"
    if case == "constant":
        assert result["sigma_point"][0] == 0
        assert result["coefficient_of_variation"][0] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("multiplier, expected", ((1, "SOURCE_PROFILE_PASS"), (1.01, "INCONCLUSIVE")))
def test_cv_one_half_boundary_is_inclusive(multiplier, expected):
    spec = make_spec((1.0, 1.0))
    offsets = np.asarray([sign * value for value in [0.5] * 7 + [0.75, 0.25, 0.0] for sign in (-1, 1)])
    values = np.column_stack((1 + offsets * multiplier, 1 + offsets))
    result = source_result(spec, values).to_dict()

    assert result["status"] == expected
    assert result["coefficient_of_variation"][0] == pytest.approx(0.5 * multiplier)
    assert result["sigma_stable"][1] is True


def test_defined_source_profile_failure_is_retained_by_final_helper():
    spec = make_spec()
    values = calibration_values(spec)
    values[:, 0] += 0.002
    calibration = source_result(spec, values)
    paired = np.tile(spec.accepted_values, (20, 1))

    assert calibration.status == "NOT_EQUIVALENT"
    assert calibration.to_dict()["source_profile"]["boolean_passed"] is True
    assert paired_result(calibration, paired, paired)["status"] == "NOT_EQUIVALENT"


@pytest.mark.parametrize("endpoint", ("source", "generic"))
def test_boolean_flip_fails_even_within_profile_tolerance(endpoint):
    accepted = list(make_spec().accepted_values)
    accepted[0] = 0.03999
    spec = make_spec(accepted)
    source_values = calibration_values(spec)
    paired = np.tile(spec.accepted_values, (20, 1))
    if endpoint == "source":
        source_values[:, 0] += 0.00002
        result = source_result(spec, source_values).to_dict()
        profile = result["source_profile"]
    else:
        paired[:, 0] += 0.00002
        result = paired_result(source_result(spec), paired, paired)
        profile = result["generic_profile"]
        assert result["paired_passed"] is True
    assert profile["profile_passed"] is True
    assert profile["boolean_passed"] is False
    assert result["status"] == "NOT_EQUIVALENT"


def test_generic_profile_fails_even_when_fresh_difference_intervals_pass():
    spec = make_spec()
    paired = np.tile(spec.accepted_values, (20, 1))
    paired[:, 0] += 0.002
    result = paired_result(source_result(spec), paired, paired)

    assert result["paired_passed"] is True
    assert result["generic_profile"]["boolean_passed"] is True
    assert result["generic_profile"]["profile_passed"] is False
    assert result["status"] == "NOT_EQUIVALENT"


@pytest.mark.parametrize("contains_zero", (True, False))
def test_intervals_crossing_envelope_are_inconclusive_even_if_profiles_pass(contains_zero):
    spec = make_spec()
    calibration = source_result(spec)
    bound = calibration.to_dict()["equivalence_half_width"][0]
    source = np.tile(spec.accepted_values, (20, 1))
    generic = source.copy()
    critical = stats.t.isf(0.05 / 38, 19)
    width = bound * (2 if contains_zero else 0.5)
    shift = 0 if contains_zero else 0.75 * bound
    generic[:, 0] += shift + PATTERN * width * math.sqrt(20) / (critical * math.sqrt(35))
    result = paired_result(calibration, source, generic)

    assert result["generic_profile"]["profile_passed"] is True
    assert result["generic_profile"]["boolean_passed"] is True
    assert result["interval_geometry"][0] == "CROSSES_BOUNDARY"
    assert result["status"] == "INCONCLUSIVE"
    assert result["paired_passed"] is False
    assert result["paired_failed"] is False


@pytest.mark.parametrize("sign", (-1, 1))
def test_wholly_outside_intervals_are_defined_failures_with_valid_statistics(sign):
    spec = make_spec()
    calibration = source_result(spec)
    bound = calibration.to_dict()["equivalence_half_width"][0]
    generic = np.tile(spec.accepted_values, (20, 1))
    source = generic.copy()
    source[:, 0] -= sign * 2 * bound
    result = paired_result(calibration, source, generic)

    assert result["generic_profile"]["profile_passed"] is True
    assert result["interval_geometry"][0] == "WHOLLY_OUTSIDE"
    assert result["status"] == "NOT_EQUIVALENT"
    assert result["paired_failed"] is True
    assert result["paired_passed"] is False
    assert result["paired_statistics_valid"] is True
    assert "definition_ambiguities" not in result


@pytest.mark.parametrize("lower, upper, expected", (
    (-0.125, 0.125, "WITHIN_ENVELOPE"),
    (0.125, 0.125, "WITHIN_ENVELOPE"),
    (-0.125, -0.125, "WITHIN_ENVELOPE"),
    (0.0, 0.0, "WITHIN_ENVELOPE"),
    (0.125, 0.25, "CROSSES_BOUNDARY"),
    (-0.25, -0.125, "CROSSES_BOUNDARY"),
    (-0.25, 0.25, "CROSSES_BOUNDARY"),
    (0.0625, 0.25, "CROSSES_BOUNDARY"),
    (-0.25, -0.0625, "CROSSES_BOUNDARY"),
    (0.25, 0.375, "WHOLLY_OUTSIDE"),
    (-0.375, -0.25, "WHOLLY_OUTSIDE"),
    (float(np.nextafter(0.125, np.inf)), float(np.nextafter(0.125, np.inf)), "WHOLLY_OUTSIDE"),
    (float(np.nextafter(-0.125, -np.inf)), float(np.nextafter(-0.125, -np.inf)), "WHOLLY_OUTSIDE"),
))
def test_accepted_interval_table_includes_directional_touching_without_epsilon(lower, upper, expected):
    assert _interval_geometry(lower, upper, 0.125, True) == expected


@pytest.mark.parametrize("case", ("valid", "invalid_source_sigma", "nonfinite_pair"))
def test_disjoint_precedes_crossing_only_with_valid_statistics(case):
    spec = make_spec()
    source_calibration = calibration_values(spec)
    if case == "invalid_source_sigma":
        source_calibration[:, 2] = spec.accepted_values[2]
    calibration = source_result(spec, source_calibration)
    bound = calibration.to_dict()["equivalence_half_width"][0]
    generic = np.tile(spec.accepted_values, (20, 1))
    source = generic.copy()
    source[:, 0] -= 2 * bound
    generic[:, 1] += PATTERN * 0.0002
    if case == "nonfinite_pair":
        generic[0, 2] = float("nan")
    result = paired_result(calibration, source, generic)

    assert result["interval_geometry"][:2] == ["WHOLLY_OUTSIDE", "CROSSES_BOUNDARY"]
    assert result["paired_passed"] is False
    assert result["paired_failed"] is (case == "valid")
    assert result["status"] == ("NOT_EQUIVALENT" if case == "valid" else "INCONCLUSIVE")


def test_invalid_source_pairs_override_finite_failed_generic_profile():
    spec = make_spec()
    calibration = source_result(spec)
    source = np.tile(spec.accepted_values, (20, 1))
    generic = source.copy()
    source[0, 2] = float("nan")
    generic[:, 0] = 0.05
    result = paired_result(calibration, source, generic)

    assert result["status"] == "INCONCLUSIVE"
    assert result["paired_statistics_valid"] is False
    assert result["paired_failed"] is False
    assert result["paired_passed"] is False
    assert result["source_calibration"] == calibration.to_dict()
    assert result["source_calibration"]["status"] == "SOURCE_PROFILE_PASS"
    assert all(math.isfinite(value) for value in result["U_replication"])
    assert result["generic_profile"]["profile_passed"] is False
    assert result["generic_profile"]["profile_matches"][0] is False
    assert result["generic_profile"]["boolean_passed"] is False
    assert result["generic_profile"]["boolean_matches"][0] is False
    assert result["source_pairs"]["values"][0][2] == "NaN"
    assert len(result["source_pairs"]["values"]) == 20


def test_invalid_pairs_override_failed_source_profile():
    spec = make_spec()
    calibration_values_with_failure = calibration_values(spec)
    calibration_values_with_failure[:, 0] += 0.002
    calibration = source_result(spec, calibration_values_with_failure)
    source = np.tile(spec.accepted_values, (20, 1))
    generic = source.copy()
    generic[0, 2] = float("nan")
    result = paired_result(calibration, source, generic)

    assert result["status"] == "INCONCLUSIVE"
    assert result["paired_statistics_valid"] is False
    assert result["paired_failed"] is False
    assert result["paired_passed"] is False
    assert result["source_calibration"] == calibration.to_dict()
    assert result["source_calibration"]["statistics_valid"] is True
    assert result["source_calibration"]["status"] == "NOT_EQUIVALENT"
    assert result["source_calibration"]["source_profile"]["profile_passed"] is False
    assert result["source_calibration"]["source_profile"]["profile_matches"][0] is False
    assert result["source_calibration"]["source_profile"]["boolean_passed"] is True
    assert result["generic_pairs"]["values"][0][2] == "NaN"
    assert len(result["generic_pairs"]["values"]) == 20


@pytest.mark.parametrize("failure", ("source_profile", "generic_profile"))
def test_crossing_does_not_erase_a_defined_profile_failure(failure):
    spec = make_spec()
    source_calibration = calibration_values(spec)
    if failure == "source_profile":
        source_calibration[:, 0] += 0.002
    generic = np.tile(spec.accepted_values, (20, 1))
    source = generic.copy()
    generic[:, 1] += PATTERN * 0.0002
    if failure == "generic_profile":
        generic[:, 0] += 0.002
        source[:, 0] += 0.002
    result = paired_result(source_result(spec, source_calibration), source, generic)

    assert "WHOLLY_OUTSIDE" not in result["interval_geometry"]
    assert result["interval_geometry"][1] == "CROSSES_BOUNDARY"
    assert result["paired_failed"] is False
    assert result["status"] == "NOT_EQUIVALENT"


@pytest.mark.parametrize("sign", (-1, 1))
def test_exact_envelope_boundary_is_included(sign):
    spec = make_spec()
    calibration = source_result(spec)
    bound = calibration.to_dict()["equivalence_half_width"][0]
    generic = np.tile(spec.accepted_values, (20, 1))
    source = generic.copy()
    generic[:, 0] = 0
    source[:, 0] = -sign * bound
    result = paired_result(calibration, source, generic)

    assert result["confidence_intervals"][0] == [sign * bound, sign * bound]
    assert result["interval_geometry"][0] == "WITHIN_ENVELOPE"


@pytest.mark.parametrize("value, boolean", ((0.04, True), (float(np.nextafter(0.04, np.inf)), False)))
def test_source_boolean_uses_original_threshold_without_rounding(value, boolean):
    accepted = list(make_spec().accepted_values)
    accepted[0] = value
    spec = make_spec(accepted)
    result = source_result(spec).to_dict()

    assert result["U_source_current"][0] == value
    assert result["B_source_current"][0] is boolean


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_comparison_is_inconclusive_and_preserves_supplied_rows(value):
    spec = make_spec()
    source = np.tile(spec.accepted_values, (20, 1))
    generic = source.copy()
    generic[0, 0] = value
    result = paired_result(source_result(spec), source, generic)

    assert result["status"] == "INCONCLUSIVE"
    assert result["paired_statistics_valid"] is False
    assert len(result["generic_pairs"]["values"]) == 20
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("group", ("calibration_bank_ids", "comparison_bank_ids"))
@pytest.mark.parametrize("count", (19, 21))
def test_spec_refuses_extra_or_missing_banks(group, count):
    spec = make_spec()
    with pytest.raises(ValueError, match="exactly 20"):
        replace(spec, **{group: tuple(f"bank-{index}" for index in range(count))})


@pytest.mark.parametrize("role", ("training-0", "validation-0", "selection-0", "certification-0", "calibration-0"))
def test_seed_registry_overlap_is_refused(role):
    spec = make_spec()
    with pytest.raises(ValueError, match="overlap"):
        replace(spec, comparison_bank_ids=(role,) + spec.comparison_bank_ids[1:])


def test_duplicate_bank_ids_and_reused_coordinates_are_refused():
    spec = make_spec()
    with pytest.raises(ValueError, match="duplicate"):
        replace(spec, calibration_bank_ids=("same",) * 20)
    with pytest.raises(ValueError, match="overlap"):
        replace(spec, comparison_coordinate_sha256s=spec.calibration_coordinate_sha256s)
    with pytest.raises(ValueError, match="duplicate"):
        replace(spec, calibration_coordinate_sha256s=(digest("same"),) * 20)


@pytest.mark.parametrize("change, message", (
    ({"spec_sha256": digest("other-design")}, "spec binding"),
    ({"endpoint_sha256": digest("round-90-parent")}, "endpoint binding"),
    ({"metric_order": tuple(reversed(PHASE5R_METRIC_ORDER))}, "metric order"),
    ({"bank_ids": tuple(f"other-{index}" for index in range(20))}, "IDs/order/count"),
    ({"coordinate_sha256s": (digest("other-inputs"),) * 20}, "input/target coordinates"),
))
def test_calibration_identity_mismatch_is_refused(change, message):
    spec = make_spec()
    banks = replace(vectors(spec, calibration_values(spec)), **change)
    with pytest.raises(ValueError, match=message):
        evaluate_source_calibration(spec, banks)


@pytest.mark.parametrize("paired", (False, True))
def test_supplied_vector_count_is_not_repaired_or_dropped(paired):
    spec = make_spec()
    banks = vectors(spec, calibration_values(spec), paired=paired)
    banks = replace(banks, bank_ids=banks.bank_ids[:-1], coordinate_sha256s=banks.coordinate_sha256s[:-1], values=banks.values[:-1])
    with pytest.raises(ValueError, match="IDs/order/count"):
        if paired:
            evaluate_paired_replication(
                source_result(spec), banks,
                vectors(spec, calibration_values(spec), paired=True, generic=True),
                generic_endpoint_sha256=digest("generic-endpoint"),
            )
        else:
            evaluate_source_calibration(spec, banks)


@pytest.mark.parametrize("field_name, replacement", (
    ("coordinate_sha256s", tuple(reversed(identity_fields()["comparison_coordinate_sha256s"]))),
    ("metric_order", tuple(reversed(PHASE5R_METRIC_ORDER))),
    ("spec_sha256", digest("other-design")),
    ("endpoint_sha256", digest("other-endpoint")),
    ("bank_ids", identity_fields()["calibration_bank_ids"]),
))
def test_paired_source_and_generic_must_have_identical_bound_coordinates(field_name, replacement):
    spec = make_spec()
    values = calibration_values(spec)
    generic = replace(vectors(spec, values, paired=True, generic=True), **{field_name: replacement})
    with pytest.raises(ValueError):
        evaluate_paired_replication(
            source_result(spec), vectors(spec, values, paired=True), generic,
            generic_endpoint_sha256=digest("generic-endpoint"),
        )


def test_required_identities_accepted_boolean_consistency_and_vector_shapes():
    spec = make_spec()
    with pytest.raises(ValueError, match="SHA-256"):
        replace(spec, source_endpoint_sha256="")
    with pytest.raises(ValueError, match="0.04"):
        replace(spec, accepted_booleans=(True,) * 19)
    with pytest.raises(ValueError, match="numeric row"):
        vectors(spec, np.zeros((20, 18)))
    with pytest.raises(ValueError, match="numeric row"):
        vectors(spec, np.zeros((20, 19), dtype=complex))
    with pytest.raises(ValueError, match="complete finite"):
        replace(spec, accepted_values=(float("nan"),) * 19)
    with pytest.raises(TypeError, match="key sequences"):
        replace(spec, metric_paths=tuple(".".join(path) for path in spec.metric_paths))
    with pytest.raises(TypeError, match="sequence of IDs"):
        replace(spec, excluded_bank_ids="training")


def test_snapshots_are_immutable_and_bind_all_thresholds_and_identities():
    spec = make_spec()
    values = calibration_values(spec)
    banks = vectors(spec, values)
    calibration = evaluate_source_calibration(spec, banks)
    before = calibration.to_dict()
    values[:] = 999
    altered_copy = calibration.to_dict()
    altered_copy["sigma_gate"][0] = 999

    assert calibration.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        calibration._payload_json = "{}"
    rules = spec.to_dict()
    assert {key: rules[key] for key in (
        "calibration_count", "comparison_count", "ddof", "source_threshold", "sd_multiplier",
        "maximum_cv", "family_alpha", "variance_bound_probability",
    )} == {
        "calibration_count": 20, "comparison_count": 20, "ddof": 1,
        "source_threshold": 0.04, "sd_multiplier": 0.1, "maximum_cv": 0.5,
        "family_alpha": 0.05, "variance_bound_probability": 0.95,
    }
    assert replace(spec, design_sha256=digest("other-design")).binding_hash() != spec.binding_hash()
    assert replace(spec, accepted_target_sha256=digest("other-target")).binding_hash() != spec.binding_hash()
    assert "independent-normal" in rules["normality_assumption"]
    assert "strict disjointness" in rules["interval_disposition_rule"]
    assert "outside_interval_ambiguity" not in rules




def test_generic_family_size_controls_bonferroni_without_a_model_loop():
    spec = make_spec((0.015, 0.025))
    pairs = calibration_values(spec)
    result = paired_result(source_result(spec), pairs, pairs)

    assert result["status"] == "EQUIVALENT"
    assert result["per_coordinate_alpha"] == 0.05 / 2
    assert np.asarray(result["covariance_D"]).shape == (2, 2)


def test_module_import_remains_host_only():
    subprocess.run([
        sys.executable, "-B", "-c",
        (
            "import sys; import mooneural.training.generic_replication_statistics; "
            "assert not {'tensorflow', 'jax', 'torch'}.intersection(sys.modules)"
        ),
    ], check=True, cwd=ROOT, capture_output=True, text=True)
