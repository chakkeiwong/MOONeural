from dataclasses import replace

import pytest

from mooneural.training.generic_role_banks import (
    RoleBank,
    RoleBankEvaluators,
    RoleBankManifest,
)
from mooneural.training.generic_training_contracts import (
    CertificationResult,
    ControlEvaluation,
    PolicyView,
    ValidationEvaluation,
)

TASK_IDS = ("task-a", "task-b")
ROLES = ("control", "validation", "certification")


def make_manifest(count=2, *, include_global=True):
    banks = []
    for role_index, role in enumerate(ROLES):
        for bank_index in range(count):
            input_hashes = {"local": "b" * 64, "target": "c" * 64}
            if include_global:
                input_hashes["global"] = "a" * 64
            banks.append(
                RoleBank(
                    role=role,
                    bank_id=f"{role}-{bank_index}",
                    seed=2026091200 + role_index * 100 + bank_index,
                    target_id=f"{role}-target",
                    target_version="v1",
                    scale_version="v1",
                    estimator_version="mse-v1",
                    input_hashes=input_hashes,
                    sample_ids=(f"{role}-sample-{bank_index}",),
                )
            )
    return RoleBankManifest(
        TASK_IDS,
        tuple(banks),
        {role: count for role in ROLES},
    )


def result_callbacks(manifest):
    def control(policy, request):
        return ControlEvaluation(
            {task: 0.01 for task in TASK_IDS},
            request=request,
        )

    def validation(policy, request):
        return ValidationEvaluation(
            {task: 0.02 for task in TASK_IDS},
            {task: request.sample_count for task in TASK_IDS},
            {"role_bank_manifest_hash": manifest.binding_hash()},
            request=request,
        )

    def certification(policy, request):
        return CertificationResult(
            {"conjunct-a": True},
            {"task-a": 0.02, "task-b": 0.02},
            {"finite": True},
            {
                "task_ids": list(TASK_IDS),
                "role_bank_manifest_hash": manifest.binding_hash(),
            },
            request=request,
        )

    return control, validation, certification


def test_manifest_binds_exact_disjoint_banks_and_round_trips():
    manifest = make_manifest()
    policy = PolicyView((1.0,), "policy")
    request = manifest.request_for("validation", policy)

    assert request.seeds == (2026091300, 2026091301)
    assert request.sample_count == 2
    assert request.metadata["role_bank_manifest_hash"] == manifest.binding_hash()
    assert manifest.from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()
    manifest.validate_request(request, policy)
    manifest.validate_request_binding(request)


def test_manifest_rejects_missing_exact_count_or_duplicate_identifiers():
    valid = make_manifest()
    with pytest.raises(ValueError, match="exact expected counts"):
        RoleBankManifest(
            TASK_IDS,
            valid.banks[:-1],
            valid.expected_counts,
        )

    duplicate_seed = replace(valid.banks[1], seed=valid.banks[0].seed)
    with pytest.raises(ValueError, match="seeds must be unique"):
        RoleBankManifest(
            TASK_IDS,
            (valid.banks[0], duplicate_seed, *valid.banks[2:]),
            valid.expected_counts,
        )

    duplicate_sample = replace(valid.banks[1], sample_ids=valid.banks[0].sample_ids)
    with pytest.raises(ValueError, match="samples must be unique"):
        RoleBankManifest(
            TASK_IDS,
            (valid.banks[0], duplicate_sample, *valid.banks[2:]),
            valid.expected_counts,
        )


def test_manifest_fails_closed_without_exact_global_inputs():
    manifest = make_manifest(include_global=False)
    with pytest.raises(ValueError, match="exact global"):
        manifest.request_for("control", PolicyView((1.0,), "policy"))


def test_role_bank_evaluators_produce_bound_results():
    manifest = make_manifest()
    control, validation, certification = result_callbacks(manifest)
    evaluators = RoleBankEvaluators(
        manifest,
        control=control,
        validation=validation,
        certification=certification,
    )
    policy = PolicyView((1.0,), "policy")

    assert evaluators.control_evaluation(policy).request.role == "control"
    assert evaluators.validation_evaluation(policy).request.seeds == (
        2026091300,
        2026091301,
    )
    assert evaluators.certification_result(policy).request.role == "certification"


def test_role_bank_evaluators_reject_unbound_or_unprovenanced_results():
    manifest = make_manifest()
    policy = PolicyView((1.0,), "policy")
    control, _validation, certification = result_callbacks(manifest)

    evaluators = RoleBankEvaluators(
        manifest,
        control=lambda _policy, _request: ControlEvaluation(
            {task: 0.01 for task in TASK_IDS},
        ),
        validation=lambda _policy, request: ValidationEvaluation(
            {task: 0.02 for task in TASK_IDS},
            {task: request.sample_count for task in TASK_IDS},
            request=request,
        ),
        certification=certification,
    )
    with pytest.raises(ValueError, match="unbound request"):
        evaluators.control_evaluation(policy)

    evaluators = RoleBankEvaluators(
        manifest,
        control=control,
        validation=lambda _policy, request: ValidationEvaluation(
            {task: 0.02 for task in TASK_IDS},
            {task: request.sample_count for task in TASK_IDS},
            request=request,
        ),
        certification=certification,
    )
    with pytest.raises(ValueError, match="provenance"):
        evaluators.validation_evaluation(policy)


def test_certification_request_round_trip_is_explicit():
    manifest = make_manifest()
    _control, _validation, certification = result_callbacks(manifest)
    policy = PolicyView((1.0,), "policy")
    request = manifest.request_for("certification", policy)
    result = certification(policy, request)
    restored = CertificationResult.from_dict(result.to_dict())
    assert restored.request.to_dict() == request.to_dict()
