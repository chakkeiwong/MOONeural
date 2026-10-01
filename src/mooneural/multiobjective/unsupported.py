"""Method-status declarations for the shared TensorFlow multiobjective package.

`IMPLEMENTED_METHODS` describes runtime-exposed methods in the reusable
library.  Runtime exposure is not by itself equivalent to reviewed-canonical
promotion under the reusable-library status-audit artifacts.

Nash-MTL, SOAP, and DenseSOAP remain blocked in the canonical multiobjective
aggregation API until their TensorFlow-native aggregation contracts are
separately reviewed.  DenseSOAP is optimizer-side only:
`common_utils.tf_optimizers.DenseSOAP` has bounded CPU float32
official-source-compatible common-utils optimizer status under the reviewed
degenerate-eigenspace policy, but that status remains outside this aggregation
API.
"""

from __future__ import annotations


IMPLEMENTED_METHODS = (
    "mgda",
    "pcgrad",
    "imtl",
    "famo",
    "cagrad",
    "gradnorm",
    "aligned",
)
BLOCKED_METHODS = ("nash", "soap", "densesoap")
SUPPORTED_METHODS = IMPLEMENTED_METHODS + BLOCKED_METHODS

METHOD_CONTRACTS = {
    "mgda": {
        "contract_version": "mgda_active_set_tf_v1",
        "reference_status": "bounded_reference_faithful_small_k",
        "evidence_status": (
            "bounded_trial_ready_with_tests_and_analytic_moo_pilot; "
            "problem_dependent_not_default_policy"
        ),
    },
    "pcgrad": {
        "contract_version": "pcgrad_stateless_tf_v1",
        "reference_status": "bounded_reference_faithful_gradient_surgery",
        "evidence_status": (
            "bounded_trial_ready_with_tests_and_analytic_moo_pilot; "
            "problem_dependent_not_default_policy"
        ),
    },
    "imtl": {
        "contract_version": "imtl_g_tf_solve_fail_closed_v1",
        "reference_status": "bounded_reference_faithful_fail_closed",
        "evidence_status": (
            "bounded_trial_ready_with_tests_and_analytic_moo_pilot; "
            "rank_deficient_cells_are_domain_skips; "
            "problem_dependent_not_default_policy"
        ),
        "domain_note": (
            "requires nonzero task gradients and a full-rank well-conditioned "
            "IMTL-G linear system; duplicate/collinear/rank-deficient "
            "gradients are domain-ineligible and fail closed"
        ),
    },
    "famo": {
        "contract_version": "famo_tf_functional_state_v1",
        "reference_status": "bounded_reference_aligned_functional_state",
        "evidence_status": (
            "bounded_trial_ready_with_tests_and_analytic_moo_pilot; "
            "problem_dependent_not_default_policy"
        ),
    },
    "cagrad": {
        "contract_version": "cagrad_tf_active_set_kkt_v1",
        "reference_status": "bounded_reference_faithful_small_k_strict_oracle",
        "c_theorem_scope": (
            "c>=1_default_blocked_a5_path_removed_explicit_non_theorem_opt_in"
        ),
        "a5_status": "blocked_removed_not_implemented",
        "evidence_status": (
            "bounded_trial_ready_with_tests_and_analytic_moo_pilot; "
            "problem_dependent_not_default_policy"
        ),
    },
    "gradnorm": {
        "contract_version": "gradnorm_tf_paper_audited_v1",
        "reference_status": (
            "bounded_usable_selected_shared_gradients_not_official_code_parity"
        ),
        "evidence_status": (
            "bounded_trial_ready_with_tests_and_analytic_moo_pilot; "
            "requires explicit state/loss contract; "
            "problem_dependent_not_default_policy"
        ),
    },
    "nash": {
        "contract_version": "blocked_nash_bargaining_solver_missing_v1",
        "reference_status": "blocked_not_implemented",
    },
    "aligned": {
        "contract_version": "aligned_mtl_tf_procrustes_operator_v1",
        "reference_status": (
            "bounded_tf_procrustes_operator_not_benchmark_reproduction"
        ),
        "evidence_status": (
            "bounded_trial_ready_with_tests_and_analytic_moo_pilot; "
            "procrustes_operator_not_full_aligned_mtl_training_loop; "
            "problem_dependent_not_default_policy"
        ),
    },
    "soap": {
        "contract_version": "blocked_soap_optimizer_out_of_scope_v1",
        "reference_status": "blocked_not_implemented",
    },
    "densesoap": {
        "contract_version": "blocked_densesoap_optimizer_out_of_scope_v1",
        "reference_status": "blocked_not_implemented",
    },
}


class UnsupportedMethodError(NotImplementedError):
    """Raised when a method is intentionally blocked from canonical use."""


def ensure_method_supported(method: str) -> str:
    """Validate and normalize a method name."""
    normalized = str(method).lower()
    if normalized not in SUPPORTED_METHODS:
        raise UnsupportedMethodError(
            f"unknown TensorFlow multiobjective method: {method}")
    if normalized in BLOCKED_METHODS:
        contract = METHOD_CONTRACTS[normalized]["contract_version"]
        raise UnsupportedMethodError(
            f"{normalized} is intentionally not implemented in "
            "mooneural.multiobjective; contract "
            f"{contract} requires a separate reviewed TensorFlow-native "
            "implementation")
    return normalized
