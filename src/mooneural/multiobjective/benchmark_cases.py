"""Dependency-free synthetic cases for multiobjective aggregation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


GradientRows = Tuple[Tuple[float, ...], ...]
LossVector = Tuple[float, ...]


@dataclass(frozen=True)
class GradientAlgebraCase:
    """Synthetic gradient-algebra case used by the Tier 0 leaderboard."""

    case_id: str
    case_family: str
    gradients: GradientRows
    losses: LossVector
    expected_gate_behavior: str
    explanatory_purpose: str
    seed: Optional[int] = None

    @property
    def objective_count(self) -> int:
        return len(self.gradients)

    @property
    def gradient_dim(self) -> int:
        return len(self.gradients[0])


def _rows(rows) -> GradientRows:
    return tuple(tuple(float(value) for value in row) for row in rows)


def _many_task_rows() -> GradientRows:
    return _rows((
        (1.0, -0.5, 0.25, 0.0, 0.75),
        (-0.8, 0.4, -0.2, 0.1, -0.6),
        (0.0, 1.0, 0.0, 0.2, -0.1),
        (0.0, -0.9, 0.1, -0.2, 0.0),
        (0.3, 0.2, 1.1, 0.0, 0.4),
        (-0.2, 0.1, -0.9, 0.3, -0.3),
        (0.0, 0.2, -0.1, 1.2, 0.1),
        (0.4, -0.1, 0.0, -0.8, 0.9),
    ))


def tier0_gradient_algebra_cases() -> Tuple[GradientAlgebraCase, ...]:
    """Return the frozen Tier 0 synthetic gradient-algebra case set."""

    return (
        GradientAlgebraCase(
            case_id="aligned_2d_k3",
            case_family="aligned",
            gradients=_rows(((1.0, 0.0), (2.0, 0.0), (0.5, 0.0))),
            losses=(1.0, 1.2, 0.8),
            expected_gate_behavior=(
                "Solvable methods return finite nonzero direction aligned "
                "with shared descent; simplex methods preserve coefficient "
                "diagnostics."
            ),
            explanatory_purpose="Checks non-conflict behavior and scale handling.",
        ),
        GradientAlgebraCase(
            case_id="opposing_2d_k2",
            case_family="opposing",
            gradients=_rows(((1.0, 0.0), (-1.0, 0.0))),
            losses=(1.0, 1.0),
            expected_gate_behavior=(
                "MGDA-like methods may return zero or near-zero direction; "
                "PCGrad must record conflict projection; solvable methods "
                "must remain finite."
            ),
            explanatory_purpose="Tests exact gradient conflict.",
        ),
        GradientAlgebraCase(
            case_id="orthogonal_2d_k2",
            case_family="orthogonal",
            gradients=_rows(((1.0, 0.0), (0.0, 1.0))),
            losses=(1.0, 1.0),
            expected_gate_behavior=(
                "No method should report nonfinite diagnostics; PCGrad "
                "should not project non-conflicts."
            ),
            explanatory_purpose="Tests non-conflicting independent tasks.",
        ),
        GradientAlgebraCase(
            case_id="dominated_2d_k3",
            case_family="dominated",
            gradients=_rows(((1.0, 0.0), (-1.0, 0.0), (10.0, 0.0))),
            losses=(1.0, 1.0, 2.0),
            expected_gate_behavior=(
                "Minimum-norm/simplex diagnostics should show dominated "
                "tasks can receive negligible weight; rows preserve explicit "
                "status."
            ),
            explanatory_purpose="Tests dominated gradient behavior.",
        ),
        GradientAlgebraCase(
            case_id="rank_deficient_k4_dim2",
            case_family="rank-deficient",
            gradients=_rows(((1.0, 0.0), (2.0, 0.0), (-1.0, 0.0), (-2.0, 0.0))),
            losses=(1.0, 1.2, 0.9, 1.1),
            expected_gate_behavior=(
                "Solvable methods must be finite, or fail closed explicitly "
                "if their contract cannot handle singular systems."
            ),
            explanatory_purpose="Tests singular/rank-deficient Gram handling.",
        ),
        GradientAlgebraCase(
            case_id="scale_imbalance_k3_dim3",
            case_family="scale-imbalanced",
            gradients=_rows(((100.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.1))),
            losses=(10.0, 1.0, 0.1),
            expected_gate_behavior=(
                "Finite outputs; diagnostics expose scale sensitivity; no "
                "default or superiority claim."
            ),
            explanatory_purpose="Tests scale imbalance and loss/task weighting behavior.",
        ),
        GradientAlgebraCase(
            case_id="zero_gradient_one_task_k3",
            case_family="zero-gradient",
            gradients=_rows(((1.0, 0.0), (0.0, 0.0), (0.0, 1.0))),
            losses=(1.0, 0.5, 1.0),
            expected_gate_behavior=(
                "Methods whose contract requires nonzero task gradients must "
                "fail closed or mark expected fail; others must remain finite."
            ),
            explanatory_purpose="Tests zero-gradient handling.",
        ),
        GradientAlgebraCase(
            case_id="many_task_k8_dim5_seeded",
            case_family="many-task stress",
            gradients=_many_task_rows(),
            losses=(1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4),
            expected_gate_behavior=(
                "Finite outputs for methods supporting K=8; explicit "
                "skip/fail for methods whose documented bounds do not allow "
                "the case."
            ),
            explanatory_purpose="Tests small many-task stress without large-scale claims.",
            seed=20260612,
        ),
        GradientAlgebraCase(
            case_id="signed_loss_invalid_k3",
            case_family="loss/stateful fail-closed",
            gradients=_rows(((1.0, 0.0), (2.0, 0.0), (0.5, 0.0))),
            losses=(1.0, -0.1, 0.5),
            expected_gate_behavior=(
                "FAMO and GradNorm must fail closed or mark expected fail "
                "due signed loss; methods not using losses may skip this "
                "loss-specific case."
            ),
            explanatory_purpose="Tests loss/stateful invalid-input behavior.",
        ),
        GradientAlgebraCase(
            case_id="near_duplicate_k4_dim3",
            case_family="near duplicate",
            gradients=_rows(((1.0, 0.0, 0.0), (1.000001, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            losses=(1.0, 1.0, 1.0, 1.0),
            expected_gate_behavior=(
                "Finite or explicit fail-closed; diagnostics preserve "
                "near-duplicate/rank information."
            ),
            explanatory_purpose="Tests numerical near-duplication.",
        ),
    )
