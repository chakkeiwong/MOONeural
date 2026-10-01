"""Public API for TensorFlow-native multiobjective aggregation.

This module exposes the runtime aggregation surface of the reusable library.
Runtime exposure is not by itself equivalent to reviewed-canonical promotion;
status should be interpreted through the reusable-library status-audit artifacts.
"""

from __future__ import annotations

import tensorflow as tf

from mooneural.multiobjective.diagnostics import common_diagnostics
from mooneural.multiobjective.aligned import aligned_flat
from mooneural.multiobjective.cagrad import cagrad_flat
from mooneural.multiobjective.famo import (
    create_famo_state,
    famo_after_step,
    famo_pre_step,
)
from mooneural.multiobjective.flattening import (
    flatten_objective_gradients,
    unflatten_like_variables,
)
from mooneural.multiobjective.gradnorm import (
    create_gradnorm_state,
    gradnorm_flat,
    gradnorm_update,
)
from mooneural.multiobjective.imtl import imtl_g_flat
from mooneural.multiobjective.mgda import mgda_flat
from mooneural.multiobjective.pcgrad import pcgrad_flat
from mooneural.multiobjective.types import AggregationResult
from mooneural.multiobjective.unsupported import (
    BLOCKED_METHODS,
    IMPLEMENTED_METHODS,
    SUPPORTED_METHODS,
    UnsupportedMethodError,
    ensure_method_supported,
)


def aggregate(
    method,
    objective_gradients,
    reference_variables,
    *,
    losses=None,
    state=None,
    seed=None,
    eps=1e-12,
    max_objectives=8,
    pcgrad_reduction="sum",
    cagrad_c=0.5,
    cagrad_rescale=1,
    cagrad_stationarity_tol=1e-6,
    cagrad_allow_outside_fixed_step_theorem=False,
    aligned_scale_mode="min",
    gradnorm_alpha=1.5,
    famo_beta=0.01,
    famo_weight_decay=0.001,
):
    """Aggregate per-objective TensorFlow gradients.

    `objective_gradients` is a sequence of gradient structures, one per
    objective.  `reference_variables` supplies shapes and dtypes for
    unflattening and for replacing `None` gradients by zeros.

    Evidence and use-status notes:
    - The runtime-exposed methods are bounded, tested implementations with
      analytic MOO pilot benchmark artifacts.  They are suitable for informed
      trial by users who understand the method assumptions and diagnostics.
      They are not default algorithms, universal recommendations, or
      downstream DSGE/HMC/scientific-readiness claims.  Multiobjective
      algorithms and update optimizers remain problem-dependent and should be
      chosen with task-specific validation.

    Method-specific contract notes:
    - `mgda` and `cagrad` are bounded small-K methods controlled by
      `max_objectives`; exceeding that bound fails closed.
    - `cagrad` is reviewed canonical for the bounded small-K active-set/KKT
      contract with `0 <= c < 1`; exact official default-output parity,
      official PyTorch runtime parity, Appendix A.5 support, and downstream
      DSGE/HMC readiness remain non-claims.  `cagrad_c >= 1` fails closed by
      default because it is outside the paper's fixed-step average-loss theorem
      scope.  Callers may set
      `cagrad_allow_outside_fixed_step_theorem=True` for a bounded one-step
      runtime computation, but that opt-in does not implement or validate
      Appendix Theorem A.5's time-varying step-size result.
    - `famo` requires nonnegative losses and explicit state; the pre-step
      aggregation exposed here must be paired with `famo_after_step` on the same
      batch, and reusing the pending pre-step state across a different batch is
      outside the supported contract.
    - `gradnorm` is a bounded usable paper-audited contract over caller-supplied
      selected shared gradients.  It requires nonnegative losses, explicit
      state, and a separate `gradnorm_update`; the API does not infer which
      model parameters count as the paper's selected shared weights.
    - `imtl` implements the IMTL-G direct closed-form linear system.  It
      requires nonzero task gradients and a full-rank, well-conditioned IMTL-G
      matrix; exact duplicate, collinear, or rank-deficient gradients are
      outside this contract and fail closed rather than receiving an implicit
      pseudoinverse or regularized fallback.
    - For `gradnorm`, the returned coefficients are the current task weights used
      for the reported aggregation step.
    - `gradnorm` runtime exposure does not imply official-code parity,
      benchmark reproduction, downstream readiness, or default-policy status.
    - `aligned` is a bounded TensorFlow-native Procrustes/SVD operator over
      caller-supplied full shared gradients.  It does not implement
      Aligned-MTL-UB, official training-loop parity, benchmark reproduction, or
      downstream/default-policy readiness.
    - Methods listed in `BLOCKED_METHODS` remain unavailable through this API.

    The returned `coefficients` field is method-specific diagnostic state: it may
    represent simplex weights, update coefficients, or task weights depending on
    the selected method.  Callers should inspect method-specific diagnostics
    rather than assuming simplex semantics.
    """
    normalized = ensure_method_supported(method)
    flat = flatten_objective_gradients(objective_gradients, reference_variables)
    if normalized == "mgda":
        coefficients, combined_flat, method_diag = mgda_flat(
            flat, eps=eps, max_objectives=max_objectives)
        out_state = ()
    elif normalized == "pcgrad":
        if seed is None:
            raise ValueError("PCGrad requires a TensorFlow stateless seed")
        coefficients, combined_flat, method_diag = pcgrad_flat(
            flat, seed=seed, reduction=pcgrad_reduction, eps=eps)
        out_state = ()
    elif normalized == "imtl":
        coefficients, combined_flat, method_diag = imtl_g_flat(flat, eps=eps)
        out_state = ()
    elif normalized == "famo":
        coefficients, combined_flat, out_state, method_diag = famo_pre_step(
            flat, losses, state, eps=eps)
        method_diag = dict(method_diag)
        method_diag["famo_beta"] = tf.cast(famo_beta, flat.dtype)
        method_diag["famo_weight_decay"] = tf.cast(famo_weight_decay, flat.dtype)
    elif normalized == "cagrad":
        cagrad_c_value = float(cagrad_c)
        outside_fixed_step_theorem = cagrad_c_value >= 1.0
        if outside_fixed_step_theorem and not cagrad_allow_outside_fixed_step_theorem:
            raise ValueError(
                "cagrad_c >= 1 is outside the CAGrad fixed-step average-loss "
                "theorem scope 0 <= c < 1. Set "
                "cagrad_allow_outside_fixed_step_theorem=True only for an "
                "explicit bounded runtime/non-theorem opt-in.")
        coefficients, combined_flat, method_diag = cagrad_flat(
            flat,
            c=cagrad_c,
            rescale=cagrad_rescale,
            eps=eps,
            max_objectives=max_objectives,
            stationarity_tol=cagrad_stationarity_tol,
            allow_outside_fixed_step_theorem=(
                cagrad_allow_outside_fixed_step_theorem),
        )
        method_diag = dict(method_diag)
        method_diag[
            "cagrad_fixed_step_average_loss_theorem_supported"
        ] = tf.constant(not outside_fixed_step_theorem)
        method_diag[
            "cagrad_outside_fixed_step_theorem_opt_in"
        ] = tf.constant(
            outside_fixed_step_theorem
            and bool(cagrad_allow_outside_fixed_step_theorem)
        )
        out_state = ()
    elif normalized == "aligned":
        coefficients, combined_flat, method_diag = aligned_flat(
            flat,
            scale_mode=aligned_scale_mode,
            eps=eps,
        )
        out_state = ()
    elif normalized == "gradnorm":
        coefficients, combined_flat, out_state, method_diag = gradnorm_flat(
            flat,
            losses,
            state,
            alpha=gradnorm_alpha,
            eps=eps,
        )
    else:
        raise UnsupportedMethodError(f"unreachable method state: {method}")

    diagnostics = common_diagnostics(
        normalized, flat, coefficients, combined_flat, eps)
    diagnostics.update(method_diag)
    return AggregationResult(
        combined_gradients=unflatten_like_variables(
            combined_flat, reference_variables),
        flat_gradient=combined_flat,
        coefficients=coefficients,
        diagnostics=diagnostics,
        state=out_state,
    )
