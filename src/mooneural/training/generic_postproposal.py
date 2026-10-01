"""Optional host-owned finite screening after one unchanged optimizer proposal."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from itertools import pairwise

import numpy as np
import tensorflow as tf

from .generic_component_descent import (
    component_descent_binding,
    make_component_descent_function,
)
from .generic_curvature_progress import curvature_progress
from .generic_displacement_blend import blend_displacements
from .generic_execution_boundary import PostproposalRejected
from .generic_relative_progress import make_relative_progress_function
from .generic_training_contracts import (
    ImmutableJSONMapping,
    canonical_json,
    stable_hash,
)


class NonfinitePostproposalLoss(ValueError):
    """Explicit numerical candidate failure; a shorter fraction may be tried."""


def relative_progress_binding():
    return {"schema": "generic_neural_solver.relative_public_progress.v1",
        "solver": "float64-feasible-seed-primal-active-set", "optimality_claim": False,
        "row_normalization": "stable-unit-norm", "target_ratio": "positive-loss-over-row-maximum-over-scaled-norm",
        "public_target": "negative-relative-public-loss", "component_target": "owner-relative-at-max-or-tie-otherwise-zero",
        "protected_target": "nonpositive", "seed": "checked-original-unit-row-equality-witness",
        "rank_cutoff": "float64-epsilon-times-max-shape-times-largest-singular-value",
        "residual_check": "original-full-rows-radius-primal-dual",
        "relative_tolerance": 1e-10, "tie_rtol": 1e-11, "max_iterations": 200,
        "scale": "source-final-displacement-norm", "component_direction": "relative_public_progress"}


def coupled_component_binding():
    return {"coupled_finite_guard": "required-cells-at-or-above-owner-T",
        "required_component_selection": "raw>=T-frozen-at-parent",
        "component_target": "owner-relative-at-max-or-tie-otherwise-required-own-relative",
        "component_loss_request_context": "parent-candidate-batch-update-D-fixed-cells-v1",
        "finite_screen": "original-active-protected-dots-and-required-cell-decrease"}


def finite_target_binding():
    return {"finite_target_refinement": "same-parent-required-cell-remainder-v1",
        "target_weight_rule": "first-failed-required-observation-max-remainder-over-negative-slope-floor-plus-one",
        "target_weight_ties": "first-supplied-component", "target_slope_check": "float64-dot-error-bound",
        "target_trial_order": "refined-grid-before-remaining-original-fractions",
        "maximum_direction_solves": 2, "target_gradient_reuse": True}


class GuardedPostproposal:
    """Blend and finitely screen a source proposal on exactly bound batch inputs.

    The loss callback takes (parameters, batch, update_index) and returns raw
    and raw/D vectors. ``batch_binding`` independently checks the batch identity
    declared by the caller and returns its stable-hash-compatible metadata.
    The parent loss call must agree with the gradient receipt. On success the
    source moments and clock are preserved, with truthful displacement/dots.
    No callback is invoked during construction or completed-state recovery.

    Optional ``component_rows(parameters, batch, update_index, context)`` is
    called once for an invalid original segment. An explicit optional finite
    trigger also tries it after a valid segment's first finite refusal. Its mapping contains
    owner_indices, cell_ids, raw_values, normalized_values, normalized_gradients
    and context. Values permit a raw/D parity check; gradient coordinates are
    the trusted provider's declaration, qualified independently by its tests.
    The optional relative-public-progress direction calls that provider on
    every update and bypasses the original displacement blend.
    """

    def __init__(self, losses, batch_binding, denominators, limits, *, binding,
                 margin_fraction=.1, fractions=(1., .5, .25, .125), relative_tolerance=1e-12,
                 component_rows=None, component_binding=None, finite_component_fallback=False,
                 extra_descent_indices=(), component_direction="equality", component_loss_values=None,
                 component_loss_binding=None, coupled_component_guard=False, finite_target_refinement=False,
                 lookahead_steps=0, residual_provider=None, residual_binding=None,
                 curvature_relative_radius=None):
        if not callable(losses) or not callable(batch_binding):
            raise TypeError("explicit loss-only and batch-binding callbacks required")
        if not isinstance(binding, Mapping) or not binding:
            raise ValueError("nonempty source/criterion/input binding required")
        canonical_json(binding)
        self.denominators = np.array(denominators, dtype=np.float64, copy=True)
        self.limits = np.array(limits, dtype=np.float64, copy=True)
        if (self.denominators.ndim != 1 or not self.denominators.size or self.denominators.shape != self.limits.shape
                or not np.isfinite(self.denominators).all() or not np.isfinite(self.limits).all()
                or np.any(self.denominators <= 0) or np.any(self.limits <= 0)):
            raise ValueError("positive finite ordered D and T required")
        if (not fractions or fractions[0] != 1. or any(type(value) is not float or not 0 < value <= 1 for value in fractions)
                or any(first <= second for first, second in pairwise(fractions))):
            raise ValueError("strictly decreasing positive finite fractions starting at one required")
        if not 0 <= margin_fraction <= 1 or not 0 <= relative_tolerance < 1:
            raise ValueError("valid explicit linear margin and relative tolerance required")
        self.denominators.setflags(write=False)
        self.limits.setflags(write=False)
        self.losses, self.batch_binding = losses, batch_binding
        self._callbacks = losses, batch_binding
        self.fractions = tuple(fractions)
        self.binding = {"schema": "generic_neural_solver.postproposal.v1", "profile": json.loads(canonical_json(binding)),
            "D": self.denominators.tolist(), "T": self.limits.tolist(), "margin_fraction": float(margin_fraction),
            "fractions": list(fractions), "relative_tolerance": float(relative_tolerance),
            "roundoff_profile": "half-interval-half-evaluation-stable-norm-v1",
            "reference": "first-protected-descent-at-source-final-displacement-norm",
            "moments": "source-proposed-slots-and-single-clock-increment", "failure": "typed-rejection-no-commit",
            "maximum_loss_calls": 1 + len(fractions), "protected_dot_absolute_tolerance": 1e-10,
            "loss_parity_rtol": 1e-11, "loss_parity_atol": 1e-18}
        self.binding["input_identity"] = "exact-gradient-batch-context-v1"
        self.binding["numerical_failure"] = "count-call-reject-fraction-continue"
        self.binding["contract_failure"] = "stop-with-receipt-complete-rollback"
        if type(coupled_component_guard) is not bool:
            raise TypeError("coupled_component_guard must be Boolean")
        if (component_loss_values is None) != (component_loss_binding is None):
            raise ValueError("component loss callback and explicit source/profile binding must be supplied together")
        if coupled_component_guard and (component_rows is None or component_loss_values is None):
            raise ValueError("coupled component guard requires gradient and loss-only component callbacks")
        if coupled_component_guard and component_direction != "relative_public_progress":
            raise ValueError("coupled component guard requires relative public progress")
        if not coupled_component_guard and component_loss_values is not None:
            raise ValueError("component loss callback requires explicit coupled component guard")
        self.coupled_component_guard = coupled_component_guard
        self._coupled_component_guard = coupled_component_guard
        if type(finite_target_refinement) is not bool:
            raise TypeError("finite_target_refinement must be Boolean")
        if finite_target_refinement and not coupled_component_guard:
            raise ValueError("finite target refinement requires the coupled component guard")
        self.finite_target_refinement = finite_target_refinement
        self._finite_target_refinement = finite_target_refinement
        self.target_refinement_solver = None
        self.component_loss_values = component_loss_values
        self._component_loss_callback = component_loss_values
        self.component_loss_binding = (None if component_loss_binding is None
                                       else ImmutableJSONMapping(component_loss_binding))
        if (component_rows is None) != (component_binding is None):
            raise ValueError("component callback and explicit source/profile binding must be supplied together")
        if type(finite_component_fallback) is not bool or finite_component_fallback and component_rows is None:
            raise ValueError("finite component fallback requires a boolean option and enabled component provider")
        self.finite_component_fallback = finite_component_fallback
        self._finite_component_fallback = finite_component_fallback
        if (not isinstance(extra_descent_indices, (list, tuple))
                or any(type(index) is not int or not 0 <= index < len(self.denominators)
                       for index in extra_descent_indices)
                or len(set(extra_descent_indices)) != len(extra_descent_indices)
                or extra_descent_indices and component_rows is None):
            raise ValueError("unique extra descent task indices and an enabled component provider required")
        self.extra_descent_indices = tuple(sorted(extra_descent_indices))
        self._extra_descent_indices = self.extra_descent_indices
        if (component_direction not in ("equality", "relative_public_progress")
                or component_direction != "equality" and component_rows is None):
            raise ValueError("declared component direction and an enabled provider required")
        self.component_direction = component_direction
        self._component_direction_profile = component_direction
        self.component_rows = component_rows
        self._component_callback = component_rows
        self.component_descent = None
        if component_rows is not None:
            if not callable(component_rows) or not isinstance(component_binding, Mapping) or not component_binding:
                raise ValueError("callable component provider and nonempty source/profile binding required")
            implementation = getattr(component_rows, "python_function", component_rows)
            solver_binding = (relative_progress_binding() if component_direction == "relative_public_progress"
                else component_descent_binding(relative_tolerance))
            self.binding["component_fallback"] = {**solver_binding,
                "provider": json.loads(canonical_json(component_binding)),
                "callback": {"module": getattr(implementation, "__module__", type(implementation).__module__),
                             "qualname": getattr(implementation, "__qualname__", type(implementation).__qualname__)},
                "trigger": "invalid-original-linear-segment-only", "maximum_provider_calls": 1,
                "coordinate_mode": "raw_over_D", "aggregation": "maximum-components",
                "request_context": "parent-batch-update-D-active-indices-v1",
                "finite_screen": "original-fractions-on-witness-no-extra-loss-budget"}
            if finite_component_fallback:
                self.binding["component_fallback"].update(
                    trigger="invalid-original-linear-segment-or-first-finite-rejection",
                    finite_screen="component-fractions-before-original-shorter-fractions",
                    finite_trigger_radius="original-valid-segment-norm")
                self.binding["maximum_loss_calls"] = 1 + 2 * len(fractions)
            if self.extra_descent_indices:
                self.binding["component_fallback"].pop("finite_trigger_radius", None)
                self.binding["component_fallback"].update(
                    trigger="every-update-with-extra-descent-preference",
                    request_context="parent-batch-update-D-active-and-extra-descent-indices-v1",
                    extra_descent_indices=list(self.extra_descent_indices),
                    extra_descent_radius="source-final-displacement-norm",
                    finite_screen="original-fractions-active-and-extra-public-decrease")
                self.binding["maximum_loss_calls"] = 1 + len(fractions)
            if component_direction == "relative_public_progress":
                self.binding["component_fallback"].pop("finite_trigger_radius", None)
                self.binding["component_fallback"].update(
                    trigger="every-update-relative-public-progress",
                    finite_screen="original-fractions-active-and-extra-public-decrease",
                    direction_radius="source-final-displacement-norm")
                self.binding["maximum_loss_calls"] = 1 + len(fractions)
                self.component_descent = make_relative_progress_function(len(self.denominators),
                    explicit_component_targets=coupled_component_guard)
            else:
                self.component_descent = make_component_descent_function(len(self.denominators),
                                                                        relative_tolerance=relative_tolerance)
            if coupled_component_guard:
                if not callable(component_loss_values) or not isinstance(component_loss_binding, Mapping) or not component_loss_binding:
                    raise ValueError("coupled component guard requires callable loss-only source binding")
                loss_implementation = getattr(component_loss_values, "python_function", component_loss_values)
                self.binding["component_fallback"].update(
                    **coupled_component_binding(),
                    component_loss_callback={
                        "module": getattr(loss_implementation, "__module__", type(loss_implementation).__module__),
                        "qualname": getattr(loss_implementation, "__qualname__", type(loss_implementation).__qualname__)},
                    component_loss_provider=json.loads(canonical_json(component_loss_binding)),
                    maximum_component_loss_calls=len(fractions))
                if finite_target_refinement:
                    self.binding["component_fallback"].update(**finite_target_binding(),
                        maximum_component_loss_calls=2 * len(fractions))
                    self.binding["maximum_loss_calls"] = 1 + 2 * len(fractions)
                    self.target_refinement_solver = make_relative_progress_function(len(self.denominators),
                        explicit_component_targets=True, explicit_component_weights=True)
        self._component_descent = self.component_descent
        self._target_refinement_solver = self.target_refinement_solver
        if type(lookahead_steps) is not int or not 0 <= lookahead_steps <= 2:
            raise ValueError("lookahead_steps must be zero, one or two")
        if lookahead_steps and (component_direction != "relative_public_progress"
                or finite_component_fallback or finite_target_refinement):
            raise ValueError("lookahead requires a fixed relative-public-progress direction")
        self.lookahead_steps = lookahead_steps
        self._lookahead_steps = lookahead_steps
        if lookahead_steps:
            self.binding["finite_selection"] = {"profile": "bounded-feasible-suffix-v1",
                "lookahead_steps": lookahead_steps, "criterion": "minimum-active-relative-decrease",
                "minimum_gain": 1e-8, "ties": "earlier-candidate", "fractions": "existing-list-only"}
        if residual_provider is None:
            if residual_binding is not None or curvature_relative_radius is not None:
                raise ValueError("curvature binding and radius require an enabled residual provider")
        else:
            if (not callable(residual_provider) or not isinstance(residual_binding, Mapping) or not residual_binding
                    or not isinstance(curvature_relative_radius, (int, float))
                    or isinstance(curvature_relative_radius, bool) or not np.isfinite(curvature_relative_radius)
                    or curvature_relative_radius <= 0):
                raise ValueError("callable residual provider, explicit binding and positive relative radius required")
            if (component_rows is not None or component_loss_values is not None or finite_component_fallback
                    or self.extra_descent_indices or component_direction != "equality"
                    or coupled_component_guard or finite_target_refinement or lookahead_steps):
                raise ValueError("curvature direction is exclusive with component, refinement and lookahead modes")
            self.binding["curvature_direction"] = {
                "schema": "generic_neural_solver.residual_curvature_postproposal.v1",
                "provider": json.loads(canonical_json(residual_binding)),
                "relative_weight_radius": float(curvature_relative_radius),
                "radius": "relative-times-max-one-parent-weight-norm",
                "request_context": "parent-batch-update-D-all-public-profile-v1",
                "residual_coordinates": "signed-residuals-squared-norm-equals-raw-over-D",
                "gradient_parity": "2-weighted-residual-response-equals-gradient-times-basis",
                "gradient_parity_rtol": 1e-9, "gradient_roundoff_factor": 256.,
                "solver": "shared-curvature-progress", "maximum_provider_calls": 1,
                "maximum_solver_calls": 1, "finite_screen": "all-public-strict-decrease-original-fractions"}
        self.residual_provider = residual_provider
        self._residual_provider = residual_provider
        self.binding_hash = stable_hash(self.binding)
        self.blend = tf.function(lambda proposal, reference, rows, active: blend_displacements(
            proposal, reference, rows, active, margin_fraction=float(margin_fraction),
            relative_tolerance=float(relative_tolerance)), autograph=False, input_signature=[
                tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.float64),
                tf.TensorSpec([len(self.denominators), None], tf.float64),
                tf.TensorSpec([len(self.denominators)], tf.bool)])

    def validate_binding(self):
        if (stable_hash(self.binding) != self.binding_hash
                or (self.losses, self.batch_binding) != self._callbacks
                or self.component_rows is not self._component_callback
                or self.component_descent is not self._component_descent
                or self.component_loss_values is not self._component_loss_callback
                or self.finite_component_fallback != self._finite_component_fallback
                or self.extra_descent_indices != self._extra_descent_indices
                or self.component_direction != self._component_direction_profile
                or self.coupled_component_guard != self._coupled_component_guard
                or self.finite_target_refinement != self._finite_target_refinement
                or self.target_refinement_solver is not self._target_refinement_solver
                or self.lookahead_steps != self._lookahead_steps
                or self.residual_provider is not self._residual_provider
                or self.denominators.tolist() != self.binding["D"]
                or self.limits.tolist() != self.binding["T"]
                or list(self.fractions) != self.binding["fractions"]):
            raise ValueError("postproposal profile or callbacks mutated after construction")

    def _curvature_direction(self, parameters, batch, update_index, normalized, gradients, receipt):
        profile = self.binding["curvature_direction"]
        context = ImmutableJSONMapping({"parameters_hash": stable_hash(parameters.tolist()),
            "batch_hash": receipt["batch_hash"], "update_index": update_index,
            "coordinate_mode": "raw_over_D", "D": self.denominators.tolist(),
            "descent_indices": list(range(len(self.denominators))), "profile_hash": stable_hash(profile)})
        details = {"provider_calls": 1, "solver_calls": 0, "context": json.loads(canonical_json(context))}
        receipt.update(curvature=details, direction_kind="residual-curvature-common-descent", blend_fraction=None)
        supplied = np.array(parameters, copy=True)
        supplied.setflags(write=False)
        payload = self.residual_provider(supplied, batch, update_index, context)
        self.validate_binding()
        if (not np.array_equal(supplied, parameters)
                or stable_hash(self.batch_binding(batch, update_index)) != receipt["batch_hash"]):
            raise ValueError("residual provider changed parent parameters or batch identity")
        if not isinstance(payload, Mapping) or set(payload) != {"residuals", "responses", "probabilities", "basis", "context"}:
            raise ValueError("residual provider must return residuals, responses, probabilities, basis and context")
        if canonical_json(payload["context"]) != canonical_json(context):
            raise ValueError("residual response has stale parent/batch/update/coordinate/profile binding")
        residuals, responses = payload["residuals"], payload["responses"]
        if (not isinstance(residuals, (list, tuple)) or not isinstance(responses, (list, tuple))
                or len(residuals) != len(self.denominators) or len(responses) != len(residuals)):
            raise ValueError("ordered residual and response arrays for every public task required")

        def array(value, rank):
            result = np.array(value, copy=True)
            if result.dtype != np.float64 or result.ndim != rank or not all(result.shape) or not np.isfinite(result).all():
                raise ValueError("nonempty finite float64 residual-provider arrays required")
            return result

        probabilities, basis = array(payload["probabilities"], 1), array(payload["basis"], 2)
        if (basis.shape[0] != len(parameters) or basis.shape[1] > len(parameters)
                or np.any(probabilities <= 0) or abs(probabilities.sum() - 1.) > 64 * np.finfo(np.float64).eps
                or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), rtol=3e-10, atol=1e-10)):
            raise ValueError("positive normalized measure and orthonormal packed-weight basis required")
        residuals = [array(value, 2) for value in residuals]
        responses = [array(value, 3) for value in responses]
        values, projected, magnitudes = [], [], []
        for residual, response in zip(residuals, responses, strict=True):
            if (response.shape != (*residual.shape, basis.shape[1]) or len(residual) != len(probabilities)):
                raise ValueError("residual/response/outer-row dimensions differ")
            values.append(np.einsum("b,bc,bc->", probabilities, residual, residual))
            projected.append(2 * np.einsum("b,bc,bck->k", probabilities, residual, response))
            magnitudes.append(2 * np.einsum("b,bc,bck->k", probabilities, np.abs(residual), np.abs(response)))
        values, projected = np.asarray(values), np.asarray(projected)
        expected = gradients @ basis
        tolerance = (profile["gradient_parity_rtol"] * np.maximum(np.abs(projected), np.abs(expected))
            + profile["gradient_roundoff_factor"] * np.finfo(np.float64).eps
            * (np.asarray(magnitudes) + np.abs(gradients) @ np.abs(basis)))
        if not np.isfinite(values).all() or not np.allclose(values, normalized, rtol=1e-11, atol=1e-18):
            raise ValueError("residual squared norms differ from current raw/D losses")
        if (not np.isfinite(projected).all() or not np.isfinite(expected).all() or not np.isfinite(tolerance).all()
                or np.any(np.abs(projected - expected) > tolerance)):
            raise ValueError("residual responses differ from current projected gradients")

        def reference(value):
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}

        details.update(residual_norms=values.tolist(), projected_gradients=projected.tolist(),
            expected_projected_gradients=expected.tolist(), gradient_tolerances=tolerance.tolist(),
            arrays={"basis": reference(basis), "probabilities": reference(probabilities),
                "residuals": [reference(value) for value in residuals], "responses": [reference(value) for value in responses]})
        radius = profile["relative_weight_radius"] * max(1., np.linalg.norm(parameters))
        details["solver_calls"] += 1
        result = curvature_progress(residuals, responses, probabilities, basis, radius)
        direction = np.asarray(result["direction"])
        if (direction.dtype != np.float64 or direction.shape != parameters.shape
                or not np.isfinite(direction).all() or not np.all(gradients @ direction < 0)):
            raise ValueError("curvature proposal must be finite common descent in original gradient coordinates")
        details["proposal"] = {key: ({name: np.asarray(value).tolist() for name, value in item.items()}
            if key == "solver" else np.asarray(item).tolist()) for key, item in result.items()}
        return direction

    def _component_direction(self, parameters, batch, update_index, normalized, gradients, mask, proposal, receipt,
                             *, original_segment_valid=False):
        details = {"invoked": True, "provider_calls": 0, "witness_calls": 0, "valid": False}
        receipt["component_fallback"] = details
        receipt["original_segment_valid"] = original_segment_valid
        receipt["direction_kind"] = "supplied-component-equality-witness"
        if self.component_direction == "relative_public_progress":
            receipt["direction_kind"] = "supplied-component-relative-public-progress"
            receipt["component_trigger"] = "relative-public-progress-preference"
            details["radius_source"] = "source-final-displacement-norm"
        elif self.extra_descent_indices:
            receipt["component_trigger"] = "extra-descent-preference"
            details["radius_source"] = "source-final-displacement-norm"
        elif original_segment_valid:
            receipt["component_trigger"] = "first-finite-rejection"
            details["radius_source"] = "original-valid-segment-norm"
        context_values = {"parameters_hash": stable_hash(parameters.tolist()),
            "batch_hash": receipt["batch_hash"], "update_index": update_index,
            "coordinate_mode": "raw_over_D", "D": self.denominators.tolist(),
            "active_indices": np.flatnonzero(mask).tolist(),
            "profile_hash": stable_hash(self.binding["component_fallback"])}
        descent_mask = mask.copy()
        if self.extra_descent_indices:
            context_values["extra_descent_indices"] = list(self.extra_descent_indices)
            descent_mask[list(self.extra_descent_indices)] = True
        context = ImmutableJSONMapping(context_values)
        supplied = np.array(parameters, dtype=np.float64, copy=True)
        supplied.setflags(write=False)
        details["provider_calls"] += 1
        payload = self.component_rows(supplied, batch, update_index, context)
        self.validate_binding()
        if (not np.array_equal(supplied, parameters)
                or stable_hash(self.batch_binding(batch, update_index)) != receipt["batch_hash"]):
            raise ValueError("component provider changed parent parameters or batch identity")
        required = {"owner_indices", "cell_ids", "raw_values", "normalized_values", "normalized_gradients", "context"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("component provider must return declared rows, values and context")
        if canonical_json(payload["context"]) != canonical_json(context):
            raise ValueError("component rows have stale parent/batch/update/coordinate/profile binding")
        owners = np.array(payload["owner_indices"], copy=True)
        cells = payload["cell_ids"]
        raw_values, values, rows = (np.array(payload[name], copy=True) for name in (
            "raw_values", "normalized_values", "normalized_gradients"))
        if (owners.ndim != 1 or owners.dtype.kind not in "iu" or not isinstance(cells, (tuple, list))
                or len(cells) != owners.size or any(not isinstance(cell, str) or not cell for cell in cells)
                or np.any(owners >= len(mask)) or np.any(owners < 0)):
            raise ValueError("component owners and unique cell identities must match public tasks")
        if len(set(zip(owners.tolist(), cells, strict=True))) != owners.size or not np.all(descent_mask[owners]):
            raise ValueError("component identities must be unique and owned by currently active tasks")
        if (any(value.dtype != np.float64 or value.shape != owners.shape for value in (raw_values, values))
                or rows.dtype != np.float64 or rows.shape != (owners.size, parameters.size)
                or not all(np.isfinite(value).all() for value in (raw_values, values, rows))
                or np.any(raw_values < 0) or np.any(values < 0)):
            raise ValueError("finite float64 component values and parameter gradient rows required")
        if not np.allclose(values, raw_values / self.denominators[owners], rtol=1e-11, atol=1e-18):
            raise ValueError("component values must be raw/D exactly once")
        public_values = normalized[owners]
        if np.any((values > public_values) & ~np.isclose(values, public_values, rtol=1e-11, atol=1e-18)):
            raise ValueError("component value exceeds its owner's public maximum")
        details.update({"context": json.loads(canonical_json(context)), "row_count": int(owners.size),
            "owner_indices": owners.tolist(), "cell_ids": list(cells), "raw_values": raw_values.tolist(),
            "normalized_values": values.tolist(), "normalized_gradients": rows.tolist(),
            "normalized_gaps": np.maximum(public_values - values, 0.).tolist()})
        if self.coupled_component_guard:
            required_components = raw_values >= self.limits[owners]
            details["required_components"] = required_components.tolist()
            receipt["component_guard"] = {
                "context": json.loads(canonical_json(context)),
                "owner_indices": owners[required_components].tolist(),
                "cell_ids": [cell for cell, selected in zip(cells, required_components, strict=True) if selected],
                "parent_raw_values": raw_values[required_components].tolist(),
                "loss_calls": 0}
        details["witness_calls"] += 1
        if self.component_direction == "relative_public_progress":
            arguments = (normalized, gradients, descent_mask, values, rows, owners.astype(np.int32), proposal)
            result = (self.component_descent(*arguments, required_components)
                      if self.coupled_component_guard else self.component_descent(*arguments))
        else:
            result = self.component_descent(gradients, descent_mask, rows, proposal)
        details["valid"] = bool(result["valid"])
        for name, value in result.items():
            array = np.asarray(value)
            details[name] = array.tolist() if np.isfinite(array).all() else None
        if not details["valid"]:
            if self.component_direction == "relative_public_progress":
                raise PostproposalRejected("invalid supplied-component relative public progress", receipt)
            if original_segment_valid and not self.extra_descent_indices:
                return None
            raise PostproposalRejected("invalid supplied-component equality witness", receipt)
        return np.asarray(result["direction"])

    def _refine_targets(self, normalized, gradients, mask, proposal, receipt, observation):
        details = receipt["component_fallback"]
        selected = np.asarray(details["required_components"], dtype=bool)
        owners = np.asarray(details["owner_indices"], dtype=np.int32)
        rows = np.asarray(details["normalized_gradients"], dtype=np.float64)
        values = np.asarray(details["normalized_values"], dtype=np.float64)
        displacement = np.asarray(observation["rounded_displacement"], dtype=np.float64)
        parent_values = values[selected]
        observed_values = np.asarray(observation["component"]["raw_values"]) / self.denominators[owners[selected]]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            products = rows[selected] * displacement
            slopes = np.sum(products, axis=1)
            rounding = np.finfo(np.float64).eps * displacement.size
            dot_error = rounding / (1. - rounding) * np.sum(np.abs(products), axis=1)
            remainders = observed_values - parent_values - slopes
            ratios = remainders / -slopes
        finite = np.isfinite(slopes) & np.isfinite(dot_error) & np.isfinite(remainders) & np.isfinite(ratios)
        failed = observed_values >= parent_values
        eligible = finite & failed & (slopes < -dot_error) & (remainders > 0.) & (ratios >= 1.)
        refinement = {"attempted": True, "witness_calls": 0, "valid": False,
            "observation_index": len(receipt["candidates"]) - 1,
            "observation_fraction": observation["fraction"],
            "rounded_displacement": displacement.tolist(),
            "eligible_required_rows": eligible.tolist()}
        receipt["target_refinement"] = refinement
        for name, array in (("slopes", slopes), ("dot_error_bounds", dot_error),
                            ("remainders", remainders), ("ratios", ratios)):
            refinement[name] = [float(value) if np.isfinite(value) else None for value in array]
        if rounding >= 1. or not np.any(eligible):
            refinement["reason"] = "no reliable finite failed-cell remainder"
            return None
        required_index = int(np.argmax(np.where(eligible, ratios, -np.inf)))
        weight = float(np.floor(ratios[required_index]) + 1.)
        if not np.isfinite(weight) or weight <= max(1., ratios[required_index]):
            refinement["reason"] = "target increment is not representable"
            return None
        component_index = int(np.flatnonzero(selected)[required_index])
        weights = np.ones(values.shape, dtype=np.float64)
        weights[component_index] = weight
        refinement.update(component_index=component_index, owner_index=int(owners[component_index]),
            cell_id=details["cell_ids"][component_index], component_target_weights=weights.tolist())
        descent_mask = mask.copy()
        descent_mask[list(self.extra_descent_indices)] = True
        refinement["witness_calls"] += 1
        details["witness_calls"] += 1
        result = self.target_refinement_solver(normalized, gradients, descent_mask, values, rows, owners,
            proposal, selected, weights)
        self.validate_binding()
        for name, value in result.items():
            array = np.asarray(value)
            refinement[name] = array.tolist() if np.isfinite(array).all() else None
        if not bool(result["valid"]):
            refinement["reason"] = "refined relative-progress direction refused"
            return None
        return np.asarray(result["direction"])

    def _component_losses(self, parameters, batch, update_index, receipt):
        if not self.coupled_component_guard:
            return True, None
        guard = receipt.get("component_guard")
        if not isinstance(guard, Mapping):
            raise TypeError("coupled component guard has no frozen parent component set")
        if not guard["owner_indices"]:
            return True, {"raw_values": [], "raw_over_T": [], "strict_decrease": True}
        context = ImmutableJSONMapping({**guard["context"],
            "candidate_parameters_hash": stable_hash(parameters.tolist()),
            "owner_indices": guard["owner_indices"], "cell_ids": guard["cell_ids"]})
        supplied = np.array(parameters, dtype=np.float64, copy=True)
        supplied.setflags(write=False)
        if self.finite_target_refinement and guard["loss_calls"] >= self.binding["component_fallback"]["maximum_component_loss_calls"]:
            raise PostproposalRejected("component loss call budget exhausted", receipt)
        guard["loss_calls"] += 1
        payload = self.component_loss_values(supplied, batch, update_index, context,
            np.asarray(guard["owner_indices"], dtype=np.int32), tuple(guard["cell_ids"]))
        self.validate_binding()
        if (not np.array_equal(supplied, parameters)
                or stable_hash(self.batch_binding(batch, update_index)) != receipt["batch_hash"]):
            raise ValueError("component loss callback changed supplied parameters or batch")
        required = {"owner_indices", "cell_ids", "raw_values", "normalized_values", "context"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("component loss callback must return fixed identities, values and context")
        owners = np.asarray(payload["owner_indices"])
        cells = payload["cell_ids"]
        raw_values = np.asarray(payload["raw_values"])
        normalized_values = np.asarray(payload["normalized_values"])
        expected_owners = np.asarray(guard["owner_indices"], dtype=np.int32)
        if (owners.dtype.kind not in "iu" or not np.array_equal(owners, expected_owners)
                or not isinstance(cells, (list, tuple))
                or tuple(cells) != tuple(guard["cell_ids"])
                or canonical_json(payload["context"]) != canonical_json(context)):
            raise ValueError("component loss identities or parent context changed")
        if (raw_values.dtype != np.float64 or normalized_values.dtype != np.float64
                or raw_values.shape != expected_owners.shape or normalized_values.shape != expected_owners.shape):
            raise ValueError("ordered float64 component loss vectors required")
        if not np.isfinite(raw_values).all() or not np.isfinite(normalized_values).all():
            raise NonfinitePostproposalLoss("nonfinite candidate component loss")
        if (np.any(raw_values < 0) or np.any(normalized_values < 0)
                or not np.allclose(normalized_values, raw_values / self.denominators[expected_owners],
                                   rtol=1e-11, atol=1e-18)):
            raise ValueError("component loss values must be nonnegative float64 raw/D values")
        parent_raw = np.asarray(guard["parent_raw_values"], dtype=np.float64)
        decreased = bool(np.all(raw_values < parent_raw))
        return decreased, {"raw_values": raw_values.tolist(),
            "raw_over_T": (raw_values / self.limits[expected_owners]).tolist(),
            "strict_decrease": decreased}

    def _losses(self, parameters, batch, update_index, expected_binding):
        self.validate_binding()
        if stable_hash(self.batch_binding(batch, update_index)) != expected_binding:
            raise ValueError("postproposal batch identity changed")
        supplied = np.array(parameters, dtype=np.float64, copy=True)
        supplied.setflags(write=False)
        values = self.losses(supplied, batch, update_index)
        self.validate_binding()
        if not np.array_equal(supplied, parameters):
            raise ValueError("loss callback changed supplied parameters")
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("loss-only callback must return raw and normalized task vectors")
        raw, normalized = (np.array(value, copy=True) for value in values)
        if any(value.dtype != np.float64 or value.shape != self.denominators.shape for value in (raw, normalized)):
            raise ValueError("ordered float64 loss vectors required")
        if not all(np.isfinite(value).all() for value in (raw, normalized)):
            raise NonfinitePostproposalLoss("nonfinite candidate loss")
        if np.any(raw < 0) or np.any(normalized < 0):
            raise ValueError("finite nonnegative float64 loss vectors required")
        if not np.allclose(normalized, raw / self.denominators, rtol=1e-11, atol=1e-18):
            raise ValueError("postproposal losses must be raw/D exactly once")
        if stable_hash(self.batch_binding(batch, update_index)) != expected_binding:
            raise ValueError("loss callback changed batch identity")
        return raw, normalized

    def __call__(self, **arguments):
        receipt = {"profile_hash": self.binding_hash, "loss_calls": 0, "candidates": [],
                   "update_index": arguments.get("update_index"), "accepted": False}
        if self._component_callback is not None:
            receipt["component_fallback"] = {"invoked": False, "provider_calls": 0, "witness_calls": 0}
        try:
            return self._apply(receipt=receipt, **arguments)
        except PostproposalRejected:
            raise
        except Exception as error:
            receipt["failure_kind"] = "contract_error"
            raise PostproposalRejected(f"{type(error).__name__}: {error}", receipt) from error

    def _apply(self, *, parameters, batch, update_index, raw_losses, normalized_losses,
               gradient_rows, optimizer, active_indices, constraint_indices, gradient_batch_binding, receipt):
        self.validate_binding()
        parameters = np.asarray(parameters)
        gradients = np.asarray(gradient_rows)
        raw = None if raw_losses is None else np.asarray(raw_losses)
        normalized = np.asarray(normalized_losses)
        active, protected = tuple(active_indices), tuple(constraint_indices)
        count = len(self.denominators)
        if (parameters.dtype != np.float64 or parameters.ndim != 1 or not parameters.size
                or gradients.dtype != np.float64 or gradients.shape != (count, len(parameters))
                or not active or sorted(active + protected) != list(range(count))):
            raise ValueError("ordered float64 parameters/gradients and complete task partition required")
        provided_losses = (normalized,) if raw is None else (raw, normalized)
        if any(value.dtype != np.float64 or value.shape != (count,) or not np.isfinite(value).all() or np.any(value < 0)
               for value in provided_losses) or not np.isfinite(parameters).all() or not np.isfinite(gradients).all():
            raise ValueError("finite source gradient receipt required")
        receipt.update(batch_hash=stable_hash(gradient_batch_binding), source_raw_available=raw is not None)
        if stable_hash(self.batch_binding(batch, update_index)) != receipt["batch_hash"]:
            raise ValueError("loss-only inputs differ from the gradient batch descriptor")
        receipt["loss_calls"] += 1
        before = self._losses(parameters.copy(), batch, update_index, receipt["batch_hash"])
        if (not np.allclose(before[1], normalized, rtol=1e-11, atol=1e-18)
                or (raw is not None and not np.allclose(before[0], raw, rtol=1e-11, atol=1e-18))):
            raise ValueError("loss-only parent differs from the source gradient receipt")
        raw = before[0]
        proposal, descent = np.asarray(optimizer[6]), np.asarray(optimizer[4])
        if (len(optimizer) != 11 or not bool(optimizer[-1]) or any(
                value.dtype != np.float64 or value.shape != parameters.shape or not np.isfinite(value).all()
                for value in (proposal, descent))):
            raise ValueError("valid source optimizer proposal required")
        def stable_norm(vector):
            maximum = np.max(np.abs(vector))
            if not maximum:
                return 0.
            scaled_norm = np.linalg.norm(vector / maximum)
            return maximum * scaled_norm if maximum <= np.finfo(np.float64).max / scaled_norm else np.inf

        mask = np.array([index in active for index in range(count)])
        direct_component = bool(self.extra_descent_indices) or self.component_direction == "relative_public_progress"
        direct_curvature = self.residual_provider is not None
        if direct_curvature:
            receipt["source_proposal"] = {"displacement_norm": stable_norm(proposal),
                "gradient_dots": (gradients @ proposal).tolist()}
            direction = self._curvature_direction(parameters, batch, update_index, before[1], gradients, receipt)
        elif direct_component:
            proposal_norm = stable_norm(proposal)
            if not np.isfinite(proposal_norm) or proposal_norm <= 0:
                raise PostproposalRejected("no finite nonzero source displacement", receipt)
            receipt["blend_fraction"] = None
            direction = self._component_direction(parameters, batch, update_index, before[1], gradients, mask,
                proposal, receipt, original_segment_valid=None)
        else:
            norm, proposal_norm = stable_norm(descent), stable_norm(proposal)
            if not np.isfinite(norm) or not np.isfinite(proposal_norm) or norm <= 0:
                raise PostproposalRejected("no finite nonzero reference direction", receipt)
            reference = -(descent / norm) * proposal_norm
            blend = self.blend(proposal, reference, gradients, mask)
            receipt["blend_fraction"] = float(blend["fraction"])
            direction = np.asarray(blend["displacement"])
            if not bool(blend["valid"]):
                if self.component_rows is None:
                    raise PostproposalRejected("no valid displacement on the declared segment", receipt)
                with np.errstate(over="ignore", invalid="ignore"):
                    difference = np.asarray(blend["reference_slopes"]) - np.asarray(blend["proposal_slopes"])
                if (not all(np.isfinite(np.asarray(value)).all() for value in blend.values())
                        or not np.isfinite(difference).all()):
                    raise PostproposalRejected("nonfinite original segment cannot invoke component fallback", receipt)
                direction = self._component_direction(parameters, batch, update_index, before[1], gradients, mask,
                                                      proposal, receipt)
        feasible = [index for index in protected if raw[index] < self.limits[index]]
        required_decrease = list(range(count)) if direct_curvature else sorted(set(active).union(self.extra_descent_indices))

        def trials():
            if direct_curvature or direct_component or not self.finite_component_fallback or not bool(blend["valid"]):
                kind = receipt.get("direction_kind", "original-segment")
                for fraction in self.fractions:
                    yield fraction, direction, kind
                    if (self.finite_target_refinement and "target_refinement" not in receipt
                            and receipt["candidates"][-1].get("component", {}).get("strict_decrease") is False):
                        alternative = self._refine_targets(before[1], gradients, mask, proposal, receipt,
                            receipt["candidates"][-1])
                        if alternative is not None:
                            for refined_fraction in self.fractions:
                                yield refined_fraction, alternative, "same-parent-target-refinement"
                return
            receipt["original_segment_valid"] = True
            yield self.fractions[0], direction, "original-segment"
            alternative = self._component_direction(parameters, batch, update_index, before[1], gradients, mask,
                direction, receipt, original_segment_valid=True)
            if alternative is not None:
                for fraction in self.fractions:
                    yield fraction, alternative, "supplied-component-equality-witness"
            for fraction in self.fractions[1:]:
                yield fraction, direction, "original-segment"

        selected, remaining_lookahead = None, None
        for fraction, trial_direction, direction_kind in trials():
            if remaining_lookahead is not None:
                if remaining_lookahead == 0:
                    break
                remaining_lookahead -= 1
            candidate = parameters + fraction * trial_direction
            direction_record = ({"direction_kind": direction_kind}
                if self.finite_component_fallback or self.finite_target_refinement else {})
            if not np.isfinite(candidate).all():
                receipt["candidates"].append({"fraction": fraction, "accepted": False, "reason": "nonfinite parameters",
                                              **direction_record})
                continue
            if self.finite_target_refinement and receipt["loss_calls"] >= self.binding["maximum_loss_calls"]:
                raise PostproposalRejected("full loss call budget exhausted", receipt)
            receipt["loss_calls"] += 1
            try:
                candidate_raw, _ = self._losses(candidate, batch, update_index, receipt["batch_hash"])
            except NonfinitePostproposalLoss as error:
                receipt["candidates"].append({"fraction": fraction, "accepted": False,
                                              "reason": str(error), "numerical_failure": True, **direction_record})
                continue
            accepted = bool(np.all(candidate_raw[required_decrease] < raw[required_decrease])
                and np.all(candidate_raw[feasible] < self.limits[feasible]))
            candidate_record = {"fraction": fraction, "accepted": accepted and not self.coupled_component_guard,
                "raw": candidate_raw.tolist(), "raw_over_T": (candidate_raw / self.limits).tolist(), **direction_record}
            receipt["candidates"].append(candidate_record)
            component_accepted = None
            if self.finite_target_refinement:
                candidate_record["rounded_displacement"] = (candidate - parameters).tolist()
                try:
                    component_accepted, component_record = self._component_losses(candidate, batch, update_index, receipt)
                except NonfinitePostproposalLoss as error:
                    candidate_record.update(reason=str(error), numerical_failure=True)
                    continue
                candidate_record["component"] = component_record
                if not component_accepted:
                    candidate_record.update(reason="required component did not strictly decrease")
            if not accepted or component_accepted is False:
                continue
            displacement = candidate - parameters
            protected_rows = gradients[list(protected)]
            magnitude = np.max(np.abs(protected_rows), axis=1, keepdims=True) if protected else np.zeros((0, 1))
            scaled = protected_rows / np.where(magnitude > 0, magnitude, 1.)
            norms = np.linalg.norm(scaled, axis=1, keepdims=True)
            unit = scaled / np.where(norms > 0, norms, 1.)
            dots = unit @ displacement
            if not np.isfinite(displacement).all() or not np.isfinite(dots).all() or np.any(dots > 1e-10):
                if direct_curvature:
                    candidate_record.update(accepted=False,
                        reason="rounded displacement failed source protected-dot validity", protected_dots=dots.tolist())
                    continue
                if ((direct_component or self.finite_component_fallback and bool(blend["valid"]))
                        and direction_kind in ("supplied-component-equality-witness", "supplied-component-relative-public-progress",
                                               "same-parent-target-refinement")):
                    receipt["candidates"][-1].update(accepted=False,
                        reason="rounded displacement failed source protected-dot validity",
                        protected_dots=dots.tolist())
                    continue
                raise PostproposalRejected("rounded displacement failed source protected-dot validity", receipt)
            if self.coupled_component_guard:
                if component_accepted is None:
                    try:
                        component_accepted, component_record = self._component_losses(candidate, batch, update_index, receipt)
                    except NonfinitePostproposalLoss as error:
                        candidate_record.update(reason=str(error), numerical_failure=True)
                        continue
                    candidate_record["component"] = component_record
                if not component_accepted:
                    candidate_record.update(reason="required component did not strictly decrease")
                    continue
                candidate_record["accepted"] = True
            updated = list(optimizer)
            updated[0], updated[6], updated[8] = (tf.constant(value, tf.float64) for value in (candidate, displacement, dots))
            if self.lookahead_steps:
                score = float(np.min((raw[list(active)] - candidate_raw[list(active)]) / raw[list(active)]))
                candidate_record.update(feasible=True, accepted=False, minimum_active_relative_decrease=score)
                if selected is None:
                    remaining_lookahead = self.lookahead_steps
                if selected is None or score > selected["score"] + self.binding["finite_selection"]["minimum_gain"]:
                    selected = {"score": score, "updated": tuple(updated), "record": candidate_record,
                        "fraction": fraction, "direction_kind": direction_kind}
                continue
            receipt.update(accepted=True, accepted_fraction=fraction, source_moments_preserved=True)
            if direct_curvature:
                receipt.update(rounded_displacement=displacement.tolist(),
                    actual_all_task_gradient_dots=(gradients @ displacement).tolist())
            if self.finite_component_fallback or self.finite_target_refinement:
                receipt["direction_kind"] = direction_kind
            return tuple(updated), receipt
        if selected is not None:
            selected["record"]["accepted"] = True
            receipt.update(accepted=True, accepted_fraction=selected["fraction"], source_moments_preserved=True,
                finite_selection={**self.binding["finite_selection"], "selected_score": selected["score"],
                    "selected_candidate_index": receipt["candidates"].index(selected["record"])})
            return selected["updated"], receipt
        raise PostproposalRejected("finite candidate budget exhausted", receipt)
