"""Guard schedule provenance only; one real split lifecycle, grouped host refusals.

Execution belongs to the packet's shared 120 CPU-second, two-invocation budget.
The manufactured comparator includes the same fresh stage-entry controls. It
does not claim equivalence to an uninterrupted stage or economic convergence.
"""

import hashlib
import json
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow as tf

from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_permanent_pass import PermanentPassState
from mooneural.training.generic_postproposal import GuardedPostproposal
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    PolicyView,
    canonical_json,
    stable_hash,
)

TRANSITION = "postproposal_schedule_transition"
DENOMINATORS = (2., 50.)
CALLS = []


def clone(value):
    return json.loads(canonical_json(value))


def schedule(count):
    return [{"update_index": index, "row_indices": [index % 3], "population_rows": 3}
            for index in range(count)]


def record_call(kind, update_index=None):
    CALLS.append((kind, update_index))


def quadratic(parameters, factor=1.):
    differences = np.asarray(parameters)[None, :] - np.array([[0.], [1.]])
    raw = factor * np.sum(differences**2, axis=1)
    denominators = np.asarray(DENOMINATORS)
    return tuple(tf.constant(value, tf.float64) for value in (
        raw, raw / denominators, factor * 2. * differences / denominators[:, None]))


def full_objective(parameters):
    record_call("full")
    return quadratic(parameters)


def full_values(parameters):
    record_call("values")
    return quadratic(parameters)[:2]


def training_objective(parameters, update_index):
    record_call("training", update_index)
    return quadratic(parameters, .5 + .1 * update_index)


def batch_identity(batch, update_index):
    record_call("batch", update_index)
    assert batch["metadata"]["update_index"] == update_index
    assert batch["metadata"]["training_schedule_entry"] == schedule(update_index + 1)[-1]
    return clone(batch)


def guard_losses(parameters, batch, update_index):
    record_call("guard", update_index)
    batch_identity(batch, update_index)
    return quadratic(parameters, .5 + .1 * update_index)[:2]


def guard(count):
    return GuardedPostproposal(guard_losses, batch_identity, DENOMINATORS, DENOMINATORS,
        binding={"criterion": "manufactured-quadratic", "task_ids": ["active", "protected"],
                 "source": "fixed quadratic", "training_schedule": schedule(count)})


def arguments(count):
    source = finite._reference(__file__)
    return {"task_ids": ("active", "protected"), "denominators": DENOMINATORS,
        "threshold": 1., "parameters": [3., 4.], "learning_rate": .01,
        "updates": count, "boundary_every": 1, "lifecycle_endpoint": "validation-only",
        "objective": full_objective, "objective_values": full_values,
        "training_objective": training_objective, "training_schedule": tuple(schedule(count)),
        "postproposal": guard(count),
        "data_binding": {"objective_recipe": {"kind": "manufactured-quadratic"},
            "objective_source": source, "training_objective_source": source, "postproposal_source": source},
        "source_evidence": {"source": source}}


def continuation_arguments(original, parent, runtime, count, stage_id):
    reference = finite._write_json(runtime.parent / f"{runtime.name}-result.json",
        {"training_completed": True, "final_state": parent.to_dict()})
    rotation = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"])
    binding = {"parent_result": reference, "parent_checkpoint": reference,
        "parent_configuration": finite._reference(runtime / "configuration.json"),
        "parent_control_archive": [finite._reference(runtime / "checkpoints/control_evidence" /
            f'{point.evaluation.raw_records["full_control_sha256"]}.json') for point in rotation.controls]}
    return {**original, "parameters": parent.policy_state["values"], "updates": count - parent.update_index,
        "training_schedule": tuple(schedule(count)), "postproposal": guard(count), "stage_id": stage_id,
        "continuation_checkpoint": parent, "continuation_binding": binding,
        TRANSITION: {"effective_update": parent.update_index, "reason": "append declared training rows"}}


def numerical(state):
    payload = state.to_dict()
    return {key: payload[key] for key in (
        "update_index", "policy_state", "optimizer_state", "method_state", "rng_state")}


def file_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def assert_lineage(runtime, runner, state, parent, previous):
    config = json.loads((runtime / "configuration.json").read_text())
    lineage = config["checkpoint_continuation"][TRANSITION]
    assert lineage == clone(runner.design.optimizer_binding[TRANSITION])
    assert lineage == clone(state.metadata["checkpoint_continuation"][TRANSITION])
    assert lineage["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
    assert lineage["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    assert lineage["effective_update"] == parent.update_index
    assert lineage["reason"] == "append declared training rows"
    assert lineage["previous_transition"] == previous
    old_binding = parent.metadata["replication_design"]["optimizer_binding"]["postproposal"]
    assert lineage["old_postproposal_hash"] == stable_hash(old_binding)
    assert lineage["new_postproposal_hash"] == stable_hash(config["postproposal"])
    assert lineage["old_schedule_hash"] == stable_hash(schedule(parent.update_index))
    assert lineage["new_schedule_hash"] == stable_hash(config["training"]["schedule"])
    return lineage


def assert_completed_recovery(runtime, args, expected, monkeypatch):
    before = file_hashes(runtime)
    forbidden = Mock(side_effect=AssertionError("completed recovery invoked numerics"))
    with monkeypatch.context() as patches:
        patches.setitem(globals(), "record_call", forbidden)
        runner, adapter, provider = finite.build_runner(runtime, **args)
        for name in ("control", "validation", "certification"):
            patches.setattr(provider, name, forbidden)
        for name in ("evaluate", "evaluate_values", "compute_task_values_and_gradients"):
            patches.setattr(adapter, name, forbidden)
        patches.setattr(runner, "batch_factory", forbidden)
        patches.setattr(runner.executor, "step", forbidden)
        actual = runner.run()
        assert actual.states[0].to_dict() == expected.states[0].to_dict()
        assert actual.evaluations == expected.evaluations
        assert actual.selection == expected.selection
        forbidden.assert_not_called()
    assert file_hashes(runtime) == before


def test_actual_guarded_split_recovery_and_third_continuation(tmp_path, monkeypatch):
    CALLS.clear()
    original = arguments(1)
    parent_root = tmp_path / "parent"
    parent_runner, _adapter, _provider = finite.build_runner(parent_root, **original)
    parent = parent_runner.run().states[0]
    prior = PermanentPassState.from_dict(parent.metadata["permanent_pass_rotation"])
    assert prior.partition["constraints"] == ["protected"]
    assert parent.update_index == parent.optimizer_state["iteration"] == 1
    assert any(parent.optimizer_state["first_moment"])
    assert any(parent.optimizer_state["second_moment"])
    child = continuation_arguments(original, parent, parent_root, 3, "guarded-child")
    immutable_parent = file_hashes(parent_root)
    parent_state = parent.to_dict()
    reference = child["continuation_binding"]["parent_result"]
    immutable_result = finite._checked(reference).read_bytes()

    forbidden = Mock(side_effect=AssertionError("invalid continuation invoked numerics"))
    with monkeypatch.context() as patches:
        patches.setitem(globals(), "record_call", forbidden)
        with pytest.raises(ValueError, match="without a verified transition"):
            finite.build_runner(tmp_path / "undeclared", **{**child, TRANSITION: None})
        with pytest.raises(ValueError, match="old/new guards"):
            finite.build_runner(tmp_path / "removed-guard", **{**child, "postproposal": None})
        with pytest.raises(ValueError, match="verified continuation parent"):
            finite.build_runner(tmp_path / "no-parent", **{**original, TRANSITION: child[TRANSITION]})
        bad_binding = clone(child["continuation_binding"])
        bad_binding["parent_checkpoint"]["sha256"] = "0" * 64
        with pytest.raises(ValueError):
            finite.build_runner(tmp_path / "changed-parent", **{**child, "continuation_binding": bad_binding})
        bad_binding = clone(child["continuation_binding"])
        bad_binding["parent_control_archive"][0]["sha256"] = "0" * 64
        with pytest.raises(ValueError):
            finite.build_runner(tmp_path / "changed-control", **{**child, "continuation_binding": bad_binding})
        forbidden.assert_not_called()

    complete_root = tmp_path / "complete-child"
    complete, adapter, _provider = finite.build_runner(complete_root, **child)
    assert numerical(complete.states[0]) == numerical(parent)
    assert complete.states[0].metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]
    assert adapter.policy_metadata == parent.policy_state["metadata"]
    assert complete.design.optimizer_binding["fresh_moments"] is False
    expected = complete.run()
    first_lineage = assert_lineage(complete_root, complete, expected.states[0], parent, None)

    resumed_root = tmp_path / "interrupted-child"
    interrupted, _adapter, _provider = finite.build_runner(resumed_root, **child)
    original_step = interrupted.executor.step

    def interrupt(state, batch, adapter):
        if state.update_index == 2:
            raise RuntimeError("guarded split interruption")
        return original_step(state, batch, adapter)

    monkeypatch.setattr(interrupted.executor, "step", interrupt)
    with pytest.raises(RuntimeError, match="guarded split interruption"):
        interrupted.run()
    before_resume = len(CALLS)
    resumed, _adapter, _provider = finite.build_runner(resumed_root, **child)
    actual = resumed.run()
    assert [clock for kind, clock in CALLS[before_resume:] if kind == "training"] == [2]
    assert numerical(actual.states[0]) == numerical(expected.states[0])
    assert actual.states[0].optimizer_state["iteration"] == actual.states[0].rng_state["next_seed_index"] == 3
    assert actual.states[0].method_state["preferred"] == "cagrad"
    assert_lineage(resumed_root, resumed, actual.states[0], parent, None)
    for runtime, runner, result in ((complete_root, complete, expected), (resumed_root, resumed, actual)):
        rotation = runner.executor.core.coordinator.read(result.states[0])
        assert [point.update_index for point in rotation.controls] == [0, 1, 1, 2, 3]
        assert rotation.controls[:len(prior.controls)] == prior.controls
        assert rotation.controls[2].stage_id == "guarded-child"
        assert rotation.permanent == prior.permanent
        transactions = [json.loads(path.read_text()) for path in sorted(
            (runtime / "checkpoints/arms/arm-0/checkpoints").glob("update-????????.json"))]
        assert [entry["event"]["postproposal"]["update_index"] for entry in transactions] == [1, 2]
        assert all(entry["event"]["postproposal"]["accepted"] for entry in transactions)
    assert_completed_recovery(resumed_root, child, actual, monkeypatch)

    third_args = continuation_arguments(original, actual.states[0], resumed_root, 4, "guarded-third")
    immutable_child = file_hashes(resumed_root)
    third_root = tmp_path / "third"
    third, _adapter, _provider = finite.build_runner(third_root, **third_args)
    assert numerical(third.states[0]) == numerical(actual.states[0])
    third_result = third.run()
    third_state = third_result.states[0]
    assert_lineage(third_root, third, third_state, actual.states[0], first_lineage)
    rotation = third.executor.core.coordinator.read(third_state)
    previous_rotation = resumed.executor.core.coordinator.read(actual.states[0])
    assert rotation.controls[:len(previous_rotation.controls)] == previous_rotation.controls
    assert [point.update_index for point in rotation.controls] == [0, 1, 1, 2, 3, 3, 4]
    assert rotation.controls[5].stage_id == "guarded-third"
    assert rotation.permanent == prior.permanent
    assert third_state.update_index == third_state.optimizer_state["iteration"] == third_state.rng_state["next_seed_index"] == 4
    assert_completed_recovery(third_root, third_args, third_result, monkeypatch)
    assert parent.to_dict() == parent_state
    assert finite._checked(reference).read_bytes() == immutable_result
    assert file_hashes(parent_root) == immutable_parent
    assert file_hashes(resumed_root) == immutable_child


@pytest.fixture(scope="module")
def helper_template():
    profile = clone(guard(2).binding)
    source = finite._reference(__file__)
    binding = {"profile": profile,
        "loss_callback": {"source": source, "qualname": guard_losses.__qualname__},
        "batch_binding_callback": {"source": source, "qualname": batch_identity.__qualname__}}
    config = {"postproposal": binding, "training": {"schedule": schedule(2)}}
    updated = clone(binding)
    updated["profile"]["profile"]["training_schedule"] = schedule(3)
    return config, updated, {"schedule": schedule(3)}, {"effective_update": 2, "reason": "extend fixture"}


def helper_parent(config, clock=2):
    previous = config.get("checkpoint_continuation", {}).get(TRANSITION)
    return CheckpointState("helper-parent", clock, PolicyView((3., 4.)).to_dict(),
        {"first_moment": [1., 2.], "second_moment": [3., 4.], "iteration": clock, "learning_rate": .01},
        {"preferred": "cagrad", "rates": {"cagrad": .01}}, {"next_seed_index": clock},
        metadata={"replication_design_sha256": "a" * 64,
            "replication_design": {"optimizer_binding": {"postproposal": clone(config.get("postproposal")),
                TRANSITION: clone(previous)}}, "checkpoint_continuation": {TRANSITION: clone(previous)}})


def set_nested(mapping, keys, value):
    for key in keys[:-1]:
        mapping = mapping[key]
    mapping[keys[-1]] = value


def test_helper_defaults_explicit_extension_and_inherited_lineage(helper_template):
    config, binding, training, declaration = clone(helper_template)
    parent = helper_parent(config)
    before = clone((parent.to_dict(), config, binding, training))
    transition = finite._postproposal_schedule_transition(parent, config, binding, training, declaration)
    assert transition["old_postproposal_hash"] == stable_hash(config["postproposal"])
    assert transition["new_postproposal_hash"] == stable_hash(binding)
    assert transition["old_schedule_hash"] == stable_hash(config["training"]["schedule"])
    assert transition["new_schedule_hash"] == stable_hash(training["schedule"])
    assert transition["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
    assert transition["parent_design_hash"] == parent.metadata["replication_design_sha256"]
    assert transition["previous_transition"] is None
    assert clone((parent.to_dict(), config, binding, training)) == before
    assert finite._postproposal_schedule_transition(parent, config, config["postproposal"], training, None) is None
    no_guard = {"training": None}
    assert finite._postproposal_schedule_transition(helper_parent(no_guard), no_guard, None, None, None) is None
    inherited_config = {"postproposal": binding, "training": training,
                        "checkpoint_continuation": {TRANSITION: transition}}
    inherited_parent = helper_parent(inherited_config, 3)
    assert finite._postproposal_schedule_transition(inherited_parent, inherited_config, binding, training, None) == transition
    final_binding = clone(binding)
    final_binding["profile"]["profile"]["training_schedule"] = schedule(4)
    following = finite._postproposal_schedule_transition(inherited_parent, inherited_config, final_binding,
        {"schedule": schedule(4)}, {"effective_update": 3, "reason": "third stage"})
    assert following["previous_transition"] == transition
    assert following["parent_checkpoint_hash"] == stable_hash(inherited_parent.to_dict())


def test_helper_rejects_absent_or_malformed_declarations(helper_template):
    config, binding, training, declaration = clone(helper_template)
    parent = helper_parent(config)
    with pytest.raises(ValueError, match="without a verified transition"):
        finite._postproposal_schedule_transition(parent, config, binding, training, None)
    invalid = [False, [], {}, {"effective_update": 2}, {"reason": "missing clock"},
        {**declaration, "effective_update": 1}, {**declaration, "effective_update": True},
        {**declaration, "effective_update": 2.}, {**declaration, "reason": ""},
        {**declaration, "reason": "  "}, {**declaration, "reason": None}, {**declaration, "extra": True}]
    for supplied in invalid:
        with pytest.raises(ValueError, match="invalid postproposal schedule transition"):
            finite._postproposal_schedule_transition(parent, config, binding, training, supplied)


def test_helper_requires_guards_training_and_exact_schedule_prefix(helper_template):
    for missing in ("old_guard", "new_guard", "old_training", "new_training"):
        config, binding, training, declaration = clone(helper_template)
        if missing == "old_guard":
            config["postproposal"] = None
        elif missing == "new_guard":
            binding = None
        elif missing == "old_training":
            config["training"] = None
        else:
            training = None
        with pytest.raises(ValueError, match="old/new guards and scheduled training"):
            finite._postproposal_schedule_transition(helper_parent(config), config, binding, training, declaration)
    for mutation in ("noop", "truncate", "prefix", "order", "old_context", "new_context", "parent_clock"):
        config, binding, training, declaration = clone(helper_template)
        clock = 2
        if mutation in ("noop", "truncate"):
            training["schedule"] = schedule(2 if mutation == "noop" else 1)
            binding["profile"]["profile"]["training_schedule"] = clone(training["schedule"])
        elif mutation in ("prefix", "order"):
            if mutation == "prefix":
                training["schedule"][0]["row_indices"] = [99]
            else:
                training["schedule"][:2] = training["schedule"][:2][::-1]
            binding["profile"]["profile"]["training_schedule"] = clone(training["schedule"])
        elif mutation == "old_context":
            config["postproposal"]["profile"]["profile"]["training_schedule"][0]["row_indices"] = [99]
        elif mutation == "new_context":
            binding["profile"]["profile"]["training_schedule"][2]["row_indices"] = [99]
        else:
            clock = declaration["effective_update"] = 3
        with pytest.raises(ValueError, match="exact generic schedule prefix"):
            finite._postproposal_schedule_transition(helper_parent(config, clock), config, binding, training, declaration)


def test_helper_rejects_other_context_callbacks_and_numerical_policy(helper_template):
    mutations = [
        (("loss_callback", "qualname"), "other_losses"),
        (("batch_binding_callback", "qualname"), "other_inputs"),
        (("loss_callback", "source", "sha256"), "0" * 64),
        (("batch_binding_callback", "source", "sha256"), "0" * 64),
        (("profile", "profile", "criterion"), "different criterion"),
        (("profile", "profile", "task_ids"), ["protected", "active"]),
        (("profile", "profile", "source"), "changed context source"),
        (("profile", "margin_fraction"), .2),
        (("profile", "fractions"), [1., .25]),
        (("profile", "D"), [3., 50.]),
        (("profile", "T"), [2., 60.]),
        (("profile", "relative_tolerance"), 1e-9),
        (("profile", "protected_dot_absolute_tolerance"), 1e-8),
        (("profile", "loss_parity_rtol"), 1e-6),
        (("profile", "reference"), "new reference"),
        (("profile", "moments"), "reset moments"),
        (("profile", "failure"), "commit rejected state"),
        (("profile", "input_identity"), "different batch semantics"),
        (("profile", "contract_failure"), "ignore mismatch"),
    ]
    for keys, value in mutations:
        config, binding, training, declaration = clone(helper_template)
        set_nested(binding, keys, value)
        with pytest.raises(ValueError, match="changed callbacks, context or numerical profile"):
            finite._postproposal_schedule_transition(helper_parent(config), config, binding, training, declaration)


def test_helper_rejects_inconsistent_lineage_copies_and_derived_identity(helper_template):
    config, binding, training, declaration = clone(helper_template)
    previous = finite._postproposal_schedule_transition(helper_parent(config), config, binding, training, declaration)
    inherited = {"postproposal": binding, "training": training,
                 "checkpoint_continuation": {TRANSITION: previous}}
    for location in ("configuration", "design", "state", "old_guard"):
        config = clone(inherited)
        parent = helper_parent(config, 3)
        metadata = clone(parent.metadata)
        if location == "configuration":
            config["checkpoint_continuation"][TRANSITION]["reason"] = "tampered"
        elif location == "design":
            metadata["replication_design"]["optimizer_binding"][TRANSITION]["reason"] = "tampered"
        elif location == "state":
            metadata["checkpoint_continuation"][TRANSITION]["reason"] = "tampered"
        else:
            metadata["replication_design"]["optimizer_binding"]["postproposal"]["profile"]["margin_fraction"] = .2
        parent = replace(parent, metadata=metadata)
        with pytest.raises(ValueError, match="lineage differs from parent design/state"):
            finite._postproposal_schedule_transition(parent, config, binding, training, None)
    for field, value in (("new_postproposal_hash", "0" * 64), ("new_schedule_hash", "0" * 64), ("effective_update", 4)):
        config = clone(inherited)
        config["checkpoint_continuation"][TRANSITION][field] = value
        with pytest.raises(ValueError, match="inconsistent profile/schedule/clock"):
            finite._postproposal_schedule_transition(helper_parent(config, 3), config, binding, training, None)
