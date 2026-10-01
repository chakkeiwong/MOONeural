"""Shared profile transitions with manufactured leaves and actual checkpoint stores."""

import copy
import json
from dataclasses import replace

import pytest
from tests.contracts import test_generic_checkpoint_continuation as checkpoints
from tests.contracts import test_generic_coupled_component_transition as coupled
from tests.contracts import test_generic_partial_continuation as partial
from tests.contracts import test_generic_postproposal_policy_transition as policies
from mooneural.training import generic_finite_objective_runner as finite
from mooneural.training.generic_permanent_pass_executor import PermanentPassUpdateError
from mooneural.training.generic_postproposal import GuardedPostproposal
from mooneural.training.generic_training_contracts import canonical_json, stable_hash

NAME = "postproposal_search_profile_transition"
FRACTIONS = (1., .5, .25, .125, .0625, .03125)


def guard(fractions=FRACTIONS[:4], *, count=None, **changes):
    context = {"criterion": "manufactured-search-profile"}
    if count is not None:
        context["training_schedule"] = list(checkpoints.schedule(count))
    return GuardedPostproposal(partial.guard_losses, partial.batch_binding, [1.] * 7, [1.] * 7,
                              binding=context, fractions=fractions, **changes)


def host_runtime(runner, provider, patches, *, reject_at=None):
    calls = checkpoints.host_runtime(runner, provider, patches)
    inherited = runner.executor.step

    def step(state, batch, adapter):
        if state.update_index == reject_at:
            partition = runner.executor.core.coordinator.read(state).partition
            raise PermanentPassUpdateError("manufactured search rejection", state,
                {"event": "update", "committed": False, "transaction_rolled_back": True,
                 "policy_before": state.policy_fingerprint, "before": partition, "after": partition,
                 "postproposal": {"accepted": False, "update_index": state.update_index,
                     "profile_hash": runner.executor.core.postproposal.binding_hash,
                     "loss_calls": 1, "candidates": []}})
        batch = replace(batch, metadata={**batch.metadata,
                                        "training_schedule_entry": {"update_index": state.update_index}})
        return inherited(state, batch, adapter)

    patches.setattr(runner.executor, "step", step)
    return calls


def saved_binding(runtime, state):
    binding = policies.saved_partial_binding(runtime, state)
    boundary = runtime / "checkpoints/arms/arm-0/boundaries" / f"boundary-{state.update_index:08d}.json"
    if boundary.is_file():
        binding["parent_checkpoint"] = finite._reference(boundary)
        binding["parent_commit_marker"] = finite._reference(boundary.with_suffix(".complete.json"))
    return binding


@pytest.fixture(scope="module", params=[False, True], ids=["unscheduled", "scheduled"])
def parent_run(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("search-profile-parent")
    args = checkpoints.fixture_arguments(root, 6)
    if not request.param:
        args.pop("training_schedule")
        args.pop("training_objective")
        args["data_binding"].pop("training_objective_source")
    args.update(postproposal=guard(count=6 if request.param else None), stage_id="search-parent")
    args["data_binding"]["postproposal_source"] = finite._reference(partial.__file__)
    args["source_evidence"]["guard_fixture"] = finite._reference(partial.__file__)
    args["source_evidence"]["search_fixture"] = finite._reference(__file__)
    runtime = root / "runtime"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    with pytest.MonkeyPatch.context() as patches:
        calls = host_runtime(runner, provider, patches, reject_at=4)
        with pytest.raises(PermanentPassUpdateError) as rejected:
            runner.run()
    assert calls["updates"] == [0, 1, 2, 3]
    parent = rejected.value.checkpoint
    return parent, args, runtime, saved_binding(runtime, parent)


def child_arguments(parent_run):
    state, args, _runtime, binding = parent_run
    return {**args, "parameters": state.policy_state["values"], "updates": 2,
        "stage_id": "search-child", "continuation_checkpoint": state,
        "continuation_binding": copy.deepcopy(binding), "partial_parent_continuation": partial.declaration(4),
        NAME: {"effective_update": 4, "reason": "Qualify appended search fractions"},
        "postproposal": guard(FRACTIONS, count=6 if args.get("training_schedule") is not None else None)}


def test_partial_entry_state_recovery_and_inheritance(parent_run, tmp_path, monkeypatch):
    parent, _original, parent_root, _binding = parent_run
    old_files = partial.hashes(parent_root)
    args = child_arguments(parent_run)
    runtime = tmp_path / "child"
    runner, adapter, provider = finite.build_runner(runtime, **args)
    assert checkpoints.numerical(runner.states[0]) == checkpoints.numerical(parent)
    assert runner.states[0].metadata["permanent_pass_rotation"] == parent.metadata["permanent_pass_rotation"]
    receipt = runner.checkpoint_continuation[NAME]
    assert tuple(receipt["old_binding"]["profile"]["fractions"]) == FRACTIONS[:4]
    assert receipt["new_binding"]["profile"]["maximum_loss_calls"] == 7
    assert receipt["parent_checkpoint_hash"] == stable_hash(parent.to_dict())
    assert runner.design.optimizer_binding[NAME] == receipt
    assert canonical_json(json.loads((runtime / "configuration.json").read_text())["checkpoint_continuation"][NAME]) == canonical_json(receipt)
    calls = host_runtime(runner, provider, monkeypatch)
    result = runner.run()
    assert calls["updates"] == [4, 5]
    assert result.states[0].update_index == 6
    saved = partial.hashes(runtime)
    rebuilt, adapter, provider = finite.build_runner(runtime, **args)
    partial.forbid_runtime(rebuilt, adapter, provider, monkeypatch)
    assert rebuilt.run().states[0].to_dict() == result.states[0].to_dict()
    assert partial.hashes(runtime) == saved
    later = {**args, "parameters": result.states[0].policy_state["values"], "updates": 1,
        "stage_id": "search-grandchild", "continuation_checkpoint": result.states[0],
        "continuation_binding": partial.completed_binding(tmp_path, result.states[0], runtime)}
    later.pop("partial_parent_continuation")
    later.pop(NAME)
    if args.get("training_schedule") is not None:
        later.update(training_schedule=checkpoints.schedule(7), postproposal=guard(FRACTIONS, count=7),
            postproposal_schedule_transition={"effective_update": 6, "reason": "Preserve search across schedule extension"})
    ordinary, _adapter, provider = finite.build_runner(tmp_path / "later", **later)
    assert ordinary.checkpoint_continuation[NAME] == receipt
    host_runtime(ordinary, provider, monkeypatch)
    final = ordinary.run().states[0]
    config = json.loads((tmp_path / "later/configuration.json").read_text())
    finite._postproposal_lineage(final, config)
    assert partial.hashes(parent_root) == old_files


def test_same_clock_rejection_recovery_and_second_extension(parent_run, tmp_path, monkeypatch):
    args = child_arguments(parent_run)
    runtime = tmp_path / "rejected-child"
    runner, _adapter, provider = finite.build_runner(runtime, **args)
    host_runtime(runner, provider, monkeypatch, reject_at=4)
    with pytest.raises(PermanentPassUpdateError) as rejected:
        runner.run()
    state = rejected.value.checkpoint
    rebuilt, adapter, provider = finite.build_runner(runtime, **args)
    partial.forbid_runtime(rebuilt, adapter, provider, monkeypatch)
    with pytest.raises(PermanentPassUpdateError):
        rebuilt.run()
    next_args = {**args, "continuation_checkpoint": state,
        "continuation_binding": saved_binding(runtime, state),
        "stage_id": "second-extension", "postproposal": guard(FRACTIONS + (.015625,),
            count=6 if args.get("training_schedule") is not None else None)}
    child, _adapter, provider = finite.build_runner(tmp_path / "second", **next_args)
    assert child.checkpoint_continuation[NAME]["previous_transition"] == runner.checkpoint_continuation[NAME]
    assert checkpoints.numerical(child.states[0]) == checkpoints.numerical(state)
    host_runtime(child, provider, monkeypatch)
    assert child.run().states[0].update_index == 6


@pytest.mark.parametrize("fault", ["undeclared", "clock", "reason", "replace_fraction", "short_horizon",
                                   "data", "state", "method", "callback"])
def test_bad_transitions_refused_before_callbacks(parent_run, tmp_path, fault):
    args = child_arguments(parent_run)
    if fault == "undeclared":
        args.pop(NAME)
    elif fault == "clock":
        args[NAME]["effective_update"] = 3
    elif fault == "reason":
        args[NAME]["reason"] = ""
    elif fault == "replace_fraction":
        args["postproposal"] = guard((1., .5, .125, .0625),
                                     count=6 if args.get("training_schedule") is not None else None)
    elif fault == "short_horizon":
        args["updates"] = 1
        if args.get("training_schedule") is not None:
            args["training_schedule"] = checkpoints.schedule(5)
    elif fault == "data":
        args["data_binding"] = {**args["data_binding"], "fixed_data": [999.]}
    elif fault == "state":
        state = args["continuation_checkpoint"]
        args["continuation_checkpoint"] = replace(state, rng_state={**state.rng_state, "next_seed_index": 99})
    elif fault == "method":
        args["method_policy_transition"] = {"effective_update": 4}
    elif fault == "callback":
        args["postproposal"] = GuardedPostproposal(policies.replacement_rows, partial.batch_binding,
            [1.] * 7, [1.] * 7, binding=args["postproposal"].binding["profile"], fractions=FRACTIONS)
    with pytest.raises(ValueError):
        finite.build_runner(tmp_path / "refused", **args)


@pytest.mark.parametrize("fault", ["old_cap", "new_cap", "fractions", "old_hash", "predecessor"])
def test_bound_lineage_reconstruction_rejects_corruption(parent_run, tmp_path, fault):
    args = child_arguments(parent_run)
    runner, _adapter, _provider = finite.build_runner(tmp_path / "lineage", **args)
    config = json.loads((tmp_path / "lineage/configuration.json").read_text())
    record = config["checkpoint_continuation"][NAME]
    if fault == "old_cap":
        record["old_binding"]["profile"]["maximum_loss_calls"] = 9
    elif fault == "new_cap":
        record["new_binding"]["profile"]["maximum_loss_calls"] = 13
    elif fault == "fractions":
        record["new_binding"]["profile"]["fractions"][-1] = .2
    elif fault == "old_hash":
        record["old_postproposal_hash"] = "0" * 64
    else:
        record["previous_policy_transition_hash"] = "0" * 64
    state = runner.design.bind_checkpoint(runner.states[0]).to_dict()
    state["metadata"]["checkpoint_continuation"][NAME] = copy.deepcopy(record)
    state["metadata"]["replication_design"]["optimizer_binding"][NAME] = copy.deepcopy(record)
    bound_state = replace(runner.design.bind_checkpoint(runner.states[0]), metadata=state["metadata"])
    with pytest.raises(ValueError):
        finite._postproposal_lineage(bound_state, config)


def test_search_composes_with_prior_and_later_schedule_policy_edges():
    def parent(config):
        state = policies.helper_parent(config).to_dict()
        record = config["checkpoint_continuation"].get(NAME)
        state["metadata"]["checkpoint_continuation"][NAME] = record
        state["metadata"]["replication_design"]["optimizer_binding"][NAME] = record
        return replace(policies.helper_parent(config), metadata=state["metadata"])

    config = policies.helper_history()
    extended = policies.helper_binding(policies.enabled_guard(fractions=FRACTIONS))
    search = finite._postproposal_search_profile_transition(parent(config), config, extended,
        {"effective_update": 21, "reason": "Append after schedule and component enablement"})
    config = {**config, "postproposal": extended,
              "checkpoint_continuation": {**config["checkpoint_continuation"], NAME: search}}
    refined = policies.helper_binding(policies.enabled_guard(fractions=FRACTIONS, finite_component_fallback=True))
    schedule, policy = finite._postproposal_transitions(parent(config), config, refined, config["training"],
        {"effective_update": 21, "reason": "Finite trigger after appended fractions", "kind": "component_trigger_refinement"}, None)
    config = {**config, "postproposal": refined,
        "checkpoint_continuation": {NAME: search, policies.SCHEDULE: schedule, policies.POLICY: policy}}
    finite._postproposal_lineage(parent(config), config)
    extended = policies.helper_binding(policies.enabled_guard(31, fractions=FRACTIONS, finite_component_fallback=True))
    training = {"schedule": list(checkpoints.schedule(31))}
    schedule, policy = finite._postproposal_transitions(parent(config), config, extended, training, None,
        {"effective_update": 21, "reason": "Extend schedule after search and trigger"}, partial={"old_schedule_length": 30})
    config = {"postproposal": extended, "training": training,
        "checkpoint_continuation": {NAME: search, policies.SCHEDULE: schedule, policies.POLICY: policy}}
    assert finite._postproposal_lineage(parent(config), config) == (schedule, policy)


def test_search_between_coupled_and_target_policy_edges():
    def parent(config):
        state = policies.helper_parent(config)
        metadata = json.loads(canonical_json(state.metadata))
        record = config.get("checkpoint_continuation", {}).get(NAME)
        metadata["checkpoint_continuation"][NAME] = record
        metadata["replication_design"]["optimizer_binding"][NAME] = record
        return replace(state, metadata=metadata)

    def binding(**options):
        supplied = coupled.guard(30, **options)
        result = policies.helper_binding(supplied)
        result["component_callback"] = {"source": finite._reference(coupled.__file__),
            "module": coupled.components.__module__, "qualname": coupled.components.__qualname__}
        if options.get("coupled"):
            result["component_callback"]["loss_only"] = {"source": finite._reference(coupled.__file__),
                "module": coupled.component_values.__module__, "qualname": coupled.component_values.__qualname__}
        return result

    config = {"postproposal": binding(), "training": {"schedule": list(checkpoints.schedule(30))}}
    coupled_binding = binding(coupled=True)
    schedule, policy = finite._postproposal_transitions(parent(config), config, coupled_binding, config["training"],
        {"effective_update": 21, "reason": "Coupled finite profile", "kind": "coupled_component_refinement"}, None)
    config.update(postproposal=coupled_binding, checkpoint_continuation={policies.SCHEDULE: schedule, policies.POLICY: policy})
    extended = binding(coupled=True, fractions=FRACTIONS)
    search = finite._postproposal_search_profile_transition(parent(config), config, extended,
        {"effective_update": 21, "reason": "Append coupled fractions"})
    config["checkpoint_continuation"][NAME] = search
    config["postproposal"] = extended
    refined = binding(coupled=True, fractions=FRACTIONS, finite_target_refinement=True)
    schedule, policy = finite._postproposal_transitions(parent(config), config, refined, config["training"],
        {"effective_update": 21, "reason": "Target after appended coupled fractions", "kind": "target_refinement"}, None)
    config.update(postproposal=refined, checkpoint_continuation={NAME: search, policies.SCHEDULE: schedule, policies.POLICY: policy})
    assert finite._postproposal_lineage(parent(config), config) == (schedule, policy)
    assert search["new_binding"]["profile"]["component_fallback"]["maximum_component_loss_calls"] == 6
    assert refined["profile"]["component_fallback"]["maximum_component_loss_calls"] == 12


def test_search_only_refuses_hidden_training_schedule_extension():
    config = policies.helper_history()
    training = {"schedule": list(checkpoints.schedule(31))}
    with pytest.raises(ValueError, match="entire training binding"):
        finite._postproposal_transitions(policies.helper_parent(config), config,
            policies.helper_binding(policies.enabled_guard(fractions=FRACTIONS)), training, None, None,
            search_profile_declaration={"effective_update": 21, "reason": "Append fractions only"})
