import json
from dataclasses import replace

import pytest

from mooneural.training.generic_replication_design import ReplicationDesignBinding
from mooneural.training.generic_replication_runner import ReplicationStage
from mooneural.training.generic_training_contracts import (
    CheckpointState,
    PolicyView,
    stable_hash,
)

TASK_IDS = ("task-a", "task-b")


def states():
    return {
        replica: CheckpointState(
            "design-test",
            0,
            PolicyView((float(replica),), f"policy-{replica}").to_dict(),
            {"first_moment": [0.0], "second_moment": [0.0], "iteration": 0, "learning_rate": 0.1},
            {"preferred": "fake", "rates": {"fake": 0.1}},
            {"seed": replica},
        )
        for replica in range(2)
    }


def stages():
    return (
        ReplicationStage("population", 0, 1, 1, 0, updates_per_round=1, population_size=2, select_after=True),
        ReplicationStage("continuation", 1, 2, 2, 1, updates_per_round=1, population_size=1),
    )


def design_for(initial):
    return ReplicationDesignBinding(
        design_id="design-test-v1",
        master_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        generic_code_hashes={"executor.py": "3" * 64},
        environment_binding={"python": "3.11", "runtime": {"numpy": "2"}},
        backend_binding={"backend": "tensorflow", "dtype": "float64"},
        role_manifest_sha256="4" * 64,
        target_hashes={"accepted": "5" * 64},
        initial_state_hashes={
            str(replica): stable_hash(state.to_dict())
            for replica, state in initial.items()
        },
        task_ids=TASK_IDS,
        threshold=0.04,
        replica_ids=(0, 1),
        expected_selected_replica=1,
        selection_key=("absolute_threshold_ratio", "replica"),
        stages=tuple(
            {
                "stage_id": stage.stage_id,
                "from_round": stage.from_round,
                "first_round": stage.first_round,
                "last_round": stage.last_round,
                "start_update": stage.start_update,
                "updates_per_round": stage.updates_per_round,
                "population_size": stage.population_size,
                "select_after": stage.select_after,
            }
            for stage in stages()
        ),
        optimizer_binding={"name": "Adam", "clip_norm": 10.0},
        permanent_pass_binding={"threshold": 0.04, "active_count": "min(3, unresolved_count)"},
        seed_registry_sha256="6" * 64,
    )


def test_design_round_trip_and_checkpoint_binding():
    initial = states()
    design = design_for(initial)
    restored = ReplicationDesignBinding.from_dict(design.to_dict())
    assert restored.to_dict() == design.to_dict()

    bound = design.bind_checkpoint(initial[0])
    design.validate_checkpoint(bound)
    assert bound.metadata["replication_design_sha256"] == design.binding_hash()
    assert bound.metadata["replication_initial_state_sha256"] == stable_hash(initial[0].to_dict())
    with pytest.raises(TypeError):
        design.environment_binding["runtime"]["python"] = "3.12"


def test_legacy_design_hash_is_unchanged():
    design = design_for(states())
    assert design.to_dict()["schema"] == "dsge_hmc.generic_replication_design.v1"
    assert design.binding_hash() == "c9aa58076cfc209709bf62a992825b9ef8725497229ecf9ab562a8b3d1b85e6f"


def test_population_design_roundtrips_and_cannot_relabel_legacy_checkpoint():
    initial = states()
    historical = design_for(initial)
    population = replace(historical, expected_selected_replica=None)
    payload = json.loads(json.dumps(population.to_dict()))
    assert payload["schema"] == "dsge_hmc.generic_replication_design.v2"
    assert payload["expected_selected_replica"] is None
    restored = ReplicationDesignBinding.from_dict(payload)
    assert restored.to_dict() == population.to_dict()
    bound = restored.bind_checkpoint(initial[0])
    restored.validate_checkpoint(CheckpointState.from_dict(json.loads(json.dumps(bound.to_dict()))))
    with pytest.raises(ValueError, match="different replication design"):
        population.bind_checkpoint(historical.bind_checkpoint(initial[0]))


@pytest.mark.parametrize("expected", (None, 1))
def test_selection_mode_cannot_be_silently_changed_by_schema(expected):
    payload = replace(design_for(states()), expected_selected_replica=expected).to_dict()
    payload["schema"] = (
        "dsge_hmc.generic_replication_design.v1" if expected is None
        else "dsge_hmc.generic_replication_design.v2"
    )
    payload["design_sha256"] = stable_hash({key: value for key, value in payload.items() if key != "design_sha256"})
    with pytest.raises(ValueError, match="selection mode mismatch"):
        ReplicationDesignBinding.from_dict(payload)


@pytest.mark.parametrize("expected", (True, 1.0, -1, 2))
def test_historical_expected_replica_remains_checked(expected):
    with pytest.raises(ValueError, match="expected selected replica"):
        replace(design_for(states()), expected_selected_replica=expected)


@pytest.mark.parametrize("defect", ("size", "key", "no-selection", "reselection"))
def test_population_design_requires_implemented_selector_and_complete_population(defect):
    population = replace(design_for(states()), expected_selected_replica=None)
    if defect == "key":
        changes = {"selection_key": ("terminal_certificate", "replica")}
    else:
        stage_records = [dict(stage) for stage in population.stages]
        if defect == "size":
            stage_records[0]["population_size"] = 3
        elif defect == "no-selection":
            stage_records[0]["select_after"] = False
        else:
            stage_records[1]["select_after"] = True
        changes = {"stages": tuple(stage_records)}
    with pytest.raises(ValueError, match="population"):
        replace(population, **changes)


def test_nested_stage_requirements_survive_json_and_checkpoint_transport():
    initial = states()
    design = replace(design_for(initial), permanent_pass_binding={
        "stage_entries": {"continuation": {"permanent_tasks": ["task-a"], "preferred_method": "fake"}},
    })
    document = json.loads(json.dumps(design.to_dict()))
    assert document == design.to_dict()
    restored = ReplicationDesignBinding.from_dict(document)
    bound = restored.bind_checkpoint(initial[0])
    restored.validate_checkpoint(CheckpointState.from_dict(json.loads(json.dumps(bound.to_dict()))))
    document["permanent_pass_binding"]["stage_entries"]["continuation"]["permanent_tasks"].clear()
    assert design.to_dict()["permanent_pass_binding"]["stage_entries"]["continuation"]["permanent_tasks"] == ["task-a"]


def test_design_deserialization_rejects_unknown_fields():
    initial = states()
    design = design_for(initial)
    payload = design.to_dict()
    payload["unexpected"] = True
    payload["design_sha256"] = stable_hash(
        {key: value for key, value in payload.items() if key != "design_sha256"}
    )
    try:
        ReplicationDesignBinding.from_dict(payload)
    except ValueError as error:
        assert "fields do not match" in str(error)
    else:
        raise AssertionError("unknown design field was accepted")


def test_design_refuses_changed_initial_state_or_stage_plan():
    initial = states()
    design = design_for(initial)
    changed = replace(initial[0], update_index=1)
    try:
        design.validate_initial_states({0: changed, 1: initial[1]})
    except ValueError as error:
        assert "initial state identity" in str(error)
    else:
        raise AssertionError("changed initial state was accepted")

    changed_stages = list(stages())
    changed_stages[0] = replace(changed_stages[0], updates_per_round=2)
    try:
        design.validate_stage_plan(changed_stages)
    except ValueError as error:
        assert "stage plan" in str(error)
    else:
        raise AssertionError("changed stage plan was accepted")
