"""Host-only scoped bank dispatch; no model evaluation or optimizer calls."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from mooneural.training.generic_role_banks import RoleBank, RoleBankManifest
from mooneural.training.generic_scheduled_role_banks import (
    ScheduledRoleBankRegistry,
    ScheduledRoleBinding,
)
from mooneural.training.generic_training_contracts import (
    EvaluationRequest,
    PolicyView,
    canonical_json,
    stable_hash,
)

TASK_IDS = ("task-a", "task-b")
POLICY = PolicyView((1.0, -2.0), "synthetic-policy")
REGISTRY_HASH = "scheduled_role_registry_hash"
BINDING_HASH = "scheduled_role_binding_hash"
SCOPE = "scheduled_role_scope"


def digest(content):
    return hashlib.sha256(content.encode()).hexdigest()


def manifest(role, tag, seeds, *, include_global=True):
    banks = []
    for seed in seeds:
        payloads = {"local": f"local:{role}:{seed}"}
        if include_global:
            payloads["global"] = f"global:{tag}:{seed}"
        hashes = {family: digest(content) for family, content in payloads.items()}
        banks.append(RoleBank(
            role=role,
            bank_id=f"{role}-{seed}",
            seed=seed,
            target_id=f"{role}-target",
            target_version=f"{tag}-target-v1",
            scale_version=f"{tag}-scales-v1",
            estimator_version="synthetic-mse-v1",
            input_hashes=hashes,
            sample_ids=(f"{role}-local-{seed}",),
            metadata={
                "generator": {"name": "synthetic-host-bytes", "revision": digest("fixture-v1")},
                "inputs": {
                    family: {
                        "path": f"banks/{tag}-{seed}-{family}.bin",
                        "sha256": hashes[family],
                        "bytes": len(content.encode()),
                    }
                    for family, content in payloads.items()
                },
                "fixture_payloads": payloads,
            },
        ))
    return RoleBankManifest(TASK_IDS, tuple(banks), {role: len(banks)})


def binding(
    role="control", *, stage_id="stage-a", arm_id=0, round_number=90,
    min_update=27000, max_update=None, seeds=(101, 102), tag="initial",
):
    return ScheduledRoleBinding(
        stage_id, arm_id, role, round_number, min_update,
        min_update if max_update is None else max_update,
        manifest(role, tag, seeds),
    )


def registry():
    bindings = (
        binding(),
        binding(round_number=91, min_update=27300, tag="round91"),
        binding(arm_id=1, tag="arm1-initial"),
        binding(round_number=110, min_update=33000, tag="stage-a-last"),
        binding(stage_id="stage-b", round_number=110, min_update=33000, tag="stage-b-entry"),
        binding("validation", round_number=None, max_update=33000, seeds=(201, 202), tag="validation-a0"),
        binding("validation", arm_id=1, round_number=None, max_update=33000,
                seeds=(201, 202), tag="validation-a1"),
        binding("validation", stage_id="stage-b", round_number=None, min_update=33000,
                max_update=39000, seeds=(251, 252), tag="validation-b0"),
        binding("certification", stage_id="stage-b", arm_id=4, round_number=None,
                min_update=33000, max_update=39000, seeds=(301, 302), tag="certification-b4"),
    )
    return ScheduledRoleBankRegistry(
        bindings, local_seed_reuse={"control": (101, 102), "validation": (201, 202)},
    )


def coordinates(entry, update_index=None):
    return {
        "stage_id": entry.stage_id,
        "arm_id": entry.arm_id,
        "round_number": entry.round_number,
        "update_index": entry.min_update if update_index is None else update_index,
    }


def request_for(inventory, index=0, *, policy=POLICY, update_index=None):
    entry = inventory.bindings[index]
    return inventory.request_for(entry.role, policy, **coordinates(entry, update_index))


def rehash(payload, field):
    payload[field] = stable_hash({key: value for key, value in payload.items() if key != field})


def copy_global_archives(entry, donors):
    banks = []
    for bank, donor in zip(entry.manifest.banks, donors, strict=True):
        payload = json.loads(canonical_json(bank.to_dict()))
        payload["input_hashes"]["global"] = donor.input_hashes["global"]
        payload["metadata"]["inputs"]["global"] = json.loads(canonical_json(donor.metadata["inputs"]["global"]))
        payload["metadata"]["fixture_payloads"]["global"] = donor.metadata["fixture_payloads"]["global"]
        banks.append(RoleBank.from_dict(payload))
    return replace(entry, manifest=replace(entry.manifest, banks=tuple(banks)))


def test_requests_preserve_the_exact_delegated_inventory_and_provenance():
    inventory = registry()
    assert inventory.task_ids == TASK_IDS
    for index, entry in enumerate(inventory.bindings):
        actual = request_for(inventory, index)
        delegated = entry.manifest.request_for(entry.role, POLICY)
        payload = actual.to_dict()
        assert payload["metadata"].pop(REGISTRY_HASH) == inventory.binding_hash()
        assert payload["metadata"].pop(SCOPE) == {**coordinates(entry), "role": entry.role}
        assert payload == delegated.to_dict()
        for bank in entry.manifest.banks:
            assert actual.metadata["input_hashes"][bank.bank_id] == dict(bank.input_hashes)
        assert actual.metadata["bank_hashes"] == [bank.binding_hash() for bank in entry.manifest.banks]
        inventory.validate_request(actual, POLICY)
        inventory.validate_request_binding(actual)
        inventory.validate_coordinates(actual, **coordinates(entry))


def test_scopes_never_union_different_stage_seed_banks():
    inventory = registry()
    first = request_for(inventory, 5)
    later = request_for(inventory, 7)
    assert first.seeds == (201, 202)
    assert later.seeds == (251, 252)
    assert first.sample_count == later.sample_count == 2
    assert first.metadata[REGISTRY_HASH] == later.metadata[REGISTRY_HASH]
    assert first.metadata["role_bank_manifest_hash"] != later.metadata["role_bank_manifest_hash"]


def test_repeated_local_control_anchors_keep_fresh_global_bytes_and_shared_update_scopes():
    inventory = registry()
    controls = [request_for(inventory, index) for index in range(5)]
    assert all(request.seeds == (101, 102) for request in controls)
    for bank_id in controls[0].metadata["bank_ids"]:
        hashes = [request.metadata["input_hashes"][bank_id] for request in controls]
        assert len({item["local"] for item in hashes}) == 1
        assert len({item["global"] for item in hashes}) == len(controls)
    assert controls[3].metadata[SCOPE]["update_index"] == controls[4].metadata[SCOPE]["update_index"]
    assert controls[3].metadata[SCOPE]["round_number"] == controls[4].metadata[SCOPE]["round_number"]
    assert controls[3].metadata[SCOPE]["stage_id"] != controls[4].metadata[SCOPE]["stage_id"]


@pytest.mark.parametrize("index", (5, 6, 7, 8))
def test_terminal_roles_allow_inclusive_early_stop_update_ranges(index):
    inventory = registry()
    entry = inventory.bindings[index]
    requests = [
        request_for(inventory, index, update_index=update)
        for update in (entry.min_update, entry.min_update + 300, entry.max_update)
    ]
    assert len({request.metadata["role_bank_manifest_hash"] for request in requests}) == 1
    for request in requests:
        assert request.metadata[SCOPE]["round_number"] is None
        inventory.validate_request(request, POLICY)
        inventory.validate_coordinates(
            request, **coordinates(entry, request.metadata[SCOPE]["update_index"]),
        )
    for update in (entry.min_update - 1, entry.max_update + 1):
        with pytest.raises(ValueError, match="outside the scheduled role range"):
            request_for(inventory, index, update_index=update)


def test_control_accepts_only_its_exact_registered_update():
    inventory = registry()
    for update in (26999, 27001):
        with pytest.raises(ValueError, match="outside the scheduled role range"):
            request_for(inventory, update_index=update)


@pytest.mark.parametrize("field,value", (
    ("stage_id", "unknown"), ("stage_id", ""), ("arm_id", 99),
    ("round_number", 99), ("update_index", 27300), ("arm_id", True),
    ("arm_id", 0.0), ("arm_id", "0"), ("round_number", 90.0),
    ("round_number", True), ("update_index", 27000.0), ("update_index", True),
))
def test_request_refuses_unknown_or_wrongly_typed_coordinates(field, value):
    inventory = registry()
    supplied = {**coordinates(inventory.bindings[0]), field: value}
    with pytest.raises(ValueError):
        inventory.request_for("control", POLICY, **supplied)


def test_terminal_request_requires_none_round_and_registered_role():
    inventory = registry()
    with pytest.raises(ValueError, match="round_number must be None"):
        inventory.request_for("validation", POLICY, **{**coordinates(inventory.bindings[5]), "round_number": 110})
    with pytest.raises(ValueError, match="unknown scheduled role"):
        inventory.request_for("training", POLICY, **coordinates(inventory.bindings[0]))


def test_policy_freshness_is_separate_from_bank_binding():
    inventory = registry()
    initial = request_for(inventory)
    changed = replace(POLICY, values=(1.5, -2.0))
    inventory.validate_request_binding(initial)
    with pytest.raises(ValueError, match="policy binding mismatch"):
        inventory.validate_request(initial, changed)
    fresh = request_for(inventory, policy=changed)
    assert fresh.policy_fingerprint != initial.policy_fingerprint
    assert fresh.metadata == initial.metadata
    inventory.validate_request(fresh, changed)


@pytest.mark.parametrize("index,changes", (
    (0, {"arm_id": 1}),
    (0, {"round_number": 91, "update_index": 27300}),
    (3, {"stage_id": "stage-b"}),
    (5, {"update_index": 27300}),
))
def test_caller_cross_check_refuses_otherwise_valid_registered_coordinates(index, changes):
    inventory = registry()
    request = request_for(inventory, index)
    supplied = {**coordinates(inventory.bindings[index]), **changes}
    with pytest.raises(ValueError, match="caller coordinates mismatch"):
        inventory.validate_coordinates(request, **supplied)


def test_early_stop_metadata_requires_cross_check_with_actual_checkpoint_update():
    inventory = registry()
    request = request_for(inventory, 5, update_index=27300)
    request.metadata[SCOPE]["update_index"] = 27600
    inventory.validate_request(request, POLICY)
    with pytest.raises(ValueError, match="caller coordinates mismatch"):
        inventory.validate_coordinates(request, **coordinates(inventory.bindings[5], 27300))


@pytest.mark.parametrize("field,value", (
    ("target_id", "wrong-target"), ("target_version", "wrong-version"),
    ("scale_version", "wrong-scale"), ("estimator_version", "wrong-estimator"),
    ("seeds", (102, 101)), ("seeds", (101, 103)),
    ("sample_count", 1), ("task_ids", tuple(reversed(TASK_IDS))),
    ("anchor_ids", ("other-anchor",)), ("seed", 101), ("cells", ("extra-cell",)),
))
def test_request_binding_refuses_changed_delegated_fields(field, value):
    inventory = registry()
    request = replace(request_for(inventory), **{field: value})
    with pytest.raises(ValueError, match="request binding mismatch"):
        inventory.validate_request_binding(request)


@pytest.mark.parametrize("field", (
    "role_bank_manifest_hash", "bank_ids", "bank_hashes", "input_hashes", "sample_ids_by_bank",
))
def test_request_binding_refuses_swapped_manifest_metadata(field):
    inventory = registry()
    request = request_for(inventory)
    donor = request_for(inventory, 5)
    request.metadata[field] = deepcopy(donor.metadata[field])
    with pytest.raises(ValueError, match="request binding mismatch"):
        inventory.validate_request_binding(request)


@pytest.mark.parametrize("mutation", (
    "missing_scope", "extra_scope", "wrong_role", "float_arm", "float_update",
    "bool_round", "wrong_registry", "missing_registry", "extra_metadata", "stale_control_update",
))
def test_request_binding_refuses_scope_or_registry_metadata_drift(mutation):
    inventory = registry()
    request = request_for(inventory)
    if mutation == "missing_scope":
        request.metadata.pop(SCOPE)
    elif mutation == "extra_scope":
        request.metadata[SCOPE]["extra"] = "unbound"
    elif mutation == "wrong_role":
        request.metadata[SCOPE]["role"] = "validation"
    elif mutation == "float_arm":
        request.metadata[SCOPE]["arm_id"] = 0.0
    elif mutation == "float_update":
        request.metadata[SCOPE]["update_index"] = 27000.0
    elif mutation == "bool_round":
        request.metadata[SCOPE]["round_number"] = True
    elif mutation == "wrong_registry":
        request.metadata[REGISTRY_HASH] = digest("wrong-registry")
    elif mutation == "missing_registry":
        request.metadata.pop(REGISTRY_HASH)
    elif mutation == "extra_metadata":
        request.metadata["unregistered"] = True
    else:
        request.metadata[SCOPE]["update_index"] = 27001
    with pytest.raises(ValueError):
        inventory.validate_request_binding(request)


def test_unscoped_requests_and_foreign_registry_requests_are_rejected():
    inventory = registry()
    delegated = inventory.bindings[0].manifest.request_for("control", POLICY)
    with pytest.raises(ValueError, match="scope metadata"):
        inventory.validate_request_binding(delegated)
    other = replace(inventory, registry_version="another-inventory")
    with pytest.raises(ValueError, match="registry binding mismatch"):
        other.validate_request_binding(request_for(inventory))


def test_registry_and_nested_provenance_are_immutable_and_to_dict_is_detached():
    bindings = [binding()]
    inventory = ScheduledRoleBankRegistry(bindings)
    original = inventory.to_dict()
    bindings.clear()
    assert len(inventory.bindings) == 1
    with pytest.raises(FrozenInstanceError):
        inventory.registry_version = "changed"
    with pytest.raises(FrozenInstanceError):
        inventory.bindings[0].min_update = 0
    with pytest.raises(TypeError):
        inventory.local_seed_reuse["control"] = (101,)
    bank = inventory.bindings[0].manifest.banks[0]
    with pytest.raises(TypeError):
        bank.metadata["inputs"]["global"]["sha256"] = digest("changed")
    detached = inventory.to_dict()
    detached["bindings"][0]["manifest"]["banks"][0]["metadata"]["inputs"]["global"]["bytes"] = 0
    detached["bindings"].clear()
    assert inventory.to_dict() == original


def test_declared_reuse_is_snapshotted_and_all_hashes_survive_real_host_byte_reload(tmp_path):
    original = registry()
    declarations = {role: list(seeds) for role, seeds in original.local_seed_reuse.items()}
    inventory = ScheduledRoleBankRegistry(original.bindings, declarations)
    declarations["control"].append(999)
    assert inventory.binding_hash() == original.binding_hash()
    for entry in inventory.bindings:
        for bank in entry.manifest.banks:
            for family, record in bank.metadata["inputs"].items():
                target = tmp_path / record["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bank.metadata["fixture_payloads"][family].encode())
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(inventory.to_dict(), indent=2))
    restored = ScheduledRoleBankRegistry.from_dict(json.loads(path.read_text()))
    assert restored.to_dict() == inventory.to_dict()
    assert restored.binding_hash() == inventory.binding_hash()
    for index, entry in enumerate(restored.bindings):
        assert ScheduledRoleBinding.from_dict(entry.to_dict()).binding_hash() == entry.binding_hash()
        assert request_for(restored, index).to_dict() == request_for(inventory, index).to_dict()
        for bank in entry.manifest.banks:
            for family, record in bank.metadata["inputs"].items():
                content = (tmp_path / record["path"]).read_bytes()
                assert len(content) == record["bytes"]
                assert hashlib.sha256(content).hexdigest() == bank.input_hashes[family] == record["sha256"]


@pytest.mark.parametrize("mutation", ("schema", "missing", "extra", "hash", "reorder", "drop", "nested"))
def test_registry_reload_refuses_schema_inventory_and_nested_hash_changes(mutation):
    payload = registry().to_dict()
    if mutation == "schema":
        payload["schema"] = "wrong-schema"
        rehash(payload, REGISTRY_HASH)
    elif mutation == "missing":
        payload.pop("local_seed_reuse")
        rehash(payload, REGISTRY_HASH)
    elif mutation == "extra":
        payload["unbound"] = True
        rehash(payload, REGISTRY_HASH)
    elif mutation == "hash":
        payload[REGISTRY_HASH] = digest("wrong")
    elif mutation == "reorder":
        payload["bindings"].reverse()
    elif mutation == "drop":
        payload["bindings"].pop()
    else:
        payload["bindings"][0]["manifest"]["banks"][0]["input_hashes"]["global"] = digest("swapped")
        rehash(payload, REGISTRY_HASH)
        rehash(payload["bindings"][0], BINDING_HASH)
        rehash(payload, REGISTRY_HASH)
    with pytest.raises(ValueError):
        ScheduledRoleBankRegistry.from_dict(payload)


@pytest.mark.parametrize("mutation", ("schema", "missing", "extra", "hash", "float_count", "normalized_hash"))
def test_binding_reload_refuses_noncanonical_fields_and_hashes(mutation):
    payload = binding().to_dict()
    if mutation == "schema":
        payload["schema"] = "wrong-schema"
    elif mutation == "missing":
        payload.pop("max_update")
    elif mutation == "extra":
        payload["extra"] = True
    elif mutation == "hash":
        payload[BINDING_HASH] = digest("wrong")
    elif mutation == "float_count":
        payload["min_update"] = 27000.0
    else:
        bank = payload["manifest"]["banks"][0]
        bank["input_hashes"]["global"] = bank["input_hashes"]["global"].upper()
        rehash(payload["manifest"], "role_bank_manifest_hash")
    if mutation != "hash":
        rehash(payload, BINDING_HASH)
    with pytest.raises(ValueError):
        ScheduledRoleBinding.from_dict(payload)


@pytest.mark.parametrize("changes", (
    {"stage_id": ""}, {"arm_id": -1}, {"arm_id": True},
    {"arm_id": 0.0}, {"round_number": None}, {"round_number": True},
    {"min_update": -1}, {"min_update": True}, {"max_update": 27000.0},
    {"min_update": 27001}, {"max_update": 27001},
))
def test_binding_refuses_invalid_scope_and_nonexact_control_bounds(changes):
    with pytest.raises(ValueError):
        replace(binding(), **changes)


def test_binding_rejects_unmaterialized_wrong_role_and_multirole_manifests():
    entry = binding()
    with pytest.raises(TypeError, match="materialized RoleBankManifest"):
        replace(entry, manifest={"arrays_generated": False})
    with pytest.raises(ValueError, match="exact global"):
        replace(entry, manifest=manifest("control", "declaration", (101, 102), include_global=False))
    with pytest.raises(ValueError, match="exactly its bound role"):
        replace(entry, manifest=manifest("validation", "wrong-role", (201, 202)))
    both = RoleBankManifest(
        TASK_IDS, (*entry.manifest.banks, *manifest("validation", "multi-role", (201, 202)).banks),
        {"control": 2, "validation": 2},
    )
    with pytest.raises(ValueError, match="exactly its bound role"):
        replace(entry, manifest=both)


def test_registry_rejects_empty_inventory_duplicate_scope_manifest_and_task_drift():
    with pytest.raises(ValueError, match="requires ScheduledRoleBinding"):
        ScheduledRoleBankRegistry(())
    with pytest.raises(ValueError, match="duplicate scheduled scope"):
        ScheduledRoleBankRegistry((binding(), binding(tag="different-manifest")))
    with pytest.raises(ValueError, match="distinct role-bank manifests"):
        ScheduledRoleBankRegistry((binding(), replace(binding(), arm_id=1)))
    other = binding(arm_id=1, tag="different-arm")
    other = replace(other, manifest=replace(other.manifest, task_ids=tuple(reversed(TASK_IDS))))
    with pytest.raises(ValueError, match="task order mismatch"):
        ScheduledRoleBankRegistry((binding(), other))


@pytest.mark.parametrize("declarations", ({}, {"control": (101,)}, {"control": (101, 102, 999)}, {"control": (101, 101)}))
def test_same_role_reuse_requires_exact_explicit_declaration(declarations):
    entries = (binding(), binding(round_number=91, min_update=27300, tag="round91"))
    with pytest.raises(ValueError, match="reus"):
        ScheduledRoleBankRegistry(entries, declarations)


def test_unused_and_wrongly_typed_reuse_declarations_are_rejected():
    for declaration in ({"control": (101,)}, {"control": (True,)}, {"control": (101.0,)}, {"training": (101,)}):
        with pytest.raises(ValueError):
            ScheduledRoleBankRegistry((binding(),), declaration)


def test_cross_role_seed_and_sample_reuse_cannot_be_declared_away():
    control = binding()
    validation = binding("validation", round_number=None, seeds=(101, 102), tag="validation")
    with pytest.raises(ValueError, match="local seeds must not overlap different roles"):
        ScheduledRoleBankRegistry((control, validation), {"control": (101, 102), "validation": (101, 102)})
    validation = binding("validation", round_number=None, seeds=(201, 202), tag="validation")
    banks = validation.manifest.banks
    validation = replace(validation, manifest=replace(
        validation.manifest,
        banks=(replace(banks[0], sample_ids=control.manifest.banks[0].sample_ids), banks[1]),
    ))
    with pytest.raises(ValueError, match="sample IDs must not overlap different roles"):
        ScheduledRoleBankRegistry((control, validation))


@pytest.mark.parametrize("change", ("missing_local", "different_local", "reused_global"))
def test_reused_control_seed_requires_identical_local_and_fresh_global_hashes(change):
    first = binding()
    second = binding(round_number=91, min_update=27300, tag="round91")
    bank = second.manifest.banks[0]
    hashes = dict(bank.input_hashes)
    if change == "missing_local":
        hashes.pop("local")
    elif change == "different_local":
        hashes["local"] = digest("changed-local")
    else:
        hashes["global"] = first.manifest.banks[0].input_hashes["global"]
    second = replace(second, manifest=replace(
        second.manifest,
        banks=(replace(bank, input_hashes=hashes), *second.manifest.banks[1:]),
    ))
    with pytest.raises(ValueError, match="local input hashes|fresh global input hashes"):
        ScheduledRoleBankRegistry((first, second), {"control": (101, 102)})


@pytest.mark.parametrize("seeds", ((101, 102), (201, 202)))
@pytest.mark.parametrize("reverse_scopes", (False, True))
def test_control_rejects_permuted_globals_under_reused_or_fresh_local_seeds(seeds, reverse_scopes):
    first = binding()
    second = binding(round_number=91, min_update=27300, seeds=seeds, tag="round91")
    second = copy_global_archives(second, reversed(first.manifest.banks))
    assert {bank.input_hashes["global"] for bank in first.manifest.banks} == {
        bank.input_hashes["global"] for bank in second.manifest.banks
    }
    for bank in second.manifest.banks:
        assert digest(bank.metadata["fixture_payloads"]["global"]) == bank.input_hashes["global"]
        assert bank.metadata["inputs"]["global"]["sha256"] == bank.input_hashes["global"]
    entries = (second, first) if reverse_scopes else (first, second)
    reuse = {"control": seeds} if seeds == (101, 102) else {}
    with pytest.raises(ValueError, match="fresh global input hashes"):
        ScheduledRoleBankRegistry(entries, reuse)


@pytest.mark.parametrize("scope_change", ({"stage_id": "stage-b"}, {"arm_id": 1}))
def test_control_global_freshness_covers_distinct_stages_and_arms(scope_change):
    first = binding()
    second = copy_global_archives(binding(tag="other-scope", **scope_change), first.manifest.banks)
    with pytest.raises(ValueError, match="fresh global input hashes"):
        ScheduledRoleBankRegistry((first, second), {"control": (101, 102)})


def test_control_can_share_global_archive_among_local_banks_in_one_scope():
    entry = binding()
    shared = copy_global_archives(entry, (entry.manifest.banks[0],) * 2)
    inventory = ScheduledRoleBankRegistry((shared,))
    request = request_for(inventory)
    assert request.seeds == (101, 102)
    assert len({hashes["global"] for hashes in request.metadata["input_hashes"].values()}) == 1
    inventory.validate_request(request, POLICY)


@pytest.mark.parametrize("role", ("validation", "certification"))
def test_terminal_global_reuse_across_scopes_remains_permitted(role):
    first = binding(role, round_number=None, seeds=(201, 202), tag="terminal-a")
    second = binding(role, stage_id="stage-b", round_number=None, min_update=33000,
                     max_update=39000, seeds=(201, 202), tag="terminal-b")
    second = copy_global_archives(second, first.manifest.banks)
    inventory = ScheduledRoleBankRegistry((first, second), {role: (201, 202)})
    assert request_for(inventory, 0).metadata["input_hashes"] == request_for(inventory, 1).metadata["input_hashes"]
    for index, entry in enumerate(inventory.bindings):
        request = request_for(inventory, index)
        inventory.validate_request(request, POLICY)
        inventory.validate_coordinates(request, **coordinates(entry))


@pytest.mark.parametrize("role", ("validation", "certification"))
def test_control_global_freshness_does_not_add_cross_role_byte_restrictions(role):
    control = binding()
    terminal = binding(role, round_number=None, seeds=(201, 202), tag="terminal")
    terminal = copy_global_archives(terminal, control.manifest.banks)
    inventory = ScheduledRoleBankRegistry((control, terminal))
    for index in range(2):
        inventory.validate_request(request_for(inventory, index), POLICY)


def test_validation_local_seed_reuse_across_stages_is_explicitly_allowed():
    first = binding("validation", round_number=None, seeds=(201, 202), tag="validation-a")
    second = binding("validation", stage_id="stage-b", round_number=None,
                     min_update=33000, max_update=39000, seeds=(201, 202), tag="validation-b")
    inventory = ScheduledRoleBankRegistry((first, second), {"validation": (201, 202)})
    assert request_for(inventory, 0).seeds == request_for(inventory, 1).seeds
    assert request_for(inventory, 0).metadata["input_hashes"] != request_for(inventory, 1).metadata["input_hashes"]


def test_request_round_trip_preserves_exact_coordinates_and_immutability_boundary():
    inventory = registry()
    request = request_for(inventory)
    restored = EvaluationRequest.from_dict(json.loads(json.dumps(request.to_dict())))
    inventory.validate_request(restored, POLICY)
    inventory.validate_coordinates(restored, **coordinates(inventory.bindings[0]))
    before = inventory.to_dict()
    restored.metadata["input_hashes"]["control-101"]["global"] = digest("changed")
    with pytest.raises(ValueError, match="request binding mismatch"):
        inventory.validate_request_binding(restored)
    assert inventory.to_dict() == before


def test_only_host_contracts_are_imported():
    script = (
        "import sys\n"
        "from mooneural.training.generic_scheduled_role_banks import ScheduledRoleBankRegistry\n"
        "assert not any(name.split('.')[0] in ('tensorflow', 'jax', 'torch', 'numpy') "
        "or name.startswith('dsge_hmc.models') for name in sys.modules)\n"
    )
    process = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10, check=False)
    assert process.returncode == 0, process.stderr
