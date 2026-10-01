"""Manufactured grouped banks; no model, optimizer or reference solves."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

from mooneural.training import generic_independent_role_bridge as module
from mooneural.training.generic_independent_role_bridge import (
    GroupedBankLossRows,
    GroupedRoleSpec,
    IndependentBankRoleBridge,
    LossCoordinates,
)
from mooneural.training.generic_replication_training_protocol import publish, reference
from mooneural.training.generic_role_banks import RoleBank, RoleBankManifest
from mooneural.training.generic_scheduled_role_banks import (
    ScheduledRoleBankRegistry,
    ScheduledRoleBinding,
)
from mooneural.training.generic_training_contracts import (
    CertificationResult,
    PolicyView,
    stable_hash,
)

TASKS = tuple(f"task-{index}" for index in range(9))
CELLS = tuple(f"cell-{index}" for index in range(9))
PROBABILITIES = tuple(np.outer([3 / 8, 1 / 4, 3 / 8], [3 / 8, 1 / 4, 3 / 8]).reshape(-1))
POLICY = PolicyView((1.,), "manufactured")


def fixture(root, *, mode="stratified", duplicate_rows=1, roles=("control", "validation", "certification"),
            scope="manufactured", dynamic=False):
    coordinates = LossCoordinates(TASKS, (2.,) * 9, (4.,) * 9, (4.,) * 9, (4.,) * 9, 1.)
    families = {f"{role}-{family}-{index}": family for role in roles
                for family, count in (("global", 3), ("local", 4)) for index in range(count)}
    grouping = GroupedRoleSpec(TASKS, CELLS, TASKS[:6], TASKS[6:], PROBABILITIES, mode, families,
                               .02, .03, 3, 4, "manufactured conservative raw losses; no error-to-truth claim")
    records, entries = [], []
    callback = FixtureProvider()
    contract = None
    if dynamic:
        source = reference(root, Path(__file__))
        contract = {"callback": {"module": FixtureProvider.__module__, "qualname": "FixtureProvider.__call__", "source": source},
                    "source_references": [source], "policy_dimension": 1, "policy_metadata": {},
                    "coordinates_hash": coordinates.binding_hash(), "grouping_hash": grouping.binding_hash()}
    for role_index, role in enumerate(roles):
        banks = []
        for ordinal, (bank_id, family) in enumerate(pair for pair in families.items() if pair[0].startswith(role + "-")):
            index = int(bank_id.rsplit("-", 1)[1])
            by_cell = family == "global" and (role != "validation" or mode == "stratified")
            cell_rows = tuple(cell for cell_index, cell in enumerate(CELLS)
                              for _ in range((1 + cell_index % 2) * duplicate_rows)) if by_cell else ()
            count = len(cell_rows) if by_cell else 2 * duplicate_rows
            row_ids = tuple(f"{bank_id}-row-{number}" for number in range(count))
            tasks = TASKS[:6] if family == "global" else TASKS[6:]
            central = np.full((count, len(tasks)), .1 + .002 * index if family == "global" else .2 + .02 * index)
            if by_cell:
                central[np.array(cell_rows) == CELLS[-1], 0] += 1.1
            elif family == "global":
                central[:, 0] += 1.1 * PROBABILITIES[-1]
            raw = central * 4.
            path = root / f"{bank_id}-inputs.json"
            publish(path, {"bank_id": bank_id, "rows": row_ids, "cells": cell_rows})
            input_ref = reference(root, path)
            sources = (input_ref,) if not dynamic else (input_ref, *contract["source_references"])
            record = GroupedBankLossRows(bank_id, row_ids, POLICY.fingerprint(), tasks, raw, raw + .04, sources, family, cell_rows)
            records.append(record)
            metadata = {"row_ids": row_ids, "row_count": count, "row_cell_ids": cell_rows, "family": family,
                        "grouping_hash": grouping.binding_hash(), "evidence_scope": scope}
            if dynamic:
                metadata.update(provider_binding_hash=stable_hash(contract), input_references={"global": input_ref})
            else:
                metadata["policy_row_hashes"] = {POLICY.fingerprint(): record.binding_hash()}
            banks.append(RoleBank(role, bank_id, 100 * (role_index + 1) + ordinal, "manufactured", "v1",
                coordinates.binding_hash(), "grouped-bank-upper-v1", {"global": input_ref["sha256"]}, (bank_id,), metadata))
        entries.append(ScheduledRoleBinding("test", 0, role, 0 if role == "control" else None, 0, 0,
            RoleBankManifest(TASKS, tuple(banks), {role: len(banks)})))
    callback.records = {record.bank_id: record for record in records}
    registry = ScheduledRoleBankRegistry(tuple(entries))
    bridge = IndependentBankRoleBridge(root, registry, coordinates, () if dynamic else records, evidence_scope=scope,
        grouping=grouping, row_provider=callback if dynamic else None, provider_binding=contract,
        cache_directory=root / "cache")
    return bridge, records, callback


class FixtureProvider:
    def __init__(self):
        self.calls, self.forbidden, self.interrupt = 0, False, False

    def __call__(self, policy, request, bank, *, output_directory):
        if self.forbidden:
            raise AssertionError("completed grouped rows were called again")
        self.calls += 1
        if self.interrupt:
            raise RuntimeError("manufactured interrupted provider")
        return replace(self.records[bank.bank_id], policy_fingerprint=policy.fingerprint())


def request(bridge, role="control", policy=POLICY):
    return bridge.registry.request_for(role, policy, stage_id="test", arm_id=0,
                                       round_number=0 if role == "control" else None, update_index=0)


def test_failing_cell_survives_and_certification_retains_57(tmp_path):
    bridge, _records, _callback = fixture(tmp_path)
    control = bridge.evaluate(POLICY, request(bridge))
    validation = bridge.evaluate(POLICY, request(bridge, "validation"))
    terminal = bridge.evaluate(POLICY, request(bridge, "certification"))
    assert control.task_upper_mse[TASKS[0]] > 1. and validation.task_mean_mse[TASKS[0]] < 1.
    assert validation.task_mean_mse[TASKS[0]] == pytest.approx(.102 + 1.1 * PROBABILITIES[-1])
    assert validation.task_mean_mse[TASKS[0]] != pytest.approx(.102 + 1.1 / 9)
    assert len(terminal.conjuncts) == 57 and sum(not value for value in terminal.conjuncts.values()) == 1
    assert not terminal.conjuncts[bridge.grouping.conjunct("global", TASKS[0], CELLS[-1])]
    assert terminal.hard_vetoes["scientific_qualification"] is False and terminal.passed is False
    assert CertificationResult.from_dict(json.loads(json.dumps(terminal.to_dict()))).to_dict() == terminal.to_dict()


def test_independent_family_counts_alpha_and_rows(tmp_path):
    results = []
    for duplicate in (1, 3):
        bridge, _records, _callback = fixture(tmp_path / str(duplicate), duplicate_rows=duplicate)
        result = bridge.evaluate(POLICY, request(bridge))
        summary = result.raw_records
        assert summary["bank_count"] == {"global": 3, "local": 4}
        assert summary["family_union_alpha_bound"] == pytest.approx(.05)
        for family, count, alpha, multiplicity in (("global", 3, .02, 54), ("local", 4, .03, 3)):
            quality = summary["quality"][family]
            assert quality["degrees_of_freedom"] == count - 1 and len(quality["tasks"]) == multiplicity
            for row in quality["tasks"].values():
                assert row["sample_count"] == count
                assert row["critical_value"] == student_t.ppf(1. - alpha / multiplicity, count - 1)
                assert row["standard_error_mse"] == pytest.approx(np.std(row["conservative_anchor_samples"], ddof=1) / np.sqrt(count))
            assert quality["threshold_definition"]["mse_matches_training_objective_coordinate"] is False
        results.append(summary["quality"])
    for family in ("global", "local"):
        for task in results[0][family]["tasks"]:
            for name in ("mean_normalized_mse", "upper_normalized_mse", "standard_error_mse"):
                assert results[0][family]["tasks"][task][name] == pytest.approx(results[1][family]["tasks"][task][name])


def test_direct_and_stratified_validation_have_same_whole_law_mean(tmp_path):
    values = []
    for mode in ("stratified", "whole-law"):
        bridge, _records, _callback = fixture(tmp_path / mode, mode=mode)
        result = bridge.evaluate(POLICY, request(bridge, "validation"))
        values.append(result.task_mean_mse)
        assert result.sample_counts == dict.fromkeys(TASKS[:6], 3) | dict.fromkeys(TASKS[6:], 4)
    assert dict(values[0]) == pytest.approx(dict(values[1]))


@pytest.mark.parametrize("fault", ("duplicate_cells", "missing_partition", "probabilities", "family_alpha"))
def test_group_spec_invalid(tmp_path, fault):
    bridge, _records, _callback = fixture(tmp_path)
    changes = {"duplicate_cells": {"cell_ids": ("same",) * 9}, "missing_partition": {"local_task_ids": TASKS[7:]},
               "probabilities": {"cell_probabilities": (1.,) * 9}, "family_alpha": {"global_alpha": 1.}}
    with pytest.raises(ValueError):
        replace(bridge.grouping, **changes[fault])


@pytest.mark.parametrize("fault", ("nonfinite", "below_central", "shape", "local_cells", "duplicate_rows"))
def test_group_rows_invalid(tmp_path, fault):
    _bridge, records, _callback = fixture(tmp_path)
    original = records[0] if fault != "local_cells" else records[3]
    changes = {"nonfinite": {"raw_mse": np.full(original.raw_mse.shape, np.nan)},
               "below_central": {"conservative_raw_mse": original.raw_mse - .1}, "shape": {"raw_mse": [[1.]]},
               "local_cells": {"row_cell_ids": (CELLS[0],) * len(original.row_ids)},
               "duplicate_rows": {"row_ids": ("same",) * len(original.row_ids)}}
    with pytest.raises(ValueError):
        replace(original, **changes[fault])


def test_missing_cells_changed_sources_and_grouping_refuse(tmp_path, monkeypatch):
    bridge, records, _callback = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid records reached estimator")
    monkeypatch.setattr(module, "evaluate_upper_mean_mse", forbidden)
    original = records[0]
    bridge.records[(POLICY.fingerprint(), original.bank_id)] = replace(original, row_cell_ids=(CELLS[0],) * len(original.row_ids))
    with pytest.raises(ValueError, match="cell"):
        bridge.evaluate(POLICY, request(bridge))
    bridge.records[(POLICY.fingerprint(), original.bank_id)] = original
    changed = replace(bridge.grouping, cell_probabilities=(1. / 9,) * 9)
    with pytest.raises(ValueError, match="grouping"):
        IndependentBankRoleBridge(tmp_path, bridge.registry, bridge.coordinates, records, evidence_scope="manufactured", grouping=changed)
    (tmp_path / f"{original.bank_id}-inputs.json").write_text("changed")
    with pytest.raises(ValueError):
        bridge.evaluate(POLICY, request(bridge))


def test_grouped_saved_completed_cache_is_no_call(tmp_path, monkeypatch):
    bridge, records, _callback = fixture(tmp_path)
    expected = bridge.evaluate(POLICY, request(bridge))
    restored = IndependentBankRoleBridge(tmp_path, bridge.registry, bridge.coordinates, records,
        evidence_scope="manufactured", grouping=bridge.grouping, cache_directory=tmp_path / "cache")
    monkeypatch.setattr(module, "evaluate_upper_mean_mse", lambda *args, **kwargs: pytest.fail("repeated estimator"))
    assert restored.evaluate(POLICY, request(restored)).to_dict() == expected.to_dict()
    assert restored.cache_hits == 1 and restored.estimator_calls == 0


@pytest.mark.parametrize("scope", ("exposed-diagnostic", "fresh-engineering"))
def test_engineering_terminal_refused(tmp_path, scope):
    with pytest.raises(ValueError, match="control-only|terminal|fresh engineering"):
        fixture(tmp_path, scope=scope)


def test_grouped_dynamic_recovery_policy_and_interrupted_intent(monkeypatch):
    import tempfile

    root = Path(__file__).resolve().parents[2]
    (root / ".pytest_cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grouped-roles-", dir=root / ".pytest_cache") as directory:
        current = Path(directory)
        bridge, _records, callback = fixture(current, roles=("control",), dynamic=False)
        source = reference(root, Path(__file__))
        contract = {"callback": {"module": FixtureProvider.__module__, "qualname": "FixtureProvider.__call__", "source": source},
                    "source_references": [source], "policy_dimension": 1, "policy_metadata": {},
                    "coordinates_hash": bridge.coordinates.binding_hash(), "grouping_hash": bridge.grouping.binding_hash()}
        entries = []
        for entry in bridge.registry.bindings:
            banks = []
            for bank in entry.manifest.banks:
                record = callback.records[bank.bank_id]
                input_ref = reference(root, current / f"{bank.bank_id}-inputs.json")
                callback.records[bank.bank_id] = replace(record, source_references=(input_ref, source))
                metadata = {key: value for key, value in bank.metadata.items() if key != "policy_row_hashes"}
                metadata.update(provider_binding_hash=stable_hash(contract), input_references={"global": input_ref})
                banks.append(replace(bank, metadata=metadata))
            entries.append(replace(entry, manifest=replace(entry.manifest, banks=tuple(banks))))
        registry = ScheduledRoleBankRegistry(tuple(entries))
        dynamic = IndependentBankRoleBridge(root, registry, bridge.coordinates, evidence_scope="manufactured",
            row_provider=callback, provider_binding=contract, grouping=bridge.grouping, cache_directory=current / "dynamic")
        first = dynamic.evaluate(POLICY, request(dynamic))
        assert callback.calls == 7 and dynamic.estimator_calls == 2
        callback.forbidden = True
        assert dynamic.evaluate(POLICY, request(dynamic)).to_dict() == first.to_dict()
        assert dynamic.row_cache_hits == 7
        callback.forbidden, callback.interrupt = False, True
        other = replace(POLICY, values=(2.,))
        with pytest.raises(RuntimeError, match="interrupted"):
            dynamic.evaluate(other, request(dynamic, policy=other))
        before = callback.calls
        with pytest.raises(ValueError, match="intent is incomplete"):
            dynamic.evaluate(other, request(dynamic, policy=other))
        assert callback.calls == before
        terminal_banks = tuple(replace(bank, role="certification", metadata={**bank.metadata, "evidence_scope": "fresh-engineering"})
                               for bank in entries[0].manifest.banks)
        terminal_entry = replace(entries[0], role="certification", round_number=None,
            manifest=RoleBankManifest(TASKS, terminal_banks, {"certification": len(terminal_banks)}))
        with pytest.raises(ValueError, match="terminal is not qualified"):
            IndependentBankRoleBridge(root, ScheduledRoleBankRegistry((terminal_entry,)), bridge.coordinates,
                evidence_scope="fresh-engineering", row_provider=callback, provider_binding=contract,
                grouping=bridge.grouping, cache_directory=current / "terminal")
