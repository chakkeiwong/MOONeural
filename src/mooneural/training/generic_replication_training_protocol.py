"""Protocol orchestration over the existing replication loop and durable store."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

from .generic_replication_design import ReplicationDesignBinding
from .generic_training_contracts import canonical_json, stable_hash

MANIFEST_SCHEMA = "dsge_hmc.replication_training_protocol.v1"
RESULT_SCHEMA = "dsge_hmc.replication_training_stage.v1"
REVIEW_SCHEMA = "dsge_hmc.replication_training_stage_review.v1"
COMPARISON_CHECKS = (
    "values", "gradients", "selected_method", "projections", "policy", "adam", "method_rates",
)
STAGES = ("one", "ten", "boundary", "full")


class TrainingProtocolError(ValueError):
    """A declared training identity or prerequisite is not satisfied."""


def require(condition, message):
    if not condition:
        raise TrainingProtocolError(message)


def reference(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    require(path.is_relative_to(root), "artifact must remain inside its root")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(root)), "sha256": digest.hexdigest()}


def verified_path(root, binding):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "path/hash reference required")
    relative = Path(binding["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "root-relative evidence path required")
    path = Path(root).resolve() / relative
    require(reference(root, path) == binding, f"artifact changed: {relative}")
    return path


def document(root, binding):
    return json.loads(verified_path(root, binding).read_text())


def publish(path, payload):
    """Publish once; identical retries reuse bytes after process interruption."""
    path = Path(path)
    content = (canonical_json(payload) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-protocol-", delete=False) as stream:
        pending = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(pending, path)
        except FileExistsError:
            require(path.read_bytes() == content, f"refusing to overwrite {path}")
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        pending.unlink()


def load_callable(root, binding):
    path = verified_path(root, binding["source"])
    name = "replication_protocol_" + binding["source"]["sha256"]
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "callable source is not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callback = getattr(module, binding["callable"], None)
    require(callable(callback), "declared runtime callable is missing")
    return callback


def calibration_binding(payload):
    from .generic_replication_calibration import SourceCalibrationBinding
    from .generic_replication_statistics import ReplicationStatisticsSpec

    supplied = payload["spec"]
    spec = ReplicationStatisticsSpec(**{field.name: supplied[field.name] for field in fields(ReplicationStatisticsSpec)})
    binding = SourceCalibrationBinding(
        spec, payload["source_global_update"], payload["provider_sha256"],
        payload["estimator_sha256"], payload["backend_sha256"],
    )
    require(binding.to_dict() == payload, "source calibration binding is not canonical")
    return binding


def training_registration(manifest):
    """Freeze launcher inputs before adding references derived from final design."""
    derived = {"design", "design_sha256", "source_calibration", "source_calibration_binding"}
    return {name: value for name, value in manifest.items() if name not in derived}


def load_manifest(root, binding):
    manifest = document(root, binding)
    require(manifest.get("schema") == MANIFEST_SCHEMA, "unknown training manifest")
    design = ReplicationDesignBinding.from_dict(document(root, manifest["design"]))
    require(design.binding_hash() == manifest["design_sha256"], "final design identity mismatch")
    require(canonical_json(design.permanent_pass_binding.get("training_protocol"))
            == canonical_json(training_registration(manifest)), "launcher inputs differ from frozen final design")
    require(manifest["source_manifest"]["sha256"] == design.source_manifest_sha256, "source manifest identity mismatch")
    for item in (manifest["source_manifest"], manifest["role_registry"], manifest["schedule"],
                 manifest["selector"], manifest["tolerances"], *manifest["sources"]):
        verified_path(root, item)
    source_hashes = {item["path"]: item["sha256"] for item in manifest["sources"]}
    require(all(source_hashes.get(path) == digest for path, digest in design.generic_code_hashes.items()),
            "generic design code hashes must bind actual declared source paths")
    for name in ("runtime_factory", "one_update_comparator", "source_calibration_stage"):
        item = manifest[name]
        require(set(item) == {"source", "callable"} and isinstance(item["callable"], str)
                and bool(item["callable"]), f"{name} callable binding required")
        verified_path(root, item["source"])
        require(source_hashes.get(item["source"]["path"]) == item["source"]["sha256"], "callable source not inventoried")
    expected = calibration_binding(manifest["source_calibration_binding"])
    require(expected.spec.design_sha256 == design.binding_hash(), "calibration must bind the same final design")
    verified_path(root, manifest["source_calibration"])
    require(manifest["selector"]["path"] in source_hashes
            and source_hashes[manifest["selector"]["path"]] == manifest["selector"]["sha256"], "source selector is not inventoried")
    require(set(manifest["limits"]) == set(STAGES), "resource limits must cover the complete ladder")
    for limit in manifest["limits"].values():
        require(set(limit) == {"seconds", "memory_bytes", "cpus", "threads", "device"}, "complete native resource limits required")
        require(all(type(limit[name]) is int and limit[name] > 0 for name in ("seconds", "memory_bytes", "threads")),
                "positive resource limits required")
        require(isinstance(limit["cpus"], list) and len(limit["cpus"]) > 0
                and len(set(limit["cpus"])) == len(limit["cpus"])
                and all(type(cpu) is int and cpu >= 0 for cpu in limit["cpus"]), "explicit CPU affinity required")
        require(limit["device"] in ("-1", "0", "1"), "one explicit device or hidden GPUs required")
    return manifest, design


def require_registered_program(root, manifest, design, *, continuing_attempt_id=None):
    from .generic_program import Program
    from .generic_replication_master import replication_master_hash

    program = Program(Path(root))
    step = program.require_step(
        "p5r.training", plan_hash=replication_master_hash(program, design), execution=True,
        continuing_attempt_id=continuing_attempt_id,
    )
    require("p5r.design" in program.complete, "replication design is still open")
    require(step["entrypoint"] == manifest["entrypoint"]["path"], "registered training entrypoint differs")
    verified_path(root, manifest["entrypoint"])
    return program, step


def require_stage_reviews(root, manifest_hash, stage, reviews, results, executor_id):
    require(stage in STAGES, "unknown training stage")
    required = STAGES[:STAGES.index(stage)]
    for previous in required:
        require(previous in reviews and previous in results, f"reviewed {previous} canary required")
        result_ref, result = results[previous]
        require(result.get("status") == "PASS" and result.get("manifest_sha256") == manifest_hash,
                f"{previous} canary did not pass under this manifest")
        for item in result.get("evidence", ()):
            verified_path(root, item)
        review = document(root, reviews[previous])
        require(review.get("schema") == REVIEW_SCHEMA and review.get("decision") == "PASS"
                and review.get("stage") == previous and review.get("result") == result_ref
                and review.get("manifest_sha256") == manifest_hash, f"{previous} independent review mismatch")
        require(isinstance(review.get("reviewer_id"), str) and bool(review["reviewer_id"])
                and review["reviewer_id"] != executor_id, "canary review needs a distinct reviewer")


def run_bounded(runner, updates, *, before_commit=None):
    """Bound the existing stage hooks without replacing its update loop."""
    from .generic_replication_runner import GenericReplicationRunner

    require(type(updates) is int and 0 < updates <= runner.stages[0].updates_per_round,
            "canary must fit the original first control round")
    target = runner.stages[0].start_update + updates
    require(all(state.update_index == runner.stages[0].start_update for state in runner.states.values()),
            "canaries must start from original parents")

    class CanaryStopped(Exception):
        pass

    class BoundedRunner(GenericReplicationRunner):
        def _stage_done(self, state, stage, last_kind):
            if state.update_index >= target:
                return updates < stage.updates_per_round or last_kind == "boundary"
            return super()._stage_done(state, stage, last_kind)

        def _stage_evaluation(self, *arguments, **keywords):
            raise CanaryStopped()

    bounded = BoundedRunner(
        runner.executor, runner.adapters, runner.states, runner.evaluation_provider,
        runner.batch_factory, runner.store, runner.stages,
        expected_selected_replica=runner.expected_selected_replica,
        threshold=runner.threshold, design=runner.design,
    )
    original_step = runner.executor.step
    if before_commit is not None:
        def checked_step(state, batch, adapter):
            next_state, event = original_step(state, batch, adapter)
            before_commit(state, next_state, batch, adapter, event)
            return next_state, event

        runner.executor.step = checked_step
    try:
        try:
            bounded.run()
        except CanaryStopped:
            pass
    finally:
        if before_commit is not None:
            runner.executor.step = original_step
    states, histories = {}, {}
    for arm_id in sorted(runner.states):
        state, events = runner.store.recover(arm_id)
        require(state.update_index == target, "canary stopped before its fixed update count")
        require(sum(event.get("event") == "update" for event in events) == updates,
                "canary committed update inventory differs")
        if updates == runner.stages[0].updates_per_round:
            require(events[-1].get("event") == "boundary", "300-update control boundary is incomplete")
        require(not any(event.get("event") == "stage_entry" for event in events), "canary entered a continuation")
        states[arm_id], histories[arm_id] = state, events
    require(runner.store.read_selection() is None, "canary must not select a production candidate")
    return states, histories


def _state_summary(states, histories, initial_update):
    return {
        "states": {str(arm_id): {"state_sha256": stable_hash(state.to_dict()),
                                "update_index": state.update_index, "policy_fingerprint": state.policy_fingerprint}
                   for arm_id, state in states.items()},
        "committed_updates": sum(sum(event.get("event") == "update" for event in events) for events in histories.values()),
        "updates_per_arm": {str(arm_id): state.update_index - initial_update for arm_id, state in states.items()},
    }


def _require_no_terminal_failure(directory):
    for path in sorted((directory / "invocations").glob("*.result.json")):
        result = json.loads(path.read_text())
        terminal = result.get("status") == "FAILED" and (
            result.get("terminal_numerical_failure") is True
            or result.get("error_type") == "PermanentPassUpdateError"
        )
        require(not terminal, f"terminal numerical failure in {path}; same stage/attempt cannot resume")


def execute_stage(*, root, artifact_root, manifest_ref, manifest, design, stage,
                  build_runner, executor_id, reviews=None, source_receipt=None):
    """Run one requested stage; checkpoint stores remain the resume authority.

    The CLI checks the live program and controller before calling this function.
    ``build_runner`` is the model-specific wrapper of the existing assembler.
    """
    root, artifact_root = Path(root).resolve(), Path(artifact_root).resolve()
    require(stage in STAGES, "unknown training stage")
    reviews = {} if reviews is None else reviews
    results = {}
    for previous in STAGES[:STAGES.index(stage)]:
        path = artifact_root / "stages" / previous / "result.json"
        if path.is_file():
            results[previous] = (reference(root, path), json.loads(path.read_text()))
    require_stage_reviews(root, manifest_ref["sha256"], stage, reviews, results, executor_id)
    if stage == "full":
        from .generic_replication_calibration import require_source_calibration_pass

        require(source_receipt is not None, "full training requires a source-only PASS receipt")
        require_source_calibration_pass(
            source_receipt, root=root, binding=calibration_binding(manifest["source_calibration_binding"]),
        )
    directory = artifact_root / "stages" / stage
    _require_no_terminal_failure(directory)
    launch_binding = {
        "manifest": manifest_ref, "stage": stage, "executor_id": executor_id,
        "reviews": {name: reviews[name] for name in STAGES[:STAGES.index(stage)]},
        "source_receipt": source_receipt if stage == "full" else None,
    }
    publish(directory / "inputs.json", launch_binding)
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        require(result["manifest_sha256"] == manifest_ref["sha256"] and result["stage"] == stage
                and result["status"] == "PASS" and result["design_sha256"] == design.binding_hash()
                and result["inputs"] == reference(root, directory / "inputs.json"),
                "saved stage result identity differs")
        for item in result["evidence"]:
            verified_path(root, item)
        return result
    attempts = directory / "invocations"
    attempts.mkdir(parents=True, exist_ok=True)
    invocation = attempts / f"{len(list(attempts.glob('*.start.json'))) + 1:04d}"
    publish(invocation.with_suffix(".start.json"), launch_binding)
    initial_update = design.stages[0]["start_update"]
    evidence = []
    try:
        if stage == "ten":
            uninterrupted = build_runner(directory / "uninterrupted")
            states, histories = run_bounded(uninterrupted, 10)
            replay = build_runner(directory / "reloaded")
            if all(not replay.store.has_initial(arm_id) or replay.store.recover(arm_id)[0].update_index <= initial_update + 5
                   for arm_id in replay.states):
                run_bounded(replay, 5)
            replay = build_runner(directory / "reloaded")
            replay_states, _replay_histories = run_bounded(replay, 10)
            require({arm_id: state.to_dict() for arm_id, state in states.items()}
                    == {arm_id: state.to_dict() for arm_id, state in replay_states.items()},
                    "ten-update uninterrupted/reload state mismatch")
            stores = (uninterrupted.store, replay.store)
            extra = {"exact_reload_comparison": True, "reload_split": 5, "executed_store_count": 2}
        else:
            runner = build_runner(directory / "store")
            if stage == "full":
                outcome = runner.run()
                states = dict(outcome.states)
                histories = {arm_id: runner.store.recover(arm_id)[1] for arm_id in states}
                require(outcome.selection is not None, "full run selection is missing")
                selected_replica = outcome.selection.get("selected_replica")
                require(type(selected_replica) is int and selected_replica in states,
                        "full run selected replica is not in the population")
                if design.expected_selected_replica is not None:
                    require(selected_replica == design.expected_selected_replica,
                            "full run source selector mismatch")
                aggregate = sum(state.update_index - initial_update for state in states.values())
                budget = sum((item["last_round"] - item["first_round"] + 1) * item["updates_per_round"]
                             * item["population_size"] for item in design.stages)
                require(aggregate <= budget, "committed updates exceed the frozen budget")
                selected = states[selected_replica]
                complete = runner.executor.core.coordinator.read(selected).complete
                endpoint_update = design.stages[-1]["start_update"] + (
                    design.stages[-1]["last_round"] - design.stages[-1]["first_round"] + 1
                ) * design.stages[-1]["updates_per_round"]
                require(complete or selected.update_index == endpoint_update, "full run stopped outside the source rule")
                extra = {"selection": outcome.selection, "evaluations": list(outcome.evaluations),
                         "aggregate_budget": budget, "selected_budget": endpoint_update - initial_update,
                         "all_permanent_early_stop": bool(complete and selected.update_index < endpoint_update)}
            else:
                comparator = load_callable(root, manifest["one_update_comparator"]) if stage == "one" else None

                def compare(before, after, batch, adapter, event):
                    arm_id = before.method_state["source_replica"]
                    comparison_path = directory / "comparisons" / f"arm-{arm_id}.json"
                    expected = {"arm_id": arm_id, "before_sha256": stable_hash(before.to_dict()),
                                "after_sha256": stable_hash(after.to_dict()), "manifest": manifest_ref,
                                "tolerances": manifest["tolerances"]}
                    if comparison_path.exists():
                        comparison = json.loads(comparison_path.read_text())
                    else:
                        measured = comparator(
                            root=root, output=comparison_path.parent, manifest=manifest,
                            before=before, after=after, batch=batch, adapter=adapter, event=event,
                        )
                        comparison = {"binding": expected, "measurement": measured}
                        publish(comparison_path, comparison)
                    require(comparison["binding"] == expected, "one-update comparison state binding differs")
                    measured = comparison["measurement"]
                    require(measured.get("decision") == "PASS" and set(measured.get("checks", {})) == set(COMPARISON_CHECKS)
                            and all(value is True for value in measured["checks"].values()), "one-update source comparison failed")
                    require(bool(measured.get("evidence")), "raw source comparison evidence required")
                    for item in measured["evidence"]:
                        verified_path(root, item)

                states, histories = run_bounded(runner, 1 if stage == "one" else 300,
                                               before_commit=compare if stage == "one" else None)
                if stage == "one":
                    for arm_id in states:
                        comparison_path = directory / "comparisons" / f"arm-{arm_id}.json"
                        comparison = json.loads(comparison_path.read_text())
                        require(comparison["measurement"]["decision"] == "PASS", "missing passing parent comparison")
                        require(comparison["binding"]["after_sha256"] == stable_hash(states[arm_id].to_dict()),
                                "committed parent differs from comparison")
                        evidence.extend(comparison["measurement"]["evidence"])
                        evidence.append(reference(root, comparison_path))
                extra = {"five_parent_comparison": stage == "one", "control_boundary": stage == "boundary"}
            stores = (runner.store,)
        for store in stores:
            evidence.extend(reference(root, path) for path in sorted(store.root.rglob("*"))
                            if path.is_file() and not path.name.startswith(".pending-"))
        result = {"schema": RESULT_SCHEMA, "stage": stage, "status": "PASS",
                  "manifest_sha256": manifest_ref["sha256"], "design_sha256": design.binding_hash(),
                  "executor_id": executor_id, "inputs": reference(root, directory / "inputs.json"),
                  "evidence": evidence, "source_calibration_receipt": source_receipt if stage == "full" else None,
                  "production_integrity_evidence": False,
                  **_state_summary(states, histories, initial_update), **extra}
        publish(result_path, result)
        publish(invocation.with_suffix(".result.json"), {"status": "COMPLETE", "result": reference(root, result_path)})
        return result
    except Exception as error:
        failure = {
            "status": "INTERRUPTED" if isinstance(error, (InterruptedError, TimeoutError)) else "FAILED",
            "error_type": type(error).__name__, "message": str(error), "inputs": launch_binding,
            "resume_authority": "existing immutable runner checkpoint stores; no failed receipt is promoted",
        }
        executor_module = sys.modules.get(__package__ + ".generic_permanent_pass_executor")
        if isinstance(error, getattr(executor_module, "PermanentPassUpdateError", ())):
            failure.update({
                "terminal_numerical_failure": True, "resume_permitted": False,
                "failed_update": {"checkpoint": error.checkpoint.to_dict(), "event": error.event,
                                  "diagnostic_only": True, "checkpoint_committed": False},
                "resume_authority": "none for this stage/attempt after a terminal numerical failure",
            })
        publish(invocation.with_suffix(".result.json"), failure)
        raise
