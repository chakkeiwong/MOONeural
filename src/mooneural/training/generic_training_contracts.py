"""Versioned, model-independent contracts for generic neural-solver work.

Phase 1 deliberately contains no numerical backend or model equations.  The
objects in this module describe the boundaries that a later adapter and
compiled execution engine must satisfy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = "dsge_hmc.generic_neural_solver_contracts.v2"
CONTRACT_SCHEMA_VERSION = SCHEMA_VERSION
SCALE_COORDINATES = ("raw", "training", "selection", "terminal")
GOVERNANCE_LANES = ("Explore", "Validate", "Admit")
DTYPES = ("float32", "float64")
BACKENDS = ("python_fake", "tensorflow")


class ContractSchemaError(ValueError):
    """Raised when a contract payload is missing or has an unknown schema."""


class ImmutableJSONMapping(Mapping):
    """An owned immutable JSON declaration with reusable canonical bytes."""

    __slots__ = ("_value", "canonical", "fingerprint")

    def __init__(self, value):
        normalized = _json_value(value)
        if not isinstance(normalized, dict):
            raise TypeError("an immutable JSON declaration must be a mapping")
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "_value", _frozen_json(normalized))
        object.__setattr__(self, "canonical", encoded)
        object.__setattr__(self, "fingerprint", hashlib.sha256(encoded.encode("utf-8")).hexdigest())

    def __setattr__(self, name, value):
        raise TypeError("immutable JSON declaration")

    def __delattr__(self, name):
        raise TypeError("immutable JSON declaration")

    def __getitem__(self, key):
        return self._value[key]

    def __iter__(self):
        return iter(self._value)

    def __len__(self):
        return len(self._value)

    def __eq__(self, other):
        if isinstance(other, ImmutableJSONMapping):
            return self.canonical == other.canonical
        return isinstance(other, Mapping) and self.canonical == canonical_json(other)

    def __deepcopy__(self, memo):
        return self

    def to_dict(self):
        return json.loads(self.canonical)


def _json_value(value: Any, *, shared=False, declarations=None) -> Any:
    if isinstance(value, ImmutableJSONMapping):
        if declarations is not None:
            declarations.append(value)
        return value if shared else value.to_dict()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON contract values must be finite")
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("contract mapping keys must be strings")
            normalized[key] = _json_value(item, shared=shared, declarations=declarations)
        return normalized
    if isinstance(value, (tuple, list)):
        return [_json_value(item, shared=shared, declarations=declarations) for item in value]
    raise TypeError(f"contract value is not JSON-safe: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a contract value deterministically and without NaN values."""

    declarations = []
    normalized = _json_value(value, shared=True, declarations=declarations)
    if not declarations:
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "".join(_canonical_parts(normalized))


def _canonical_parts(value):
    if isinstance(value, ImmutableJSONMapping):
        yield value.canonical
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("contract mapping keys must be strings")
        yield "{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield ","
            yield json.dumps(key)
            yield ":"
            yield from _canonical_parts(value[key])
        yield "}"
    elif isinstance(value, (tuple, list)):
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from _canonical_parts(item)
        yield "]"
    else:
        yield json.dumps(_json_value(value), allow_nan=False, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    """Return the SHA-256 fingerprint of a canonical contract value."""

    if isinstance(value, ImmutableJSONMapping):
        return value.fingerprint
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty(name: str, value: str) -> str:
    """Require an actual, non-empty string instead of silently coercing input."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_string(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _nonempty(name, value)


def _strict_bool(name: str, value: bool) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _strict_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _strict_sequence(name: str, values: Sequence[Any]) -> Sequence[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return values


def _unique_strings(name: str, values: Sequence[str]) -> tuple[str, ...]:
    _strict_sequence(name, values)
    normalized = tuple(_nonempty(name, value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique values")
    return normalized


def _finite(name: str, value: float) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a real numeric primitive")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive(name: str, value: float) -> float:
    normalized = _finite(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _sha256(name: str, value: str) -> str:
    normalized = _nonempty(name, value).lower()
    if len(normalized) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return normalized


def _schema(payload: Mapping[str, Any], expected: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("contract payload must be a mapping")
    actual = payload.get("schema")
    if actual != expected:
        raise ContractSchemaError(
            f"unknown contract schema: expected {expected}, got {actual}"
        )
    return dict(payload)


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("contract metadata must be a mapping")
    return dict(_json_value(value, shared=True))


def _verify_hash(name: str, payload: Mapping[str, Any], field: str) -> None:
    supplied = payload.get(field)
    if supplied is None:
        raise ContractSchemaError(f"{name} is missing {field}")
    expected = stable_hash(
        {key: value for key, value in payload.items() if key != field}
    )
    if _sha256(field, supplied) != expected:
        raise ContractSchemaError(f"{name} {field} does not match payload")


def _deserialize(cls, payload, suffix, hash_field=None, **decoders):
    data = _schema(payload, f"{SCHEMA_VERSION}.{suffix}")
    if hash_field is not None:
        _verify_hash(suffix, data, hash_field)
        del data[hash_field]
    del data["schema"]
    expected = {item.name for item in fields(cls)}
    if set(data) != expected:
        raise ContractSchemaError(f"{suffix} fields do not match schema")
    for name, decode in decoders.items():
        data[name] = decode(data[name])
    return cls(**data)


def _frozen_json(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _frozen_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_json(item) for item in value)
    return value


def _tensor_metadata(dtype: str, backend: str) -> None:
    if _nonempty("dtype", dtype) not in DTYPES:
        raise ValueError("unsupported tensor dtype")
    if _nonempty("backend", backend) not in BACKENDS:
        raise ValueError("unsupported tensor backend")


@dataclass(frozen=True)
class TaskDefinition:
    """One ordered objective/task with no model-specific cardinality."""

    task_id: str = ""
    order: int = 0
    objective_family: str = "generic"
    hard: bool = False
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")
        _optional_string("name", self.name)
        _strict_bool("hard", self.hard)
        task_id = self.task_id or (self.name or "")
        if self.name is not None and self.task_id and self.name != self.task_id:
            raise ValueError("task_id and name must identify the same task")
        object.__setattr__(self, "task_id", _nonempty("task_id", task_id))
        object.__setattr__(self, "name", task_id)
        if type(self.order) is not int or self.order < 0:
            raise ValueError("task order must be a non-negative integer")
        object.__setattr__(
            self,
            "objective_family",
            _nonempty("objective_family", self.objective_family),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.task_definition",
            "task_id": self.task_id,
            "name": self.name,
            "order": self.order,
            "objective_family": self.objective_family,
            "hard": self.hard,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskDefinition:
        return _deserialize(cls, payload, "task_definition")


@dataclass(frozen=True)
class TaskRegistry:
    """Ordered task inventory; order and identity cannot drift mid-run."""

    tasks: tuple[TaskDefinition, ...]
    order_version: str = "v1"

    def __post_init__(self) -> None:
        normalized = tuple(_strict_sequence("tasks", self.tasks))
        if not all(isinstance(item, TaskDefinition) for item in normalized):
            raise TypeError("tasks must contain TaskDefinition instances")
        if not normalized:
            raise ValueError("task registry must contain at least one task")
        if len({item.task_id for item in normalized}) != len(normalized):
            raise ValueError("task IDs must be unique")
        orders = tuple(item.order for item in normalized)
        if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
            raise ValueError("task order must be strictly increasing")
        object.__setattr__(self, "tasks", normalized)
        object.__setattr__(
            self, "order_version", _nonempty("order_version", self.order_version)
        )

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(item.task_id for item in self.tasks)

    def contains(self, task_id: str) -> bool:
        return _nonempty("task_id", task_id) in self.task_ids

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": f"{SCHEMA_VERSION}.task_registry",
            "order_version": self.order_version,
            "tasks": [item.to_dict() for item in self.tasks],
        }
        payload["registry_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskRegistry:
        return _deserialize(
            cls,
            payload,
            "task_registry",
            "registry_hash",
            tasks=lambda items: tuple(
                TaskDefinition.from_dict(item)
                for item in _strict_sequence("tasks", items)
            ),
        )


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Objective metadata kept separate from the numerical evaluator."""

    objective_id: str = ""
    order: int = 0
    task_id: str | None = None
    threshold: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "objective_id", _nonempty("objective_id", self.objective_id)
        )
        if self.task_id is not None:
            object.__setattr__(self, "task_id", _nonempty("task_id", self.task_id))
        if type(self.order) is not int or self.order < 0:
            raise ValueError("objective order must be a non-negative integer")
        if self.threshold is not None:
            object.__setattr__(
                self, "threshold", _positive("threshold", self.threshold)
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.objective_definition",
            "objective_id": self.objective_id,
            "order": self.order,
            "task_id": self.task_id,
            "threshold": self.threshold,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveDefinition:
        return _deserialize(cls, payload, "objective_definition")


@dataclass(frozen=True)
class ObjectiveRegistry:
    objectives: tuple[ObjectiveDefinition, ...]
    order_version: str = "v1"

    def __post_init__(self) -> None:
        normalized = tuple(_strict_sequence("objectives", self.objectives))
        if not all(isinstance(item, ObjectiveDefinition) for item in normalized):
            raise TypeError("objectives must contain ObjectiveDefinition instances")
        if not normalized:
            raise ValueError("objective registry must contain at least one objective")
        if len({item.objective_id for item in normalized}) != len(normalized):
            raise ValueError("objective IDs must be unique")
        orders = tuple(item.order for item in normalized)
        if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
            raise ValueError("objective order must be strictly increasing")
        object.__setattr__(self, "objectives", normalized)
        object.__setattr__(
            self, "order_version", _nonempty("order_version", self.order_version)
        )

    @property
    def objective_ids(self) -> tuple[str, ...]:
        return tuple(item.objective_id for item in self.objectives)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": f"{SCHEMA_VERSION}.objective_registry",
            "order_version": self.order_version,
            "objectives": [item.to_dict() for item in self.objectives],
        }
        payload["registry_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectiveRegistry:
        return _deserialize(
            cls,
            payload,
            "objective_registry",
            "registry_hash",
            objectives=lambda items: tuple(
                ObjectiveDefinition.from_dict(item)
                for item in _strict_sequence("objectives", items)
            ),
        )


@dataclass(frozen=True)
class ScaleSpec:
    """Metric multiplier relative to raw units, not a residual denominator."""

    scale_id: str
    coordinate: str
    factor: float = 1.0
    source: str = "declared"
    version: str = "v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_id", _nonempty("scale_id", self.scale_id))
        coordinate = _nonempty("coordinate", self.coordinate).lower()
        if coordinate not in SCALE_COORDINATES:
            raise ValueError(f"unknown scale coordinate: {coordinate}")
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "factor", _positive("factor", self.factor))
        object.__setattr__(self, "source", _nonempty("source", self.source))
        object.__setattr__(self, "version", _nonempty("version", self.version))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.scale_spec",
            "scale_id": self.scale_id,
            "coordinate": self.coordinate,
            "factor": self.factor,
            "source": self.source,
            "version": self.version,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScaleSpec:
        return _deserialize(cls, payload, "scale_spec")


@dataclass(frozen=True)
class NormalizationMetadata:
    """Raw/training/selection/terminal scale registry for one task."""

    task_id: str
    raw_scale: ScaleSpec
    training_scale: ScaleSpec
    selection_scale: ScaleSpec
    terminal_scale: ScaleSpec
    scale_version: str = "v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _nonempty("task_id", self.task_id))
        scales = (
            self.raw_scale,
            self.training_scale,
            self.selection_scale,
            self.terminal_scale,
        )
        if not all(isinstance(item, ScaleSpec) for item in scales):
            raise TypeError("normalization scales must be ScaleSpec instances")
        coordinates = tuple(item.coordinate for item in scales)
        if coordinates != SCALE_COORDINATES:
            raise ValueError(
                "normalization scales must cover raw/training/selection/terminal"
            )
        if self.raw_scale.factor != 1.0:
            raise ValueError("raw scale must be the identity multiplier")
        scale_ids = tuple(item.scale_id for item in scales)
        if len(set(scale_ids)) != len(scale_ids):
            raise ValueError("scale IDs must be distinct across coordinates")
        object.__setattr__(
            self, "scale_version", _nonempty("scale_version", self.scale_version)
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def raw(self) -> ScaleSpec:
        return self.raw_scale

    @property
    def training(self) -> ScaleSpec:
        return self.training_scale

    @property
    def selection(self) -> ScaleSpec:
        return self.selection_scale

    @property
    def terminal(self) -> ScaleSpec:
        return self.terminal_scale

    def scale_for(self, coordinate: str) -> ScaleSpec:
        normalized = _nonempty("coordinate", coordinate).lower()
        if normalized not in SCALE_COORDINATES:
            raise ValueError(f"unknown scale coordinate: {normalized}")
        return {
            "raw": self.raw_scale,
            "training": self.training_scale,
            "selection": self.selection_scale,
            "terminal": self.terminal_scale,
        }[normalized]

    def validate_application(
        self,
        coordinate: str,
        applied_scale_ids: Sequence[str] = (),
    ) -> None:
        scale = self.scale_for(coordinate)
        ids = tuple(
            _nonempty("scale_id", item)
            for item in _strict_sequence("applied_scale_ids", applied_scale_ids)
        )
        if detect_double_scaling(ids):
            raise ValueError("double_scaling_detected")
        if ids != (scale.scale_id,) and not (coordinate == "raw" and not ids):
            raise ValueError("normalization_scale_binding_mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.normalization_metadata",
            "task_id": self.task_id,
            "raw_scale": self.raw_scale.to_dict(),
            "training_scale": self.training_scale.to_dict(),
            "selection_scale": self.selection_scale.to_dict(),
            "terminal_scale": self.terminal_scale.to_dict(),
            "scale_version": self.scale_version,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NormalizationMetadata:
        return _deserialize(
            cls,
            payload,
            "normalization_metadata",
            **{f"{name}_scale": ScaleSpec.from_dict for name in SCALE_COORDINATES},
        )


def detect_double_scaling(applied_scale_ids: Sequence[str]) -> bool:
    """Return whether a value has the same normalization applied twice."""

    normalized = tuple(
        _nonempty("scale_id", item)
        for item in _strict_sequence("applied_scale_ids", applied_scale_ids)
    )
    return len(normalized) != len(set(normalized))


@dataclass(frozen=True)
class MetricCoordinates:
    """A task metric retaining every scale coordinate and its provenance."""

    task_id: str
    raw: float
    training: float
    selection: float
    terminal: float
    normalization: NormalizationMetadata
    applied_scale_ids: tuple[str, ...] = ()
    gate_ratio: float | None = None
    coordinate: str = "training"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _nonempty("task_id", self.task_id))
        if not isinstance(self.normalization, NormalizationMetadata):
            raise TypeError("normalization must be NormalizationMetadata")
        if self.normalization.task_id != self.task_id:
            raise ValueError("metric and normalization task IDs must match")
        for name in ("raw", "training", "selection", "terminal"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
            if not math.isclose(
                getattr(self, name),
                self.raw * self.normalization.scale_for(name).factor,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError(f"inconsistent metric coordinate: {name}")
        if self.gate_ratio is not None:
            object.__setattr__(
                self, "gate_ratio", _finite("gate_ratio", self.gate_ratio)
            )
        ids = tuple(
            _nonempty("scale_id", item)
            for item in _strict_sequence("applied_scale_ids", self.applied_scale_ids)
        )
        self.normalization.validate_application(self.coordinate, ids)
        object.__setattr__(self, "applied_scale_ids", ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.metric_coordinates",
            "task_id": self.task_id,
            "raw": self.raw,
            "training": self.training,
            "selection": self.selection,
            "terminal": self.terminal,
            "gate_ratio": self.gate_ratio,
            "coordinate": self.coordinate,
            "applied_scale_ids": list(self.applied_scale_ids),
            "normalization": self.normalization.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MetricCoordinates:
        return _deserialize(
            cls,
            payload,
            "metric_coordinates",
            normalization=NormalizationMetadata.from_dict,
        )


@dataclass(frozen=True)
class PolicyView:
    """Immutable policy view passed into read-only evaluation calls."""

    values: tuple[float, ...]
    policy_id: str = "policy"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = tuple(
            _finite("policy value", item)
            for item in _strict_sequence("values", self.values)
        )
        if not values:
            raise ValueError("policy values must not be empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "policy_id", _nonempty("policy_id", self.policy_id))
        object.__setattr__(self, "metadata", _frozen_json(_mapping(self.metadata)))

    def fingerprint(self) -> str:
        return stable_hash(
            {
                "policy_id": self.policy_id,
                "values": list(self.values),
                "metadata": self.metadata,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.policy_view",
            "policy_id": self.policy_id,
            "values": list(self.values),
            "metadata": _json_value(self.metadata),
            "policy_fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PolicyView:
        data = _schema(payload, f"{SCHEMA_VERSION}.policy_view")
        fingerprint = _sha256("policy_fingerprint", data.pop("policy_fingerprint"))
        policy = _deserialize(cls, data, "policy_view")
        if policy.fingerprint() != fingerprint:
            raise ContractSchemaError("policy fingerprint mismatch")
        return policy


FrozenPolicyView = PolicyView


@dataclass(frozen=True)
class TaskBatchTensors:
    """Shape and provenance contract for a backend-native task batch."""

    task_ids: tuple[str, ...]
    sample_count: int
    policy_dimension: int
    dtype: str = "float64"
    backend: str = "python_fake"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hard_check_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_ids", _unique_strings("task_ids", self.task_ids))
        object.__setattr__(
            self,
            "hard_check_ids",
            _unique_strings("hard_check_ids", self.hard_check_ids),
        )
        if not self.task_ids:
            raise ValueError("task_ids must not be empty")
        _tensor_metadata(self.dtype, self.backend)
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if type(self.policy_dimension) is not int or self.policy_dimension <= 0:
            raise ValueError("policy_dimension must be a positive integer")
        object.__setattr__(self, "dtype", _nonempty("dtype", self.dtype))
        object.__setattr__(self, "backend", _nonempty("backend", self.backend))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.task_batch_tensors",
            "hard_check_ids": list(self.hard_check_ids),
            "task_ids": list(self.task_ids),
            "sample_count": self.sample_count,
            "policy_dimension": self.policy_dimension,
            "dtype": self.dtype,
            "backend": self.backend,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskBatchTensors:
        return _deserialize(cls, payload, "task_batch_tensors")


@dataclass(frozen=True)
class TaskEvaluationTensors:
    """Values and matching gradient rows in training-normalized coordinates."""

    task_ids: tuple[str, ...]
    values: tuple[float, ...]
    rows: tuple[tuple[float, ...], ...]
    hard_max: tuple[float, ...] = ()
    policy_dimension: int = 0
    coordinate: str = "training"
    dtype: str = "float64"
    backend: str = "python_fake"
    normalization_hash: str | None = None
    hard_check_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        task_ids = _unique_strings("task_ids", self.task_ids)
        hard_check_ids = _unique_strings("hard_check_ids", self.hard_check_ids)
        if not task_ids:
            raise ValueError("task_ids must not be empty")
        _tensor_metadata(self.dtype, self.backend)
        for name in ("values", "rows", "hard_max"):
            _strict_sequence(name, getattr(self, name))
        for row in self.rows:
            _strict_sequence("gradient row", row)
        values = tuple(_finite("task value", item) for item in self.values)
        rows = tuple(
            tuple(_finite("gradient value", item) for item in row) for row in self.rows
        )
        hard_max = tuple(_finite("hard maximum", item) for item in self.hard_max)
        if len(task_ids) != len(values) or len(values) != len(rows):
            raise ValueError(
                "task IDs, values, and gradient rows must have matching lengths"
            )
        if len(hard_max) != len(hard_check_ids):
            raise ValueError(
                "hard_max must match the independent hard-check key registry"
            )
        dimension = self.policy_dimension
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("policy_dimension must be a positive integer")
        if any(len(row) != dimension for row in rows):
            raise ValueError("gradient rows must match policy_dimension")
        if self.coordinate != "training":
            raise ValueError("task evaluation tensors must use training coordinates")
        if self.normalization_hash is not None:
            object.__setattr__(
                self,
                "normalization_hash",
                _sha256("normalization_hash", self.normalization_hash),
            )
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "hard_max", hard_max)
        object.__setattr__(self, "hard_check_ids", hard_check_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.task_evaluation_tensors",
            "hard_check_ids": list(self.hard_check_ids),
            "task_ids": list(self.task_ids),
            "values": list(self.values),
            "rows": [list(row) for row in self.rows],
            "hard_max": list(self.hard_max),
            "policy_dimension": self.policy_dimension,
            "coordinate": self.coordinate,
            "dtype": self.dtype,
            "backend": self.backend,
            "normalization_hash": self.normalization_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskEvaluationTensors:
        return _deserialize(cls, payload, "task_evaluation_tensors")

    def validate_batch(self, batch: TaskBatchTensors, normalization_hash: str) -> None:
        for name in (
            "task_ids",
            "policy_dimension",
            "dtype",
            "backend",
            "hard_check_ids",
        ):
            if getattr(self, name) != getattr(batch, name):
                raise ValueError(f"evaluation batch binding mismatch: {name}")
        if self.normalization_hash != _sha256("normalization_hash", normalization_hash):
            raise ValueError("evaluation normalization binding mismatch")


@dataclass(frozen=True)
class EvaluationRequest:
    """Host metadata binding an evaluation to a role and target."""

    role: str
    target_id: str
    seeds: tuple[int, ...] = ()
    cells: tuple[str, ...] = ()
    anchor_ids: tuple[str, ...] = ()
    sample_count: int = 1
    estimator_version: str = "v1"
    target_version: str = "v1"
    scale_version: str = "v1"
    policy_fingerprint: str = ""
    task_ids: tuple[str, ...] = ()
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _nonempty("role", self.role))
        object.__setattr__(self, "target_id", _nonempty("target_id", self.target_id))
        seeds = tuple(
            _strict_int("seed", item) for item in _strict_sequence("seeds", self.seeds)
        )
        if self.seed is not None:
            _strict_int("seed", self.seed)
            if seeds and self.seed not in seeds:
                raise ValueError("seed and seeds disagree")
            if not seeds:
                seeds = (self.seed,)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(item < 0 for item in seeds)
        ):
            raise ValueError("unique non-negative resolved seeds are required")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "cells", _unique_strings("cells", self.cells))
        object.__setattr__(
            self, "anchor_ids", _unique_strings("anchor_ids", self.anchor_ids)
        )
        object.__setattr__(self, "task_ids", _unique_strings("task_ids", self.task_ids))
        if not self.task_ids:
            raise ValueError("evaluation task_ids must not be empty")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        for name in ("estimator_version", "target_version", "scale_version"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        object.__setattr__(
            self,
            "policy_fingerprint",
            _sha256("policy_fingerprint", self.policy_fingerprint),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.evaluation_request",
            "role": self.role,
            "target_id": self.target_id,
            "seeds": list(self.seeds),
            "seed": self.seed,
            "cells": list(self.cells),
            "anchor_ids": list(self.anchor_ids),
            "sample_count": self.sample_count,
            "estimator_version": self.estimator_version,
            "target_version": self.target_version,
            "scale_version": self.scale_version,
            "policy_fingerprint": self.policy_fingerprint,
            "task_ids": list(self.task_ids),
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvaluationRequest:
        return _deserialize(cls, payload, "evaluation_request")


def _finite_metric_map(name: str, values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = {}
    for key, value in values.items():
        normalized[_nonempty(f"{name} key", key)] = _finite(name, value)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _bool_map(name: str, values: Mapping[str, bool]) -> dict[str, bool]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {
        _nonempty(name, key): _strict_bool(name, value) for key, value in values.items()
    }


def _evaluation_binding(request, metrics, role):
    if request is not None:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest")
        if request.role != role or set(request.task_ids) != set(metrics):
            raise ValueError("evaluation role or task binding mismatch")


def _request_from_dict(payload):
    return None if payload is None else EvaluationRequest.from_dict(payload)


@dataclass(frozen=True)
class ControlEvaluation:
    task_upper_mse: Mapping[str, float]
    raw_records: Mapping[str, Any] = field(default_factory=dict)
    source_cell_results: Mapping[str, Any] = field(default_factory=dict)
    request: EvaluationRequest | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_upper_mse",
            _finite_metric_map("task_upper_mse", self.task_upper_mse),
        )
        _evaluation_binding(self.request, self.task_upper_mse, "control")
        object.__setattr__(self, "raw_records", _mapping(self.raw_records))
        object.__setattr__(
            self, "source_cell_results", _mapping(self.source_cell_results)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.control_evaluation",
            "task_upper_mse": _json_value(self.task_upper_mse),
            "raw_records": _json_value(self.raw_records),
            "source_cell_results": _json_value(self.source_cell_results),
            "request": None if self.request is None else self.request.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControlEvaluation:
        return _deserialize(
            cls, payload, "control_evaluation", request=_request_from_dict
        )


@dataclass(frozen=True)
class ValidationEvaluation:
    task_mean_mse: Mapping[str, float]
    sample_counts: Mapping[str, int]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    request: EvaluationRequest | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_mean_mse",
            _finite_metric_map("task_mean_mse", self.task_mean_mse),
        )
        if not isinstance(self.sample_counts, Mapping):
            raise TypeError("sample_counts must be a mapping")
        counts = {
            _nonempty("sample count key", key): _strict_int("sample count", value)
            for key, value in self.sample_counts.items()
        }
        if set(counts) != set(self.task_mean_mse) or any(
            value <= 0 for value in counts.values()
        ):
            raise ValueError(
                "validation sample counts must cover task means with positive counts"
            )
        object.__setattr__(self, "sample_counts", counts)
        object.__setattr__(self, "provenance", _mapping(self.provenance))
        _evaluation_binding(self.request, self.task_mean_mse, "validation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.validation_evaluation",
            "task_mean_mse": _json_value(self.task_mean_mse),
            "sample_counts": self.sample_counts,
            "provenance": _json_value(self.provenance),
            "request": None if self.request is None else self.request.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValidationEvaluation:
        return _deserialize(
            cls, payload, "validation_evaluation", request=_request_from_dict
        )


@dataclass(frozen=True)
class CertificationResult:
    conjuncts: Mapping[str, bool]
    upper_records: Mapping[str, Any]
    hard_vetoes: Mapping[str, bool]
    estimator_metadata: Mapping[str, Any] = field(default_factory=dict)
    request: EvaluationRequest | None = None

    def __post_init__(self) -> None:
        conjuncts = _bool_map("conjuncts", self.conjuncts)
        vetoes = _bool_map("hard_vetoes", self.hard_vetoes)
        if not conjuncts:
            raise ValueError("certification must retain at least one conjunct")
        if not vetoes:
            raise ValueError("certification must retain at least one hard veto")
        object.__setattr__(self, "conjuncts", conjuncts)
        object.__setattr__(self, "hard_vetoes", vetoes)
        object.__setattr__(self, "upper_records", _mapping(self.upper_records))
        object.__setattr__(
            self, "estimator_metadata", _mapping(self.estimator_metadata)
        )
        if self.request is not None:
            if not isinstance(self.request, EvaluationRequest):
                raise TypeError("certification request must be an EvaluationRequest")
            if self.request.role != "certification":
                raise ValueError("certification request must use the certification role")

    @property
    def passed(self) -> bool:
        return all(self.conjuncts.values()) and all(self.hard_vetoes.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.certification_result",
            "conjuncts": self.conjuncts,
            "upper_records": _json_value(self.upper_records),
            "hard_vetoes": self.hard_vetoes,
            "estimator_metadata": _json_value(self.estimator_metadata),
            "request": None if self.request is None else self.request.to_dict(),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CertificationResult:
        data = _schema(payload, f"{SCHEMA_VERSION}.certification_result")
        passed = _strict_bool("passed", data.pop("passed"))
        result = _deserialize(
            cls,
            data,
            "certification_result",
            request=_request_from_dict,
        )
        if passed != result.passed:
            raise ContractSchemaError("certification passed flag mismatch")
        return result


@dataclass(frozen=True)
class RoleBinding:
    """One declared role/seed/target/scale binding."""

    role: str
    seed: int
    target_id: str
    scale_version: str
    sample_ids: tuple[str, ...] = ()
    source_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    target_version: str = "v1"
    estimator_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _nonempty("role", self.role))
        if _strict_int("role seed", self.seed) < 0:
            raise ValueError("role seed must be non-negative")
        object.__setattr__(self, "target_id", _nonempty("target_id", self.target_id))
        for name in ("target_version", "estimator_version"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        object.__setattr__(
            self, "scale_version", _nonempty("scale_version", self.scale_version)
        )
        object.__setattr__(
            self, "sample_ids", _unique_strings("sample_ids", self.sample_ids)
        )
        if self.source_hash is not None:
            object.__setattr__(
                self, "source_hash", _sha256("source_hash", self.source_hash)
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA_VERSION}.role_binding",
            "target_version": self.target_version,
            "estimator_version": self.estimator_version,
            "role": self.role,
            "seed": self.seed,
            "target_id": self.target_id,
            "scale_version": self.scale_version,
            "sample_ids": list(self.sample_ids),
            "source_hash": self.source_hash,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoleBinding:
        return _deserialize(cls, payload, "role_binding")


@dataclass(frozen=True)
class RoleManifest:
    bindings: tuple[RoleBinding, ...]
    manifest_version: str = "v1"

    def __post_init__(self) -> None:
        normalized = tuple(_strict_sequence("bindings", self.bindings))
        if not all(isinstance(item, RoleBinding) for item in normalized):
            raise TypeError("bindings must contain RoleBinding instances")
        if not normalized:
            raise ValueError("role manifest must contain at least one binding")
        if len({item.role for item in normalized}) != len(normalized):
            raise ValueError("role names must be unique")
        seen: dict[str, str] = {}
        for binding in normalized:
            for sample_id in binding.sample_ids:
                previous = seen.setdefault(sample_id, binding.role)
                if previous != binding.role:
                    raise ValueError("role sample sets must be disjoint")
        object.__setattr__(self, "bindings", normalized)
        object.__setattr__(
            self,
            "manifest_version",
            _nonempty("manifest_version", self.manifest_version),
        )

    def binding_hash(self) -> str:
        return self.to_dict()["role_manifest_hash"]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": f"{SCHEMA_VERSION}.role_manifest",
            "manifest_version": self.manifest_version,
            "bindings": [item.to_dict() for item in self.bindings],
        }
        payload["role_manifest_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RoleManifest:
        return _deserialize(
            cls,
            payload,
            "role_manifest",
            "role_manifest_hash",
            bindings=lambda items: tuple(
                RoleBinding.from_dict(item)
                for item in _strict_sequence("bindings", items)
            ),
        )

    def validate_request(self, request: EvaluationRequest, policy: PolicyView) -> None:
        binding = next(
            (item for item in self.bindings if item.role == request.role), None
        )
        if binding is None or request.seeds != (binding.seed,):
            raise ValueError("evaluation role or seed binding mismatch")
        if (request.target_id, request.scale_version) != (
            binding.target_id,
            binding.scale_version,
        ):
            raise ValueError("evaluation target or scale binding mismatch")
        if request.policy_fingerprint != policy.fingerprint():
            raise ValueError("evaluation policy binding mismatch")
        if (request.target_version, request.estimator_version) != (
            binding.target_version,
            binding.estimator_version,
        ):
            raise ValueError("evaluation target or estimator version binding mismatch")


@dataclass(frozen=True)
class CheckpointState:
    """Serializable policy plus optimizer/method/RNG state at a boundary."""

    attempt_id: str
    update_index: int
    policy_state: Any
    optimizer_state: Mapping[str, Any]
    method_state: Mapping[str, Any] | None
    rng_state: Mapping[str, Any]
    policy_fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _nonempty("attempt_id", self.attempt_id))
        if type(self.update_index) is not int or self.update_index < 0:
            raise ValueError("update_index must be a non-negative integer")
        policy = PolicyView.from_dict(self.policy_state)
        object.__setattr__(self, "policy_state", _frozen_json(policy.to_dict()))
        object.__setattr__(self, "optimizer_state", _mapping(self.optimizer_state))
        if self.method_state is not None:
            object.__setattr__(self, "method_state", _mapping(self.method_state))
        object.__setattr__(self, "rng_state", _mapping(self.rng_state))
        computed = policy.fingerprint()
        if self.policy_fingerprint is not None and self.policy_fingerprint != computed:
            raise ValueError(
                "policy fingerprint does not match checkpoint policy state"
            )
        object.__setattr__(self, "policy_fingerprint", computed)
        metadata = dict(self.metadata)
        for name in ("task_registry", "role_manifest"):
            if name in metadata and not isinstance(metadata[name], ImmutableJSONMapping):
                metadata[name] = ImmutableJSONMapping(metadata[name])
        object.__setattr__(self, "metadata", _mapping(metadata))

    def to_dict(self, *, shared=False) -> dict[str, Any]:
        payload = {
            "schema": f"{SCHEMA_VERSION}.checkpoint_state",
            "attempt_id": self.attempt_id,
            "update_index": self.update_index,
            "policy_state": _json_value(self.policy_state),
            "optimizer_state": _json_value(self.optimizer_state),
            "method_state": None
            if self.method_state is None
            else _json_value(self.method_state),
            "rng_state": _json_value(self.rng_state),
            "policy_fingerprint": self.policy_fingerprint,
            "metadata": _json_value(self.metadata, shared=shared),
        }
        payload["checkpoint_state_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointState:
        return _deserialize(cls, payload, "checkpoint_state", "checkpoint_state_hash")


@dataclass(frozen=True)
class CheckpointRecord:
    """Binding from an atomic checkpoint file to its completion marker."""

    attempt_id: str
    checkpoint_id: str
    checkpoint_path: str
    checkpoint_hash: str
    update_index: int
    completion_marker: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _nonempty("attempt_id", self.attempt_id))
        object.__setattr__(
            self, "checkpoint_id", _nonempty("checkpoint_id", self.checkpoint_id)
        )
        object.__setattr__(
            self, "checkpoint_path", _nonempty("checkpoint_path", self.checkpoint_path)
        )
        object.__setattr__(
            self, "checkpoint_hash", _sha256("checkpoint_hash", self.checkpoint_hash)
        )
        if type(self.update_index) is not int or self.update_index < 0:
            raise ValueError("checkpoint update_index must be non-negative")
        object.__setattr__(
            self,
            "completion_marker",
            _nonempty("completion_marker", self.completion_marker),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": f"{SCHEMA_VERSION}.checkpoint_record",
            "attempt_id": self.attempt_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "update_index": self.update_index,
            "completion_marker": self.completion_marker,
            "metadata": _json_value(self.metadata),
        }
        payload["checkpoint_record_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointRecord:
        return _deserialize(cls, payload, "checkpoint_record", "checkpoint_record_hash")


@dataclass(frozen=True)
class ResultRecord:
    """Phase result retaining counters, fingerprints, and explicit nonclaims."""

    attempt_id: str
    status: str
    update_count: int
    policy_fingerprint: str
    checkpoint_hash: str
    decision: str
    nonclaims: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_id", _nonempty("attempt_id", self.attempt_id))
        object.__setattr__(self, "status", _nonempty("status", self.status))
        if type(self.update_count) is not int or self.update_count < 0:
            raise ValueError("update_count must be non-negative")
        object.__setattr__(
            self,
            "policy_fingerprint",
            _sha256("policy_fingerprint", self.policy_fingerprint),
        )
        object.__setattr__(
            self, "checkpoint_hash", _sha256("checkpoint_hash", self.checkpoint_hash)
        )
        object.__setattr__(self, "decision", _nonempty("decision", self.decision))
        object.__setattr__(
            self, "nonclaims", _unique_strings("nonclaims", self.nonclaims)
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": f"{SCHEMA_VERSION}.result_record",
            "attempt_id": self.attempt_id,
            "status": self.status,
            "update_count": self.update_count,
            "policy_fingerprint": self.policy_fingerprint,
            "checkpoint_hash": self.checkpoint_hash,
            "decision": self.decision,
            "nonclaims": list(self.nonclaims),
            "metadata": _json_value(self.metadata),
        }
        payload["result_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResultRecord:
        return _deserialize(cls, payload, "result_record", "result_hash")


def validate_task_partition(
    registry: TaskRegistry,
    active_tasks: Sequence[str],
    constraint_tasks: Sequence[str],
) -> None:
    """Validate a policy-defined active/constraint partition."""

    active = _unique_strings("active_tasks", active_tasks)
    constraints = _unique_strings("constraint_tasks", constraint_tasks)
    known = set(registry.task_ids)
    unknown = (set(active) | set(constraints)) - known
    if unknown:
        raise ValueError(f"unknown task IDs in policy partition: {sorted(unknown)}")
    overlap = set(active) & set(constraints)
    if overlap:
        raise ValueError(f"active and constraint tasks overlap: {sorted(overlap)}")
    if set(active) | set(constraints) != known:
        raise ValueError("task partition is missing registered tasks")


@runtime_checkable
class ModelAdapter(Protocol):
    """Logical adapter interface; numerical TensorSpecs arrive in Phase 2."""

    def adapter_identity(self) -> Mapping[str, Any]: ...

    def compute_task_values_and_gradients(
        self,
        policy: PolicyView,
        batch: TaskBatchTensors,
        update_index: int,
    ) -> TaskEvaluationTensors: ...

    def compute_control_upper_mse(
        self,
        policy: FrozenPolicyView,
        request: EvaluationRequest,
    ) -> ControlEvaluation: ...

    def compute_validation_mean_mse(
        self,
        policy: FrozenPolicyView,
        request: EvaluationRequest,
    ) -> ValidationEvaluation: ...

    def compute_certification_result(
        self,
        policy: FrozenPolicyView,
        request: EvaluationRequest,
    ) -> CertificationResult: ...


TaskSpec = TaskDefinition
ObjectiveSpec = ObjectiveDefinition
NormalizationSpec = NormalizationMetadata
ObjectiveMetricBundle = MetricCoordinates
CheckpointSchema = CheckpointState


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "GOVERNANCE_LANES",
    "SCALE_COORDINATES",
    "SCHEMA_VERSION",
    "CertificationResult",
    "CheckpointRecord",
    "CheckpointSchema",
    "CheckpointState",
    "ContractSchemaError",
    "ControlEvaluation",
    "EvaluationRequest",
    "FrozenPolicyView",
    "MetricCoordinates",
    "ModelAdapter",
    "NormalizationMetadata",
    "NormalizationSpec",
    "ObjectiveDefinition",
    "ObjectiveMetricBundle",
    "ObjectiveRegistry",
    "ObjectiveSpec",
    "PolicyView",
    "ResultRecord",
    "RoleBinding",
    "RoleManifest",
    "ScaleSpec",
    "TaskBatchTensors",
    "TaskDefinition",
    "TaskEvaluationTensors",
    "TaskRegistry",
    "TaskSpec",
    "ValidationEvaluation",
    "canonical_json",
    "detect_double_scaling",
    "stable_hash",
    "validate_task_partition",
]
