"""Host-owned immutable tensor bundles for saved-input model boundaries."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _shape(name: str, value: Sequence[int | None]) -> tuple[int | None, ...]:
    normalized = tuple(value)
    if any(item is not None and (type(item) is not int or item < 0) for item in normalized):
        raise ValueError(f"{name} contains an invalid dimension")
    return normalized


def _digest(value: str) -> str:
    normalized = _nonempty("digest", value).lower()
    if len(normalized) != 64:
        raise ValueError("digest must be a SHA-256 value")
    try:
        int(normalized, 16)
    except ValueError as error:
        raise ValueError("digest must be a SHA-256 value") from error
    return normalized


def array_fingerprint(names: Sequence[str], arrays: Sequence[np.ndarray]) -> str:
    if len(names) != len(arrays):
        raise ValueError("fingerprint names and arrays must have matching lengths")
    digest = hashlib.sha256()
    for name, value in zip(names, arrays, strict=True):
        array = np.ascontiguousarray(value)
        digest.update(_nonempty("field name", name).encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorFieldSpec:
    """Shape and row-sharing rule for one saved tensor field."""

    name: str
    rank: int
    tail_shape: tuple[int | None, ...]
    row_axis: int | None = 0
    dtype: str = "float64"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty("field name", self.name))
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("field rank must be a positive integer")
        object.__setattr__(self, "tail_shape", _shape("tail shape", self.tail_shape))
        if self.row_axis not in (None, 0):
            raise ValueError("saved-input row_axis must be zero or None")
        expected = self.rank if self.row_axis is None else self.rank - 1
        if len(self.tail_shape) != expected:
            raise ValueError("field tail shape does not match rank and row axis")
        if self.dtype != "float64":
            raise ValueError("saved-input contracts currently require float64")

    def validate(self, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{self.name} must be a NumPy array at the host boundary")
        if value.dtype != np.dtype(self.dtype):
            raise ValueError(f"{self.name} dtype must be {self.dtype}")
        if value.ndim != self.rank:
            raise ValueError(f"{self.name} rank mismatch")
        actual_tail = value.shape if self.row_axis is None else value.shape[1:]
        if any(
            expected is not None and actual != expected
            for actual, expected in zip(actual_tail, self.tail_shape, strict=True)
        ):
            raise ValueError(f"{self.name} shape mismatch")
        if self.row_axis is not None and value.shape[0] <= 0:
            raise ValueError(f"{self.name} must contain at least one row")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{self.name} contains non-finite values")
        copied = np.array(value, dtype=self.dtype, copy=True, order="C")
        copied.setflags(write=False)
        return copied


@dataclass(frozen=True)
class FrozenTensorBundle:
    """Ordered, read-only arrays with exact fields and shared row count."""

    specs: tuple[TensorFieldSpec, ...]
    arrays: tuple[np.ndarray, ...]
    _row_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        arrays = tuple(self.arrays)
        if not specs or len(specs) != len(arrays):
            raise ValueError("tensor bundle fields must be non-empty and aligned")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("tensor bundle field names must be unique")
        if not all(isinstance(spec, TensorFieldSpec) for spec in specs):
            raise TypeError("tensor bundle specs must be TensorFieldSpec instances")
        normalized = tuple(
            spec.validate(value) for spec, value in zip(specs, arrays, strict=True)
        )
        row_counts = tuple(
            value.shape[0]
            for spec, value in zip(specs, normalized, strict=True)
            if spec.row_axis is not None
        )
        if row_counts and len(set(row_counts)) != 1:
            raise ValueError("row-bearing tensor fields must share row count")
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "arrays", normalized)
        object.__setattr__(self, "_row_count", row_counts[0] if row_counts else 0)

    @classmethod
    def from_mapping(
        cls,
        specs: Sequence[TensorFieldSpec],
        values: Mapping[str, np.ndarray],
    ) -> FrozenTensorBundle:
        ordered_specs = tuple(specs)
        if tuple(values) != tuple(spec.name for spec in ordered_specs):
            raise ValueError("saved-input field order or inventory mismatch")
        return cls(ordered_specs, tuple(values[spec.name] for spec in ordered_specs))

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def mapping(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(dict(zip(self.names, self.arrays, strict=True)))

    @property
    def fingerprint(self) -> str:
        return array_fingerprint(self.names, self.arrays)

    def as_tuple(self) -> tuple[np.ndarray, ...]:
        return self.arrays

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "dsge_hmc.generic_frozen_tensor_bundle.v1",
            "fields": [
                {
                    "name": spec.name,
                    "rank": spec.rank,
                    "tail_shape": list(spec.tail_shape),
                    "row_axis": spec.row_axis,
                    "dtype": spec.dtype,
                    "shape": list(value.shape),
                }
                for spec, value in zip(self.specs, self.arrays, strict=True)
            ],
            "row_count": self.row_count,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class SourceHashBinding:
    """Hash binding for trusted source code and saved input records."""

    source_hashes: Mapping[str, str]
    input_hash: str
    target_id: str
    target_version: str = "v1"

    def __post_init__(self) -> None:
        if not isinstance(self.source_hashes, Mapping) or not self.source_hashes:
            raise ValueError("source hash binding must not be empty")
        normalized = {
            _nonempty("source path", path): _digest(digest)
            for path, digest in self.source_hashes.items()
        }
        object.__setattr__(self, "source_hashes", MappingProxyType(normalized))
        object.__setattr__(self, "input_hash", _digest(self.input_hash))
        object.__setattr__(self, "target_id", _nonempty("target ID", self.target_id))
        object.__setattr__(self, "target_version", _nonempty("target version", self.target_version))

    def metadata(self) -> dict[str, Any]:
        return {
            "source_hashes": dict(self.source_hashes),
            "input_hash": self.input_hash,
            "target_id": self.target_id,
            "target_version": self.target_version,
        }


__all__ = [
    "FrozenTensorBundle",
    "SourceHashBinding",
    "TensorFieldSpec",
    "array_fingerprint",
]
