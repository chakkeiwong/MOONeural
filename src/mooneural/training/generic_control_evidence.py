"""Immutable full controls with compact, exactly rehydratable history entries."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from mooneural.artifacts import atomic_write_text

from .generic_training_contracts import ControlEvaluation, canonical_json, stable_hash

CONTROL_REFERENCE_SCHEMA = "dsge_hmc.generic_control_evidence_reference.v1"
CONTROL_REFERENCE_PREFIX = "dsge_hmc.generic_control_evidence_reference."


class ControlEvidenceError(ValueError):
    """A compact control cannot be bound to its immutable full evidence."""


def has_control_reference(control):
    schema = control.raw_records.get("schema")
    return isinstance(schema, str) and schema.startswith(CONTROL_REFERENCE_PREFIX)


def full_control_hash(control):
    if not has_control_reference(control):
        return stable_hash(control.to_dict())
    if control.raw_records["schema"] != CONTROL_REFERENCE_SCHEMA:
        raise ControlEvidenceError("unknown control evidence reference schema")
    if set(control.raw_records) != {"schema", "full_control_sha256"}:
        raise ControlEvidenceError("invalid control evidence reference fields")
    digest = control.raw_records["full_control_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ControlEvidenceError("invalid full control evidence hash")
    return digest


def _compact(control, digest):
    return replace(control, raw_records={
        "schema": CONTROL_REFERENCE_SCHEMA, "full_control_sha256": digest,
    })


class ControlEvidenceArchive:
    """Single-writer archive; cache only verified identity and file signatures."""

    def __init__(self, root):
        self.root = Path(root)
        self._verified = {}

    def begin_verification(self):
        """Require fresh content checks in the next recovery pass."""
        self._verified.clear()

    def _path(self, digest):
        return self.root / f"{digest}.json"

    @staticmethod
    def _signature(path):
        try:
            status = path.stat()
        except FileNotFoundError as error:
            raise ControlEvidenceError("full control evidence is missing") from error
        return (status.st_dev, status.st_ino, status.st_size,
                status.st_mtime_ns, status.st_ctime_ns)

    @staticmethod
    def _write(path, content):
        atomic_write_text(path, content)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def archive(self, control):
        if not isinstance(control, ControlEvaluation):
            raise TypeError("a typed control evaluation is required")
        if has_control_reference(control):
            self.verify(control)
            return control
        content = canonical_json(control.to_dict())
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        compact = _compact(control, digest)
        path = self._path(digest)
        if path.exists():
            self.verify(compact)
        else:
            if digest in self._verified:
                raise ControlEvidenceError("verified full control evidence was removed")
            self._write(path, content)
            self._verified[digest] = (self._signature(path), stable_hash(compact.to_dict()))
        return compact

    def _read(self, control, digest, signature):
        path = self._path(digest)
        content = path.read_bytes()
        if self._signature(path) != signature:
            raise ControlEvidenceError("full control evidence changed during verification")
        if hashlib.sha256(content).hexdigest() != digest:
            raise ControlEvidenceError("full control evidence hash mismatch")
        expanded = ControlEvaluation.from_dict(json.loads(content))
        if has_control_reference(expanded):
            raise ControlEvidenceError("full control evidence cannot contain another reference")
        if _compact(expanded, digest).to_dict() != control.to_dict():
            raise ControlEvidenceError("compact control differs from full control evidence")
        if stable_hash(expanded.to_dict()) != digest:
            raise ControlEvidenceError("full control evidence is not canonical")
        self._verified[digest] = (signature, stable_hash(control.to_dict()))
        return expanded

    def verify(self, control, *, use_cache=False):
        """Check content, or reuse this instance's immutable-file verification."""
        if not has_control_reference(control):
            return
        digest = full_control_hash(control)
        signature = self._signature(self._path(digest))
        cached = self._verified.get(digest)
        if cached is not None:
            if signature != cached[0]:
                raise ControlEvidenceError("verified full control evidence file changed")
            if stable_hash(control.to_dict()) != cached[1]:
                raise ControlEvidenceError("compact control differs from verified full evidence")
        if cached is None or not use_cache:
            self._read(control, digest, signature)

    def rehydrate(self, control):
        if not has_control_reference(control):
            return ControlEvaluation.from_dict(control.to_dict())
        digest = full_control_hash(control)
        signature = self._signature(self._path(digest))
        cached = self._verified.get(digest)
        if cached is not None and signature != cached[0]:
            raise ControlEvidenceError("verified full control evidence file changed")
        return self._read(control, digest, signature)
