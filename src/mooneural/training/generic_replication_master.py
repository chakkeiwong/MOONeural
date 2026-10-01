"""Keep scientific design identity across generated position-only updates."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from .generic_program import PLAN_PATH, ProgramError
from .generic_program_boundary import POSITION_END, POSITION_START


def _scientific_bytes(content):
    start, end = POSITION_START.encode(), POSITION_END.encode()
    if content.count(start) != 1 or content.count(end) != 1:
        raise ProgramError("master must contain exactly one generated position block")
    before, remaining = content.split(start)
    if end not in remaining:
        raise ProgramError("master generated position delimiters are reversed")
    _position, after = remaining.split(end)
    return before, after


def replication_master_hash(program, design):
    """Validate the frozen master snapshot; retain exact live attempt checks."""
    if program.plan_hash == design.master_sha256:
        return program.plan_hash
    reference = design.permanent_pass_binding.get("master_snapshot")
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ProgramError("changed master requires the design's explicit master snapshot")
    relative = Path(reference["path"])
    root = program.root.resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ProgramError("master snapshot must be repository relative")
    if reference["sha256"] != design.master_sha256:
        raise ProgramError("master snapshot is not the frozen design master")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != design.master_sha256:
        raise ProgramError("frozen master snapshot changed")
    current = program.snapshot[PLAN_PATH].encode()
    if _scientific_bytes(content) != _scientific_bytes(current):
        raise ProgramError("scientific master content changed outside generated position")
    return program.plan_hash


__all__ = ["replication_master_hash"]
