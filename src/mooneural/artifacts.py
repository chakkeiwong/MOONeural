"""Artifact helpers for long-running experiment launchers."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

try:  # pragma: no cover - numpy is present in this project.
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def json_clean(value: Any) -> Any:
    """Convert common numerical values to strict JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return json_clean(value.tolist())
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _replace_atomically(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    tmp_name = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text through a same-directory temporary file and ``os.replace``."""
    out = Path(path)
    _replace_atomically(out, str(text))
    return out


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write strict JSON atomically."""
    text = json.dumps(
        json_clean(payload), indent=indent, sort_keys=True, allow_nan=False
    ) + "\n"
    return atomic_write_text(path, text)


def append_jsonl(path: str | Path, payload: Any) -> Path:
    """Append one JSONL record and fsync the handle for monitor durability."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(json_clean(payload), sort_keys=True, allow_nan=False)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return out


def safe_load_json(path: str | Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def stable_config_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_clean(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class StageEventWriter:
    """Timestamped JSONL stage-event writer."""

    def __init__(self, path: str | Path | None):
        self.path = None if path is None else Path(path)

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        payload = {
            "created_at_unix": float(time.time()),
            "event": str(event),
            **fields,
        }
        cleaned = json_clean(payload)
        if self.path is not None:
            append_jsonl(self.path, cleaned)
        return cleaned
