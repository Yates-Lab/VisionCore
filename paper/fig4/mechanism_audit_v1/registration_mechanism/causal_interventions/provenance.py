"""Fail-closed provenance and resume markers for causal intervention parts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_record(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    value = Path(path).stat()
    result = {
        "path": str(Path(path).resolve()),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }
    if hash_file:
        result["sha256"] = sha256_file(Path(path))
    return result


def write_complete_marker(
    directory: Path,
    *,
    schema_version: str,
    stage: str,
    scope: str,
    identity: dict[str, Any],
    input_fingerprint: dict[str, Any],
    products: Sequence[Path],
    diagnostics: dict[str, Any],
) -> Path:
    if not products or any(not Path(path).is_file() for path in products):
        raise RuntimeError("Cannot mark an intervention part complete without every product")
    marker = Path(directory) / "complete.json"
    atomic_json(
        marker,
        {
            "complete": True,
            "schema_version": schema_version,
            "stage": stage,
            "scope": scope,
            "identity": identity,
            "input_fingerprint": input_fingerprint,
            "products": [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": sha256_file(Path(path)),
                    "size_bytes": Path(path).stat().st_size,
                }
                for path in products
            ],
            "diagnostics": diagnostics,
        },
    )
    return marker


def valid_complete_marker(
    directory: Path,
    *,
    schema_version: str,
    stage: str,
    scope: str,
    identity: dict[str, Any],
    input_fingerprint: dict[str, Any],
    required_product_names: Sequence[str],
) -> dict[str, Any] | None:
    marker = Path(directory) / "complete.json"
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not (
        value.get("complete") is True
        and value.get("schema_version") == schema_version
        and value.get("stage") == stage
        and value.get("scope") == scope
        and value.get("identity") == json_ready(identity)
        and value.get("input_fingerprint") == json_ready(input_fingerprint)
    ):
        return None
    products = value.get("products")
    if not isinstance(products, list):
        return None
    by_name = {Path(item.get("path", "")).name: item for item in products}
    if not set(required_product_names).issubset(by_name):
        return None
    for item in products:
        path = Path(item.get("path", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            return None
    return value
