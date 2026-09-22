"""Resolve the repository-local Python dependency closure of explicit entry points.

Production manifests should identify executable roots deliberately and hash
their transitive repository-local imports.  Directory globs are forbidden:
they make unrelated exploratory scripts part of a release merely because the
files share a folder.
"""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
from typing import Iterable


def _module_path(root: Path, module: str) -> Path | None:
    """Resolve a dotted module name to a Python file below ``root``."""
    if not module:
        return None
    relative = Path(*module.split("."))
    candidates = (root / relative.with_suffix(".py"), root / relative / "__init__.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _module_name(root: Path, path: Path) -> tuple[str, bool]:
    relative = path.resolve().relative_to(root.resolve())
    if relative.suffix != ".py":
        raise ValueError(f"Python source expected, got {relative}")
    parts = list(relative.with_suffix("").parts)
    is_package = bool(parts and parts[-1] == "__init__")
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _from_base(module: str, is_package: bool, level: int, imported: str | None) -> str:
    if level == 0:
        return imported or ""
    package = module if is_package else module.rpartition(".")[0]
    parts = package.split(".") if package else []
    ascend = level - 1
    if ascend > len(parts):
        return ""
    if ascend:
        parts = parts[:-ascend]
    if imported:
        parts.extend(imported.split("."))
    return ".".join(parts)


def local_imports(root: Path, path: Path) -> tuple[Path, ...]:
    """Return direct repository-local imports of one Python source file."""
    module, is_package = _module_name(root, path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    resolved: set[Path] = set()

    def resolve(module_name: str) -> Path | None:
        candidate = _module_path(root, module_name)
        if candidate is not None:
            return candidate
        # Several inherited paper scripts put their own directory on sys.path
        # and use bare sibling imports (for example ``from _fig3_data``).
        sibling = path.parent / Path(*module_name.split("."))
        for local in (sibling.with_suffix(".py"), sibling / "__init__.py"):
            if local.is_file():
                return local.resolve()
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidate = resolve(alias.name)
                if candidate is not None:
                    resolved.add(candidate)
        elif isinstance(node, ast.ImportFrom):
            base = _from_base(module, is_package, node.level, node.module)
            candidate = resolve(base)
            if candidate is not None:
                resolved.add(candidate)
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = resolve(".".join(filter(None, (base, alias.name))))
                if child is not None:
                    resolved.add(child)
    return tuple(sorted(resolved))


def source_closure(root: Path, entrypoints: Iterable[Path]) -> tuple[Path, ...]:
    """Resolve entry points and all transitive repository-local imports."""
    root = root.resolve()
    queue: deque[Path] = deque()
    for entrypoint in entrypoints:
        path = Path(entrypoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"production source is missing: {path}")
        path.relative_to(root)
        queue.append(path)
    visited: set[Path] = set()
    while queue:
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        parent = path.parent
        while parent != root and root in parent.parents:
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer.resolve() not in visited:
                queue.append(initializer.resolve())
            parent = parent.parent
        queue.extend(item for item in local_imports(root, path) if item not in visited)
    return tuple(sorted(visited))


def relative_source_closure(
    root: Path, entrypoints: Iterable[Path]
) -> tuple[str, ...]:
    """Return a stable manifest-ready relative-path representation."""
    root = root.resolve()
    return tuple(str(path.relative_to(root)) for path in source_closure(root, entrypoints))
