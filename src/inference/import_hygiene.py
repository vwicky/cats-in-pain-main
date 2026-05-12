"""
Resolve collisions between multiple repo subtrees that each expose a top-level
``models`` package (e.g. AudioSep vs ``video/pose-models``).

Python caches loaded packages in ``sys.modules``; the first ``import models``
wins for the whole process unless we drop stale entries and ensure the right
tree is first on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def strip_pose_models_tree_from_sys_path(repo_root: Path) -> None:
    """
    Remove every ``sys.path`` entry that points at ``<repo>/video/pose-models``.

    If that directory stays on the path, Python may resolve ``import models``
    to the **regular** ST-GCN package (``models/__init__.py``), which shadows
    AudioSep's ``models/audiosep.py`` even when ``AudioSep`` is listed earlier.
    """
    anchor = (repo_root / "video" / "pose-models").resolve()
    victims: list[str] = []
    for p in list(sys.path):
        if not p:
            continue
        try:
            r = Path(p).resolve()
        except OSError:
            continue
        if r == anchor:
            victims.append(p)
            continue
        try:
            if r.is_dir() and anchor.is_dir() and r.samefile(anchor):
                victims.append(p)
        except OSError:
            pass
    for p in victims:
        while p in sys.path:
            sys.path.remove(p)


def clear_models_package_cache() -> None:
    """Drop ``models`` and submodules from ``sys.modules`` (AudioSep vs ST-GCN handoff)."""
    for key in list(sys.modules):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]


def prioritize_sys_path(entry: Path | str) -> None:
    """Move ``entry`` to the front of ``sys.path`` (resolved for stability)."""
    s = str(Path(entry).resolve())
    while s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


def _models_origin_hint() -> str | None:
    m = sys.modules.get("models")
    if m is None:
        return None
    file = getattr(m, "__file__", None)
    if file:
        return str(file)
    path = getattr(m, "__path__", None)
    if path is not None:
        try:
            return str(next(iter(path)))
        except StopIteration:
            return None
    return None


def drop_cached_models_package_unless_origin_contains(needle: str) -> None:
    """
    Remove ``models`` (+ submodules) from ``sys.modules`` if the cached package
    is not the one under ``needle`` (path substring, e.g. ``pose-models`` or
    ``AudioSep``).
    """
    norm_needle = needle.replace("\\", "/")
    hint = _models_origin_hint()
    if hint is not None and norm_needle in hint.replace("\\", "/"):
        return
    if hint is None and sys.modules.get("models") is None:
        return
    for key in list(sys.modules):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]
