from __future__ import annotations

import os
from pathlib import Path


def job_link_dir(data_dir: Path, job_id: str) -> Path:
    return data_dir / "jobs" / job_id


def write_run_reference(link_dir: Path, run_dir: Path) -> None:
    link_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir.resolve()
    link_path = link_dir / "run_dir"
    txt_path = link_dir / "run_dir.txt"
    try:
        if os.name == "nt":
            raise OSError("avoid symlink on windows unknown privileges")
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target, target_is_directory=True)
        if txt_path.exists():
            txt_path.unlink()
    except OSError:
        txt_path.write_text(str(target), encoding="utf-8")


def resolve_run_dir_for_job(data_dir: Path, job_id: str) -> Path | None:
    base = job_link_dir(data_dir, job_id)
    link_path = base / "run_dir"
    if link_path.is_symlink() or link_path.is_dir():
        return link_path.resolve()
    txt = base / "run_dir.txt"
    if txt.is_file():
        p = Path(txt.read_text(encoding="utf-8").strip())
        return p.resolve()
    return None


def safe_resolve_under_run_dir(run_dir: Path, relative: str) -> Path:
    """Resolve relative path under run_dir; raise if path escapes."""
    if ".." in relative.split("/") or relative.startswith("/"):
        raise ValueError("invalid path")
    resolved = (run_dir / relative).resolve()
    root = run_dir.resolve()
    s_res = str(resolved)
    s_root = str(root)
    if os.name == "nt":
        s_res = s_res.lower()
        s_root = s_root.lower()
    if not s_res.startswith(s_root):
        raise PermissionError("path traversal")
    return resolved
