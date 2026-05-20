from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_pipeline_subprocess(
    *,
    video_path: Path,
    repo_root: Path,
    device: str,
    cat_threshold: float,
    split_window_sec: float,
    split_step_sec: float,
    output_dir: str,
    timeout_sec: int,
    multicat_video_only: bool = False,
    multicat_max_cats: int = 8,
    multicat_min_track_coverage: float = 0.15,
    multicat_decision_threshold: float = 0.5,
    multicat_summary_strategy: str = "coverage_weighted_mean",
) -> dict[str, Any]:
    """
    Run inference via subprocess so timeouts can terminate the process tree.
    Parses JSON printed to stdout by pipeline main().
    """
    repo_root = repo_root.resolve()
    script = repo_root / "src" / "inference" / "pipeline.py"
    if not script.is_file():
        raise FileNotFoundError(f"Pipeline script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--video",
        str(video_path),
        "--output-dir",
        output_dir,
        "--device",
        device,
        "--cat-threshold",
        str(cat_threshold),
        "--split-window-sec",
        str(split_window_sec),
        "--split-step-sec",
        str(split_step_sec),
        "--multicat-max-cats",
        str(multicat_max_cats),
        "--multicat-min-track-coverage",
        str(multicat_min_track_coverage),
        "--multicat-decision-threshold",
        str(multicat_decision_threshold),
        "--multicat-summary-strategy",
        str(multicat_summary_strategy),
    ]
    if multicat_video_only:
        cmd.append("--multicat-video-only")
    # Only ``src`` on PYTHONPATH: ``video/pose-models`` must NOT be on the path
    # when resolving top-level ``import models`` for AudioSep (regular package at
    # pose-models/models shadows AudioSep's models/). ST-GCN loader prepends
    # pose-models in-process when loading the video branch.
    srcp = str((repo_root / "src").resolve())
    website_root = str((repo_root / "website").resolve())
    try:
        from observability.pipeline import metrics_env_for_subprocess
        from observability.worker import scan_pipeline_output

        env = metrics_env_for_subprocess(website_root, srcp)
    except ImportError:
        env = {**os.environ, "PYTHONPATH": srcp}
        scan_pipeline_output = None  # type: ignore[assignment]

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        stdout, stderr = proc.communicate()
        e.stdout = stdout
        e.stderr = stderr
        if scan_pipeline_output is not None:
            scan_pipeline_output((stderr or stdout or "")[:8000])
        try:
            from observability.registry import mark_subprocess_dead

            mark_subprocess_dead(proc.pid)
        except ImportError:
            pass
        raise
    finally:
        try:
            from observability.registry import mark_subprocess_dead

            mark_subprocess_dead(proc.pid)
        except ImportError:
            pass

    if proc.returncode != 0:
        err = (stderr or stdout or "").strip()
        if scan_pipeline_output is not None:
            scan_pipeline_output(err)
        raise RuntimeError(f"Pipeline exited {proc.returncode}: {err[-4000:]}")

    out = (stdout or "").strip()
    if not out:
        raise RuntimeError("Pipeline produced no stdout JSON")
    # Pipeline prints full JSON; take from first { in case of log leakage
    idx = out.find("{")
    if idx < 0:
        raise RuntimeError("Pipeline stdout did not contain JSON object")
    return json.loads(out[idx:])
