from __future__ import annotations

import getpass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_local_database_url() -> str:
    """Homebrew Postgres: superuser is usually your macOS login, not ``postgres``."""
    user = getpass.getuser()
    return f"postgresql+psycopg://{user}@localhost:5432/catpain_web"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(default_factory=default_local_database_url)
    repo_root: Path = Field(
        default=Path(__file__).resolve().parent.parent.parent,
    )
    data_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent / "data",
    )
    frontend_origin: str = Field(default="http://localhost:5173")
    allow_local_paths: bool = Field(default=False)
    local_path_base_dir: Path | None = Field(default=None)
    max_upload_mb: int = Field(default=200)
    max_video_duration_sec: float | None = Field(default=600.0)
    job_timeout_sec: int = Field(default=900)
    positive_label: int = Field(default=1)
    max_artifact_stream_mb: int = Field(default=500)
    pipeline_output_dir: str = Field(default="runs/inference")
    wrapper_version: str = Field(default="0.1.0")
    enable_multicat_video: bool = Field(default=True)
    multicat_max_cats_default: int = Field(default=8)
    multicat_min_track_coverage_default: float = Field(default=0.15)
    multicat_decision_threshold_default: float = Field(default=0.5)
    multicat_summary_strategy_default: str = Field(default="coverage_weighted_mean")


def get_settings() -> Settings:
    return Settings()


def try_import_pipeline(repo_root: Path) -> bool:
    """Verify inference package imports (does not run models)."""
    import sys

    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import importlib.util

        spec = importlib.util.find_spec("inference.pipeline")
        return spec is not None
    except Exception:
        return False


def ffmpeg_available() -> bool:
    try:
        r = __import__("subprocess").run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def torch_device_probe() -> dict:
    out: dict = {"cuda_available": False, "mps_available": False}
    try:
        import torch

        out["cuda_available"] = bool(torch.cuda.is_available())
        out["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception as e:
        out["error"] = str(e)
    return out


def data_dir_writable(data_dir: Path) -> bool:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        p = data_dir / ".write_test"
        p.write_text("ok", encoding="utf-8")
        p.unlink(missing_ok=True)
        return True
    except OSError:
        return False
