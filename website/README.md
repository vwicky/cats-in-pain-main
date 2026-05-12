# Cat Pain — local-first web MVP

Stack under `website/`:

- **backend** — FastAPI (`backend/main.py`): jobs API, artifact streaming, health checks
- **worker** — `worker/runner.py` polls PostgreSQL, runs the existing pipeline **via subprocess** (hard timeouts, repo `src/inference/pipeline.py` unchanged)
- **db** — SQLAlchemy + PostgreSQL (`db/models.py`)
- **frontend** — React + Vite (`frontend/`)
- **shared** — Python helpers + TypeScript types (`shared/`)

## Prerequisites

- Python 3.11+ (recommended)
- Node 20+ (for the UI)
- PostgreSQL 16+
- `ffmpeg` / `ffprobe` on `PATH` (same as the main pipeline)
- Full **repository** checkout with pipeline dependencies installed (see root `requirements.txt` — PyTorch, etc.). The web layer only adds small packages in `website/requirements.txt`.

## Configure

```bash
cd website
cp .env.example .env
# Edit DATABASE_URL if needed
```

Optional: `ALLOW_LOCAL_PATHS=1` and `LOCAL_PATH_BASE_DIR=...` for `local_path` job mode.

## Database

Create DB (example):

```bash
createdb catpain_web
# or psql -c "CREATE DATABASE catpain_web;"
```

Tables are created automatically on API startup (`init_db()`).

## One-command launch (recommended)

From the **repository root** (folder that contains `website/` and `src/`):

```bash
bash website/launch.sh
```

**If your shell is already inside `website/`** (prompt ends with `website` and `pwd` shows `.../Cats-in-Pain-Bachelors/website`), run:

```bash
bash launch.sh
# or: ./launch.sh
```

Do **not** use `bash website/launch.sh` from inside `website/` — that path does not exist there.

From `website/` you can also run `make launch` or `make dev` (same as `bash launch.sh`).

The script:

- Sets `PYTHONPATH` and `REPO_ROOT` for you
- Puts common Homebrew Postgres bin dirs on `PATH` (`postgresql@16`, `@15`)
- Checks `pg_isready` on `127.0.0.1:5432`; may run `brew services start postgresql@16` if Homebrew has it
- Runs `createdb catpain_web` if possible (ignores “already exists”)
- Starts **API** (`:8000`), **worker**, and **Vite** (`:5173`) in one terminal (Ctrl+C stops API + worker)
- If `${REPO_ROOT}/.venv` exists, its `bin` is prepended to `PATH` so you do not have to `activate` manually

First-time dependencies:

```bash
cd website
make up
# or: bash launch.sh --install-deps
```

## Run (three processes)

### Important: `PYTHONPATH` and current directory

The FastAPI package is **`website/backend`**, not the repository root. If you run:

```bash
uvicorn backend.main:app ...
```

from **`Cats-in-Pain-Bachelors/`** (parent of `website/`) without fixing the path, you get:

`ModuleNotFoundError: No module named 'backend'`.

**Fix — pick one:**

1. **Recommended:** `cd website` first, then `export PYTHONPATH="$PWD"` (as below).
2. **Stay at repo root:** add the website tree to `PYTHONPATH` and set `REPO_ROOT` to the repo root:

```bash
cd /path/to/Cats-in-Pain-Bachelors   # repo root (contains website/ and src/)
export PYTHONPATH="$PWD/website"
export REPO_ROOT="$PWD"
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
# Worker from repo root (same env vars)
export PYTHONPATH="$PWD/website"
export REPO_ROOT="$PWD"
python website/worker/runner.py
```

Put `website/.env` next to `website/requirements.txt`; when using option 2, either `cd website` before starting processes or copy/symlink `.env` if tools expect it in CWD (simplest: open two terminals and `cd website` for API + worker).

From `website/`:

```bash
export PYTHONPATH="$PWD"
export REPO_ROOT="$(cd .. && pwd)"   # repo root with src/inference
pip install -r requirements.txt
pip install -r ../requirements.txt   # pipeline / ML stack (if not already)

# Terminal 1
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2
PYTHONPATH=. python worker/runner.py

# Terminal 3
cd frontend && npm install && npm run dev
```

### PostgreSQL on macOS (Homebrew)

There is no `service` command like on Linux. Typical:

```bash
brew services start postgresql@16
# or: pg_ctl -D /opt/homebrew/var/postgresql@16 start
```

Then create the database if needed (`createdb catpain_web`) and match `DATABASE_URL` in `website/.env`.

### Troubleshooting: `connection refused` on port 5432

Startup calls `init_db()` and requires PostgreSQL listening on the host/port in `DATABASE_URL` (default `127.0.0.1:5432`).

**Nothing listening?**

1. **Docker** — requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or another install where `docker` is on your `PATH`). If `docker: command not found`, use option 2 or install Docker first.

   ```bash
   cd website
   bash scripts/start_postgres_docker.sh
   ```

2. **Homebrew** — no Docker needed:

   ```bash
   brew install postgresql@16
   brew services start postgresql@16
   createdb catpain_web
   ```

   In `website/.env`, use your **macOS username**, not `postgres`, or remove `DATABASE_URL` entirely so the app picks `$(whoami)` automatically. Example:

   `DATABASE_URL=postgresql+psycopg://yourname@localhost:5432/catpain_web`

   An old `.env` line like `postgres:postgres@...` causes `role "postgres" does not exist` on Homebrew.

3. Check: `nc -zv 127.0.0.1 5432` or `pg_isready -h 127.0.0.1 -p 5432`

Or use **`make dev`** (runs `dev.sh` — starts backend, worker, and Vite; requires Postgres).

Open http://localhost:5173 — API proxied to http://127.0.0.1:8000 .

## Docker Compose

From `website/`:

```bash
docker compose up --build
```

This bind-mounts the **entire repo** into `/workspace`, sets `REPO_ROOT=/workspace`, and starts Postgres + API + worker. Install **heavy** ML dependencies inside the image or extend the Dockerfile (base image only installs `website/requirements.txt`).

## API quick test

```bash
curl -s http://127.0.0.1:8000/health | jq
# Upload job (multipart)
curl -s -X POST http://127.0.0.1:8000/jobs \
  -F mode=upload \
  -F device=cpu \
  -F cat_threshold=0.5 \
  -F split_window_sec=6 \
  -F split_step_sec=3 \
  -F file=@/path/to/video.mp4
```

Poll `GET /jobs/{id}` until `done`, then `GET /jobs/{id}/result` (normalized + `raw`) and `GET /jobs/{id}/artifacts`.

## Cleanup policy

- **No automatic deletion** of runs or uploads.
- `DELETE /jobs/{id}` removes the DB row; add `?delete_run=1` to also remove the linked pipeline run directory and job link folder (explicit opt-in).

## Design notes

- Progress stages are **monotonic**; `n_windows` for split mode is precomputed with the same windowing logic as the pipeline.
- Artifact paths are resolved with strict prefix checks to prevent traversal escapes.
- Large artifact streams are blocked when file size exceeds `MAX_ARTIFACT_STREAM_MB` (MVP guard).
