#!/usr/bin/env bash
# Start the full local stack: API (8000) + worker + Vite (5173).
# Run from anywhere:  bash website/launch.sh
# Or from website/:   ./launch.sh
#
# Options:
#   --install-deps   pip install website/requirements.txt and npm install in frontend/
#   --no-frontend    only API + worker
#   --no-brew-start  do not try "brew services start postgresql@16" when DB is down
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${PWD}"
export REPO_ROOT="${REPO_ROOT:-$(cd .. && pwd)}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  export PATH="${REPO_ROOT}/.venv/bin:$PATH"
  echo "Using Python from ${REPO_ROOT}/.venv"
fi

INSTALL_DEPS=0
START_FRONTEND=1
TRY_BREW_PG=1
for arg in "$@"; do
  case "$arg" in
    --install-deps) INSTALL_DEPS=1 ;;
    --no-frontend) START_FRONTEND=0 ;;
    --no-brew-start) TRY_BREW_PG=0 ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--install-deps] [--no-frontend] [--no-brew-start]" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${REPO_ROOT}/src/inference/pipeline.py" ]]; then
  echo "REPO_ROOT must point at the Cats-in-Pain-Bachelors repo (with src/inference/pipeline.py)." >&2
  echo "Current REPO_ROOT=${REPO_ROOT}" >&2
  exit 1
fi

# Homebrew PostgreSQL binaries are often off PATH
for _pg in /opt/homebrew/opt/postgresql@16/bin /opt/homebrew/opt/postgresql@15/bin /usr/local/opt/postgresql@16/bin; do
  if [[ -d "$_pg" ]]; then
    export PATH="${_pg}:$PATH"
    break
  fi
done

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  echo "Installing Python deps (website/requirements.txt)…"
  pip install -r requirements.txt
  echo "Installing frontend deps…"
  (cd frontend && npm install)
fi

if [[ ! -d frontend/node_modules ]]; then
  echo "frontend/node_modules missing; run: (cd frontend && npm install)" >&2
  echo "Or re-run with:  bash launch.sh --install-deps" >&2
  exit 1
fi

ensure_postgres() {
  if command -v pg_isready >/dev/null 2>&1; then
    pg_isready -h 127.0.0.1 -p 5432 -q
  else
    echo "pg_isready not found. Install PostgreSQL (e.g. brew install postgresql@16) or add its bin dir to PATH." >&2
    return 1
  fi
}

if ! ensure_postgres; then
  if [[ "$TRY_BREW_PG" -eq 1 ]] && command -v brew >/dev/null 2>&1; then
    for formula in postgresql@16 postgresql@15; do
      if brew list "$formula" &>/dev/null; then
        echo "Starting ${formula} via Homebrew…"
        brew services start "$formula" || true
        sleep 2
        break
      fi
    done
  fi
fi

if ! ensure_postgres; then
  echo "" >&2
  echo "PostgreSQL is not accepting connections on 127.0.0.1:5432." >&2
  echo "  brew services start postgresql@16" >&2
  echo "  createdb catpain_web" >&2
  echo "See website/README.md" >&2
  exit 1
fi

if command -v createdb >/dev/null 2>&1; then
  createdb -h 127.0.0.1 -p 5432 catpain_web 2>/dev/null || true
else
  echo "Tip: install client tools so createdb exists, or run: createdb -h 127.0.0.1 -p 5432 catpain_web" >&2
fi

cleanup() {
  [[ -n "${UVICORN_PID:-}" ]] && kill "${UVICORN_PID}" 2>/dev/null || true
  [[ -n "${WORKER_PID:-}" ]] && kill "${WORKER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo ""
echo "=== Cat Pain web (local)"
echo "    REPO_ROOT=${REPO_ROOT}"
echo "    API       http://127.0.0.1:8000   docs: /docs"
echo "    Frontend  http://127.0.0.1:5173  (proxies /jobs to API)"
echo ""

echo "Starting backend (uvicorn)…"
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!

echo "Starting worker…"
python worker/runner.py &
WORKER_PID=$!

sleep 1

if [[ "$START_FRONTEND" -eq 1 ]]; then
  echo "Starting frontend (Vite)…"
  (cd frontend && npm run dev)
else
  echo "Frontend skipped; press Ctrl+C to stop API + worker."
  wait "${UVICORN_PID}" "${WORKER_PID}" 2>/dev/null || wait
fi
