#!/usr/bin/env bash
# Start a local PostgreSQL 16 instance for the web MVP (Docker required).
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "docker: command not found." >&2
  echo "" >&2
  echo "Install Docker Desktop for Mac (https://www.docker.com/products/docker-desktop/), then re-run this script," >&2
  echo "or use PostgreSQL without Docker, for example:" >&2
  echo "  brew install postgresql@16" >&2
  echo "  brew services start postgresql@16" >&2
  echo "  createdb catpain_web" >&2
  exit 127
fi

NAME="${POSTGRES_CONTAINER_NAME:-catpain-pg}"
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Container $NAME already exists; starting if stopped..."
  docker start "$NAME"
else
  echo "Creating $NAME (postgres:16-alpine on port 5432)..."
  docker run -d \
    --name "$NAME" \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=catpain_web \
    -p 5432:5432 \
    postgres:16-alpine
fi
echo "Postgres should be listening on localhost:5432"
echo "DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/catpain_web"
