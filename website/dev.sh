#!/usr/bin/env bash
# Backward-compatible alias for launch.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch.sh" "$@"
