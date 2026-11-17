#!/usr/bin/env bash
set -euo pipefail

python_bin="${1:-python3}"
port="${2:-8000}"
token="${RA_ACCESS_TOKEN:-}"

if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    printf '%s\n' "Usage: bash boot.sh [python] [port 1-65535]" >&2
    exit 2
fi
if (( ${#token} < 32 || ${#token} > 256 )) || [[ ! "$token" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf '%s\n' "Set RA_ACCESS_TOKEN to a URL-safe token of 32-256 characters." >&2
    exit 2
fi

exec "$python_bin" -m research_assist.cli serve --host 127.0.0.1 --port "$((10#$port))"
