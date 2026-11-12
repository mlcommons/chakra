#!/usr/bin/env bash
# Run one GitHub workflow in Docker with act, before you push.
# Usage: scripts/act.sh [WORKFLOW_FILE]    (default: python_lint.yml, a file in .github/workflows/)
#
# Before each run, this script removes the shared act cache volume (act-toolcache) and the act containers that use it.
# Why: act keeps one volume for all runs. A package that one run installed is then visible to the next run. This does
# not happen on a GitHub runner. A failed act job also leaves its container running, and that container holds the volume.
set -euo pipefail

cd "$(dirname "$0")/.."

workflow="${1:-python_lint.yml}"
file=".github/workflows/$workflow"
if [ ! -f "$file" ]; then
    echo "act: no workflow file $file" >&2
    exit 2
fi

case "$(uname -m)" in
    x86_64) arch=amd64 ;;
    arm64 | aarch64) arch=arm64 ;;
    *)
        echo "act: unsupported CPU type $(uname -m)" >&2
        exit 1
        ;;
esac

# The removal below would break an act run that is active on this computer, in this repository or in another one.
# A missing pgrep must not count as "no act run": check for the program first.
if ! command -v pgrep >/dev/null; then
    echo "act: pgrep is not installed, so this script cannot tell if another act run is active." >&2
    exit 1
fi
if pgrep -x act >/dev/null; then
    echo "act: another act run is active. Wait for it, or stop it." >&2
    exit 1
fi

# Docker Desktop and OrbStack use different sockets: ask the active Docker context.
host="${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}"
export DOCKER_HOST="$host"

stale=$(docker ps -aq --filter volume=act-toolcache)
listed="${stale//$'\n'/ }"
echo "act: removing the volume act-toolcache and these containers: ${listed:-none}" >&2
while IFS= read -r id; do
    if [ -n "$id" ]; then
        docker rm -f "$id" >/dev/null
    fi
done <<<"$stale"
docker volume rm -f act-toolcache >/dev/null

exec act pull_request -W "$file" --container-architecture "linux/$arch" --container-daemon-socket -
