#!/usr/bin/env bash
# The CI "Protobuf Gencode/Runtime Smoke Test".
# Import the generated protobuf modules with the newest protobuf, then with the lowest one that pyproject.toml allows.
# Usage: scripts/proto_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

check='import chakra.schema.protobuf.et_def_pb2, chakra.schema.protobuf.storage_pb2'
uv run python -c "$check"
floor=$(uv run python -c "import re; from importlib.metadata import requires; print(next(re.search(r'>=([0-9.]+)', r).group(1) for r in requires('mlc-chakra') if r.startswith('protobuf')))")
uv run --with "protobuf==$floor" python -c "$check"
