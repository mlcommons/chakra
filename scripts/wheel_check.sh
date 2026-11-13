#!/usr/bin/env bash
# Check the package the way a user and CI get it: build a wheel, install it (not editable) into a new environment,
# and test it there. `uv sync` installs the project in editable mode, and an editable install can hide a file that
# the wheel does not hold.
# Usage: scripts/wheel_check.sh
set -euo pipefail

cd "$(dirname "$0")/.."

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python="$work/venv/bin/python"

# Build from a clean copy of the files that git tracks or could add, as CI builds from a clean checkout. A stale
# build/ folder (setuptools reuses it) cannot leak into the wheel, and the repository stays clean.
source="$work/source"
while IFS= read -r -d '' file; do
    if [ -f "$file" ]; then
        mkdir -p "$source/$(dirname "$file")"
        cp -p "$file" "$source/$file"
    fi
done < <(git ls-files -z --cached --others --exclude-standard)
if [ ! -f "$source/pyproject.toml" ]; then
    echo "wheel_check: could not copy the files. Run it in a git checkout." >&2
    exit 1
fi
uv build --quiet --wheel --out-dir "$work/dist" "$source"
wheel=$(echo "$work"/dist/*.whl)
uv venv --quiet "$work/venv"
uv pip install --quiet --python "$python" "$wheel"

# 1. The wheel holds the generated protobuf modules and the .proto files.
listing=$("$python" -m zipfile -l "$wheel")
for file in chakra/schema/protobuf/et_def_pb2.py chakra/schema/protobuf/storage_pb2.py \
    chakra/schema/protobuf/et_def.proto chakra/schema/protobuf/storage.proto; do
    if ! grep -q "$file" <<<"$listing"; then
        echo "wheel_check: $file is missing from the wheel" >&2
        exit 1
    fi
done

# 2. Import the generated modules with the newest protobuf, then with the lowest one that pyproject.toml allows.
check='import chakra.schema.protobuf.et_def_pb2, chakra.schema.protobuf.storage_pb2'
"$python" -c "$check"
floor=$("$python" -c "import re; from importlib.metadata import requires; print(next(re.search(r'>=([0-9.]+)', r).group(1) for r in requires('mlc-chakra') if r.startswith('protobuf')))")
uv pip install --quiet --python "$python" "protobuf==$floor"
"$python" -c "$check"

# 3. The commands start. With no arguments, argparse exits with 2, as in the CI end-to-end workflow.
# chakra_trace_link is not tried here: it needs PARAM (et_replay).
for command in chakra_converter chakra_jsonizer chakra_timeline_visualizer chakra_visualizer; do
    "$work/venv/bin/$command" >/dev/null 2>&1 || [ $? -eq 2 ] || {
        echo "wheel_check: $command does not start" >&2
        exit 1
    }
done
echo "wheel_check: the wheel is complete and works with protobuf $floor and the newest protobuf"
