#!/usr/bin/env bash
# Generate schema/protobuf/*_pb2.py and *_pb2.pyi from the .proto files. Commit the result.
# grpcio-tools 1.80.0 emits gencode 6.31.1: keep the protobuf lower bound in pyproject.toml in step.
# Usage: scripts/gen_proto.sh [--check]   (--check writes nothing and fails if the files are out of date)
# Env:   PYTHON  interpreter for the throwaway venv (default: python3)
set -euo pipefail

cd "$(dirname "$0")/.."

GRPCIO_TOOLS="grpcio-tools==1.80.0"
PROTO_DIR="schema/protobuf"
PYTHON="${PYTHON:-python3}"

check=0
case "${1:-}" in
    "") ;;
    --check) check=1 ;;
    *)
        echo "usage: $0 [--check]" >&2
        exit 2
        ;;
esac

shopt -s nullglob
protos=("$PROTO_DIR"/*.proto)
if [ "${#protos[@]}" -eq 0 ]; then
    echo "gen_proto: no .proto files in $PROTO_DIR" >&2
    exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

"$PYTHON" -m venv "$work/venv"
# Wheels only: fail fast instead of compiling grpcio-tools.
"$work/venv/bin/python" -m pip install --quiet --disable-pip-version-check --only-binary :all: "$GRPCIO_TOOLS" || {
    echo "gen_proto: could not install $GRPCIO_TOOLS with $PYTHON (needs a network and Python 3.9 to 3.14)" >&2
    exit 1
}

# --proto_path sets the descriptor names and the imports between the modules.
mkdir "$work/out"
"$work/venv/bin/python" -m grpc_tools.protoc --proto_path="$PROTO_DIR" \
    --python_out="$work/out" --pyi_out="$work/out" "${protos[@]}"

# The files are committed, so drop protoc's marker that says they are not. No sed -i: it differs on macOS.
for file in "$work/out"/*_pb2.py; do
    sed '/^# NO CHECKED-IN PROTOBUF GENCODE$/d' "$file" >"$file.tmp"
    mv "$file.tmp" "$file"
done

current=("$PROTO_DIR"/*_pb2.py "$PROTO_DIR"/*_pb2.pyi)

if [ "$check" -eq 1 ]; then
    mkdir "$work/current"
    if [ "${#current[@]}" -gt 0 ]; then
        cp "${current[@]}" "$work/current/"
    fi
    if ! diff -ru "$work/current" "$work/out"; then
        echo "gen_proto: $PROTO_DIR is out of date. Run scripts/gen_proto.sh and commit the result." >&2
        exit 1
    fi
    echo "gen_proto: $PROTO_DIR is up to date"
    exit 0
fi

# Also removes the outputs of a deleted .proto file.
if [ "${#current[@]}" -gt 0 ]; then
    rm -f "${current[@]}"
fi
cp "$work/out"/* "$PROTO_DIR/"
echo "gen_proto: wrote $(cd "$work/out" && echo *) to $PROTO_DIR"
