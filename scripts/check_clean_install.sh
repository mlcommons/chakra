#!/usr/bin/env bash
#
# Prove that a fresh install of Chakra works on the current Python.
#
# Steps:
#   1. build the wheel and install it in a fresh venv, plus the PARAM et_replay package
#   2. run all six tools with --help (exit code 0)
#   3. import the generated protobuf modules in a fresh interpreter
#   4. run chakra_trace_link and chakra_converter on the simple_onerank fixture (traces collected on a GPU)
#
# After each tool runs we check: exit code 0, the output exists and decodes, no ERROR/CRITICAL
# log line and no Traceback in stdout, stderr or debug.log. WARNING lines are counted, not fatal.
#
# Usage:   scripts/check_clean_install.sh
# Env:     PYTHON          interpreter used to create the venvs (default: python3)
#          PARAM_REF       git ref of facebookresearch/param to install (default: pinned below)
#          TRACE_SETS      space-separated fixture sets to run (default: every directory in the fixture)
#          KEEP_WORKDIR=1  do not delete the temporary work directory on exit (default: 0)
#          FAILURE_ARTIFACTS_DIR  if the run fails, copy the logs and trace outputs into this directory

set -euo pipefail

PYTHON="${PYTHON:-python3}"
PARAM_REF="${PARAM_REF:-7b19f586dd8b267333114992833a0d7e0d601630}"
PARAM_SPEC="git+https://github.com/facebookresearch/param.git@${PARAM_REF}#subdirectory=et_replay"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="${REPO_ROOT}/tests/data/simple_onerank_traces.tar.gz"
TOOLS=(
    chakra_converter
    chakra_generator
    chakra_jsonizer
    chakra_timeline_visualizer
    chakra_trace_link
    chakra_visualizer
)
INVALID_GRAPH_MSG="Critical path analysis encountered an invalid graph structure"

WORK="$(mktemp -d)"
LOGS="${WORK}/logs"
mkdir -p "${LOGS}"

# On exit: if the run failed and FAILURE_ARTIFACTS_DIR is set, copy the logs and the trace sets (linked JSON, .et,
# debug.log and input traces) there so CI can upload them; then delete the work directory unless KEEP_WORKDIR=1.
cleanup() {
    local rc=$?
    if [[ ${rc} -ne 0 && -n "${FAILURE_ARTIFACTS_DIR:-}" ]]; then
        mkdir -p "${FAILURE_ARTIFACTS_DIR}"
        cp -r "${LOGS}" "${FAILURE_ARTIFACTS_DIR}/" 2>/dev/null || true
        [[ -d "${WORK}/trace" ]] && cp -r "${WORK}/trace" "${FAILURE_ARTIFACTS_DIR}/" 2>/dev/null || true
        echo "Logs and traces of the failed run copied to ${FAILURE_ARTIFACTS_DIR}" >&2
    fi
    if [[ "${KEEP_WORKDIR:-0}" == "1" ]]; then
        echo "Work directory kept at ${WORK}"
    else
        rm -rf "${WORK}"
    fi
}
trap cleanup EXIT
TOTAL_WARNINGS=0

step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

# Run a command with stdout+stderr captured to ${LOGS}/<name>.log; replay the log and fail on a non-zero exit.
run_logged() {
    local name="$1"
    shift
    local rc=0
    "$@" >"${LOGS}/${name}.log" 2>&1 || rc=$?
    cat "${LOGS}/${name}.log"
    [[ ${rc} -eq 0 ]] || fail "${name} exited with code ${rc}"
}

# Fail on Traceback or ERROR/CRITICAL log lines in the given files; add WARNING lines to the running count.
# Chakra logs as "LEVEL:root:msg" or "LEVEL [date] msg" and HTA/traceinsight as "[date] file:line [LEVEL]: msg",
# so a level is matched either at the line start or as "[LEVEL]:".
check_logs() {
    local name="$1"
    shift
    local f
    for f in "$@"; do
        [[ -f "${f}" ]] || continue
        if grep -q "${INVALID_GRAPH_MSG}" "${f}"; then
            fail "${name}: '${INVALID_GRAPH_MSG}' found in ${f}"
        fi
        if grep -n '^Traceback (most recent call last)' "${f}"; then
            fail "${name}: Traceback found in ${f}"
        fi
        if grep -nE '^(ERROR|CRITICAL)([: ]|$)|\[(ERROR|CRITICAL)\]:' "${f}"; then
            fail "${name}: ERROR-level log line found in ${f}"
        fi
        local n
        n="$(grep -cE '^WARNING([: ]|$)|\[WARNING\]:' "${f}" || true)"
        TOTAL_WARNINGS=$((TOTAL_WARNINGS + n))
        echo "${name}: ${n} WARNING line(s) in $(basename "${f}")"
    done
}

# ---------------------------------------------------------------------------------------------------------------------
step "Build the wheel (python -m build) in a throwaway venv"
"${PYTHON}" -m venv "${WORK}/build-venv"
"${WORK}/build-venv/bin/python" -m pip install --quiet --upgrade pip build
# Build from a copy so the build never writes build/ or *.egg-info into the working tree.
mkdir "${WORK}/src"
(cd "${REPO_ROOT}" && git ls-files --cached --others --exclude-standard -z | tar --null -T - -cf -) | tar -xf - -C "${WORK}/src"
"${WORK}/build-venv/bin/python" -m build --wheel --outdir "${WORK}/dist" "${WORK}/src"
WHEEL="$(ls "${WORK}"/dist/*.whl)"
echo "Built ${WHEEL}"

step "Install the wheel and PARAM et_replay in a fresh venv"
"${PYTHON}" -m venv "${WORK}/venv"
VPY="${WORK}/venv/bin/python"
BIN="${WORK}/venv/bin"
"${VPY}" -m pip install --quiet --upgrade pip
"${VPY}" -m pip install "${WHEEL}"
"${VPY}" -m pip install "${PARAM_SPEC}"
"${VPY}" -m pip list

# ---------------------------------------------------------------------------------------------------------------------
step "Run every tool with --help"
for tool in "${TOOLS[@]}"; do
    [[ -x "${BIN}/${tool}" ]] || fail "${tool} is not installed in the venv"
    run_logged "${tool}_help" "${BIN}/${tool}" --help >/dev/null
    echo "ok: ${tool} --help"
done

# ---------------------------------------------------------------------------------------------------------------------
step "Import the generated protobuf modules in a fresh interpreter"
# Run from ${WORK} with -I so the source checkout cannot shadow the installed package.
(
    cd "${WORK}"
    "${VPY}" -I - <<'EOF'
import importlib
import importlib.resources as resources

for module in ("chakra.schema.protobuf.et_def_pb2", "chakra.schema.protobuf.storage_pb2"):
    importlib.import_module(module)
    print("imported", module)

package_files = resources.files("chakra.schema.protobuf")
for proto in ("et_def.proto", "storage.proto"):
    assert package_files.joinpath(proto).is_file(), f"{proto} is missing from the installed package"
    print("packaged", proto)
EOF
) || fail "protobuf import check failed"

# ---------------------------------------------------------------------------------------------------------------------
step "Extract the fixture"
mkdir -p "${WORK}/trace"
tar -xzf "${FIXTURE}" -C "${WORK}/trace"
if [[ -n "${TRACE_SETS:-}" ]]; then
    read -r -a SETS <<<"${TRACE_SETS}"
else
    SETS=()
    for dir in "${WORK}"/trace/*/; do SETS+=("$(basename "${dir}")"); done
fi
[[ ${#SETS[@]} -gt 0 ]] || fail "fixture contains no trace sets"
echo "Trace sets: ${SETS[*]}"

# Link and convert one trace set (a directory with host.0.json and device.0.json), checking every output.
run_trace_set() {
    local set="$1"
    local dir="${WORK}/trace/${set}"
    local host_trace="${dir}/host.0.json"
    local device_trace="${dir}/device.0.json"
    [[ -f "${host_trace}" && -f "${device_trace}" ]] || fail "${set}: missing host.0.json or device.0.json"

    step "[${set}] chakra_trace_link"
    local linked="${dir}/linked.json"
    run_logged "${set}_trace_link" "${BIN}/chakra_trace_link" \
        --rank 0 \
        --chakra-host-trace "${host_trace}" \
        --chakra-device-trace "${device_trace}" \
        --output-file "${linked}"
    check_logs "${set}_trace_link" "${LOGS}/${set}_trace_link.log"
    [[ -s "${linked}" ]] || fail "${set}: trace_link did not produce ${linked}"
    "${VPY}" -I -c 'import json, sys; json.load(open(sys.argv[1]))' "${linked}" \
        || fail "${set}: linked trace is not valid JSON"
    echo "ok: linked trace parses as JSON"

    step "[${set}] chakra_converter"
    local et_out="${dir}/chakra.et"
    local debug_log="${dir}/debug.log"
    run_logged "${set}_converter" "${BIN}/chakra_converter" --log-filename "${debug_log}" \
        PyTorch --input "${linked}" --output "${et_out}"
    check_logs "${set}_converter" "${LOGS}/${set}_converter.log" "${debug_log}"
    [[ -s "${et_out}" ]] || fail "${set}: converter did not produce ${et_out}"
    "${VPY}" -I - "${et_out}" <<'EOF' || fail "${set}: converted .et does not decode or has no nodes"
import sys

from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
from chakra.src.third_party.utils.protolib import decodeMessage, openFileRd

et = openFileRd(sys.argv[1])
metadata = GlobalMetadata()
assert decodeMessage(et, metadata), "could not read the GlobalMetadata header"
nodes = 0
node = Node()
while decodeMessage(et, node):
    nodes += 1
assert nodes > 0, "the .et file contains 0 nodes"
print(f"ok: .et decodes, {nodes} nodes")
EOF
}

for set in "${SETS[@]}"; do
    run_trace_set "${set}"
done

# ---------------------------------------------------------------------------------------------------------------------
step "Summary"
echo "All checks passed on $("${VPY}" --version). Total WARNING lines (not fatal): ${TOTAL_WARNINGS}"
