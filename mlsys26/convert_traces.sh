#!/usr/bin/env bash
# convert_traces.sh
# Links Chakra host+device traces and converts them to protobuf (.et) format
# for all ranks in the Mixtral-8x7B NeMo trace set.
#
# Usage:
#   source <chakra-env>/bin/activate
#   bash mlsys26/convert_traces.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_DIR="${SCRIPT_DIR}/traces/nemo-chakra-mixtral-8x7B-traces"
LINKED_DIR="${SCRIPT_DIR}/traces/linked"
ET_DIR="${SCRIPT_DIR}/traces/et"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ ! -d "${TRACE_DIR}" ]]; then
    echo "[ERROR] Trace directory not found: ${TRACE_DIR}"
    echo "        Run download_nemo_chakra_traces.sh first."
    exit 1
fi

mkdir -p "${LINKED_DIR}" "${ET_DIR}"

# Automatically detect number of ranks from host_*.json files
NUM_RANKS=$(ls "${TRACE_DIR}"/host_*.json 2>/dev/null | wc -l)
if [[ "${NUM_RANKS}" -eq 0 ]]; then
    echo "[ERROR] No host_*.json files found in ${TRACE_DIR}"
    exit 1
fi
echo "[INFO] Found ${NUM_RANKS} rank(s) in ${TRACE_DIR}"

# ---------------------------------------------------------------------------
# Step 1: chakra_trace_link  (host + device → linked JSON)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: chakra_trace_link ==="
for ((rank=0; rank<NUM_RANKS; rank++)); do
    HOST_TRACE="${TRACE_DIR}/host_${rank}.json"
    DEVICE_TRACE="${TRACE_DIR}/device_${rank}.json"
    LINKED_OUT="${LINKED_DIR}/rank${rank}_linked.json"

    echo "[rank ${rank}] Linking ${HOST_TRACE} + ${DEVICE_TRACE} -> ${LINKED_OUT}"
    chakra_trace_link \
        --chakra-host-trace "${HOST_TRACE}" \
        --chakra-device-trace "${DEVICE_TRACE}" \
        --rank "${rank}" \
        --output-file "${LINKED_OUT}"
done
echo "[INFO] All ranks linked."

# ---------------------------------------------------------------------------
# Step 2: chakra_converter  (linked JSON → protobuf .et)
# ASTRA-sim expects files named {prefix}.{npu_id}.et
# e.g. chakra_trace.0.et, chakra_trace.1.et, ...
# so we use --output <ET_DIR>/chakra_trace.<rank> → chakra_trace.<rank>.et
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: chakra_converter ==="
for ((rank=0; rank<NUM_RANKS; rank++)); do
    LINKED_IN="${LINKED_DIR}/rank${rank}_linked.json"
    ET_OUT="${ET_DIR}/chakra_trace.${rank}.et"

    echo "[rank ${rank}] Converting ${LINKED_IN} -> ${ET_OUT}"
    chakra_converter PyTorch \
        --input "${LINKED_IN}" \
        --output "${ET_OUT}"
done
echo "[INFO] All ranks converted."

echo ""
echo "=== Done ==="
echo "Linked JSON traces : ${LINKED_DIR}/"
echo "Protobuf .et traces: ${ET_DIR}/"
echo "  Files: chakra_trace.0.et ... chakra_trace.$((NUM_RANKS-1)).et"
echo "  ASTRA-sim workload prefix: /traces/chakra_trace"
