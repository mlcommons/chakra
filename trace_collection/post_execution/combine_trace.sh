set -ex pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_DIR="${TRACE_DIR:-${SCRIPT_DIR}/traces}"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ ! -d "${TRACE_DIR}" ]]; then
    echo "[ERROR] Trace directory not found: ${TRACE_DIR}"
    exit 1
fi

# Automatically detect number of ranks from host*.json files
NUM_RANKS=$(ls "${TRACE_DIR}"/host*.json 2>/dev/null | wc -l)
if [[ "${NUM_RANKS}" -eq 0 ]]; then
    echo "[ERROR] No host_*.json files found in ${TRACE_DIR}"
    exit 1
fi
echo "[INFO] Found ${NUM_RANKS} rank(s) in ${TRACE_DIR}"

# ---------------------------------------------------------------------------
# Step 1: chakra_trace_link  (host + device → linked JSON in TRACE_DIR)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: chakra_trace_link ==="
for ((rank=0; rank<NUM_RANKS; rank++)); do
    HOST_TRACE="${TRACE_DIR}/host.${rank}.json"
    DEVICE_TRACE="${TRACE_DIR}/device.${rank}.json"
    LINKED_OUT="${TRACE_DIR}/linked.${rank}.json"

    echo "[rank ${rank}] Linking ${HOST_TRACE} + ${DEVICE_TRACE} -> ${LINKED_OUT}"
    chakra_trace_link \
        --chakra-host-trace "${HOST_TRACE}" \
        --chakra-device-trace "${DEVICE_TRACE}" \
        --rank "${rank}" \
        --output-file "${LINKED_OUT}"
done
echo "[INFO] All ranks linked."

# ---------------------------------------------------------------------------
# Step 2: chakra_converter  (linked JSON → protobuf .et in TRACE_DIR)
# ASTRA-sim expects files named {prefix}.{npu_id}.et
# e.g. chakra_trace.0.et, chakra_trace.1.et, ...
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: chakra_converter ==="
for ((rank=0; rank<NUM_RANKS; rank++)); do
    LINKED_IN="${TRACE_DIR}/linked.${rank}.json"
    ET_OUT="${TRACE_DIR}/chakra_trace.${rank}.et"

    echo "[rank ${rank}] Converting ${LINKED_IN} -> ${ET_OUT}"
    chakra_converter --log-filename /dev/null \
        PyTorch \
        --input "${LINKED_IN}" \
        --output "${ET_OUT}"
done
echo "[INFO] All ranks converted."

# ---------------------------------------------------------------------------
# Step 3: chakra_jsonizer  (protobuf .et to JSON in TRACE_DIR)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: chakra_jsonizer ==="
for ((rank=0; rank<NUM_RANKS; rank++)); do
    ET_OUT="${TRACE_DIR}/chakra_trace.${rank}.et"
    JSON_OUT="${TRACE_DIR}/chakra_trace.${rank}.json"

    echo "[rank ${rank}] Converting ${ET_OUT} -> ${JSON_OUT}"
    chakra_jsonizer \
        --input "${ET_OUT}" \
        --output "${JSON_OUT}"
done
echo "[INFO] All ranks converted."

echo ""
echo "=== Done ==="
echo "  Files: chakra_trace.0.et ... chakra_trace.$((NUM_RANKS-1)).et"
echo ""