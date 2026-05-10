#!/bin/bash
set -e

## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Inside the Docker container, ASTRA-sim is built at /app/astra-sim.
# PROJECT_DIR is kept for reference but ASTRA_SIM is set explicitly
# to avoid path resolution issues with the mlsys26 mount point.
PROJECT_DIR="/app/astra-sim"
EXAMPLE_DIR="${PROJECT_DIR:?}/examples"

# start
echo "[ASTRA-sim] Compiling ASTRA-sim with the Analytical Network Backend..."
echo ""

# Compile
# "${PROJECT_DIR:?}"/build/astra_analytical/build.sh

echo ""
echo "[ASTRA-sim] Compilation finished."
echo "[ASTRA-sim] Running ASTRA-sim Example with Analytical Network Backend..."
echo ""


# paths
ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
# Chakra .et traces are mounted at /traces inside the Docker container.
# ASTRA-sim reads /traces/chakra_trace.0.et ... /traces/chakra_trace.7.et
WORKLOAD="/traces/chakra_trace"
SYSTEM="${SCRIPT_DIR:?}/system.json"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory/analytical/no_memory_expansion.json"

# Temporary debug knobs:
# 1) suppress unreleased-node teardown warnings
# 2) avoid early exit when event queue drains before all ranks finish
# 3) cap retries to avoid infinite loops
: "${ASTRA_SIM_SKIP_UNRELEASED_NODE_CHECK:=1}"
: "${ASTRA_SIM_DISABLE_EARLY_EXIT:=1}"
: "${ASTRA_SIM_EMPTY_QUEUE_STALL_LIMIT:=20000}"
export ASTRA_SIM_SKIP_UNRELEASED_NODE_CHECK
export ASTRA_SIM_DISABLE_EARLY_EXIT
export ASTRA_SIM_EMPTY_QUEUE_STALL_LIMIT


run_case() {
    local case_name="$1"
    local logging_dir="${SCRIPT_DIR:?}/${case_name}"
    local network="${SCRIPT_DIR:?}/${case_name}.yml"

    mkdir -p "${logging_dir:?}"
    echo "[ASTRA-sim] Running case: ${case_name}"

    "${ASTRA_SIM:?}" \
        --workload-configuration="${WORKLOAD}" \
        --system-configuration="${SYSTEM:?}" \
        --remote-memory-configuration="${REMOTE_MEMORY:?}" \
        --network-configuration="${network:?}" \
        --logging-folder="${logging_dir:?}" > /dev/null 2>&1 || true
}

for case_name in \
    fc75 fc150 fc300 fc600 fc900\
    s75 s150 s300 s600 s900 \
    r75 r150 r300 r600 r900; do
    run_case "${case_name}" &
done
wait


# finalize
echo ""
echo "[ASTRA-sim] Finished the execution."
