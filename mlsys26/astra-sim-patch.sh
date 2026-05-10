#!/usr/bin/env bash
set -euo pipefail

DOCKERFILE="${1:-Dockerfile}"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "Error: $DOCKERFILE not found"
  exit 1
fi

# cp "$DOCKERFILE" "${DOCKERFILE}.bak"

sed -i \
  -e 's/^ARG ABSL_VER=20240722\.0$/ARG ABSL_VER=20250814.1/' \
  -e 's/^## Download Abseil 20240722\.0.*/## Download Abseil 20250814.1 (Latest LTS as of 10\/31\/2024)/' \
  -e 's/^ARG PROTOBUF_VER=29\.0$/ARG PROTOBUF_VER=33.0/' \
  -e 's/^## Download Protobuf 29\.0.*/## Download Protobuf 33.0 (=v6.33.0, latest stable version as of Feb\/01\/2025)/' \
  -e 's/protobuf==5\.\${PROTOBUF_VER}/protobuf==6.${PROTOBUF_VER}/' \
  "$DOCKERFILE"

python3 - "$DOCKERFILE" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()

# Update all C++ standard settings from 14 -> 17
text = re.sub(r'(-DCMAKE_CXX_STANDARD=)14\b', r'\g<1>17', text)

path.write_text(text)
PY

# Patch the CMakeLists.txt that lives alongside the Dockerfile
CMAKEFILE="$(dirname "$DOCKERFILE")/CMakeLists.txt"

if [[ ! -f "$CMAKEFILE" ]]; then
  echo "Warning: $CMAKEFILE not found, skipping CMakeLists.txt patch"
else
  python3 - "$CMAKEFILE" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()

# Remove hardcoded abseil .so linker lines that are baked into the repo
# but break builds when the abseil version changes.
cleaned, n = re.subn(
    r'\ntarget_link_libraries\(AstraSim PRIVATE /usr/local/lib/libabsl_log_internal[^\n]+\)',
    '',
    text,
)

if n == 0:
    print(f"No abseil link libraries found in {sys.argv[1]}, nothing to remove")
else:
    path.write_text(cleaned)
    print(f"Removed {n} abseil link librar{'y' if n == 1 else 'ies'} from {sys.argv[1]}")
PY
  echo "Patched $CMAKEFILE"
fi

echo "Patched $DOCKERFILE"
# echo "Backup saved as ${DOCKERFILE}.bak"