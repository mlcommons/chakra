#!/usr/bin/env bash
# Run a command on the files that git knows or could add: tracked files and new files, but not ignored or deleted ones.
# Usage: scripts/run_on_git_files.sh PATTERN... -- COMMAND [ARGUMENT...]
# Example: scripts/run_on_git_files.sh '*.py' -- pyright
# Run it in the repository. It stops when no file matches, so a check cannot pass on an empty list.
set -euo pipefail

usage() {
    echo "usage: $0 PATTERN... -- COMMAND [ARGUMENT...]" >&2
    exit 2
}

patterns=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    patterns+=("$1")
    shift
done
[ "${#patterns[@]}" -gt 0 ] || usage
[ "$#" -ge 2 ] || usage
shift # the "--"

files=()
while IFS= read -r -d '' file; do
    # A leading ./ keeps a file named -x.py from looking like an option. A link to a directory is not a file.
    if [ -f "$file" ]; then
        files+=("./$file")
    fi
done < <(git ls-files -z --cached --others --exclude-standard -- "${patterns[@]}")

if [ "${#files[@]}" -eq 0 ]; then
    echo "run_on_git_files: no files match ${patterns[*]} (run it in a git checkout)" >&2
    exit 1
fi

exec "$@" "${files[@]}"
