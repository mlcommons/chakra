# Developer tasks for chakra. Run `just` to list them.

# uv uses this Python, the version that CI uses. A .python-version file would also reach pyenv, mise and asdf users.
export UV_PYTHON := "3.11"

default:
    @just --list

# Install the environment: Python 3.11, the dev tools and PARAM (uv sync)
setup:
    uv sync

# Format the code and fix the lint errors that ruff can fix
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Lint: format check, ruff and vulture (the CI "Python Lint" job, plus the format check that CI lacks)
lint:
    uv run ruff format --check .
    uv run ruff check .
    uv run vulture src/ tests/

# Lint the shell scripts in scripts/ with shellcheck
lint-scripts:
    uvx --from shellcheck-py==0.11.0.1 shellcheck scripts/*.sh

# Only the dev tools are installed, as in CI. Every Python file that git does not ignore is checked, new files too.
# Pyright must not scan .venv. With the project installed, pyright reports more issues than CI does.
# Type check with pyright, as CI does
types:
    scripts/run_on_git_files.sh '*.py' -- uv run --isolated --only-group dev pyright

# Unit tests. Use `python -m pytest`, not `pytest`: 3 test files import `src.*`, which needs the repo root on sys.path
[positional-arguments]
test *args:
    CHAKRA_TOOLING_TESTS=1 uv run python -m pytest -vv tests "$@"

# The slow tests build the package: as an editable install, and as a wheel that has a file left out. They need a network.
test-slow:
    CHAKRA_TEST_SLOW=1 uv run python -m pytest tests/test_tooling.py tests/test_scripts.py -k "test_editable_install_builds_and_imports or test_wheel_check_catches_a_file_that_is_missing_from_the_wheel"

# It imports the generated modules with the newest protobuf, then with the lowest one that pyproject.toml allows.
# The CI "Protobuf Gencode/Runtime Smoke Test"
proto-smoke:
    scripts/proto_smoke.sh

# The other checks use the editable install that `uv sync` makes. It can hide a file that the wheel does not hold.
# Build the wheel from a clean copy, install it (not editable) into a new environment, and test it there, as CI does
wheel-check:
    scripts/wheel_check.sh

# Check the C++ format with clang-format 16 (the CI "C++ Lint" job)
cpp-fmt-check:
    scripts/run_on_git_files.sh '*.cc' '*.cpp' '*.h' '*.hh' -- uvx --from 'clang-format==16.*' clang-format --dry-run --Werror

# The full check. Run it before you open a pull request.
gate: lint lint-scripts types test test-slow proto-smoke wheel-check cpp-fmt-check

# It fails now: the workflows use old actions (checkout@v2, setup-python@v2). It is not part of `gate`.
# Lint the GitHub workflow files with actionlint (needs Docker)
ci-github-lint:
    docker run --rm -v "$PWD":/repo -w /repo rhysd/actionlint:1.7.7 -color
